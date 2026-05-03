"""Typer-based CLI wrapper for dream-trainer training scripts.

The single entrypoint [cli][dream_trainer.utils.cli.cli.cli] turns any
`(config) -> None` training function into a full-featured command-line
tool. Out of the box it exposes:

- A default command that runs `main(trainer_config)` end-to-end with
  options for checkpoint resumption (`--resume`) and partial-checkpoint
  initialisation (`--init-from`).
- Debugging subcommands: `find-graph-breaks`, `summarize`.
- Profiling subcommands: `benchmark`, `profile`, `importtime`.
- An `export` subcommand for offline checkpoint conversion.
- One auto-generated `--<modifier-name>` flag per entry in the
  [MODIFIERS][dream_trainer.utils.modifiers.MODIFIERS] registry.

Typer's signature-based introspection is extended at runtime (see
`add_modifiers_to_signature`) so that modifier flags appear on every
subcommand that accepts `**modifiers`.
"""

import os
from inspect import Parameter, Signature, signature
from typing import TYPE_CHECKING, Annotated, Any, Callable, Literal, TypeVar

import typer
from typer import Option

from dream_trainer.utils import logger
from dream_trainer.utils.dist.core import get_dist_rank
from dream_trainer.utils._logger import setup_logger
from dream_trainer.utils.modifiers import MODIFIERS

if TYPE_CHECKING:
    from dream_trainer import DreamTrainerConfig

TrainerConfig = TypeVar("TrainerConfig", bound="DreamTrainerConfig")


setup_logger()


def torch_setup() -> None:
    """Apply dream-trainer's default PyTorch process-level settings.

    Flips a handful of global knobs that almost every training job wants:

    - `torch.set_float32_matmul_precision("high")` — allow TF32 on matmul.
    - `torch.set_flush_denormal(True)` — zero-out denormals to avoid
      CPU slowdowns.
    - `torch.backends.cudnn.enabled = True` with `benchmark = False` —
      enable cuDNN but skip per-call algorithm search (which is only
      useful when input shapes never change).
    - Silence a noisy root-mesh slicing warning emitted by torch's
      DTensor layer.
    """
    import torch

    torch.set_float32_matmul_precision("high")

    torch.set_flush_denormal(True)

    torch.backends.cudnn.enabled = True

    torch.backends.cudnn.benchmark = False

    import warnings

    warnings.filterwarnings("ignore", message="Slicing a flattened dim from root mesh")


def cli(main: Callable[[TrainerConfig], None], trainer_config: TrainerConfig) -> None:
    """Build and run a typer CLI around `main(trainer_config)`.

    Call this from the `if __name__ == "__main__"` block of your training
    script after instantiating your config. A fully-featured CLI will be
    mounted with subcommands for debugging, profiling, and export, plus
    every registered modifier surfaced as a `--<name>` flag.

    Args:
        main: Your training function. It must accept a single positional
            argument (the trainer config) and return `None`. The function
            is wrapped with [entrypoint][dream_trainer.utils.entrypoint.entrypoint]
            before being invoked, so it can freely assume a distributed
            environment is live.
        trainer_config: The trainer config instance to drive the run.
            Modifier flags mutate this object in place before `main` is
            called.

    Example:
        ```python
        def main(config: MyTrainerConfig):
            trainer = MyTrainer(config)
            trainer.fit()

        if __name__ == "__main__":
            from dream_trainer.utils.cli import cli
            cli(main, MyTrainerConfig())
        ```
    """
    _cli = typer.Typer(
        pretty_exceptions_show_locals=False,
        add_completion=False,
        chain=True,
    )

    local_modifiers = MODIFIERS[trainer_config.__class__]

    def apply_modifiers(modifiers: dict[str, Any]) -> None:
        """Dispatch user-supplied modifier values to their registered functions.

        Iterates the subset of [MODIFIERS][dream_trainer.utils.modifiers.MODIFIERS]
        applicable to `trainer_config.__class__` (preserving registration
        `z_index` order) and for each modifier consults the value parsed by
        typer:

        - `None` -> flag not passed, skip.
        - `True` -> boolean flag passed, call `fn(trainer_config)`.
        - anything else -> call `fn(trainer_config, value)`.

        Args:
            modifiers: The `**modifiers` dict collected by each subcommand.
        """
        applied = []
        for modifier, (fn, _) in local_modifiers.items():
            if modifier not in modifiers:
                continue

            match modifiers[modifier]:
                case None:
                    continue
                case True:
                    fn(trainer_config)
                    applied.append(modifier)
                case _:
                    fn(trainer_config, modifiers[modifier])
                    applied.append(f"{modifier}={modifiers[modifier]}")

        if applied:
            logger.info(f"Running with modifiers: {', '.join(applied)}")
        else:
            logger.info("Running with no modifiers")

    def add_modifiers_to_signature(func: Callable) -> None:
        """Splice modifier flags into `func`'s signature so typer picks them up.

        Typer builds its argument parser from each command's
        `inspect.signature`. Because our subcommands collect modifiers via
        `**modifiers`, typer would otherwise see no modifier flags at all.
        This helper replaces `func.__signature__` with an explicit
        `Signature` object that lists every registered modifier as an
        `Annotated[..., Option(...)]` parameter under the "Modifiers" rich
        help panel, followed by the original command's non-`**kwargs`
        parameters.

        Args:
            func: The subcommand (or default callback) to patch.
        """
        func.__signature__ = Signature(
            [
                Parameter(
                    name,
                    Parameter.POSITIONAL_OR_KEYWORD,
                    annotation=Annotated[
                        type_,
                        Option(
                            f"--{name.replace('_', '-')}",
                            help=modifier.__doc__,
                            rich_help_panel="Modifiers",
                            show_default=False,
                        ),
                    ],
                    default=None,
                )
                for name, (modifier, type_) in local_modifiers.items()
                if name  # Exclude identity modifier (registered under "")
            ]
            + list(signature(func).parameters.values())[:-1]  # Remove **kwargs
        )

    @_cli.callback(invoke_without_command=True)
    def _(
        *,
        ctx: typer.Context,
        resume: Annotated[
            str | None, Option("--resume", help="Experiment to resume from")
        ] = None,
        init_from: Annotated[
            str | None,
            Option(
                "--init-from",
                help="Checkpoint to initialize from. Equivalent to resuming only named_modules",
            ),
        ] = None,
        cfg: Annotated[bool, Option("--cfg", help="Print the config")] = False,
        **modifiers,
    ):
        """Default command: run training end-to-end.

        Invoked when the user runs the script with no subcommand. Applies
        modifiers, optionally prints the resolved config, wires up
        resume/init-from logic via the checkpoint callbacks, and finally
        dispatches to `main` inside the distributed entrypoint.

        Args:
            ctx: Typer context. If a subcommand was also invoked, this
                callback short-circuits and lets the subcommand take over.
            resume: If set, load the named experiment's latest checkpoint
                and continue training from there.
            init_from: If set, initialise model weights from the named
                checkpoint but start training from step 0. Mutually
                exclusive with `resume`.
            cfg: If `True`, print the resolved config (after modifiers
                are applied) and exit without training.
            **modifiers: The auto-generated `--<modifier>` flags, parsed
                into a dict by typer.

        Raises:
            ValueError: If both `resume` and `init_from` are provided.
        """
        if ctx.invoked_subcommand is not None:
            return

        apply_modifiers(modifiers)
        torch_setup()

        if cfg:
            from .cfg import print_config

            return print_config(trainer_config)

        from dream_trainer.utils.entrypoint import entrypoint

        if resume and init_from:
            raise ValueError(
                f"Cannot resume and init from a checkpoint at the same time. Got resume={resume} and init_from={init_from}"
            )

        if init_from:
            from dream_trainer.callbacks import LoadPartialCheckpointCallback

            trainer_config.callbacks.append(
                LoadPartialCheckpointCallback(init_from, resume_mode="last")
            )

        if resume:
            from dream_trainer.callbacks import LoadPartialCheckpointCallback

            try:
                trainer_config.callbacks.pop(LoadPartialCheckpointCallback.__name__)
                logger.info(
                    "Removed LoadPartialCheckpointCallback as we are resuming from a checkpoint"
                )
            except KeyError:
                pass

        trainer_config.experiment = resume or trainer_config.experiment
        entrypoint(main)(trainer_config)

    @_cli.command(rich_help_panel="Debugging")
    def find_graph_breaks(
        output_path: Annotated[
            str,
            Option("--output-path", help="Path to the output file"),
        ] = "graph_breaks.log",
        skip: Annotated[
            int,
            Option(
                "--skip", help="Skip the first N training steps before finding graph breaks"
            ),
        ] = 0,
        fullgraph: Annotated[
            bool,
            Option(
                "--fullgraph",
                help="Whether to compile with full graph. This will raise an exception on the first graph break (usually with a more descriptive error message ",
            ),
        ] = False,
        **modifiers,
    ):
        """Find graph breaks in any compiled regions of the model.

        Installs a [FindGraphBreaksCallback][dream_trainer.callbacks.FindGraphBreaksCallback]
        before running the training loop. Breaks are written to
        `output_path`; pass `--fullgraph` to bail out on the first one
        with a descriptive traceback.

        Args:
            output_path: Where to write the graph-break log.
            skip: Number of training steps to skip before listening for
                breaks (some models only hit unstable regions after warmup).
            fullgraph: If `True`, compile with `fullgraph=True` so the
                first break raises instead of being silently logged.
            **modifiers: Modifier flags injected by `add_modifiers_to_signature`.
        """
        from dream_trainer.callbacks import FindGraphBreaksCallback
        from dream_trainer.utils.entrypoint import entrypoint

        apply_modifiers(modifiers)
        torch_setup()

        trainer_config.callbacks.append(
            FindGraphBreaksCallback(log_file=output_path, skip=skip, fullgraph=fullgraph)
        )
        entrypoint(main)(trainer_config)

    @_cli.command(rich_help_panel="Profiling")
    def importtime(**modifiers):
        """Display a waterfall of import times up to the training script.

        Re-launches the current process with `python -X importtime` under
        a single-rank torchelastic harness and shells out to `tuna` to
        render the result.

        Args:
            **modifiers: Modifier flags injected by `add_modifiers_to_signature`.
        """
        from .import_times import get_importtime

        apply_modifiers(modifiers)
        torch_setup()

        return get_importtime()

    @_cli.command(rich_help_panel="Profiling")
    def benchmark(
        *,
        skip: Annotated[
            int,
            Option(
                "--skip",
                help="Number of training steps to skip before collecting data (warmup)",
            ),
        ] = 8,
        print_every: Annotated[
            int,
            Option(
                "--print-every", help="Print statistics every N training steps (0 to disable)"
            ),
        ] = 1,
        window_size: Annotated[
            int,
            Option("--window-size", help="Size of the sliding window for running statistics"),
        ] = 4,
        **modifiers,
    ):
        """Low-overhead continuous benchmarking of step, forward, backward, and optimizer times.

        Uses CUDA events for accurate GPU timing with minimal overhead
        compared to full profiling. Useful for identifying variance in
        iteration speed.

        Args:
            skip: Number of training steps to skip before collecting
                data (warmup window).
            print_every: Print statistics every N training steps (pass
                `0` to disable periodic printing).
            window_size: Size of the sliding window used to compute
                running statistics.
            **modifiers: Modifier flags injected by `add_modifiers_to_signature`.
        """
        from dream_trainer.callbacks import BenchmarkCallback
        from dream_trainer.utils.entrypoint import entrypoint

        apply_modifiers(modifiers)
        torch_setup()

        trainer_config.callbacks.append(
            BenchmarkCallback(
                skip=skip,
                print_every=print_every,
                window_size=window_size,
            )
        )
        entrypoint(main)(trainer_config)

    @_cli.command(rich_help_panel="Profiling")
    def profile(
        *,
        skip: Annotated[
            int, Option("--skip", help="The number of training steps to skip before profiling")
        ] = 5,
        warmup: Annotated[
            int, Option("--warmup", help="The number of training steps to warmup for")
        ] = 1,
        cycle: Annotated[
            int, Option("--cycle", help="The number of training steps to cycle through")
        ] = 4,
        repeat: Annotated[
            int, Option("--repeat", help="The number of times to repeat cycles")
        ] = 1,
        with_stack: Annotated[
            bool, Option("--with-stack/--without-stack", help="Whether to profile with stack")
        ] = False,
        record_shapes: Annotated[
            bool, Option("--record-shapes", help="Whether to record shapes")
        ] = False,
        profile_memory: Annotated[
            bool, Option("--profile-memory", help="Whether to profile memory")
        ] = False,
        output_dir: Annotated[
            str, Option("--output-dir", help="Path to the output directory")
        ] = "./trace",
        **modifiers,
    ):
        """Run a full `torch.profiler` trace around the training loop.

        Installs a [ProfileCallback][dream_trainer.callbacks.ProfileCallback]
        that drives a `torch.profiler.profile` context with the schedule
        `(wait=skip, warmup=warmup, active=cycle, repeat=repeat)`. The
        collected trace is written to `output_dir/rank_<rank>/`.

        Args:
            skip: Number of training steps to skip before the profiler
                starts (maps to `schedule(wait=skip)`).
            warmup: Number of warmup steps before active collection
                (maps to `schedule(warmup=warmup)`).
            cycle: Number of active training steps per profiler cycle
                (maps to `schedule(active=cycle)`).
            repeat: Number of profiler cycles to run before tearing down.
            with_stack: If `True`, record Python and C++ stack frames.
                Also triggers an `export_stacks` dump.
            record_shapes: If `True`, record operator input shapes.
            profile_memory: If `True`, enable memory profiling and dump
                a memory timeline + snapshot alongside the trace.
            output_dir: Root output directory for trace artefacts.
                Per-rank subdirectories are appended automatically.
            **modifiers: Modifier flags injected by `add_modifiers_to_signature`.
        """
        from functools import partial

        import torch.profiler
        from dream_trainer.callbacks.profile import (
            DEFAULT_MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT,
            DEFAULT_MEMORY_SNAPSHOT_FILENAME,
            DEFAULT_MEMORY_TIMELINE_DEVICE,
            ProfileCallback,
            trace_handler,
        )
        from dream_trainer.utils.entrypoint import entrypoint

        apply_modifiers(modifiers)
        torch_setup()

        schedule = torch.profiler.schedule(
            wait=skip,
            warmup=warmup,
            active=cycle,
            repeat=repeat,
        )

        output_dir = os.path.expanduser(output_dir)
        output_dir = os.path.abspath(output_dir)
        output_dir = os.path.join(output_dir, f"rank_{get_dist_rank()}")

        effective_profile_memory = profile_memory

        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CUDA,
                torch.profiler.ProfilerActivity.CPU,
            ],
            with_stack=with_stack,
            profile_memory=effective_profile_memory,
            schedule=schedule,
            on_trace_ready=partial(
                trace_handler,
                output_dir=output_dir,
                export_stacks=with_stack,
                export_memory_timeline=profile_memory,
                memory_timeline_device=DEFAULT_MEMORY_TIMELINE_DEVICE,
                dump_memory_snapshot=profile_memory,
                memory_snapshot_path=os.path.join(
                    os.path.abspath(output_dir), DEFAULT_MEMORY_SNAPSHOT_FILENAME
                )
                if profile_memory
                else None,
                stop_memory_recording=profile_memory,
            ),
            record_shapes=record_shapes,
            experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True),  # type: ignore
        ) as profiler:
            trainer_config.callbacks.append(
                ProfileCallback(
                    profiler=profiler,
                    memory_snapshot_enabled=profile_memory,
                    memory_snapshot_max_entries=DEFAULT_MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT,
                    memory_snapshot_output_dir=output_dir,
                )
            )
            entrypoint(main)(trainer_config)

    @_cli.command(rich_help_panel="Export")
    def export(
        *,
        checkpoint_path: Annotated[str, Option("--checkpoint-path", help="Checkpoint path")],
        output_path: Annotated[
            str, Option("--output-path", help="Path to the output file")
        ] = "export.pt",
        ignore_frozen_params: Annotated[
            bool, Option("--ignore-frozen-params", help="Whether to ignore frozen params")
        ] = False,
        submodules: Annotated[
            list[str] | None, Option("--submodules", help="Submodules to export")
        ] = [],
        exclude_submodules: Annotated[
            list[str] | None, Option("--exclude-submodules", help="Submodules to exclude")
        ] = [],
        resume_mode: Annotated[
            Literal["min", "max", "last"] | None,
            Option("--resume-mode", help="Resume mode (min, max, last)"),
        ] = None,
        resume_step: Annotated[
            int | None,
            Option("--resume-step", help="Resume step"),
        ] = None,
        overwrite: Annotated[
            bool, Option("--overwrite", help="Whether to overwrite the output file")
        ] = False,
        **modifiers,
    ):
        """Export a checkpoint to a standalone `.pt` file.

        Forces the run onto a single GPU in DDP mode with logging,
        checkpointing and FP8 quantization disabled, then installs an
        [ExportCallback][dream_trainer.callbacks.ExportCallback] that
        loads the selected checkpoint and writes the requested submodule
        state-dict to `output_path`.

        Args:
            checkpoint_path: Experiment directory / DCP checkpoint path to
                load weights from.
            output_path: Destination `.pt` file.
            ignore_frozen_params: If `True`, skip any parameter with
                `requires_grad=False` when building the export.
            submodules: Optional list of submodule names to include
                (empty -> export everything).
            exclude_submodules: Optional list of submodule names to exclude.
            resume_mode: One of `"min"`, `"max"`, `"last"` — selects which
                checkpoint to load based on step. Mutually exclusive with
                `resume_step`.
            resume_step: Specific training step to load. Mutually
                exclusive with `resume_mode`.
            overwrite: If `True`, overwrite `output_path` if it already exists.
            **modifiers: Modifier flags injected by `add_modifiers_to_signature`.

        Raises:
            AssertionError: If neither or both of `resume_mode` /
                `resume_step` are provided.
        """
        import os

        from dream_trainer.callbacks import ExportCallback
        from dream_trainer.configs import DeviceParameters
        from dream_trainer.utils.entrypoint import entrypoint

        apply_modifiers(modifiers)
        torch_setup()

        assert resume_mode is not None or resume_step is not None, (
            "Either resume_mode or resume_step must be provided"
        )
        assert resume_mode is None or resume_step is None, (
            "Either resume_mode or resume_step must be provided, not both"
        )

        trainer_config.callbacks.append(
            ExportCallback(
                checkpoint_path,
                output_path,
                submodules,
                exclude_submodules,
                ignore_frozen_params,
                overwrite,
                resume_mode or resume_step,
            )
        )

        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

        trainer_config.device_parameters = DeviceParameters.DDP()
        trainer_config.logging_parameters.enabled = False
        trainer_config.callbacks.pop("AsyncCheckpointCallback")
        trainer_config.callbacks.pop("CheckpointCallback")
        trainer_config.callbacks.pop("Fp8Quantization")

        entrypoint(main)(trainer_config)

    @_cli.command(rich_help_panel="Debugging")
    def summarize(
        *,
        depth: Annotated[
            int, Option("--depth", help="The depth of the model to summarize")
        ] = 1,
        model_names: Annotated[
            list[str], Option("--model-names", help="The names of the models to summarize")
        ] = [],
        **modifiers,
    ):
        """Print a [ModelSummary][dream_trainer.callbacks.ModelSummary] for the configured models.

        Installs the `ModelSummary` callback, then runs the usual training
        entrypoint. The callback emits its summary during `pre_fit` and
        does not interfere with subsequent steps, so this subcommand also
        serves as a quick "does my config even construct" smoke test.

        Args:
            depth: How deep to recurse into submodule hierarchy before
                collapsing rows.
            model_names: Restrict the summary to the named models. Empty
                list (default) summarises every registered model.
            **modifiers: Modifier flags injected by `add_modifiers_to_signature`.
        """
        from dream_trainer.callbacks import ModelSummary
        from dream_trainer.utils.entrypoint import entrypoint

        apply_modifiers(modifiers)
        torch_setup()

        trainer_config.callbacks.append(ModelSummary(depth=depth, model_names=model_names))

        entrypoint(main)(trainer_config)

    add_modifiers_to_signature(_)
    add_modifiers_to_signature(find_graph_breaks)
    add_modifiers_to_signature(importtime)
    add_modifiers_to_signature(benchmark)
    add_modifiers_to_signature(profile)
    add_modifiers_to_signature(export)
    add_modifiers_to_signature(summarize)

    _cli()
