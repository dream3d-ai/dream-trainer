# Using the CLI

!!! abstract "TL;DR"
    - `cli()` is a typer-based drop-in for `@entrypoint` that adds subcommands (`benchmark`, `profile`, `summarize`, `find-graph-breaks`, `importtime`, `export`) and turns every registered modifier into a `--<flag>`.
    - Install once with `pip install "dream-trainer[cli]"` and swap `@entrypoint(MyConfig)` for `cli(main, MyConfig())`.
    - Root flags: `--cfg` (print resolved config, no run), `--resume <experiment>` (continue from last checkpoint), `--init-from <path>` (warm-start weights only).
    - Rule of thumb: `@entrypoint` for simple scripts; `cli()` when you want ad-hoc diagnostics or resume without editing code.

`cli()` wraps the same `def main(config)` function `@entrypoint` takes. The training path stays identical — you get a command surface on top of it.

## When to use it

| Use | Launcher |
| --- | --- |
| Single-file script, one launch shape | `@entrypoint` |
| Ad-hoc benchmark / profile / graph-break runs | `cli()` |
| Resume or warm-start without editing the trainer | `cli()` |
| Flip debug knobs (no-compile, no-log, no-ckpt, grad-accum) per-run | `cli()` |

!!! tip "Opt-in via extra"
    The CLI pulls `typer` and a handful of formatting helpers. Install on demand:

    === "pip"

        ```bash
        pip install "dream-trainer[cli]"
        ```

    === "uv"

        ```bash
        uv add "dream-trainer[cli]"
        ```

## Quick start

Swap `@entrypoint` for `cli()` in your training script:

=== "With `cli()`"

    ```python
    from dream_trainer.utils.cli import cli

    def main(config: MyTrainerConfig) -> None:
        MyTrainer(config).fit()

    if __name__ == "__main__":
        cli(main, MyTrainerConfig())
    ```

=== "With `@entrypoint`"

    ```python
    from dream_trainer.utils.entrypoint import entrypoint

    @entrypoint
    def main() -> None:
        MyTrainer(MyTrainerConfig()).fit()

    if __name__ == "__main__":
        main()
    ```

`cli()` wraps `main` with `@entrypoint` internally, so the distributed setup is identical. List the generated surface:

```bash
python train.py --help
```

## Root-level flags

The default command runs training. Its flags let you inspect, resume, or warm-start without touching code.

### `--cfg` — inspect the resolved config

```bash
python train.py --cfg
```

Applies any modifiers you passed, prints the resulting config, and exits without training. Useful for reviewing what a modifier chain actually produces.

### `--resume <experiment>` — continue training

```bash
python train.py --resume my-experiment-2024-04-19
```

Loads the latest checkpoint for the named experiment and continues. If an earlier run crashed mid-epoch, this is the fastest way back.

!!! note "Experiment vs. path"
    `--resume` takes an **experiment name**, not a checkpoint directory. Dream Trainer's checkpoint layout resolves the path from `root_dir / project / group / experiment`.

### `--init-from <path>` — warm-start weights only

```bash
python train.py --init-from /checkpoints/base-run/checkpoints
```

Loads model weights from the checkpoint via [`LoadPartialCheckpointCallback`](checkpointing.md#partial-loading) but starts training from step 0 — no optimizer, scheduler, dataloader, or counter state. The fine-tune warm-start pattern.

!!! danger "Mutually exclusive"
    Passing both `--resume` and `--init-from` raises. Resume loads *all* trainer state; init-from loads *only* model weights.

## Subcommands

Each subcommand installs a single callback and runs `fit()` normally. Modifier flags work on every subcommand.

### Debugging

**`summarize`** — print a model summary via [`ModelSummary`](api/callbacks/performance.md), then run training. Doubles as a "does my config construct" smoke test.

```bash
python train.py summarize --depth 2
```

**`find-graph-breaks`** — log `torch.compile` graph breaks to a file. Pass `--fullgraph` to raise on the first break with a readable traceback.

```bash
python train.py find-graph-breaks --output-path graph_breaks.log --skip 5
```

Same behavior as adding [`FindGraphBreaksCallback`](debugging.md#finding-graph-breaks) to your callback collection — no code edit required.

### Profiling

**`benchmark`** — low-overhead CUDA-event timing of step / forward / backward / optimizer. Equivalent to the [`BenchmarkCallback`](performance.md#profiling) recipe.

```bash
python train.py benchmark --skip 8 --print-every 1 --window-size 4
```

**`profile`** — full `torch.profiler` trace around the training loop. Writes per-rank traces to `output-dir/rank_<n>/`. Equivalent to the [`ProfileCallback`](performance.md#profiling) recipe.

```bash
python train.py profile --skip 5 --warmup 1 --cycle 4 --repeat 1 --with-stack
```

**`importtime`** — waterfall of Python import times up to the training script, rendered via `tuna`. Use this when startup latency is the bottleneck.

```bash
python train.py importtime
```

### Export

**`export`** — offline checkpoint conversion. Forces a single-rank DDP run with logging and checkpointing disabled, loads the selected checkpoint, and writes a standalone `.pt` file.

```bash
python train.py export \
  --checkpoint-path /checkpoints/prod-run/checkpoints \
  --output-path model.pt \
  --resume-mode last
```

Pass `--submodules` / `--exclude-submodules` to filter, and `--ignore-frozen-params` to skip non-trainable state.

## Modifiers

Modifiers are tiny functions that mutate the config before `main` runs. Each is surfaced as a kebab-case flag on every command.

### Built-in modifiers

| Flag | Effect |
| --- | --- |
| `--no-compile` | Disable `torch.compile`, compiled autograd, and async TP. |
| `--no-log` | Disable all logging backends. |
| `--no-ckpt` | Drop `CheckpointCallback` and `AsyncCheckpointCallback`. |
| `--no-sanity` | Skip sanity validation. |
| `--no-fp8` | Remove `Fp8Quantization` from the callback stack. |
| `--ckpt-acts` | Turn on activation checkpointing. |
| `--cpu-offload` | Offload optimizer state (and configured parameters) to CPU. |
| `--single-device` | Force a single-device run while preserving `compile_model` / `cpu_offload` / `checkpoint_activations`. |
| `--force-ddp` | Force DDP on a single device (collective debugging). |
| `--force-fsdp` | Force FSDP on a single device. |
| `--grad-accum-steps N` | Set `gradient_accumulation_steps` to `N`. |
| `--detect-anomaly` | Turn on `torch.autograd.set_detect_anomaly`. |
| `--warn-on-sync` | Warn on device-to-host synchronisation. |
| `--display-fsdp-call-order` | Enable the `OptimizeFSDP` debug display. |

Stack modifiers as needed:

```bash
python train.py --no-compile --no-log --grad-accum-steps 4 profile --skip 5
```

!!! tip "Modifier application order"
    Modifiers run in registration `z_index` order, not CLI order. If two modifiers touch the same field, the higher `z_index` wins. Set `z_index=` on `@register_modifier` when ordering matters.

### Adding your own

Register a modifier against your trainer config — importing the module is enough:

```python
from dream_trainer import DreamTrainerConfig
from dream_trainer.utils.modifiers import register_modifier


@register_modifier()
def lr(config: DreamTrainerConfig, value: float):
    """Override the learning rate."""
    config.optimizer.lr = value
```

- `fn(config)` → boolean flag (`--my-flag`).
- `fn(config, value: T)` → value flag (`--my-flag 1.5`).

The name kebab-cases automatically (`lr` → `--lr`, `grad_accum_steps` → `--grad-accum-steps`). See [Configuration](configuration.md) for where modifiers fit in the broader config story.

## Full reference

- [API Reference → CLI Utilities](api/utilities/cli.md) — full typer surface and registry internals.
- [Debugging](debugging.md) — when to reach for `find-graph-breaks`.
- [Performance](performance.md) — when to reach for `benchmark` / `profile`.
- [Checkpointing](checkpointing.md) — how `--resume` and `--init-from` interact with checkpoint state.
