# Installation

!!! abstract "TL;DR"
    - `pip install "dream-trainer[metrics,wandb]"` for the default trainer path.
    - Add extras — `rich`, `torchao`, `torchft`, `cli` — only when you need them.
    - Launch with `torchrun` or Dream Trainer's `@entrypoint` helper.

## Requirements

| Requirement | Version |
| --- | --- |
| Python | 3.10+ |
| PyTorch | 2.7.1+ |
| Launcher | `torchrun` or `@entrypoint` |

## Package install

=== "pip"

    ```bash
    pip install "dream-trainer[metrics,wandb]"
    ```

=== "uv"

    ```bash
    uv add "dream-trainer[metrics,wandb]"
    ```

!!! note "Why `wandb` by default?"
    The current top-level trainer composition imports the WandB mixin. Installing the `wandb` extra avoids import-time surprises even if you disable logging via `WandbLoggingParameters(enabled=False)`.

### Optional extras

| Extra | When to install |
| --- | --- |
| `rich` | Richer trainer and model summaries. |
| `torchao` | FP8 and low-precision quantization workflows. |
| `torchft` | Fault-tolerant training with `FaultToleranceCallback`. |
| `cli` | Typer-based command surface — `benchmark`, `profile`, modifiers. See [Using the CLI](cli.md). |

```bash
pip install "dream-trainer[rich,torchao]"
```

!!! warning "`torchft` packaging"
    Verify the package metadata exposes the `torchft` extra before relying on `dream-trainer[torchft]` — the underlying package is installed alongside Dream Trainer rather than through the extra in some releases.

## Local workspace install

From the monorepo, use the project tooling rather than a bare `pip install`:

=== "All extras"

    ```bash
    uv sync --all-extras
    ```

=== "Docs only"

    ```bash
    uv sync --group docs
    ```

    Installs MkDocs, the `mkdocs-shadcn` theme, mkdocstrings, and the small build plugins used by `mkdocs.yml`.

## Sanity check

Confirm the runtime can import the pieces the quick start needs:

```bash
python - <<'PY'
import torch
import dream_trainer
from dream_trainer import DreamTrainer, DreamTrainerConfig
from dream_trainer.configs import DeviceParameters, TrainingParameters

print("torch:", torch.__version__)
print("dream_trainer:", dream_trainer.__file__)
print("single device:", DeviceParameters.SINGLE_DEVICE(compile_model=False))
PY
```

## How launch works

Dream Trainer's `@entrypoint` helper inspects the process environment:

- **Already distributed** (`torchrun`, Slurm, etc.) — calls your `main()` directly.
- **Bare Python** — spawns a local launch across the visible devices.

Match the launch shape to your `DeviceParameters`:

| Launch shape | `DeviceParameters` |
| --- | --- |
| Single rank | `DeviceParameters.SINGLE_DEVICE()` |
| Multi-GPU, no sharding | `DeviceParameters.DDP()` |
| Sharded | `DeviceParameters.FSDP()` |
| Hybrid sharded | `DeviceParameters.HSDP(...)` |

!!! tip "Richer launch surface"
    For subcommands (`benchmark`, `profile`, `summarize`), config modifiers, and `--resume` / `--init-from` flags, see [Using the CLI](cli.md).

## Troubleshooting

??? question "`ModuleNotFoundError: torchmetrics`"
    Install the metrics extra. The default `DreamTrainer` includes `EvalMetricMixin`, so metric support is part of the common path.

    ```bash
    pip install "dream-trainer[metrics]"
    ```

??? question "`ModuleNotFoundError: wandb`"
    Install the WandB extra. You can still disable logging with `WandbLoggingParameters(enabled=False)`.

    ```bash
    pip install "dream-trainer[wandb]"
    ```

## Next steps

- [Quick Start](getting-started.md) — build the smallest useful trainer.
- [Core Concepts](core-concepts.md) — lifecycle, mixins, and callbacks.
- [Using the CLI](cli.md) — once `[cli]` is installed.
