# Common Utilities API

Modules:

- `dream_trainer.utils.entrypoint`
- `dream_trainer.utils.materialize`
- `dream_trainer.utils.serialize`
- `dream_trainer.utils.names`
- `dream_trainer.utils.dataloader`
- `dream_trainer.utils.common`
- `dream_trainer.utils.logging`
- `dream_trainer.utils._logger`

Utilities support launch, materialization, serialization, names, dataloader state, and logging.

## Important Utilities

| Utility | Purpose |
| --- | --- |
| `entrypoint` | Decorator that uses an existing distributed environment or launches local ranks from visible CUDA devices. |
| materialization helpers | Support meta-device model setup and parameter materialization. |
| serialization helpers | Support structured state save and load behavior. |
| dataloader helpers | Support dataloader state and iterable behavior. |
| name helpers | Support stable component names for tracking. |
| `logger` | Package logger that can be overridden by applications. |

## Entrypoint Usage

```python
from dream_trainer.utils.entrypoint import entrypoint


@entrypoint
def main():
    Trainer(config).fit()
```

For single-device configs, keep one CUDA device visible. For local multi-GPU configs, expose the devices you want Dream Trainer to launch.

## See It In Use

- [Parallelism — How Launches Work](../../parallelism.md#how-launches-work) — how `entrypoint` adapts to torchrun, Slurm, or bare CUDA visibility.
- [Tutorial 1 — The Entrypoint](../../tutorials/first-trainer.md#the-entrypoint) — `@entrypoint` wrapping `main()`.
