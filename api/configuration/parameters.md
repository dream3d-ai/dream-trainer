# Configuration Parameters API

Modules:

- `dream_trainer.configs`
- `dream_trainer.trainer.dream`
- `dream_trainer.trainer.base`

Dream Trainer uses dataclasses for experiment and runtime configuration.

## Primary Config Classes

| Class | Import | Purpose |
| --- | --- | --- |
| `DreamTrainerConfig` | `from dream_trainer import DreamTrainerConfig` | Default config base for user trainers. |
| `BaseTrainerConfig` | `from dream_trainer.trainer import BaseTrainerConfig` | Config base for the concrete lifecycle engine. |
| `AbstractTrainerConfig` | `from dream_trainer.trainer import AbstractTrainerConfig` | Lowest-level trainer config contract. |
| `WandbLoggingParameters` | `from dream_trainer.configs import WandbLoggingParameters` | Toggle WandB-backed logging behavior. |
| `FaultToleranceParameters` | `from dream_trainer.configs import FaultToleranceParameters` | Configure optional torchft fault tolerance. |

## Common Pattern

```python
@dataclass(kw_only=True)
class MyConfig(DreamTrainerConfig):
    learning_rate: float = 3e-4
    batch_size: int = 64
```

Keep runtime setup out of dataclass defaults. Create CUDA tensors, modules, optimizers, and dataloaders inside trainer hooks after Dream Trainer has configured the distributed world.

## See It In Use

- [Configuration](../../configuration.md) — the how-to guide on writing configs, named variants, and the config/trainer boundary.
- [Tutorial 1 — The Config](../../tutorials/first-trainer.md#the-config) — a minimal `DreamTrainerConfig` subclass.
