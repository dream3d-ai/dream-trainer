# Dream Trainer API

Module: `dream_trainer.trainer.dream`

`DreamTrainer` is the default trainer class for user code. It combines the base lifecycle with setup, metrics, WandB logging, and quantization support.

## Public Classes

| Class | Purpose |
| --- | --- |
| `DreamTrainerConfig` | Default config base for Dream Trainer subclasses. |
| `DreamTrainer` | Main user-facing trainer base class. |

## Typical Usage

```python
from dream_trainer import DreamTrainer, DreamTrainerConfig


class MyTrainer(DreamTrainer):
    config: DreamTrainerConfig
```

Subclass `DreamTrainer` when you want:

- meta-device model configuration
- explicit weight initialization after materialization
- optimizer and scheduler tracking
- rank-aware dataloader setup
- metric registration and reset behavior
- WandB-compatible logging hooks
- quantization setup hooks
- callback-driven checkpointing, profiling, progress, and logging

## Required Hooks

Most trainers implement:

- `configure_models`
- `init_weights`
- `model_state_dict`
- `configure_optimizers`
- `configure_dataloaders`
- `configure_metrics`
- `training_step`
- `validation_step`

Distributed trainers may also implement:

- `apply_replicate`
- `apply_fully_shard`
- `apply_tensor_parallel`
- `apply_context_parallel`
- `apply_pipeline_parallel`
- `apply_activation_checkpointing`
- `apply_compile`

## See It In Use

- [Trainer Guide](../../trainer-guide.md) — hook-by-hook reference with copy-paste examples.
- [Tutorial 1 — Meet Meteor](../../tutorials/first-trainer.md) — a minimal `DreamTrainer` subclass on a single GPU.
- [Tutorial 4 — Production Shape](../../tutorials/production.md) — the same trainer split across config/train/data/model modules.
