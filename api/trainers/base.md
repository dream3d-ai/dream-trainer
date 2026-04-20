# Base Trainer API

Module: `dream_trainer.trainer.base`

`BaseTrainer` implements the core training lifecycle: configuration, setup, sanity validation, training epochs, validation epochs, callbacks, gradient accumulation, optimizer stepping, state dicts, and checkpoint integration.

## Public Classes

| Class | Purpose |
| --- | --- |
| `BaseTrainerConfig` | Configuration base that includes device, training, logging, callback, and fault-tolerance parameters. |
| `BaseTrainer` | Concrete lifecycle engine used by higher-level trainer classes. |

## Core Methods

| Method | Purpose |
| --- | --- |
| `fit` | Run setup, sanity validation, training, validation, callbacks, and teardown. |
| `validate` | Run validation outside or inside the fit loop. |
| `backward` | Apply trainer-managed backward behavior, including accumulation scaling. |
| `step` | Validate gradients, clip gradients, step optimizer, zero gradients, and update schedulers. |
| `state_dict` | Collect model, optimizer, scheduler, callback, trainer, and dataloader state. |
| `load_state_dict` | Restore checkpointed trainer state. |

## User-Implemented Methods

Subclasses are expected to provide model-specific behavior through hooks:

- `configure_models`
- `init_weights`
- `model_state_dict`
- `configure_optimizers`
- `configure_dataloaders`
- `training_step`
- `validation_step`

## Notes

`BaseTrainer` is still framework code. Most projects should subclass `DreamTrainer`, which composes `BaseTrainer` with setup, metric, logger, and quantization mixins.

## See It In Use

- [Core Concepts — The Lifecycle](../../core-concepts.md#the-lifecycle) — the lifecycle diagram `fit()` follows.
- [Trainer Guide](../../trainer-guide.md) — how user-implemented hooks fit into the lifecycle.
