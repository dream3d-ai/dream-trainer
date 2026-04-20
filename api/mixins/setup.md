# Setup Mixins API

Modules:

- `dream_trainer.trainer.mixins.setup`
- `dream_trainer.trainer.mixins.setup.models`
- `dream_trainer.trainer.mixins.setup.optimizers`
- `dream_trainer.trainer.mixins.setup.dataloader`

Setup mixins provide the default configure-and-setup flow used by `DreamTrainer`.

## Public Classes

| Class | Purpose |
| --- | --- |
| `SetupConfigMixin` | Config contract for setup behavior. |
| `SetupMixin` | Coordinates model, optimizer, scheduler, dataloader, and metric setup. |
| `ModelSetupConfigMixin` | Config support for model setup. |
| `ModelSetupMixin` | Tracks models and materializes meta-device modules. |
| `OptimizerAndSchedulerSetupConfigMixin` | Config support for optimizers and schedulers. |
| `OptimizerAndSchedulerSetupMixin` | Tracks optimizers and scheduler mappings. |
| `DataLoaderSetupConfigMixin` | Config support for dataloaders. |
| `DataLoaderSetupMixin` | Tracks train and validation dataloaders. |

## Setup Order

Dream Trainer setup follows this order:

1. Configure models on the meta device.
2. Materialize model parameters and buffers.
3. Run `init_weights`.
4. Apply parallelism and compile hooks.
5. Configure optimizers.
6. Configure schedulers.
7. Configure dataloaders.
8. Configure metrics.

This order lets large models be declared without immediate allocation while keeping user initialization and optimizer creation explicit.

## See It In Use

- [Core Concepts — The Lifecycle](../../core-concepts.md#the-lifecycle) — the canonical lifecycle diagram.
- [Core Concepts — Meta Device, Then Real Device](../../core-concepts.md#meta-device-then-real-device) — why `configure_models` runs before materialization.
