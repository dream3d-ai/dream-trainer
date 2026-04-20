# Training Parameters API

Module: `dream_trainer.configs.trainer`

`TrainingParameters` controls fit-loop length, validation cadence, gradient accumulation, and gradient clipping.

## Fields

| Field | Purpose |
| --- | --- |
| `n_epochs` | Number of epochs to train. |
| `train_steps_per_epoch` | Explicit train steps per epoch, useful for iterable datasets. |
| `gradient_accumulation_steps` | Number of microbatches per optimizer step. |
| `val_steps_per_epoch` | Explicit validation steps per epoch. |
| `val_every_n_steps` | Step-based validation cadence. |
| `num_sanity_val_steps` | Validation batches to run before training starts. |
| `gradient_clip_val` | Optional gradient norm clipping value. |

## Example

```python
TrainingParameters(
    n_epochs=10,
    train_steps_per_epoch=1_000,
    val_steps_per_epoch=100,
    gradient_accumulation_steps=4,
    num_sanity_val_steps=2,
    gradient_clip_val=1.0,
)
```

Use explicit step counts for iterable datasets or streaming dataloaders.

## See It In Use

- [Configuration](../../configuration.md) — how `TrainingParameters` composes with the rest of `DreamTrainerConfig`.
- [Trainer Guide — `training_step`](../../trainer-guide.md#6-training_step-one-step-of-the-loop) — `gradient_accumulation_steps` and `self.is_accumulating_gradients`.
