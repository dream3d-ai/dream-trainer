# Checkpoint Parameters API

Module: `dream_trainer.configs.checkpoint`

`CheckpointParameters` configures checkpoint save cadence, resume behavior, retention, and load strictness.

## Fields

| Field | Purpose |
| --- | --- |
| `enable` | Enables or disables checkpoint behavior. |
| `root_dir` | Directory where checkpoint state is written. |
| `resume_mode` | Resume from `"last"`, `"min"`, or `"max"` according to `monitor`. |
| `monitor` | Metric key used for best-checkpoint selection. |
| `checkpoint_every_n_train_epochs` | Save cadence by train epoch. |
| `checkpoint_every_n_val_epochs` | Save cadence by validation epoch. |
| `keep_top_k` | Number of checkpoint replicas retained. |
| `strict_load` | Whether load should fail on unmatched state. |
| `model_weights_only` | Whether model-only behavior is used for weight loads. |
| `pin_memory` | Whether checkpoint loading pins memory. |
| `resume_data` | Whether dataloader or dataset state should be restored. |

## Example

```python
CheckpointParameters(
    root_dir="/checkpoints/run",
    monitor="val/loss",
    resume_mode="min",
    checkpoint_every_n_val_epochs=1,
    keep_top_k=3,
    strict_load=False,
    resume_data=True,
)
```

Exactly one of `checkpoint_every_n_train_epochs` or `checkpoint_every_n_val_epochs` should be set when checkpointing is enabled.

## See It In Use

- [Checkpointing](../../checkpointing.md) — full DCP state layout, resume modes, partial load.
- [Callbacks — Checkpoint Callback API](../callbacks/checkpoint.md) — the callbacks that consume these parameters.
- [Tutorial 4 — `config.py`](../../tutorials/production.md#configpy) — a real `CheckpointParameters` in a production config.
