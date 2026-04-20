# Checkpoint Callback API

Modules:

- `dream_trainer.callbacks.checkpoint.base`
- `dream_trainer.callbacks.checkpoint.async_`
- `dream_trainer.callbacks.checkpoint.partial`
- `dream_trainer.callbacks.checkpoint.export`

Checkpoint callbacks save, load, partially load, and export trainer state through PyTorch Distributed Checkpoint.

## Public Classes

| Class | Purpose |
| --- | --- |
| `CheckpointCallback` | Saves and resumes full trainer checkpoints. |
| `AsyncCheckpointCallback` | Writes checkpoints asynchronously and handles cleanup in the background. |
| `LoadPartialCheckpointCallback` | Loads model weights or partial state for fine-tuning and initialization. |
| `ExportCallback` | Exports supported modules from checkpoint state. |

## Configuration

Checkpoint callbacks use `CheckpointParameters` from `dream_trainer.configs`.

```python
CheckpointParameters(
    root_dir="/checkpoints/run",
    monitor="val/loss",
    checkpoint_every_n_val_epochs=1,
    keep_top_k=3,
)
```

## Guidance

- Use full checkpointing for normal resume.
- Use async checkpointing when checkpoint writes block training.
- Use partial loading for fine-tuning, model surgery, or pretrained initialization.
- Keep `model_state_dict` keys stable across runs.

## See It In Use

- [Checkpointing](../../checkpointing.md) — full DCP state layout, resume modes, partial load.
- [Tutorial 4 — Why `AsyncCheckpointCallback` Now](../../tutorials/production.md#why-asynccheckpointcallback-now) — when to switch from sync to async.
- [Configuration — Checkpoint Parameters](../configuration/checkpoint.md) — field-by-field reference.
