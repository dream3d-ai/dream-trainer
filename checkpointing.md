# Checkpointing

!!! abstract "TL;DR"
    - Dream Trainer checkpoints **full trainer state** — model, optimizer, scheduler, dataloader, callback, trainer counters — via PyTorch Distributed Checkpoint (DCP).
    - Implement `model_state_dict` with `get_model_state_dict(...)` so the same code works single-GPU and sharded.
    - Use `CheckpointCallback` for synchronous saves; `AsyncCheckpointCallback` when writes stall training; `LoadPartialCheckpointCallback` for fine-tune warm-starts.
    - Pick a `monitor=` metric that your step hooks actually return.

Dream Trainer checkpointing is built around complete trainer state and PyTorch Distributed Checkpoint. The goal is to resume training with the same model, optimizer, scheduler, dataloader, callback, and trainer counters whenever those states are available.

## State layout

`BaseTrainer.state_dict()` returns a nested structure:

```python
{
    "trainer": {
        "global_step": ...,
        "current_epoch": ...,
        "local_batches": ...,
        "callbacks": ...,
    },
    "models": ...,
    "optimizers": ...,
    "schedulers": ...,
    "dataloaders": ...,
}
```

| Section | Where it comes from |
| --- | --- |
| `trainer` | Trainer counters + callback `state_dict()` aggregates. |
| `models` | Your `model_state_dict` hook. |
| `optimizers` | DCP helper using the model-to-optimizer mapping from `configure_optimizers`. |
| `schedulers` | Optimizer-associated scheduler state. |
| `dataloaders` | Any dataloader that exposes `state_dict` / `load_state_dict`. |

## Implementing `model_state_dict`

Use DCP helpers so the same code works single-GPU and sharded.

=== "Single model"

    ```python
    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict


    def model_state_dict(self, **_):
        return {"model": get_model_state_dict(self.model, options=StateDictOptions())}
    ```

=== "Multiple models"

    ```python
    def model_state_dict(self, **_):
        return {
            "encoder": get_model_state_dict(self.encoder, options=StateDictOptions()),
            "decoder": get_model_state_dict(self.decoder, options=StateDictOptions()),
        }
    ```

!!! warning "Don't use `model.state_dict()` directly"
    For sharded models, plain `model.state_dict()` collects local shards only. Use `get_model_state_dict` so DCP can save and load them consistently across mesh shapes.

## `CheckpointCallback`

`CheckpointCallback` saves and loads full trainer state.

```python
callbacks.CheckpointCallback(
    CheckpointParameters(
        enable=True,
        root_dir="/checkpoints",
        monitor="val/loss",
        checkpoint_every_n_val_epochs=1,
        keep_top_k=5,
        resume_mode="last",
    )
)
```

The checkpoint path is built from `root_dir / project / group / experiment / checkpoints`.

## Choosing a monitor

The monitored metric **must** be present in the latest train step or validation epoch result. If the callback can't find it when it needs to save, it raises.

Valid sources:

- Scalar keys returned from your step hooks: `return {"train/loss": loss}`.
- Metric-collection outputs computed in validation — Dream Trainer appends these with names like `metrics/mse` when the metric attribute is named `metrics`.

!!! danger "Monitor-not-found is a startup-time error"
    Mismatched monitor keys are not caught until the first save/resume. Grep for the name in your trainer before shipping.

## Resume modes

| `resume_mode` | Behavior |
| --- | --- |
| `"last"` | Resume the newest checkpoint by step. |
| `"min"` | Resume the checkpoint with the lowest monitored metric. |
| `"max"` | Resume the checkpoint with the highest monitored metric. |
| `int` | Select a specific step (when supported by checkpoint discovery). |

## Data resumption

`resume_data=True` restores dataloader state when dataloaders expose `state_dict` / `load_state_dict`.

!!! tip "When to disable data resumption"
    Set `resume_data=False` when you intentionally change datasets between runs or only want model and optimizer continuity. Leaving it on after a dataset layout change produces silent non-determinism.

## Async checkpointing

`AsyncCheckpointCallback` starts save work in the background and waits for outstanding saves before starting another save or closing. Use it when checkpoint writes are large enough to affect training throughput.

## Partial loading

Use `LoadPartialCheckpointCallback` to load model weights **without** restoring trainer counters, optimizer state, scheduler state, or dataloader state — the fine-tune warm-start pattern.

```python
callbacks.LoadPartialCheckpointCallback("/checkpoints/source-run/checkpoints")
```

## Strict loading

`CheckpointParameters(strict_load=False)` allows partial loads through DCP's load planner. Use strict loading when you expect exact model and optimizer state matches.

## Common mistakes

??? bug "Forgetting `model_state_dict`"
    Checkpoints cannot save model state correctly. Tests pass but `trainer.fit()` resume loses weights.

??? bug "Monitoring a metric that is never returned or computed"
    The callback raises at save time — not at trainer startup. Grep for the monitored key.

??? bug "Reusing `resume_mode=\"last\"` with a stale experiment name"
    You'll silently resume a previous run's checkpoint you didn't mean to continue.

??? bug "Restoring dataloader state after changing the dataset layout"
    The step counter resumes but the iteration order no longer matches. Disable `resume_data` on dataset changes.

??? bug "Using plain `model.state_dict()` for sharded models"
    Collects local shards only. Always go through `get_model_state_dict(...)`.

## Next steps

- [Callbacks](callbacks.md) — lifecycle integration of checkpoint callbacks.
- [Debugging](debugging.md) — checkpoint-related failure modes.
