# Logging And Metrics

!!! abstract "TL;DR"
    - **Trainer returns scalars. Callbacks log them.** Don't call `wandb.log` directly from your trainer.
    - Assign `MetricCollection` / `Metric` objects in `configure_metrics` — Dream Trainer moves, resets, and computes them for you.
    - Default stack: `LoggerCallback` (scalars) + `LRLoggerCallback` (learning rates) + `ProgressBar` (local) + `MediaLoggerCallback` (images/video).
    - Logging runs on global rank 0 only — keep media sample-limited.

Dream Trainer separates metric **computation** from metric **logging**. Trainers update metrics and return scalar logs; callbacks decide where those values go.

## WandB logging

The default `DreamTrainer` includes `WandBLoggerMixin`. Configure it with `WandbLoggingParameters`:

```python
logging_parameters=WandbLoggingParameters(enabled=False)
```

!!! tip "Disable locally, enable remotely"
    Set `enabled=False` for local smoke tests. When enabled, WandB runs **only on global rank zero** — no special per-rank handling required.

## Scalar logs

Return scalar tensors, floats, or ints from `training_step` and `validation_step`:

```python
return {
    "train/loss": loss,
    "train/grad_norm": grad_norm,
}
```

`LoggerCallback` filters returned dictionaries to numeric scalar values and calls the trainer's `log_dict` method.

```python
callbacks.LoggerCallback(log_every_n_train_batches=8)
```

!!! warning "Throughput note"
    Use a lower logging frequency for fast training loops to avoid per-step logging overhead. `log_every_n_train_batches=8` is a reasonable baseline.

## Metrics

Assign `MetricCollection` or `Metric` objects as trainer attributes in `configure_metrics`:

```python
def configure_metrics(self):
    self.metrics = self.config.metrics
```

Dream Trainer tracks those attributes, moves them to `self.world.device`, resets them before validation, and computes them after validation.

Validation output combines step-returned logs with computed metrics:

```python
def validation_step(self, batch, batch_idx):
    pred = self.model(batch["x"])
    self.metrics.update(pred, batch["y"])
    return {"val/loss": loss}
```

Metric names are prefixed by the trainer attribute name. A collection assigned to `self.metrics` produces names like `metrics/mse`.

!!! danger "Assign metrics only in `configure_metrics`"
    Assigning `self.metrics = ...` from another hook prevents auto-tracking and silently breaks reset and compute boundaries.

## Logging callbacks at a glance

| Callback | Purpose |
| --- | --- |
| `LoggerCallback` | Records scalar step/epoch values. Optionally logs source code via `code_dir=...`. |
| `LRLoggerCallback` | Logs optimizer learning rates each step. |
| `MediaLoggerCallback` | Logs images, videos, and figure artifacts from validation. |
| `MetricLoggerCallback` | Publishes computed metric-collection outputs to the logger mixin. |
| `ModelWatchCallback` | WandB `wandb.watch` integration for weights/gradient histograms. |
| `ProgressBar` | Local-only progress with a chosen scalar metric. |

## Media logs

Use `MediaLoggerCallback` and logger mixin media methods for image or video outputs. Validation steps often return media pairs for a callback to consume:

```python
return {"samples": (target, prediction)}
```

!!! warning "Keep media sample-limited"
    Large image or video logging can dominate validation time. Cap sample counts and log only at checkpoint boundaries, not every batch.

## Code logging

`LoggerCallback(code_dir="../")` can log Python source to WandB. Use a project root or training package directory and rely on `.gitignore` patterns to avoid uploading generated files.

## Local progress

```python
callbacks.ProgressBar(metric="train/loss")
```

The progress bar reads scalar values from the step result and shows separate loop and dataloader latency estimates.

## Common mistakes

??? bug "Returning non-scalar tensors from step logs"
    `LoggerCallback` filters for numeric scalars. Returning a `[B, H]` tensor does not get logged. Aggregate first (`loss.mean()`, `metric.compute()`).

??? bug "Forgetting `self.metrics.update(...)` in validation"
    Metric output values are zero — and no callback warns. Update before returning.

??? bug "Assigning metrics outside `configure_metrics`"
    Auto-tracking, move-to-device, and reset boundaries all stop working silently.

??? bug "Enabling WandB with no credentials or network path"
    `wandb.init` blocks with a login prompt. Either set up credentials, set `WandbLoggingParameters(enabled=False)`, or use offline mode.

??? bug "Logging media every batch"
    Dominates validation time. Log at checkpoint boundaries only.

## Next steps

- [Callbacks](callbacks.md) — lifecycle integration of logger callbacks.
- [Configuration](configuration.md) — setting `logging_parameters` on the trainer config.
