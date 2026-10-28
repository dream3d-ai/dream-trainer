# Logging And Metrics

!!! abstract "TL;DR"
    - **Trainer returns scalars. Callbacks log them.** Don't call `wandb.log` directly from your trainer.
    - Assign `MetricCollection` / `Metric` objects in `configure_metrics` — Dream Trainer moves, resets, and computes them for you.
    - Default stack: `LoggerCallback` (scalars) + `LRLoggerCallback` (learning rates) + `ProgressBar` (local) + `MediaLoggerCallback` (images/video).
    - Training scalars are reduced across microbatches and data-parallel ranks; only global rank 0 writes logs.

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

`LoggerCallback` filters returned dictionaries to numeric scalar values. By default,
it logs once after each completed gradient update, averaging all microbatches in that
update across data-parallel ranks. For example, with 128 accumulation microbatches,
128 returned losses become one mean loss at `trainer/global_step=1`.

Each metric has its own observation count: a gradient norm returned only at the
update boundary is not divided by 128. Accumulation uses detached FP32 values,
and host conversion and data-parallel reduction happen at logging time.

```python
callbacks.LoggerCallback()  # Every completed gradient update
callbacks.LoggerCallback(log_every_n_train_steps=8)  # Mean over 8 updates
```

!!! warning "Throughput note"
    Use a larger `log_every_n_train_steps` for fast training loops. The old
    `log_every_n_train_batches` keyword remains as a deprecated alias, but its
    interval now counts gradient updates rather than microbatches.

Returned scalars are treated as equally weighted observations, matching the
trainer's equal weighting of microbatch losses during backward. This is not a
token-weighted or sample-weighted average for variable-size batches. Every scalar
uses mean reduction, including metrics named `min` or `max`; these become means
of the reported batch extrema. Missing keys are excluded from that metric's count.

An epoch ending on an update boundary flushes any shorter logging interval.
An unfinished accumulation window is not logged as a completed update. The last
training microbatch is no longer emitted again as an epoch metric. Validation
logging and direct `trainer.log_dict(...)` calls keep their existing behavior.

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
| `LoggerCallback` | Records averaged training-update scalars and validation values. Optionally logs source code via `code_dir=...`. |
| `LRLoggerCallback` | Logs rates used by the optimizers once per completed update, before scheduler changes. |
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
