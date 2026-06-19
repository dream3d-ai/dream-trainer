# Evaluation Metric Mixins API

Module: `dream_trainer.trainer.mixins.eval_metric`

Metric mixins integrate `torchmetrics` metric collections into the trainer lifecycle.

## Public Classes

| Class | Purpose |
| --- | --- |
| `EvalMetricConfigMixin` | Config contract for metric setup. |
| `EvalMetricMixin` | Tracks, moves, resets, and computes validation metrics. |

## Usage

```python
def configure_metrics(self):
    self.metrics = self.config.metrics
```

Metrics assigned as trainer attributes are handled by the trainer lifecycle. Update metrics in `validation_step`, then let Dream Trainer reset and compute them at the correct validation boundaries.

## Notes

The metric mixin depends on `torchmetrics`. Install the metrics extra when using `DreamTrainer` through the current top-level import path.

## See It In Use

- [Tutorial 5 — Custom Metrics](../../tutorials/custom-components.md) — a custom `Perplexity` metric wired through `configure_metrics`.
- [Logging And Metrics](../../logging-metrics.md) — how metric results surface through logger callbacks.
