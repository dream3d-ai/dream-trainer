# Logging Callback API

Modules:

- `dream_trainer.callbacks.loggers.base`
- `dream_trainer.callbacks.loggers.learning_rate`
- `dream_trainer.callbacks.loggers.media`
- `dream_trainer.callbacks.loggers.metric`
- `dream_trainer.callbacks.loggers.wandb_watch`
- `dream_trainer.callbacks.progress_bar`

Logging callbacks collect trainer outputs, optimizer state, metrics, media, and progress information.

## Public Classes

| Class | Purpose |
| --- | --- |
| `LoggerCallback` | Logs scalar outputs returned by train and validation steps. |
| `LRLoggerCallback` | Logs optimizer learning rates. |
| `MediaLoggerCallback` | Logs image or video outputs. |
| `MetricLoggerCallback` | Logs computed metric values when available. |
| `ModelWatchCallback` | Adds WandB model watching when WandB is configured. |
| `ProgressBar` | Displays local progress and a selected metric. |

## Guidance

Return stable scalar keys such as `train/loss`, `train/grad_norm`, and `val/loss`. Keep media logging sparse, especially in distributed runs.

## See It In Use

- [Logging And Metrics](../../logging-metrics.md) — how step returns flow into logger callbacks.
- [Callbacks — Pick A Stack](../../callbacks.md#pick-a-stack) — minimum-viable, production, and debugging stacks.
- [Tutorial 4 — Why WandB, Why `ModelWatchCallback`](../../tutorials/production.md#why-wandb-why-modelwatchcallback) — production logging setup.
