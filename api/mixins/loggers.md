# Logger Mixins API

Modules:

- `dream_trainer.trainer.mixins.loggers`
- `dream_trainer.trainer.mixins.loggers.wandb`
- `dream_trainer.trainer.mixins.loggers.types`

Logger mixins provide trainer-side logging methods used by logging callbacks.

## Public Classes

| Class | Purpose |
| --- | --- |
| `LoggerConfigMixin` | Base config contract for logger mixins. |
| `LoggerMixin` | Trainer-side scalar and media logging interface. |
| `WandBLoggerConfigMixin` | Config support for WandB-backed logging. |
| `WandBLoggerMixin` | WandB-backed logger implementation. |

## Related Callbacks

- `LoggerCallback`
- `LRLoggerCallback`
- `MediaLoggerCallback`
- `MetricLoggerCallback`
- `ModelWatchCallback`

Callbacks collect training outputs and call logger mixin methods. Trainers should return stable scalar keys from `training_step` and `validation_step`.

## See It In Use

- [Logging And Metrics](../../logging-metrics.md) — step-return contract and how callbacks consume it.
- [Callbacks — Logging Callback API](../callbacks/logging.md) — the callbacks that drive these mixins.
