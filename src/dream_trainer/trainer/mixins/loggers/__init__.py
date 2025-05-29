from dream_trainer.trainer.mixins.loggers.types import LoggerConfigMixin, LoggerMixin

try:
    from .wandb import WandBLoggerConfigMixin, WandBLoggerMixin

except ImportError:
    pass

__all__ = [
    "LoggerMixin",
    "LoggerConfigMixin",
    "WandBLoggerMixin",
    "WandBLoggerConfigMixin",
]
