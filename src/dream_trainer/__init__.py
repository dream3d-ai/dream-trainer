from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .trainer import DreamTrainer, DreamTrainerConfig


__all__ = ["DreamTrainer", "DreamTrainerConfig"]


def __getattr__(name: str):
    if name in __all__:
        from .trainer import DreamTrainer, DreamTrainerConfig

        return {
            "DreamTrainer": DreamTrainer,
            "DreamTrainerConfig": DreamTrainerConfig,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
