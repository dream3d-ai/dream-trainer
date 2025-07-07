try:
    from .trainer import DreamTrainer, DreamTrainerConfig

    __all__ = ["DreamTrainer", "DreamTrainerConfig"]
except ImportError:
    # If DreamTrainer import fails due to optional dependencies,
    # still allow other modules to be imported
    __all__ = []
