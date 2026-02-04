from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass(kw_only=True)
class CheckpointParameters:
    enable: bool = True
    root_dir: str | Path

    resume_mode: Literal["min", "max", "last"] = "last"
    monitor: str = "train/loss"

    checkpoint_every_n_train_epochs: int | None = None
    checkpoint_every_n_val_epochs: int | None = None

    keep_top_k: int = 5
    strict_load: bool = False

    model_weights_only: bool = True
    pin_memory: bool = False

    resume_data: bool = True
    """
    If True, resume dataloader/dataset state from checkpoint.
    If False, skip loading dataloader state, allowing dataset changes between runs.
    """

    def __post_init__(self):
        if self.enable and (self.keep_top_k <= 1 and self.keep_top_k >= 0):
            raise ValueError(
                "We need to maintain at least 2 checkpoint replicas, "
                "as the last one may be in the process of being saved."
                "Please set keep_top_k to a value greater than 1."
            )

        if (
            self.enable
            and self.checkpoint_every_n_train_epochs is None
            and self.checkpoint_every_n_val_epochs is None
        ):
            raise ValueError(
                "checkpoint_every_n_train_epochs and checkpoint_every_n_val_epochs cannot both be None"
            )

        if (
            self.checkpoint_every_n_train_epochs is not None
            and self.checkpoint_every_n_val_epochs is not None
        ):
            raise ValueError(
                "checkpoint_every_n_train_epochs and checkpoint_every_n_val_epochs cannot both be set"
            )
