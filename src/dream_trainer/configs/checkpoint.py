from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

AsyncCheckpointMode = Literal["disabled", "async", "async_with_pinned_mem"]
_ASYNC_CHECKPOINT_MODES = ("disabled", "async", "async_with_pinned_mem")


@dataclass(kw_only=True)
class CheckpointParameters:
    enable: bool = True
    root_dir: str | Path
    storage_options: dict[str, Any] | None = None

    resume_mode: Literal["min", "max", "last"] = "last"
    monitor: str = "train/loss"

    checkpoint_every_n_train_epochs: int | None = None
    checkpoint_every_n_val_epochs: int | None = None

    keep_top_k: int = 5
    strict_load: bool = False

    model_weights_only: bool = True
    async_mode: AsyncCheckpointMode = "disabled"
    """
    "disabled": synchronous DCP save.
    "async": DCP async_save using its default thread-based path.
    "async_with_pinned_mem": DCP async_save using pinned-memory staging and process upload.
    """

    resume_data: bool = True
    """
    If True, resume dataloader/dataset state from checkpoint.
    If False, skip loading dataloader state, allowing dataset changes between runs.
    """

    def __post_init__(self):
        if self.async_mode not in _ASYNC_CHECKPOINT_MODES:
            raise ValueError(
                f"Invalid async_mode {self.async_mode!r}. "
                f"Expected one of {_ASYNC_CHECKPOINT_MODES}."
            )

        if self.keep_top_k < 0:
            raise ValueError("keep_top_k must be non-negative")

        if self.keep_top_k == 1:
            raise ValueError(
                "We need to maintain at least 2 checkpoint replicas, "
                "as the last one may be in the process of being saved. "
                "Please set keep_top_k to 0 to keep all checkpoints, "
                "or to a value greater than 1."
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
