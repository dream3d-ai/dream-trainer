from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


@dataclass(kw_only=True)
class CheckpointParameters:
    enable: bool = True
    root_dir: str | Path

    resume_mode: Literal["min", "max", "last"] = "last"
    monitor: str = "train/loss"

    checkpoint_every_n_epochs: int | None = None
    checkpoint_every_n_steps: int | None = None

    keep_top_k: int = 5
    strict_load: bool = False
    ignore_frozen_params: bool = False
    flatten_optimizer_state_dict: bool = True

    model_weights_only: bool = True
    exclude_from_loading: list[str] = field(default_factory=list)
    pin_memory: bool = False

    def __post_init__(self):
        if self.enable and self.keep_top_k <= 1:
            raise ValueError(
                "We need to maintain at least 2 checkpoint replicas, "
                "as the last one may be in the process of being saved."
                "Please set keep_top_k to a value greater than 1."
            )

        if (
            self.enable
            and self.checkpoint_every_n_epochs is None
            and self.checkpoint_every_n_steps is None
        ):
            raise ValueError(
                "checkpoint_every_n_epochs and checkpoint_every_n_steps cannot both be None"
            )

        if (
            self.checkpoint_every_n_epochs is not None
            and self.checkpoint_every_n_steps is not None
        ):
            raise ValueError(
                "checkpoint_every_n_epochs and checkpoint_every_n_steps cannot both be set"
            )
