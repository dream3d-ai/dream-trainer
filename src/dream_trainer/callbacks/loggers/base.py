import os
from typing import Any

import torch
from typing_extensions import override

from dream_trainer.trainer.mixins.loggers import LoggerMixin

from ..callback import RankZeroCallback


def filter_logs(result: dict[str, Any]) -> dict[str, Any]:
    """Filter out non-numeric values from the result dictionary."""
    filter_input = lambda value: isinstance(value, (int, float)) or (
        isinstance(value, torch.Tensor) and value.squeeze().ndim == 0
    )

    return {k: v for k, v in result.items() if filter_input(v)}


class LoggerCallback(RankZeroCallback[LoggerMixin]):
    """
    A base callback that logs all returned metrics from training and validation steps and
    epochs.

    This callback only logs scalar metrics.
    """

    def __init__(
        self,
        log_every_n_train_batches: int | None = 8,
        log_every_n_val_batches: int | None = None,
        code_dir: str = "./",
    ):
        """Initialize the logger callback.

        Args:
            log_every_n_train_batches: If set, only log training metrics every N batches. If None, log every batch.
            log_every_n_val_batches: If set, only log validation metrics every N batches. If None, log every batch.
            code_dir: Directory storing the code to be saved.
        """
        self.log_every_n_train_batches = log_every_n_train_batches
        self.log_every_n_val_batches = log_every_n_val_batches
        self.code_dir = code_dir

    @override
    def pre_configure(self):
        self.trainer.log_config(self.trainer.config)

    @override
    def pre_fit(self):
        self.trainer.log_code(
            self.code_dir, gitignore_path=os.path.join(self.code_dir, ".gitignore")
        )

    @override
    def post_train_step(self, result: dict[str, torch.Tensor | int | float], batch_idx: int):
        if (
            self.log_every_n_train_batches is None
            or batch_idx % self.log_every_n_train_batches == 0
        ):
            self.trainer.log_dict(filter_logs(result))

    @override
    def post_train_epoch(self, result: dict[str, torch.Tensor | int | float]):
        self.trainer.log_dict(filter_logs(result))

    @override
    def post_validation_step(self, result: dict[str, Any], batch_idx: int):
        if (
            self.log_every_n_val_batches is None
            or batch_idx % self.log_every_n_val_batches == 0
        ):
            self.trainer.log_dict(filter_logs(result))

    @override
    def post_validation_epoch(self, result: dict[str, Any]):
        self.trainer.log_dict(filter_logs(result))
