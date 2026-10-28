import os
import warnings
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor
from typing_extensions import override

from dream_trainer.trainer.mixins.loggers import LoggerMixin

from ..callback import Callback


def filter_logs(result: dict[str, Any]) -> dict[str, Any]:
    """Filter out non-numeric values from the result dictionary."""
    filter_input = lambda value: (
        isinstance(value, (int, float))
        or (isinstance(value, torch.Tensor) and value.squeeze().ndim == 0)
    )

    return {k: v for k, v in result.items() if filter_input(v)}


class LoggerCallback(Callback[LoggerMixin]):
    """
    A base callback that logs all returned metrics from training and validation steps and
    epochs.

    Training scalars are averaged over microbatches and data-parallel ranks.
    Only global rank zero writes logs.
    """

    def __init__(
        self,
        log_every_n_train_steps: int | None = 1,
        log_every_n_val_batches: int | None = None,
        code_dir: str | None = None,
        *,
        log_every_n_train_batches: int | None = None,
    ):
        """Initialize the logger callback.

        Args:
            log_every_n_train_steps: Log averaged training metrics every N completed
                gradient updates. None is equivalent to 1.
            log_every_n_val_batches: If set, only log validation metrics every N batches. If None, log every batch.
            code_dir: Directory storing the code to be saved.
            log_every_n_train_batches: Deprecated alias for log_every_n_train_steps.
                Its interval now counts gradient updates, not microbatches.
        """
        if log_every_n_train_batches is not None:
            warnings.warn(
                "log_every_n_train_batches is deprecated; use log_every_n_train_steps. "
                "The interval now counts completed gradient updates.",
                FutureWarning,
                stacklevel=2,
            )
            log_every_n_train_steps = log_every_n_train_batches
        if log_every_n_train_steps is not None and log_every_n_train_steps < 1:
            raise ValueError("log_every_n_train_steps must be positive")
        self.log_every_n_train_steps = log_every_n_train_steps or 1
        self.log_every_n_val_batches = log_every_n_val_batches
        self.code_dir = code_dir

        self._train_sums: dict[str, torch.Tensor] = {}
        self._train_counts: dict[str, int] = {}
        self._at_update_boundary = False
        self._accumulated_val_logs: dict[str, Any] = {}

    @override
    def pre_configure(self):
        if self.trainer.world.is_global_zero:
            self.trainer.log_config(self.trainer.config)

    @override
    def pre_fit(self):
        if self.trainer.world.is_global_zero and self.code_dir is not None:
            self.trainer.log_code(
                self.code_dir, gitignore_path=os.path.join(self.code_dir, ".gitignore")
            )

    @override
    def post_train_step(self, result: dict[str, torch.Tensor | int | float], batch_idx: int):
        for key, value in filter_logs(result).items():
            if isinstance(value, torch.Tensor):
                value = value.detach()
                if isinstance(value, DTensor):
                    value = value.full_tensor()
            value = torch.as_tensor(
                value, device=self.trainer.world.device, dtype=torch.float32
            ).reshape(())
            if key in self._train_sums:
                self._train_sums[key].add_(value)
            else:
                # Own the storage: callers may reuse a metric tensor next batch.
                self._train_sums[key] = value.clone()
            self._train_counts[key] = self._train_counts.get(key, 0) + 1

        self._at_update_boundary = not self.trainer.is_accumulating_gradients
        # post_train_step runs after optimizer.step(), before global_step increments.
        completed_step = self.trainer.global_step + 1
        if self._at_update_boundary and completed_step % self.log_every_n_train_steps == 0:
            self._log_train_metrics(completed_step)

    def _log_train_metrics(self, completed_step: int):
        world = self.trainer.world
        dp_mesh = world.get_mesh("dp")
        group = dp_mesh.get_group() if dp_mesh is not None and dp_mesh.size() > 1 else None
        keys = sorted(self._train_sums)
        if group is not None:
            # Sparse metrics (e.g. grad norms or modality-specific losses) need their
            # own counts, and can be absent on some ranks.
            rank_keys: list[list[str] | None] = [None] * dist.get_world_size(group)
            dist.all_gather_object(rank_keys, keys, group=group)
            keys = sorted({key for names in rank_keys if names is not None for key in names})

        if keys:
            sums = torch.stack(
                [
                    self._train_sums[key]
                    if key in self._train_sums
                    else torch.zeros((), device=world.device)
                    for key in keys
                ]
            )
            counts = torch.tensor(
                [self._train_counts.get(key, 0) for key in keys],
                device=world.device,
                dtype=torch.float32,
            )
            totals = torch.stack((sums, counts))
            if group is not None:
                dist.all_reduce(totals, group=group)
            if world.is_global_zero:
                means = (totals[0] / totals[1]).cpu().tolist()
                self.trainer.log_dict(
                    {**dict(zip(keys, means)), "trainer/global_step": completed_step}
                )
        self._train_sums.clear()
        self._train_counts.clear()

    @override
    def pre_epoch(self):
        if self.trainer.world.is_global_zero:
            self.trainer.log_dict({"trainer/epoch": self.trainer.current_epoch})

    @override
    def post_train_epoch(self, result: dict[str, torch.Tensor | int | float]):
        # Flush a shorter logging interval only if it ends on a completed update.
        # Never re-log the last microbatch as an epoch metric.
        if self._at_update_boundary:
            self._log_train_metrics(self.trainer.global_step)

    @override
    def post_validation_step(self, result: dict[str, Any], batch_idx: int):
        if not self.trainer.world.is_global_zero:
            return
        self._accumulated_val_logs.update(filter_logs(result))

        if (
            self.log_every_n_val_batches is None
            or batch_idx % self.log_every_n_val_batches == 0
        ):
            self.trainer.log_dict(self._accumulated_val_logs)
            self._accumulated_val_logs.clear()

    @override
    def post_validation_epoch(self, result: dict[str, Any]):
        if self.trainer.world.is_global_zero:
            self.trainer.log_dict(filter_logs(result))

    @override
    def post_fit(self):
        if self.trainer.world.is_global_zero:
            self.trainer.finish()

    @override
    def on_interrupt(self):
        if self.trainer.world.is_global_zero:
            return self.trainer.finish(exit_code=1)
