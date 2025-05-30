import shutil
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch import Tensor
from typing_extensions import override

from dream_trainer.configs import CheckpointParameters
from dream_trainer.utils import logger

from ..callback import Callback
from .types import Checkpoint
from .utils import find_checkpoints, find_current_checkpoint


class CheckpointCallback(Callback):
    """
    Base checkpoint callback that implements the entire checkpointing
    workflow except for storage-specific details. Subclasses must supply
    a storage-specific _setup_paths() implementation and may override
    the file-open keyword arguments.
    """

    config: CheckpointParameters

    # Paths
    _root_dir: Path
    _checkpoint_dir: Path

    # State
    _current_metric: Tensor | None
    _did_resume: bool

    def __init__(self, config: CheckpointParameters):
        self.config = config

    # ##################
    # Metric Reporting #
    # ##################

    def _report_metric(self, result: dict[str, Any]):
        metric = result.get(self.config.monitor)
        if metric is None:
            return

        if not isinstance(metric, (Tensor, float, int)):
            raise ValueError(
                f"Metric must be a scalar tensor, float, or int. Got {type(metric)}"
            )

        if not isinstance(metric, Tensor):
            metric = torch.tensor(metric)

        if metric.shape != torch.Size([]):
            raise ValueError("Metric must be a scalar tensor")

        self._current_metric = metric

    # ####################################
    # Common Checkpoint Loading & Saving #
    # ####################################

    def load(self):
        current_checkpoint = find_current_checkpoint(
            self._checkpoint_dir, self.config.resume_mode
        )
        if current_checkpoint is None:
            logger.info(f"Training {self.trainer.experiment} from scratch")
            return

        logger.info(
            f"Loading checkpoint {current_checkpoint.checkpoint_id} for {self.trainer.experiment}",
        )

        state_dict = self.trainer.state_dict()
        dcp.state_dict_loader.load(
            state_dict,
            checkpoint_id=str(self._checkpoint_dir / current_checkpoint.checkpoint_id),
            process_group=self.pg,
        )

        self._did_resume = True
        self._current_metric = None

        self.trainer.world.barrier()
        logger.info(
            f"Resumed {self.trainer.experiment} from step {current_checkpoint.step}",
        )

    def _save(self, checkpoint: Checkpoint):
        state_dict = self.trainer.state_dict()
        dcp.state_dict_saver.save(
            state_dict,
            checkpoint_id=str(self._checkpoint_dir / checkpoint.checkpoint_id),
            process_group=self.pg,
        )
        self._cleanup_checkpoints()
        self.trainer.world.barrier()

    def save(self):
        if self._current_metric is None:
            if self._did_resume:
                # Skip saving as we just loaded the checkpoint
                self._did_resume = False
                return
            else:
                raise ValueError(
                    f"Monitoring {self.config.monitor} but it was not reported in the last epoch"
                )

        current_metric = self.trainer.world.all_reduce(self._current_metric, op="mean")
        assert isinstance(current_metric, Tensor) and current_metric.numel() == 1, (
            f"Monitored checkpoint metric must be a scalar tensor, got {current_metric}"
        )
        checkpoint = Checkpoint(step=self.trainer.global_step, metric=current_metric.item())

        self._save(checkpoint)
        logger.info(f"Saved checkpoint to {checkpoint.checkpoint_id}")
        self._cleanup_checkpoints()

    # ################
    # Callback Hooks #
    # ################

    @override
    def post_setup(self):
        # Setup paths
        self._root_dir = (
            Path(self.config.root_dir)
            / self.trainer.project
            / self.trainer.group
            / self.trainer.experiment
            / "checkpoints"
        )

        # Setup process group
        self.pg = dist.new_group(backend="gloo")
        self._did_resume = False
        self._current_metric = None

    # Load & save for fit

    @override
    def pre_fit(self):
        self.load()

    @override
    def post_fit(self):
        if self._current_metric is None:
            return
        self.save()

    # Report metrics

    @override
    def post_train_step(self, result: dict[str, Any], _):
        self._report_metric(result)

    @override
    def post_validation_epoch(self, result: dict[str, Any]):
        self._report_metric(result)
        if (
            self.config.checkpoint_every_n_epochs is not None
            and self.trainer.current_epoch % self.config.checkpoint_every_n_epochs == 0
        ) or (
            self.config.checkpoint_every_n_steps is not None
            and self.trainer.global_step % self.config.checkpoint_every_n_steps == 0
        ):
            self.save()

    @override
    def pre_train_epoch(self):
        if (
            self.config.checkpoint_every_n_epochs is not None
            and self.trainer.current_epoch % self.config.checkpoint_every_n_epochs == 0
        ):
            self.save()

    def _cleanup_checkpoints(self):
        if self.config.keep_top_k <= 0:
            return

        checkpoints = find_checkpoints(self._checkpoint_dir, self.config.resume_mode)
        purge_checkpoints = checkpoints[-self.config.keep_top_k :]

        if self.trainer.world.is_global_zero:
            for checkpoint in purge_checkpoints:
                # TODO: Work with cloud storage
                shutil.rmtree(
                    self._checkpoint_dir / checkpoint.checkpoint_id, ignore_errors=True
                )
        self.trainer.world.barrier()
