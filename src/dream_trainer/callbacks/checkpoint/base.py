import gc
import os
import shutil
from functools import cached_property
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch import Tensor
from torch.distributed import ReduceOp
from typing_extensions import override

from dream_trainer.configs import CheckpointParameters
from dream_trainer.trainer import DreamTrainer
from dream_trainer.utils import logger

from ..callback import Callback
from .types import Checkpoint
from .utils import find_checkpoints, find_current_checkpoint


class CheckpointCallback(Callback[DreamTrainer]):
    """
    Base checkpoint callback that implements the entire checkpointing
    workflow except for storage-specific details. Subclasses must supply
    a storage-specific _setup_paths() implementation and may override
    the file-open keyword arguments.
    """

    config: CheckpointParameters

    # State
    _current_metric: Tensor | None
    _did_resume: bool

    def __init__(self, config: CheckpointParameters):
        self.config = config

    ##############
    # Properties #
    ##############

    @cached_property
    def root_dir(self) -> Path:
        return (
            Path(self.config.root_dir)
            / self.trainer.project
            / self.trainer.group
            / self.trainer.experiment
            / "checkpoints"
        )

    @property
    def should_checkpoint(self) -> bool:
        if self.config.checkpoint_every_n_epochs is not None:
            return (
                self.trainer.current_epoch > 0
                and self.trainer.current_epoch % self.config.checkpoint_every_n_epochs == 0
            )

        if self.config.checkpoint_every_n_steps is not None:
            return (
                self.trainer.global_step > 0
                and self.trainer.global_step % self.config.checkpoint_every_n_steps == 0
            )

        raise ValueError(
            "If checkpointing is enabled, one of checkpoint_every_n_epochs or checkpoint_every_n_steps must be set"
        )

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

        if metric.numel() != 1:
            raise ValueError(f"Metric must be a scalar tensor, got {metric.shape}")

        self._current_metric = metric

    # ####################################
    # Common Checkpoint Loading & Saving #
    # ####################################

    def _load(self, checkpoint: Checkpoint, state_dict: dict[str, Any]):
        logger.info(f"Loading checkpoint {checkpoint.checkpoint_id}")
        dcp.state_dict_loader.load(
            state_dict,
            checkpoint_id=str(self.root_dir / checkpoint.checkpoint_id),
            process_group=self.pg,
        )
        logger.info(f"Resumed {self.trainer.experiment} from step {checkpoint.step}")
        self.trainer.world.barrier()

    def _save(self, checkpoint: Checkpoint):
        logger.info(f"Saving checkpoint {checkpoint.checkpoint_id}")
        dcp.state_dict_saver.save(
            self.trainer.state_dict(),
            checkpoint_id=str(self.root_dir / checkpoint.checkpoint_id),
            process_group=self.pg,
        )
        logger.info(f"Saved checkpoint to {self.root_dir / checkpoint.checkpoint_id}")
        self._cleanup_checkpoints()
        self.trainer.world.barrier()

    @torch.no_grad()
    def load(self, checkpoint: Checkpoint):
        gc.collect(generation=1)
        self._load(checkpoint, self.trainer.state_dict())

        self._did_resume = True
        self._current_metric = None

    @torch.no_grad()
    def save(self):
        if self._did_resume:
            self._did_resume = False
            return  # Skip saving as we just loaded the checkpoint

        if self._current_metric is None:
            raise ValueError(f"{self.config.monitor} was not reported in the last epoch")

        self.trainer.world.all_reduce(self._current_metric, op=ReduceOp.AVG)
        checkpoint = Checkpoint(
            step=self.trainer.global_step, metric=self._current_metric.item()
        )

        gc.collect(generation=1)
        self._save(checkpoint)
        gc.collect(generation=1)

    # ################
    # Callback Hooks #
    # ################

    @override
    def pre_launch(self):
        # Remove the callback if checkpointing is disabled
        if not self.config.enable:
            self.trainer.callbacks.pop(self.__class__.__name__)

    @override
    def post_setup(self):
        # Setup paths
        os.makedirs(self.root_dir, exist_ok=True)

        # Setup process group for loading and saving
        self.pg = dist.new_group(backend="gloo")
        self._did_resume = False
        self._current_metric = None

    @override
    def pre_fit(self):
        # Load a checkpoint if it exists
        checkpoint = find_current_checkpoint(self.root_dir, self.config.resume_mode)
        if checkpoint is None:
            logger.info(f"Training {self.trainer.experiment} from scratch")
            return

        self.load(checkpoint)

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
    def pre_train_epoch(self):
        # Checkpointing runs at the start of each epoch
        if self.should_checkpoint:
            self.save()

    def _cleanup_checkpoints(self):
        checkpoints = find_checkpoints(self.root_dir, self.config.resume_mode)
        purge_checkpoints = checkpoints[self.config.keep_top_k :]

        if self.trainer.world.is_global_zero:
            for checkpoint in purge_checkpoints:
                print(f"Purging checkpoint {checkpoint.checkpoint_id}")
                # TODO: Work with cloud storage
                shutil.rmtree(self.root_dir / checkpoint.checkpoint_id, ignore_errors=True)
        self.trainer.world.barrier()
