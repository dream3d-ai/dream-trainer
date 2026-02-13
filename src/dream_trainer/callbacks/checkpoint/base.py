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
    _last_saved_step: int | None

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
        if self.config.checkpoint_every_n_train_epochs is not None:
            return (
                self.trainer.current_epoch > 0
                and (self.trainer.current_epoch) % self.config.checkpoint_every_n_train_epochs
                == 0
            )

        if self.config.checkpoint_every_n_val_epochs is not None:
            # When val_every_n_steps is None, validation happens once per training epoch
            if self.trainer.training_parameters.val_every_n_steps is None:
                val_epoch = (
                    self.trainer.current_epoch + 1
                )  # +1 because we're in post_validation_epoch
            else:
                val_epoch = (
                    self.trainer.global_step
                    // self.trainer.training_parameters.val_every_n_steps
                )
            return val_epoch > 0 and val_epoch % self.config.checkpoint_every_n_val_epochs == 0

        raise ValueError(
            "One of checkpoint_every_n_train_epochs or checkpoint_every_n_val_epochs must be set"
        )

    # ##################
    # Metric Reporting #
    # ##################

    @torch.no_grad()
    def _report_metric(self, result: dict[str, Any]):
        metric = result.get(self.config.monitor)
        if metric is None:
            return

        if not isinstance(metric, (Tensor, float, int)):
            raise ValueError(
                f"Metric must be a scalar tensor, float, or int. Got {type(metric)}"
            )

        if not isinstance(metric, Tensor):
            metric = torch.tensor(metric, dtype=torch.float32)

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
            planner=dcp.default_planner.DefaultLoadPlanner(
                allow_partial_load=not self.config.strict_load
            ),
        )
        logger.info(f"Resumed {self.trainer.experiment} from step {checkpoint.step}")
        self.trainer.world.barrier()

    def _checkpoint_exists(self, checkpoint: Checkpoint):
        save_path = self.root_dir / checkpoint.checkpoint_id
        path_exists = torch.tensor([save_path.exists()])
        dist.broadcast(path_exists, src=0, group=self.pg)
        if path_exists:
            if self.config.resume_mode == "last":
                raise ValueError(
                    f"Checkpoint {checkpoint.checkpoint_id} already exists, but resume_mode is 'last'. This should never happen."
                )
            else:
                logger.warning(
                    f"Checkpoint {checkpoint.checkpoint_id} already exists, skipping save."
                )

        return path_exists

    def _save(self, checkpoint: Checkpoint, state_dict: dict[str, Any]):
        save_path = self.root_dir / checkpoint.checkpoint_id
        if self._checkpoint_exists(checkpoint):
            return

        logger.info(f"Saving checkpoint {checkpoint.checkpoint_id}")
        dcp.state_dict_saver.save(
            state_dict,
            checkpoint_id=str(save_path),
            process_group=self.pg,
        )
        logger.info(f"Saved checkpoint to {save_path}")
        self._cleanup_checkpoints()
        self.trainer.world.barrier()

    @torch.no_grad()
    def load(self, checkpoint: Checkpoint):
        gc.collect(generation=1)

        state_dict = self.trainer.state_dict()
        self._load(checkpoint, state_dict)
        self.trainer.load_state_dict(
            state_dict,
            strict=self.config.strict_load,
            resume_data=self.config.resume_data,
        )

        # Prevent re-saving the loaded checkpoint
        self._last_saved_step = self.trainer.global_step
        self._current_metric = None

    @torch.no_grad()
    def save(self):
        # Skip saving if we're at the exact same step we resumed from
        # (avoids re-saving a checkpoint we just loaded)
        if self._last_saved_step == self.trainer.global_step:
            return

        if self._current_metric is None:
            raise ValueError(f"{self.config.monitor} was not reported in the last epoch")

        self.trainer.world.all_reduce(self._current_metric, op=ReduceOp.AVG)
        checkpoint = Checkpoint(
            step=self.trainer.global_step, metric=self._current_metric.item()
        )

        gc.collect(generation=1)
        self._save(checkpoint, self.trainer.state_dict())
        self._last_saved_step = self.trainer.global_step
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
        self._last_saved_step = None

    @override
    def pre_fit(self):
        # Load a checkpoint if it exists
        logger.info(f"Checkpoint directory: {self.root_dir}")
        checkpoint = find_current_checkpoint(self.root_dir, self.config.resume_mode)
        if checkpoint is None:
            logger.info(f"Training {self.trainer.experiment} from scratch")
            return

        self.load(checkpoint)

    @override
    def post_fit(self):
        # Skip if no metric was reported (nothing to checkpoint)
        if self._current_metric is None:
            return

        # Skip if we already checkpointed at this step (e.g., from post_validation_epoch)
        # This prevents duplicate checkpoint errors, especially with resume_mode="last"
        if self._last_saved_step == self.trainer.global_step:
            return

        self.save()

    # Report metrics

    @override
    def post_train_step(self, result: dict[str, Any], _):
        self._report_metric(result)

    @override
    def post_validation_epoch(self, result: dict[str, Any]):
        self._report_metric(result)

        # Skip checkpointing during sanity validation
        if self.trainer.is_sanity_validation:
            return

        if self.config.checkpoint_every_n_val_epochs is not None and self.should_checkpoint:
            self.save()

    @override
    def pre_train_epoch(self):
        if self.config.checkpoint_every_n_train_epochs is not None and self.should_checkpoint:
            self.save()

    def _cleanup_checkpoints(self):
        checkpoints = find_checkpoints(self.root_dir, self.config.resume_mode)
        purge_checkpoints = (
            checkpoints[self.config.keep_top_k :] if self.config.keep_top_k > 0 else []
        )

        if self.trainer.world.is_global_zero:
            for checkpoint in purge_checkpoints:
                print(f"Purging checkpoint {checkpoint.checkpoint_id}")
                # TODO: Work with cloud storage
                shutil.rmtree(self.root_dir / checkpoint.checkpoint_id, ignore_errors=True)
        self.trainer.world.barrier()
