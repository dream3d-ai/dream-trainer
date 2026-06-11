from pathlib import Path
from typing import Any, Literal, cast, override

import torch.distributed as dist
import torch.distributed.checkpoint as dcp

from dream_trainer import DreamTrainer
from dream_trainer.callbacks import Callback
from dream_trainer.utils import logger

from .s3 import S3CheckpointStorage, is_s3_uri
from .utils import find_current_checkpoint


class LoadPartialCheckpointCallback(Callback[DreamTrainer]):
    """
    Partially load a checkpoint.

    This callback will only load model weights. Other configurations like optimizers,
    schedulers, experiment id, etc... will be initialized from the existing config. This is
    useful for e.g. fine-tuning a model from a checkpoint.
    """

    def __init__(
        self,
        path: str | Path,
        resume_mode: Literal["min", "max", "last"] | int = "last",
        *,
        storage_options: dict[str, Any] | None = None,
    ):
        self.path: str | Path = str(path).rstrip("/") if is_s3_uri(path) else Path(path)
        self.resume_mode: Literal["min", "max", "last"] | int = resume_mode
        self.storage_options = dict(storage_options or {}) if storage_options else None

    @override
    def post_setup(self):
        self.pg = cast(dist.ProcessGroup, dist.new_group(backend="gloo"))

    @override
    def pre_fit(self):
        storage = None
        available: list[str]
        if is_s3_uri(self.path):
            assert isinstance(self.path, str)
            storage = S3CheckpointStorage(self.path, self.storage_options)
            mode = "last" if isinstance(self.resume_mode, int) else self.resume_mode
            checkpoints = storage.find_checkpoints(mode)
            available = [checkpoint.checkpoint_id for checkpoint in checkpoints]
            if isinstance(self.resume_mode, int):
                checkpoints = [
                    checkpoint
                    for checkpoint in checkpoints
                    if checkpoint.step == self.resume_mode
                ]
                checkpoint = checkpoints[0] if len(checkpoints) == 1 else None
            else:
                checkpoint = checkpoints[0] if checkpoints else None
        else:
            assert isinstance(self.path, Path)
            checkpoint = find_current_checkpoint(self.path, self.resume_mode)
            available = [p.name for p in self.path.iterdir()] if self.path.exists() else []

        if checkpoint is None:
            raise ValueError(
                f"No checkpoint found at {self.path}. Available checkpoints: {', '.join(available)}"
            )

        logger.info(f"Loading weights from {checkpoint.checkpoint_id}")

        # Only partial-load model weights. Trainer subclasses may add arbitrary
        # top-level state for full resume, which should not affect weight init.
        state_dict = {"models": self.trainer.model_state_dict()}

        # Load weights
        load_kwargs: dict[str, Any] = {
            "checkpoint_id": (
                storage.checkpoint_path(checkpoint.checkpoint_id)
                if storage is not None
                else str(self.path / checkpoint.checkpoint_id)
            ),
            "process_group": self.pg,
            "planner": dcp.default_planner.DefaultLoadPlanner(allow_partial_load=True),
        }
        if storage is not None:
            load_kwargs["storage_reader"] = storage.reader(checkpoint.checkpoint_id)

        dcp.state_dict_loader.load(
            state_dict,
            **load_kwargs,
        )
        self.trainer.load_state_dict(state_dict, strict=False, resume_data=False)

        self.trainer.world.barrier()
        del self.pg

        logger.info(f"Loaded weights from {checkpoint.checkpoint_id}")
        self.trainer.callbacks.pop(self.__class__.__name__)  # Remove callback
