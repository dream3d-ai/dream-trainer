from pathlib import Path
from typing import Literal, override

import torch.distributed as dist
import torch.distributed.checkpoint as dcp

from dream_trainer import DreamTrainer
from dream_trainer.callbacks import Callback
from dream_trainer.utils import logger

from .utils import find_current_checkpoint


class LoadPartialCheckpointCallback(Callback[DreamTrainer]):
    """
    Partially load a checkpoint.

    This callback will only load model weights. Other configurations like optimizers,
    schedulers, experiment id, etc... will be initialized from the existing config. This is
    useful for e.g. fine-tuning a model from a checkpoint.
    """

    def __init__(self, path: str | Path, resume_mode: Literal["min", "max", "last"] = "last"):
        self.path = Path(path)
        self.resume_mode: Literal["min", "max", "last"] = resume_mode

    @override
    def post_setup(self):
        self.pg = dist.new_group(backend="gloo")

    @override
    def pre_fit(self):
        checkpoint = find_current_checkpoint(self.path, self.resume_mode)
        if checkpoint is None:
            available = [p.name for p in self.path.iterdir()] if self.path.exists() else []
            raise ValueError(
                f"No checkpoint found at {self.path}. Available checkpoints: {', '.join(available)}"
            )

        logger.info(f"Loading weights from {checkpoint.checkpoint_id}")

        # Only load necessary states
        state_dict = self.trainer.state_dict(flatten_optimizer_state_dict=True)

        state_dict.pop("trainer")
        state_dict.pop("optimizers")
        state_dict.pop("schedulers")
        state_dict.pop("dataloaders")

        assert len(state_dict) == 1 and "models" in state_dict, (
            "State dict must contain only models. Got: " + str(state_dict.keys())
        )

        # Load weights
        dcp.state_dict_loader.load(
            state_dict,
            checkpoint_id=str(self.path / checkpoint.checkpoint_id),
            process_group=self.pg,
            planner=dcp.DefaultLoadPlanner(allow_partial_load=True),
        )
        self.trainer.load_state_dict(state_dict, strict=False)

        self.trainer.world.barrier()
        del self.pg

        logger.info(f"Loaded weights from {checkpoint.checkpoint_id}")
        self.trainer.callbacks.pop(self.__class__.__name__)  # Remove callback
