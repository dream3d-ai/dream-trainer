import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterable

from dream_trainer.configs import DeviceParameters
from dream_trainer.trainer.world import DistributedWorld

if TYPE_CHECKING:
    import torch.nn as nn
    from torch.optim.lr_scheduler import LRScheduler
    from torch.optim.optimizer import Optimizer


@dataclass(kw_only=True)
class AbstractTrainerConfig:
    seed: int | None = 42

    project: str
    group: str
    experiment: str

    device_parameters: DeviceParameters


class AbstractTrainer(ABC):
    """
    An abstract trainer that encapsulates the base necessary components of a trainer. Contains a world
    "world" object containing components for distributed and parallel training.
    """

    config: AbstractTrainerConfig

    def __init__(self, config: AbstractTrainerConfig):
        self.config = config

        self.seed = config.seed or random.randint(0, 1000)

        self.project = config.project
        self.group = config.group
        self.experiment = config.experiment

        self.device_parameters = config.device_parameters

        self.world = DistributedWorld(config.device_parameters)

        # Trainer State:  NOTE: Keep track of these yourself
        self.global_step = 0  # Number of optimizer steps taken
        self.local_batches = 0  # Number of batches processed since program start
        self.current_epoch = 0

    @abstractmethod
    def named_models(self) -> dict[str, "nn.Module"]: ...

    @abstractmethod
    def named_optimizers(self) -> dict[str, "Optimizer"]: ...

    @abstractmethod
    def named_schedulers(self) -> dict[str, "LRScheduler"] | None: ...

    @abstractmethod
    def get_module(self, fqn: str) -> "nn.Module": ...

    @property
    @abstractmethod
    def train_dataloader(self) -> Iterable: ...

    @property
    @abstractmethod
    def val_dataloader(self) -> Iterable: ...

    @abstractmethod
    def state_dict(self) -> dict[str, Any]: ...

    @abstractmethod
    def load_state_dict(self, state_dict: dict[str, Any]) -> None: ...

    @abstractmethod
    def fit(self): ...

    @abstractmethod
    def setup(self): ...

    @abstractmethod
    def configure(self): ...

    ###################
    # Utility Methods #
    ###################

    def get_name_by_model(self, model: "nn.Module") -> str:
        name = next((name for name, m in self.named_models().items() if m is model), None)
        if name is None:
            raise ValueError(f"Model {model} not found in {self.named_models()}")
        return name

    def get_name_by_optimizer(self, optimizer: "Optimizer") -> str:
        name = next(
            (name for name, o in self.named_optimizers().items() if o is optimizer),
            None,
        )
        if name is None:
            raise ValueError(f"Optimizer {optimizer} not found in {self.named_optimizers()}")
        return name

    def get_name_by_scheduler(self, scheduler: "LRScheduler") -> str:
        name = next(
            (name for name, s in (self.named_schedulers() or {}).items() if s is scheduler),
            None,
        )
        if name is None:
            raise ValueError(f"Scheduler {scheduler} not found in {self.named_schedulers()}")
        return name

    def get_scheduler_from_optimizer(self, optimizer: "Optimizer") -> "LRScheduler | None":
        for scheduler in (self.named_schedulers() or {}).values():
            if scheduler.optimizer is optimizer:
                return scheduler
        return None
