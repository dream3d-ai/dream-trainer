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
    """
    Abstract configuration class for trainers.

    This dataclass defines the base configuration parameters that all trainer
    implementations must provide. It serves as the foundation for specific
    trainer configurations.

    Attributes:
        seed: Random seed for reproducibility. If None, a random seed between
            0 and 1000 will be generated.
        project: Name of the project for experiment tracking and organization.
        group: Group name for categorizing related experiments.
        experiment: Unique name for this specific experiment run.
        device_parameters: Configuration for device and distributed training setup.
    """

    seed: int | None = 42

    project: str
    group: str
    experiment: str

    device_parameters: DeviceParameters


class AbstractTrainer(ABC):
    """
    Abstract base class for all trainer implementations.

    This class defines the interface and common functionality that all trainers
    must implement. It encapsulates the core components needed for training
    including distributed world management, state tracking, and abstract methods
    for model, optimizer, and dataloader access.

    The trainer maintains global training state and provides utility methods
    for managing models, optimizers, and schedulers by name.

    Attributes:
        config: Configuration object containing training parameters.
        world: DistributedWorld object managing distributed and parallel training.
        global_step: Number of optimizer steps taken across all epochs.
        local_batches: Number of batches processed since program start.
        current_epoch: Current epoch number (0-indexed).
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
    def named_models(self) -> dict[str, "nn.Module"]:
        """
        Return a dictionary mapping model names to their corresponding modules.

        This method should return all models used in training, organized by
        unique string identifiers. These names are used throughout the training
        process for logging, checkpointing, and model-specific operations.

        Returns:
            dict[str, nn.Module]: Dictionary mapping model names to PyTorch modules.
                For example: {"encoder": encoder_model, "decoder": decoder_model}
        """
        ...

    @abstractmethod
    def named_optimizers(self) -> dict[str, "Optimizer"]:
        """
        Return a dictionary mapping optimizer names to their corresponding optimizers.

        This method should return all optimizers used in training, organized by
        unique string identifiers. Each optimizer should correspond to one or more
        models returned by named_models().

        Returns:
            dict[str, Optimizer]: Dictionary mapping optimizer names to PyTorch optimizers.
                For example: {"adam": adam_optimizer, "sgd": sgd_optimizer}
        """
        ...

    @abstractmethod
    def named_schedulers(self) -> dict[str, "LRScheduler"] | None:
        """
        Return a dictionary mapping scheduler names to their corresponding schedulers.

        This method should return all learning rate schedulers used in training,
        organized by unique string identifiers. Each scheduler should be associated
        with an optimizer from named_optimizers(). Return None if no schedulers
        are used.

        Returns:
            dict[str, LRScheduler] | None: Dictionary mapping scheduler names to
                PyTorch schedulers, or None if no schedulers are used.
                For example: {"cosine": cosine_scheduler, "linear": linear_scheduler}
        """
        ...

    @abstractmethod
    def get_module(self, fqn: str) -> "nn.Module":
        """
        Retrieve a module by its fully qualified name (FQN).

        This method provides access to nested modules within the trainer's models
        using dot-separated paths. It's primarily used for accessing specific
        layers or components for callbacks, analysis, or targeted operations.

        Args:
            fqn: Fully qualified name of the module, using dot notation.
                For example: "encoder.layer1.conv1" or "decoder.attention"

        Returns:
            nn.Module: The requested module.

        Raises:
            AttributeError: If the module with the given FQN doesn't exist.
        """
        ...

    @property
    @abstractmethod
    def train_dataloader(self) -> Iterable:
        """
        Return the training dataloader.

        This property should provide access to the dataloader used for training.
        The dataloader should yield batches of data that will be passed to
        the training_step method.

        Returns:
            Iterable: An iterable that yields training batches. Typically a
                PyTorch DataLoader, but can be any iterable that produces
                batches of training data.
        """
        ...

    @property
    @abstractmethod
    def val_dataloader(self) -> Iterable:
        """
        Return the validation dataloader.

        This property should provide access to the dataloader used for validation.
        The dataloader should yield batches of data that will be passed to
        the validation_step method.

        Returns:
            Iterable: An iterable that yields validation batches. Typically a
                PyTorch DataLoader, but can be any iterable that produces
                batches of validation data.
        """
        ...

    @abstractmethod
    def state_dict(self) -> dict[str, Any]:
        """
        Return the complete state dictionary of the trainer.

        This method should capture all necessary state for resuming training,
        including model parameters, optimizer states, scheduler states, trainer
        metadata, and any other stateful components.

        The returned dictionary should be serializable and contain all information
        needed to exactly resume training from the current point.

        Returns:
            dict[str, Any]: Complete state dictionary containing all trainer state.
                Typically includes keys like "models", "optimizers", "schedulers",
                "trainer", and "dataloaders".
        """
        ...

    @abstractmethod
    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """
        Load a complete state dictionary into the trainer.

        This method should restore all trainer state from a checkpoint, including
        model parameters, optimizer states, scheduler states, trainer metadata,
        and any other stateful components.

        Args:
            state_dict: Complete state dictionary to load, typically obtained
                from a previous call to state_dict(). Should contain all
                necessary components to resume training.
        """
        ...

    @abstractmethod
    def fit(self):
        """
        Execute the complete training pipeline.

        This is the main entry point for training. It should handle the entire
        training lifecycle including initialization, training loops, validation,
        checkpointing, and cleanup.

        Implementations should ensure proper setup and teardown of distributed
        training resources.
        """
        ...

    @abstractmethod
    def setup(self):
        """
        Set up the trainer components after configuration.

        This method is called after configure() and should perform any setup
        that requires the trainer to be fully configured. This includes
        initializing models, optimizers, dataloaders, and any other components
        that depend on the configuration.

        This is where heavy initialization should occur, such as loading
        pretrained weights, setting up distributed training, or preparing datasets.
        """
        ...

    @abstractmethod
    def configure(self):
        """
        Configure the trainer components.

        This method is called early in the training pipeline and should set up
        the basic configuration for training. This includes defining models,
        optimizers, schedulers, and other training components.

        This method should be lightweight and primarily focused on defining
        the training components rather than initializing them.
        """
        ...

    ###################
    # Utility Methods #
    ###################

    def get_name_by_model(self, model: "nn.Module") -> str:
        """
        Get the name associated with a given model instance.

        This utility method performs a reverse lookup to find the name key
        for a given model object in the named_models() dictionary.

        Args:
            model: The model instance to look up.

        Returns:
            str: The name associated with the model.

        Raises:
            ValueError: If the model is not found in named_models().
        """
        name = next((name for name, m in self.named_models().items() if m is model), None)
        if name is None:
            raise ValueError(f"Model {model} not found in {self.named_models()}")
        return name

    def get_name_by_optimizer(self, optimizer: "Optimizer") -> str:
        """
        Get the name associated with a given optimizer instance.

        This utility method performs a reverse lookup to find the name key
        for a given optimizer object in the named_optimizers() dictionary.

        Args:
            optimizer: The optimizer instance to look up.

        Returns:
            str: The name associated with the optimizer.

        Raises:
            ValueError: If the optimizer is not found in named_optimizers().
        """
        name = next(
            (name for name, o in self.named_optimizers().items() if o is optimizer),
            None,
        )
        if name is None:
            raise ValueError(f"Optimizer {optimizer} not found in {self.named_optimizers()}")
        return name

    def get_name_by_scheduler(self, scheduler: "LRScheduler") -> str:
        """
        Get the name associated with a given scheduler instance.

        This utility method performs a reverse lookup to find the name key
        for a given scheduler object in the named_schedulers() dictionary.

        Args:
            scheduler: The scheduler instance to look up.

        Returns:
            str: The name associated with the scheduler.

        Raises:
            ValueError: If the scheduler is not found in named_schedulers() or
                if named_schedulers() returns None.
        """
        name = next(
            (name for name, s in (self.named_schedulers() or {}).items() if s is scheduler),
            None,
        )
        if name is None:
            raise ValueError(f"Scheduler {scheduler} not found in {self.named_schedulers()}")
        return name

    def get_scheduler_from_optimizer(self, optimizer: "Optimizer") -> "LRScheduler | None":
        """
        Find the scheduler associated with a given optimizer.

        This utility method searches through all schedulers to find one that
        is configured to work with the specified optimizer.

        Args:
            optimizer: The optimizer to find the associated scheduler for.

        Returns:
            LRScheduler | None: The scheduler associated with the optimizer,
                or None if no scheduler is found or if named_schedulers() returns None.
        """
        for scheduler in (self.named_schedulers() or {}).values():
            if scheduler.optimizer is optimizer:
                return scheduler
        return None
