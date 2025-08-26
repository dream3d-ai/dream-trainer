from abc import abstractmethod
from dataclasses import dataclass
from typing import Iterable

import torch.nn as nn
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer
from typing_extensions import override

from dream_trainer.trainer.abstract import AbstractTrainer, AbstractTrainerConfig
from dream_trainer.utils import logger
from dream_trainer.utils.common import configuration_ctx


@dataclass(kw_only=True)
class OptimizerAndSchedulerSetupConfigMixin(AbstractTrainerConfig):
    """Configuration mixin for optimizer and scheduler setup.

    This class serves as a base configuration for trainers that need optimizer
    and learning rate scheduler setup capabilities. It inherits from
    AbstractTrainerConfig and can be extended with optimizer-specific parameters
    like learning rates, weight decay, scheduler configurations, etc.

    Example:
        @dataclass
        class MyTrainerConfig(OptimizerAndSchedulerSetupConfigMixin):
            learning_rate: float = 1e-4
            weight_decay: float = 0.1
            warmup_steps: int = 1000
    """

    ...


class OptimizerAndSchedulerSetupMixin(AbstractTrainer):
    """Mixin that handles optimizer and learning rate scheduler configuration and setup.

    This mixin provides a framework for configuring and managing optimizers and
    schedulers in a trainer. It automatically tracks optimizer and scheduler instances,
    manages their relationships, and provides convenient access methods.

    The mixin enforces a two-phase setup:
    1. Configuration: Define optimizers and schedulers
    2. Setup: Initialize and link optimizers with their schedulers

    Attributes:
        config (OptimizerAndSchedulerSetupConfigMixin): Configuration for optimizers/schedulers
        _optimizer_names (list[str]): Names of optimizer attributes
        _scheduler_names (list[str]): Names of scheduler attributes
        _optimizer_scheduler_map (dict[str, str | None]): Maps optimizer names to their
            controlling scheduler names
    """

    config: OptimizerAndSchedulerSetupConfigMixin

    ###########################
    # AbstractTrainer Methods #
    ###########################

    @override
    def named_optimizers(self) -> dict[str, Optimizer]:
        """Return a dictionary mapping optimizer names to their instances.

        This method provides access to all optimizers registered during the
        configure_optimizers phase. Optimizer names are collected automatically
        when optimizers are assigned as attributes during configuration.

        Returns:
            dict[str, Optimizer]: Dictionary where keys are optimizer attribute names
                and values are the corresponding Optimizer instances.

        Example:
            >>> trainer.named_optimizers()
            {'optimizer': Adam(...), 'discriminator_opt': SGD(...)}
        """
        return {name: getattr(self, name) for name in self._optimizer_names}

    @override
    def named_schedulers(self) -> dict[str, LRScheduler]:
        """Return a dictionary mapping scheduler names to their instances.

        This method provides access to all schedulers registered during the
        configure_schedulers phase. Scheduler names are collected automatically
        when schedulers are assigned as attributes during configuration.

        Returns:
            dict[str, LRScheduler]: Dictionary where keys are scheduler attribute names
                and values are the corresponding LRScheduler instances.

        Example:
            >>> trainer.named_schedulers()
            {'scheduler': CosineAnnealingLR(...), 'warmup': LinearWarmup(...)}
        """
        return {name: getattr(self, name) for name in self._scheduler_names}

    @override
    def get_model_by_optimizer(self, name: str) -> nn.Module:
        """Return the model associated with a given optimizer name."""
        return self.named_models()[self._optimizer_model_map[name]]

    @override
    def get_optimizers_by_model(self, name: str) -> list[Optimizer]:
        """Return the optimizers associated with a given model name."""

        if name not in self.named_models().keys():
            raise ValueError(f"Model {name} not found")

        optimizer_names = []
        for optimizer_name, model_name in self._optimizer_model_map.items():
            if model_name == name:
                optimizer_names.append(optimizer_name)

        if not optimizer_names:
            raise ValueError(f"No optimizer found for model {name}")

        optimizers = []
        for name, optim in self.named_optimizers():
            if name in optimizer_names:
                optimizers.append(optim)

        return optimizers

    ########################
    # User-Defined Methods #
    ########################

    @abstractmethod
    def configure_optimizers(self) -> dict[nn.Module, Optimizer | Iterable[Optimizer]]:
        """Configure and instantiate all optimizers used by the trainer.

        This method must be implemented by subclasses to define and instantiate
        all optimizers. Optimizers should be assigned as attributes to the trainer
        instance. The method is called after models have been set up, so model
        parameters are available.

        Returns:
            dict[nn.Module, Optimizer]: A mapping from model to their
                corresponding optimizers.

        Example:
            def configure_optimizers(self):
                self.optimizer = torch.optim.AdamW(
                    self.model.parameters(),
                    lr=self.config.learning_rate,
                    weight_decay=self.config.weight_decay
                )
                self.discriminator_opt = torch.optim.SGD(
                    self.discriminator.parameters(),
                    lr=0.01
                )

                return {
                    self.model: self.optimizer,
                    self.discriminator: self.discriminator_opt
                }
        """
        pass

    def configure_schedulers(self) -> dict[Optimizer, LRScheduler]:
        """Configure and instantiate learning rate schedulers.

        This optional method can be overridden to define learning rate schedulers
        for the optimizers. Schedulers should be assigned as attributes and must
        be initialized with their corresponding optimizer instances.

        The method is called after configure_optimizers(), so optimizer instances
        are available via self.optimizer_name or self.get_optimizer().

        Returns:
            dict[Optimizer, LRScheduler]: A mapping from optimizer to their
                corresponding schedulers.

        Example:
            def configure_schedulers(self):
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    T_max=self.config.max_steps
                )
                self.warmup = WarmupScheduler(
                    self.discriminator_optimizer,
                    warmup_steps=self.config.warmup_steps
                )

                return {
                    self.optimizer: self.scheduler,
                    self.discriminator_optimizer: self.warmup
                }
        """
        return {}

    #######################
    # Convenience Methods #
    #######################

    def get_optimizer(self, name: str) -> Optimizer:
        """Retrieve an optimizer by name.

        Args:
            name: The attribute name of the optimizer to retrieve

        Returns:
            Optimizer: The requested optimizer instance

        Raises:
            AttributeError: If no optimizer with the given name exists

        Example:
            >>> opt = trainer.get_optimizer("optimizer")
            >>> opt.step()
        """
        return getattr(self, name)

    def get_scheduler(self, name: str) -> LRScheduler:
        """Retrieve a scheduler by name.

        Args:
            name: The attribute name of the scheduler to retrieve

        Returns:
            LRScheduler: The requested scheduler instance

        Raises:
            AttributeError: If no scheduler with the given name exists

        Example:
            >>> sched = trainer.get_scheduler("scheduler")
            >>> sched.step()
        """
        return getattr(self, name)

    def get_scheduler_by_optimizer(self, name: str) -> LRScheduler:
        """Retrieve a scheduler by optimizer name.

        Args:
            name: The attribute name of the optimizer to retrieve a scheduler for

        Returns:
            LRScheduler: The requested scheduler instance

        Raises:
            AttributeError: If no optimizer with the given name exists

        Example:
            >>> sched = trainer.get_scheduler_by_optimizer("optimizer")
            >>> sched.step()
        """
        return self._optimizer_scheduler_map[name]

    ###################
    # Private Methods #
    ###################

    def _configure_optimizers(self):
        """Internal method to configure optimizers with tracking.

        This method wraps the user's configure_optimizers() call in a context
        that automatically tracks optimizer instances as they're created.
        """
        with configuration_ctx(self, self._optimizer_names, Optimizer):
            model_optimizer_map = self.configure_optimizers()

        for model, optim in model_optimizer_map.items():
            if isinstance(optim, Optimizer):
                self._optimizer_model_map[self.get_name_by_optimizer(optim)] = (
                    self.get_name_by_model(model)
                )
            else:
                for opt in optim:
                    self._optimizer_model_map[self.get_name_by_optimizer(opt)] = (
                        self.get_name_by_model(model)
                    )

    def _configure_schedulers(self):
        """Internal method to configure schedulers and map them to optimizers.

        This method:
        1. Wraps the user's configure_schedulers() call to track scheduler instances
        2. Automatically discovers which scheduler controls which optimizer
        3. Creates the _optimizer_scheduler_map for tracking relationships
        """
        with configuration_ctx(self, self._scheduler_names, LRScheduler):
            optimizer_scheduler_map = self.configure_schedulers()

        for optim, scheduler in optimizer_scheduler_map.items():
            self._optimizer_scheduler_map[self.get_name_by_optimizer(optim)] = (
                self.get_name_by_scheduler(scheduler)
            )

    def _setup_optimizers_and_schedulers(self):
        """Set up all optimizers and schedulers.

        This method initializes the tracking lists and calls the optimizer
        and scheduler configuration methods in the correct order. It ensures
        optimizers are configured before schedulers, as schedulers depend
        on optimizer instances.

        The method is typically called during the trainer's setup phase after
        models have been configured and set up.
        """
        self._optimizer_names: list[str] = []
        self._scheduler_names: list[str] = []

        self._optimizer_model_map: dict[str, str] = {}
        self._optimizer_scheduler_map: dict[str, str] = {}

        self._configure_optimizers()
        self._configure_schedulers()

        logger.info("Setup Optimizers and Schedulers")
