from abc import abstractmethod
from dataclasses import dataclass
from typing import override

from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer

from dream_trainer.trainer.abstract import AbstractTrainer, AbstractTrainerConfig
from dream_trainer.utils import logger
from dream_trainer.utils.common import configuration_ctx


@dataclass(kw_only=True)
class OptimizerAndSchedulerSetupConfigMixin(AbstractTrainerConfig): ...


class OptimizerAndSchedulerSetupMixin(AbstractTrainer):
    config: OptimizerAndSchedulerSetupConfigMixin

    ###########################
    # AbstractTrainer Methods #
    ###########################

    @override
    def named_optimizers(self) -> dict[str, Optimizer]:
        return {name: getattr(self, name) for name in self._optimizer_names}

    @override
    def named_schedulers(self) -> dict[str, LRScheduler]:
        return {name: getattr(self, name) for name in self._scheduler_names}

    ########################
    # User-Defined Methods #
    ########################

    @abstractmethod
    def configure_optimizers(self):
        pass

    def configure_schedulers(self):
        pass

    #######################
    # Convenience Methods #
    #######################

    def get_optimizer(self, name: str) -> Optimizer:
        return getattr(self, name)

    def get_scheduler(self, name: str) -> LRScheduler:
        return getattr(self, name)

    ###################
    # Private Methods #
    ###################

    def _configure_optimizers(self):
        with configuration_ctx(self, self._optimizer_names, Optimizer):
            self.configure_optimizers()

    def _configure_schedulers(self):
        with configuration_ctx(self, self._scheduler_names, LRScheduler):
            self.configure_schedulers()

        # Find which scheduler controls which optimizer
        self._optimizer_scheduler_map: dict[str, str | None] = {
            optimizer_name: next(
                (
                    scheduler_name
                    for scheduler_name, scheduler in self.named_schedulers().items()
                    if scheduler.optimizer is optimizer
                ),
                None,
            )
            for optimizer_name, optimizer in self.named_optimizers().items()
        }

    def _setup_optimizers_and_schedulers(self):
        self._optimizer_names: list[str] = []
        self._scheduler_names: list[str] = []

        self._configure_optimizers()
        self._configure_schedulers()

        logger.info("Setup Optimizers and Schedulers")
