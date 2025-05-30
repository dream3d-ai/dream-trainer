from abc import abstractmethod
from dataclasses import dataclass
from typing import Iterable

from torch.utils.data import DataLoader
from typing_extensions import override

from dream_trainer.trainer.abstract import AbstractTrainer, AbstractTrainerConfig
from dream_trainer.utils import logger


def get_epoch_length(dataloader: DataLoader, length: int | None):
    if length is not None:
        return length

    try:
        return len(dataloader)
    except TypeError:
        raise ValueError(
            f"The underlying dataset of {dataloader} does not have __len__ defined. "
            f"Please specify training_parameters.{{stage}}_steps_per_epoch instead. "
        )


@dataclass(kw_only=True)
class DataLoaderSetupConfigMixin(AbstractTrainerConfig): ...


class DataLoaderSetupMixin(AbstractTrainer):
    config: DataLoaderSetupConfigMixin

    ###########################
    # AbstractTrainer Methods #
    ###########################

    @property
    @override
    def train_dataloader(self) -> Iterable:
        return self._train_dataloader

    @property
    @override
    def val_dataloader(self) -> Iterable:
        return self._val_dataloader

    ########################
    # User-Defined Methods #
    ########################

    @abstractmethod
    def configure_dataloaders(self) -> tuple[Iterable, Iterable]:
        """
        Configure and return a tuple of train and validation dataloader.
        """
        pass

    def _setup_dataloaders(self):
        self._train_dataloader, self._val_dataloader = self.configure_dataloaders()
        logger.info("Setup Dataloaders")
