from abc import abstractmethod
from dataclasses import dataclass
from typing import Iterable

from torch.utils.data import DataLoader
from typing_extensions import override

from dream_trainer.trainer.abstract import AbstractTrainer, AbstractTrainerConfig
from dream_trainer.utils import logger


def get_epoch_length(dataloader: DataLoader, length: int | None):
    """Determine the number of steps per epoch for a dataloader.

    This utility function calculates the epoch length, either from an explicitly
    provided value or by querying the dataloader's length. This is useful for
    dataloaders that may not have a well-defined length (e.g., infinite datasets).

    Args:
        dataloader: The DataLoader instance to get the length from
        length: Optional explicit length. If provided, this value is returned
            directly without querying the dataloader

    Returns:
        int: The number of steps/batches per epoch

    Raises:
        ValueError: If length is None and the dataloader's dataset doesn't
            implement __len__, making it impossible to determine epoch length

    Example:
        >>> epoch_len = get_epoch_length(train_loader, config.train_steps_per_epoch)
        >>> for step in range(epoch_len):
        ...     batch = next(dataloader_iter)
    """
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
class DataLoaderSetupConfigMixin(AbstractTrainerConfig):
    """Configuration mixin for dataloader setup functionality.

    This class serves as a base configuration for trainers that need dataloader
    setup capabilities. It inherits from AbstractTrainerConfig and can be extended
    with dataloader-specific parameters like batch sizes, number of workers,
    prefetch settings, and dataset configurations.

    Example:
        @dataclass
        class MyTrainerConfig(DataLoaderSetupConfigMixin):
            batch_size: int = 32
            num_workers: int = 4
            prefetch_factor: int = 2
            train_dataset_path: str = "data/train"
            val_dataset_path: str = "data/val"
    """

    ...


class DataLoaderSetupMixin(AbstractTrainer):
    """Mixin that handles dataloader configuration and setup for training.

    This mixin provides a framework for configuring training and validation
    dataloaders in a trainer. It manages dataloader lifecycle and provides
    convenient property access to the configured dataloaders.

    The mixin expects subclasses to implement configure_dataloaders() to define
    how dataloaders should be created based on the configuration.

    Attributes:
        config (DataLoaderSetupConfigMixin): Configuration for dataloaders
        _train_dataloader (Iterable): Internal training dataloader instance
        _val_dataloader (Iterable): Internal validation dataloader instance

    Example:
        class MyTrainer(DataLoaderSetupMixin):
            def configure_dataloaders(self):
                train_dataset = MyDataset(self.config.train_path)
                val_dataset = MyDataset(self.config.val_path)

                train_loader = DataLoader(
                    train_dataset,
                    batch_size=self.config.batch_size,
                    shuffle=True
                )
                val_loader = DataLoader(
                    val_dataset,
                    batch_size=self.config.batch_size,
                    shuffle=False
                )

                return train_loader, val_loader
    """

    config: DataLoaderSetupConfigMixin

    ###########################
    # AbstractTrainer Methods #
    ###########################

    @property
    @override
    def train_dataloader(self) -> Iterable:
        """Access the training dataloader.

        Returns:
            Iterable: The configured training dataloader instance. This is typically
                a torch.utils.data.DataLoader but can be any iterable that yields
                batches of training data.

        Note:
            The dataloader is created during the setup phase by calling
            configure_dataloaders(). Accessing this property before setup
            will raise an AttributeError.
        """
        return self._train_dataloader

    @property
    @override
    def val_dataloader(self) -> Iterable:
        """Access the validation dataloader.

        Returns:
            Iterable: The configured validation dataloader instance. This is typically
                a torch.utils.data.DataLoader but can be any iterable that yields
                batches of validation data.

        Note:
            The dataloader is created during the setup phase by calling
            configure_dataloaders(). Accessing this property before setup
            will raise an AttributeError.
        """
        return self._val_dataloader

    ########################
    # User-Defined Methods #
    ########################

    @abstractmethod
    def configure_dataloaders(self) -> tuple[Iterable, Iterable]:
        """Configure and create training and validation dataloaders.

        This method must be implemented by subclasses to define how dataloaders
        are created. It should instantiate and return both training and validation
        dataloaders based on the trainer's configuration.

        The method is called during the setup phase after models and optimizers
        have been configured, allowing dataloaders to be aware of model parallelism
        settings if needed.

        Returns:
            tuple[Iterable, Iterable]: A tuple containing (train_dataloader, val_dataloader).
                Both should be iterables that yield batches of data. Typically these are
                torch.utils.data.DataLoader instances.

        Example:
            def configure_dataloaders(self):
                # Create datasets
                train_dataset = TextDataset(
                    self.config.train_path,
                    tokenizer=self.tokenizer,
                    max_length=self.config.max_seq_len
                )
                val_dataset = TextDataset(
                    self.config.val_path,
                    tokenizer=self.tokenizer,
                    max_length=self.config.max_seq_len
                )

                # Create dataloaders with distributed sampling if needed
                train_sampler = DistributedSampler(
                    train_dataset,
                    num_replicas=self.world.size,
                    rank=self.world.rank
                ) if self.world.size > 1 else None

                train_loader = DataLoader(
                    train_dataset,
                    batch_size=self.config.batch_size,
                    sampler=train_sampler,
                    shuffle=(train_sampler is None),
                    num_workers=self.config.num_workers,
                    pin_memory=True
                )

                val_loader = DataLoader(
                    val_dataset,
                    batch_size=self.config.batch_size,
                    shuffle=False,
                    num_workers=self.config.num_workers,
                    pin_memory=True
                )

                return train_loader, val_loader
        """
        pass

    def _setup_dataloaders(self):
        """Internal method to setup dataloaders.

        This method calls the user-defined configure_dataloaders() and stores
        the returned dataloaders as internal attributes. It's called during
        the trainer's setup phase as the final step after models and optimizers
        have been configured.
        """
        self._train_dataloader, self._val_dataloader = self.configure_dataloaders()
        logger.info("Setup Dataloaders")
