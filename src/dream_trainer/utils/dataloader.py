from itertools import repeat
from typing import Any, Iterable, Iterator, cast

import dist_utils.ops as dist_ops
import torch
from dist_utils.ops import apply_to_collection, broadcast_collection
from torch.distributed.device_mesh import DeviceMesh

from dream_trainer.utils import logger

Batch = dict[str, Any]


class DataLoaderCycler:
    """
    A utility class to cycle through a DataLoader indefinitely, with optional support for
    broadcasting batches across a device mesh for distributed training.

    Args:
        dataloader (DataLoader[Batch]): The DataLoader to cycle through.
        broadcast_locally (bool): If True, broadcast batches to all local devices in the mesh.
        device_mesh (DeviceMesh | None): The device mesh for distributed training.
        device (torch.device | None): The device to move batches to.
    """

    def __init__(
        self,
        dataloader: Iterable[Batch],
        device_mesh: DeviceMesh | None = None,
        device: torch.device | None = None,
    ):
        """
        Initialize the DataLoaderCycler.

        Args:
            dataloader (DataLoader[Batch]): The DataLoader to cycle through.
            device_mesh (DeviceMesh | None, optional): The device mesh to which to broadcast batches.
            device (torch.device | None, optional): The device to move batches to. Defaults to None.
        """
        self.dataloader = dataloader
        self.device_mesh = device_mesh
        self.device = device

    def _standard_iterator(self) -> Iterator[Batch]:
        """
        Standard infinite iterator over the dataloader, moving each batch to the specified device.

        Yields:
            Batch: The next batch from the dataloader, with tensors moved to the specified device.
        """
        for batch in self.dataloader:
            yield apply_to_collection(
                cast(Batch, batch),
                function=lambda t: t.to(self.device),
                dtype=torch.Tensor,
            )

    def _broadcast_iterator(self) -> Iterator[Batch]:
        """
        Infinite iterator that broadcasts each batch from rank 0 to all other ranks in the device mesh.

        Yields:
            Batch: The next batch, broadcasted to all local devices and moved to the specified device.

        Raises:
            ValueError: If device_mesh is not set.
        """
        if self.device_mesh is None:
            raise ValueError("Device mesh is not set")

        process_group = self.device_mesh.get_group()
        local_rank = self.device_mesh.get_local_rank()

        iterator = cast(
            Iterator[Batch],
            iter(self.dataloader) if local_rank == 0 else ({} for _ in repeat(None)),
        )
        if local_rank != 0:  # Clean-up dataloaders on other processes
            del self.dataloader

        # TODO: could make broadcast_collection async and broadcast the next batch eagerly
        for batch in iterator:
            yield broadcast_collection(batch, process_group, device=self.device)

    def __iter__(self) -> Iterator[Batch]:
        while True:
            it = (
                self._broadcast_iterator()
                if self.device_mesh is not None
                else self._standard_iterator()
            )
            num_batches = 0
            for batch in it:
                yield batch
                num_batches += 1

            assert dist_ops.global_agreement(num_batches), (
                "Ranks received different number of batches"
            )

    def state_dict(self) -> dict[str, Any]:
        state_dict: dict[str, Any] = {}
        if hasattr(self, "dataloader") and hasattr(self.dataloader, "state_dict"):
            state_dict = self.dataloader.state_dict()  # type: ignore

        if state_dict and self.device_mesh is not None:
            state_dict = broadcast_collection(
                state_dict, self.device_mesh.get_group() if self.device_mesh else None
            )
        return state_dict

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if hasattr(self.dataloader, "load_state_dict"):
            self.dataloader.load_state_dict(state_dict)  # type: ignore


def get_epoch_length(dataloader: Iterable, length: int | None) -> int:
    if length is not None:
        return length

    try:
        return len(dataloader)  # type: ignore
    except TypeError:
        raise ValueError(
            f"The underlying dataset of {dataloader} does not have __len__ defined. "
            f"Please specify training_parameters.{{stage}}_steps_per_epoch instead. "
        )


def get_train_dataloader_steps(
    dataloader: Iterable,
    train_steps_per_epoch: int | None,
    train_batch_size: int = 1,
    dp_size: int = 1,
) -> tuple[int, int, int]:
    """
    Calculate training dataloader steps, effective minibatch size, and gradient accumulation steps.

    Args:
        dataloader (Iterable): The training dataloader.
        train_steps_per_epoch (int | None): Number of training steps per epoch. If None, uses the length of the dataloader.
        train_batch_size (int): The total batch size for training (across all devices/processes).
        dp_size (int): Data parallel size (number of processes/devices).

    Returns:
        tuple: (train_batch_size, num_train_steps, gradient_accumulation_steps)
            - train_batch_size (int): The total batch size for training.
            - num_train_steps (int): Number of training steps per epoch (possibly adjusted for gradient accumulation).
            - gradient_accumulation_steps (int): Number of steps to accumulate gradients before optimizer step.

    Raises:
        ValueError: If batch size cannot be determined, or if effective minibatch size is invalid.
    """
    num_train_steps: int = get_epoch_length(dataloader, train_steps_per_epoch)

    if train_steps_per_epoch is not None and train_steps_per_epoch > num_train_steps:
        logger.warning(
            f"train_steps_per_epoch, {train_steps_per_epoch}, "
            f"is greater than the number of batches in the dataloader, {num_train_steps}. ",
        )

    dataloader_batch_size: int | None = getattr(dataloader, "batch_size", None)
    if dataloader_batch_size is None:
        dataloader_batch_size = getattr(getattr(dataloader, "dataset", {}), "batch_size", None)

    if dataloader_batch_size is None:
        raise ValueError(
            "Neither dataloader nor dataloader.dataset has non-None 'batch_size' attribute. "
            "Please ensure one or the other specifies an integer batch size "
            "to correctly compute the effective minibatch size and gradient accumulation."
        )

    effective_minibatch_size: int = dataloader_batch_size * dp_size

    if effective_minibatch_size > train_batch_size:
        raise ValueError(
            f"Effective minibatch size, {effective_minibatch_size}, is greater than train_batch_size, {train_batch_size}"
        )
    if train_batch_size % effective_minibatch_size != 0:
        raise ValueError(
            f"train_batch_size, {train_batch_size}, must be divisible by effective minibatch size, {effective_minibatch_size}"
        )

    gradient_accumulation_steps = train_batch_size // effective_minibatch_size

    # _num_train_batches is the number of dataloader batches per epoch
    if train_steps_per_epoch is not None:
        num_train_steps *= gradient_accumulation_steps

    return train_batch_size, num_train_steps, gradient_accumulation_steps


def get_val_dataloader_steps(
    dataloader: Iterable,
    val_steps_per_epoch: int | None,
    num_sanity_val_steps: int = 0,
    dp_size: int = 1,
) -> tuple[int, int]:
    """
    Calculate validation dataloader steps and sanity validation steps, accounting for data parallelism.

    Args:
        dataloader (Iterable): The validation dataloader.
        val_steps_per_epoch (int | None): Number of validation steps per epoch. If None, uses the length of the dataloader.
        num_sanity_val_steps (int): Number of sanity validation steps to run before training.
        dp_size (int): Data parallel size (number of processes/devices).

    Returns:
        tuple: (num_val_batches, num_sanity_val_steps)
            - num_val_batches (int): Number of validation batches per epoch (divided by dp_size).
            - num_sanity_val_steps (int): Number of sanity validation steps (divided by dp_size).
    """
    _num_val_batches: int = get_epoch_length(dataloader, val_steps_per_epoch)
    if val_steps_per_epoch is not None and val_steps_per_epoch > _num_val_batches:
        logger.warning(
            f"val_batches_per_epoch, {val_steps_per_epoch}, "
            f"is greater than the number of batches in the dataloader, {_num_val_batches}. "
        )

    return _num_val_batches // dp_size, num_sanity_val_steps // dp_size
