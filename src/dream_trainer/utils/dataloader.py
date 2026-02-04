from math import ceil
from typing import Any, Iterable

from dream_trainer.utils import logger

Batch = dict[str, Any]


def get_dataloader_length(dataloader: Iterable) -> int | None:
    try:
        return len(dataloader)  # type: ignore
    except TypeError:
        return None


def get_train_dataloader_steps(
    dataloader: Iterable,
    train_steps_per_epoch: int | None,
    gradient_accumulation_steps: int = 1,
    dp_size: int = 1,
) -> tuple[int, int, int]:
    """
    Calculate training dataloader steps, global batch size, and gradient accumulation steps.

    Args:
        dataloader (Iterable): The training dataloader.
        train_steps_per_epoch (int | None): Number of training steps per epoch. If None, uses the length of the dataloader.
        gradient_accumulation_steps (int): Number of steps to accumulate gradients before optimizer step.
        dp_size (int): Data parallel size (number of processes/devices).

    Returns:
        tuple: (global_batch_size, num_train_steps, gradient_accumulation_steps)
            - global_batch_size (int): The total batch size for training (dataloader_batch_size * dp_size * gradient_accumulation_steps).
            - num_train_steps (int): Number of training steps per epoch (possibly adjusted for gradient accumulation).
            - gradient_accumulation_steps (int): Number of steps to accumulate gradients before optimizer step.

    Raises:
        ValueError: If batch size cannot be determined from the dataloader.
    """
    dataloader_length = get_dataloader_length(dataloader)
    num_train_steps = train_steps_per_epoch or dataloader_length

    if num_train_steps is None:
        raise ValueError(
            f"The underlying dataset of {dataloader} does not have __len__ defined. "
            f"Please specify training_parameters.{{stage}}_steps_per_epoch or train_steps_per_epoch instead. "
        )

    if dataloader_length is not None and num_train_steps > dataloader_length:
        logger.warning(
            f"train_steps_per_epoch, {train_steps_per_epoch}, "
            f"is greater than the number of batches in the dataloader, {dataloader_length}. ",
        )

    dataloader_batch_size: int | None = getattr(dataloader, "batch_size", None)
    if dataloader_batch_size is None:
        dataloader_batch_size = getattr(getattr(dataloader, "dataset", {}), "batch_size", None)

    if dataloader_batch_size is None:
        raise ValueError(
            "Neither dataloader nor dataloader.dataset has non-None 'batch_size' attribute. "
            "Please ensure one or the other specifies an integer batch size "
            "to correctly compute the global batch size."
        )

    global_batch_size: int = dataloader_batch_size * dp_size * gradient_accumulation_steps

    # _num_train_batches is the number of dataloader batches per epoch
    if train_steps_per_epoch is not None:
        num_train_steps *= gradient_accumulation_steps

    return global_batch_size, num_train_steps, gradient_accumulation_steps


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
    dataloader_length = get_dataloader_length(dataloader)
    _num_val_batches = val_steps_per_epoch or dataloader_length

    if _num_val_batches is None:
        raise ValueError(
            f"The underlying dataset of {dataloader} does not have __len__ defined. "
            f"Please specify val_steps_per_epoch instead. "
        )

    if dataloader_length is not None and _num_val_batches > dataloader_length:
        logger.warning(
            f"val_batches_per_epoch, {val_steps_per_epoch}, "
            f"is greater than the number of batches in the dataloader, {dataloader_length}. "
        )

    return _num_val_batches // dp_size, ceil(num_sanity_val_steps / dp_size)
