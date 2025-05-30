from typing import Iterable

import torch.nn as nn
from tqdm import tqdm
from typing_extensions import override

from dream_trainer.trainer.abstract import AbstractTrainer
from dream_trainer.utils import logger

from .callback import Callback

try:
    from rich.console import Console  # type: ignore # noqa: F401
    from rich.table import Table  # type: ignore # noqa: F401
except ImportError as e:
    raise ImportError(
        "rich is not installed. Please install it with `pip install dream-trainer[rich]` to use the TrainerSummary callback."
    ) from e


def summarize_models(named_models: dict[str, nn.Module], title: str = "Model Summary"):
    num_params = lambda module: sum(p.numel() for p in module.parameters())
    trainable = lambda module: sum(p.numel() for p in module.parameters() if p.requires_grad)
    size = lambda module: sum(p.element_size() * p.numel() for p in module.parameters())

    summary_data = [
        (name, type(module).__name__, num_params(module), trainable(module), size(module))
        for name, module in named_models.items()
    ]

    summary_data = list(zip(*summary_data))

    total_params = sum(summary_data[2])
    total_trainable_params = sum(summary_data[3])
    total_size = sum(summary_data[4])
    non_trainable_params = total_params - total_trainable_params

    summary_data[2] = tuple(
        map(lambda x: tqdm.format_sizeof(x).replace("G", "B"), summary_data[2])
    )
    summary_data[3] = tuple(
        map(lambda x: tqdm.format_sizeof(x).replace("G", "B"), summary_data[3])
    )
    summary_data[4] = tuple(map(lambda x: tqdm.format_sizeof(x, "B", 1024), summary_data[4]))

    summary_data = list(
        zip(("Name", "Type", "Params", "Trainable Params", "Size"), summary_data)
    )

    table = Table(title=title)
    console = Console()
    for column in summary_data:
        table.add_column(column[0], justify="right", no_wrap=True)

    for row in zip(*[column[1] for column in summary_data]):
        table.add_row(*row)

    console.print()
    console.print(table)
    console.print(f"Trainable Parameters: {tqdm.format_sizeof(total_trainable_params)}")
    if non_trainable_params > 0:
        console.print(f"Non-Trainable Parameters: {tqdm.format_sizeof(non_trainable_params)}")
    console.print(f"Total Parameters: {tqdm.format_sizeof(total_params)}")
    console.print(f"Total Size: {tqdm.format_sizeof(total_size, 'B', 1024)}")
    console.print()


def summarize_dataloaders(
    train_dataloader: Iterable,
    val_dataloader: Iterable,
    title: str = "Dataloader Summary",
):
    train_batch_size = getattr(train_dataloader, "batch_size", None) or getattr(
        getattr(train_dataloader, "dataset", {}), "batch_size", None
    )

    val_batch_size = getattr(val_dataloader, "batch_size", None) or getattr(
        getattr(val_dataloader, "dataset", {}), "batch_size", None
    )

    if train_batch_size is None or val_batch_size is None:
        logger.warning("Batch size not found. Cannot summarize dataloaders.")
        return

    table = Table(title=title)
    console = Console()
    table.add_column("Dataloader")
    table.add_column("Split")
    table.add_column("Batches")
    table.add_column("Batch Size")

    table.add_row(
        getattr(train_dataloader, "dataset", train_dataloader).__class__.__name__,
        "Train",
        f"{getattr(train_dataloader, '__len__', lambda: 0)():,}",
        f"{train_batch_size:,}",
    )
    table.add_row(
        getattr(val_dataloader, "dataset", val_dataloader).__class__.__name__,
        "Validation",
        f"{getattr(val_dataloader, '__len__', lambda: 0)():,}",
        f"{val_batch_size:,}",
    )

    console.print()
    console.print(table)
    console.print()


def summarize(trainer: AbstractTrainer):
    summarize_models(trainer.named_models(), title="Model Summary")
    summarize_dataloaders(
        trainer.train_dataloader, trainer.val_dataloader, title="Dataloader Summary"
    )


class TrainerSummary(Callback):
    @override
    def pre_fit(self):
        summarize(self.trainer)
