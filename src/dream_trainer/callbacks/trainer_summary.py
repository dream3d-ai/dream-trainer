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


def summarize_models(
    named_models: dict[str, nn.Module], title: str = "Model Summary", depth: int = 1
):
    num_params = lambda module: sum(p.numel() for p in module.parameters())
    trainable = lambda module: sum(p.numel() for p in module.parameters() if p.requires_grad)
    size = lambda module: sum(p.element_size() * p.numel() for p in module.parameters())
    fmt_params = lambda x: tqdm.format_sizeof(x).replace("G", "B")
    fmt_size = lambda x: tqdm.format_sizeof(x, "B", 1024)

    def collect_rows(name: str, module: nn.Module, current_depth: int):
        rows = [
            (name, type(module).__name__, num_params(module), trainable(module), size(module))
        ]
        if current_depth < depth:
            for child_name, child in module.named_children():
                rows.extend(collect_rows(f"{name}.{child_name}", child, current_depth + 1))
        return rows

    rows: list[tuple[str, str, int, int, int]] = []
    for name, module in named_models.items():
        rows.extend(collect_rows(name, module, 1))

    total_params = sum(num_params(m) for m in named_models.values())
    total_trainable = sum(trainable(m) for m in named_models.values())
    total_size = sum(size(m) for m in named_models.values())
    non_trainable = total_params - total_trainable

    table = Table(title=title)
    console = Console()
    for col in ("Name", "Type", "Params", "Trainable Params", "Size"):
        table.add_column(col, justify="right", no_wrap=True)

    for name, type_name, p, tp, sz in rows:
        table.add_row(name, type_name, fmt_params(p), fmt_params(tp), fmt_size(sz))

    console.print()
    console.print(table)
    console.print(f"Trainable Parameters: {tqdm.format_sizeof(total_trainable)}")
    if non_trainable > 0:
        console.print(f"Non-Trainable Parameters: {tqdm.format_sizeof(non_trainable)}")
    console.print(f"Total Parameters: {tqdm.format_sizeof(total_params)}")
    console.print(f"Total Size: {tqdm.format_sizeof(total_size, 'B', 1024)}")
    console.print()


def summarize_dataloaders(
    train_dataloader: Iterable,
    val_dataloader: Iterable,
    title: str = "Dataloader Summary",
):
    # Check if dataloaders have their own summary attribute
    train_summary = getattr(train_dataloader, "summarize", None)
    val_summary = getattr(val_dataloader, "summarize", None)

    if train_summary is not None and val_summary is not None:
        console = Console()
        console.print()
        console.print("[bold]Train Dataloader Summary:[/bold]")
        console.print(train_summary())
        console.print()
        console.print("[bold]Validation Dataloader Summary:[/bold]")
        console.print(val_summary())
        return

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


def summarize(trainer: AbstractTrainer, depth: int = 1):
    summarize_models(trainer.named_models(), title="Model Summary", depth=depth)
    summarize_dataloaders(
        trainer.train_dataloader, trainer.val_dataloader, title="Dataloader Summary"
    )


class TrainerSummary(Callback):
    def __init__(self, depth: int = 1):
        super().__init__()
        self.depth = depth

    @override
    def pre_fit(self):
        summarize(self.trainer, depth=self.depth)
