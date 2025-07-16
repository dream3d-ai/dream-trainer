import os
from pathlib import Path
from typing import Literal

from .types import Checkpoint


def sort_checkpoints(
    checkpoints: list[Checkpoint],
    mode: Literal["min", "max", "last"],
) -> list[Checkpoint]:
    if mode == "last":
        return sorted(checkpoints, key=lambda c: c.step, reverse=True)
    elif mode == "min":
        return sorted(checkpoints)
    elif mode == "max":
        return sorted(checkpoints, reverse=True)
    else:
        raise ValueError(f"Invalid resume mode {mode}")


def find_checkpoints(
    checkpoint_dir: Path, mode: Literal["min", "max", "last"] = "last"
) -> list[Checkpoint]:
    """
    Finds all checkpoints in the given directory and sorts them by the given mode.
    """
    checkpoints = [
        Checkpoint.from_path(checkpoint_dir / f)
        for f in os.listdir(checkpoint_dir)
        if Checkpoint.is_valid_checkpoint_path(checkpoint_dir / f)
    ]

    return sort_checkpoints(checkpoints, mode)


def find_top_k_checkpoints(
    checkpoint_dir: Path,
    mode: Literal["min", "max", "last"],
    k: int = 1,
) -> list[Checkpoint]:
    return find_checkpoints(checkpoint_dir, mode)[:k]


def find_current_checkpoint(
    checkpoint_dir: Path,
    mode: Literal["min", "max", "last"],
) -> Checkpoint | None:
    checkpoints = find_top_k_checkpoints(checkpoint_dir, mode, k=1)
    return checkpoints[0] if len(checkpoints) == 1 else None
