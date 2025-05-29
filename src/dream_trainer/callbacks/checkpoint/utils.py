import os
from pathlib import Path
from typing import Literal

from .types import Checkpoint


def sort_checkpoints(
    checkpoints: list[Checkpoint],
    mode: Literal["min", "max", "last"],
) -> list[Checkpoint]:
    checkpoints = sorted(checkpoints, key=lambda c: c.step)

    if mode == "last":
        return checkpoints
    elif mode == "min":
        return sorted(checkpoints, key=lambda c: c.metric)
    elif mode == "max":
        return sorted(checkpoints, key=lambda c: c.metric, reverse=True)
    else:
        raise ValueError(f"Invalid resume mode {mode}")


def find_checkpoints(
    checkpoint_dir: Path, mode: Literal["min", "max", "last"] = "last"
) -> list[Checkpoint]:
    checkpoints = []
    for f in os.listdir(checkpoint_dir):
        try:
            checkpoints.append(Checkpoint.from_path(checkpoint_dir / f))
        except ValueError:
            pass

    return sort_checkpoints(checkpoints, mode)


def find_top_k_checkpoints(
    checkpoint_dir: Path,
    mode: Literal["min", "max", "last"],
    k: int = 1,
) -> list[Checkpoint]:
    checkpoints = find_checkpoints(checkpoint_dir, mode)
    return checkpoints[:k]


def find_current_checkpoint(
    checkpoint_dir: Path,
    mode: Literal["min", "max", "last"],
) -> Checkpoint | None:
    checkpoints = find_top_k_checkpoints(checkpoint_dir, mode, k=1)
    if len(checkpoints) == 0:
        return None
    return checkpoints[0]
