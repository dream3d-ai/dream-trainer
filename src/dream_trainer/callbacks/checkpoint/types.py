import os
import re
from dataclasses import dataclass
from functools import total_ordering
from pathlib import Path

CHECKPOINT_REGEX = re.compile(r"step=(\d+)__metric=([\d_]+)")


@dataclass(kw_only=True)
@total_ordering
class Checkpoint:
    step: int
    metric: float

    @property
    def checkpoint_id(self) -> str:
        return f"step={self.step:06d}__metric={self.metric:.6f}".replace(".", "_")

    @staticmethod
    def is_valid_checkpoint_path(path: Path) -> bool:
        return CHECKPOINT_REGEX.search(path.name) is not None and os.path.exists(
            path / ".metadata"
        )

    @classmethod
    def from_path(cls, path: Path) -> "Checkpoint":
        match = CHECKPOINT_REGEX.search(path.name)
        if match is None:
            raise ValueError(f"Invalid checkpoint path {path}")

        step = int(match.group(1))
        metric = float(match.group(2).replace("_", "."))
        return cls(step=step, metric=metric)

    def __str__(self):
        return f"Checkpoint(step={self.step}, metric={self.metric:.2f})"

    def __eq__(self, other):
        if not isinstance(other, Checkpoint):
            return NotImplemented

        return (self.metric, self.step) == (other.metric, other.step)

    def __lt__(self, other):
        if not isinstance(other, Checkpoint):
            return NotImplemented

        return (self.metric, self.step) < (other.metric, other.step)
