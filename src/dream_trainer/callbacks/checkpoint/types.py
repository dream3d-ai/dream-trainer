import re
from dataclasses import dataclass
from pathlib import Path

CHECKPOINT_REGEX = re.compile(r"step=(\d+),metric=([+-]?\d+\.\d+)")


@dataclass(kw_only=True)
class Checkpoint:
    step: int
    metric: float

    @property
    def checkpoint_id(self):
        return f"step={self.step:03d},metric={self.metric:.6f}"

    @classmethod
    def from_path(cls, path: Path) -> "Checkpoint":
        match = CHECKPOINT_REGEX.search(path.name)
        if match is None:
            raise ValueError(f"Invalid checkpoint path {path}")

        step = int(match.group(1))
        metric = float(match.group(2))
        return cls(step=step, metric=metric)

    def __str__(self):
        return f"Checkpoint(step={self.step}, metric={self.metric:.2f})"

    def __eq__(self, other):
        if isinstance(other, Checkpoint):
            return self.step == other.step
        raise ValueError("Cannot compare Checkpoint to non-Checkpoint object")
