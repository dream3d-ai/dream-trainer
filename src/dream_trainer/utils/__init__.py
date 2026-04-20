"""Shared utility exports for Dream Trainer."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._logger import logger

__all__ = ["logger"]


def __getattr__(name: str):
    if name == "logger":
        from ._logger import logger

        return logger
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
