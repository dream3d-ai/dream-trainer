import logging
import os
import sys
from functools import partial
from typing import Literal, cast

from tqdm import tqdm

LOG_LEVEL = cast(
    Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    os.environ.get("LOG_LEVEL", "INFO"),
)
LOGGER_NAME = "dream_trainer"
logger = logging.getLogger(LOGGER_NAME)


class _DreamTrainerFormatter(logging.Formatter):
    """Match the compact elapsed-time format previously used by loguru."""

    def format(self, record: logging.LogRecord) -> str:
        elapsed_ms = int(record.relativeCreated)
        minutes, remainder = divmod(elapsed_ms, 60_000)
        seconds, millis = divmod(remainder, 1_000)
        elapsed = f"{minutes:02}:{seconds:02}.{millis:03}"
        message = record.getMessage()

        if record.levelno == logging.DEBUG:
            return (
                f"{elapsed} | {record.levelname:<8} | "
                f"{record.module}:{record.funcName}:{record.lineno} - {message}\n"
            )

        return f"{elapsed} | {record.levelname:<8} | {message}\n"


class _TqdmLoggingHandler(logging.Handler):
    """Write logs through tqdm when available so progress bars stay intact."""

    def __init__(self) -> None:
        super().__init__()
        self._use_tqdm = os.environ.get("NO_TQDM_WRITE", "0") != "1"
        self._tqdm_write = partial(tqdm.write, end="") if self._use_tqdm else None

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = self.format(record)
            if self._tqdm_write is not None:
                self._tqdm_write(message)
            else:
                sys.stdout.write(message)
                sys.stdout.flush()
        except Exception:
            self.handleError(record)


def setup_logger(
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = LOG_LEVEL,
) -> logging.Logger:
    """Configure dream-trainer's stdlib logger."""

    os.environ["LOG_LEVEL"] = level

    handler = _TqdmLoggingHandler()
    handler.setLevel(logging.NOTSET)
    handler.setFormatter(_DreamTrainerFormatter())

    logger.handlers.clear()
    logger.setLevel(logging._nameToLevel[level])
    logger.propagate = False
    logger.addHandler(handler)

    return logger


setup_logger()

__all__ = ["logger", "setup_logger"]
