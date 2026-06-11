from loguru import logger


def setup_logger():
    """
    Sets up Loguru logging pretty

    INFO messages of the form:
        00:00 | INFO     | Sets up Loguru logging pretty
    DEBUG messages of the form:
        00:00 | DEBUG    | dream_trainer.utils:setup_logger:13 - Sets up Loguru debug logging pretty too
    """
    import os
    import sys
    from functools import partial
    from typing import Literal, cast

    from tqdm import tqdm

    os.environ["LOGURU_LEVEL"] = os.environ.get(
        "LOGURU_LEVEL", os.environ.get("LOG_LEVEL", "INFO")
    )
    LOG_LEVEL = cast(
        Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], os.environ["LOGURU_LEVEL"]
    )

    def _format_string(record):
        t = record["elapsed"]
        elapsed = f"{t.seconds // 60:02}:{t.seconds % 60:02}.{t.microseconds // 1000:03}"
        return (
            f"<green>{elapsed}</green> | "
            f"<level>{record['level']: <8}</level> | "
            "<level>{message}</level>"
            "\n"
        )

    def _format_debug_string(record):
        t = record["elapsed"]
        elapsed = f"{t.seconds // 60:02}:{t.seconds % 60:02}.{t.microseconds // 1000}"
        return (
            f"<green>{elapsed}</green> | "
            f"<level>{record['level']: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
            "\n"
        )

    if os.environ.get("NO_TQDM_WRITE", "0") == "1":
        sink = sys.stdout
    else:
        # Use TQDM write to log nicely with
        sink = partial(tqdm.write, end="")

    logger.remove()
    logger.add(
        sink,
        format=_format_string,
        filter=lambda r: r["level"].name != "DEBUG",
        level=LOG_LEVEL,
        colorize=True,
    )
    logger.add(
        sink,
        format=_format_debug_string,
        filter=lambda r: r["level"].name == "DEBUG",
        level=LOG_LEVEL,
        colorize=True,
    )


setup_logger()

__all__ = [
    "logger",
]
