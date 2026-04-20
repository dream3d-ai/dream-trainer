"""Rich-powered config pretty-printer used by the `--cfg` CLI flag."""

from dream_trainer.utils.serialize import asdict


def print_config(trainer_config) -> None:
    """Pretty-print a trainer config to stdout.

    The config is first converted to a plain dict (with private fields and
    non-serialisable values filtered out), then rendered with `rich`.

    Args:
        trainer_config: Any dataclass-based trainer config (typically a
            `DreamTrainerConfig` subclass).
    """
    from rich import print as rprint
    from rich.pretty import Pretty

    rprint(Pretty(asdict(trainer_config)))
