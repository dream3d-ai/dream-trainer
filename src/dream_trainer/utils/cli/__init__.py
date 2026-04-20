"""Typer-based CLI for dream-trainer training scripts.

See [dream_trainer.utils.cli.cli][] for usage; importing this package
also normalises `sys.argv` to work around VSCode's debug launcher
quoting single-argv entries containing spaces.
"""

from .cli import cli

__all__ = ["cli"]


import sys

argv = [arg.split(" ") for arg in sys.argv]
argv = [item for sublist in argv for item in sublist]
sys.argv = argv
