# CLI Utilities API

The `dream_trainer.utils.cli` package provides a typer-based command-line
wrapper for training scripts, together with a lightweight registry of
config *modifiers* that are auto-surfaced as `--<flag>` options.

Modules:

- `dream_trainer.utils.cli` — the `cli(main, trainer_config)` entrypoint
  and its subcommands (`benchmark`, `profile`, `importtime`,
  `find-graph-breaks`, `summarize`, `export`).
- `dream_trainer.utils.modifiers` — the `MODIFIERS` registry and the
  `@register_modifier` decorator used to add new flags.

## Installation

The CLI dependencies are optional — install with the `cli` extra:

```bash
pip install "dream-trainer[cli]"
```

## Entrypoint

Wrap your training function in `cli(...)` and you get a full-featured
command-line interface for free:

```python
from dream_trainer.utils.cli import cli

def main(config: MyTrainerConfig):
    MyTrainer(config).fit()

if __name__ == "__main__":
    cli(main, MyTrainerConfig())
```

Running `python train.py --help` will now list every registered modifier,
plus the built-in `--resume`, `--init-from`, and `--cfg` options along
with the debugging and profiling subcommands.

::: dream_trainer.utils.cli.cli

## Modifier Registry

Modifiers are small functions that mutate a trainer config in place and
are exposed as CLI flags. Register them with the `@register_modifier`
decorator:

```python
from dream_trainer import DreamTrainerConfig
from dream_trainer.utils.modifiers import register_modifier

@register_modifier()
def no_compile(config: DreamTrainerConfig):
    """Disable torch.compile."""
    config.device_parameters.compile_model = False
```

::: dream_trainer.utils.modifiers.register

## Built-in Modifiers

A collection of generic modifiers ships with dream-trainer; importing
`dream_trainer.utils.modifiers` (as the CLI does) registers them
automatically.

::: dream_trainer.utils.modifiers.base

::: dream_trainer.utils.modifiers.grad_accum
