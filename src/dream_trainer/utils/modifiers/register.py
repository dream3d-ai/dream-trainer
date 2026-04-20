"""Registry for CLI config modifiers.

Modifiers are small functions that mutate a `DreamTrainerConfig` in place,
typically to flip a debugging flag, disable a callback, or tweak a knob for
a quick experiment. They are exposed as CLI flags by
[dream_trainer.utils.cli][] so that users can run e.g.
`python train.py --no-compile --no-log` without editing source.

The registry keys on trainer config classes so that subclasses inherit the
modifiers of their parents. Call `register_modifier` (as a decorator) in any
module that should contribute modifiers; importing that module is enough to
make the new flag appear on the CLI.
"""

import inspect
from typing import Callable, TypeVar

from dream_trainer import DreamTrainerConfig
from dream_trainer.trainer.abstract import AbstractTrainerConfig

T = TypeVar("T", bound=AbstractTrainerConfig)


class ModifierRegistry:
    """Class-scoped registry of modifier functions.

    Each entry maps a trainer config class to a dict of
    `{name: (fn, value_type, z_index)}`. Lookup via `registry[TrainerCls]`
    flattens all entries whose key is an ancestor of `TrainerCls`, so a
    subclass automatically inherits every modifier registered against its
    bases.

    The flattened result is sorted by `z_index` (ascending), giving
    registrants control over the order in which modifiers are applied when
    several are passed together on the CLI.
    """

    def __init__(self):
        self._registry: dict[type, dict[str, tuple[Callable[..., None], type, int]]] = {}

    def __getitem__(self, trainer_type: type[T]) -> dict[str, tuple[Callable[..., None], type]]:
        """Return modifiers applicable to `trainer_type`.

        Args:
            trainer_type: The concrete trainer config class being launched.

        Returns:
            An ordered mapping of modifier name to `(fn, value_type)` pairs.
            Entries are sorted by the registrant-provided `z_index`. The
            `z_index` is dropped from the returned tuple because callers
            only care about resolution order, not the key itself.
        """
        items = {
            name: modifier
            for trainer_type_, modifier_dict in self._registry.items()
            for name, modifier in modifier_dict.items()
            if issubclass(trainer_type, trainer_type_)
        }

        return {
            name: (fn, type_)
            for name, (fn, type_, _) in sorted(items.items(), key=lambda item: item[1][2])
        }

    def __setitem__(
        self,
        trainer_type: type[T],
        value: dict[
            str, tuple[Callable[..., None], type, int] | tuple[Callable[..., None], type]
        ],
    ):
        """Replace all modifiers for `trainer_type`.

        Args:
            trainer_type: The trainer config class to associate modifiers with.
            value: A mapping of modifier name to either
                `(fn, value_type, z_index)` or `(fn, value_type)` (in which
                case `z_index` defaults to `0`).
        """
        self._registry[trainer_type] = {
            name: modifier if len(modifier) == 3 else (modifier[0], modifier[1], 0)
            for name, modifier in value.items()
        }

    def items(self):
        """Iterate over `(trainer_type, modifier_dict)` pairs in registration order."""
        return self._registry.items()

    def setdefault(
        self,
        trainer_type: type[T],
        default: dict[
            str, tuple[Callable[..., None], type, int] | tuple[Callable[..., None], type]
        ],
    ):
        """Get or insert the modifier dict for `trainer_type`.

        Equivalent to `dict.setdefault`, but normalises the inserted value so
        every entry is a 3-tuple `(fn, value_type, z_index)` with a default
        `z_index` of `0`.

        Args:
            trainer_type: The trainer config class to look up.
            default: The mapping to insert if `trainer_type` is not yet
                registered.

        Returns:
            The modifier dict for `trainer_type` (either existing or newly
            inserted).
        """
        return self._registry.setdefault(
            trainer_type,
            {
                name: modifier if len(modifier) == 3 else (modifier[0], modifier[1], 0)
                for name, modifier in default.items()
            },
        )

    def keys(self) -> list[type]:
        """Return every trainer config class with at least one registered modifier."""
        return list(self._registry.keys())

    def values(self) -> list[dict[str, tuple[Callable[..., None], type, int]]]:
        """Return every registered modifier dict, one per trainer class."""
        return list(self._registry.values())

    def __repr__(self):
        return f"ModifierRegistry({self._registry})"


MODIFIERS = ModifierRegistry()
"""Global modifier registry consulted by [dream_trainer.utils.cli.cli][]."""


def register_modifier(
    name: str | None = None,
    trainer: type[T] | list[type[T]] = DreamTrainerConfig,
    z_index: int = 0,
):
    """Register a modifier function with `MODIFIERS`.

    Use this decorator to expose a config-mutation as a CLI flag. The
    decorated function must accept a trainer config as its first positional
    argument and, optionally, a single additional argument whose annotation
    determines the type of the CLI option.

    Signature rules:

    - `fn(config)` -> the flag is a boolean (`--my-flag`, no value).
    - `fn(config, value: T)` -> the flag takes a value of type `T`
      (`--my-flag 42`, `--my-flag path/to/thing`).

    Anything else raises `ValueError`.

    Args:
        name: The CLI-visible name for this modifier. Defaults to
            `fn.__name__`. The name is converted to kebab-case when the
            corresponding `--my-flag` option is generated.
        trainer: The trainer config class (or list of classes) this
            modifier applies to. The modifier is available on any subclass
            of the provided class(es). Defaults to `DreamTrainerConfig`,
            which makes the modifier universally available.
        z_index: Ordering hint. When multiple modifiers are applied at
            once, they are invoked in ascending `z_index` order. Use this
            to ensure, for example, that a modifier that sets batch size
            runs before one that disables dataloader workers.

    Returns:
        A decorator that registers the wrapped function and returns it
        unchanged.

    Example:
        ```python
        from dream_trainer import DreamTrainerConfig
        from dream_trainer.utils.modifiers import register_modifier

        @register_modifier()
        def no_compile(config: DreamTrainerConfig):
            '''Disable torch.compile.'''
            config.device_parameters.compile_model = False

        @register_modifier()
        def batch_size(config: DreamTrainerConfig, n: int):
            '''Override the train/val batch size.'''
            config.train_dataloader.batch_size = n
        ```
    """
    trainers = trainer if isinstance(trainer, list) else [trainer]

    def wrapper(fn: Callable[..., None]) -> Callable[..., None]:
        signature = inspect.signature(fn)
        parameters = list(signature.parameters.values())

        if len(parameters) == 1:
            type_ = bool
        elif len(parameters) == 2:
            type_ = parameters[1].annotation
        else:
            raise ValueError(
                "Modifier functions must take at most two arguments: the trainer config and optionally a value"
            )

        modifier_name = fn.__name__ if name is None else name
        for t in trainers:
            MODIFIERS.setdefault(t, {})[modifier_name] = (fn, type_, z_index)
        return fn

    return wrapper
