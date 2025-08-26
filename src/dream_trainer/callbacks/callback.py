import contextlib
from typing import Any, Generic, Iterable, TypeVar, get_origin

import torch.nn as nn
from torch.distributed.checkpoint.stateful import Stateful
from torch.optim import Optimizer

from dream_trainer.trainer.abstract import AbstractTrainer

_AbstractTrainer = TypeVar("_AbstractTrainer", bound=AbstractTrainer)


class Callback(Generic[_AbstractTrainer]):
    """
    A base class for any hooks that need to be called during the training loop.

    When subclassing, instantiate the generic with the mixins you need. If the generic is not
    instantiated, the callback will default to AbstractTrainer.

    Example:
        class MyCallback(Callback[LoggerMixin]):
            pass

    """

    _dependency: type[_AbstractTrainer] | tuple[type[_AbstractTrainer], ...] | Any
    trainer: _AbstractTrainer

    def __init_subclass__(cls) -> None:
        """
        Extracts the dependency defined in the generic Callback signature and defaults to
        AbstractTrainer whenever no dependency is defined.
        """
        super().__init_subclass__()
        if "_dependency" in cls.__dict__:
            # User defined dependencies
            return

        # A tuple of the inherited classes (before removing typing)
        orig_bases = getattr(cls, "__orig_bases__", [])
        if len(orig_bases) != 1:
            raise NotImplementedError(
                "We currently do not support callbacks with multiple generic inheritance."
            )

        base = orig_bases[0]
        origin = get_origin(base)
        assert issubclass(origin, Callback) or (
            issubclass(origin, Generic) and base.__args__[0] is _AbstractTrainer
        ), f"{cls} does not inherit from Callback"

        _abstract_trainer = base.__args__[0]
        if isinstance(_abstract_trainer, TypeVar):
            assert _abstract_trainer is _AbstractTrainer
            _abstract_trainer = AbstractTrainer  # Generic wasn't substituted

        cls._dependency = _abstract_trainer  # type: ignore

    def pre_launch(self):
        """
        Called before the launch of the distributed world.
        """
        raise NotImplementedError("This function should never be called")

    def pre_configure(self):
        """
        Called before the model, optimizers, schedulers and dataloaders are configured.
        """
        raise NotImplementedError("This function should never be called")

    def post_configure(self):
        """
        Called after the model, optimizers, schedulers and dataloaders are configured.
        """
        raise NotImplementedError("This function should never be called")

    def pre_setup(self):
        """
        Called before the model, optimizers, schedulers and dataloaders are setup.
        """
        raise NotImplementedError("This function should never be called")

    def post_setup(self):
        """
        Called after the model, optimizers, schedulers and dataloaders are setup.
        """
        raise NotImplementedError("This function should never be called")

    def pre_fit(self):
        """
        Called when the fit process starts.

        Args:
            trainer (FSDP2Trainer): The trainer instance.
        """
        raise NotImplementedError("This function should never be called")

    def post_fit(self):
        """
        Called when the fit process ends.

        Args:
            trainer (FSDP2Trainer): The trainer instance.
        """
        raise NotImplementedError("This function should never be called")

    def pre_epoch(self):
        """
        Called at the beginning of each epoch.

        Args:
            trainer (FSDP2Trainer): The trainer instance.
        """
        raise NotImplementedError("This function should never be called")

    def post_epoch(self):
        """
        Called at the end of each epoch.

        Args:
            trainer (FSDP2Trainer): The trainer instance.
        """
        raise NotImplementedError("This function should never be called")

    def pre_train_epoch(self):
        """
        Called at the beginning of each training epoch.

        Args:
            trainer (FSDP2Trainer): The trainer instance.
        """
        raise NotImplementedError("This function should never be called")

    def post_train_epoch(self, result: dict[str, Any]):
        """
        Called at the end of each training epoch.

        Args:
            trainer (FSDP2Trainer): The trainer instance.
            result (dict[str, Any]): The result of the training epoch.
        """
        raise NotImplementedError("This function should never be called")

    def pre_train_step(self, batch: dict[str, Any], batch_idx: int):
        """
        Called at the start of each training step.

        Args:
            trainer (FSDP2Trainer): The trainer instance.
            batch (dict[str, Any]): The current batch.
            batch_idx (int): The index of the current batch.
        """
        raise NotImplementedError("This function should never be called")

    def post_train_step(self, result: dict[str, Any], batch_idx: int):
        """
        Called at the end of each training step.

        Args:
            trainer (FSDP2Trainer): The trainer instance.
            result (Any): The result of the training step.
            batch_idx (int): The index of the current batch.
        """
        raise NotImplementedError("This function should never be called")

    def pre_optimizer_step(self, model: nn.Module, optimizer: Optimizer):
        """
        Called at the before each optimizer step.
        """
        raise NotImplementedError("This function should never be called")

    def post_optimizer_step(self, model: nn.Module, optimizer: Optimizer):
        """
        Called at the end of each optimizer step.
        """
        raise NotImplementedError("This function should never be called")

    def pre_optimizer_zero_grad(self, model: nn.Module, optimizer: Optimizer):
        """
        Called at the before each optimizer zero grad.
        """
        raise NotImplementedError("This function should never be called")

    def post_optimizer_zero_grad(self, model: nn.Module, optimizer: Optimizer):
        """
        Called at the end of each optimizer zero grad.
        """
        raise NotImplementedError("This function should never be called")

    def pre_validation_epoch(self):
        """
        Called at the beginning of each validation epoch.

        Args:
            trainer (FSDP2Trainer): The trainer instance.
        """
        raise NotImplementedError("This function should never be called")

    def post_validation_epoch(self, result: dict[str, Any]):
        """
        Called at the end of each validation epoch.

        Args:
            trainer (FSDP2Trainer): The trainer instance.
            result (Any): The result of the validation epoch.
        """
        raise NotImplementedError("This function should never be called")

    def pre_validation_step(self, batch: dict[str, Any], batch_idx: int):
        """
        Called at the start of each validation step.

        Args:
            trainer (FSDP2Trainer): The trainer instance.
            batch_idx (int): The index of the current batch.
        """
        raise NotImplementedError("This function should never be called")

    def post_validation_step(self, result: dict[str, Any], batch_idx: int):
        """
        Called at the end of each validation step.

        Args:
            trainer (FSDP2Trainer): The trainer instance.
            result (Any): The result of the validation step.
            batch_idx (int): The index of the current batch.
        """
        raise NotImplementedError("This function should never be called")

    def on_interrupt(self):
        """
        Called when the program is interrupted or crashes.
        """
        raise NotImplementedError("This function should never be called")

    def train_context(self) -> contextlib._GeneratorContextManager[None]:
        """
        Training steps happen under this context manager.
        """
        raise NotImplementedError("This function should never be called")

    def validation_context(self) -> contextlib._GeneratorContextManager[None]:
        """
        Validation steps happen under this context manager.
        """
        raise NotImplementedError("This function should never be called")

    def state_dict(self) -> dict[str, Any]:
        return {}

    def load_state_dict(self, state_dict: dict[str, Any]):
        pass

    def pop_self(self):
        """
        Remove the callback from the trainer.callbacks list.
        """
        assert hasattr(self.trainer, "callbacks") and isinstance(
            getattr(self.trainer, "callbacks"), CallbackCollection
        ), f"Cannot find CallbackCollection in Trainer {self.trainer.__class__.__name__}"

        getattr(self.trainer, "callbacks").pop(self.__class__.__name__)


class RankZeroCallback(Callback[_AbstractTrainer]):
    """
    Equivalent to the base Callback, but ensures all functions are only called
    on the rank zero process.
    """

    pass


class CallbackCollection(dict[str, Callback | RankZeroCallback], Stateful):
    def __init__(
        self,
        callbacks: Iterable[Callback | RankZeroCallback]
        | dict[str, Callback | RankZeroCallback]
        | Iterable[tuple[str, Callback | RankZeroCallback]] = [],
    ):
        # Get all methods defined in Callback above
        self.trigger_names = {m for m in dir(Callback) if not m.startswith("_")}

        # Allow callback initialization to match dict initialization
        if not isinstance(callbacks, dict):
            callbacks = [
                callback
                if isinstance(callback, tuple)
                else (callback.__class__.__name__, callback)
                for callback in callbacks
            ]

        super().__init__(callbacks)  # type: ignore
        self.refresh()

    def initialize(self, trainer: AbstractTrainer):
        self.trainer = trainer

        for callback in self.values():
            deps = deps if isinstance(deps := callback._dependency, tuple) else (deps,)
            assert all(isinstance(trainer, dependency) for dependency in deps), (
                f"{callback.__class__.__name__} requires dependencies {deps} but got {trainer.__class__.__name__}"
            )

            callback.trainer = trainer

    def __iter__(self):
        return iter(self.values())

    def state_dict(self):
        return {
            name: getattr(callback, "state_dict")()
            for name, callback in self.items()
            if hasattr(callback, "state_dict")
        }

    def load_state_dict(self, state_dicts: dict[str, Any]):
        for name, state_dict in state_dicts.items():
            self[name].load_state_dict(state_dict)

    def refresh(self, init: bool = False):
        """
        Organizes callbacks by existing methods for efficient execution.
        """
        self.triggers = {
            trigger: [
                callback
                for callback in self.values()
                # Only add triggers that were overridden in the callback
                if not getattr(callback, trigger).__qualname__.startswith("Callback.")
            ]
            for trigger in self.trigger_names
        }

        if init and hasattr(self, "trainer"):
            # When a new callback is added and we've already initialized, ensure the new
            # callback's dependencies are satisfied and its trainer is populated
            self.initialize(self.trainer)

    def _call_callbacks(self, trigger: str):
        def call(*args, **kwargs):
            output = (
                getattr(callback, trigger)(*args, **kwargs)
                for callback in self.triggers[trigger]
                if not (
                    isinstance(callback, RankZeroCallback)
                    and not callback.trainer.world.is_global_zero
                )
            )
            return [out for out in output if out is not None]

        return call

    def append(self, callback: "Callback | RankZeroCallback"):
        self[callback.__class__.__name__] = callback
        self.refresh(init=True)

    def __getattr__(self, name: str):
        if name in self.trigger_names:
            return self._call_callbacks(name)

        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    #########################
    # Override dict methods #
    #########################
    def __deepcopy__(self, memo: dict[int, Any]):
        from copy import deepcopy

        return CallbackCollection(deepcopy(tuple(self.values()), memo))

    def __repr__(self):
        return f"CallbackCollection({list(self.keys())})"

    def __setitem__(self, key: str, value: Callback | RankZeroCallback):
        super().__setitem__(key, value)
        self.refresh(init=True)

    def __delitem__(self, key: str):
        super().__delitem__(key)
        self.refresh()

    def update(self, **kwargs: Callback | RankZeroCallback):
        super().update(**kwargs)
        self.refresh(init=True)

    def clear(self):
        super().clear()
        self.refresh()

    def pop(self, key: str, default: Callback | RankZeroCallback | None = None):
        value = super().pop(key, default)
        self.refresh()
        return value

    def popitem(self) -> tuple[str, Callback | RankZeroCallback]:
        item = super().popitem()
        self.refresh()
        return item

    def setdefault(self, key: str, default: Callback | RankZeroCallback):
        value = super().setdefault(key, default)
        self.refresh(init=True)
        return value
