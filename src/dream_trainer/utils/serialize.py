import copy
import importlib
from dataclasses import fields
from pathlib import Path
from typing import Any, Callable

default_dataclass_filter = lambda name, value: not name.startswith("_")


def asdict(
    obj: Any, *, filt: Callable[[str, Any], bool] = default_dataclass_filter
) -> dict[str, Any]:
    """Modified dataclass asdict that filters expensive items before iterating"""
    if not hasattr(type(obj), "__dataclass_fields__"):
        raise TypeError("asdict() should be called on dataclass instances")
    return _asdict_inner(obj, filt)  # type: ignore


def _asdict_inner(obj: Any, filt: Callable[[str, Any], bool] = default_dataclass_filter):
    if hasattr(type(obj), "__dataclass_fields__"):
        _obj = {
            "__cls__": f"{obj.__module__}.{obj.__class__.__name__}",
            **{
                f.name: _asdict_inner(getattr(obj, f.name), filt)
                for f in fields(obj)
                if filt(f.name, getattr(obj, f.name))
            },
        }
        return _obj
    elif isinstance(obj, (list, tuple)):
        return type(obj)(_asdict_inner(v, filt) for v in obj)
    elif isinstance(obj, dict):
        return type(obj)(
            (_asdict_inner(k, filt), _asdict_inner(v, filt)) for k, v in obj.items()
        )
    else:
        return copy.deepcopy(obj)


def fromdict(obj: dict[str, Any] | Any):
    """Convert a dictionary to a dataclass"""
    if isinstance(obj, (list, tuple, set)):
        return type(obj)(fromdict(o) for o in obj)
    if not isinstance(obj, dict):
        return obj
    if (_cls := obj.pop("__cls__", None)) is not None:
        cls = getattr(
            importlib.import_module(".".join(_cls.split(".")[:-1])), _cls.split(".")[-1]
        )
        return cls(**{k: fromdict(v) for k, v in obj.items()})

    return {k: fromdict(v) for k, v in obj.items()}


def get_path_fields(cls):
    """Find all fields that are of type Path"""
    return [name for name, type_ in cls.__annotations__.items() if type_ is Path]
