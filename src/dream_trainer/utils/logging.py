from dataclasses import fields
from typing import Any, Callable

default_dataclass_filter = lambda name, value: not name.startswith("_")


def config_to_dict(
    config: Any, *, filt: Callable[[str, Any], bool] = default_dataclass_filter
) -> dict[str, Any]:
    """Modified dataclass asdict that filters expensive items before iterating"""
    if not hasattr(type(config), "__dataclass_fields__"):
        raise TypeError("asdict() should be called on dataclass instances")
    return _asdict_inner(config, filt)


def _asdict_inner(
    obj: Any, filt: Callable[[str, Any], bool] = default_dataclass_filter
) -> dict[str, Any]:
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
    elif isinstance(obj, list):
        return [_asdict_inner(v, filt) for v in obj]  # type: ignore
    elif isinstance(obj, tuple):
        return tuple(_asdict_inner(v, filt) for v in obj)  # type: ignore
    elif isinstance(obj, dict):
        return dict((_asdict_inner(k, filt), _asdict_inner(v, filt)) for k, v in obj.items())  # type: ignore
    else:
        return str(obj)  # type: ignore
