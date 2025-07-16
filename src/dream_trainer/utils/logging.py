"""Logging utilities for converting dataclasses to dictionaries with filtering support.

This module provides functionality to convert dataclass instances into dictionaries
while applying filters to exclude certain fields. This is particularly useful for
logging configurations where some fields may contain expensive-to-serialize objects
or private data that shouldn't be logged.
"""

from dataclasses import fields
from typing import Any, Callable

default_dataclass_filter = lambda name, value: not name.startswith("_")


def config_to_dict(
    config: Any, *, filt: Callable[[str, Any], bool] = default_dataclass_filter
) -> dict[str, Any]:
    """Convert a dataclass instance to a dictionary with optional field filtering.

    This function provides a modified version of dataclasses.asdict() that allows
    filtering out specific fields before conversion. This is useful when certain
    fields contain expensive-to-serialize objects or sensitive data.

    Args:
        config: A dataclass instance to convert to a dictionary.
        filt: A callable that takes a field name and value, returning True if the
            field should be included in the output dictionary. Defaults to excluding
            fields whose names start with an underscore.

    Returns:
        A dictionary representation of the dataclass with filtered fields. The
        dictionary includes a special "__cls__" key containing the fully qualified
        class name.

    Raises:
        TypeError: If config is not a dataclass instance.

    Example:
        >>> @dataclass
        ... class Config:
        ...     name: str = "example"
        ...     _private: str = "hidden"
        ...     value: int = 42
        >>>
        >>> result = config_to_dict(Config())
        >>> # result will contain 'name' and 'value' but not '_private'
    """
    if not hasattr(type(config), "__dataclass_fields__"):
        raise TypeError("asdict() should be called on dataclass instances")
    return _asdict_inner(config, filt)


def _asdict_inner(
    obj: Any, filt: Callable[[str, Any], bool] = default_dataclass_filter
) -> dict[str, Any]:
    """Recursively convert objects to dictionary representations.

    This helper function handles the recursive conversion of various object types:
    - Dataclass instances are converted to dictionaries with field filtering
    - Lists and tuples are converted element-wise
    - Dictionaries are converted with both keys and values processed
    - All other objects are converted to strings

    Args:
        obj: The object to convert. Can be a dataclass instance, list, tuple,
            dict, or any other type.
        filt: A callable that takes a field name and value, returning True if the
            field should be included when converting dataclass instances.

    Returns:
        A dictionary, list, tuple, or string representation of the input object.
        Dataclass instances will have a "__cls__" key with their fully qualified
        class name.

    Note:
        Non-dataclass objects are converted to strings to ensure serializability.
        This means complex objects like model instances will be represented as
        their string representation.
    """
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
