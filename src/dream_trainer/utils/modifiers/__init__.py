"""Config-modifier system used by the dream-trainer CLI.

Importing this module has the side effect of registering all generic
modifiers (`base`, `grad_accum`) into the global
[MODIFIERS][dream_trainer.utils.modifiers.MODIFIERS] registry. Downstream
projects should import this module before defining their own
`@register_modifier`-decorated functions so the ordering is deterministic.
"""

from . import base, grad_accum
from .register import MODIFIERS, ModifierRegistry, register_modifier

__all__ = [
    "base",
    "grad_accum",
    "MODIFIERS",
    "ModifierRegistry",
    "register_modifier",
]
