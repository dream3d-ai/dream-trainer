"""Gradient-accumulation modifier."""

from dream_trainer import DreamTrainerConfig

from .register import register_modifier


@register_modifier()
def grad_accum_steps(config: DreamTrainerConfig, n: int):
    """Set the number of gradient-accumulation steps per optimizer update."""
    config.training_parameters.gradient_accumulation_steps = n
