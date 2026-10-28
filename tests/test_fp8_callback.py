from types import SimpleNamespace

import torch.nn as nn
from torchao.float8.float8_linear import Float8Linear

from dream_trainer.callbacks.fp8 import callback as fp8_callback
from dream_trainer.callbacks.fp8.callback import Fp8Quantization


class _Optimizer:
    def __init__(self):
        self.hooks = []

    def register_step_post_hook(self, hook):
        self.hooks.append(hook)


def _trainer(models: dict[str, nn.Module]) -> SimpleNamespace:
    trainer = SimpleNamespace(
        **models,
        named_models=lambda: {name: getattr(trainer, name) for name in models},
        world=SimpleNamespace(fsdp_config=None),
        _quantized_models=[],
        optimizers={name: _Optimizer() for name in models},
    )
    trainer.get_optimizers_by_model = lambda name: [trainer.optimizers[name]]
    return trainer


def test_tensorwise_training_recipe_converts_and_binds_hooks_per_model(monkeypatch):
    # Big enough to pass AutoFilterForTensorwise; meta so no memory is allocated.
    models = {
        name: nn.Sequential(nn.Linear(4096, 2048, device="meta")) for name in ("gen", "disc")
    }
    trainer = _trainer(models)
    callback = Fp8Quantization({"gen": "tensorwise", "disc": "tensorwise"})
    callback.trainer = trainer  # type: ignore[assignment]

    callback.post_configure()

    assert trainer._quantized_models == ["gen", "disc"]
    assert isinstance(trainer.gen[0], Float8Linear)
    assert isinstance(trainer.disc[0], Float8Linear)

    seen = []
    monkeypatch.setattr(fp8_callback, "precompute_float8_dynamic_scale_for_fsdp", seen.append)
    callback.post_setup()

    for name in ("gen", "disc"):
        (hook,) = trainer.optimizers[name].hooks
        hook(trainer.optimizers[name], (), {})
    assert seen == [trainer.gen, trainer.disc]
