"""Fresh-process native compiler-cache correctness fixture."""

import json
from pathlib import Path
from types import SimpleNamespace

import torch
import typer
from torch._dynamo.utils import counters

from dream_trainer.callbacks import CompileCacheCallback


def main(directory: Path, output: Path, width: int = 8, scale: float = 1.0):
    torch.set_num_threads(1)
    torch.manual_seed(123)
    model = torch.nn.Linear(width, width)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    callback = CompileCacheCallback(directory)
    callback.trainer = SimpleNamespace(
        device_parameters=SimpleNamespace(compile_model=True),
        world=SimpleNamespace(device=torch.device("cpu"), world_mesh=None),
        is_accumulating_gradients=False,
    )
    before_rng = torch.get_rng_state().clone()
    before = {key: value.clone() for key, value in model.state_dict().items()}
    callback.pre_configure()
    torch.testing.assert_close(model.state_dict(), before, rtol=0, atol=0)
    assert torch.equal(before_rng, torch.get_rng_state())
    assert not optimizer.state

    @torch.compile(fullgraph=True)
    def objective(x):
        return (model(x).sin() * scale).square().mean()

    x = torch.randn(4, width)
    loss = objective(x)
    expected = (model(x).sin() * scale).square().mean()
    torch.testing.assert_close(loss, expected)
    loss.backward()
    gradients = [parameter.grad.clone() for parameter in model.parameters()]
    optimizer.step()
    optimizer.zero_grad()
    before_rng = torch.get_rng_state().clone()
    callback.post_train_step({}, 0)
    assert torch.equal(before_rng, torch.get_rng_state())
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "gradients": gradients,
            "loss": loss.detach(),
        },
        output,
    )
    output.with_suffix(".json").write_text(
        json.dumps({name: dict(values) for name, values in counters.items()}, indent=2)
    )


if __name__ == "__main__":
    typer.run(main)
