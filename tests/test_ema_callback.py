from contextlib import contextmanager
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Shard
from torch.optim.swa_utils import get_ema_multi_avg_fn

from dream_trainer.callbacks.ema.averaged_model import EMA
from dream_trainer.callbacks.ema.callback import EMACallback


@contextmanager
def single_rank_cpu_mesh(tmp_path, monkeypatch):
    monkeypatch.setenv("GLOO_SOCKET_IFNAME", "lo")
    init_method = f"file://{tmp_path / 'dist_init'}"
    dist.init_process_group("gloo", rank=0, world_size=1, init_method=init_method)
    try:
        yield init_device_mesh("cpu", (1,), mesh_dim_names=("dp",))
    finally:
        dist.destroy_process_group()


def sharded_dtensor(data: torch.Tensor, mesh) -> DTensor:
    return DTensor.from_local(data, mesh, [Shard(0)], run_check=False)


class DTensorModule(nn.Module):
    def __init__(self, mesh):
        super().__init__()
        self.weight = nn.Parameter(
            sharded_dtensor(
                torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32),
                mesh,
            )
        )


def test_ema_tracks_dtensor_parameters_with_multi_avg_fn(tmp_path, monkeypatch):
    with single_rank_cpu_mesh(tmp_path, monkeypatch) as mesh:
        model = DTensorModule(mesh)
        ema = EMA(model, multi_avg_fn=get_ema_multi_avg_fn(0.5))
        ema.initialize_from_model(model, n_averaged=1)

        assert isinstance(ema.parameters["weight"], DTensor)

        with torch.no_grad():
            model.weight.copy_(
                sharded_dtensor(
                    torch.tensor([[3.0, 4.0], [5.0, 6.0]], dtype=torch.float32),
                    mesh,
                )
            )

        ema.update_parameters(model)

        assert isinstance(ema.state_dict()["parameters"]["weight"], DTensor)
        torch.testing.assert_close(
            ema.parameters["weight"].to_local(),
            torch.tensor([[2.0, 3.0], [4.0, 5.0]], dtype=torch.float32),
        )

        with torch.no_grad():
            model.weight.copy_(sharded_dtensor(torch.zeros(2, 2), mesh))
        ema.copy_to(model)
        torch.testing.assert_close(
            model.weight.to_local(),
            torch.tensor([[2.0, 3.0], [4.0, 5.0]], dtype=torch.float32),
        )


def test_ema_cpu_offload_preserves_dtensor_state(tmp_path, monkeypatch):
    with single_rank_cpu_mesh(tmp_path, monkeypatch) as mesh:
        model = DTensorModule(mesh)
        ema = EMA(model, multi_avg_fn=get_ema_multi_avg_fn(0.5), device=torch.device("cpu"))
        ema.initialize_from_model(model, n_averaged=1)

        assert isinstance(ema.parameters["weight"], DTensor)
        assert ema.parameters["weight"].to_local().device.type == "cpu"

        with torch.no_grad():
            model.weight.copy_(
                sharded_dtensor(
                    torch.tensor([[3.0, 4.0], [5.0, 6.0]], dtype=torch.float32),
                    mesh,
                )
            )

        ema.update_parameters(model)

        assert isinstance(ema.state_dict()["parameters"]["weight"], DTensor)
        assert ema.state_dict()["parameters"]["weight"].to_local().device.type == "cpu"
        torch.testing.assert_close(
            ema.parameters["weight"].to_local(),
            torch.tensor([[2.0, 3.0], [4.0, 5.0]], dtype=torch.float32),
        )


def test_ema_updates_from_post_optimizer_step_for_matching_model_only():
    model = nn.Linear(1, 1, bias=False)
    other_model = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.0)
        other_model.weight.fill_(10.0)

    callback = EMACallback(["model", "other_model"], decay=0.5)
    callback.trainer = SimpleNamespace(
        global_step=0,
        named_models=lambda: {"model": model, "other_model": other_model},
        get_model=lambda name: {"model": model, "other_model": other_model}[name],
    )
    callback.post_setup()

    with torch.no_grad():
        model.weight.fill_(3.0)
        other_model.weight.fill_(30.0)

    callback.post_optimizer_step(model, object())

    torch.testing.assert_close(
        callback.ema_models["model"].parameters["weight"],
        torch.full_like(model.weight, 2.0),
    )
    torch.testing.assert_close(
        callback.ema_models["other_model"].parameters["weight"],
        torch.full_like(other_model.weight, 10.0),
    )


def test_ema_reinitializes_from_model_after_checkpoint_without_ema_state():
    model = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.0)

    callback = EMACallback(["model"], decay=0.5)
    callback.trainer = SimpleNamespace(
        named_models=lambda: {"model": model},
        get_model=lambda name: {"model": model}[name],
    )
    callback.post_setup()

    with torch.no_grad():
        model.weight.fill_(7.0)

    callback.post_load_state_dict(set())

    torch.testing.assert_close(
        callback.ema_models["model"].parameters["weight"],
        torch.full_like(model.weight, 7.0),
    )
    torch.testing.assert_close(
        callback.ema_models["model"].n_averaged,
        torch.ones_like(callback.ema_models["model"].n_averaged),
    )
