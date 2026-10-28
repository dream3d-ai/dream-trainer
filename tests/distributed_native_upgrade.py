"""Two-rank native backend, optimizer failure, and checkpoint acceptance.

Run with torchrun --standalone --nproc-per-node=2 and --backend nccl or nccl2.
"""

import json
import os
from copy import deepcopy
from pathlib import Path

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import typer
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    parallelize_module,
)

from dream_trainer.configs import Comm, DeviceParameters
from dream_trainer.trainer.world.distributed_world import DistributedWorld
from dream_trainer.utils.dist.core import (
    _process_group_cache,
    create_global_process_group,
    create_local_process_group,
)
from test_optimizer_validation import StepTrainer


def local(tensor):
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


def full_state(model):
    return {
        name: (value.full_tensor() if isinstance(value, DTensor) else value).detach().clone()
        for name, value in model.state_dict().items()
    }


def main(output: Path, backend: str = "nccl", mode: str = "fsdp", bf16: bool = False):
    config = DeviceParameters(
        comm=Comm(backend=backend, init_timeout_seconds=60, train_timeout_seconds=30),
        compile_model=False,
        async_tensor_parallel=False,
        _dp_shard=2 if mode == "fsdp" else 1,
        _dp_replicate=2 if mode == "ddp" else 1,
        _tensor_parallel=2 if mode == "tp" else 1,
    )
    world = DistributedWorld(config)
    world.launch()
    rank = dist.get_rank()
    try:
        for name in world.world_mesh.mesh_dim_names:
            actual = dist.get_backend(world.world_mesh[name].get_group())
            assert backend in actual.split(",") or f"cuda:{backend}" in actual.split(","), (
                name,
                actual,
                backend,
            )
        for mesh in world.world_mesh._flatten_mapping.values():
            actual = dist.get_backend(mesh.get_group())
            assert backend in actual.split(",") or f"cuda:{backend}" in actual.split(","), (
                mesh.mesh_dim_names,
                actual,
                backend,
            )
        assert create_local_process_group(backend) is create_local_process_group(backend)
        assert dist.get_backend(create_global_process_group(backend)) == backend
        cpu_group = dist.new_group(backend="gloo")
        torch.manual_seed(123)
        model = torch.nn.Sequential(
            torch.nn.Linear(16, 32), torch.nn.GELU(), torch.nn.Linear(32, 8)
        ).to(world.device)
        if mode == "fsdp":
            for layer in (model[0], model[2], model):
                fully_shard(
                    layer,
                    mesh=world.get_mesh("dp_shard"),
                    mp_policy=MixedPrecisionPolicy(
                        param_dtype=torch.bfloat16 if bf16 else torch.float32,
                        reduce_dtype=torch.float32,
                    ),
                )
        elif mode == "ddp":
            from torch.distributed._composable.replicate import replicate

            replicate(model, device_mesh=world.get_mesh("dp_replicate"))
        elif mode == "tp":
            parallelize_module(
                model,
                world.get_mesh("tp"),
                {
                    "0": ColwiseParallel(),
                    "2": RowwiseParallel(),
                },
            )
        else:
            raise ValueError(mode)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9)
        trainer = StepTrainer(model, optimizer, scheduler, world)
        inputs = torch.randn(4, 16, device=world.device)
        losses = []
        for _ in range(2):
            model(inputs).float().square().mean().backward()
            trainer.step(optimizer)
        state = get_state_dict(model, optimizer)
        dcp.save(
            {"model": state[0], "optimizer": state[1]},
            checkpoint_id=str(output / f"{mode}-{bf16}-checkpoint"),
            process_group=cpu_group,
        )
        saved_scheduler = scheduler.state_dict()
        for _ in range(2):
            loss = model(inputs).float().square().mean()
            losses.append(float(loss.detach()))
            loss.backward()
            trainer.step(optimizer)
        expected = full_state(model)
        expected_optimizer = deepcopy(get_state_dict(model, optimizer)[1])
        state = get_state_dict(model, optimizer)
        dcp.load(
            {"model": state[0], "optimizer": state[1]},
            checkpoint_id=str(output / f"{mode}-{bf16}-checkpoint"),
            process_group=cpu_group,
        )
        set_state_dict(model, optimizer, model_state_dict=state[0], optim_state_dict=state[1])
        scheduler.load_state_dict(saved_scheduler)
        for expected_loss in losses:
            loss = model(inputs).float().square().mean()
            assert float(loss.detach()) == expected_loss
            loss.backward()
            trainer.step(optimizer)
        torch.testing.assert_close(full_state(model), expected, rtol=0, atol=0)
        actual_leaves, actual_spec = torch.utils._pytree.tree_flatten(
            get_state_dict(model, optimizer)[1]
        )
        expected_leaves, expected_spec = torch.utils._pytree.tree_flatten(expected_optimizer)
        assert actual_spec == expected_spec
        for actual, expected in zip(actual_leaves, expected_leaves):
            if isinstance(actual, torch.Tensor):
                torch.testing.assert_close(local(actual), local(expected), rtol=0, atol=0)
            else:
                assert actual == expected
        failures = []
        for target in ("parameter", "gradient", "missing", "norm"):
            loss = model(inputs).float().square().mean()
            loss.backward()
            parameter = next(model.parameters())
            original = local(parameter).detach().clone()
            if rank == 1:
                if target == "parameter":
                    with torch.no_grad():
                        local(parameter).flatten()[0] = float("nan")
                elif target == "gradient":
                    local(parameter.grad).flatten()[0] = float("inf")
                elif target == "missing":
                    optimizer.zero_grad()
                else:
                    local(parameter.grad).fill_(1e30)
            before = {
                name: local(value).detach().clone() for name, value in model.named_parameters()
            }
            trainer.events.clear()
            try:
                trainer.step(optimizer)
            except RuntimeError as error:
                failures.append(str(error))
            else:
                raise AssertionError(f"rank {rank} accepted invalid {target}")
            assert not trainer.events
            for name, value in model.named_parameters():
                torch.testing.assert_close(
                    local(value), before[name], rtol=0, atol=0, equal_nan=True
                )
            if target == "parameter":
                with torch.no_grad():
                    local(parameter).copy_(original)
            optimizer.zero_grad()
        output.mkdir(parents=True, exist_ok=True)
        (output / f"{mode}-{bf16}-rank-{rank}.json").write_text(
            json.dumps(
                {
                    "backend": backend,
                    "mode": mode,
                    "bf16": bf16,
                    "losses": losses,
                    "failures": failures,
                    "resume": "exact",
                    "pid": os.getpid(),
                },
                indent=2,
            )
        )
        print(f"PASS {backend} {mode} bf16={bf16} rank={rank}", flush=True)
    finally:
        world.destroy()
    assert not _process_group_cache
    world.launch()
    try:
        value = torch.ones((), device=world.device)
        world.all_reduce(value)
        assert value.item() == 2
        assert dist.get_backend(create_global_process_group(backend)) == backend
        print(f"PASS fresh world {backend} rank={rank}", flush=True)
    finally:
        world.destroy()


if __name__ == "__main__":
    typer.run(main)
