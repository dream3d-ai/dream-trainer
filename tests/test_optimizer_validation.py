from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from dream_trainer.trainer.base import BaseTrainer


class StepTrainer:
    """Exercise the real update implementation with ordinary PyTorch state."""

    _optimizer_parameters = BaseTrainer._optimizer_parameters
    _validate_optimizer_parameters = BaseTrainer._validate_optimizer_parameters
    _validate_gradients = BaseTrainer._validate_gradients
    _validate_step = BaseTrainer._validate_step
    step = BaseTrainer.step

    def __init__(self, model, optimizer, scheduler=None, world=None):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.world = world or SimpleNamespace(device=next(model.parameters()).device)
        self.events = []
        self.callbacks = SimpleNamespace(
            **{
                name: lambda *args, name=name: self.events.append(name)
                for name in (
                    "pre_optimizer_step",
                    "post_optimizer_step",
                    "pre_optimizer_zero_grad",
                    "post_optimizer_zero_grad",
                )
            }
        )

    def get_name_by_optimizer(self, optimizer):
        return "optimizer"

    def get_model_by_optimizer(self, name):
        return self.model

    def get_scheduler_from_optimizer(self, optimizer):
        return self.scheduler

    def total_gradient_norm(self, parameters, p, foreach):
        if hasattr(self.world, "get_total_norm"):
            return self.world.get_total_norm(parameters, p, foreach)
        return torch.nn.utils.get_total_norm([p.grad for p in parameters], p, False, foreach)

    def clip_gradient_norm(self, parameters, total_norm, foreach):
        torch.nn.utils.clip_grads_with_norm_(parameters, 0.25, total_norm, foreach=foreach)


def test_update_matches_pytorch_with_accumulation_unused_parameters_and_resume():
    torch.manual_seed(7)
    model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.GELU(), torch.nn.Linear(8, 2))
    model.register_parameter("unused", torch.nn.Parameter(torch.ones(2)))
    model.register_parameter("frozen", torch.nn.Parameter(torch.ones(2), requires_grad=False))
    reference = deepcopy(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    control = torch.optim.AdamW(reference.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9)
    control_scheduler = torch.optim.lr_scheduler.StepLR(control, 1, gamma=0.9)
    trainer = StepTrainer(model, optimizer, scheduler)
    for step in range(3):
        for _ in range(2):
            x = torch.randn(3, 4)
            loss = model(x).square().mean() / 2
            expected = reference(x).square().mean() / 2
            torch.testing.assert_close(loss, expected, rtol=0, atol=0)
            loss.backward()
            expected.backward()
        for actual, wanted in zip(model.parameters(), reference.parameters()):
            if actual.grad is not None:
                torch.testing.assert_close(actual.grad, wanted.grad, rtol=0, atol=0)
        norm = trainer.step(optimizer)
        wanted_norm = torch.nn.utils.clip_grad_norm_(reference.parameters(), 0.25, foreach=True)
        control.step()
        control.zero_grad()
        control_scheduler.step()
        torch.testing.assert_close(norm, wanted_norm, rtol=0, atol=0)
        torch.testing.assert_close(model.state_dict(), reference.state_dict(), rtol=0, atol=0)
        torch.testing.assert_close(optimizer.state_dict(), control.state_dict(), rtol=0, atol=0)
        assert scheduler.state_dict() == control_scheduler.state_dict()
        if step == 1:
            state = deepcopy(optimizer.state_dict())
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
            optimizer.load_state_dict(state)
            saved_scheduler = scheduler.state_dict()
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9)
            scheduler.load_state_dict(saved_scheduler)
            trainer.optimizer, trainer.scheduler = optimizer, scheduler
    assert (
        trainer.events
        == [
            "pre_optimizer_step",
            "post_optimizer_step",
            "pre_optimizer_zero_grad",
            "post_optimizer_zero_grad",
        ]
        * 3
    )


@pytest.mark.parametrize("target", ["parameter", "gradient", "missing", "norm"])
@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_bad_state_cannot_mutate_optimizer(target, value):
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters())
    trainer = StepTrainer(model, optimizer)
    model(torch.ones(1, 2)).sum().backward()
    if target == "parameter":
        model.weight.data[0, 0] = value
    elif target == "gradient":
        model.weight.grad[0, 0] = value
    elif target == "missing":
        optimizer.zero_grad()
    else:
        trainer.total_gradient_norm = lambda *args, **kwargs: torch.tensor(value)
    before = deepcopy(model.state_dict())
    with pytest.raises(RuntimeError):
        trainer.step(optimizer)
    torch.testing.assert_close(model.state_dict(), before, rtol=0, atol=0, equal_nan=True)
    assert not optimizer.state
    assert not trainer.events


def test_independent_optimizers():
    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 1))
    optimizers = [torch.optim.SGD(layer.parameters(), lr=0.1) for layer in model]
    model(torch.ones(1, 2)).sum().backward()
    for optimizer in optimizers:
        StepTrainer(model, optimizer).step(optimizer)
    assert all(parameter.grad is None for parameter in model.parameters())


def test_meta_parameter_rejected_before_norm():
    model = torch.nn.Linear(2, 2, device="meta")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    trainer = StepTrainer(model, optimizer, world=SimpleNamespace(device=torch.device("cpu")))
    with pytest.raises(RuntimeError, match="meta optimizer"):
        trainer.step(optimizer)


def test_validation_host_reads_do_not_scale_with_parameter_count():
    model = torch.nn.Sequential(*(torch.nn.Linear(2, 2) for _ in range(16)))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    model(torch.ones(1, 2)).sum().backward()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
        StepTrainer(model, optimizer)._validate_step(optimizer)
    reads = sum(
        event.count
        for event in profile.key_averages()
        if event.key == "aten::_local_scalar_dense"
    )
    assert reads == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("invalid_cpu_gradient", [False, True])
def test_validation_combines_cpu_and_cuda_state(invalid_cpu_gradient):
    model = torch.nn.Module()
    model.register_parameter("cpu_weight", torch.nn.Parameter(torch.ones(2)))
    model.register_parameter("cuda_weight", torch.nn.Parameter(torch.ones(2, device="cuda")))
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    trainer = StepTrainer(model, optimizer, world=SimpleNamespace(device=torch.device("cuda")))
    if invalid_cpu_gradient:
        model.cpu_weight.grad[0] = float("nan")
        with pytest.raises(RuntimeError, match="non-finite gradients"):
            trainer._validate_step(optimizer)
    else:
        assert [id(parameter) for parameter in trainer._validate_step(optimizer)] == [
            id(parameter) for parameter in model.parameters()
        ]
