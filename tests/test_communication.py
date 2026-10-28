from unittest.mock import Mock

import pytest
import torch

from dream_trainer.configs import Comm
from dream_trainer.utils.dist import core


def test_backend_selection(monkeypatch):
    monkeypatch.setattr(torch.distributed, "is_backend_available", lambda backend: True)
    assert Comm().resolve_backend("cuda") == "nccl"
    assert Comm().resolve_backend("cpu") == "gloo"
    assert Comm(backend="nccl2").resolve_backend("cuda") == "nccl2"
    with pytest.raises(ValueError, match="CUDA"):
        Comm(backend="nccl2").resolve_backend("cpu")
    with pytest.raises(ValueError, match="Unknown"):
        Comm(backend="typo").resolve_backend("cuda")
    monkeypatch.setattr(torch.distributed, "is_backend_available", lambda backend: False)
    with pytest.raises(RuntimeError, match="unavailable"):
        Comm(backend="nccl2").resolve_backend("cuda")


@pytest.mark.parametrize("backend", ["nccl", "nccl2", "cuda:nccl2,cpu:gloo", "gloo"])
def test_barrier_uses_supplied_group_and_local_device(monkeypatch, backend):
    group = object()
    monkeypatch.setattr(core.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(core.dist, "get_backend", lambda selected: backend)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 1)
    barrier = Mock()
    monkeypatch.setattr(core.dist, "barrier", barrier)
    core.safe_barrier(group)
    expected = {"group": group}
    if "nccl" in backend:
        expected["device_ids"] = [1]
    barrier.assert_called_once_with(**expected)


def test_local_groups_created_in_global_order_and_cached_per_backend(monkeypatch):
    monkeypatch.setattr(core.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(core, "get_dist_world_size", lambda: 4)
    monkeypatch.setattr(core, "get_dist_local_world_size", lambda: 2)
    monkeypatch.setattr(core, "get_dist_rank", lambda: 3)
    created = []

    def new_group(**kwargs):
        group = object()
        created.append((kwargs, group))
        return group

    monkeypatch.setattr(core.dist, "new_group", new_group)
    core.clear_process_group_cache()
    first = core.create_local_process_group("nccl")
    assert first is core.create_local_process_group("nccl")
    second = core.create_local_process_group("nccl2")
    assert second is not first
    assert [entry[0] for entry in created] == [
        {"ranks": [0, 1], "backend": "nccl"},
        {"ranks": [2, 3], "backend": "nccl"},
        {"ranks": [0, 1], "backend": "nccl2"},
        {"ranks": [2, 3], "backend": "nccl2"},
    ]
    core.clear_process_group_cache()
    assert core.create_local_process_group("nccl") is not first
    core.clear_process_group_cache()
