from pathlib import Path
import json
import os
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from dream_trainer.callbacks import CompileCacheCallback


def callback(directory, enabled=True, interval=2):
    result = CompileCacheCallback(directory, save_every_n_steps=interval)
    result.trainer = SimpleNamespace(
        device_parameters=SimpleNamespace(compile_model=enabled),
        world=SimpleNamespace(device=torch.device("cpu"), world_mesh=None),
        is_accumulating_gradients=False,
    )
    return result


def test_cache_lifecycle_and_accumulation(tmp_path, monkeypatch):
    save = Mock(return_value=(b"artifacts", "info"))
    load = Mock(return_value="info")
    monkeypatch.setattr(torch.compiler, "save_cache_artifacts", save)
    monkeypatch.setattr(torch.compiler, "load_cache_artifacts", load)
    first = callback(tmp_path)
    first.pre_configure()
    load.assert_not_called()
    first.trainer.is_accumulating_gradients = True
    first.post_train_step({}, 0)
    save.assert_not_called()
    first.trainer.is_accumulating_gradients = False
    for index in range(3):
        first.post_train_step({}, index)
    assert save.call_count == 2
    first.post_fit()
    assert save.call_count == 3
    assert first.cache_path.is_file()
    assert list(first.cache_path.parent.iterdir()) == [first.cache_path]
    second = callback(tmp_path)
    second.pre_configure()
    load.assert_called_once_with(b"artifacts")


@pytest.mark.parametrize("mode", ["corrupt", "incompatible", "load_error"])
def test_cache_misses_are_nonfatal(tmp_path, monkeypatch, mode):
    monkeypatch.setattr(torch.compiler, "save_cache_artifacts", lambda: (b"valid", "info"))
    load = Mock(return_value="info")
    monkeypatch.setattr(torch.compiler, "load_cache_artifacts", load)
    first = callback(tmp_path)
    first.pre_configure()
    first.post_fit()
    if mode == "corrupt":
        first.cache_path.write_bytes(first.cache_path.read_bytes()[:-2])
    elif mode == "incompatible":
        first.cache_path.write_bytes(
            first.cache_path.read_bytes().replace(b'"format": 1', b'"format": 2')
        )
    else:
        load.side_effect = RuntimeError("rejected")
    callback(tmp_path).pre_configure()
    assert load.call_count == (1 if mode == "load_error" else 0)


def test_failed_publication_preserves_previous_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.compiler, "save_cache_artifacts", lambda: (b"old", "info"))
    first = callback(tmp_path)
    first.pre_configure()
    first.post_fit()
    before = first.cache_path.read_bytes()
    monkeypatch.setattr(torch.compiler, "save_cache_artifacts", lambda: (b"new", "info"))
    monkeypatch.setattr(
        "dream_trainer.callbacks.compile_cache.os.replace", Mock(side_effect=OSError("full"))
    )
    first.post_fit()
    assert first.cache_path.read_bytes() == before
    assert list(first.cache_path.parent.iterdir()) == [first.cache_path]


def test_disabled_compilation_does_not_touch_cache(tmp_path, monkeypatch):
    load, save = Mock(), Mock()
    monkeypatch.setattr(torch.compiler, "load_cache_artifacts", load)
    monkeypatch.setattr(torch.compiler, "save_cache_artifacts", save)
    disabled = callback(tmp_path, enabled=False)
    disabled.pre_configure()
    disabled.post_train_step({}, 0)
    disabled.post_fit()
    load.assert_not_called()
    save.assert_not_called()
    assert not list(tmp_path.iterdir())


def test_partition_changes_with_compiler_version(tmp_path, monkeypatch):
    first = callback(tmp_path)
    first.pre_configure()
    monkeypatch.setattr(torch, "__version__", "different")
    second = callback(tmp_path)
    second.pre_configure()
    assert first.cache_path != second.cache_path


@pytest.mark.parametrize("interval", [0, -1, 1.5, True])
def test_invalid_interval(interval):
    with pytest.raises(ValueError):
        CompileCacheCallback(Path("unused"), interval)


def test_real_cache_across_fresh_processes(tmp_path):
    script = Path(__file__).with_name("compiler_cache_case.py")
    states = {}
    for name, cache, width, scale in (
        ("build", "shared", 8, 1.0),
        ("load", "shared", 8, 1.0),
        ("shape", "shared", 12, 1.0),
        ("shape_control", "shape_control", 12, 1.0),
        ("code", "shared", 8, 2.0),
        ("code_control", "code_control", 8, 2.0),
    ):
        output = tmp_path / f"{name}.pt"
        env = {
            **os.environ,
            "TORCHINDUCTOR_CACHE_DIR": str(tmp_path / f"inductor-{name}"),
            "TRITON_CACHE_DIR": str(tmp_path / f"triton-{name}"),
            "TORCHINDUCTOR_COMPILE_THREADS": "1",
            "OMP_NUM_THREADS": "1",
        }
        result = subprocess.run(
            [
                sys.executable,
                str(script),
                str(tmp_path / cache),
                str(output),
                "--width",
                str(width),
                "--scale",
                str(scale),
            ],
            env=env,
            capture_output=True,
            text=True,
            timeout=180,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        states[name] = torch.load(output, weights_only=True)
    for first, second in (
        ("build", "load"),
        ("shape", "shape_control"),
        ("code", "code_control"),
    ):
        torch.testing.assert_close(states[first], states[second], rtol=0, atol=0)
    counters = json.loads((tmp_path / "load.json").read_text())
    assert counters.get("inductor", {}).get("fxgraph_cache_hit", 0) > 0
