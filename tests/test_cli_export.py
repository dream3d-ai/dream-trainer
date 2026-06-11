from dataclasses import dataclass

import pytest

from dream_trainer.utils.cli.cli import _resolve_export_checkpoint_path


@dataclass
class _CheckpointConfig:
    root_dir: object
    storage_options: dict[str, str] | None = None


class _CheckpointCallback:
    def __init__(self, config: _CheckpointConfig):
        self.config = config


class _TrainerConfig:
    project = "test-project"
    group = "test-group"

    def __init__(self, callbacks: dict[str, _CheckpointCallback]):
        self.callbacks = callbacks


def test_export_resolver_preserves_explicit_checkpoint_path_with_storage_options():
    storage_options = {"profile": "training"}
    config = _TrainerConfig(
        {
            "CheckpointCallback": _CheckpointCallback(
                _CheckpointConfig("/checkpoints", storage_options)
            )
        }
    )

    checkpoint_path, resolved_storage_options = _resolve_export_checkpoint_path(
        trainer_config=config,
        checkpoint_path="/manual/checkpoints",
        run_name=None,
    )

    assert checkpoint_path == "/manual/checkpoints"
    assert resolved_storage_options == storage_options


def test_export_resolver_builds_checkpoint_path_from_run_name_and_callback_config(
    tmp_path,
):
    storage_options = {"endpoint_url": "https://s3.example.com"}
    config = _TrainerConfig(
        {
            "AsyncCheckpointCallback": _CheckpointCallback(
                _CheckpointConfig(tmp_path, storage_options)
            )
        }
    )

    checkpoint_path, resolved_storage_options = _resolve_export_checkpoint_path(
        trainer_config=config,
        checkpoint_path=None,
        run_name="run-123",
    )

    assert (
        checkpoint_path
        == tmp_path / "test-project" / "test-group" / "run-123" / "checkpoints"
    )
    assert resolved_storage_options == storage_options


def test_export_resolver_requires_one_checkpoint_source():
    config = _TrainerConfig({})

    with pytest.raises(ValueError, match="Exactly one"):
        _resolve_export_checkpoint_path(
            trainer_config=config,
            checkpoint_path=None,
            run_name=None,
        )

    with pytest.raises(ValueError, match="Exactly one"):
        _resolve_export_checkpoint_path(
            trainer_config=config,
            checkpoint_path="/manual/checkpoints",
            run_name="run-123",
        )
