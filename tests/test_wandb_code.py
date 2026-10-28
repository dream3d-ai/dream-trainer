import errno
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

wandb = pytest.importorskip("wandb")

from dream_trainer.configs.logger import WandbLoggingParameters  # noqa: E402
from dream_trainer.trainer.mixins.loggers.wandb import WandBLoggerMixin  # noqa: E402
from wandb.sdk.lib.filenames import filtered_dir  # noqa: E402


def make_logger(run, *, enabled=True, rank_zero=True):
    return SimpleNamespace(
        _wandb=run,
        logging_parameters=WandbLoggingParameters(enabled=enabled),
        world=SimpleNamespace(is_global_zero=rank_zero),
        _should_ignore=lambda patterns: WandBLoggerMixin._should_ignore(None, patterns),
    )


def test_snapshot_filters_environments_generated_files_and_external_symlinks(tmp_path):
    root = tmp_path / "project"
    sources = ["train.py", "models/src/model.py", "dream-trainer/src/trainer.py"]
    excluded = [
        ".venv/lib/python3.14/site-packages/torch/test.py",
        "nested/.venv/lib/dependency.py",
        "nested/venv/lib/dependency.py",
        "other/lib/site-packages/dependency.py",
        "wandb/run/tmp/code/train.py",
        "nested/.wandb/train.py",
        "node_modules/lib/tool.py",
        ".git/script.py",
        "nested/__pycache__/generated.py",
        "generated/output.py",
    ]
    for relative in sources + excluded:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("pass\n")
    outside = tmp_path / "sibling.py"
    outside.write_text("pass\n")
    (root / "linked.py").symlink_to(outside)
    (root / ".gitignore").write_text("generated/\n")
    run = Mock()
    WandBLoggerMixin.log_code.__wrapped__(
        make_logger(run), str(root), gitignore_path=str(root / ".gitignore")
    )
    files = list(filtered_dir(**run.log_code.call_args.kwargs))
    assert {str(path.relative_to(root)) for path in map(Path, files)} == set(sources)


def test_ignore_patterns_match_directory_boundaries(tmp_path):
    exclude = WandBLoggerMixin._should_ignore(None, ["generated/", "*.private.py"])
    assert exclude(str(tmp_path / "nested/generated/code.py"), str(tmp_path))
    assert exclude(str(tmp_path / "nested/code.private.py"), str(tmp_path))
    assert not exclude(str(tmp_path / "generated_model.py"), str(tmp_path))


def test_snapshot_does_not_mutate_or_reuse_ignore_patterns(tmp_path):
    run = Mock()
    trainer = make_logger(run)
    gitignore = tmp_path / ".gitignore"
    gitignore.write_text("ignored.py\n")
    patterns = ["private.py"]
    WandBLoggerMixin.log_code.__wrapped__(trainer, str(tmp_path), patterns, str(gitignore))
    assert patterns == ["private.py"]
    assert run.log_code.call_args.kwargs["exclude_fn"](
        str(tmp_path / "ignored.py"), str(tmp_path)
    )
    WandBLoggerMixin.log_code.__wrapped__(trainer, str(tmp_path), gitignore_path=None)
    assert not run.log_code.call_args.kwargs["exclude_fn"](
        str(tmp_path / "ignored.py"), str(tmp_path)
    )


@pytest.mark.parametrize("enabled,rank_zero", [(False, True), (True, False)])
def test_snapshot_skips_disabled_and_nonzero_rank(enabled, rank_zero):
    run = Mock()
    WandBLoggerMixin.log_code.__wrapped__(
        make_logger(run, enabled=enabled, rank_zero=rank_zero)
    )
    run.log_code.assert_not_called()


def test_snapshot_disk_full_is_reported_without_unhandled_thread_error(monkeypatch):
    run = Mock()
    run.log_code.side_effect = OSError(errno.ENOSPC, "No space left on device")
    warning = Mock()
    monkeypatch.setattr("dream_trainer.trainer.mixins.loggers.wandb.logger.warning", warning)
    unhandled = Mock()
    monkeypatch.setattr("threading.excepthook", unhandled)
    thread = WandBLoggerMixin.log_code(make_logger(run), gitignore_path=None)
    thread.join(timeout=5)
    assert not thread.is_alive()
    unhandled.assert_not_called()
    warning.assert_called_once()
    assert "No space left on device" in warning.call_args.args[0]


def test_real_offline_wandb_artifact_contains_only_source(tmp_path, monkeypatch):
    monkeypatch.setenv("WANDB_DATA_DIR", str(tmp_path / "staging"))
    monkeypatch.setenv("WANDB_CACHE_DIR", str(tmp_path / "cache"))
    root = tmp_path / "project"
    root.mkdir()
    (root / "train.py").write_text("pass\n")
    env = root / ".venv/lib"
    env.mkdir(parents=True)
    (env / "dependency.py").write_text("pass\n")
    with wandb.init(project="dream-trainer-tests", mode="offline", dir=str(tmp_path)) as run:
        # Use the installed SDK's actual file staging path without network access.
        artifact = run.log_code(
            root=str(root),
            include_fn=lambda path: path.endswith(".py"),
            exclude_fn=WandBLoggerMixin._should_ignore(None, []),
        )
        assert artifact is not None
        assert set(artifact.manifest.entries) == {"train.py"}
