import threading
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from dream_trainer.callbacks.checkpoint import async_ as async_checkpoint
from dream_trainer.callbacks.checkpoint import base as checkpoint_base
from dream_trainer.callbacks.checkpoint.async_ import AsyncCheckpointCallback
from dream_trainer.callbacks.checkpoint.types import Checkpoint
from dream_trainer.configs.checkpoint import CheckpointParameters


KEEP_TOP_K_ONE_MESSAGE = r"process of being saved\. Please set keep_top_k to 0"


def checkpoint_config(**overrides):
    kwargs = {
        "root_dir": Path("/tmp/checkpoints"),
        "checkpoint_every_n_train_epochs": 1,
    }
    kwargs.update(overrides)
    return CheckpointParameters(**kwargs)


def test_checkpoint_config_accepts_async_modes():
    assert checkpoint_config(async_mode="disabled").async_mode == "disabled"
    assert checkpoint_config(async_mode="async").async_mode == "async"
    assert (
        checkpoint_config(async_mode="async_with_pinned_mem").async_mode
        == "async_with_pinned_mem"
    )


def test_checkpoint_config_rejects_unknown_async_mode():
    with pytest.raises(ValueError, match="Invalid async_mode"):
        checkpoint_config(async_mode="background")


def test_checkpoint_config_keep_top_k_zero_keeps_all():
    assert checkpoint_config(keep_top_k=0).keep_top_k == 0


def test_checkpoint_config_rejects_keep_top_k_one():
    with pytest.raises(ValueError, match=KEEP_TOP_K_ONE_MESSAGE):
        checkpoint_config(keep_top_k=1)


def test_checkpoint_config_rejects_keep_top_k_one_when_disabled():
    with pytest.raises(ValueError, match=KEEP_TOP_K_ONE_MESSAGE):
        checkpoint_config(enable=False, keep_top_k=1)


def test_async_checkpoint_rejects_disabled_mode():
    with pytest.raises(ValueError, match="AsyncCheckpointCallback.*disabled"):
        AsyncCheckpointCallback(
            checkpoint_config(
                async_mode="disabled",
                checkpoint_every_n_train_epochs=1,
            )
        )


class DummyWorld:
    is_global_zero = True

    def __init__(self):
        self.barrier_calls = 0
        self.reduced = []

    def barrier(self):
        self.barrier_calls += 1

    def all_reduce(self, tensor, op=None):
        self.reduced.append((tensor, op))


class DummyTrainer:
    project = "project"
    group = "group"
    experiment = "experiment"
    is_sanity_validation = False

    def __init__(self):
        self.world = DummyWorld()
        self.training_parameters = SimpleNamespace(val_every_n_steps=None)
        self.global_step = 0
        self.current_epoch = 0
        self.local_batches = 0
        self.loaded_state_dict = None
        self.load_kwargs = None
        self.post_load_state_dict_calls = []
        self.callbacks = SimpleNamespace(
            post_load_state_dict=self.post_load_state_dict_calls.append
        )

    def state_dict(self):
        return {
            "trainer": {
                "global_step": self.global_step,
                "current_epoch": self.current_epoch,
                "local_batches": self.local_batches,
                "callbacks": {"EMACallback": {"stale": True}},
            },
            "models": {"model": {"weight": "initial"}},
            "optimizers": {},
            "schedulers": {},
            "dataloaders": {},
        }

    def load_state_dict(self, state_dict, **kwargs):
        self.loaded_state_dict = state_dict
        self.load_kwargs = kwargs
        trainer_state = state_dict["trainer"]
        loaded_callback_names = set(trainer_state.get("callbacks", {}))
        self.global_step = trainer_state["global_step"]
        self.current_epoch = trainer_state["current_epoch"]
        self.local_batches = trainer_state["local_batches"]
        self.callbacks.post_load_state_dict(loaded_callback_names)


def attach_callback(callback, trainer, tmp_path):
    callback.trainer = trainer
    callback.pg = object()
    callback._current_metric = None
    callback._last_saved_step = None
    callback.__dict__["root_dir"] = tmp_path
    return callback


def create_checkpoint_dir(root, checkpoint):
    path = root / checkpoint.checkpoint_id
    path.mkdir(parents=True)
    (path / ".metadata").write_text("ok")
    return path


def test_stale_checkpoints_use_last_sort_for_integer_resume_mode(monkeypatch, tmp_path):
    callback = attach_callback(
        checkpoint_base.CheckpointCallback(
            checkpoint_config(resume_mode=3, keep_top_k=2)
        ),
        DummyTrainer(),
        tmp_path,
    )
    checkpoints = [
        Checkpoint(step=5, metric=5.0),
        Checkpoint(step=4, metric=4.0),
        Checkpoint(step=3, metric=3.0),
    ]
    modes = []

    def find_checkpoints(root_dir, mode):
        modes.append(mode)
        return checkpoints

    monkeypatch.setattr(checkpoint_base, "find_checkpoints", find_checkpoints)

    assert callback._stale_checkpoints() == checkpoints[2:]
    assert modes == ["last"]


def test_async_cleanup_keep_top_k_zero_keeps_all(monkeypatch, tmp_path):
    callback = attach_callback(
        AsyncCheckpointCallback(
            checkpoint_config(
                async_mode="async",
                keep_top_k=0,
                checkpoint_every_n_train_epochs=1,
            )
        ),
        DummyTrainer(),
        tmp_path,
    )
    queued = []
    callback.purge_queue = SimpleNamespace(put=lambda item: queued.append(item))

    for step in range(3):
        create_checkpoint_dir(tmp_path, Checkpoint(step=step, metric=float(step)))

    callback._cleanup_checkpoints()

    assert queued == []


def test_async_cleanup_only_global_zero_queues_deletes(monkeypatch, tmp_path):
    callback = attach_callback(
        AsyncCheckpointCallback(
            checkpoint_config(
                async_mode="async",
                keep_top_k=2,
                checkpoint_every_n_train_epochs=1,
            )
        ),
        DummyTrainer(),
        tmp_path,
    )
    callback.trainer.world.is_global_zero = False
    queued = []
    callback.purge_queue = SimpleNamespace(put=lambda item: queued.append(item))

    for step in range(4):
        create_checkpoint_dir(tmp_path, Checkpoint(step=step, metric=float(step)))

    callback._cleanup_checkpoints()

    assert queued == []


def test_checkpoint_exists_returns_python_bool(monkeypatch, tmp_path):
    callback = attach_callback(
        AsyncCheckpointCallback(
            checkpoint_config(
                async_mode="async",
                resume_mode="max",
                checkpoint_every_n_train_epochs=1,
            )
        ),
        DummyTrainer(),
        tmp_path,
    )
    monkeypatch.setattr(checkpoint_base.dist, "broadcast", lambda *args, **kwargs: None)

    create_checkpoint_dir(tmp_path, Checkpoint(step=1, metric=1.0))

    assert callback._checkpoint_exists(Checkpoint(step=1, metric=1.0)) is True
    assert callback._checkpoint_exists(Checkpoint(step=2, metric=2.0)) is False


def test_async_checkpoint_load_is_synchronous(monkeypatch, tmp_path):
    callback = attach_callback(
        AsyncCheckpointCallback(
            checkpoint_config(async_mode="async", checkpoint_every_n_train_epochs=1)
        ),
        DummyTrainer(),
        tmp_path,
    )
    checkpoint = Checkpoint(step=17, metric=0.25)

    def load(state_dict, **kwargs):
        state_dict["trainer"]["global_step"] = 17
        state_dict["trainer"]["current_epoch"] = 3
        state_dict["trainer"]["local_batches"] = 51
        state_dict["models"]["model"]["weight"] = "loaded"

    monkeypatch.setattr(async_checkpoint.dcp.state_dict_loader, "load", load)
    monkeypatch.setattr(
        callback,
        "_checkpoint_callback_names",
        lambda checkpoint: {"CheckpointCallback"},
    )

    callback.load(checkpoint)

    assert callback.trainer.global_step == 17
    assert callback.trainer.current_epoch == 3
    assert callback.trainer.local_batches == 51
    assert callback.trainer.loaded_state_dict["models"]["model"]["weight"] == "loaded"
    assert callback.trainer.loaded_state_dict["trainer"]["callbacks"] == {}
    assert callback.trainer.load_kwargs == {"strict": False, "resume_data": True}
    assert callback.trainer.post_load_state_dict_calls == [set()]
    assert callback._did_resume is True
    assert callback._last_saved_step == 17
    assert not hasattr(callback, "_load_future") or callback._load_future is None


def finished_future(result=None):
    future = Future()
    future.set_result(result)
    return future


def failed_future(exc):
    future = Future()
    future.set_exception(exc)
    return future


class RecordingFuture(Future):
    def __init__(self, events):
        super().__init__()
        self.events = events
        self.set_result("metadata")

    def result(self, timeout=None):
        self.events.append("wait")
        return super().result(timeout=timeout)


class FakeAsyncSaveResponse:
    def __init__(self, staging_completion, upload_completion):
        self.staging_completion = staging_completion
        self.upload_completion = upload_completion


def test_async_with_pinned_mem_uses_process_checkpointer_and_stager(
    monkeypatch, tmp_path
):
    callback = attach_callback(
        AsyncCheckpointCallback(
            checkpoint_config(
                async_mode="async_with_pinned_mem",
                checkpoint_every_n_train_epochs=1,
            )
        ),
        DummyTrainer(),
        tmp_path,
    )
    calls = []
    staging_future = finished_future("staged")
    upload_future = finished_future("uploaded")

    class FakeStager:
        def __init__(self, options):
            self.options = options
            self.closed = False

        def close(self):
            self.closed = True

    def async_save(state_dict, **kwargs):
        calls.append(kwargs)
        return FakeAsyncSaveResponse(staging_future, upload_future)

    monkeypatch.setattr(
        async_checkpoint.torch,
        "accelerator",
        SimpleNamespace(is_available=lambda: True),
        raising=False,
    )
    monkeypatch.setattr(async_checkpoint, "AsyncSaveResponse", FakeAsyncSaveResponse)
    monkeypatch.setattr(async_checkpoint, "DefaultStager", FakeStager)
    monkeypatch.setattr(async_checkpoint.dcp.state_dict_saver, "async_save", async_save)
    monkeypatch.setattr(callback, "_checkpoint_exists", lambda checkpoint: False)
    monkeypatch.setattr(callback, "_cleanup_checkpoints", lambda: None)

    callback._current_metric = torch.tensor(1.0)
    callback.trainer.global_step = 9
    callback.save()

    assert callback.staging_future is staging_future
    assert callback._save_future is upload_future
    assert callback.staging
    assert callback._stager.options == async_checkpoint.StagingOptions(
        True, True, True, True
    )
    assert (
        calls[0]["async_checkpointer_type"]
        is async_checkpoint.AsyncCheckpointerType.PROCESS
    )
    assert calls[0]["async_stager"] is callback._stager


def test_async_with_pinned_mem_waits_for_staging_before_next_train_step(tmp_path):
    callback = attach_callback(
        AsyncCheckpointCallback(
            checkpoint_config(
                async_mode="async_with_pinned_mem",
                checkpoint_every_n_train_epochs=1,
            )
        ),
        DummyTrainer(),
        tmp_path,
    )
    staging_future = Future()
    upload_future = Future()
    returned = threading.Event()
    errors = []

    callback.staging = True
    callback.staging_future = staging_future
    callback._save_future = upload_future

    def run_pre_train_step():
        try:
            callback.pre_train_step({}, 0)
        except Exception as exc:
            errors.append(exc)
        finally:
            returned.set()

    thread = threading.Thread(target=run_pre_train_step, daemon=True)
    thread.start()

    assert not returned.wait(timeout=0.05)

    staging_future.set_result("staged")
    thread.join(timeout=1)

    assert not thread.is_alive()
    assert errors == []
    assert callback.staging_future is None
    assert not callback.staging
    assert callback._save_future is upload_future
    assert not upload_future.done()


def test_async_with_pinned_mem_uses_fresh_stager_for_each_save(
    monkeypatch, tmp_path
):
    callback = attach_callback(
        AsyncCheckpointCallback(
            checkpoint_config(
                async_mode="async_with_pinned_mem",
                checkpoint_every_n_train_epochs=1,
            )
        ),
        DummyTrainer(),
        tmp_path,
    )
    events = []
    stagers = []

    class FakeStager:
        def __init__(self, options):
            self.id = len(stagers)
            self.options = options
            self.closed = False
            stagers.append(self)
            events.append(("create", self.id))

        def close(self):
            self.closed = True
            events.append(("close", self.id))

    def async_save(state_dict, **kwargs):
        return FakeAsyncSaveResponse(finished_future("staged"), finished_future("done"))

    monkeypatch.setattr(async_checkpoint, "AsyncSaveResponse", FakeAsyncSaveResponse)
    monkeypatch.setattr(async_checkpoint, "DefaultStager", FakeStager)
    monkeypatch.setattr(async_checkpoint.dcp.state_dict_saver, "async_save", async_save)
    monkeypatch.setattr(callback, "_checkpoint_exists", lambda checkpoint: False)
    monkeypatch.setattr(callback, "_cleanup_checkpoints", lambda: None)

    callback._current_metric = torch.tensor(1.0)
    callback.trainer.global_step = 5
    callback.save()

    assert len(stagers) == 1
    assert not stagers[0].closed

    callback.trainer.global_step = 6
    callback.save()

    assert len(stagers) == 2
    assert stagers[0] is not stagers[1]
    assert stagers[0].closed
    assert not stagers[1].closed
    assert events.index(("close", 0)) < events.index(("create", 1))


def test_pinned_memory_staging_options_disable_non_blocking_without_accelerator(
    monkeypatch,
):
    monkeypatch.setattr(
        async_checkpoint.torch,
        "accelerator",
        SimpleNamespace(is_available=lambda: False),
        raising=False,
    )

    options = async_checkpoint._pinned_memory_staging_options()

    assert options == async_checkpoint.StagingOptions(True, True, True, False)


def test_async_with_pinned_mem_rejects_non_async_save_response(
    monkeypatch, tmp_path
):
    callback = attach_callback(
        AsyncCheckpointCallback(
            checkpoint_config(
                async_mode="async_with_pinned_mem",
                checkpoint_every_n_train_epochs=1,
            )
        ),
        DummyTrainer(),
        tmp_path,
    )
    stagers = []

    class FakeStager:
        def __init__(self, options):
            self.closed = False
            stagers.append(self)

        def close(self):
            self.closed = True

    monkeypatch.setattr(async_checkpoint, "DefaultStager", FakeStager)
    monkeypatch.setattr(
        async_checkpoint.dcp.state_dict_saver,
        "async_save",
        lambda state_dict, **kwargs: finished_future("wrong-shape"),
    )
    monkeypatch.setattr(callback, "_checkpoint_exists", lambda checkpoint: False)

    callback._current_metric = torch.tensor(1.0)
    callback.trainer.global_step = 5

    with pytest.raises(TypeError, match="AsyncSaveResponse"):
        callback.save()

    assert stagers[0].closed


def test_async_post_fit_waits_for_in_flight_save(monkeypatch, tmp_path):
    callback = attach_callback(
        AsyncCheckpointCallback(
            checkpoint_config(async_mode="async", checkpoint_every_n_train_epochs=1)
        ),
        DummyTrainer(),
        tmp_path,
    )
    waited = []
    future = Future()
    save_started = threading.Event()
    post_fit_returned = threading.Event()
    post_fit_errors = []

    def save(state_dict, **kwargs):
        waited.append(("started", kwargs["checkpoint_id"]))
        save_started.set()
        return future

    def run_post_fit():
        try:
            callback.post_fit()
        except Exception as exc:
            post_fit_errors.append(exc)
        finally:
            post_fit_returned.set()

    monkeypatch.setattr(callback, "_checkpoint_exists", lambda checkpoint: False)
    monkeypatch.setattr(async_checkpoint.dcp.state_dict_saver, "async_save", save)
    monkeypatch.setattr(
        callback, "_cleanup_checkpoints", lambda: waited.append(("cleanup", None))
    )

    callback._current_metric = torch.tensor(1.0)
    callback.trainer.global_step = 5
    thread = threading.Thread(target=run_post_fit, daemon=True)
    thread.start()

    assert save_started.wait(timeout=1)
    assert not post_fit_returned.wait(timeout=0.05)
    assert callback._save_future is future
    assert waited == [("started", str(tmp_path / "step=000005__metric=1_000000"))]

    future.set_result("metadata")
    thread.join(timeout=1)

    assert not thread.is_alive()
    assert post_fit_errors == []
    assert callback._save_future is None
    assert callback._last_saved_step == 5
    assert waited[-1] == ("cleanup", None)


def test_async_save_failure_does_not_mark_step_saved_or_cleanup(monkeypatch, tmp_path):
    callback = attach_callback(
        AsyncCheckpointCallback(
            checkpoint_config(async_mode="async", checkpoint_every_n_train_epochs=1)
        ),
        DummyTrainer(),
        tmp_path,
    )
    cleanup_calls = []
    error = RuntimeError("upload failed")

    monkeypatch.setattr(callback, "_checkpoint_exists", lambda checkpoint: False)
    monkeypatch.setattr(
        async_checkpoint.dcp.state_dict_saver,
        "async_save",
        lambda state_dict, **kwargs: failed_future(error),
    )
    monkeypatch.setattr(
        callback, "_cleanup_checkpoints", lambda: cleanup_calls.append("cleanup")
    )

    callback._current_metric = torch.tensor(1.0)
    callback.trainer.global_step = 5
    callback.save()

    with pytest.raises(RuntimeError, match="upload failed"):
        callback._wait_save()

    assert callback._last_saved_step is None
    assert cleanup_calls == []


def test_async_second_save_waits_before_state_dict(monkeypatch, tmp_path):
    events = []

    class RecordingTrainer(DummyTrainer):
        def state_dict(self):
            events.append(("state_dict", self.global_step))
            return super().state_dict()

    callback = attach_callback(
        AsyncCheckpointCallback(
            checkpoint_config(async_mode="async", checkpoint_every_n_train_epochs=1)
        ),
        RecordingTrainer(),
        tmp_path,
    )

    monkeypatch.setattr(callback, "_checkpoint_exists", lambda checkpoint: False)
    monkeypatch.setattr(
        async_checkpoint.dcp.state_dict_saver,
        "async_save",
        lambda state_dict, **kwargs: finished_future("next-metadata"),
    )
    monkeypatch.setattr(
        callback, "_cleanup_checkpoints", lambda: events.append("cleanup")
    )

    callback._save_future = RecordingFuture(events)
    callback._pending_saved_step = 4
    callback._pending_checkpoint_path = tmp_path / "step=000004__metric=1_000000"
    callback._current_metric = torch.tensor(2.0)
    callback.trainer.global_step = 5

    callback.save()

    assert events.index("wait") < events.index(("state_dict", 5))


def test_async_close_shuts_down_purge_thread_when_save_wait_fails(
    monkeypatch, tmp_path
):
    callback = attach_callback(
        AsyncCheckpointCallback(
            checkpoint_config(
                async_mode="async", checkpoint_every_n_train_epochs=1, keep_top_k=2
            )
        ),
        DummyTrainer(),
        tmp_path,
    )
    events = []

    class FakeThread:
        def is_alive(self):
            return True

        def join(self):
            events.append("join")

    def fail_wait():
        raise RuntimeError("failed")

    monkeypatch.setattr(callback, "_wait_save", fail_wait)
    callback.purge_thread = FakeThread()

    with pytest.raises(RuntimeError, match="failed"):
        callback._close()

    assert isinstance(callback.purge_queue.get_nowait(), async_checkpoint.Terminate)
    assert events == ["join"]
