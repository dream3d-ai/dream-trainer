import queue
from concurrent.futures import Future
from math import isnan
from pathlib import Path
from types import SimpleNamespace

import pytest

from dream_trainer.callbacks.checkpoint import async_ as async_checkpoint
from dream_trainer.callbacks.checkpoint import base as checkpoint_base
from dream_trainer.callbacks.checkpoint import s3 as checkpoint_s3
from dream_trainer.callbacks.checkpoint.async_ import AsyncCheckpointCallback
from dream_trainer.callbacks.checkpoint.base import CheckpointCallback
from dream_trainer.callbacks.checkpoint.types import CHECKPOINT_REGEX, Checkpoint
from dream_trainer.configs.checkpoint import CheckpointParameters


def checkpoint_config(**overrides):
    kwargs = {
        "root_dir": "s3://checkpoints",
        "checkpoint_every_n_train_epochs": 1,
        "storage_options": {
            "region": "auto",
            "endpoint_url": "https://r2.example.test",
            "access_key_id": "test-key",
            "secret_access_key": "test-secret",
        },
    }
    kwargs.update(overrides)
    return CheckpointParameters(**kwargs)


class DummyWorld:
    is_global_zero = True

    def __init__(self):
        self.barrier_calls = 0

    def barrier(self):
        self.barrier_calls += 1


class DummyTrainer:
    project = "project"
    group = "group"
    experiment = "experiment"

    def __init__(self):
        self.world = DummyWorld()


def attach_callback(callback):
    callback.trainer = DummyTrainer()
    callback.pg = object()
    callback._last_saved_step = None
    callback._current_metric = None
    return callback


def test_checkpoint_root_dir_preserves_s3_uri():
    callback = attach_callback(CheckpointCallback(checkpoint_config()))

    assert (
        callback.root_dir
        == "s3://checkpoints/project/group/experiment/checkpoints"
    )


def test_s3_checkpoint_storage_lists_metadata_backed_checkpoints():
    fs = FakeS3FileSystem(
        [
            "runs/checkpoints/step=000001__metric=2_000000/.metadata",
            "runs/checkpoints/step=000001__metric=2_000000/__0_0.distcp",
            "runs/checkpoints/step=000003__metric=1_000000/.metadata",
            "runs/checkpoints/not-a-checkpoint/.metadata",
        ]
    )
    storage = checkpoint_s3.S3CheckpointStorage(
        "s3://bucket/runs/checkpoints",
        {"region": "auto"},
        fs=fs,
    )

    checkpoints = storage.find_checkpoints("last")

    assert [checkpoint.step for checkpoint in checkpoints] == [3, 1]


def test_s3_checkpoint_storage_deletes_checkpoint_prefix():
    fs = FakeS3FileSystem(
        [
            "runs/checkpoints/step=000001__metric=2_000000/.metadata",
            "runs/checkpoints/step=000001__metric=2_000000/__0_0.distcp",
            "runs/checkpoints/step=000002__metric=1_000000/.metadata",
        ]
    )
    storage = checkpoint_s3.S3CheckpointStorage(
        "s3://bucket/runs/checkpoints",
        {"region": "auto"},
        fs=fs,
    )

    storage.delete_checkpoint("step=000001__metric=2_000000")

    assert fs.deleted == [
        "s3://bucket/runs/checkpoints/step=000001__metric=2_000000/.metadata",
        "s3://bucket/runs/checkpoints/step=000001__metric=2_000000/__0_0.distcp",
    ]


def test_s3_checkpoint_storage_reader_consumes_force_path_style(
    monkeypatch: pytest.MonkeyPatch,
):
    calls = []

    class FakeS3StorageReader:
        def __init__(self, **kwargs):
            calls.append(kwargs)

    class FakeS3ClientConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(checkpoint_s3, "_s3_storage_reader", lambda: FakeS3StorageReader)
    monkeypatch.setattr(checkpoint_s3, "_s3_client_config", lambda: FakeS3ClientConfig)
    storage = checkpoint_s3.S3CheckpointStorage(
        "s3://bucket/runs/checkpoints",
        {
            "endpoint_url": "https://r2.example.test",
            "force_path_style": True,
        },
        fs=FakeS3FileSystem([]),
    )

    storage.reader("step=000001__metric=2_000000")

    assert "force_path_style" not in calls[0]
    assert calls[0]["s3client_config"].kwargs == {"force_path_style": True}


@pytest.mark.parametrize("metric", [-0.5, float("inf"), float("-inf"), float("nan")])
def test_checkpoint_id_round_trips_special_metric_values(metric):
    checkpoint = Checkpoint(step=1, metric=metric)

    assert CHECKPOINT_REGEX.search(checkpoint.checkpoint_id) is not None
    parsed = Checkpoint.from_path(Path(checkpoint.checkpoint_id))

    assert parsed.step == checkpoint.step
    if isnan(metric):
        assert isnan(parsed.metric)
    else:
        assert parsed.metric == metric


def test_s3_checkpoint_save_uses_storage_writer(monkeypatch: pytest.MonkeyPatch):
    callback = attach_callback(CheckpointCallback(checkpoint_config()))
    checkpoint = Checkpoint(step=5, metric=1.0)
    save_calls = []
    storage = FakeCheckpointStorage(callback.root_dir, exists=False)

    monkeypatch.setattr(callback, "_checkpoint_storage", storage)
    monkeypatch.setattr(checkpoint_base.dist, "broadcast", lambda *args, **kwargs: None)

    def fake_save(state_dict, **kwargs):
        save_calls.append((state_dict, kwargs))

    monkeypatch.setattr(checkpoint_base.dcp.state_dict_saver, "save", fake_save)

    callback._save(checkpoint, {"trainer": {}})

    assert save_calls[0][1]["checkpoint_id"] == (
        "s3://checkpoints/project/group/experiment/checkpoints/"
        "step=000005__metric=1_000000"
    )
    assert save_calls[0][1]["storage_writer"] == "writer:step=000005__metric=1_000000"
    assert "storage_reader" not in save_calls[0][1]


def test_s3_checkpoint_load_uses_storage_reader(monkeypatch: pytest.MonkeyPatch):
    callback = attach_callback(CheckpointCallback(checkpoint_config()))
    checkpoint = Checkpoint(step=5, metric=1.0)
    load_calls = []
    storage = FakeCheckpointStorage(callback.root_dir, exists=True)

    monkeypatch.setattr(callback, "_checkpoint_storage", storage)

    def fake_load(state_dict, **kwargs):
        load_calls.append((state_dict, kwargs))

    monkeypatch.setattr(checkpoint_base.dcp.state_dict_loader, "load", fake_load)

    callback._load(checkpoint, {"trainer": {}})

    assert load_calls[0][1]["checkpoint_id"] == (
        "s3://checkpoints/project/group/experiment/checkpoints/"
        "step=000005__metric=1_000000"
    )
    assert load_calls[0][1]["storage_reader"] == "reader:step=000005__metric=1_000000"
    assert callback.trainer.world.barrier_calls == 1


def test_s3_async_checkpoint_save_uses_storage_writer(
    monkeypatch: pytest.MonkeyPatch,
):
    callback = attach_callback(
        AsyncCheckpointCallback(checkpoint_config(async_mode="async"))
    )
    checkpoint = Checkpoint(step=5, metric=1.0)
    save_calls = []
    storage = FakeCheckpointStorage(callback.root_dir, exists=False)

    monkeypatch.setattr(callback, "_checkpoint_storage", storage)
    monkeypatch.setattr(callback, "_checkpoint_exists", lambda checkpoint: False)

    def fake_async_save(state_dict, **kwargs):
        save_calls.append((state_dict, kwargs))
        future = Future()
        future.set_result("saved")
        return future

    monkeypatch.setattr(
        async_checkpoint.dcp.state_dict_saver,
        "async_save",
        fake_async_save,
    )

    callback._save(checkpoint, {"trainer": {}})

    assert save_calls[0][1]["checkpoint_id"] == (
        "s3://checkpoints/project/group/experiment/checkpoints/"
        "step=000005__metric=1_000000"
    )
    assert save_calls[0][1]["storage_writer"] == "writer:step=000005__metric=1_000000"


def test_s3_pinned_async_checkpoint_save_uses_storage_writer_and_process_stager(
    monkeypatch: pytest.MonkeyPatch,
):
    callback = attach_callback(
        AsyncCheckpointCallback(checkpoint_config(async_mode="async_with_pinned_mem"))
    )
    checkpoint = Checkpoint(step=5, metric=1.0)
    save_calls = []
    stagers = []
    staging_future = Future()
    upload_future = Future()
    storage = FakeCheckpointStorage(callback.root_dir, exists=False)

    class FakeAsyncSaveResponse:
        def __init__(self, staging_completion, upload_completion):
            self.staging_completion = staging_completion
            self.upload_completion = upload_completion

    class FakeStager:
        def __init__(self, options):
            self.options = options
            stagers.append(self)

        def close(self):
            pass

    def fake_async_save(state_dict, **kwargs):
        save_calls.append((state_dict, kwargs))
        return FakeAsyncSaveResponse(staging_future, upload_future)

    monkeypatch.setattr(callback, "_checkpoint_storage", storage)
    monkeypatch.setattr(callback, "_checkpoint_exists", lambda checkpoint: False)
    monkeypatch.setattr(async_checkpoint, "AsyncSaveResponse", FakeAsyncSaveResponse)
    monkeypatch.setattr(async_checkpoint, "DefaultStager", FakeStager)
    monkeypatch.setattr(
        async_checkpoint.dcp.state_dict_saver,
        "async_save",
        fake_async_save,
    )

    callback._save(checkpoint, {"trainer": {}})

    assert save_calls[0][1]["checkpoint_id"] == (
        "s3://checkpoints/project/group/experiment/checkpoints/"
        "step=000005__metric=1_000000"
    )
    assert save_calls[0][1]["storage_writer"] == "writer:step=000005__metric=1_000000"
    assert (
        save_calls[0][1]["async_checkpointer_type"]
        is async_checkpoint.AsyncCheckpointerType.PROCESS
    )
    assert save_calls[0][1]["async_stager"] is stagers[0]
    assert callback.staging_future is staging_future
    assert callback._save_future is upload_future


def test_s3_async_cleanup_queues_checkpoint_prefix_with_storage_options(
    monkeypatch: pytest.MonkeyPatch,
):
    storage_options = {"region": "auto", "endpoint_url": "https://r2.example.test"}
    callback = attach_callback(
        AsyncCheckpointCallback(
            checkpoint_config(
                async_mode="async",
                keep_top_k=2,
                storage_options=storage_options,
            )
        )
    )
    storage = FakeCheckpointStorage(
        callback.root_dir,
        exists=True,
        checkpoints=[
            Checkpoint(step=5, metric=0.5),
            Checkpoint(step=4, metric=1.0),
            Checkpoint(step=3, metric=2.0),
        ],
    )
    queued = []

    monkeypatch.setattr(callback, "_checkpoint_storage", storage)
    callback.purge_queue = SimpleNamespace(put=lambda item: queued.append(item))

    callback._cleanup_checkpoints()

    assert queued == [
        (
            "s3://checkpoints/project/group/experiment/checkpoints/"
            "step=000003__metric=2_000000",
            storage_options,
        )
    ]


def test_s3_purge_thread_deletes_checkpoint_prefix_with_storage_options(
    monkeypatch: pytest.MonkeyPatch,
):
    purge_queue = queue.Queue()
    storage_options = {"region": "auto", "endpoint_url": "https://r2.example.test"}
    calls = []

    monkeypatch.setattr(
        async_checkpoint,
        "delete_s3_checkpoint_prefix",
        lambda path, options: calls.append((path, options)),
    )

    purge_queue.put(("s3://bucket/run/checkpoints/step=000003__metric=2_000000", storage_options))
    purge_queue.put(async_checkpoint.Terminate())

    async_checkpoint._purge_thread(purge_queue)

    assert calls == [
        (
            "s3://bucket/run/checkpoints/step=000003__metric=2_000000",
            storage_options,
        )
    ]


def test_s3_current_checkpoint_supports_integer_resume_mode(
    monkeypatch: pytest.MonkeyPatch,
):
    callback = attach_callback(
        CheckpointCallback(checkpoint_config(resume_mode=5))
    )
    storage = FakeCheckpointStorage(
        callback.root_dir,
        exists=True,
        checkpoints=[
            Checkpoint(step=7, metric=1.0),
            Checkpoint(step=5, metric=2.0),
        ],
    )

    monkeypatch.setattr(callback, "_checkpoint_storage", storage)

    checkpoint = callback._current_checkpoint()

    assert checkpoint == Checkpoint(step=5, metric=2.0)
    assert storage.find_modes == ["last"]


def test_s3_current_checkpoint_returns_first_checkpoint_for_resume_mode(
    monkeypatch: pytest.MonkeyPatch,
):
    callback = attach_callback(CheckpointCallback(checkpoint_config(resume_mode="last")))
    storage = FakeCheckpointStorage(
        callback.root_dir,
        exists=True,
        checkpoints=[
            Checkpoint(step=7, metric=1.0),
            Checkpoint(step=5, metric=2.0),
        ],
    )

    monkeypatch.setattr(callback, "_checkpoint_storage", storage)

    checkpoint = callback._current_checkpoint()

    assert checkpoint == Checkpoint(step=7, metric=1.0)
    assert storage.find_modes == ["last"]


class FakeCheckpointStorage:
    def __init__(self, root_dir, *, exists, checkpoints=None):
        self.root_dir = root_dir
        self._exists = exists
        self.checkpoints = checkpoints or []
        self.find_modes = []

    def checkpoint_path(self, checkpoint_id):
        return checkpoint_s3.join_s3_uri(self.root_dir, checkpoint_id)

    def exists(self, checkpoint_id):
        return self._exists

    def writer(self, checkpoint_id):
        return f"writer:{checkpoint_id}"

    def reader(self, checkpoint_id):
        return f"reader:{checkpoint_id}"

    def find_checkpoints(self, mode):
        self.find_modes.append(mode)
        return self.checkpoints


class FakeS3FileSystem:
    def __init__(self, keys):
        self._client = FakeS3Client(keys)
        self.deleted = []

    def rm_file(self, path):
        self.deleted.append(path)


class FakeS3Client:
    def __init__(self, keys):
        self.keys = keys

    def list_objects(self, bucket, prefix, delimiter="", max_keys=1000):
        del delimiter, max_keys
        assert bucket == "bucket"
        return [
            SimpleNamespace(
                object_info=[
                    SimpleNamespace(key=key)
                    for key in self.keys
                    if key.startswith(prefix)
                ]
            )
        ]
