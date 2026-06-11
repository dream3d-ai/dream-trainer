from pathlib import Path
from types import SimpleNamespace

from dream_trainer.callbacks.checkpoint import partial
from dream_trainer.callbacks.checkpoint.partial import LoadPartialCheckpointCallback
from dream_trainer.callbacks.checkpoint.types import Checkpoint


class TrainerWithExtraState:
    def __init__(self):
        self.world = SimpleNamespace(barrier=lambda: None)
        self.callbacks = RecordingCallbacks()
        self.loaded_state_dict = None
        self.load_kwargs = None

    def state_dict(self):
        raise AssertionError("partial checkpoint loading should not inspect full trainer state")

    def model_state_dict(self):
        return {"model": {"weight": "target"}}

    def load_state_dict(self, state_dict, **kwargs):
        self.loaded_state_dict = state_dict
        self.load_kwargs = kwargs
        loaded_callback_names = set(state_dict.get("trainer", {}).get("callbacks", {}))
        self.callbacks.post_load_state_dict(loaded_callback_names)


class RecordingCallbacks(dict):
    def __init__(self):
        super().__init__()
        self.post_load_state_dict_calls = []

    def post_load_state_dict(self, loaded_callback_names):
        self.post_load_state_dict_calls.append(loaded_callback_names)


def test_partial_checkpoint_loads_model_state_only(monkeypatch, tmp_path: Path):
    checkpoint = Checkpoint(step=1, metric=0.0)
    callback = LoadPartialCheckpointCallback(tmp_path)
    trainer = TrainerWithExtraState()
    trainer.callbacks[callback.__class__.__name__] = callback
    callback.trainer = trainer
    callback.pg = object()
    process_group = callback.pg

    load_calls = []

    def load(state_dict, **kwargs):
        load_calls.append((state_dict, kwargs))
        state_dict["models"]["model"]["weight"] = "loaded"

    monkeypatch.setattr(partial, "find_current_checkpoint", lambda path, mode: checkpoint)
    monkeypatch.setattr(partial.dcp.state_dict_loader, "load", load)

    callback.pre_fit()

    expected_state_dict = {"models": {"model": {"weight": "loaded"}}}
    assert load_calls[0][0] == expected_state_dict
    assert load_calls[0][1]["checkpoint_id"] == str(tmp_path / checkpoint.checkpoint_id)
    assert load_calls[0][1]["process_group"] is process_group
    assert trainer.loaded_state_dict == expected_state_dict
    assert trainer.load_kwargs == {"strict": False, "resume_data": False}
    assert trainer.callbacks.post_load_state_dict_calls == [set()]
    assert callback.__class__.__name__ not in trainer.callbacks


def test_partial_checkpoint_loads_s3_checkpoint_with_storage_reader(monkeypatch):
    checkpoint = Checkpoint(step=3, metric=0.5)
    storage_options = {"region": "auto", "endpoint_url": "https://r2.example.test"}
    storage_calls = []
    trainer = TrainerWithExtraState()
    callback = LoadPartialCheckpointCallback(
        "s3://bucket/run/checkpoints",
        storage_options=storage_options,
    )
    trainer.callbacks[callback.__class__.__name__] = callback
    callback.trainer = trainer
    callback.pg = object()
    process_group = callback.pg
    load_calls = []

    monkeypatch.setattr(
        partial,
        "S3CheckpointStorage",
        lambda path, options: storage_calls.append((path, options))
        or FakeS3CheckpointStorage(
            path,
            options,
            checkpoints=[checkpoint, Checkpoint(step=2, metric=1.0)],
        ),
        raising=False,
    )

    def load(state_dict, **kwargs):
        load_calls.append((state_dict, kwargs))
        state_dict["models"]["model"]["weight"] = "loaded"

    monkeypatch.setattr(partial.dcp.state_dict_loader, "load", load)

    callback.pre_fit()

    assert storage_calls == [("s3://bucket/run/checkpoints", storage_options)]
    assert load_calls[0][1]["checkpoint_id"] == (
        "s3://bucket/run/checkpoints/step=000003__metric=0_500000"
    )
    assert load_calls[0][1]["storage_reader"] == "reader:step=000003__metric=0_500000"
    assert load_calls[0][1]["process_group"] is process_group
    assert trainer.loaded_state_dict == {"models": {"model": {"weight": "loaded"}}}
    assert trainer.callbacks.post_load_state_dict_calls == [set()]
    assert callback.__class__.__name__ not in trainer.callbacks


class FakeS3CheckpointStorage:
    def __init__(self, root_dir, storage_options, checkpoints):
        self.root_dir = root_dir
        self.storage_options = storage_options
        self.checkpoints = checkpoints
        self.find_modes = []

    def find_checkpoints(self, mode):
        self.find_modes.append(mode)
        return self.checkpoints

    def checkpoint_path(self, checkpoint_id):
        return f"{self.root_dir}/{checkpoint_id}"

    def reader(self, checkpoint_id):
        return f"reader:{checkpoint_id}"
