import contextlib
from types import SimpleNamespace

import pytest
import torch

from dream_trainer.callbacks import EarlyStoppingCallback
from dream_trainer.trainer.base import BaseTrainer


def test_early_stopping_tracks_metric_and_restores_state():
    trainer = SimpleNamespace(
        global_step=1,
        current_epoch=0,
        is_sanity_validation=True,
        stop_requested=False,
    )
    callback = EarlyStoppingCallback(monitor="metrics/bpb", patience=0)
    callback.trainer = trainer
    callback.post_validation_epoch({"metrics/bpb": torch.tensor(4.0)})
    assert callback.history == []

    trainer.is_sanity_validation = False
    callback.post_validation_epoch({"metrics/bpb": torch.tensor(1.0)})
    restored = EarlyStoppingCallback(monitor="metrics/bpb", patience=0)
    restored.trainer = trainer
    restored.load_state_dict(callback.state_dict())
    trainer.current_epoch = 1
    restored.post_validation_epoch({"metrics/bpb": torch.tensor(1.1)})

    assert restored.best_metric == 1.0
    assert restored.epochs_without_improvement == 1
    assert len(restored.history) == 2
    assert trainer.stop_requested

    with pytest.raises(KeyError, match="metrics/bpb"):
        restored.post_validation_epoch({})


def test_stop_request_exits_training_epoch_without_short_loader_error():
    trainer = SimpleNamespace()
    trainer._num_train_batches = 3
    trainer._num_gradient_accumulation_steps = 1
    trainer.train_dataloader = [{"x": torch.tensor(1.0)} for _ in range(3)]
    trainer.training_parameters = SimpleNamespace(val_every_n_steps=1)
    trainer.device_parameters = SimpleNamespace(comm=SimpleNamespace(train_timeout_seconds=10))
    trainer.world = SimpleNamespace(
        device=torch.device("cpu"),
        barrier=lambda: None,
        train_context=lambda: contextlib.nullcontext(),
        set_pg_timeouts=lambda timeout: None,
    )
    finished = []
    trainer.callbacks = SimpleNamespace(
        pre_train_epoch=lambda: None,
        pre_train_step=lambda batch, batch_idx: None,
        train_context=lambda: [],
        post_train_step=lambda result, batch_idx: None,
        post_train_epoch=lambda result: finished.append(result),
    )
    trainer.training_step = lambda batch, batch_idx: {"loss": torch.tensor(1.0)}
    trainer.pre_train_step = lambda batch, batch_idx: batch
    trainer.train_context = lambda: contextlib.nullcontext()
    trainer.is_accumulating_gradients = False
    trainer.train = lambda: None
    trainer.perform_validation_epoch = lambda: setattr(trainer, "stop_requested", True)
    trainer.local_batches = 0
    trainer.global_step = 0
    trainer._local_step = 0
    trainer._train_pg_timeout_reduced = False
    trainer.stop_requested = False

    BaseTrainer.perform_training_epoch(trainer)

    assert trainer.global_step == 1
    assert trainer.local_batches == 1
    assert len(finished) == 1


def test_stop_request_runs_post_fit():
    class NoopCallbacks:
        def __getattr__(self, name):
            return lambda *args, **kwargs: None

    trainer = SimpleNamespace(
        callbacks=NoopCallbacks(),
        world=SimpleNamespace(launch=lambda: None, barrier=lambda: None),
        seed=0,
        configure=lambda: None,
        setup=lambda: None,
        _setup_trainer_metadata=lambda: None,
        post_setup=lambda: None,
        pre_fit=lambda: None,
        perform_sanity_validation_steps=lambda: None,
        training_parameters=SimpleNamespace(n_epochs=5),
        current_epoch=0,
        stop_requested=False,
    )
    events = []
    trainer.perform_training_epoch = lambda: setattr(trainer, "stop_requested", True)
    trainer.post_fit = lambda: events.append("post_fit")

    BaseTrainer._fit(trainer)

    assert trainer.current_epoch == 1
    assert events == ["post_fit"]
