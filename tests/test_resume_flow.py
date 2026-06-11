import contextlib
from types import SimpleNamespace

import pytest
import torch

from dream_trainer.trainer.base import BaseTrainer


class MinimalTrainer(BaseTrainer):
    def configure_dataloaders(self):
        raise NotImplementedError

    def configure_metrics(self):
        raise NotImplementedError

    def configure_models(self):
        raise NotImplementedError

    def configure_optimizers(self):
        raise NotImplementedError

    def init_weights(self):
        raise NotImplementedError

    def model_state_dict(self, **_):
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        raise NotImplementedError

    def validation_step(self, batch, batch_idx):
        raise NotImplementedError


def test_sanity_validation_is_skipped_after_resume():
    trainer = object.__new__(MinimalTrainer)
    trainer.global_step = 12
    trainer._num_sanity_val_steps = 2
    trainer._num_val_steps = 5
    trainer.is_sanity_validation = False
    calls = []

    trainer.perform_validation_epoch = lambda: calls.append("validation")

    BaseTrainer.perform_sanity_validation_steps(trainer)

    assert calls == []
    assert trainer._num_val_steps == 5
    assert trainer.is_sanity_validation is False


def test_sanity_validation_runs_from_scratch():
    trainer = object.__new__(MinimalTrainer)
    trainer.global_step = 0
    trainer._num_sanity_val_steps = 2
    trainer._num_val_steps = 5
    trainer.is_sanity_validation = False
    calls = []

    trainer.perform_validation_epoch = lambda: calls.append(
        SimpleNamespace(
            num_val_steps=trainer._num_val_steps,
            is_sanity_validation=trainer.is_sanity_validation,
        )
    )

    BaseTrainer.perform_sanity_validation_steps(trainer)

    assert calls == [SimpleNamespace(num_val_steps=2, is_sanity_validation=True)]
    assert trainer._num_val_steps == 5
    assert trainer.is_sanity_validation is False


def test_state_dict_does_not_checkpoint_gradient_accumulation_counter():
    trainer = object.__new__(MinimalTrainer)
    trainer.global_step = 12
    trainer.current_epoch = 3
    trainer.local_batches = 7
    trainer.callbacks = SimpleNamespace(state_dict=lambda: {})
    trainer.model_state_dict = lambda: {}
    trainer.named_optimizers = lambda: {}
    trainer.named_schedulers = lambda: {}
    trainer.named_rngs = lambda: {}
    trainer._train_dataloader = []
    trainer._val_dataloader = []

    state = BaseTrainer.state_dict(trainer)

    assert state["trainer"] == {
        "global_step": 12,
        "current_epoch": 3,
        "callbacks": {},
    }


def test_load_state_dict_keeps_fresh_gradient_accumulation_counter():
    trainer = object.__new__(MinimalTrainer)
    trainer.global_step = 0
    trainer.current_epoch = 0
    trainer.local_batches = 0
    trainer.callbacks = SimpleNamespace(
        load_state_dict=lambda _: None,
        post_load_state_dict=lambda _: None,
    )
    trainer.named_models = lambda: {}
    trainer.named_optimizers = lambda: {}
    trainer.named_schedulers = lambda: {}
    trainer.named_rngs = lambda: {}
    trainer._train_dataloader = []
    trainer._val_dataloader = []

    BaseTrainer.load_state_dict(
        trainer,
        {
            "trainer": {
                "global_step": 12,
                "current_epoch": 3,
                "local_batches": 7,
                "callbacks": {},
            },
        },
    )

    assert trainer.global_step == 12
    assert trainer.current_epoch == 3
    assert trainer.local_batches == 0


def test_rng_state_round_trips_through_checkpoint():
    from dream_trainer.trainer.mixins.setup.rngs import RNGSetupMixin

    class RNGTrainer(MinimalTrainer):
        def configure_rngs(self):
            self.val_noise_rng = torch.Generator(device="cpu").manual_seed(0)

    trainer = object.__new__(RNGTrainer)
    RNGSetupMixin._setup_rngs(trainer)
    assert list(trainer.named_rngs()) == ["val_noise_rng"]

    trainer.global_step = 1
    trainer.current_epoch = 0
    trainer.local_batches = 0
    trainer.callbacks = SimpleNamespace(
        state_dict=lambda: {},
        load_state_dict=lambda _: None,
        post_load_state_dict=lambda _: None,
    )
    trainer.model_state_dict = lambda: {}
    trainer.named_models = lambda: {}
    trainer.named_optimizers = lambda: {}
    trainer.named_schedulers = lambda: {}
    trainer._train_dataloader = []
    trainer._val_dataloader = []

    # Advance the generator past its initial state, then checkpoint
    torch.rand(8, generator=trainer.val_noise_rng)
    state = BaseTrainer.state_dict(trainer)
    expected = torch.rand(8, generator=trainer.val_noise_rng)

    # Advance further so resume actually has to rewind the stream
    torch.rand(8, generator=trainer.val_noise_rng)
    BaseTrainer.load_state_dict(trainer, state)

    assert torch.equal(torch.rand(8, generator=trainer.val_noise_rng), expected)


def test_training_epoch_raises_when_dataloader_ends_early():
    trainer = object.__new__(MinimalTrainer)
    trainer._num_train_batches = 2
    trainer._num_gradient_accumulation_steps = 1
    trainer._train_dataloader = [{"x": torch.tensor(1.0)}]
    trainer.training_parameters = SimpleNamespace(val_every_n_steps=100)
    trainer.world = SimpleNamespace(
        device=torch.device("cpu"),
        barrier=lambda: None,
        train_context=lambda: contextlib.nullcontext(),
        set_pg_timeouts=lambda timeout: None,
        world_mesh=None,
    )
    post_train_epoch_calls = []
    trainer.callbacks = SimpleNamespace(
        pre_train_epoch=lambda: None,
        pre_train_step=lambda batch, batch_idx: None,
        train_context=lambda: [],
        post_train_step=lambda result, batch_idx: None,
        post_train_epoch=lambda result: post_train_epoch_calls.append(result),
    )
    trainer.named_models = lambda: {}
    trainer.training_step = lambda batch, batch_idx: {"loss": torch.tensor(1.0)}
    trainer.pre_train_step = lambda batch, batch_idx: batch
    trainer.train = lambda: None
    trainer.perform_validation_epoch = lambda: None
    trainer.local_batches = 0
    trainer.global_step = 0
    trainer._local_step = 0

    with pytest.raises(RuntimeError, match="fewer training batches than expected"):
        BaseTrainer.perform_training_epoch(trainer)

    assert post_train_epoch_calls == []
