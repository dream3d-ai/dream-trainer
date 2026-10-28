import contextlib
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from dream_trainer.callbacks.callback import CallbackCollection
from dream_trainer.callbacks.loggers import LoggerCallback, LRLoggerCallback
from dream_trainer.trainer.base import BaseTrainer


def make_trainer(callback, global_step=0):
    trainer = SimpleNamespace(
        global_step=global_step,
        local_batches=0,
        current_epoch=0,
        is_accumulating_gradients=False,
        world=SimpleNamespace(
            device=torch.device("cpu"),
            is_global_zero=True,
            get_mesh=lambda name: None,
        ),
        logs=[],
    )
    trainer.log_dict = lambda logs: trainer.logs.append(dict(logs))
    callback.trainer = trainer
    return trainer


def microbatch(callback, result, accumulation=1, batch_idx=None):
    trainer = callback.trainer
    trainer.is_accumulating_gradients = (trainer.local_batches + 1) % accumulation != 0
    callback.post_train_step(result, trainer.local_batches if batch_idx is None else batch_idx)
    if not trainer.is_accumulating_gradients:
        trainer.global_step += 1
    trainer.local_batches += 1


def test_default_logs_full_update_mean_and_does_not_dilute_sparse_grad_norm():
    callback = LoggerCallback()
    trainer = make_trainer(callback)
    value = torch.tensor(0.0, dtype=torch.bfloat16, requires_grad=True)
    for index in range(128):
        with torch.no_grad():
            value.fill_(index)
        result = {"loss": value, "media": torch.ones(2), "text": "ignored"}
        if index == 127:
            result["grad_norm"] = 3.0
        microbatch(callback, result, accumulation=128)
        assert len(trainer.logs) == (index == 127)
    assert trainer.logs == [{"loss": 63.5, "grad_norm": 3.0, "trainer/global_step": 1}]
    callback.post_train_epoch(result)
    assert len(trainer.logs) == 1  # No duplicate last-microbatch/epoch loss.


def test_update_interval_resume_numbering_and_epoch_tail():
    callback = LoggerCallback(log_every_n_train_steps=2)
    trainer = make_trainer(callback, global_step=10)
    for value in (1.0, 3.0, 5.0, 7.0, 9.0, 11.0):
        microbatch(callback, {"loss": value}, accumulation=2)
    assert trainer.logs == [{"loss": 4.0, "trainer/global_step": 12}]
    callback.post_train_epoch({"loss": 11.0})
    assert trainer.logs[-1] == {"loss": 10.0, "trainer/global_step": 13}


def test_unfinished_update_is_not_logged_and_can_cross_epoch_boundary():
    callback = LoggerCallback()
    trainer = make_trainer(callback)
    microbatch(callback, {"loss": 2.0}, accumulation=2)
    callback.post_train_epoch({"loss": 2.0})
    assert trainer.logs == []
    microbatch(callback, {"loss": 6.0}, accumulation=2, batch_idx=0)
    assert trainer.logs == [{"loss": 4.0, "trainer/global_step": 1}]


def test_single_microbatch_updates_and_legacy_interval():
    with pytest.warns(FutureWarning, match="completed gradient updates"):
        callback = LoggerCallback(log_every_n_train_batches=2)
    trainer = make_trainer(callback)
    microbatch(callback, {"loss": 2})
    microbatch(callback, {"loss": 4})
    assert trainer.logs == [{"loss": 3.0, "trainer/global_step": 2}]
    with pytest.raises(ValueError, match="positive"):
        LoggerCallback(log_every_n_train_steps=0)


def test_validation_still_logs_on_rank_zero_only():
    callback = LoggerCallback()
    trainer = make_trainer(callback)
    callback.post_validation_step({"val/loss": 2.0, "media": torch.ones(2)}, 0)
    callback.post_validation_epoch({"metrics/mse": 3.0})
    assert trainer.logs == [{"val/loss": 2.0}, {"metrics/mse": 3.0}]
    trainer.world.is_global_zero = False
    callback.post_validation_step({"val/loss": 100.0}, 1)
    callback.post_validation_epoch({"metrics/mse": 100.0})
    assert len(trainer.logs) == 2


def test_real_training_loop_logs_after_both_optimizers_update():
    class Trainer:
        is_accumulating_gradients = BaseTrainer.is_accumulating_gradients
        perform_training_epoch = BaseTrainer.perform_training_epoch

    trainer = Trainer()
    trainer.global_step = trainer.local_batches = trainer._local_step = 0
    trainer.current_epoch = 0
    trainer._num_train_batches = 4
    trainer._num_gradient_accumulation_steps = 2
    trainer._train_pg_timeout_reduced = True
    trainer.train_dataloader = [torch.tensor(x) for x in (1.0, 3.0, 5.0, 7.0)]
    trainer.training_parameters = SimpleNamespace(val_every_n_steps=100)
    trainer.world = SimpleNamespace(
        device=torch.device("cpu"),
        is_global_zero=True,
        get_mesh=lambda name: None,
        barrier=lambda: None,
    )
    trainer.train = lambda: None
    trainer.pre_train_step = lambda batch, batch_idx: batch
    trainer.train_context = contextlib.nullcontext
    trainer.logs = []
    updates = []
    trainer.log_dict = lambda logs: trainer.logs.append((dict(logs), list(updates)))
    parameters = [torch.nn.Parameter(torch.tensor(1.0)) for _ in range(2)]
    optimizers = {
        name: torch.optim.SGD([parameter], lr=rate)
        for name, parameter, rate in zip(("generator", "discriminator"), parameters, (0.1, 0.2))
    }
    trainer.named_optimizers = lambda: optimizers
    schedulers = [torch.optim.lr_scheduler.StepLR(o, 1, gamma=0.5) for o in optimizers.values()]
    trainer.callbacks = CallbackCollection([LoggerCallback(), LRLoggerCallback()])
    for callback in trainer.callbacks:
        callback.trainer = trainer

    def training_step(batch, batch_idx):
        for parameter in parameters:
            (parameter * batch / 2).backward()
        if not trainer.is_accumulating_gradients:
            for (name, optimizer), scheduler in zip(optimizers.items(), schedulers):
                optimizer.step()
                updates.append((name, trainer.global_step + 1))
                trainer.callbacks.post_optimizer_step(None, optimizer)
                optimizer.zero_grad()
                scheduler.step()
        return {"loss": batch}

    trainer.training_step = training_step
    trainer.perform_training_epoch()
    assert trainer.global_step == 2
    loss_logs = [(logs, updates) for logs, updates in trainer.logs if "loss" in logs]
    assert [logs for logs, _ in loss_logs] == [
        {"loss": 2.0, "trainer/global_step": 1},
        {"loss": 6.0, "trainer/global_step": 2},
    ]
    assert [len(updates) for _, updates in loss_logs] == [2, 4]
    lr_logs = [logs for logs, _ in trainer.logs if "learning_rate/generator" in logs]
    assert lr_logs == [
        {
            "learning_rate/generator": 0.1,
            "learning_rate/discriminator": 0.2,
            "trainer/global_step": 1,
        },
        {
            "learning_rate/generator": 0.05,
            "learning_rate/discriminator": 0.1,
            "trainer/global_step": 2,
        },
    ]


def _distributed_logging_worker(rank, rendezvous):
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2)
    try:
        callback = LoggerCallback()
        trainer = make_trainer(callback)
        trainer.world.is_global_zero = rank == 0
        trainer.world.get_mesh = lambda name: SimpleNamespace(
            size=lambda: 2, get_group=lambda: dist.group.WORLD
        )
        collection = CallbackCollection([callback])
        for index in range(2):
            trainer.is_accumulating_gradients = index == 0
            result = {"loss": float(1 + index * 2 + rank * 4)}
            if rank == 1:
                result["rank_one_only"] = float(10 + index * 2)
                result["conditional"] = float(10 + index * 2)
            elif index == 1:
                result["conditional"] = 100.0
            if index == 1:
                result["grad_norm"] = float(2 + rank * 2)
            collection.post_train_step(result, index)
        if rank == 0:
            assert trainer.logs[0].pop("conditional") == pytest.approx(122 / 3)
            assert trainer.logs == [
                {"loss": 4.0, "rank_one_only": 11.0, "grad_norm": 3.0, "trainer/global_step": 1}
            ]
        else:
            assert trainer.logs == []
    finally:
        dist.destroy_process_group()


def test_data_parallel_reduction_includes_other_ranks_and_sparse_metrics(tmp_path):
    mp.spawn(
        _distributed_logging_worker,
        args=(f"file://{tmp_path / 'rendezvous'}",),
        nprocs=2,
        join=True,
    )
