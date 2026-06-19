---
title: "Tutorial 4 — Shaping Meteor for Production"
---

# Tutorial 4 — Shaping Meteor for Production

<small>🎓 Tutorial · ~30 min · any number of GPUs</small>

## Where We Left Off

[Tutorial 3](fsdp.md) gave us **Meteor v2**: a 1.3B-parameter FSDP-sharded trainer, compiled, loss decreasing, checkpoints that survive restarts. It's correct. It's fast. It all lives in one `train.py`.

One file was fine for teaching the lifecycle. It is not fine for a run you actually care about.

## The Problem

Four things we tolerated at v2 stop working the moment you cross the "research prototype" line:

1. **The file is becoming a god object.** Model, data, config, trainer, entrypoint — all in one place. Reviewers can't tell what changed between runs.
2. **Checkpoint writes block training.** At 1.3B params the save takes tens of seconds. Every `checkpoint_every_n_val_epochs` hit freezes the rank-zero training loop.
3. **Logging goes to stdout only.** There's no dashboard, no run comparison, no history. When a run regresses, you have nothing to diff against.
4. **There's no run hygiene.** No LR logging, no model-watch, no structural summary at the start. The first five minutes of debugging any bad run is "wait, what did we even configure?"

None of these is a rewrite. Each is a specific move. Meteor v3 is the same trainer as v2 with these four problems fixed.

## The Split — One File Becomes Five

The shape of a Meteor repo at v3:

```text
meteor/
  config.py      # MeteorConfig + named factories
  model.py       # Meteor nn.Module
  data.py        # dataset + dataloader factory
  train.py       # MeteorTrainer + entrypoint
  metrics.py     # (lands in Tutorial 5)
```

The boundary is the same one [Configuration](../configuration.md) draws: **configs describe, trainer performs**. Everything that's pure-data goes to `config.py`; everything that does I/O or CUDA work goes where it belongs.

!!! tip "Split when the file hurts, not when a style guide says"
    One file through v2 was the right call — you could see the whole trainer in a single scroll. Splitting earlier would have made the lifecycle harder to teach. Split when the file is actually in your way: when `git diff` is too wide to read, when two people can't edit it without conflicts, when grep returns too many false positives.

### `config.py`

```python
from dataclasses import dataclass, field

from dream_trainer import DreamTrainerConfig
from dream_trainer.configs import (
    CheckpointParameters,
    DeviceParameters,
    TrainingParameters,
    WandbLoggingParameters,
)


@dataclass(kw_only=True)
class ModelConfig:
    vocab_size: int = 50_257
    d_model: int = 2048
    n_heads: int = 16
    n_layers: int = 24
    seq_len: int = 2048


@dataclass(kw_only=True)
class DataConfig:
    path: str
    batch_size: int
    num_workers: int = 8


@dataclass(kw_only=True)
class OptimizerConfig:
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    warmup_steps: int = 1_000


@dataclass(kw_only=True)
class MeteorConfig(DreamTrainerConfig):
    model: ModelConfig
    train_data: DataConfig
    val_data: DataConfig
    optimizer: OptimizerConfig


def v3_config() -> MeteorConfig:
    return MeteorConfig(
        project="meteor",
        group="v3",
        model=ModelConfig(),
        train_data=DataConfig(path="s3://…/train", batch_size=4),
        val_data=DataConfig(path="s3://…/val", batch_size=4),
        optimizer=OptimizerConfig(),
        device_parameters=DeviceParameters.FSDP(compile_model=True),
        training_parameters=TrainingParameters(
            n_epochs=1,
            train_steps_per_epoch=100_000,
            val_steps_per_epoch=500,
            num_sanity_val_steps=2,
            gradient_clip_val=1.0,
        ),
        checkpoint_parameters=CheckpointParameters(
            enable=True,
            root_dir="s3://meteor-checkpoints/v3",
            monitor="val/loss",
            resume_mode="min",
            checkpoint_every_n_val_epochs=1,
            keep_top_k=3,
        ),
        logging_parameters=WandbLoggingParameters(
            enabled=True,
            entity="your-org",
            project="meteor",
            tags=["v3", "fsdp", "1.3B"],
        ),
    )


def debug_config() -> MeteorConfig:
    return dataclasses.replace(
        v3_config(),
        group="debug",
        device_parameters=DeviceParameters.SINGLE_DEVICE(compile_model=False),
        training_parameters=TrainingParameters(
            n_epochs=1,
            train_steps_per_epoch=20,
            val_steps_per_epoch=4,
            num_sanity_val_steps=1,
            gradient_clip_val=1.0,
        ),
        logging_parameters=WandbLoggingParameters(enabled=False),
    )
```

Two factories now instead of one. Production runs on `v3_config()`; you reach for `debug_config()` the instant something looks wrong.

### `data.py`

```python
import torch
from torch.utils.data import DataLoader, DistributedSampler

from .config import DataConfig


class TokenStream(torch.utils.data.Dataset):
    def __init__(self, path: str, seq_len: int):
        ...   # your tokenizer + shard reader here


def build_loader(cfg: DataConfig, *, rank: int, world_size: int, shuffle: bool) -> DataLoader:
    dataset = TokenStream(cfg.path, seq_len=2048)
    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=shuffle,
    )
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        sampler=sampler,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
```

The key discipline: `build_loader` accepts `rank` and `world_size` explicitly. The trainer passes `self.world.dp_rank` and `self.world.dp_size` at call time. The data module doesn't know or care what kind of parallelism is active — the mesh resolves that.

### `train.py`

```python
import torch
import torch.nn.functional as F
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict
from torch.distributed.fsdp import fully_shard
from torch.optim import AdamW

from dream_trainer import DreamTrainer, callbacks
from dream_trainer.utils.entrypoint import entrypoint

from .config import MeteorConfig, v3_config
from .data import build_loader
from .model import Meteor


class MeteorTrainer(DreamTrainer):
    config: MeteorConfig

    def configure_models(self):
        self.model = Meteor(**vars(self.config.model))

    def init_weights(self):
        self.model.init_weights()

    def apply_compile(self):
        self.model.compile(mode="max-autotune-no-cudagraphs", dynamic=False)

    def apply_fully_shard(self, config):
        for layer in self.model.layers:
            fully_shard(layer, **config)
        fully_shard(self.model, **config)

    def model_state_dict(self, **_):
        return {"model": get_model_state_dict(self.model, options=StateDictOptions())}

    def configure_optimizers(self):
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.optimizer.learning_rate,
            weight_decay=self.config.optimizer.weight_decay,
        )
        return {self.model: self.optimizer}

    def configure_schedulers(self):
        scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=1e-3,
            total_iters=self.config.optimizer.warmup_steps,
        )
        return {self.optimizer: scheduler}

    def configure_dataloaders(self):
        kwargs = dict(rank=self.world.dp_rank, world_size=self.world.dp_size)
        return (
            build_loader(self.config.train_data, shuffle=True, **kwargs),
            build_loader(self.config.val_data, shuffle=False, **kwargs),
        )

    def training_step(self, batch, batch_idx):
        tokens = batch
        logits = self.model(tokens[:, :-1])
        loss = F.cross_entropy(
            logits.reshape(-1, self.config.model.vocab_size),
            tokens[:, 1:].reshape(-1),
        )
        self.backward(loss)
        logs = {"train/loss": loss}
        if not self.is_accumulating_gradients:
            logs["train/grad_norm"] = self.step(self.optimizer)
        return logs

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        tokens = batch
        logits = self.model(tokens[:, :-1])
        loss = F.cross_entropy(
            logits.reshape(-1, self.config.model.vocab_size),
            tokens[:, 1:].reshape(-1),
        )
        return {"val/loss": loss}


@entrypoint
def main():
    config = v3_config()
    config.callbacks = callbacks.CallbackCollection([
        callbacks.LoggerCallback(log_every_n_train_batches=20, code_dir="../"),
        callbacks.LRLoggerCallback(),
        callbacks.ProgressBar(metric="train/loss"),
        callbacks.AsyncCheckpointCallback(config.checkpoint_parameters),
        callbacks.ModelWatchCallback(),
        callbacks.TrainerSummary(),
        callbacks.ModelSummary(),
    ])
    MeteorTrainer(config).fit()


if __name__ == "__main__":
    main()
```

The trainer itself barely grew. The hook count is the same as v2. What changed is **what the hooks read from**: structured config objects instead of flat fields, a real data module instead of synthetic tensors, named factories instead of one `v2_config`.

## Why `AsyncCheckpointCallback` Now

`CheckpointCallback` writes synchronously on the main rank. That's fine when saves are quick. At 1.3B parameters under FSDP the save takes ~30s; every validation epoch you'd stall the whole world.

`AsyncCheckpointCallback` stages the DCP write to a background thread and returns to training immediately. It also handles background cleanup of evicted checkpoints (the ones past `keep_top_k`) without blocking.

!!! tip "Swap the callback, keep the config"
    Both callbacks take the same `CheckpointParameters`. Moving from `CheckpointCallback(checkpoint_parameters)` to `AsyncCheckpointCallback(checkpoint_parameters)` is a one-line change. If you ever need to debug a checkpoint correctness issue, swap back temporarily to rule out the async path.

## Why WandB, Why `ModelWatchCallback`

`LoggerCallback` with `WandbLoggingParameters(enabled=True)` routes every scalar return from `training_step` and `validation_step` to WandB automatically. `LRLoggerCallback` adds per-optimizer learning rate curves. `ModelWatchCallback` hooks `wandb.watch` so weight and gradient histograms show up alongside loss curves.

Together, the first time you see a regression you can open the dashboard and compare three runs side by side instead of tailing three log files.

## Running It

### Single node, four GPUs

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m meteor.train
```

### Multi-node with torchrun

```bash
torchrun \
  --nproc-per-node=8 \
  --nnodes=$NNODES \
  --node-rank=$RANK \
  --master-addr=$MASTER_ADDR \
  --master-port=$MASTER_PORT \
  -m meteor.train
```

`entrypoint` uses the torchrun environment when it exists and spawns locally when it doesn't — the same `train.py` runs in both.

## What Splitting Bought You

Concretely, on a fresh run with v3:

- A teammate can PR a new `ModelConfig` field without touching the trainer.
- `debug_config()` runs on one GPU for two minutes as a correctness smoke test before a long job.
- A 30s checkpoint save no longer pauses training.
- Three parallel runs show up side by side on WandB, tagged by `group`.
- `TrainerSummary` prints what `DeviceParameters` and mesh were used — the first sanity check when a run looks odd.

Nothing on this list is *new capability*. Each is "the same thing, without the operational drag."

## Sanity Checks Before Moving On

- [ ] `python -m meteor.train` (with `debug_config`) runs end-to-end on one GPU in a few minutes.
- [ ] `python -m meteor.train` with `v3_config` on your target world size runs a validation epoch and writes a checkpoint without stalling.
- [ ] WandB shows loss, LR, and gradient histograms.
- [ ] Killing mid-run and restarting resumes from the checkpoint — and the resumed run continues the same WandB run, not a new one.

## The New Thing We Have Now

**Meteor v3**: split config/train/data/model, async checkpointing, WandB logging, LR + model-watch + structure summaries. Same hooks as v2, production-shaped.

One thing is still missing, though. Meteor's validation is a single loss scalar. For anything bigger than a toy experiment you want perplexity, per-domain breakdowns, maybe a tokens-seen curriculum. That's custom metrics and a custom callback — reusable parts that attach to the trainer without changing it.

In [Tutorial 5 — Extending Meteor with Custom Components](custom-components.md) we add a metric collection and a curriculum-learning callback, and see how Dream Trainer's extension points keep the core trainer stable.
