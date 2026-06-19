---
title: "Tutorial 1 — Meet Meteor"
---

# Tutorial 1 — Meet Meteor

<small>🎓 Tutorial · ~20 min · 1 GPU</small>

This is the first page of a five-part series. We're going to build and grow one project — **Meteor**, a tiny GPT-style language model — from a single-file single-GPU trainer into a production-shaped repo that runs across many nodes. Each tutorial starts with *where we left off*, introduces *one new problem*, and ends with *the new thing we have now*.

If you just want a reference, start with [Trainer Guide](../trainer-guide.md). If you want to build intuition by watching a real project grow, start here.

## Where We're Starting

Nothing yet. A clean directory and one file:

```text
meteor/
  train.py
```

Our goals for v0 are the smallest honest version of a trainer:

1. Meteor v0 is **~125M parameters** (GPT-2 Small scale).
2. It runs on **one GPU** against synthetic data.
3. It logs loss, saves a checkpoint, shows a progress bar.

That's the whole target. We don't scale, parallelize, or tune anything. When v0 is running, v1 gets interesting.

## The Model, Plain PyTorch

Before Dream Trainer shows up at all, write the model. It's an ordinary `nn.Module`. No hooks, no mixins, no magic:

```python
# train.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class Meteor(nn.Module):
    def __init__(self, *, vocab_size: int, d_model: int, n_heads: int, n_layers: int, seq_len: int):
        super().__init__()
        self.tok_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(seq_len, d_model)
        self.layers = nn.ModuleList(
            [nn.TransformerEncoderLayer(d_model, n_heads, 4 * d_model, batch_first=True, norm_first=True)
             for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        B, T = tokens.shape
        pos = torch.arange(T, device=tokens.device)
        x = self.tok_embed(tokens) + self.pos_embed(pos)
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=tokens.device)
        for layer in self.layers:
            x = layer(x, src_mask=mask, is_causal=True)
        return self.head(self.norm(x))

    def init_weights(self):
        nn.init.normal_(self.tok_embed.weight, std=0.02)
        nn.init.normal_(self.pos_embed.weight, std=0.02)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
```

Two things about `init_weights`. First, it lives on the model, not the trainer — that's where weight initialization usually belongs. Second, we'll call it from the trainer's `init_weights` hook, and the reason matters.

## The Config

A dataclass holds the experiment's choices. No CUDA, no tensors, no side effects:

```python
from dataclasses import dataclass, field

from torch.optim import AdamW

from dream_trainer import DreamTrainerConfig
from dream_trainer.configs import DeviceParameters, TrainingParameters, WandbLoggingParameters


@dataclass(kw_only=True)
class MeteorConfig(DreamTrainerConfig):
    # Model
    vocab_size: int = 50_257
    d_model: int = 768
    n_heads: int = 12
    n_layers: int = 12
    seq_len: int = 1024

    # Optim
    learning_rate: float = 3e-4
    batch_size: int = 8


def v0_config() -> MeteorConfig:
    return MeteorConfig(
        project="meteor",
        group="v0",
        device_parameters=DeviceParameters.SINGLE_DEVICE(compile_model=False),
        training_parameters=TrainingParameters(
            n_epochs=1,
            train_steps_per_epoch=200,
            val_steps_per_epoch=20,
            num_sanity_val_steps=2,
            gradient_clip_val=1.0,
        ),
        logging_parameters=WandbLoggingParameters(enabled=False),
    )
```

We wrap the config values in a **named factory** (`v0_config`) instead of hard-coding a single instance. This pattern pays off in Tutorial 2 the moment we want a `ddp_config` alongside it.

## The Trainer

Now the Dream Trainer part. We implement the seven hooks from [Trainer Guide](../trainer-guide.md) and nothing else:

```python
from typing import Any, Iterable

from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict
from torch.utils.data import DataLoader, TensorDataset

from dream_trainer import DreamTrainer


class MeteorTrainer(DreamTrainer):
    config: MeteorConfig

    def configure_models(self):
        self.model = Meteor(
            vocab_size=self.config.vocab_size,
            d_model=self.config.d_model,
            n_heads=self.config.n_heads,
            n_layers=self.config.n_layers,
            seq_len=self.config.seq_len,
        )

    def init_weights(self):
        self.model.init_weights()

    def model_state_dict(self, **_: Any) -> dict[str, Any]:
        return {"model": get_model_state_dict(self.model, options=StateDictOptions())}

    def configure_optimizers(self):
        self.optimizer = AdamW(self.model.parameters(), lr=self.config.learning_rate)
        return {self.model: self.optimizer}

    def configure_dataloaders(self) -> tuple[Iterable, Iterable]:
        def fake_loader():
            tokens = torch.randint(0, self.config.vocab_size, (1024, self.config.seq_len))
            return DataLoader(TensorDataset(tokens), batch_size=self.config.batch_size, shuffle=True)
        return fake_loader(), fake_loader()

    def training_step(self, batch, batch_idx: int) -> dict[str, Any]:
        (tokens,) = batch
        logits = self.model(tokens[:, :-1])
        loss = F.cross_entropy(logits.reshape(-1, self.config.vocab_size), tokens[:, 1:].reshape(-1))
        self.backward(loss)

        logs: dict[str, Any] = {"train/loss": loss}
        if not self.is_accumulating_gradients:
            logs["train/grad_norm"] = self.step(self.optimizer)
        return logs

    @torch.no_grad()
    def validation_step(self, batch, batch_idx: int) -> dict[str, Any]:
        (tokens,) = batch
        logits = self.model(tokens[:, :-1])
        loss = F.cross_entropy(logits.reshape(-1, self.config.vocab_size), tokens[:, 1:].reshape(-1))
        return {"val/loss": loss}
```

!!! danger "`init_weights` runs after materialization, not inside `configure_models`"
    Under Dream Trainer's lifecycle, `configure_models` creates the module on the **meta device** — the tensors have shapes but no storage. Only after materialization does `init_weights` run on real tensors. If you move the `nn.init` calls into `configure_models`, they silently initialize meta tensors and the results are thrown away. This is the single most common first bug.

!!! tip "Use `self.backward` and `self.step`, not the PyTorch equivalents"
    `self.backward(loss)` scales for gradient accumulation. `self.step(self.optimizer)` handles clipping, callbacks, scheduler stepping, and zeroing grads. Calling `loss.backward()` or `optimizer.step()` directly breaks all of that silently. See [Trainer Guide](../trainer-guide.md#6-training_step-one-step-of-the-loop).

## The Entrypoint

```python
from dream_trainer import callbacks
from dream_trainer.configs import CheckpointParameters
from dream_trainer.utils.entrypoint import entrypoint


@entrypoint
def main():
    config = v0_config()
    config.callbacks = callbacks.CallbackCollection([
        callbacks.LoggerCallback(log_every_n_train_batches=10),
        callbacks.ProgressBar(metric="train/loss"),
        callbacks.CheckpointCallback(CheckpointParameters(
            enable=True,
            root_dir="/tmp/meteor/v0",
            monitor="val/loss",
            resume_mode="min",
            checkpoint_every_n_val_epochs=1,
        )),
    ])
    MeteorTrainer(config).fit()


if __name__ == "__main__":
    main()
```

## Running It

```bash
CUDA_VISIBLE_DEVICES=0 python train.py
```

You should see a progress bar, a decreasing `train/loss`, and — after the first validation epoch — a checkpoint in `/tmp/meteor/v0/`.

!!! info "What just happened"
    `entrypoint` detected one visible GPU and launched locally. Dream Trainer built the distributed world (of size 1), created the mesh (all dimensions equal 1), ran `configure_models` under meta-device, materialized the model, called `init_weights`, built the optimizer, then started calling `training_step` and `validation_step` in the right order. You wrote seven methods; Dream Trainer ran ten phases around them.

## Sanity Checks Before Moving On

Before Tutorial 2, confirm:

- [ ] The run completes end-to-end (no OOM on a single GPU with `d_model=768`).
- [ ] `train/loss` decreases — even on synthetic data the transformer should overfit fast.
- [ ] A checkpoint exists under `/tmp/meteor/v0/`.
- [ ] Killing and restarting the script resumes from the checkpoint.

If any of these fails, fix it here. Every problem compounds when we add parallelism.

## The New Thing We Have Now

A working **Meteor v0**:

- A `MeteorConfig` with a named factory.
- A `MeteorTrainer` that implements exactly the seven hooks every trainer needs.
- A callback stack of three items: logger, progress bar, checkpoint.
- Synthetic data, single GPU, reproducible.

In [Tutorial 2 — Scaling Meteor to Multiple GPUs](multi-gpu.md) we hit the first real problem: Meteor trains fine, but we want **4× more data throughput**. Time to add data parallelism.
