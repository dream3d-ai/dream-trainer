# Quick Start

!!! abstract "TL;DR"
    - Define a `DreamTrainerConfig` subclass with the knobs for your run.
    - Subclass `DreamTrainer` and implement 8 hooks: `configure_models`, `init_weights`, `model_state_dict`, `configure_optimizers`, `configure_dataloaders`, `configure_metrics`, `training_step`, `validation_step`.
    - Wrap `main()` with `@entrypoint` and call `trainer.fit()`.

This guide builds the smallest useful Dream Trainer training script: a synthetic regression dataset, a tiny MLP, metrics, callbacks, and the distributed `entrypoint` launcher. It follows the same shape a production trainer uses — define a config, define a trainer, implement the setup hooks, then call `trainer.fit()`.


## Install

Follow the full environment guidance in [Installation](installation.md). For the current package shape, install the metric and WandB extras even if this first run disables WandB logging:

=== "pip"

    ```bash
    pip install "dream-trainer[metrics,wandb]"
    ```

=== "uv"

    ```bash
    uv add "dream-trainer[metrics,wandb]"
    ```

Run the script with the single-device debug config:

```bash
CUDA_VISIBLE_DEVICES=0 python quickstart.py
```

!!! warning "One rank only"
    This quick start uses `DeviceParameters.SINGLE_DEVICE()`. Keep the launch to a single rank; the distributed story starts in [Tutorial 2 — Scaling Meteor](tutorials/multi-gpu.md).

## A complete `quickstart.py`

```python
from dataclasses import dataclass, field
from typing import Any, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict
from torch.utils.data import DataLoader, Dataset
from torchmetrics import MeanSquaredError, MetricCollection

from dream_trainer import DreamTrainer, DreamTrainerConfig, callbacks
from dream_trainer.configs import DeviceParameters, TrainingParameters, WandbLoggingParameters
from dream_trainer.utils.entrypoint import entrypoint


class RegressionDataset(Dataset):
    def __init__(self, *, n_samples: int, input_dim: int, seed: int):
        # Keep the dataset deterministic so the quick start behaves the same
        # each time you run it.
        generator = torch.Generator().manual_seed(seed)
        self.x = torch.randn(n_samples, input_dim, generator=generator)
        weights = torch.randn(input_dim, 1, generator=generator)
        bias = torch.randn(1, generator=generator)
        noise = 0.05 * torch.randn(n_samples, 1, generator=generator)
        self.y = self.x @ weights + bias + noise

    def __len__(self) -> int:
        # Dream Trainer uses dataloader length to infer epoch length when
        # train_steps_per_epoch or val_steps_per_epoch is not provided.
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        # Batches are dictionaries because training_step receives the batch
        # exactly as the dataloader yields it.
        return {"x": self.x[idx], "y": self.y[idx]}


@dataclass(kw_only=True)
class ToyTrainerConfig(DreamTrainerConfig):
    input_dim: int = 16
    hidden_dim: int = 64
    learning_rate: float = 1e-3
    batch_size: int = 32
    metrics: MetricCollection = field(
        default_factory=lambda: MetricCollection({"mse": MeanSquaredError()})
    )


class ToyTrainer(DreamTrainer):
    config: ToyTrainerConfig

    def configure_models(self) -> None:
        # Dream Trainer calls configure_models under a meta-device context.
        # Define module structure here, but do not load weights or rely on
        # allocated parameter storage yet.
        self.model = nn.Sequential(
            nn.Linear(self.config.input_dim, self.config.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.config.hidden_dim, 1),
        )

    def init_weights(self) -> None:
        # Models are materialized after configure_models. Initialize weights,
        # load pretrained checkpoints, or run custom init here because tensors
        # now have real storage on the training device.
        for module in self.model.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def model_state_dict(self, **_: Any) -> dict[str, Any]:
        # Checkpoint callbacks call trainer.state_dict(), which delegates model
        # state collection to this method. Use PyTorch DCP helpers so the same
        # code works for local and distributed model states.
        return {"model": get_model_state_dict(self.model, options=StateDictOptions())}

    def configure_optimizers(self) -> dict[nn.Module, torch.optim.Optimizer]:
        # Optimizers are created after model setup, so they see the materialized
        # parameters Dream Trainer will train.
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
        )
        # Returning the model-to-optimizer mapping lets Dream Trainer restore
        # optimizer state and associate schedulers with the right model.
        return {self.model: self.optimizer}

    def configure_dataloaders(self) -> tuple[Iterable, Iterable]:
        # Build dataloaders after the distributed world exists. In production,
        # pass self.world.dp_rank and self.world.dp_size into sharded dataloader
        # factories here.
        train_dataset = RegressionDataset(
            n_samples=2048,
            input_dim=self.config.input_dim,
            seed=0,
        )
        val_dataset = RegressionDataset(
            n_samples=512,
            input_dim=self.config.input_dim,
            seed=1,
        )
        return (
            DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True),
            DataLoader(val_dataset, batch_size=self.config.batch_size),
        )

    def configure_metrics(self) -> None:
        # Metrics assigned as trainer attributes are auto-tracked, moved to the
        # trainer device, reset before validation, and computed after validation.
        self.metrics = self.config.metrics

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> dict[str, Any]:
        # training_step owns the experiment-specific forward pass and loss.
        # Dream Trainer wraps the call with the training context and callbacks.
        pred = self.model(batch["x"])
        loss = F.mse_loss(pred, batch["y"])

        # Use self.backward instead of loss.backward directly so gradient
        # accumulation scaling stays consistent with TrainingParameters.
        self.backward(loss)

        logs: dict[str, Any] = {"train/loss": loss}
        if not self.is_accumulating_gradients:
            # self.step handles gradient validation, gradient clipping,
            # optimizer.step(), optimizer.zero_grad(), scheduler stepping, and
            # optimizer callbacks.
            logs["train/grad_norm"] = self.step(self.optimizer)

        # Scalar tensors returned here can be consumed by logger callbacks.
        return logs

    @torch.no_grad()
    def validation_step(
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int,
    ) -> dict[str, Any]:
        # validation_step runs with gradients disabled by the trainer loop.
        # Update metrics here and return any scalar logs for callbacks.
        pred = self.model(batch["x"])
        loss = F.mse_loss(pred, batch["y"])
        self.metrics.update(pred, batch["y"])
        return {"val/loss": loss}


@entrypoint
def main() -> None:
    # entrypoint creates a local distributed environment when one is not
    # already provided by torchrun or a cluster launcher.
    config = ToyTrainerConfig(
        project="dream-trainer-quickstart",
        group="local",
        # Disable compile in the first example so the trainer does not need an
        # apply_compile hook yet.
        device_parameters=DeviceParameters.SINGLE_DEVICE(compile_model=False),
        training_parameters=TrainingParameters(
            n_epochs=2,
            train_steps_per_epoch=32,
            val_steps_per_epoch=8,
            num_sanity_val_steps=2,
            gradient_clip_val=1.0,
        ),
        # WandB is installed for today's DreamTrainer import path, but disabled
        # so the quick start runs without networked experiment logging.
        logging_parameters=WandbLoggingParameters(enabled=False),
        callbacks=callbacks.CallbackCollection(
            [
                # LoggerCallback records scalar values returned by
                # training_step and validation_step.
                callbacks.LoggerCallback(log_every_n_train_batches=1),
                # ProgressBar displays local training progress and the chosen
                # scalar metric.
                callbacks.ProgressBar(metric="train/loss"),
            ]
        ),
    )

    # The trainer owns lifecycle orchestration; fit() launches configure,
    # setup, sanity validation, training, validation, callbacks, and teardown.
    ToyTrainer(config).fit()


if __name__ == "__main__":
    main()
```

Run it:

```bash
CUDA_VISIBLE_DEVICES=0 python quickstart.py
```

!!! tip "What you'll see"
    Dream Trainer initializes a single-process distributed world, runs two sanity validation batches, trains for two epochs, runs validation, and reports progress for `train/loss`.

## What the hooks do

For the deeper lifecycle model behind these hooks, read [Core Concepts](core-concepts.md).

| Hook | Purpose |
| --- | --- |
| `configure_models` | Instantiate modules. Dream Trainer calls this under a meta-device context so large models can be configured before materialization. |
| `init_weights` | Initialize parameters after Dream Trainer materializes the model on the training device. |
| `model_state_dict` | Define the model state that distributed checkpointing should save and load. |
| `configure_optimizers` | Create optimizers after model setup, then return the model-to-optimizer mapping. |
| `configure_dataloaders` | Return train and validation iterables. In distributed runs, this is where production code passes `rank=self.world.dp_rank` and `world_size=self.world.dp_size` into dataloader factories. |
| `configure_metrics` | Register metric collections as trainer attributes so Dream Trainer can move and reset them. |
| `training_step` | Compute the loss, call `self.backward(loss)`, and call `self.step(self.optimizer)` when gradient accumulation is complete. |
| `validation_step` | Run evaluation logic, update metrics, and return scalar validation logs. |

## Why the example disables compile

`DeviceParameters.SINGLE_DEVICE(compile_model=False)` keeps the first script focused on the trainer lifecycle. If `compile_model=True`, Dream Trainer calls `apply_compile`, and your trainer must implement that hook.

```python
def apply_compile(self):
    self.model.compile(mode="max-autotune-no-cudagraphs", dynamic=False)
```

!!! info "The same pattern applies to parallelism"
    When you switch to DDP, FSDP, TP, or PP, Dream Trainer calls explicit hooks: `apply_replicate`, `apply_fully_shard`, `apply_tensor_parallel`, `apply_pipeline_parallel`. You implement only the hooks relevant to the parallelism dimensions you enable.

## Optional: upgrade to the CLI

`@entrypoint` is the simplest launcher. If you want subcommands (`benchmark`, `profile`, `summarize`, `find-graph-breaks`) or flags like `--resume` / `--init-from` without editing code, swap in `cli()`:

```bash
pip install "dream-trainer[cli]"
```

```python
from dream_trainer.utils.cli import cli


def main(config: ToyTrainerConfig) -> None:
    ToyTrainer(config).fit()


if __name__ == "__main__":
    cli(main, ToyTrainerConfig(...))
```

```bash
python quickstart.py --help              # list every subcommand and modifier flag
python quickstart.py --cfg               # print the resolved config and exit
python quickstart.py --no-compile        # flip a built-in modifier for one run
python quickstart.py profile --skip 5    # run the trainer under torch.profiler
```

`cli()` wraps `main` with `@entrypoint` internally, so the distributed setup is identical. See [Using the CLI](cli.md) for the full surface.

## From quick start to production

Production trainers use the same skeleton with richer components:

| Change | Why |
| --- | --- |
| Split `config.py` and `train.py` | Keeps experiment configuration reviewable and separate from lifecycle code. |
| Store factories on the config | Makes model/optimizer/scheduler/metric choices explicit per-run. |
| Config-backed dataloaders with `rank` and `world_size` | Required for sharded data parallelism. |
| Swap `SINGLE_DEVICE()` → `DDP()` / `FSDP()` / `HSDP(...)` | Dream Trainer calls the parallelism hooks your `DeviceParameters` enables. |
| Add production callbacks | Checkpointing, logging, profiling, FP8, EMA, fault tolerance. |

A production callback collection commonly grows into something like:

```python
callbacks=callbacks.CallbackCollection(
    [
        callbacks.TrainerSummary(),
        callbacks.LoggerCallback(code_dir="../"),
        callbacks.LRLoggerCallback(),
        callbacks.ProgressBar(metric="train/loss"),
        callbacks.CheckpointCallback(...),
    ]
)
```

!!! tip "The control flow never changes"
    The model and data change. The control flow stays familiar: config object, trainer class, setup hooks, step hooks, and `trainer.fit()`. That continuity is the payoff for the lifecycle discipline.
