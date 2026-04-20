---
title: Configuration
---

# Configuration

<small>🛠️ How-to · ~12 min read · copy-paste friendly</small>

!!! abstract "TL;DR"
    - Configs are **typed Python dataclasses**, not YAML.
    - **Config describes choices. Trainer hooks perform setup.** Don't construct CUDA tensors or imported models in config defaults.
    - Organize variants with **named factories** (`debug_config()`, `fsdp_config()`) and compose with `dataclasses.replace`.
    - **Modifiers** are small functions registered on `MODIFIERS` that mutate a config — auto-surfaced as `--<flag>` options by [the CLI](cli.md#modifiers).

Your trainer needs to run in three modes: a fast smoke test on your laptop, a nightly FSDP run on a shared cluster, and a one-off large-scale run when you have a bug to reproduce. The training code shouldn't change between them. What changes is configuration.

Dream Trainer configs are **typed Python dataclasses**, not YAML. This page explains how to structure them, where the boundary between config and trainer hooks sits, and how to organize variants so experiments stay reviewable.

## The Boundary

> **Config describes choices. Trainer hooks perform setup.**

Everything follows from that rule. The config holds dimensions, hyperparameters, paths, and *factories* that know how to build things. The trainer builds them — at the right lifecycle phase, on the right rank, with access to the distributed world.

**Good config fields:**

- dimensions and hyperparameters
- file paths
- batch sizes and worker counts
- dtype, device, and parallelism choices
- callback parameters
- small factory objects that build runtime components later

**Do not put in config defaults:**

- constructed CUDA tensors
- instantiated models
- opened files or network connections
- dataloaders that depend on rank or world size
- optimizer instances tied to unmaterialized parameters
- any side effect that fires at import time

Dream Trainer creates the distributed world **before** calling setup hooks. Anything that needs rank, device, process group, or materialized parameters belongs in the trainer hook that has that runtime context. If your config module allocates a `torch.zeros(…, device="cuda")` at import time, you've crossed the boundary.

## The Base Shape

Every `DreamTrainerConfig` includes:

| Field | Purpose |
| --- | --- |
| `project`, `group`, `experiment` | Run identity (used by WandB). |
| `seed` | Reproducibility. |
| `device_parameters` | Device, dtype, compile, and parallelism choices. |
| `training_parameters` | Epochs, step counts, validation cadence, accumulation, clipping. |
| `callbacks` | Lifecycle extensions as a `CallbackCollection`. |
| `logging_parameters` | WandB logging behavior. |

Your subclass adds whatever factories and hyperparameters your model needs:

```python
from dataclasses import dataclass, field

from torchmetrics import MeanSquaredError, MetricCollection

from dream_trainer import DreamTrainerConfig


@dataclass(kw_only=True)
class MyTrainerConfig(DreamTrainerConfig):
    model: MyModelConfig
    optimizer: OptimizerConfig
    train_data: DataConfig
    val_data: DataConfig
    metrics: MetricCollection = field(
        default_factory=lambda: MetricCollection({"mse": MeanSquaredError()})
    )
```

## File Layout

For anything beyond a one-file quick start, split into two files:

- `config.py` — the `Config` subclass and named factory functions (`debug_config()`, `fsdp_config()`, …).
- `train.py` — the trainer class and a `train(config)` entrypoint.

This mirrors production trainers. The same trainer class runs every variant; the variant picks the config.

## Training Parameters

`TrainingParameters` controls loop length and optimizer cadence:

```python
TrainingParameters(
    n_epochs=100,
    train_steps_per_epoch=1200,
    gradient_accumulation_steps=1,
    val_steps_per_epoch=256,
    val_every_n_steps=None,    # None → validate at epoch boundary
    num_sanity_val_steps=32,
    gradient_clip_val=1.0,
)
```

If you omit step counts, Dream Trainer infers them from the dataloaders. Iterable datasets without `__len__` must provide explicit counts.

!!! warning "Use `self.backward`, not `loss.backward()`"
    `gradient_accumulation_steps` only works if you use `self.backward(loss)` in `training_step` and gate optimizer steps with `self.is_accumulating_gradients`. Calling `loss.backward()` directly skips the accumulation scaling and your grad norms will be wrong by a factor of `gradient_accumulation_steps`.

## Device Parameters

Start with a preset:

```python
DeviceParameters.SINGLE_DEVICE(compile_model=False)
DeviceParameters.DDP()
DeviceParameters.FSDP()
DeviceParameters.HSDP(dp_shard=8)
```

Override individual fields when you need them:

```python
DeviceParameters(
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.float32,
    compile_model=True,
    compiled_autograd=True,
    _dp_shard=8,
    _dp_replicate=1,
    _tensor_parallel=1,
    _context_parallel=1,
    _pipeline_parallel=1,
)
```

At most one parallelism dimension can be set to `"auto"`; Dream Trainer resolves it from the remaining dimensions and world size. See [Parallelism](parallelism.md) for when to use each mode.

## Logging Parameters

`WandbLoggingParameters` controls WandB behavior. Disabling logging keeps the `WandBLoggerMixin` path intact (so `self.log_dict(...)` still works) but doesn't send anything externally:

```python
logging_parameters=WandbLoggingParameters(enabled=False)
```

Enable it when you want metrics, media, code snapshots, or model watching to land in a WandB run.

## Callbacks In Config

Callbacks live in the config, not in the trainer class. This is how you swap debug and production stacks without touching training code:

```python
callbacks=callbacks.CallbackCollection([
    callbacks.LoggerCallback(log_every_n_train_batches=20),
    callbacks.LRLoggerCallback(),
    callbacks.ProgressBar(metric="train/loss"),
])
```

See [Callbacks](callbacks.md) for the three canonical stacks (minimum / production / debugging).

## Why Configs As Code

Python dataclasses give you things YAML cannot:

- typed nested objects your editor can follow
- normal inheritance between families of configs
- factory functions that build runtime components lazily
- shared constants, helper functions, and imports
- auto-complete, rename, go-to-definition
- `isinstance` checks and `mypy` validation before a run starts

Attribute access stays clean:

```python
config.training_parameters.gradient_accumulation_steps
config.device_parameters.tensor_parallel
config.model.hidden_dim
```

Those are real attributes. You can navigate, refactor, and type-check them. YAML configs turn all of that into strings.

!!! info "YAML still has a place"
    Python configs should be the **source of truth**, but exporting a resolved config to YAML or JSON is useful for run records, dashboards, reproducibility snapshots, and diffing two runs. Use YAML as output, not input.

## A Complete Pattern

Here is a `config.py` that shows the full shape — typed model/data/optimizer subconfigs, factory methods that defer construction to setup time, and named variants for debug and large-scale runs.

```python
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torchmetrics import MeanSquaredError, MetricCollection

from dream_trainer import DreamTrainerConfig, callbacks
from dream_trainer.configs import (
    CheckpointParameters,
    DeviceParameters,
    TrainingParameters,
    WandbLoggingParameters,
)


@dataclass(kw_only=True)
class MLPConfig:
    input_dim: int = 128
    hidden_dim: int = 512
    output_dim: int = 1
    depth: int = 4

    def build(self) -> nn.Module:
        layers: list[nn.Module] = []
        dim = self.input_dim
        for _ in range(self.depth):
            layers.extend([nn.Linear(dim, self.hidden_dim), nn.SiLU()])
            dim = self.hidden_dim
        layers.append(nn.Linear(dim, self.output_dim))
        return nn.Sequential(*layers)


@dataclass(kw_only=True)
class DataConfig:
    path: Path
    batch_size: int = 32
    num_workers: int = 8
    shuffle: bool = True

    def build_dataloader(self, *, rank: int, world_size: int) -> DataLoader:
        dataset = MyDataset(self.path)
        sampler = DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=self.shuffle,
        )
        return DataLoader(
            dataset, batch_size=self.batch_size, num_workers=self.num_workers,
            sampler=sampler, pin_memory=True,
        )


@dataclass(kw_only=True)
class OptimizerConfig:
    learning_rate: float = 3e-4
    weight_decay: float = 0.1

    def build(self, parameters: Iterable[torch.nn.Parameter]) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            parameters, lr=self.learning_rate, weight_decay=self.weight_decay,
        )


@dataclass(kw_only=True)
class RegressionTrainerConfig(DreamTrainerConfig):
    train_data: DataConfig
    val_data: DataConfig
    model: MLPConfig = field(default_factory=MLPConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    metrics: MetricCollection = field(
        default_factory=lambda: MetricCollection({"mse": MeanSquaredError()})
    )
```

The trainer consumes those factories in setup hooks — not in the config:

```python
class RegressionTrainer(DreamTrainer):
    config: RegressionTrainerConfig

    def configure_models(self):
        self.model = self.config.model.build()

    def configure_optimizers(self):
        self.optimizer = self.config.optimizer.build(self.model.parameters())
        return {self.model: self.optimizer}

    def configure_dataloaders(self):
        return (
            self.config.train_data.build_dataloader(
                rank=self.world.dp_rank, world_size=self.world.dp_size,
            ),
            self.config.val_data.build_dataloader(
                rank=self.world.dp_rank, world_size=self.world.dp_size,
            ),
        )
```

Notice that `build_dataloader` takes `rank` and `world_size` as arguments — the config doesn't know them, the trainer supplies them from `self.world` at setup time. This is the boundary in action.

## Named Variants

Run variants are **named functions**, not mutated global state.

```python
def debug_config() -> RegressionTrainerConfig:
    return RegressionTrainerConfig(
        project="regression",
        group="debug",
        model=MLPConfig(input_dim=32, hidden_dim=64, depth=2),
        train_data=DataConfig(path=Path("/data/debug/train"), batch_size=16),
        val_data=DataConfig(path=Path("/data/debug/val"), batch_size=16, shuffle=False),
        device_parameters=DeviceParameters.SINGLE_DEVICE(compile_model=False),
        training_parameters=TrainingParameters(
            n_epochs=1, train_steps_per_epoch=20, val_steps_per_epoch=4,
            num_sanity_val_steps=2,
        ),
        logging_parameters=WandbLoggingParameters(enabled=False),
        callbacks=callbacks.CallbackCollection([
            callbacks.LoggerCallback(log_every_n_train_batches=1),
            callbacks.ProgressBar(metric="train/loss"),
        ]),
    )


def fsdp_1b_config() -> RegressionTrainerConfig:
    return RegressionTrainerConfig(
        project="regression",
        group="fsdp-1b",
        model=MLPConfig(input_dim=1024, hidden_dim=8192, depth=32),
        optimizer=OptimizerConfig(learning_rate=3e-4, weight_decay=0.1),
        train_data=DataConfig(path=Path("/data/full/train"), batch_size=2),
        val_data=DataConfig(path=Path("/data/full/val"), batch_size=2, shuffle=False),
        device_parameters=DeviceParameters.FSDP(
            tensor_parallel="auto", dp_shard="auto",
            compile_model=True, compiled_autograd=True,
        ),
        training_parameters=TrainingParameters(
            n_epochs=10, train_steps_per_epoch=10_000, val_steps_per_epoch=500,
            gradient_accumulation_steps=8, num_sanity_val_steps=4,
            gradient_clip_val=1.0,
        ),
        logging_parameters=WandbLoggingParameters(enabled=True),
        callbacks=callbacks.CallbackCollection([
            callbacks.LoggerCallback(log_every_n_train_batches=20),
            callbacks.ProgressBar(metric="train/loss"),
            callbacks.AsyncCheckpointCallback(
                CheckpointParameters(
                    root_dir="/checkpoints/regression-1b",
                    monitor="val/loss", resume_mode="min",
                    checkpoint_every_n_val_epochs=1, keep_top_k=3,
                ),
            ),
        ]),
    )
```

Named variants make PRs reviewable. "Changed `fsdp_1b_config`" is a different claim than "changed some nested key in `default.yaml`".

## Deriving Variants

When two variants share most fields, use `dataclasses.replace` or inheritance instead of duplicating:

=== "dataclasses.replace"

    ```python
    from dataclasses import replace

    def compiled_debug_config() -> RegressionTrainerConfig:
        base = debug_config()
        return replace(
            base,
            device_parameters=DeviceParameters.SINGLE_DEVICE(compile_model=True),
        )
    ```

=== "Inheritance"

    ```python
    @dataclass(kw_only=True)
    class BaseImageConfig(DreamTrainerConfig):
        image_size: int = 256
        channels: int = 3


    @dataclass(kw_only=True)
    class DiffusionConfig(BaseImageConfig):
        noise_schedule: str = "cosine"


    @dataclass(kw_only=True)
    class DiscriminatorConfig(BaseImageConfig):
        use_spectral_norm: bool = True
    ```

    Use inheritance when a *family* of configs shares structure. If it starts hiding too much, fall back to explicit factory functions.

!!! tip "Use `default_factory` for mutable defaults"
    Mutable dataclass defaults (`MetricCollection`, `CallbackCollection`) must use `field(default_factory=...)` — otherwise every config instance shares the same object.

    ```python
    metrics: MetricCollection = field(
        default_factory=lambda: MetricCollection({"mse": MeanSquaredError()}),
    )
    ```

## Defaults To Revisit

A few defaults trip up new trainers. Revisit them when you copy a config from an existing run:

- `compile_model=True` is the DDP/FSDP preset default. If you haven't implemented `apply_compile`, set it to `False` for your first run.
- `param_dtype=torch.bfloat16` is the mixed-precision default. Make sure your model math and dataset tensors are compatible.
- `val_every_n_steps=None` runs validation at epoch boundaries. Set an explicit step cadence for long epochs.
- `num_sanity_val_steps=0` skips sanity validation entirely. Set this above zero on any new trainer — the first validation pass before training catches 90% of setup bugs.
- `resume_data=True` resumes dataloader state on checkpoint load. Set to `False` when you resume model state but intentionally change the data.

## Checklist

Before shipping a config module:

- [ ] Configs are dataclasses with typed fields.
- [ ] Heavy runtime objects are built in trainer hooks, not config defaults.
- [ ] Mutable defaults use `field(default_factory=...)`.
- [ ] Dataloader factories receive `rank` and `world_size` at setup time.
- [ ] Debug and production variants are named functions, not mutated state.
- [ ] Device scale changes are expressed through `DeviceParameters`.
- [ ] Callback stacks are config-owned.
- [ ] The same trainer class runs every variant.
- [ ] Nothing in the config module has import-time side effects.

## Next Steps

- [Trainer Guide](trainer-guide.md) — the hooks that consume these configs.
- [Parallelism](parallelism.md) — what each `DeviceParameters` preset actually does.
- [AI Agent Friendly](agents.md) — why typed config surfaces help code generation.
