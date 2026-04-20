---
title: API Reference
---

# API Reference

<small>📚 Reference · stable import paths, terse summaries</small>

Dream Trainer's public API is organized around trainers, mixins, callbacks, configuration dataclasses, and distributed utilities. The pages in this section name the stable import paths and summarize what each module owns.

If you're not sure what you're looking for, start with the Meteor tutorial arc — every API shows up in use across those five pages:

- [Tutorial 1 — Meet Meteor](../tutorials/first-trainer.md) · [2 — Scaling Meteor](../tutorials/multi-gpu.md) · [3 — Sharding Meteor](../tutorials/fsdp.md) · [4 — Production Shape](../tutorials/production.md) · [5 — Extending Meteor](../tutorials/custom-components.md)

## Primary Imports

```python
from dream_trainer import DreamTrainer, DreamTrainerConfig, callbacks
from dream_trainer.configs import (
    CheckpointParameters,
    DeviceParameters,
    TrainingParameters,
    WandbLoggingParameters,
)
from dream_trainer.utils.entrypoint import entrypoint
```

## Reference Sections

| Section | Contents | Start here if… |
| --- | --- | --- |
| [Trainers](trainers/dream.md) | `AbstractTrainer`, `BaseTrainer`, `DreamTrainer`. | You're subclassing a trainer. |
| [Mixins](mixins/setup.md) | Setup, metrics, logging, quantization behavior. | You want to know what `DreamTrainer` is composed of. |
| [Callbacks](callbacks/base.md) | Lifecycle hooks for checkpointing, logging, performance, debugging, and training extensions. | You're writing or picking a callback. |
| [Configuration](configuration/parameters.md) | Dataclasses for training, device meshes, logging, checkpointing, and fault tolerance. | You're writing a config. |
| [Utilities](utilities/world.md) | Distributed world helpers, vendored distributed utilities, launch entrypoint, materialization, serialization, common helpers. | You need `self.world.*`, rank helpers, or the `@entrypoint` decorator. |

## Design Notes

- Models remain ordinary `torch.nn.Module` objects.
- Trainers own lifecycle and distributed bookkeeping.
- User code owns model construction, initialization, optimizer mapping, dataloaders, loss computation, and parallelism policy.
- Callbacks own reusable cross-cutting behavior.
- Configuration dataclasses make experiment choices reviewable and reproducible.

