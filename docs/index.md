---
title: dream-trainer
summary: Dream Trainer is a composable PyTorch training framework for custom training systems. It gives you reusable building blocks for trainer lifecycle, distributed setup, callbacks, checkpointing, and configuration so you can keep the algorithm explicit, scale through config, and avoid rebuilding trainer infrastructure for every project.
hide:
  - toc
show_datetime: false
---


## Quickstart

Install and run a single-file trainer in under a minute.

=== "pip"

    ```bash
    pip install "dream-trainer[metrics,wandb]"
    ```

=== "uv"

    ```bash
    uv add "dream-trainer[metrics,wandb]"
    ```

=== "From source"

    ```bash
    git clone https://github.com/dream3d/dream-trainer
    cd dream-trainer && uv sync --all-extras
    ```

[Quick Start →](getting-started.md){ .home-cta } &nbsp; [Installation →](installation.md){ .home-cta-ghost }

## A trainer in ten lines

```python
from dream_trainer import DreamTrainer

class MyTrainer(DreamTrainer):
    def configure_models(self):
        self.model = MyModel(self.config.model)

    def configure_optimizers(self):
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.lr)
        return {self.model: self.optimizer}

    def training_step(self, batch, batch_idx):
        logits = self.model(batch["input"])
        loss = F.cross_entropy(logits, batch["target"])
        self.backward(loss)
        return {"train/loss": loss, "train/grad_norm": self.step(self.optimizer)}

MyTrainer(config).fit()
```

No hidden decorators, no framework-owned `forward`. Dream Trainer handles distributed launch, device mesh construction, meta-device materialization, parallelism ordering, checkpoint state, and callback dispatch. Everything else is ordinary PyTorch.

## What you get

<div class="home-grid" markdown>

<div class="home-card" markdown>
### Device meshes, named

`DeviceParameters.FSDP()`, `DeviceParameters.HSDP(...)`, or a custom mesh. Dimensions like `pp`, `dp_replicate`, `dp_shard`, `cp`, `tp` are first-class.

[Parallelism →](parallelism.md)
</div>

<div class="home-card" markdown>
### Explicit parallelism hooks

`apply_replicate`, `apply_fully_shard`, `apply_tensor_parallel`, `apply_pipeline_parallel`. You implement only what your model needs. Nothing is guessed.

[Trainer Guide →](trainer-guide.md)
</div>

<div class="home-card" markdown>
### DCP checkpoints that resume

Full trainer state — model, optimizer, scheduler, dataloader, callbacks, counters — via PyTorch Distributed Checkpoint. Resume works across mesh shapes.

[Checkpointing →](checkpointing.md)
</div>

<div class="home-card" markdown>
### Callbacks that compose

Logger, progress, LR, profile, benchmark, graph-break, FP8, EMA, async checkpoint, fault tolerance. Register what you need per run.

[Callbacks →](callbacks.md)
</div>

<div class="home-card" markdown>
### Diagnostics without code edits

`python train.py profile`, `benchmark`, `find-graph-breaks`, `summarize`. The CLI wraps `@entrypoint` and turns every registered modifier into a `--flag`.

[Using the CLI →](cli.md)
</div>

<div class="home-card" markdown>
### Typed configs, not YAML

`DreamTrainerConfig` dataclasses describe the run. The trainer performs setup. Named factories (`debug_config()`, `fsdp_config()`) swap in one line.

[Configuration →](configuration.md)
</div>

</div>

## Built on modern PyTorch

Dream Trainer is a thin lifecycle substrate over the stable PyTorch distributed surface. No private forks, no vendored kernels.

- **`torch`** 2.7+ — `nn.Module`, `torch.compile`, `torch.autograd`.
- **FSDP2** — fully-sharded data parallel via `fully_shard`.
- **DTensor** — TP, CP, async TP, 2D parallelism.
- **DCP** — `torch.distributed.checkpoint` for sharded save/load.
- **`torch.profiler`** — driven by `ProfileCallback`, no custom instrumentation.
- **torchao** — optional FP8 quantization via `Fp8Quantization`.
- **torchft** — optional fault tolerance via `FaultToleranceCallback`.

## Where to go next

<div class="home-grid" markdown>

<div class="home-card" markdown>
### Learn by doing

Five-part Meteor arc: build a tiny GPT-style trainer from single-GPU to multi-node, FSDP, production shape.

[Tutorials →](tutorials/first-trainer.md)
</div>

<div class="home-card" markdown>
### Learn by lookup

Task-oriented recipes — trainer hooks, configuration, parallelism, callbacks, checkpointing, logging, debugging, performance.

[How-To Guides →](trainer-guide.md)
</div>

<div class="home-card" markdown>
### Understand the design

Core concepts, lifecycle phases, mixins vs callbacks, and the decisions behind the API.

[Core Concepts →](core-concepts.md)
</div>

<div class="home-card" markdown>
### Full API

Every public symbol with source links — trainers, mixins, callbacks, configs, utilities.

[API Reference →](api/index.md)
</div>

</div>

## Is Dream Trainer for you?

| If you want to… | Dream Trainer? |
| --- | --- |
| Train on 1 GPU with a readable loop | Probably overkill — a raw script is simpler. |
| Scale the same trainer from 1 to 1000 GPUs without rewriting the body | **Yes.** This is the point. |
| Combine FSDP + TP + CP + PP in one run | **Yes.** Named mesh dimensions make this first-class. |
| Have the framework pick your parallelism strategy automatically | No — Dream Trainer refuses to guess model-specific parallelism. |
| Debug a distributed failure by reading ordinary PyTorch objects | **Yes.** Nothing is hidden behind decorators. |

For a deeper comparison with other distributed training frameworks, see [Comparison](comparison.md).
