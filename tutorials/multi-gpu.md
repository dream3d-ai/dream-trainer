---
title: "Tutorial 2 — Scaling Meteor"
---

# Tutorial 2 — Scaling Meteor to Multiple GPUs

<small>🎓 Tutorial · ~20 min · 4 GPUs on one node</small>

## Where We Left Off

At the end of [Tutorial 1](first-trainer.md) we had **Meteor v0**: a 125M-parameter GPT trainer, one file, one GPU, synthetic data, a decreasing loss, and a checkpoint under `/tmp/meteor/v0/`.

## The Problem

On one GPU, the batch size that fits comfortably is 8. That's fine for a toy experiment, but our effective batch size is tiny — gradient estimates are noisy and wall-clock time per token is bad. We want **4× throughput** without changing the model.

We have a 4-GPU node sitting there. The right tool here is **data parallelism**: every GPU holds the full model and its own batch shard; gradients get averaged at step time. DDP. Start with DDP before reaching for anything more exotic — if it works, the answer is DDP.

!!! info "Why DDP first"
    DDP is the simplest way to verify that dataloaders, metrics, logging, and checkpoints all behave correctly across ranks. If you skip it and start with FSDP, you will fight rank-aware data bugs and sharded checkpointing bugs simultaneously. One problem at a time.

## What Changes — The Diff

Going from v0 to v1 touches three places in `train.py`:

1. **Config**: swap `SINGLE_DEVICE` for `DDP` and add a `v1_config` factory.
2. **Trainer**: implement two new hooks — `apply_replicate` and `apply_compile`.
3. **Dataloaders**: pass `dp_rank` and `dp_size` so every rank sees different data.

That's the entire diff. Everything else — the model, `init_weights`, `training_step`, checkpointing — stays.

## 1. New Config Variant

Add to the factory module:

```python
def v1_config() -> MeteorConfig:
    return MeteorConfig(
        project="meteor",
        group="v1",
        device_parameters=DeviceParameters.DDP(),
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

Two things to notice. We kept `v0_config` — you'll want to fall back to single-GPU for debugging. And `DeviceParameters.DDP()` enables compile by default; that's why the next step adds `apply_compile`.

## 2. Two New Hooks On The Trainer

```python
from torch.distributed._composable.replicate import replicate


class MeteorTrainer(DreamTrainer):
    # ... (everything from v0 stays)

    def apply_compile(self):
        self.model.compile(mode="max-autotune-no-cudagraphs", dynamic=False)

    def apply_replicate(self, dp_replicate_mesh):
        replicate(self.model, device_mesh=dp_replicate_mesh)
```

Dream Trainer calls these in the right order — **compile before replicate** — so the compiled graph sees the unwrapped module. This is the correct order for composable DDP; the common internet advice to "compile after DDP wrapping" is wrong for this path.

!!! warning "Compile + async TP are coupled"
    If you ever turn off compile (for debugging), you must also turn off async tensor parallelism on the same `DeviceParameters`. They are paired because async TP rewrites collectives at compile time — with no compile, there's nothing to rewrite. `DDP()` default doesn't use TP at all, so you're fine here; keep it in mind when you reach Tutorial 3.

## 3. Rank-Aware Dataloaders

This is the subtle change. Replace the v0 dataloader with one that shards:

```python
from torch.utils.data import DistributedSampler


def configure_dataloaders(self) -> tuple[Iterable, Iterable]:
    tokens = torch.randint(0, self.config.vocab_size, (1024, self.config.seq_len))
    dataset = TensorDataset(tokens)

    def loader(shuffle: bool) -> DataLoader:
        sampler = DistributedSampler(
            dataset,
            num_replicas=self.world.dp_size,
            rank=self.world.dp_rank,
            shuffle=shuffle,
        )
        return DataLoader(dataset, batch_size=self.config.batch_size, sampler=sampler)

    return loader(shuffle=True), loader(shuffle=False)
```

!!! danger "Rank-aware dataloaders are not optional under DP"
    If you skip `rank=self.world.dp_rank, num_replicas=self.world.dp_size`, every rank sees the same batch. The gradients are perfectly redundant, your loss still decreases, and the bug is **silent** — no crash, just effective batch size of one GPU with four times the compute. If there is exactly one line in distributed training you should never forget, it is this one. See [Parallelism](../parallelism.md#data-parallel-ddp-vs-fsdp-vs-hsdp).

Note `self.world.dp_rank` and `self.world.dp_size` — not the global rank. Under pure DDP they happen to be the same, but this code keeps working when we add sharding or tensor parallelism later.

## Running It

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py
```

The `entrypoint` helper sees four visible GPUs and no distributed env vars, so it launches four local processes. If your cluster uses `torchrun`:

```bash
torchrun --nproc-per-node=4 train.py
```

Dream Trainer detects the distributed environment and uses it instead of spawning.

## Does It Work?

Three checks, in order, before moving on:

=== "1. Different data per rank"

    Add a one-time assertion inside `training_step` during development:

    ```python
    if batch_idx == 0:
        print(f"rank={self.world.dp_rank}: tokens[0,:5]={batch[0][0,:5].tolist()}")
    ```

    Each rank should print different numbers. If they all print the same, your sampler is wrong.

=== "2. Throughput scales"

    Your `ProgressBar` should show ~4× more training steps per minute than Tutorial 1 (on the same 4 GPUs). If it scales sub-linearly, the bottleneck is dataloading, not compute. More `num_workers`, or cache the dataset on fast disk.

=== "3. Checkpoints still work"

    Kill the run mid-training and restart it. The loss should resume from where it stopped. DCP handles the state merge; you don't have to do anything.

## What Didn't Change

This is the point of the whole exercise. Compare v0 and v1:

- `configure_models` — same.
- `init_weights` — same.
- `model_state_dict` — same.
- `training_step` — same.
- `validation_step` — same.

The model code stayed ordinary PyTorch. Distributed policy became two trainer hooks. That's the split Dream Trainer is built around.

## The New Thing We Have Now

**Meteor v1**: DDP-parallel, compiled, rank-aware data. Same model, ~4× throughput.

But here's the next problem lurking. Right now Meteor is 125M parameters — replicated across 4 GPUs, that's 500MB per GPU just for the weights. Tomorrow we want to push to **1.3B parameters**. The math gets uncomfortable: 5GB of weights, plus gradients, plus Adam state (another 2× for first and second moments), plus activations. Replicated, it won't fit.

In [Tutorial 3 — Sharding Meteor with FSDP](fsdp.md) we watch the v1 trainer OOM on the bigger model and fix it with sharding.
