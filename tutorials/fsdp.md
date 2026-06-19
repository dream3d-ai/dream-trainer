---
title: "Tutorial 3 — Sharding Meteor"
---

# Tutorial 3 — Sharding Meteor with FSDP

<small>🎓 Tutorial · ~25 min · 4 GPUs on one node</small>

## Where We Left Off

[Tutorial 2](multi-gpu.md) gave us **Meteor v1**: a DDP-parallel 125M-parameter trainer, compiled, rank-aware data, 4× the throughput of v0. The model, gradients, and Adam state fit comfortably on a single GPU — DDP just replicates them.

Today we want to train **Meteor v2: 1.3B parameters**. Twenty-four layers, 2048-wide, 16 heads. Same trainer, bigger numbers.

## What Actually Goes Wrong

Before writing any fix, run v1 with the bigger config and read the error. This is how distributed bugs tell the truth:

```python
def v2_config_broken() -> MeteorConfig:
    return MeteorConfig(
        project="meteor",
        group="v2-broken",
        d_model=2048,
        n_heads=16,
        n_layers=24,
        seq_len=2048,
        batch_size=4,
        device_parameters=DeviceParameters.DDP(),   # still DDP
        training_parameters=TrainingParameters(n_epochs=1, train_steps_per_epoch=50),
    )
```

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py
```

A few steps in, every rank crashes with the same message:

```text
torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 10.12 GiB.
GPU 0 has a total capacity of 79.15 GiB of which 8.42 GiB is free.
```

Do the napkin math. At 1.3B parameters in BF16:

- Model weights: ~2.6 GB
- Gradients: another ~2.6 GB
- Adam state (first and second moments, FP32): ~10.4 GB
- Activations at seq=2048, batch=4: several GB more, scaling with layers

That's ~20 GB per GPU *before* activations and workspace — and DDP forces every GPU to hold all of it. On an 80 GB card you'd be tight; on 40 GB cards you're dead.

!!! info "Why replication runs out"
    DDP pays the full memory footprint of the model, gradients, and optimizer state on every GPU. Adding GPUs to a DDP run adds *throughput*, not *capacity*. When the model fits replicated, DDP is the right tool. When it doesn't, you need to stop replicating.

## The Fix — Shard, Don't Replicate

FSDP shards parameters, gradients, and optimizer state across the `dp_shard` mesh dimension. Each GPU holds only its slice; parameters are all-gathered per layer as they're needed, then released. Memory per GPU drops to roughly 1/N of the replicated footprint.

The diff from v1 is tighter than you'd expect.

### Step 1 — Config

```python
def v2_config() -> MeteorConfig:
    return MeteorConfig(
        project="meteor",
        group="v2",
        d_model=2048,
        n_heads=16,
        n_layers=24,
        seq_len=2048,
        batch_size=4,
        device_parameters=DeviceParameters.FSDP(
            compile_model=False,
            async_tensor_parallel=False,
        ),
        training_parameters=TrainingParameters(n_epochs=1, train_steps_per_epoch=50),
    )
```

!!! tip "Disable compile for the first FSDP run"
    Compile + FSDP works, but compile + FSDP + a silent rank-aware-data bug + an unfamiliar sharding policy is too many moving parts at once. Get non-compiled FSDP correct first, then turn compile back on. The next section shows both steps.

### Step 2 — Replace `apply_replicate` With `apply_fully_shard`

Delete the replicate hook from v1. Add:

```python
from torch.distributed.fsdp import fully_shard


def apply_fully_shard(self, config):
    for layer in self.model.layers:
        fully_shard(layer, **config)
    fully_shard(self.model, **config)
```

Two lines of policy, and that's the whole sharding decision. Notice:

- **Each transformer block is sharded independently.** Dream Trainer all-gathers per-layer as the forward pass walks through them. Sharding the whole model as one unit forces a full all-gather on every forward — you'd get correctness but throw away the memory win.
- **The root module is sharded last.** This is the FSDP2 convention for composable wrapping.
- **`config` comes from Dream Trainer.** It's built from your `DeviceParameters`: mixed precision settings, mesh, offload policy. You don't assemble it yourself.

### Step 3 — Keep Weight Init Inside `init_weights`

This didn't change from v0/v1, but it matters more now. Under FSDP, `configure_models` runs on the meta device and `init_weights` runs after sharding. That split is why Dream Trainer has two hooks — so large models can be "built" before any GPU allocates real memory for them.

```python
def configure_models(self):
    self.model = Meteor(...)   # meta device, ~0 memory footprint

def init_weights(self):
    self.model.init_weights()  # real tensors, only this rank's shard
```

!!! danger "Don't load pretrained weights in `configure_models`"
    Same rule as v0, amplified. A 1.3B checkpoint loaded into meta-device tensors has nowhere to go — the `.data = ...` is silently dropped. If you're loading pretrained weights, do it in `init_weights` on the already-sharded real tensors. `LoadPartialCheckpointCallback` handles this for most cases.

### Step 4 — Dataloaders And `model_state_dict` Don't Change

This is the payoff for being disciplined in earlier tutorials:

- `configure_dataloaders` already uses `self.world.dp_rank` and `self.world.dp_size`. Those still work — the mesh now interprets them as "your position in the sharded dp group" instead of "your position in the replicated dp group".
- `model_state_dict` already uses `get_model_state_dict(self.model, options=StateDictOptions())`. That helper produces a DCP-compatible sharded state dict automatically — the same function works for single GPU, DDP, and FSDP.

If you'd taken a shortcut in v0 and used plain `self.model.state_dict()`, you'd be rewriting it now. You didn't. Good.

## Running It

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py
```

Every rank now only allocates roughly 1/4 of the weight, gradient, and optimizer footprint. The 1.3B model that OOM'd under DDP fits, runs, logs loss.

## Turning Compile Back On

Once non-compiled FSDP is stable, flip the flag:

```python
device_parameters=DeviceParameters.FSDP(compile_model=True)
```

…and reinstate `apply_compile` from v1. Dream Trainer's lifecycle runs `apply_compile` *before* FSDP wrapping, which is the correct order for FSDP2 — the compiled graph sees the unwrapped module.

!!! warning "Compile after FSDP is wrong for FSDP2"
    If you've been reading older tutorials online, you'll see advice to compile *after* FSDP wrapping. That was correct for FSDP1 and is **incorrect** for FSDP2. Dream Trainer enforces the right order for you; if you override the ordering manually, you'll get graph breaks and worse performance.

## When To Reach For HSDP Instead

`FSDP()` shards across every data-parallel rank. On a single node with fast NVLink that's fine. Across multiple nodes, full-world all-gather starts hitting the slow inter-node fabric on every layer.

`HSDP(dp_shard=8)` shards within groups of 8 (typically one node) and replicates across groups. All-gather stays inside NVLink; gradient reduction rides the slower inter-node network just once per step.

Same hook — `apply_fully_shard` — the mesh does all the work. Switch to HSDP when the topology of your cluster warrants it, not preemptively.

## Sanity Checks Before Moving On

- [ ] Non-compiled FSDP v2 runs without OOM.
- [ ] Loss decreases — FSDP doesn't silently change numerics when set up correctly.
- [ ] Compiled FSDP matches the non-compiled loss curve within noise.
- [ ] Killing and restarting the run resumes from the checkpoint — DCP stitches shards back together across restarts and even across different mesh shapes.

## The New Thing We Have Now

**Meteor v2**: 1.3B parameters, FSDP-sharded, compiled, still running on 4 GPUs of the same node.

What we have now is a trainer that is genuinely *scalable* — the same code will run on 64 or 256 GPUs. What it **isn't** yet is a production trainer. Everything's in one `train.py`, logging goes to stdout, checkpoint writes block the training step, and there's no WandB, no EMA, no async save.

In [Tutorial 4 — Shaping Meteor for Production](production.md) we split the file, plug in a real callback stack, and make checkpoints non-blocking.
