---
title: Design Philosophy
---

# Design Philosophy

<small>📖 Explanation · ~8 min read · no runnable code</small>

Dream Trainer is a library of opinions. Some are visible in the API; some only show up after your trainer has been running for a week. This page collects the opinions in one place so you know what you are buying into.

The shortest version: **Dream Trainer owns the lifecycle. You own the model.** Everything below is a consequence of that split.

## Plain PyTorch Stays Visible

Training frameworks cluster around two failure modes. One is to hide PyTorch entirely behind a new abstraction — a `LightningModule`, an `Accelerator`, a `FabricStrategy` — so the framework can automate decisions on your behalf. The other is to hide almost nothing, and make you reassemble distributed plumbing in every experiment.

Dream Trainer picks a third path. Your model is an `nn.Module`. Your optimizer is an `Optimizer`. Your dataloader is any iterable. Your `training_step` returns a dict of scalars. You can `print(self.model)` and see the module you wrote. You can `list(self.optimizer.param_groups)` and see the parameter groups you assigned.

What does this buy you? Two things that frameworks rarely deliver together:

- **Transfer.** Anything you learn about PyTorch transfers. Anything you know about PyTorch applies. You do not have to learn a separate dialect of "Lightning-flavored PyTorch" or "Accelerate-flavored PyTorch" before you can debug your own code.
- **Inspection.** When something goes wrong — a gradient is `NaN`, a parameter didn't update, a checkpoint refuses to load — you can step into the trainer with a debugger and read ordinary PyTorch state. Nothing is hidden behind decorators or closures.

The cost is that you write slightly more glue for ordinary things. `configure_models`, `configure_optimizers`, `configure_dataloaders` are all explicit methods on your trainer. In exchange, those methods are the *only* places model-specific decisions live, and when you read your trainer a year later, you can see what it does.

## Structure Without Hiding The Sharp Edges

Modern PyTorch distributed training has an ordering problem. Roughly:

> Launch the world → build the device mesh → create the model on meta tensors → apply pipeline parallelism → apply tensor parallelism → apply activation checkpointing → compile → wrap with FSDP or DDP → materialize on device → initialize weights → build optimizers → build dataloaders → run sanity validation → train.

Get this order wrong and the symptoms are bizarre. Apply FSDP before tensor parallelism and your TP plan silently refers to the wrong sub-modules. Compile *after* FSDP and the compiled graph breaks at every DTensor boundary — this was the right order for FSDP1 and is wrong for FSDP2. Build the optimizer before materialization and it captures meta tensors instead of real parameters. Each of these has burned someone, and the errors don't always point at the cause.

Dream Trainer centralizes the ordering. You don't write it. But it also doesn't *hide* it — the ordering is visible, each step is a named method you can override, and the in-source comment at [base.py:898](https://github.com/dream3d/dream-trainer/blob/main/src/dream_trainer/trainer/base.py) spells out the full sequence. The shape is:

- If you ask for tensor parallelism, you implement `apply_tensor_parallel(tp_mesh)`. Dream Trainer calls it at the right moment.
- If you ask for FSDP, you implement `apply_fully_shard(config)`. Dream Trainer calls it at the right moment.
- If you want compile, you implement `apply_compile()`. Dream Trainer calls it at the right moment.

The policy is yours. The schedule is ours. This is the meaningful difference from a framework that tries to infer parallelism from your model — model-specific parallelism cannot be inferred cleanly, so we don't pretend to.

## Composable By Default

"Composition over inheritance" is easy to say and hard to do. Dream Trainer's version is concrete:

- `BaseTrainer` owns the lifecycle loop, callback dispatch, state management, checkpoint integration, gradient accumulation, and validation cadence. Nothing model-specific.
- `SetupMixin` tracks models, optimizers, schedulers, and dataloaders as trainer attributes, then calls your `configure_*` and `apply_*` hooks in the correct order.
- `EvalMetricMixin` integrates `torchmetrics` — moves collections to the trainer device, resets them before validation, computes them after.
- `WandBLoggerMixin` adds `log_dict`, `log_image`, `log_video` methods so logger callbacks have a uniform target.
- `QuantizeMixin` tracks quantized models and exposes a hook for quantization policy.

The default `DreamTrainer = BaseTrainer + SetupMixin + EvalMetricMixin + WandBLoggerMixin`. That is a composition, not a contract. If your research trainer doesn't need metrics, you can skip `EvalMetricMixin`. If you need something we don't ship — a custom metric system, a different logger, a new form of quantization — you can write a mixin for it and compose.

The analogous pattern for behaviors that don't live inside the trainer is **callbacks**. Progress bars, checkpointing, EMA, FP8, profiling, graph-break inspection — none of these belong in `training_step`, and none of them belong inside a mixin either, because they don't need access to model internals. They need access to lifecycle events. That's what callbacks are for.

> **Rule of thumb.** If the behavior cares about what the model *is*, it's a hook (mixin method on the trainer). If it cares about what the trainer *is doing*, it's a callback.

## Modern Distributed First

Dream Trainer's baseline assumption is that you want to use PyTorch's modern distributed machinery: composable FSDP (FSDP2), DTensor-aware tensor parallelism, named device meshes, distributed checkpointing (DCP), mixed precision via DTensor, loss parallelism, and — increasingly — context parallelism and pipeline parallelism as first-class primitives.

"Modern distributed first" has a few concrete consequences:

- **Device mesh is the substrate.** Every parallelism decision projects onto a named dimension: `pp`, `dp_replicate`, `dp_shard`, `cp`, `tp`. Presets like `DeviceParameters.FSDP()` and `DeviceParameters.HSDP(dp_shard=8)` just pick which dimensions are active.
- **Meta-device model creation is the default.** A 70B model doesn't fit on a single GPU's worth of RAM as a plain `nn.Module`, but its *structure* fits fine on meta tensors. Dream Trainer creates your model on meta, lets you apply parallelism, then materializes only the parts each rank owns. This is not an optional performance feature — it's how large models are built.
- **Checkpoints are sharded.** `BaseTrainer.state_dict()` returns a full training state that DCP can save and load across whatever mesh shape the run happens to use. Same checkpoint, different mesh, resumes cleanly. You don't have to rematerialize a full-state `.pt` file.
- **Compile is explicit.** If you want `torch.compile`, you implement `apply_compile()`. Async tensor parallelism, compiled autograd, and graph-break inspection all plug into this explicit hook. We do not call `torch.compile` behind your back.

The default single-GPU experience is still straightforward — you can write a trainer and run it on one GPU without ever touching these concepts. But the shape of the API does not change when you scale. The thing you ran on one GPU is the thing that runs on 128 GPUs with tensor + context + FSDP parallelism, because the distributed lifecycle was already the substrate.

## What Dream Trainer Is Not

No philosophy is complete without its negations.

- **Not a managed trainer.** Dream Trainer does not choose your optimizer, your precision, your schedule, your wrapping policy, or your tensor-parallel plan. If you want those decisions made for you, a managed framework is the right tool.
- **Not a beginner's training loop.** The target reader already knows what `nn.Module`, `DistributedSampler`, `torch.distributed.init_process_group`, and mixed precision are. Dream Trainer assumes that baseline and builds on it.
- **Not a single-file training script.** For a 20-line MNIST example, a raw loop is clearer. Dream Trainer's structure starts paying rent when the same trainer needs to run on 1 GPU in research, 8 GPUs in a pre-production test, and 512 GPUs in a real run without changing the body.
- **Not a replacement for `torch.distributed`.** Dream Trainer uses the PyTorch distributed primitives directly. We don't wrap them in a new language. Reading the Dream Trainer source means reading `DeviceMesh`, `DTensor`, `fully_shard`, `parallelize_module`, and `dcp.save` — the same names you'd use yourself.

## Where To Go Next

- [Core Concepts](core-concepts.md) — the mental model: lifecycle, device mesh, meta-device setup, mixins versus callbacks.
- [Quick Start](getting-started.md) — a single-file runnable trainer that makes the design decisions concrete.
- [Comparison](comparison.md) — how Dream Trainer sits next to Lightning, Accelerate, and torchtitan.
