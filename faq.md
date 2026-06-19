---
title: FAQ
---

# FAQ

<small>🛠️ Reference · common questions, terse answers</small>

Grouped by topic. When a question has a longer answer elsewhere, the answer links out.

## Positioning

### Who is Dream Trainer for?

Advanced PyTorch users who need reusable distributed training infrastructure without hiding the model, optimizer, dataloader, or training step behind a monolithic abstraction. If you've outgrown one-off `train.py` scripts but don't want a framework that rewrites your `forward`, this is aimed at you.

### Is Dream Trainer a replacement for PyTorch?

No. It's a PyTorch-native substrate. Models remain `nn.Module` objects, training steps are ordinary Python, distributed behavior sits on PyTorch primitives (DTensor, FSDP2, composable DDP, DCP). See [Design Philosophy](design-philosophy.md).

### Why not just write a raw PyTorch loop?

Do, for tiny scripts. Reach for Dream Trainer when you repeatedly need: launch/setup ordering, device mesh validation, gradient accumulation, sanity validation, validation loops, metric reset cadence, rank-aware dataloaders, callback lifecycle, DCP checkpointing, and the ten other invariants that go wrong silently.

### Why not Lightning, Accelerate, or DeepSpeed?

Those fit when their abstractions match. Dream Trainer fits better when you want explicit control over modern PyTorch distributed features (DTensor mesh, FSDP2, async TP, context parallelism), model-specific parallelism hooks, and production lifecycle without changing the shape of your model code. See [Comparison](comparison.md).

### Does Dream Trainer own my model?

No. You construct normal PyTorch modules in `configure_models`. Dream Trainer materializes, places, shards, compiles, and checkpoints them according to your hooks and config.

## Installation And Environment

### Which Python and PyTorch versions are supported?

Python 3.11+ and a recent PyTorch nightly with FSDP2, DTensor, and `torch.compile` support. See [Installation](installation.md) for the current matrix.

### How do I install optional extras?

```bash
pip install "dream-trainer[wandb,metrics,fp8,cli]"
```

Extras are opt-in on purpose — quantization, fault tolerance, and CLI helpers pull heavy dependencies that most runs don't need. See [Installation](installation.md) for the extras table, and [Using the CLI](cli.md) for what the `[cli]` extra unlocks.

## Lifecycle

### Why does `init_weights` exist separately from `configure_models`?

`configure_models` runs under the meta device — parameters have shapes but no storage. Dream Trainer then applies parallelism hooks and materializes tensors on the real device. Only after materialization can weights be initialized. If you call `nn.init.*` inside `configure_models`, you initialize meta tensors and the results are silently discarded. See [Core Concepts — Meta Device, Then Real Device](core-concepts.md#meta-device-then-real-device).

### Why does `training_step` call `self.backward` instead of `loss.backward()`?

`self.backward(loss)` scales for gradient accumulation. Calling `loss.backward()` directly breaks accumulation silently — the gradients aren't scaled and your effective batch size drifts from what the config says.

### Why does `self.step` take an optimizer?

`self.step(optimizer)` is where gradient validation, gradient clipping, optimizer stepping, scheduler stepping, optimizer callbacks, and zeroing gradients all happen atomically for one optimizer. Calling `optimizer.step()` directly skips all of that.

### What order do the lifecycle phases run in?

`launch → mesh → configure (meta) → apply_pipeline_parallel → apply_tensor_parallel → apply_activation_checkpointing → apply_compile → apply_fully_shard / apply_replicate → materialize → init_weights → configure_optimizers → configure_schedulers → configure_dataloaders → configure_metrics → sanity validation → fit`. The diagram on [Core Concepts](core-concepts.md#the-lifecycle) is authoritative.

### Can I use multiple models or optimizers?

Yes. Assign each as a trainer attribute. Return a model-to-optimizer mapping from `configure_optimizers`, and return stable model keys from `model_state_dict`. See [Trainer Guide — Multiple Models And Frozen Modules](trainer-guide.md#multiple-models-and-frozen-modules).

## Parallelism

### Should I start with DDP or FSDP?

Start with whatever fits the model + one batch on a single GPU. Then: DDP if the replicated model fits per-GPU; FSDP if it doesn't. See [Parallelism — Pick A Mode First](parallelism.md#pick-a-mode-first).

### Why is my compiled FSDP run slower than non-compiled?

Usually one of: compile is wrapping too small a region (try `apply_compile` on the whole model before FSDP); dynamic shapes are triggering recompiles (`FindGraphBreaksCallback` shows them); or async TP is off while compile is on. Run `BenchmarkCallback` to get a stable number.

### Can I switch from DDP to FSDP without retraining?

Yes. DCP checkpoints are mesh-shape-agnostic. Save under DDP (`dp_replicate=4`), load under FSDP (`dp_shard=4`), and the state gathers correctly. Keep `model_state_dict` keys stable between runs.

### Does Dream Trainer support tensor parallelism out of the box?

It supports the `tp` mesh dimension and calls your `apply_tensor_parallel(tp_mesh)` hook. You define the plan — Dream Trainer does not infer layouts because the correct policy depends on which tensors share contraction dimensions. See [Parallelism — Tensor Parallelism](parallelism.md#tensor-parallelism).

### Can I combine FSDP with tensor parallelism?

Yes. Set both `dp_shard > 1` and `tensor_parallel > 1` in `DeviceParameters`. Implement both `apply_fully_shard` and `apply_tensor_parallel`. Dream Trainer runs them in the correct order.

## Checkpointing

### Should I use checkpoint callbacks or `torch.save`?

Use checkpoint callbacks for training state — they integrate trainer counters, optimizer state, scheduler state, callback state, and DCP sharded model state. Use `torch.save` only for isolated artifacts that are not part of trainer resume.

### When should I use `AsyncCheckpointCallback` instead of `CheckpointCallback`?

As soon as save time is non-trivial (say, >1 second). Synchronous saves stall the rank-zero training loop; async saves run in a background thread. They share the same `CheckpointParameters`, so the switch is one line.

### How do I resume from a checkpoint?

Enable `CheckpointParameters(enable=True, resume_mode="last")` (or `"min"`/`"max"` by `monitor`). When the trainer starts, it checks `root_dir` and resumes automatically if a checkpoint is present. No code changes between a fresh run and a resume.

### Can I load only model weights (e.g., for fine-tuning)?

Yes — use `LoadPartialCheckpointCallback`. It loads model state without trainer/optimizer/scheduler/callback state, which is the usual shape for starting a new fine-tuning run from a pretrained checkpoint.

## Logging And Metrics

### How do I disable WandB?

```python
WandbLoggingParameters(enabled=False)
```

The package may still require the WandB extra to be installed for the current top-level import path, but logging stays off for local runs.

### Where do scalar logs come from?

Whatever `training_step` and `validation_step` return. `LoggerCallback` reads the dict, logs each key on its configured cadence, and dispatches to the configured backend (stdout, WandB, or both). Use stable keys (`train/loss`, `train/grad_norm`, `val/loss`) — checkpoint monitors and dashboards depend on them.

### How do I log images or videos?

Return them from `training_step`/`validation_step` under a key like `media/samples`, and add `MediaLoggerCallback(log_every_n_train_batches=500)`. Keep media logging sparse — it's expensive in distributed runs.

### Why aren't my metrics updating?

Three common causes: (1) metrics assigned in `configure_models` instead of `configure_metrics` (wrong device); (2) `update()` called in `training_step` instead of `validation_step` (Dream Trainer only resets+computes at validation boundaries); (3) metric state not registered with `add_state(... dist_reduce_fx=...)` in a custom metric (it won't gather across ranks).

## Performance And Debugging

### Can I run without compile?

Yes:

```python
DeviceParameters.SINGLE_DEVICE(compile_model=False)
# or
DeviceParameters.FSDP(compile_model=False, async_tensor_parallel=False)
```

For any mode with `async_tensor_parallel=True`, compile must also be on — they're paired.

### How do I profile a slow step?

Add `ProfileCallback(wait=3, warmup=2, active=4, output_dir="/tmp/profile")` for a few hundred steps, then open the trace in Chrome tracing or Perfetto. `BenchmarkCallback(warmup_steps=20, measure_steps=50)` gives a single warmed-up timing number for A/B comparison.

### My FSDP run has too many graph breaks. What do I do?

`FindGraphBreaksCallback` prints every Dynamo graph break with its source location. Common culprits: Python-level branches on tensor values, eager fallback ops inside `training_step`, dynamic shapes on input dims. Fix them by making shapes static where possible and hoisting non-tensor branches out of the compiled region.

### How do I catch memory creep during long runs?

`GarbageCollectionCallback(n=200)` forces GC on a predictable cadence, which also smooths step-time variance. If memory still grows, the leak is usually accumulating Python objects (lists, dicts) inside `training_step` — move them out or clear them per step.

## Extending

### Can I write my own callback?

Yes — subclass `Callback` and implement whichever lifecycle hooks you need. `CallbackCollection` inspects each callback once and builds a dispatch table, so unused hooks cost nothing. See [Callbacks — The Callback Interface](callbacks.md#the-callback-interface).

### How do I make a callback survive a restart?

Implement `state_dict()` and `load_state_dict(state)`. Dream Trainer includes callback state in trainer checkpoint state, so resumed runs pick up where they left off. See [Tutorial 5 — A Curriculum Callback](tutorials/custom-components.md#a-curriculum-callback).

### Can I mix custom mixins into a trainer?

Yes. Subclass `DreamTrainer` with your mixin in the MRO. Keep mixins narrow — one concern per mixin. If the mixin needs to reach deep into `self.model`, it's probably a hook, not a mixin.

### Why are API pages hand-written?

Mkdocstrings is configured, but generated pages require the docs environment to import Dream Trainer under a supported Python version with the optional dependencies installed. The current pages are written by hand so strict docs builds remain reliable during the docs rebuild.
