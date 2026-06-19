---
title: Callbacks
---

# Callbacks

<small>🛠️ How-to · copy-paste friendly</small>

!!! abstract "TL;DR"
    - **Callbacks care about *when*; hooks care about *what*.** Checkpointing, logging, progress, profiling → callback. Parameter groups, state-dict layout → hook.
    - Compose with `callbacks.CallbackCollection([...])` on the config.
    - Start with the **Minimum viable stack** (Logger + ProgressBar), grow to the **Production stack** (+ Checkpoint + LRLogger + TrainerSummary), add **Diagnostics** (Profile, Benchmark, FindGraphBreaks) as needed.
    - Custom callbacks implement `state_dict` / `load_state_dict` to participate in checkpointing.

Your trainer works. Now you need checkpoints, progress bars, WandB logging, a learning-rate scheduler log, EMA, FP8 quantization, and a profiler — without turning `training_step` into a 500-line method and without `configure_models` knowing about any of them. That's what callbacks are for.

Callbacks hook into the lifecycle around your trainer. Your trainer doesn't know they exist. You can run the same trainer with a different callback stack for research and for production and the training logic doesn't change.

## Hooks vs Callbacks — The Rule

Dream Trainer distinguishes extension points by what they care about:

- **Hooks live on the trainer.** They care about what your model *is* — parameters, layers, state dict layout. `configure_models`, `training_step`, `model_state_dict`, `apply_fully_shard` are hooks.
- **Callbacks live outside the trainer.** They care about what the trainer is *doing* — "before every optimizer step", "at the end of each validation epoch". They don't need to know what your model computes.

A custom gradient-clipping scheme is a hook (it needs parameter groups). "Save a checkpoint every 1000 steps" is a callback (it only needs to know when step 1000 happens).

## Pick A Stack

Your callback stack depends on what you're doing with the trainer. Three canonical stacks cover most cases.

=== "Minimum viable"

    For a single-GPU dev loop where you just want to see it train. Nothing else.

    ```python
    from dream_trainer import callbacks

    callbacks.CallbackCollection([
        callbacks.LoggerCallback(log_every_n_train_batches=8),
        callbacks.ProgressBar(metric="train/loss"),
    ])
    ```

    That's it. Loss logs to stdout, progress bar shows throughput. Add things when you need them, not preemptively.

=== "Production"

    A real run with checkpointing, async save, WandB logging, LR tracking, and progress.

    ```python
    from dream_trainer import callbacks
    from dream_trainer.configs import CheckpointParameters

    callbacks.CallbackCollection([
        callbacks.LoggerCallback(log_every_n_train_batches=20, code_dir="../"),
        callbacks.LRLoggerCallback(),
        callbacks.ProgressBar(metric="train/loss"),
        callbacks.AsyncCheckpointCallback(
            CheckpointParameters(
                enable=True,
                root_dir="/checkpoints/my-run",
                monitor="val/loss",
                resume_mode="min",
                checkpoint_every_n_val_epochs=1,
                keep_top_k=3,
            ),
        ),
        callbacks.ModelWatchCallback(),
    ])
    ```

    `AsyncCheckpointCallback` is strictly better than `CheckpointCallback` once your save is non-trivial — saving on the main rank in-band stalls training; async saves in the background.

=== "Debugging"

    When something is wrong — slow training, recompiles, memory creep, or an unexplained throughput cliff.

    ```python
    from dream_trainer import callbacks

    callbacks.CallbackCollection([
        callbacks.LoggerCallback(log_every_n_train_batches=1),     # log every step
        callbacks.ProgressBar(metric="train/loss"),
        callbacks.FindGraphBreaksCallback(),                       # recompile culprits
        callbacks.ProfileCallback(                                 # torch.profiler traces
            wait=3, warmup=2, active=4, repeat=1,
            output_dir="/tmp/profile",
        ),
        callbacks.BenchmarkCallback(warmup_steps=20, measure_steps=50),
        callbacks.GarbageCollectionCallback(n=200),                # predictable GC
    ])
    ```

    Run this stack for a few hundred steps, read the trace in Chrome tracing or Perfetto, then go back to the production stack.

!!! tip "Swap stacks with config, not code"
    The callback list is a `CallbackCollection` inside your config, not inside the trainer class. Swapping `debug_config()` for `production_config()` is the only change you need to make to switch stacks.

## The Callback Interface

A callback is a class inheriting `Callback` that implements whichever lifecycle hooks it needs. Only overridden methods get called — `CallbackCollection` inspects each callback once at init time and builds a dispatch table.

```python
from dream_trainer.callbacks import Callback


class LogEveryHundredSteps(Callback):
    def post_train_step(self, batch, result, batch_idx):
        if batch_idx % 100 == 0:
            print(f"step {batch_idx}: {result['train/loss']:.4f}")
```

That's the whole contract.

### Available hook points

| Phase | Hooks |
| --- | --- |
| Launch | `pre_launch` |
| Configure | `pre_configure`, `post_configure` |
| Setup | `pre_setup`, `post_setup` |
| Fit | `pre_fit`, `post_fit` |
| Epochs | `pre_epoch`, `post_epoch`, `pre_train_epoch`, `post_train_epoch`, `pre_validation_epoch`, `post_validation_epoch` |
| Steps | `pre_train_step`, `post_train_step`, `pre_validation_step`, `post_validation_step` |
| Optimizer | `pre_optimizer_step`, `post_optimizer_step`, `pre_optimizer_zero_grad`, `post_optimizer_zero_grad` |
| Interrupts | `on_interrupt` |
| Context | `train_context`, `validation_context` |

If a behavior doesn't have a natural home in this list, it probably belongs in a trainer hook instead.

### Rank-zero work

Any I/O-heavy callback (logging to stdout, writing to a shared filesystem) should run only on global rank zero. Inherit from `RankZeroCallback` and the dispatch is filtered automatically:

```python
from dream_trainer.callbacks import RankZeroCallback


class PrintEveryEpoch(RankZeroCallback):
    def post_train_epoch(self):
        print("epoch done")
```

### Trainer dependencies

Callbacks can declare the trainer interface they require. This is how `WandBLoggerCallback` guarantees the trainer has the `WandBLoggerMixin`:

```python
class MyCallback(Callback[DreamTrainer]):
    def post_train_step(self, batch, result, batch_idx):
        self.trainer.log_dict({"custom": result["train/loss"]})  # type-checked
```

If you attach a callback to a trainer that doesn't satisfy its declared interface, `CallbackCollection` raises at init time — not at the first call in the middle of training.

### Callback state

A callback that owns resumable state (EMA weights, curriculum step counters, adaptive schedules) should implement `state_dict` and `load_state_dict`. Dream Trainer includes callback state in trainer checkpoint state, so resumed runs pick up where they left off.

```python
class MyCurriculum(Callback):
    def __init__(self):
        self.stage = 0

    def state_dict(self):
        return {"stage": self.stage}

    def load_state_dict(self, state):
        self.stage = state["stage"]
```

## Built-in Callback Catalogue

A quick reference for what ships in the box. See the API docs under [Callbacks](api/callbacks/base.md) for full signatures.

| Category | Callback | What it does |
| --- | --- | --- |
| Logging | `LoggerCallback` | Scalar train/val logs from step returns. |
| | `LRLoggerCallback` | Logs optimizer learning rates. |
| | `MediaLoggerCallback` | Logs image/video batches. |
| | `ModelWatchCallback` | WandB model watch. |
| Progress | `ProgressBar` | Per-epoch train/val progress. |
| | `TrainerSummary`, `ModelSummary` | Structure summaries when `rich` is installed. |
| Checkpointing | `CheckpointCallback` | Saves trainer state via DCP. |
| | `AsyncCheckpointCallback` | Non-blocking save + background cleanup. |
| | `LoadPartialCheckpointCallback` | Load model weights only (fine-tuning). |
| | `ExportCallback` | Export supported modules from a checkpoint. |
| Debugging | `FindGraphBreaksCallback` | Inspect compiled regions for graph breaks. |
| | `ProfileCallback` | `torch.profiler` traces. |
| | `BenchmarkCallback` | Timed measurement after warmup. |
| Performance | `OptimizeFSDP` | Traces FSDP execution; configures prefetch. |
| | `GarbageCollectionCallback` | Predictable GC cadence. |
| Training extensions | `EMACallback` | Exponential moving averages of parameters. |
| | `Fp8Quantization` | torchao FP8 integration. |
| | `FaultToleranceCallback` | torchft integration (when installed). |
| | `WeightTransferCallback` | Transfer weights between modules. |

## When Not To Use A Callback

Callbacks are for reusable lifecycle behavior. Do not reach for a callback to implement:

- Model architecture or weight initialization → trainer hook (`configure_models`, `init_weights`).
- Parameter groups for optimizers → `configure_optimizers`.
- Tensor-parallel or FSDP wrapping policy → `apply_tensor_parallel`, `apply_fully_shard`.
- Loss computation or validation semantics → `training_step`, `validation_step`.

The test is simple: if implementing the behavior requires reaching into `self.model`'s layers or parameters, it's a hook, not a callback.

## Next Steps

- [Checkpointing](checkpointing.md) — full DCP state layout, resume modes, partial load.
- [Logging And Metrics](logging-metrics.md) — how step returns flow into logger callbacks.
- [Trainer Guide](trainer-guide.md) — the hook side of the ownership split.
