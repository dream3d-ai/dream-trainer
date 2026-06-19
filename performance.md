# Performance

!!! abstract "TL;DR"
    - **Optimize after the trainer is correct.** Change one knob at a time.
    - Enable `compile` first (largest single win). Then measure. Then enable the rest.
    - Use **callbacks**, not hand-rolled timers: `ProfileCallback`, `BenchmarkCallback`, `OptimizeFSDP`.
    - Know your bottleneck before reaching for activation checkpointing or gradient accumulation — they trade throughput for memory.

Optimize Dream Trainer runs **after** the trainer is correct. Start with a single working run, then turn on compile, parallelism, checkpointing, profiling, and FSDP optimizations deliberately.

## First principles

- Keep the model-specific policy explicit.
- Change one performance setting at a time.
- Measure before and after each change.
- Keep a debug path with `compile_model=False`.
- Use callbacks for profiling and benchmarking rather than ad-hoc timing in the trainer.

## Compile

Dream Trainer calls `apply_compile` when `device_parameters.compile_model=True`.

```python
def apply_compile(self):
    self.model.compile(mode="max-autotune-no-cudagraphs", dynamic=False)
```

=== "Debug"

    ```python
    DeviceParameters.SINGLE_DEVICE(compile_model=False)
    ```

=== "Production"

    ```python
    DeviceParameters.FSDP(compile_model=True)
    ```

!!! tip "Turn compile on after correctness is stable"
    Disable while diagnosing. Turn back on once forward, backward, validation, and checkpointing are all stable.

## Compiled autograd

```python
DeviceParameters(
    compile_model=True,
    compiled_autograd=True,
)
```

!!! warning "Keep a non-compiled fallback"
    Compiled autograd can improve throughput but makes debugging harder. Use graph-break tools and a non-compiled fallback path.

## Async tensor parallelism

`async_tensor_parallel=True` requires `compile_model=True`. Dream Trainer enables symmetric memory for the tensor-parallel process group and configures PyTorch compile for async tensor parallelism when a `tp` mesh exists.

!!! danger "Stable TP first"
    Use this only after tensor-parallel layout and compile behavior are stable.

## FSDP prefetch

`OptimizeFSDP` traces FSDP module execution and configures prefetching:

```python
callbacks.OptimizeFSDP(prefetch=1)
```

| `prefetch` | Memory | Overlap |
| --- | --- | --- |
| 0 | Lowest | Minimal |
| 1 (default) | Low | Good |
| 2+ | Higher | Best (when memory permits) |

Start with `prefetch=1`, measure, then increase only if memory allows.

## Activation checkpointing

Trades compute for memory:

```python
DeviceParameters.FSDP(checkpoint_activations=True)
```

```python
def apply_activation_checkpointing(self):
    ...
```

!!! warning "Know your bottleneck"
    Use activation checkpointing when activation memory is the bottleneck. Expect slower iteration time if recomputation dominates.

## Gradient accumulation

Use `gradient_accumulation_steps` to increase effective batch size without increasing memory:

```python
TrainingParameters(gradient_accumulation_steps=4)
```

In `training_step`, always use the canonical pattern:

```python
self.backward(loss)
if not self.is_accumulating_gradients:
    self.step(self.optimizer)
```

Dream Trainer controls gradient scaling and distributed sync behavior around this pattern.

## Profiling

Use profiling callbacks instead of hand-written timers:

| Callback | Use for |
| --- | --- |
| `ProfileCallback` | Full torch profiler trace (CPU + CUDA, kernel timings, memory). |
| `BenchmarkCallback` | Repeated step timing with warmup and statistics. |

```python
callbacks.ProfileCallback(...)
callbacks.BenchmarkCallback(...)
```

!!! tip "Measure after warmup"
    Compile, lazy initialization, and first-checkpoint setup can distort early steps. Both callbacks accept a warmup window.

!!! tip "Ad-hoc profiling / benchmarking"
    If `[cli]` is installed, you can run these without editing the trainer: `python train.py profile` or `python train.py benchmark`. See [Using the CLI](cli.md).

## Checkpoint throughput

Large checkpoints can block training. Use async checkpointing when save time is a bottleneck:

```python
callbacks.AsyncCheckpointCallback(checkpoint_parameters)
```

Keep `keep_top_k` bounded so cleanup doesn't become a storage problem. Monitor checkpoint timing separately from model iteration timing.

## Communication timeouts

```python
DeviceParameters(
    comm=Comm(init_timeout_seconds=600, train_timeout_seconds=300)
)
```

Use longer init timeouts for first-step compilation and lazy setup. Tighter training timeouts surface stalled collectives sooner after warmup.

## Throughput checklist

- [ ] Confirm dataloader latency with `ProgressBar`.
- [ ] Profile one representative step after warmup.
- [ ] Turn on compile only after correctness is stable.
- [ ] Use DDP before FSDP if memory allows; use FSDP when memory requires sharding.
- [ ] Use activation checkpointing only when memory-bound.
- [ ] Use async checkpointing for large checkpoint writes.
- [ ] Revisit batch size, gradient accumulation, and data prefetch together.
- [ ] Keep media logging sparse.
- [ ] Measure with and without each callback that does heavy work.

## Next steps

- [Debugging](debugging.md) — when an optimization changes behavior.
- [Parallelism](parallelism.md) — when scaling to larger device meshes.
- [Using the CLI](cli.md) — ad-hoc benchmark / profile / find-graph-breaks.
