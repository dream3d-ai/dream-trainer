# Performance Callback API

Modules:

- `dream_trainer.callbacks.profile`
- `dream_trainer.callbacks.benchmark`
- `dream_trainer.callbacks.optimize_fsdp`
- `dream_trainer.callbacks.graph_breaks`
- `dream_trainer.callbacks.gc`

Performance callbacks help inspect, measure, and tune training runs without embedding profiling logic in the trainer.

## Public Classes

| Class | Purpose |
| --- | --- |
| `ProfileCallback` | Records profiler traces. |
| `BenchmarkCallback` | Measures step timing after warmup. |
| `OptimizeFSDP` | Traces FSDP module order and configures prefetching. |
| `FindGraphBreaksCallback` | Records Dynamo graph-break explanations. |
| `GarbageCollectionCallback` | Controls garbage collection cadence. |

## Guidance

Use these callbacks during targeted investigations. Remove expensive profiling and graph-break callbacks from normal production runs unless you intentionally want ongoing traces.

## See It In Use

- [Callbacks — Debugging stack](../../callbacks.md#pick-a-stack) — the recommended callback bundle for tracking down throughput cliffs.
- [Performance](../../performance.md) — throughput tuning: FSDP prefetch, compile modes, activation checkpointing tradeoffs.
- [Debugging](../../debugging.md) — when a parallelism mode silently misbehaves.
