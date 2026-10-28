# Native trainer upgrade acceptance

Measured on two RTX PRO 6000 Blackwell Max-Q GPUs with PyTorch 2.14.0+cu130 and
Python 3.14.2. The baseline is dream-trainer commit
`7b306a4dbb31ac8d8291a7370d7c86a03c3dd2db`; both packages use the same installed
PyTorch, workspace model, optimizer and captured FineWeb inputs.

The [benchmark](../../native_upgrade.py) runs the real Slowrun trainer and Muon
optimizer with four layers, width 128, two attention heads, 128-token sequences,
the full GPT-2 vocabulary and one sequence per rank. Deterministic algorithms,
ATen GEMMs, full causal attention, and disabled dropout hold numerical controls
fixed. The production compile hook and optimizer foreach mode remain active.
This is a reduced workload measuring trainer overhead, not training quality or
full-model throughput.

Each arm ran in three fresh two-rank processes. Each process used new Inductor
and Triton directories. Cache-load arms imported only their corresponding
per-rank compiler bundles. There were 128 updates per run: 16 warmup updates,
111 measured updates, and one final profiled update. Logging and checkpoint saves
were disabled. Compiler export happened on the first update and at shutdown,
outside the steady-state window.

| Configuration | First update, median seconds | Tokens/s, median | Tokens/s range | Peak allocated GPU MiB |
| --- | ---: | ---: | ---: | ---: |
| Baseline NCCL | 63.1 | 12,890 | 12,221–14,275 | 433.0 |
| Updated NCCL, cold cache/export | 63.9 | 14,695 | 14,355–14,856 | 433.0 |
| Updated NCCL, loaded cache | 21.3 | 15,243 | 14,878–15,533 | 433.0 |
| Updated NCCL2, cold/no export | 63.9 | 15,419 | 13,953–15,610 | 491.3 |

First-update timing starts immediately before trainer construction, so it excludes
Python imports and configuration construction. It includes model setup,
compilation, the optimizer update and any first-update compiler export. Each
trial uses the slower rank's throughput and larger rank latency; the table gives
medians over trials. Peak memory is the maximum PyTorch allocated-memory counter
over ranks and trials after setup, not total device residency or host memory.

The updated NCCL arm has 14.0% higher median steady-state throughput here.
Compiler-cache loading cuts first-update time by 66.7%, approximately 3× faster.
NCCL2's throughput range overlaps updated NCCL and its measured allocated-memory
peak is higher. It remains opt-in; these results do not establish a general NCCL2
speedup. Short measurement windows and three repetitions limit extrapolation.

All 24 per-rank reports have exactly matching inputs, initial/final model hashes,
optimizer/scheduler hashes, and full loss sequences relative to the corresponding
baseline rank. Whole-update host scalar reads fall from 161 to 98. The focused
profiler test separately verifies that successful parameter/gradient validation
uses one scalar read regardless of parameter count; the norm check is separate.

Acceptance also covers:

- 88 unit tests, including exact updates and resume, accumulation, unused/frozen
  parameters, independent optimizers, mixed CPU/CUDA state, callback ordering,
  corrupt/incompatible cache files, atomic-write failure, disabled compilation,
  and real fresh-process cache hits after changes to shapes or code constants.
- Eight two-rank configurations: NCCL/NCCL2 × DDP, FSDP, TP, and BF16 FSDP.
  Each checks exact checkpoint continuation and optimizer state, four injected
  failures before optimizer mutation, backend selection through flattened meshes,
  teardown, and a successful fresh world. Continuation losses also match exactly
  between NCCL and NCCL2 for each configuration.
- Ruff on changed Python files, `git diff --check`, and the workspace's
  `uv lock --check --offline`.

`summary.json` contains unrounded aggregates. The adjacent per-rank JSON files
and command records retain timings, compiler counters, state hashes and launch
environments. `manifest.json` identifies the source files and baseline revision.
Raw logs and temporary compiler/checkpoint artifacts remain under
`/tmp/dream-trainer-native-upgrade`; binaries are not included in this report.

The existing timeout tests were corrected to initialize their bypassed constructor
state and check timeout reduction after the first complete optimizer update.
Their two previous failures were reproduced against the archived baseline.
No graph execution engine, cancellation supervisor, TorchFT recovery, checkpoint
staging redesign, or logger redesign is included in this change.
