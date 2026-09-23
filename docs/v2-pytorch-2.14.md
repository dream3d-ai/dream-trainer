---
title: Dream Trainer v2 — PyTorch 2.14 Execution Design
---

**Status: design proposal, September 16, 2026.** Complements the [v2 API proposal](v2-proposal.md). Dream Trainer APIs below are proposed, not implemented. Source review does not establish runtime correctness or performance.

See [how v2 fits together](v2-overview.md) for the integrated public API and lifecycle.

**Scope revision, September 17:** follow the [define-once parallelism contract](../../dream-trainer-v2/docs/define-once-parallelism.md)
for the product boundary. The library must resolve reusable model layout declarations and run
configuration into a supported N-D execution plan. The earlier project-owned preparation
approach below is an advanced extension path, not the default scaling experience.

**Scope follows the simplifying assumptions in the main proposal.** The feature survey and structural discussion below identify implementation choices and extension opportunities, not a requirement to ship every option. Start with CUDA/NCCL, BF16, fixed membership and parameter ownership, direct Python training, and regional compilation. Internal execution records can be small concrete dataclasses. Defer a general executor/plugin framework, backend capability registry, custom-stream dependency graph, and per-resource checkpoint hazard tracking until a measured workload requires them. Model-specific native PyTorch configuration remains ordinary preparation code.

**Support starts at PyTorch 2.14.0.** Declare `torch>=2.14.0` in v2 packaging and reject older installations before creating process groups. Test the minimum release and each newer release claimed as supported; isolate changes to unstable APIs in focused adapters. Start compiled CI on Python 3.12, with other Python versions admitted through the same checks. PyTorch 2.14's Python 3.15 support is eager-only. Eager execution remains a first-class debugging mode on the supported PyTorch baseline. [Release notes](https://github.com/pytorch/pytorch/releases/tag/v2.14.0).

The reference source is PyTorch tag `v2.14.0`, commit `2b3ec34829036a65cd9d1398ea72a0167dc37470`. TorchTitan was reviewed at `9a80ce528a6ec9f08b3d6f9d171bb2648ff839a5`. Its main CI includes nightly environments and separate release-validation jobs pinned to 2.14.0. Borrow implementation ideas with their tests and dependencies; a feature on TorchTitan main is not automatically a released PyTorch contract. [Nightly CI](https://github.com/pytorch/torchtitan/blob/9a80ce528a6ec9f08b3d6f9d171bb2648ff839a5/.github/workflows/unit_test_cpu.yaml), [release GPU workflow](https://github.com/pytorch/torchtitan/blob/9a80ce528a6ec9f08b3d6f9d171bb2648ff839a5/.github/workflows/validate_release_gpu.yml).

**The user-facing style and compiler boundaries serve different purposes.** The same operations can be composed through functions or an optional Trainer class. `Runtime` prepares execution resources and `Update` owns optimizer state and semantics in both forms. Tensor-heavy regions are the compiler targets, independently of whether the calling algorithm is a function or method. The managed runner, checkpoint registry, and execution dependencies are shared. The user can change an objective or add a discriminator without implementing a graph scheduler or adopting lifecycle mixins.

For the initial API, `CompilePolicy` is a closed union of immutable `Eager` and `RegionalCompile` dataclasses. Exhaustive handling should expose an unsupported policy to the type checker. Training execution is a separate concern: the default directly calls `train_step`, while pipeline schedules and joint graphs require explicit executor contracts. The proposed usage is:

```python
runtime = dt.RuntimeConfig(
    distributed=dt.FSDP(shard="auto"),
    precision=dt.Precision.bf16(),
    compile=dt.RegionalCompile(fullgraph=True),
)
```

The model-specific preparation function chooses concrete regions. For example, this fragment is suitable for a model whose adapter has validated fixed or bucketed block input shapes:

```python
def compile_dit_blocks(model: DiT, runtime: dt.Runtime) -> DiT:
    policy = runtime.compile_policy
    if isinstance(policy, dt.RegionalCompile):
        for block in model.blocks:
            block.compile(fullgraph=policy.fullgraph, dynamic=False)
    return model
```

This fragment is the compilation part of preparation, not a universal FSDP/checkpointing order. Prefer placing it directly in `prepare_dit` unless it is reused. `Module.compile()` installs compilation on the existing module's call path, allowing the trainer to retain its concrete object and custom methods. It does not repair PyTorch's untyped module-call boundary; retain the typed integration checks from the main proposal. This API predates 2.14. [Tagged module implementation](https://github.com/pytorch/pytorch/blob/v2.14.0/torch/nn/modules/module.py).

Keep experimental shape decorators and compiler option dictionaries inside typed adapters. A function-compilation adapter must preserve its input/output signature; a module adapter must preserve the declared concrete interface. Do not let an upstream `Any` annotation erase types throughout an experiment. Small contract fixtures should check the inferred types after each such boundary.

**Adopt the following execution contracts.**

| Area | Proposed v2 behavior | PyTorch basis |
| --- | --- | --- |
| Sharding | Native FSDP2 `fully_shard`, named DeviceMesh dimensions, DTensor placements, explicit reshard/prefetch and reduction precision. No FSDP1 compatibility layer. | Existing foundations available through the [2.14 public FSDP namespace](https://github.com/pytorch/pytorch/blob/v2.14.0/torch/distributed/fsdp/__init__.py). |
| Accumulation | Each Update controls its own participating FSDP modules. Distinguish delaying all synchronization from delaying only HSDP all-reduce; expose the memory tradeoff. | [`set_requires_gradient_sync` and `set_requires_all_reduce`](https://github.com/pytorch/pytorch/blob/v2.14.0/torch/distributed/fsdp/_fully_shard/_fully_shard.py). |
| Compilation | Compile selected model/loss regions with explicit graph-break policy. Keep data iteration, phase changes, logging, and checkpoint coordination outside. | In-place module compilation; [2.14 compiler configuration](https://github.com/pytorch/pytorch/blob/v2.14.0/torch/compiler/config.py). |
| Shapes | Declare fixed, bucketed, or dynamic inputs per model adapter. Bound expected specializations and measure recompiles across training and evaluation. Dynamic dimensions are independent of loss normalization. | New experimental [`dynamic_spec`](https://github.com/pytorch/pytorch/blob/v2.14.0/torch/fx/experimental/dynamic_spec.py) attaches shape contracts, currently using unbacked symbols. |
| Communication overlap | Preserve upstream compiler defaults and profile actual exposed communication. FSDP prefetch remains a separate model-adapter choice. | Inductor's [`enable_simple_overlap=True`](https://github.com/pytorch/pytorch/blob/v2.14.0/torch/_inductor/config.py) reorders starts/waits when collectives are visible in its graph. |
| Optimizers | Use ordinary PyTorch optimizers. Select fused/foreach implementations per validated optimizer and layout. Compile the optimizer separately only when its mutation, LR, and scheduler behavior pass checks. | Update semantics remain explicit; compilation must not absorb or reorder multiple optimizer phases. |
| Activation memory | Non-reentrant checkpoint calls with explicit RNG preservation. Selective checkpointing/offload uses an explicit saved-tensor-hook policy and model-specific preparation order. | New [`respect_saved_tensors_hooks`](https://github.com/pytorch/pytorch/blob/v2.14.0/torch/utils/checkpoint.py) controls whether SAC-saved tensors honor surrounding hooks. |
| Checkpoints | Wrap DCP with a typed save handle separating snapshot completion from durable completion. Wait before conflicting model, optimizer, buffer, or algorithm-state mutation. | [`AsyncSaveResponse`](https://github.com/pytorch/pytorch/blob/v2.14.0/torch/distributed/checkpoint/state_dict_saver.py) exposes staging and upload futures; the async API is experimental and needs an adapter. |
| Process groups | Runtime owns initialization, timeouts, diagnostics, and teardown. Use public timeout management. | New [`torch.distributed.set_timeout`](https://github.com/pytorch/pytorch/blob/v2.14.0/torch/distributed/distributed_c10d.py) replaces reliance on the current library's private timeout call. |

BF16 is the initial GPU recipe baseline, with parameter storage, compute, and gradient reduction dtypes specified separately. FP8/MXFP8 recipes belong to model adapters with pinned TorchAO dependencies and matched quality checks. A kernel supporting a low-precision dtype is not evidence that a training recipe is numerically acceptable.

**Treat new optimizations as measured choices.**

- **NVGEMM:** offer an explicit autotuning option for eligible CUDA recipes. In 2.14 it is absent from the default GEMM candidate list, requires `cutlass.operators` and autotuning, and rejects unbacked GEMM dimensions and AOTInductor mode. Preserve ATen/Triton alternatives and report which kernels actually win. The new dynamic-shape feature and this GEMM backend therefore do not compose automatically. [Candidate configuration](https://github.com/pytorch/pytorch/blob/v2.14.0/torch/_inductor/config.py), [eligibility checks](https://github.com/pytorch/pytorch/blob/v2.14.0/torch/_inductor/utils.py).
- **Compile on one rank:** evaluate through a constrained artifact-building backend. Use `torch.compiler.config.compile_on_one_rank`; it makes captured device/group references rank-independent. It does not itself establish artifact distribution, cache validity, or semantic equivalence between ranks. [Compiler configuration](https://github.com/pytorch/pytorch/blob/v2.14.0/torch/compiler/config.py).
- **CUDA graphs:** opt in for regions with validated shapes, allocations, RNG behavior, and output lifetimes. Metric accumulation and retained outputs must not alias buffers overwritten on replay. Warmup is real execution: discarded warmup must restore the entire model/optimizer/data/RNG state, or run in an isolated benchmark process.
- **`nccl2`:** retain `nccl` for the initial compiled CUDA path. The rewritten backend is an eager-only preview in 2.14; validate it as a distinct experimental runtime option. Process-group reconfiguration alone does not deliver end-to-end recovery of sharded optimizer state and data progress. [2.14 distributed changes](https://pytorch.org/blog/pytorch-2-14-release-blog/).
- **Symmetric-memory communication:** enable only for a measured topology and supported collective path. The documented NCCL symmetric kernels require NCCL 2.27+, a single direct-NVLink domain, and `NCCL_WIN_ENABLE`; CUDA capability alone is insufficient. [Requirements in the 2.14 release article](https://pytorch.org/blog/pytorch-2-14-release-blog/).

Adaptive research loops stay ordinary Python by default. A newly capturable control-flow operator is not a blanket training contract: the tagged `torch.while_loop` documentation still warns about training support, despite containing autograd machinery. Adopt such operators only in an adapter with explicit gradient and capture tests. [Tagged while-loop implementation](https://github.com/pytorch/pytorch/blob/v2.14.0/torch/_higher_order_ops/while_loop.py).

**Prototype the next compiler step without making it a prerequisite.** TorchTitan's GraphTrainer is a useful comparison for joint forward/backward graphs, compiler-visible sharding, and ahead-of-time preparation. Inspect its SimpleFSDP and graph transformations as a separate execution design; it is not an alias for native FSDP2 plus `torch.compile`. The initial research task is to compare one fixed-shape diffusion workload against the regional FSDP2 path using identical math, update order, and data. Only add a public joint-graph execution policy after establishing supported mutation, RNG, accumulation, and checkpoint contracts. [GraphTrainer implementation at the reviewed commit](https://github.com/pytorch/torchtitan/tree/9a80ce528a6ec9f08b3d6f9d171bb2648ff839a5/torchtitan/experiments/graph_trainer).

<a id="structural-changes"></a>

**Structural changes: make ownership and dependencies explicit.** Inspection of the current library found overlap mechanisms already present: model setup enables separate reduce-scatter groups, and `OptimizeFSDP` configures prefetch. However, that callback also unshards the roots of every registered model before a step, and learns an order through temporary hooks. The accumulation helper uses one trainer-wide transition flag. These mechanisms need coordinated, phase-aware ownership. This is a code finding, not a measurement of their performance. [Model setup](../src/dream_trainer/trainer/mixins/setup/models.py), [prefetch callback](../src/dream_trainer/callbacks/optimize_fsdp.py), [training control flow](../src/dream_trainer/trainer/base.py).

| Component | Change | Contract |
| --- | --- | --- |
| Trainer | Keep the shallow typed class. | Owns objective definitions, phase order, and update cadence through direct calls. |
| Model preparation | Add an internal typed model execution record to explicit registration. | Maps concrete modules to sharding units, compiled regions, persistent layouts, and synchronization domains. These mappings need not be one-to-one. |
| `OptimizeFSDP` callback | Move its useful mechanisms into the model's execution adapter. | Prefetch only the relevant phase's units with an explicit memory policy; use native implicit scheduling where an explicit order is unvalidated. |
| Update | Delegate model synchronization; retain optimizer semantics. | Requires ready gradients and snapshot-safe writes, publishes parameter readiness, then advances declared update state. |
| DistributedWorld | Split resource ownership from model execution. | Runtime owns named meshes, process groups, backend capabilities, and shutdown; model adapters own communication use and ordering. |
| Async checkpoint callback | Retain storage/staging code behind a runtime service. | Correctness waits cannot depend on callback registration order. |
| Data and metrics | Add bounded device prefetch and detached device-side reductions. | Explicit stream/lifetime ownership, memory limits, and checkpointable consumption progress. |

**Tune communication units independently of compilation.** `fully_shard` groups determine collective granularity. A model adapter should choose them bottom-up, record prefetch relationships where execution order is known, and expose memory choices separately from compilation region size. An entire model in one communication group can leave little compute available to hide its transfer; excessively small groups incur launch/latency costs. Profile group sizes and exposed waits on the target topology. [FSDP grouping and prefetch APIs](https://github.com/pytorch/pytorch/blob/v2.14.0/torch/distributed/fsdp/_fully_shard/_fully_shard.py).

Phase-specific plans matter for adversarial models, reused modules, and varying depth. A single observed order is not proof of a valid schedule across all phases. Training must preserve compatible collective order across ranks. Avoid permanently installing a trace-derived prefetch plan after one arbitrary research iteration.

**Use readiness dependencies to avoid unnecessary serialization.** Model execution and Update cooperate on gradients ready for clipping/update and parameter writes complete for the next gather. Initially protect checkpoint sources with a staging wait before the next mutating iteration. For FSDP, `set_post_optim_event` permits a narrower optimizer-to-all-gather dependency. An event must cover all writes to the parameters being gathered; custom optimizers must join their work back to the calling stream before returning. Reading live weights for EMA may overlap a gather when no mutation conflicts; evaluation weight swapping requires separate protection. The backend owns the events rather than exposing raw stream management in the ordinary trainer API. [FSDP event contract](https://github.com/pytorch/pytorch/blob/v2.14.0/torch/distributed/fsdp/_fully_shard/_fully_shard.py).

Expose advanced FSDP communication knobs as validated policy options: separate reduce-scatter groups, and a bounded number of in-flight reduce-scatter input buffers. Both APIs are experimental. The former can permit concurrent collectives; the latter trades memory for avoiding buffer-reuse waits. Neither establishes a universal throughput gain. Keep upstream conservative defaults for unmeasured adapters. [Tagged communication controls](https://github.com/pytorch/pytorch/blob/v2.14.0/torch/distributed/fsdp/_fully_shard/_fully_shard.py).

**Separate compiler optimization from runtime overlap.** Native FSDP overlap can work across compiled blocks, while compiler scheduling only sees operations inside its captured graphs. Merely making model blocks larger does not expose surrounding hook-driven communication. A constrained joint-graph executor can expose more dependencies for scheduling, but must own its gradient production and supported mutation model. Its interface should state which operations it performs so `Update` never repeats backward or an already-captured optimizer step. TorchTitan's GraphTrainer README documents joint forward/backward capture with optional optimizer capture, and currently asks for nightly PyTorch; exact 2.14 compatibility needs verification per adopted path. [GraphTrainer design](https://github.com/pytorch/torchtitan/blob/9a80ce528a6ec9f08b3d6f9d171bb2648ff839a5/torchtitan/experiments/graph_trainer/README.md).

**Represent sharding layouts where they matter.** Named mesh dimensions and persistent parameter placements belong to model execution metadata. `DataParallelMeshDims` supports identifying DP dimensions when parameters already use DTensor on a larger mesh. Objective denominators, gradient norms, and metrics need their own declared reduction domains; a replicated dimension must not multiply their totals. [Mesh and precision types](https://github.com/pytorch/pytorch/blob/v2.14.0/torch/distributed/fsdp/_fully_shard/_fsdp_api.py).

Optimizer compute may require a different layout from persistent storage. Keep an adapter boundary that can redistribute for an update, manage temporary storage, and restore the persistent layout before publishing parameter readiness. TorchTitan's FlexShard demonstrates this distinction with DistMuon; it is a separate implementation to evaluate, not a PyTorch 2.14 core API. Retain ordinary AdamW as the simple direct path. [FlexShard design at the reviewed commit](https://github.com/pytorch/torchtitan/blob/9a80ce528a6ec9f08b3d6f9d171bb2648ff839a5/torchtitan/distributed/flex_shard/README.md).

**Remove avoidable host synchronization and bound asynchronous work.** Current optimizer validation converts per-parameter finite checks into Python conditions. Retain a correct distributed nonfinite policy using aggregated device checks, with full parameter-by-parameter diagnostics available explicitly. Keep metrics detached on device until the reporting boundary; preserve any required waits for consistent update/skip decisions. Measure synchronization costs rather than assuming every check is free or eliminating correctness checks.

Device prefetch should own a transfer stream, readiness events, and tensor lifetimes, normally retaining only one upcoming microbatch initially. `non_blocking=True` on a copy alone does not establish copy/compute overlap. Account for prefetched inputs alongside FSDP buffers and checkpoint staging in the memory budget. Data checkpoints must distinguish fetched inputs from completed training consumption, saving or reconstructing queued inputs as required. Do not introduce a general asynchronous task scheduler for this bounded pipeline.

**Backend selection belongs to runtime construction.** Initially validate the known NCCL/FSDP2 configuration directly, without a general capability registry. Future backends must establish support for the requested compilation, capture, memory, and recovery behaviors. Configure process groups collectively before use and reject incompatible combinations. Let PyTorch's FSDP/DTensor implementations issue standard collectives directly; a library-wide Python wrapper around every collective would add another tracing boundary. Custom communication implementations are deferred until required by a workload. `nccl2` is a possible separately validated eager extension on 2.14.

Keep transport failure handling separate from training recovery. Reconfigured ranks require a coordinated decision about meshes, sharded state, accumulation, optimizer clocks, and data progress. Initial supported recovery should restart from a completed checkpoint; transparent continuation after an arbitrary collective failure is not implied by backend reconfiguration.

Remove implicit registration via class-level `__setattr__` interception and stop advertising pipeline parallelism as a model-setup flag: the current pipeline path raises `NotImplementedError`. A future PP executor must own its microbatch schedule and explicit interactions with Update. Keep generic callbacks for reporting; move FP8 preparation, EMA updates, prefetch, and checkpoint consistency to the component that owns their state and ordering. [Registration mechanism](../src/dream_trainer/utils/common.py), [pipeline setup path](../src/dream_trainer/trainer/mixins/setup/models.py).

**Acceptance requires correctness and useful end-to-end performance.** Implement these checks alongside the three representative experiments from the main proposal:

1. Type-check concrete model/batch/config types through preparation, compilation, construction, and data transfer. Reject unsupported execution combinations with object-specific errors before training.
2. Compare eager and compiled one- and two-device gradients and parameter updates against direct PyTorch controls, including uneven objective weights, frozen phases, nonfinite failure handling, optimizer bundles, and independent Update clocks. Declare numerical tolerances per precision recipe.
3. Test every supported shape bucket, dynamic range, train/eval phase, and parameter-freezing pattern. Record expected graph specializations and fail on unexplained ongoing recompilation.
4. Compare uninterrupted training with save/restore continuation, including data and RNG progress, EMA, mutable buffers, and multiple optimizers. Exercise delayed staging and failed saves. Reuse the existing DCP staging work where it satisfies these contracts.
5. Measure cold-start compilation, warm steady-state throughput, peak allocated/reserved GPU memory, checkpoint stalls, and time to a fixed training budget. Include data movement and reporting in end-to-end measurements. Matched samples/tokens, precision, topology, and objectives are required for comparisons.
6. For low precision or other numerical changes, compare held-out quality at matched compute; a stable loss or faster step is insufficient. Promote optimizations per workload and hardware after establishing their benefit, rather than enabling every new flag globally.

The immediate implementation target is a small, typed runtime on 2.14 with an explicit regional compilation path. The optimized recipe should be inspectable: a contributor must be able to find exactly where models are sharded, regions compiled, gradients synchronized, and snapshots protected.
