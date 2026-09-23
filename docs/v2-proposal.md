---
title: Dream Trainer v2 API Proposal
---

**Status: design proposal, September 16, 2026.** These APIs do not exist yet. Examples use project-defined models, losses, data builders, and preparation functions. This document specifies the intended developer experience and behavior; it is not a runnable implementation.

Start with [how v2 fits together](v2-overview.md) for the integrated architecture and lifecycle.

**Scope revision, September 17:** the [define-once parallelism contract](../../dream-trainer-v2/docs/define-once-parallelism.md)
supersedes this document's earlier narrowing to project-owned preparation and 1D FSDP as the
product scope. Model authors declare parallelism once; the library owns composition and
application across supported configurations. Single-device/FSDP is an implementation milestone.

Dream Trainer v2 should optimize for robust typing, clear APIs, and a structure that developers and coding agents can change confidently during rapid experimentation. Composable functions implement its operations; an optional trainer class keeps models, optimizers, data, RNGs, and algorithm methods together. A typed factory explicitly builds and registers components, then returns fully initialized state or a trainer. A function or method owns a complete logical iteration, including accumulation, forward/backward phases, and optimizer updates. One small runner handles the surrounding lifecycle for both styles.

These priorities are acceptance criteria. An API should preserve concrete types through preparation and data movement, make required dependencies visible in signatures, and let a contributor determine execution order by following direct calls. Public examples must type-check. A new objective or an additional optimizer should require local edits to an experiment rather than changes to the core lifecycle.

**Recommended initial scope: choose assumptions that remove machinery.** These are proposed product constraints to validate against the three representative trainers, not claims about every existing experiment. This initial scope takes precedence over the extension possibilities discussed later.

| Assumption | Simplification and cost |
| --- | --- |
| Production training targets CUDA/NCCL and BF16 on PyTorch 2.14+. CPU remains available for small correctness checks. | Start with one production execution stack. Defer alternative accelerators, FP16 loss scaling, and transparent backend switching. |
| One process owns one GPU; topology and registered components are fixed after build. | No live rank replacement, parameter-group mutation, model replacement, or resource migration. Recover by restarting from a complete checkpoint on the same topology. |
| Each trainable parameter and its FSDP synchronization unit has one owning Update. | Optimizers own disjoint parameters. Multiple optimizers sharing a synchronization unit must belong to one Update and commit together. No general coordination graph for overlapping Updates. |
| All ranks in a communication domain follow compatible phase and collective order. | Algorithm branches remain ordinary Python; the runtime does not reconcile independently chosen distributed schedules. Rank-local objective values may differ. |
| Model preparation is explicit, project-owned Python; each supported recipe has a tested transformation order. | No generic transformation planner, automatic pass ordering, or promise that every combination of sharding, quantization, and compilation works. |
| Framework-managed writes, including attached update hooks, obey the calling compute stream's dependencies. Custom work on another stream must join before returning. | A small fixed set of readiness events replaces arbitrary stream dependency tracking. FSDP and data transfer can still manage their own internal streams. |
| A completed `train_step` leaves no pending accumulation or partial optimizer phase. | Checkpoint only between logical iterations. No mid-backward snapshots or rollback of partially applied optimizer updates. |
| Initially permit at most one checkpoint in flight; staging completes before the next iteration can mutate state. | Persistence may overlap training after staging, without snapshot versioning or per-resource mutation tracking. New saves wait for the previous save. |

The first distributed recipes should cover single-device execution and native FSDP2 data parallelism; add other mesh layouts only for concrete validated workloads. Start compiled recipes with fixed or bucketed region shapes, while preserving variable-length data and explicit weighted objectives. Dynamic shape adapters, graph executors, FP8, alternative communication backends, and optimizer layout redistribution are later workload-driven additions. They do not require public plugin registries or speculative interfaces in the first implementation.

Preserve multiple models, multiple optimizers, frozen teachers, explicit adversarial phases, differing update frequencies, and weighted losses. These are essential algorithm choices. Models and optimizer ownership remain fixed while computations, data, and objective weights can change across iterations. Tied parameters inside one owning Update are compatible with that rule; two independent Updates writing the same parameter are not.

The ordinary public vocabulary remains Runtime, Update, Run, and an optional Trainer, plus typed configuration/data values and functions operating on them. Model execution records, readiness events, and communication capabilities are internal implementation details. Start with concrete implementations and extract a Protocol only when a second real implementation or meaningful test substitution needs it. A native PyTorch option can remain in `prepare_dit` without gaining a corresponding framework class or flag.

**PyTorch 2.14.0 is the minimum for v2.** Its dependency metadata and startup validation must reject older versions. There will be no pre-2.14 compatibility paths. Build on native FSDP2, DTensor/DeviceMesh, regional compilation, and distributed checkpointing, with explicit policies for shapes, precision, and synchronization. These foundations predate 2.14; the new baseline lets v2 use their current interfaces consistently. See the [PyTorch 2.14 execution design](v2-pytorch-2.14.md) for source-verified features, defaults, limitations, and validation requirements.

The essential split is between **Trainer**, **Runtime**, and **Update**. Trainer owns the experiment's algorithm and references to its components. Runtime owns distributed resources, per-model execution records, and the state registry. An Update binds an ordinary PyTorch optimizer, or a fixed bundle stepped together, to its participating models, optional schedulers and EMA, clipping policy, and update counter. Run describes how to construct and execute the trainer. Sharding groups, compiled regions, and optimizer parameter groups are independent boundaries; their relationships are validated during construction.

The class facade is a shallow, keyword-only dataclass base `Trainer[Config]` with two required methods, `build(cls, config, runtime) -> Self` and `train_step(self, step) -> None`, and an optional `validate(self, evaluation) -> None`. The base stores typed `config` and `runtime` fields; subclasses declare their components as required fields. The runner creates the runtime, calls the trainer's build method once, restores registered state, and then invokes training and evaluation. The constructor only assigns already-created dependencies. There is no publicly usable trainer whose required attributes appear later during setup. Function-based runs use the same construction, restore, and iteration sequence without requiring a Trainer instance.

Mixins remain available for cohesive, explicitly typed helper methods. They must declare the capabilities they consume and must not rely on cooperative lifecycle chains, mutate registration through attribute interception, or implicitly add optimizer actions. The build or training method can call a reusable helper directly; stateful services such as checkpointing and EMA are composed as objects. There is no separate Recipe wrapper in the default API.

**A typical experiment should fit in one file.** The following sketches diffusion training with equally sized microbatches, one optimizer, gradient accumulation, and an EMA:

```python
from dataclasses import dataclass
from typing import Self

import torch
import dream_trainer as dt


@dataclass(frozen=True)
class Config:
    lr: float = 3e-4
    microbatch_size: int = 8
    accumulation: int = 4


@dataclass(frozen=True)
class DiffusionBatch:
    latents: torch.Tensor
    labels: torch.Tensor


MODEL_UPDATE = dt.UpdateKey("model")


@dataclass(kw_only=True)
class DiffusionTrainer(dt.Trainer[Config]):
    model: DiT
    update: dt.Update
    train_data: dt.DataSource[DiffusionBatch]
    noise: torch.Generator

    @classmethod
    def build(cls, config: Config, runtime: dt.Runtime) -> Self:
        cfg, rt = config, runtime
        model = rt.model(
            "model",
            factory=make_dit,
            prepare=prepare_dit,
            initialize=lambda model: model.init_weights(),
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
        update = rt.update(
            MODEL_UPDATE,
            optimizer,
            models=(model,),
            clip_grad=1.0,
            ema=dt.EMA(model, decay=0.999),
        )
        train_data = rt.data(
            "train",
            make_loader(
                batch_size=cfg.microbatch_size,
                rank=rt.data_rank,
                world_size=rt.data_world_size,
            ),
            move=move_diffusion_batch,
        )
        return cls(
            config=cfg,
            runtime=rt,
            model=model,
            update=update,
            train_data=train_data,
            noise=rt.rng("noise", scope="data"),
        )

    def train_step(self, step: dt.Step) -> None:
        self.update.zero_grad()

        for micro in step.microbatches(
            self.train_data, count=self.config.accumulation
        ):
            with self.update.sync(enabled=micro.last):
                with self.runtime.autocast():
                    loss = self.compute_loss(micro.batch)
                self.update.backward(loss / micro.count)

            step.mean("train/loss", loss, weight=len(micro.batch.latents))

        self.update.step()

    def compute_loss(self, batch: DiffusionBatch) -> torch.Tensor:
        return flow_loss(self.model, batch, generator=self.noise)
```

`rt.model` preserves the concrete module type, so the factory's `DiT` becomes a `DiT` in the trainer. The optimizer remains a PyTorch optimizer. `micro.batch` is inferred as `DiffusionBatch`, with tensors moved by a typed `move_diffusion_batch(batch: DiffusionBatch, device: torch.device) -> DiffusionBatch` function. An Update references these objects. It does not replace the model forward or wrap the optimizer in another optimization algorithm. Project helpers such as `make_dit`, `prepare_dit`, `make_loader`, and `flow_loss` must expose typed signatures too; these documentation sketches are not complete standalone type-checking fixtures.

Declare trainer fields only for components that its methods need to access directly. The optimizer can remain a local variable in `build` because Update retains it. A project needing optimizer-specific methods can declare an additional concrete optimizer field without changing the lifecycle.

The field list is useful documentation and a constructor contract. Tests can construct a trainer directly with tiny models and a CPU runtime; they need not execute the production build path. A normal explicit constructor can provide the same guarantees if dataclass inheritance becomes restrictive. Do not add a second generic State object unless it solves a concrete reuse problem.

**Functions and classes share one implementation.** Expose cohesive operations as ordinary functions with explicit state arguments. Stateful resources remain ordinary objects; these functions can mutate them and are not advertised as pure. Convenient methods delegate to the same operation, with the same guards and bookkeeping:

| Function form | Convenience form | Shared contract |
| --- | --- | --- |
| `build_model(runtime, name, ...)` | `runtime.model(name, ...)` | Preparation, initialization, type preservation, and explicit state registration. |
| `create_update(runtime, key, optimizers, ...)` | `runtime.update(key, optimizers, ...)` | Ownership validation and registered optimizer/update state. |
| `microbatches(step, data, count=N)` | `step.microbatches(data, count=N)` | Typed complete-window acquisition and data movement. |
| `sync_gradients(update, enabled=...)` | `update.sync(enabled=...)` | Per-model forward/backward synchronization context. |
| `backward(update, loss)` | `update.backward(loss)` | Backpropagation of the explicitly normalized loss. |
| `step_update(update)` | `update.step()` | Gradient checks/clipping, optimizer writes, readiness, and update clocks. |

Ellipses in this table stand for the explicit typed parameters described elsewhere, not public untyped `**kwargs`. Functions are the implementation entry points; convenience methods perform direct delegation. Do not maintain separate class/function versions of optimizer logic, checkpoint sequencing, or distributed scheduling. State ownership must not depend on a function being called through a Trainer.

A functional experiment can use a plain project dataclass. A shared state object is useful here because the same algorithm is intentionally reused from both functions and a class:

```python
@dataclass(kw_only=True)
class DiffusionState:
    model: DiT
    data: dt.DataSource[DiffusionBatch]
    update: dt.Update
    noise: torch.Generator


def diffusion_step(
    state: DiffusionState,
    config: Config,
    runtime: dt.Runtime,
    step: dt.Step,
) -> None:
    state.update.zero_grad()
    for micro in dt.microbatches(step, state.data, count=config.accumulation):
        with dt.sync_gradients(state.update, enabled=micro.last):
            with runtime.autocast():
                loss = flow_loss(state.model, micro.batch, generator=state.noise)
            dt.backward(state.update, loss / micro.count)
        step.mean("train/loss", loss, weight=len(micro.batch.latents))
    dt.step_update(state.update)
```

`build_diffusion(config: Config, runtime: Runtime) -> DiffusionState` uses the same explicit model/update/data/RNG registration as the class example, then returns the plain dataclass. It must register any additional checkpointable mutable algorithm state too. The dataclass is not automatically traversed or serialized.

```python
run = dt.Run.from_functions(
    config=Config(),
    build=build_diffusion,
    train_step=diffusion_step,
    max_steps=100_000,
)
dt.run(run)
```

`Run.from_functions` couples `build: Callable[[C, Runtime], S]` with `train_step: Callable[[S, C, Runtime, Step], None]` and an optional `validate: Callable[[S, C, Runtime, Evaluation], None]`. It accepts the same runtime, checkpoint, and observer options as the class run. The generic factory returns `Run[C]`; `S` remains fully typed inside the binding closure rather than being widened to `Any`. A differently typed state/configuration in the step function must fail static checking.

The class facade can reuse those exact functions:

```python
@dataclass(kw_only=True)
class ComposedDiffusionTrainer(dt.Trainer[Config]):
    state: DiffusionState

    @classmethod
    def build(cls, config: Config, runtime: dt.Runtime) -> Self:
        return cls(
            config=config,
            runtime=runtime,
            state=build_diffusion(config, runtime),
        )

    def train_step(self, step: dt.Step) -> None:
        diffusion_step(self.state, self.config, self.runtime, step)
```

A project can instead keep the direct-field trainer shown earlier and write its algorithm inline. Extract a shared function/state bundle only when reuse helps. Both forms preserve concrete types and use the same managed runner.

Internally, each run binds construction to a pair of typed training/evaluation callables. The runner restores the explicit runtime registry before invoking either callable, then executes one common loop. It need not inspect function signatures, look up attributes dynamically, discover plugins, or branch on a training style during each iteration. Closures and trainer objects are not checkpoint payloads; registered resource state is.

Lower-level custom loops can call the same preparation and update functions directly, with an explicitly constructed runtime. Using only these functions does not automatically provide managed iteration counters, checkpoint staging boundaries, evaluation schedules, failure handling, or cleanup. Custom loops own that lifecycle, while the managed function runner supplies it without requiring a Trainer subclass. No hidden global runtime or automatic context-variable lookup is needed.

**Typing must survive the entire user path.** The core type relationships are:

| Boundary | Required relationship |
| --- | --- |
| Trainer configuration | `Trainer[C].config` is `C`; `Run[C]` couples `type[Trainer[C]]` with a configuration of the same type. |
| Function run | A builder returning `S` is paired with a step consuming `S` and the same config type `C`; closure binding preserves both types. |
| Model preparation | A factory of model type `M` and preparation preserving `M` return `M`, retaining custom model methods. |
| Data source | An iterable of `B`, with a transfer function from `B` to `B`, produces `DataSource[B]`. |
| Accumulation | `DataSource[B]` produces `MicrobatchWindow[B]` and `Microbatch[B]`; `micro.batch` remains `B`. |
| Construction | Every required model, update, RNG, and data field is a constructor argument; no `None` placeholders, suppressed uninitialized attributes, or post-construction type casts. |
| Results | Update outcomes and observer events are named dataclasses with precise fields. |
| Persistence | Heterogeneous checkpoint payloads and third-party typing gaps are contained in small adapters with runtime validation. |

Batch types belong to experiments. Use a dataclass when fields are fixed, or a TypedDict where existing data pipelines benefit from mapping semantics. The framework does not require a universal batch schema. Multiple data sources can have different batch types in the same trainer; the data source carries that type, so Trainer needs only its configuration generic parameter.

Public APIs should use precise parameter and return types, keyword-only options, and small Protocols where structural substitution is useful. Avoid `dict[str, Any]`, `Callable[..., Any]`, dynamic attribute lookup, and untyped `**kwargs` across user-facing boundaries. Any exceptions must be explicit, localized adapters. Generic preparation must actually preserve the declared model interface: a transformation returning a different wrapper needs an accurately typed adapter instead of a blanket cast claiming it returns the original type.

Ordinary tensor typing does not prove shapes, layouts, device placement, or distributed correctness. In particular, PyTorch's module-call boundary can expose `Any` even when a model's forward is annotated. Model integrations need audited typed call boundaries or stubs, with fixtures checking the output types. Preserve the normal module call and its hooks; never call `forward` directly merely to satisfy a type checker. See the [PyTorch 2.14 module source](https://github.com/pytorch/pytorch/blob/v2.14.0/torch/nn/modules/module.py). Validate tensor contracts and supported parallelism combinations separately at runtime and in integration tests.

Pin one authoritative type checker in CI, proposed as Pyright in strict mode, and check the library plus complete runnable examples. Strict mode is one part of the policy; explicit public `Any` still needs an API rule and review. See [Pyright configuration](https://github.com/microsoft/pyright/blob/main/docs/configuration.md). Maintain both positive fixtures and negative fixtures that must reject wrong configurations, wrong batch types, misspelled fields, incomplete construction, and incorrect model preparation results. Tests should inspect diagnostics, rather than hiding invalid calls behind ignore comments.

Function and class entry points also need matched continuation checks: with the same construction, data, seed, and step function, they must produce matching parameters, update counters, metrics, and checkpoint state. Type fixtures must reject mismatched builder/step state types. The hybrid binding design described here has not yet been checked by the earlier stand-in probe.

During this proposal, a small dependency-free contract probe passed mypy strict checking and preserved the inferred concrete model, batch, trainer, and run-config types. Invalid fixtures produced the expected five diagnostics for a wrong configuration, two missing constructor fields, a wrong loss argument, and a nonexistent batch field. This checks the generic construction design with stand-in types; it is not validation of PyTorch integration or the proposed runtime.

`step.mean` accumulates detached weighted statistics on device. Reporting reduces their sums and weights over the appropriate data domains at the reporting boundary. It must not perform an unconditional `.item()` or collective on every microbatch. Model-parallel replication requires a declared metric reduction domain so replicated observations are not counted twice.

**Launching is independent of defining the algorithm.** A Python factory returns a complete run description:

```python
def experiment() -> dt.Run[Config]:
    return dt.Run(
        trainer=DiffusionTrainer,
        config=Config(),
        runtime=dt.RuntimeConfig(
            compile=dt.RegionalCompile(fullgraph=True),
        ),
        max_steps=100_000,
        checkpoint=dt.Checkpoint(
            root="runs/dit/checkpoints",
            every=dt.Every(1_000, clock=MODEL_UPDATE.clock),
            keep_last=3,
        ),
        observers=[dt.Console()],
    )


if __name__ == "__main__":
    dt.run(experiment())
```

`max_steps` counts completed calls to `trainer.train_step`. Update-based schedules explicitly name their clock. Single-process execution uses one device. The library never starts workers merely because multiple GPUs are visible.

For the initial production scope, RuntimeConfig defaults to CUDA/NCCL, BF16 compute with FP32 parameter storage and gradient reduction, and native FSDP2 across launched data-parallel ranks when more than one rank is present. Single-rank execution uses a local module. CPU correctness tests select an explicit CPU test runtime. Compilation remains explicit in the run above because its selected regions and shapes depend on the project preparation function. These are proposed defaults to validate with the initial recipes, not behavior of the existing library.

`RegionalCompile` requests compilation of regions explicitly selected by the model's preparation function. It does not decorate `train_step` or automatically discover every compilable method. A preparation adapter must consume the policy and record the selected regions; unsupported requests fail during setup. `Eager()` selects the same algorithm for debugging and correctness comparisons. Compilation is the intended default for validated GPU recipes, with shape and transform contracts documented per recipe.

Compilation policy is separate from training execution. The default runner directly calls `train_step`; a future pipeline schedule or joint-graph executor would need its own explicit contract for forwards, backwards, microbatches, and updates. Such an executor cannot be introduced merely by replacing the model preparation function.

Inside `train_step`, `step.state.iterations` is the number of previously completed training iterations, and `self.update.steps` is the number of successful updates to that optimizer. These are read-only views. Warmup conditions and alternating schedules can use them directly; a successful Update advances its counter immediately, and the runner advances the iteration counter only after the method returns successfully. `UpdateKey` gives the run description a typed reference to a future update clock; it carries a stable serialized name and must resolve to a registered update before training. Normal code uses object references instead of string-based object lookup.

```bash
python train.py
torchrun --standalone --nproc-per-node=8 train.py
dream train experiments.dit:experiment --resume latest
torchrun --standalone --nproc-per-node=8 -m dream_trainer train experiments.dit:experiment
```

`dream` and `python -m dream_trainer` expose the same thin Typer CLI. Python factories are the source of configuration. CLI options cover execution concerns such as resume, output path, profiling, and a short run. Avoid a separate configuration language or dynamically generating flags for every nested object.

The run manifest records resolved configuration data, named components, package versions, source references, and runtime topology. Arbitrary Python closures are not serialized as configuration. Project configuration should primarily use dataclasses and ordinary values; factories remain code identified by their source.

For compiled runs, also record compiler policy, selected regions, shape assumptions, precision and activation policies, and relevant CUDA/NCCL/kernel-package versions. Report requested versus effective optimizations, compile/warmup time, recompiles, steady-state throughput, and peak memory. A request for an unavailable backend must not silently become an apparently optimized run.

**Multi-model training uses the same primitives.** A trainer's build method can register a generator and discriminator, create two ordinary optimizers, and supply the declared `generator_update` and `discriminator_update` fields to its constructor. Its training method can then be:

```python
def train_step(self, step: dt.Step) -> None:
    self.generator_update.zero_grad()
    self.discriminator_update.zero_grad()

    for micro in step.microbatches(self.train_data, count=self.config.accumulation):
        real = micro.batch.images

        with dt.freeze(self.discriminator, eval_mode=True):
            with self.generator_update.sync(enabled=micro.last):
                with self.runtime.autocast():
                    fake = self.generator(real)
                    g_loss = self.generator_loss(fake, real)
                self.generator_update.backward(g_loss / micro.count)

        with self.discriminator_update.sync(enabled=micro.last):
            with self.runtime.autocast():
                d_loss = self.discriminator_loss(real, fake.detach())
            self.discriminator_update.backward(d_loss / micro.count)

        step.mean("train/g_loss", g_loss, weight=len(real))
        step.mean("train/d_loss", d_loss, weight=len(real))

    self.generator_update.step()
    self.discriminator_update.step()
```

This preserves the existing pattern of accumulating both objectives against the current weights and updating both at the shared boundary. Moving an optimizer update between phases changes the algorithm and remains an explicit edit to the training method. A 5:1 update ratio is an ordinary loop containing five complete discriminator updates and one generator update; the trainer also explicitly chooses whether those updates reuse data or consume new batches.

`dt.freeze` temporarily disables parameter gradients, optionally changes module training modes, and restores their exact previous values on exit. It does not enter `no_grad`: gradients can still flow through a frozen discriminator into generator outputs. Implementations must preserve mixed child-module training modes and previously accumulated parameter gradients.

**An Update has a precise contract.**

| Operation | Behavior |
| --- | --- |
| `zero_grad()` | Clears the owned optimizers' gradients using the configured PyTorch policy. |
| `sync(enabled=...)` | Configures the participating distributed modules for this forward/backward region; manages and restores state per module. |
| `backward(loss)` | Backpropagates the supplied loss. It never implicitly divides by an accumulation count. |
| `step()` | Requires snapshot-safe writes, checks the supported distributed finite-state condition, clips owned gradients, invokes owned optimizers, runs bound post-optimizer hooks, and publishes parameter readiness. |
| Successful update | Advances schedulers and the named counter, then runs post-update effects such as EMA before returning. |
| Failed update | Stops the run; recovery uses the previous completed checkpoint. |

`step()` returns an `UpdateResult` with a device-resident gradient norm on success. The initial BF16 policy stops on nonfinite updates, consistently across participating ranks. Dynamic loss scaling and skip-update policies are outside the initial scope; they should add their own explicit outcome semantics if a real workload needs them.

`step()` does not implicitly clear gradients. This keeps the visible sequence consistent with ordinary PyTorch. Calling the raw optimizer's `step()` bypasses these services; custom optimizer execution must go through an explicit adapter if the runner is to manage its counters and state.

One optimizer can own parameters spanning several registered models. Multiple optimizers can target disjoint parameter sets in one model. If they share an FSDP synchronization unit, register them as a fixed tuple in one Update; `step()` invokes all of them at the same algorithmic boundary and then advances the Update counter once. This covers combinations such as Muon and AdamW without treating their shared communication as independent. An exception midway through the tuple is not transactionally reversible. The registry rejects parameters or FSDP synchronization units owned by different Updates. Independently scheduled updates require disjoint ownership and synchronization units.

The construction change for a bundle is small. Here both optimizers were built against disjoint final parameter sets, and each optimizer supports the model's prepared layout:

```python
update = rt.update(
    MODEL_UPDATE,
    (matrix_optimizer, other_optimizer),
    models=(model,),
    clip_grad=1.0,
    ema=dt.EMA(model, decay=0.999),
)
```

The training method is unchanged: one backward can produce gradients for both, and one `update.step()` applies both in the declared order. This syntax does not make an optimizer compatible with arbitrary DTensor layouts; that remains part of the recipe's validation.

The `sync` context covers both forward and backward. Nested or sequential contexts for different models must not share a trainer-wide transition flag. Some FSDP accumulation modes retain larger gradients; select such policies explicitly and show their estimated memory implications in the run description.

Internally, synchronization state belongs to the participating model execution records and communication domains. `Update.sync` is a convenience that delegates to those records when the mapping is unambiguous. Optimizer ownership does not create a separate communication domain. An execution record holds actual sharding units, compiled regions, mesh/layout metadata, phase-specific prefetch policy, and dependency state while `self.model` retains its concrete model type. These are inspectable records and small backend adapters, not a declarative algorithm graph.

The runtime coordinates gradient readiness, optimizer writes, and parameter prefetch using a small fixed set of backend dependencies. On supported FSDP paths an event after owned optimizer writes and bound post-optimizer hooks can release the next all-gather before unrelated work on the compute stream finishes. The single-owner and stream-ordering contracts make this dependency explicit. Post-update observers/effects cannot mutate the published live weights; mutations belong to the earlier owned phase. A future layout-changing optimizer adapter must restore persistent layouts before publishing readiness. See the [structural execution changes](v2-pytorch-2.14.md#structural-changes) and [extension contracts](v2-extensions.md).

**A microbatch window is bounded, and its normalization is visible.** `step.microbatches(train, count=N)` obtains a complete window of N microbatches, retains their CPU-side inputs, and moves one microbatch to the device at a time. Its iterator exposes `batch`, `index`, `count`, and `last`. A caller can name the window before iterating to compute normalization metadata. The default short-window policy stops before any backward work if a complete window is unavailable; ranks must agree on exhaustion before entering training collectives. A future partial-window policy must define its actual denominator explicitly.

The division by `micro.count` in the examples assumes equal per-microbatch and per-rank weighting. Variable video lengths, masks, or packed tokens require an explicit denominator:

```python
window = step.microbatches(self.train_data, count=self.config.accumulation)
normalization = self.update.normalizer(
    window,
    weight=lambda batch: batch.valid_mask.sum(),
)

self.update.zero_grad()
for micro in window:
    with self.update.sync(enabled=micro.last):
        with self.runtime.autocast():
            loss_sum = masked_loss_sum(self.model, micro.batch)
        self.update.backward(loss_sum, normalization=normalization)
self.update.step()
```

The normalizer counts valid objective elements across the complete accumulation window and the declared data partition, then scales local loss sums consistently with the backend's gradient reduction semantics. It must account for whether gradients are summed or averaged and avoid counting TP replicas as additional examples. CP-sharded objectives need an explicit partition description. Reject an ambiguous layout rather than inferring it from arbitrary tensor shapes. Distinct loss terms may require distinct denominators.

**Model preparation remains model-specific Python.** The common `rt.model` lifecycle is:

```text
construct on meta
  -> prepare(model, runtime)
  -> materialize local parameters and buffers
  -> initialize(model)
  -> validate and register the resulting model
```

`prepare_dit` is a normal project function containing the tested order of transformations for that model: for example, quantization, TP/CP adaptations, activation checkpointing, regional compilation, and FSDP. It returns the resulting model. There is no claim that one ordering is valid for every backend. The runtime exposes meshes and policies; the model adapter chooses units and supported combinations.

Keep mode changes, parameter freezing, accumulation loops, metrics, and checkpoint coordination in ordinary Python. Compile selected module calls and tensor-only loss regions; independently validate any compiled optimizer path. For modules, prefer in-place `nn.Module.compile()` where supported by the preparation adapter so the trainer retains its concrete model object. `fullgraph=True` applies to each selected region and makes unexpected graph breaks visible. It does not promise whole-iteration capture. Phase-specific guards and recompilation behavior still require tests.

Lower-level construction and registration are available when a model requires preparation after materialization or specialized checkpoint initialization. Such a model builder can return its prepared module for explicit registration. Optimizers are always constructed against the final parameter objects. This avoids requiring a growing sequence of lifecycle hooks for special cases.

Initially, materialization and initialization happen before checkpoint restoration. That straightforward ordering initializes nonpersistent buffers correctly. Skipping expensive initialization on resume is a later optimization requiring a model-specific completeness contract.

**State registration determines checkpoint contents.** Registering a model, Update, data source, or RNG assigns a stable name and adds its state adapter to the run. Updates register their optimizers, schedulers, EMA, and counters together. Additional project state uses `rt.state("name", stateful_object)`, where the object implements `state_dict` and `load_state_dict`. Assigning an object to `self.model` or another trainer attribute does not register it; the explicit runtime call does.

A checkpoint includes named model and optimizer state; schedulers and EMA; completed logical iterations and update counters; supported Python, NumPy, CPU/CUDA and named RNG streams; stateful data/sampler progress; registered algorithm state; schema version; and the run manifest. Frozen assets can eventually be referenced by immutable identifiers instead of copied, but references must be verified during restore.

`rt.rng("noise", scope="data")` denotes an independent named stream per data replica, shared as required within the model-parallel group. Name-derived seeds must use a stable hash. It must not derive seeds blindly from global rank, because TP/CP participants may require coordinated random inputs. Exact continuation initially targets the same validated topology; loading weights on a different topology is a separate promise from identical data/RNG continuation.

```python
dt.run(experiment(), resume="latest")       # Full training continuation.
dt.run(experiment(), init_from="seed")      # Model weights; fresh run state.
```

The runner restores registered state after build and before consuming training data. Full resume is strict by default. Missing optimizer, EMA, RNG, or data state must not be silently treated as an exact resume. Ordinary non-stateful iterables remain usable for training, but full data continuation requires a supported stateful source or explicit restart-data mode.

Only completed training iterations are eligible for checkpoints in the first implementation. The training method must finish its accumulation windows and optimizer phases before returning. A crash midway through a multi-optimizer iteration resumes from the previous complete checkpoint. Returning does not make a sequence of optimizer updates transactionally reversible in memory.

Async checkpointing separates staging completion from durable save completion. The runtime waits before any conflicting mutation of the snapshot sources, including mutable model buffers during forward. Training inputs and RNG state need consistent snapshots too. Save failures are surfaced, retention acts only after confirmed completion, and shutdown drains or reports pending saves. These rules belong to the checkpoint/runtime contract rather than observer order.

Initially enforce this with one simple boundary: finish snapshot staging before the next training iteration or evaluation that can mutate state. Permit one save in flight, wait for it before starting another, and allow persistence to overlap subsequent training. This intentionally gives up overlap with staging in exchange for eliminating per-resource hazard tracking. More aggressive staging overlap requires measured benefit and a separate consistency design.

**Evaluation and observation stay optional.** A trainer may implement `validate(self, evaluation: Evaluation) -> None`. A run schedules it with an explicit clock, such as `Every(1_000, clock=MODEL_UPDATE.clock)`. The method controls which models, data, metrics, and generation loops it uses, and can share model references and sampling helpers with `train_step`. It can select EMA weights with an explicit context. Evaluation contexts restore exact model modes and weights on exit; separate registered evaluation RNGs avoid perturbing training streams.

WandB, progress bars, and profilers are observers receiving immutable event metadata and reported values. They do not receive unrestricted mutable trainer state as their primary API. Optional integrations load only when selected. The package's CPU core must import without WandB, torchmetrics, S3 packages, or quantization packages installed.

Algorithmic behavior is configured where it acts: EMA on an Update, quantization in model preparation, discriminator warmup inside the training method, and fault tolerance in an execution backend. Avoid offering every observer a callback before and after every internal operation.

**Extensions can live entirely outside this package.** Publish small typed contracts for observers, model preparation functions, owned Update hooks, readiness after restoration, managed resources, registered state, and checkpoint backends. An external feature can bind to several contracts through an ordinary Python factory or installer. Bindings are explicit and frozen before restore/training; component owners invoke them in a documented order. Function-based and class-based runs share this mechanism. The core knows when a post-optimizer function must run and when a save must finish staging, without knowing about TorchAO recipes, WandB clients, or storage vendors. See the [extension design and FP8 example](v2-extensions.md).

**The run should be inspectable before training.** Proposed CLI commands include:

```bash
dream inspect experiments.dit:experiment
dream train experiments.dit:experiment --steps 10
dream benchmark experiments.dit:experiment --warmup 10 --steps 50
dream profile experiments.dit:experiment --steps 20
```

Static inspection shows declared configuration without allocating the training model; runtime inspection reports resolved mesh, model preparation, parameter ownership, update policies, state names, and schedules. Trainer construction can execute arbitrary Python, so a static inspect command cannot claim to discover every dynamically registered object.

Diagnostics should identify the object and phase that failed: for example, an optimizer was built before a transform replaced its parameters, two Updates overlap unsupported FSDP units, a resume is missing a named RNG, or a requested parallelism combination has no validated adapter. Expensive per-parameter finite scans are opt-in diagnostics, not an unconditional cost on every optimizer update.

**Agent-friendly structure is a design requirement.** Keep definitions next to the operations they describe, import dependencies explicitly, and keep execution order visible in one runner. A proposed layout is:

```text
src/dream_trainer/
    trainer.py       # Optional typed class facade
    runner.py        # Typed function/method binding and one common lifecycle
    runtime.py       # Distributed resources and explicit registration
    execution/      # Model records, synchronization, readiness, backend adapters
    update.py        # Optimizer execution and its state
    data.py          # Typed data sources and accumulation windows
    checkpoint/     # Snapshot consistency, persistence, schema migration
    observers/      # Optional reporting integrations
    extensions.py   # Public context types and small extension contracts
    adapters/       # Experimental PyTorch and external-library boundaries
examples/
    diffusion/      # Complete typed experiment and tiny test configuration
    adversarial/    # Explicit two-model training with resume coverage
    rem/            # Representative project-specific training scheme
```

Use direct methods and explicit construction rather than dynamic plugin discovery, synthesized methods, or callback dependency graphs. A contributor should be able to find the optimizer update, the loss scaling, and the checkpoint boundary with a text search and follow each with go-to-definition. Avoid splitting a straightforward training method into many single-use helpers purely to reduce line count.

Run configurations are immutable dataclass values where practical. Derive an ablation with `dataclasses.replace` or a small Python factory; log the resolved config and its changes. Each example should provide a cheap configuration for exercising construction, one logical step, and restore without loading production data. Type errors and CPU checks provide the first feedback loop; two-device correctness tests and matched benchmarks validate distributed changes. Framework tests can use injected tiny components through the normal constructor.

The documentation should identify the canonical edit location for common research tasks: change an objective in `compute_loss`, add a model in the trainer's declared fields and build method, change update order in `train_step`, and add a sharding strategy in a model preparation adapter. Feature-specific tests sit beside the relevant component. A short contributor guide records the lifecycle and verification commands; complete executable examples remain the authoritative API documentation.

**The implementation should start with the three existing workload shapes.** Port a conventional diffusion trainer, the adversarial autoencoder, and a REM/adversarial trainer. Validate one- and two-device execution, equal and weighted accumulation, independent multi-model synchronization, optimizer/scheduler/EMA clocks, and save/resume continuation. Compare matched gradients, parameters, data, and RNG outcomes against direct PyTorch controls, and measure throughput and peak memory. Extend to TP/CP/PP combinations only with explicit integration tests for each supported adapter.

The first supported surface should be the composable operations, Runtime, Update, Run, an optional Trainer facade, named state, the checkpoint service, basic observers, and typed eager/regional compilation policies. Function-based runs and class-based runs share one implementation and are part of the initial contract. It should not include a declarative training graph, plugin discovery, a second model abstraction, or a universal phase scheduler. A constrained comparison against TorchTitan's GraphTrainer may be a separate research spike after the regional baseline works. Only extract a training-executor interface once a second execution backend demonstrates a benefit on a representative workload; arbitrary Python training functions or methods are not promised to be capturable.

Existing v0.3 trainers should remain on their current API while they migrate individually. Reuse tested checkpoint storage and materialization code where its contracts fit. A v1 checkpoint importer should map state names and reject unsupported omissions; migration can preserve algorithm method organization while replacing lifecycle inheritance and implicit registration. The decision to ship a broader v2 library should follow successful reuse across the three representative trainers.
