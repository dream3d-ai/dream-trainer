---
title: Dream Trainer v2 — Extension Contracts
---

**Status: proposal, September 16, 2026.** These interfaces are not implemented. Complements the [core API](v2-proposal.md) and [execution design](v2-pytorch-2.14.md). Example `lab.*` packages below are illustrative external integrations, not dependencies shipped with this repository. FP8 is an extension example; the initial core execution baseline remains BF16.

See [how v2 fits together](v2-overview.md) for the shared runner and an assembled experiment.

The library should define stable lifecycle boundaries and ownership rules. Feature implementations can live in an experiment directory or separate package and bind through ordinary imports. An extension may use several boundaries, but it does not need a monolithic base class or unrestricted access to a trainer.

The current [CallbackCollection](../src/dream_trainer/callbacks/callback.py) discovers overridden methods, dispatches through dynamic attribute lookup, and attaches the trainer to each callback. The current [FP8 callback](../src/dream_trainer/callbacks/fp8/callback.py) mixes hardware/configuration changes, module conversion, and optimizer hook installation. The current [async checkpointer](../src/dream_trainer/callbacks/checkpoint/async_.py) uses callbacks for staging protection and cleanup. Retain useful implementations while replacing these implicit dependencies with explicit bindings.

**Use three kinds of extension point.**

| Kind | Examples | Contract |
| --- | --- | --- |
| Observation | Console, WandB, progress, profiling, media reporting | Typed run events and detached reporting values. No live training-state mutation or training collectives. |
| Owned operations | Model conversion, post-optimizer FP8 work, EMA, registered algorithm state | Concrete model/update references bound during construction; narrow phase-specific callables with documented mutation and rank semantics. |
| Services | Checkpoint backend and storage, external resource lifetimes | Small Protocols that the owning runner/runtime invokes at required boundaries; errors and completion are explicit. |

A cohesive feature can implement more than one kind. An FP8 integration can provide conversion, runtime maintenance, state adapters, and diagnostics without adding FP8 branches to the runner. A checkpoint integration can supply backend/storage implementations and report save completion through the ordinary observer path.

**Keep the initial hook surface small and typed.**

| Binding | When called | Allowed effects |
| --- | --- | --- |
| Project preparation function | In the explicit model construction sequence | Transform the model before constructing its optimizers, in the recipe's tested order. |
| `runtime.on_ready(name, fn)` | Once after all construction and optional checkpoint restore, before training/evaluation | Validate or rebuild derived state on each participating rank. Do not replace restored persistent state with fresh defaults. |
| `update.on_after_optimizer(name, fn)` | Once after all optimizers in that Update succeed, before publishing model readiness | Maintain or modify the Update's owned weights/buffers and derived execution state. Any collectives follow the bound model domain. |
| `update.on_after_update(name, fn)` | After readiness, scheduler progression, and counter advancement; before `step()` returns | Read live weights and update the extension's own registered state, such as EMA shadows. Cannot mutate published model weights or execution buffers. |
| Observer event handler | At documented run/reporting/evaluation/checkpoint boundaries | Reporting and instrumentation, with rank scope selected explicitly. |
| Managed context/resource | Open during construction; close in guaranteed teardown | Acquire/release clients, writers, tracing sessions, streams, or other resources owned by the integration. |

`ReadyContext`, `AfterOptimizer`, and `AfterUpdate` are distinct named dataclasses, not a bag of optional fields. They expose only relevant metadata: fresh/resume mode, update identity/count, and appropriate result information. Each registration accepts one precisely typed Callable; functions and bound methods work equally well. A bound hook captures concrete model references when it needs them. Runtime does not inspect closure contents or infer dependencies from model names.

For resources, expose an explicit `runtime.enter_context(context_manager)` operation backed by managed teardown. Observers and checkpoint backends have their own typed open/close lifecycle, invoked by the runner while process groups are alive. Registration is complete during build and then frozen. This avoids hooks installing new hooks halfway through a run.

Settings required before runtime/process-group creation belong in the ordinary run-configuration factory. An integration may provide a typed helper that constructs or validates that configuration. It must not quietly change communication or compiler settings after those resources have been created. The same explicit factory mechanism works for both functional and class-based experiments.

**Example: one external FP8 feature binds to several boundaries.** An experiment's build function, usable directly or through a Trainer, can contain:

```python
from lab.torchao import Float8


fp8 = Float8(config.fp8)


def prepare(model: DiT, runtime: dt.Runtime) -> DiT:
    model = fp8.convert(model)
    return prepare_dit(model, runtime)


model = runtime.model(
    "model",
    factory=make_dit,
    prepare=prepare,
    initialize=lambda model: model.init_weights(),
)
optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
update = runtime.update(MODEL_UPDATE, optimizer, models=(model,))

fp8.bind(name="fp8/model", model=model, update=update, runtime=runtime)
```

The example assumes the selected recipe supports conversion at that preparation point and preserves the declared DiT interface. Recipes with different materialization requirements use an explicit project builder instead. Model preparation establishes the final parameter objects before optimizer construction.

`Float8.bind` is an ordinary method implemented by the external package. The core does not discover it or recognize its class. For a recipe requiring FSDP float8 scale precomputation, a binding can implement its two runtime callbacks using normal functions:

```python
def bind_fsdp_scales(
    name: str,
    model: torch.nn.Module,
    update: dt.Update,
    runtime: dt.Runtime,
) -> None:
    from torchao.float8 import precompute_float8_dynamic_scale_for_fsdp

    def ready(context: dt.ReadyContext) -> None:
        precompute_float8_dynamic_scale_for_fsdp(model)

    def after_optimizer(context: dt.AfterOptimizer) -> None:
        precompute_float8_dynamic_scale_for_fsdp(model)

    runtime.on_ready(name, ready)
    update.on_after_optimizer(name, after_optimizer)
```

TorchAO documents this operation after an optimizer step and implements a collective for applicable float8 weights. Binding it to the Update means it runs after the entire optimizer bundle, rather than once after each individual optimizer. The on-ready recomputation is this proposal's recipe-specific initialization/restore policy and needs validation. Not every FP8 recipe needs this hook. Pin and test the integration's TorchAO/PyTorch combination. [TorchAO implementation](https://github.com/pytorch/ao/blob/main/torchao/float8/fsdp_utils.py).

The package registers persistent state with `runtime.state("fp8/model", state_adapter)` only when the recipe actually has persistent extra state. Recomputable scale caches need not become checkpoint payloads. A stateful extension must define names, schema/version handling, and restore requirements; a missing required state entry must not silently initialize a fresh training history.

**External logging and checkpointing use different contracts.** A run may select:

```python
from lab.checkpointing import DCPBackend
from lab.logging import Wandb


run = dt.Run(
    trainer=DiffusionTrainer,
    config=Config(),
    max_steps=100_000,
    observers=[Wandb(project="diffusion")],
    checkpoint=dt.Checkpoint(
        backend=DCPBackend(root="runs/dit/checkpoints"),
        every=dt.Every(1_000, clock=MODEL_UPDATE.clock),
        keep_last=3,
    ),
)
```

`Checkpoint.backend` selects a service implementation; the simpler `Checkpoint(root=...)` form from the main proposal is convenience construction of the standard DCP adapter. These are alternative construction forms, with ambiguous root/backend combinations rejected.

An Observer contract has typed opening, event delivery, and closing. The initial events can be a closed union of run started, iteration completed, metrics reported, evaluation completed, checkpoint completed, and run ended. Event payloads contain immutable metadata and detached reporting snapshots, with no mutable Trainer reference. This is an API contract, not a security sandbox: Python closures can still capture arbitrary objects. Metric reductions occur on the required ranks before rank-zero logging; observer handlers never hide training collectives. Device-to-host reporting is bounded by reporting cadence, rather than repeated for every sink and microbatch.

A CheckpointBackend contract owns loading and saving through the state registry, plus resource cleanup. Its save handle exposes separate typed staging-completion and persistence-completion signals, and propagates failure. The runner retains the invariant that staging completes before subsequent training-state mutation and that only one save is in flight initially. The backend provides the serialization, storage, and transport implementation. PyTorch DCP already distinguishes these completion points; the adapter must preserve them. [PyTorch 2.14 checkpoint documentation](https://docs.pytorch.org/docs/2.14/distributed.checkpoint.html), [tagged async save implementation](https://github.com/pytorch/pytorch/blob/v2.14.0/torch/distributed/checkpoint/state_dict_saver.py).

**Ordering is part of the public contract.** The common runner's construction sequence is:

```text
create runtime and open selected services
  -> build models, data, Updates, and extension bindings/state
  -> freeze and validate registrations
  -> restore all registered state if requested
  -> run on_ready hooks
  -> begin training
```

For an Update, the sequence is:

```text
ready gradients and snapshot-safe state
  -> finite checks and clipping
  -> owned optimizers in declared order
  -> after_optimizer hooks
  -> publish parameter readiness
  -> advance schedulers and Update counter
  -> after_update hooks, including attached EMA
  -> return UpdateResult
```

EMA's `ema=` convenience argument, if retained, installs an ordinary bound extension using these contracts; the core needs no type-specific EMA dispatch. Its buffers and shadow weights are registered state, and its evaluation swap uses an explicit scoped context with restoration on exit.

Within one phase, hooks run in explicit registration order. If two operations have a semantic dependency, compose them in one ordinary function or installer with visible call order. Do not introduce arbitrary numeric priorities, automatic dependency sorting, or a universal hook around every line of code. Named bindings identify failures and appear in inspection output, including owner, phase, order, and rank domain. Reject duplicate names within the same owner/phase; the same feature name may intentionally bind to several phases.

Update/readiness hooks run synchronously in the owning domain, with the existing stream-ordering contract. A rank-zero-only hook cannot issue model collectives. Hook errors propagate and fail the run; callbacks after a failed phase do not continue as if the update succeeded. A failure after partial optimizer or extension-state mutation recovers from a completed checkpoint. Observer I/O may use bounded background delivery and an explicit reporting-failure policy, but model/update hooks cannot be silently moved to that queue. Teardown drains checkpoint work and closes resources even on errors, while preserving the original failure and reporting cleanup failures too.

Hooks may not request a checkpoint in the middle of an Update, change optimizer ownership, or bypass the staging rule. A request to stop or save must be routed to the runner's next completed iteration boundary. An integration that needs additional backward passes, new data flow, or a different optimizer algorithm should supply an explicit training function or component for the experiment to call. Those operations remain extensible without hidden changes to the algorithm through event handlers.

**Publish a small extension contract suite.** External integrations should be able to verify: typed callback signatures; single invocation after an optimizer bundle; fresh-build and restore ordering; state completeness; staging protection; cleanup after exceptions; identical behavior in functional and class-based runs; and two-rank collective participation where required. Core CI should exercise a fake external integration without importing WandB, TorchAO, or a storage SDK. Each real integration owns its package/version pins and feature-specific numerical/performance checks. Stable contracts and useful fixtures allow functionality to remain outside the core without relying on private lifecycle methods.
