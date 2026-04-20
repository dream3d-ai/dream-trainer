# Abstract Trainer API

Module: `dream_trainer.trainer.abstract`

`AbstractTrainer` defines the protocol every trainer implementation must satisfy. It is the lowest-level contract for lifecycle methods, component lookup, distributed state, and checkpoint state.

## Public Classes

| Class | Purpose |
| --- | --- |
| `AbstractTrainerConfig` | Base configuration contract shared by trainer implementations. |
| `AbstractTrainer` | Abstract interface for configured trainers. |

## Responsibilities

`AbstractTrainer` describes methods for:

- configured model, optimizer, scheduler, dataloader, and metric lookup
- lifecycle phases such as configure, setup, fit, validate, and teardown
- distributed world access
- optimizer-to-model association
- trainer `state_dict` and `load_state_dict`
- interrupt handling

Most users subclass `DreamTrainer`, not `AbstractTrainer` directly. Use this API when building a custom trainer stack or writing callbacks that need to target the broadest possible trainer interface.

## See It In Use

- [Base Trainer](base.md) — the concrete lifecycle implementation.
- [Dream Trainer](dream.md) — the default trainer users subclass.
- [Core Concepts — Mixins vs Callbacks](../../core-concepts.md#mixins-vs-callbacks) — why the abstract/base/dream split exists.
