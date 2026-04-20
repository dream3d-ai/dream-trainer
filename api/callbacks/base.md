# Base Callback API

Module: `dream_trainer.callbacks.callback`

Callbacks attach reusable behavior to the trainer lifecycle.

## Public Classes

| Class | Purpose |
| --- | --- |
| `Callback` | Base class for lifecycle callbacks. |
| `RankZeroCallback` | Callback base for work that should only run on global rank zero. |
| `CallbackCollection` | Ordered collection that dispatches implemented callback hooks. |

## Lifecycle Hooks

Callbacks may implement hooks around:

- launch
- configure
- setup
- fit
- train and validation epochs
- train and validation steps
- optimizer step and zero-grad operations
- interrupt handling
- train and validation contexts

Only implemented hooks are dispatched.

## State

Callbacks can implement `state_dict` and `load_state_dict`. Dream Trainer includes callback state in trainer checkpoints, which allows reusable behavior such as EMA, counters, or background state to resume with the run.

## See It In Use

- [Callbacks](../../callbacks.md) — the how-to guide, including picking a callback stack.
- [Tutorial 5 — A Curriculum Callback](../../tutorials/custom-components.md#a-curriculum-callback) — a resumable custom callback with state.
- [Core Concepts — Mixins vs Callbacks](../../core-concepts.md#mixins-vs-callbacks) — where the callback boundary lies.
