# Distributed Utilities API

Modules:

- `dream_trainer.utils.dist`
- `dream_trainer.utils.dist.core`
- `dream_trainer.utils.dist.context`
- `dream_trainer.utils.dist.decorators`
- `dream_trainer.utils.dist.ops`
- `dream_trainer.utils.dist.utils`

These helpers expose lightweight distributed coordination primitives outside the trainer's `world` abstraction. Use them when configuration code, CLI tools, dataloaders, or utility functions need rank information or simple cross-rank coordination without depending on `DistributedWorld`.

## What Lives Here

| Module | Purpose |
| --- | --- |
| `core` | Rank, world-size, node-local, and dataloader-worker queries. |
| `context` | Context managers for temporary process groups and ordered execution across ranks. |
| `decorators` | Decorators that run a function only on rank 0 or on one process per node. |
| `ops` | Small distributed coordination helpers like collection broadcast and agreement checks. |
| `utils` | Recursive collection helpers used by the broadcast utilities. |

## When To Use These

Reach for `dream_trainer.utils.dist` when you need:

- global or local rank information before the trainer has launched a `world`
- one process per node to perform setup work
- a cheap agreement check across ranks for configuration or dataset metadata
- recursive tensor handling inside nested Python containers

If you're inside trainer code and need mesh-aware behavior, prefer `self.world.*` instead.

## Common Imports

```python
from dream_trainer.utils import dist
from dream_trainer.utils.dist.ops import apply_to_collection, global_agreement
```

## Core

::: dream_trainer.utils.dist.core

## Context Managers

::: dream_trainer.utils.dist.context

## Decorators

::: dream_trainer.utils.dist.decorators

## Operations

::: dream_trainer.utils.dist.ops

## Supporting Utilities

::: dream_trainer.utils.dist.utils
