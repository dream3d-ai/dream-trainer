# Contributing

Dream Trainer contributions should preserve the core design: plain PyTorch remains visible, distributed policy is explicit, and reusable infrastructure lives in mixins or callbacks instead of hiding model behavior.

## Before Changing Code

- Read [Core Concepts](core-concepts.md).
- Read the source around the lifecycle phase you are changing.
- Check whether the behavior belongs in a trainer hook, a mixin, a callback, a config dataclass, or a utility.
- Keep unrelated refactors out of focused changes.

## Development Setup

The package requires Python 3.10 or newer.

Install the package with the extras needed for the code path you are working on:

```bash
pip install -e ".[metrics,wandb]"
```

Optional extras:

```bash
pip install -e ".[rich,torchao]"
```

## Code Style

Use the repo's existing style:

- ordinary PyTorch modules and optimizers
- explicit trainer hooks
- dataclass configs
- callback lifecycle methods for reusable side effects
- type hints where they clarify public contracts
- focused comments only where ordering or distributed behavior is non-obvious

Run formatting and lint checks with the tools configured in `pyproject.toml`:

```bash
ruff check .
```

## Documentation Changes

Docs live in `docs/` and are built with MkDocs plus the `mkdocs-shadcn` theme.

Before adding a page:

- Add it to `mkdocs.yml` only when the file exists.
- Avoid broken Markdown links.
- Prefer links to existing pages over future placeholders.
- Keep examples runnable without private repository dependencies unless explicitly labeled production-pattern-only.
- Use current source as the source of truth.
- Prefer standard Python-Markdown and pymdown extensions supported by `mkdocs-shadcn`; avoid Material-only Markdown features.

Check docs with:

```bash
git diff --check -- docs mkdocs.yml
mkdocs build --strict
```

Run those commands from the `dream-trainer/` directory when checking MkDocs.

## Voice Guide

The docs have a consistent voice. Before writing or editing a page, know which [Diátaxis](https://diataxis.fr) mode it lives in, because the register changes by mode.

| Mode | Home | Register |
| --- | --- | --- |
| **Tutorial** (`tutorials/`) | Learning by doing | Narrator walking beside the reader. Cumulative, one project across pages. Show errors and fixes. |
| **How-To** (How-To Guides in nav) | Accomplishing a task | Neutral, crisp, decision-oriented. Open with a decision tree if the reader is choosing. |
| **Explanation** (`index.md`, `core-concepts.md`, `design-philosophy.md`, `comparison.md`) | Building the mental model | Essayistic. Analogies allowed, one per concept. No runnable code. |
| **Reference** (`api/**`) | Looking up facts | Dry, exhaustive, consistent. `mkdocstrings` + a short human intro. |

### Five voice rules

Apply these across all modes:

1. **Lead with the reader's problem, not the library's feature.** Replace "X does Y" openers with "You're trying to Z. Here's how X handles it." An opener like "Callbacks add cross-cutting behavior" tells the reader what the author knows. An opener like "Your trainer works. Now you need checkpoints, logging, profiling, and progress bars without turning `training_step` into a 500-line method" tells the reader why they should care.
2. **Show the sharp edges.** Dream Trainer's pitch is "structure without hiding the sharp edges." The docs should model that. Use explicit `!!! warning` and `!!! danger` admonitions for the subtle ordering constraints that matter (meta-device → apply_tp → compile → apply_fully_shard → materialize → init_weights → optimizers). Don't bury them in prose.
3. **Use analogies sparingly but deliberately.** One apt analogy per concept is worth more than a paragraph of definitions. "The device mesh is a named coordinate system" is earning rent; "FSDP is like a team sport" is not.
4. **Vary rhythm by mode.** Tutorials read like a narrator. Reference is terse. How-to is neutral. Explanation can be essayistic. If a tutorial page reads like a reference page, rewrite it.
5. **No hedge words.** "Usually", "generally", "often", "typically" are symptoms of not owning the decision. Either state the rule ("Use FSDP when the model doesn't fit replicated") or state the condition ("If the model fits replicated, use DDP; otherwise FSDP"). Remove hedges in review.

### Visual vocabulary

The `mkdocs.yml` config already loads `admonition`, `pymdownx.details`, `pymdownx.tabbed`, `pymdownx.superfences` with mermaid, and `pymdownx.snippets`. Use them:

- **Admonitions with consistent roles:** `!!! info` for "why this matters"; `!!! warning` for subtle gotchas; `!!! tip` for production patterns; `!!! danger` for "don't do this, here's the failure mode"; `!!! example` for long collapsible snippets.
- **Tabbed alternatives** (`=== "DDP"` / `=== "FSDP"`) when the reader is choosing a path. Two sequential code blocks hide the diff; tabs show it.
- **Mermaid decision trees** at the top of how-to pages with a real choice (parallelism, callbacks, checkpoint modes). Replace paragraph-form "choose X if... otherwise Y" with a diagram.
- **Canonical lifecycle diagram** via `--8<-- "includes/lifecycle.md"`. Do not re-invent the lifecycle in prose on a new page — include the snippet.
- **Page-type badge** as the first line under the title: `<small>📖 Explanation · ~8 min read</small>` or `<small>🛠️ How-to · copy-paste friendly</small>` or `<small>🎓 Tutorial · 30 min, requires 2 GPUs</small>`. Sets expectations immediately.

### Self-check before merging docs

- [ ] The page opens with the reader's problem, not the library's feature.
- [ ] No hedge words remain.
- [ ] Sharp edges are called out with admonitions, not buried in prose.
- [ ] The lifecycle is not re-listed in prose — the snippet is included.
- [ ] If the page describes a choice, a decision tree or tabs show the alternatives.
- [ ] The register matches the page's Diátaxis mode.

## Adding Trainer Features

Use this ownership rule:

| Behavior | Where It Belongs |
| --- | --- |
| Model architecture | trainer hook |
| Weight initialization or loading | `init_weights` |
| Optimizer ownership | `configure_optimizers` |
| Dataloader construction | `configure_dataloaders` or dataloader config factory |
| Loss computation | `training_step` |
| Validation semantics | `validation_step` |
| Cross-cutting lifecycle behavior | callback |
| Reusable setup behavior | mixin |
| Distributed mesh defaults | config or world utilities |

## Adding Callbacks

When adding a callback:

- Implement only the lifecycle hooks it needs.
- Use `RankZeroCallback` for single-writer side effects.
- Add `state_dict` and `load_state_dict` if the callback owns resumable state.
- Declare trainer interface expectations clearly.
- Keep expensive work opt-in and configurable.

## Adding Distributed Behavior

Distributed changes should be tested in the smallest useful order:

1. Single device.
2. DDP.
3. FSDP or HSDP.
4. Tensor, context, or pipeline parallelism.
5. Compile and performance optimizations.

Keep a non-compiled debug path where possible.

## Pull Request Checklist

- [ ] The change follows existing trainer, mixin, callback, or config boundaries.
- [ ] Public behavior is documented.
- [ ] Examples avoid private dependencies unless clearly labeled.
- [ ] `git diff --check` passes.
- [ ] `mkdocs build --strict` passes when docs change.
- [ ] Distributed behavior is tested in a suitable target environment when the change requires it.

## Changelog

There is no documented release or changelog process in the repo yet. Add a changelog page only after releases are tracked consistently.
