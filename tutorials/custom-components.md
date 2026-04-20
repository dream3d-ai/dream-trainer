---
title: "Tutorial 5 — Extending Meteor"
---

# Tutorial 5 — Extending Meteor with Custom Components

<small>🎓 Tutorial · ~25 min · any number of GPUs</small>

## Where We Left Off

[Tutorial 4](production.md) gave us **Meteor v3**: a split-file 1.3B trainer with FSDP, compile, async checkpointing, WandB, LR and structure summaries. It produces a clean `val/loss` curve.

A single scalar is a thin view of the model. Before Meteor v4 is useful to anyone but its author, we want:

1. **Better validation signal.** Perplexity, token-level accuracy, maybe per-domain slices.
2. **A curriculum.** Start training on short sequences, grow to long sequences over the first few thousand steps — the kind of policy that needs lifecycle awareness and resumable state.

Both are extensions, not modifications. v3's hooks don't change. We add a metric collection and a callback.

## The Rule, One More Time

Before writing either, re-apply the rule from [Core Concepts](../core-concepts.md#mixins-vs-callbacks):

- **Metrics care about what the model *computes*.** They belong with the trainer — declared in config, used in `validation_step`. A metric is not a callback.
- **A curriculum cares about *when* the trainer does things.** It doesn't touch parameters or layers; it adjusts a knob based on step count. That's a callback.

If you confuse which is which, you'll end up writing either a 500-line `validation_step` (everything crammed into hooks) or a callback that reaches into `self.trainer.model.layers[0]` to read a weight (callback pretending to be a hook). The split keeps each piece simple.

## Custom Metrics — `metrics.py`

Metrics live in their own module so the trainer doesn't grow a new responsibility. Meteor uses `torchmetrics` for the mechanics — it handles cross-rank all-reduce correctly under any mesh shape.

```python
# meteor/metrics.py
import torch
import torchmetrics as tm
from torchmetrics import MetricCollection


class Perplexity(tm.Metric):
    full_state_update = False

    def __init__(self):
        super().__init__()
        self.add_state("total_nll", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_tokens", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, logits: torch.Tensor, targets: torch.Tensor):
        nll = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            reduction="sum",
        )
        self.total_nll += nll.detach()
        self.total_tokens += targets.numel()

    def compute(self) -> torch.Tensor:
        return torch.exp(self.total_nll / self.total_tokens)


def meteor_metrics() -> MetricCollection:
    return MetricCollection(
        {
            "perplexity": Perplexity(),
            "token_accuracy": tm.Accuracy(task="multiclass", num_classes=50_257, top_k=1),
        },
        prefix="val/",
    )
```

The config references the collection factory; the trainer installs it on itself.

```python
# config.py — additions
from .metrics import meteor_metrics


@dataclass(kw_only=True)
class MeteorConfig(DreamTrainerConfig):
    ...
    metrics: MetricCollection = field(default_factory=meteor_metrics)
```

```python
# train.py — one hook + one line in validation_step
class MeteorTrainer(DreamTrainer):
    ...

    def configure_metrics(self):
        self.metrics = self.config.metrics

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        tokens = batch
        logits = self.model(tokens[:, :-1])
        targets = tokens[:, 1:]
        loss = F.cross_entropy(
            logits.reshape(-1, self.config.model.vocab_size),
            targets.reshape(-1),
        )
        self.metrics.update(logits, targets)
        return {"val/loss": loss}
```

That's the entire integration. Dream Trainer:

- resets `self.metrics` at the start of every validation epoch,
- calls `.update()` lines in `validation_step`,
- runs `.compute()` after the last validation batch and logs the result,
- gathers across ranks correctly under DDP, FSDP, HSDP, whatever you configured.

You only wrote the `.update()` line. That's the split working.

!!! tip "Metrics belong in `configure_metrics`, not `configure_models`"
    It's tempting to assign `self.metrics = ...` in `configure_models` since it's where "things get set up". Don't. `configure_models` is under meta-device; metric state is a real tensor. `configure_metrics` runs later, on real device, and Dream Trainer knows to reset and gather metric state correctly — which it can't do for arbitrary attributes set in `configure_models`.

## A Curriculum Callback

The policy: start at `seq_len=512`, grow to the full `2048` linearly over the first 4000 steps. Pad short sequences so the model shape doesn't change.

This is not a model change. The model always sees sequences of length `seq_len`. What changes is what the *dataloader produces*, based on a knob that the callback updates.

The callback owns:

- A `current_seq_len` value that the data module reads each step.
- A `state_dict` / `load_state_dict` pair so a resumed run picks up at the same curriculum step.

```python
# meteor/callbacks.py
from dream_trainer.callbacks import Callback


class SequenceLengthCurriculum(Callback):
    def __init__(self, start: int, end: int, warmup_steps: int):
        self.start = start
        self.end = end
        self.warmup_steps = warmup_steps
        self.current = start
        self.step = 0

    def pre_train_step(self, batch, batch_idx):
        frac = min(1.0, self.step / self.warmup_steps)
        self.current = int(self.start + frac * (self.end - self.start))
        self.step += 1

    def state_dict(self):
        return {"step": self.step, "current": self.current}

    def load_state_dict(self, state):
        self.step = state["step"]
        self.current = state["current"]
```

Two things make this callback cheap to maintain:

- **It doesn't inherit anything except `Callback`.** No mixin dependency, no trainer type parameter, no lifecycle other than `pre_train_step`.
- **It owns its state.** Dream Trainer automatically includes `state_dict()` in the aggregate trainer state saved by `AsyncCheckpointCallback`. Resume just works.

The data module reads `current`:

```python
# data.py — curriculum-aware variant
from .callbacks import SequenceLengthCurriculum


def build_loader(cfg: DataConfig, curriculum: SequenceLengthCurriculum, *, rank, world_size, shuffle):
    class CurriculumDataset(torch.utils.data.Dataset):
        def __init__(self, inner, curriculum):
            self.inner = inner
            self.curriculum = curriculum

        def __len__(self):
            return len(self.inner)

        def __getitem__(self, idx):
            tokens = self.inner[idx]
            return tokens[: self.curriculum.current]

    dataset = CurriculumDataset(TokenStream(cfg.path, seq_len=2048), curriculum)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=shuffle)
    return DataLoader(dataset, batch_size=cfg.batch_size, sampler=sampler, num_workers=cfg.num_workers)
```

Wiring in the entrypoint:

```python
@entrypoint
def main():
    config = v3_config()
    curriculum = SequenceLengthCurriculum(start=512, end=2048, warmup_steps=4000)
    config.callbacks = callbacks.CallbackCollection([
        callbacks.LoggerCallback(log_every_n_train_batches=20),
        callbacks.LRLoggerCallback(),
        callbacks.ProgressBar(metric="train/loss"),
        callbacks.AsyncCheckpointCallback(config.checkpoint_parameters),
        callbacks.ModelWatchCallback(),
        callbacks.TrainerSummary(),
        callbacks.ModelSummary(),
        curriculum,
    ])
    trainer = MeteorTrainer(config)
    # Dataloader factory needs the curriculum instance; wire through config in real code.
    trainer.fit()
```

In production you'd route `curriculum` into the config rather than plumbing it through the entrypoint by hand — the pattern is the same.

!!! warning "Rank-zero vs per-rank callbacks"
    `SequenceLengthCurriculum` runs on every rank. That's correct here — every rank needs the same `current` value to produce consistent batch shapes. If it ran only on rank zero, ranks would disagree about sequence length and your all-reduce would trip.

    I/O-heavy callbacks (uploading a checkpoint, posting to a dashboard) should inherit `RankZeroCallback` instead, so Dream Trainer filters dispatch to rank zero automatically. The rule: if the callback *reads or modifies training state consistently across ranks*, use `Callback`. If it *writes to an external system once*, use `RankZeroCallback`.

## A Tour Of The Extension Points

You've now touched every extension point Dream Trainer offers. Here's the map:

| Concern | Extension point | Where Meteor uses it |
| --- | --- | --- |
| Model architecture | `configure_models` | `Meteor(**vars(self.config.model))` |
| Weight initialization | `init_weights` | `self.model.init_weights()` |
| Parallelism policy | `apply_fully_shard`, `apply_compile`, `apply_replicate`, `apply_tensor_parallel`, ... | v1 used `apply_replicate`, v2+ use `apply_fully_shard` |
| Optimizer ownership | `configure_optimizers` | AdamW with the parameter groups we want |
| LR schedule | `configure_schedulers` | v3 added linear warmup |
| Rank-aware data | `configure_dataloaders` + data module | `build_loader(..., rank=self.world.dp_rank)` |
| Validation metrics | `configure_metrics` + `validation_step.update()` | v4 perplexity and accuracy |
| Lifecycle behavior | `Callback` / `RankZeroCallback` | v4 curriculum |
| State persistence | `model_state_dict` on the trainer, `state_dict` on callbacks | DCP handles the rest |

What's notable is what's *not* on this list: there's no "how do I change the training loop", no "how do I override `fit`". Every behavior you'd want has a dedicated place. That's the payoff for the two rules we kept repeating:

- Hooks care about what the model is. Callbacks care about what the trainer is doing.
- Config describes. Trainer performs.

## Sanity Checks Before Moving On

- [ ] `val/perplexity` and `val/token_accuracy` appear in WandB alongside `val/loss`.
- [ ] Killing a run at step 2000 and restarting resumes with `current ≈ 1280` — the curriculum state came back.
- [ ] A run launched with `SINGLE_DEVICE` produces the same metric values (within float noise) as a DDP run of world size 1 — metric all-reduce is a no-op there.
- [ ] `self.metrics.update(...)` doesn't appear in `training_step`; metrics are a validation concept.

## The New Thing We Have Now

**Meteor v4**: perplexity and accuracy metrics, a resumable curriculum callback, every extension point exercised without touching the trainer's existing seven hooks.

## Where To Go Next

The Meteor arc is complete. You've built up from a 125M-parameter single-GPU trainer to a 1.3B-parameter production trainer with custom metrics and lifecycle extensions. Every step was a small diff on the previous one, because the lifecycle didn't change.

From here:

- [Trainer Guide](../trainer-guide.md) — the hook catalogue, cross-referenced by "when you need it".
- [Parallelism](../parallelism.md) — if you want to go past FSDP into tensor, context, or pipeline parallelism.
- [Callbacks](../callbacks.md) — the built-in callback catalogue: EMA, FP8, fault tolerance, profiling.
- [Design Philosophy](../design-philosophy.md) — why the API is shaped this way, in essay form.

If something in Meteor's arc didn't land, open an issue. The tutorials are supposed to be the path we *wish* someone had shown us — and every page is negotiable.
