# dream-trainer

Composable distributed training framework built around PyTorch DTensor abstractions.

Dream Trainer gives you reusable building blocks for the trainer lifecycle, distributed setup, callbacks, checkpointing, and configuration — so the training algorithm stays explicit and ordinary PyTorch, while the framework handles distributed launch, device mesh construction, meta-device materialization, parallelism ordering, checkpoint state, and callback dispatch.

**Docs:** https://dream3d.ai/trainer

## Installation

```bash
pip install "dream-trainer[metrics,wandb]"
```

### Optional extras

| Extra | When to install |
| --- | --- |
| `wandb` | WandB logging (recommended — the default trainer composition imports the WandB mixin). |
| `metrics` | `torchmetrics` support via `EvalMetricMixin`. |
| `rich` | Richer trainer and model summaries. |
| `torchao` | FP8 and low-precision quantization workflows. |
| `cli` | Typer-based command surface — `benchmark`, `profile`, config modifiers. |
| `s3` | Save, resume, and export checkpoints directly to `s3://` URIs. |

> [!NOTE]
> The `s3` extra builds `s3torchconnectorclient` from source, which requires
> `cmake` and `clang` to be installed on your system (e.g.
> `apt install cmake clang`).

## A trainer in ten lines

```python
from dream_trainer import DreamTrainer

class MyTrainer(DreamTrainer):
    def configure_models(self):
        self.model = MyModel(self.config.model)

    def configure_optimizers(self):
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.lr)
        return {self.model: self.optimizer}

    def training_step(self, batch, batch_idx):
        logits = self.model(batch["input"])
        loss = F.cross_entropy(logits, batch["target"])
        self.backward(loss)
        return {"train/loss": loss, "train/grad_norm": self.step(self.optimizer)}

MyTrainer(config).fit()
```

No hidden decorators, no framework-owned `forward`. Launch with `torchrun` or Dream Trainer's `@entrypoint` helper — see the [Quick Start](https://dream3d.ai/trainer/getting-started/).

## What you get

- **Composable trainer mixins** — model, optimizer, dataloader, RNG, metric, and logging setup as independent pieces you combine into your own trainer.
- **DTensor-native parallelism** — FSDP2, tensor parallel, pipeline parallel, and hybrid sharding configured through `DeviceParameters`, not code rewrites.
- **Checkpointing** — distributed checkpointing (DCP) with sync or async saves (including pinned-memory staging), top-k retention, partial loads, weight export, and S3 storage.
- **Callbacks** — lifecycle hooks for EMA, benchmarking, fault tolerance, graph-break detection, media logging, and your own extensions.
- **Config-driven CLI** — `train`, `benchmark`, `profile`, and `export` subcommands with config modifiers, `--resume`, and `--init-from`.

## Development

```bash
git clone https://github.com/dream3d-ai/dream-trainer
cd dream-trainer
uv sync --all-extras
uv run pytest tests/
```

## License

[BSD 3-Clause](LICENSE) — (c) Dream3D, Inc. and affiliates.
