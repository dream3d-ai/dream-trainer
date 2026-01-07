# Dream Trainer

**A composable distributed training framework built on PyTorch DTensor**

Dream Trainer simplifies advanced distributed training by providing a flexible, mixin-based architecture that embraces PyTorch's next-generation DTensor abstractions. Write clean training code that scales from a single GPU to thousands.

<div class="grid cards" markdown>

-   :material-clock-fast:{ .lg .middle } __Get Started in 5 Minutes__

    ---

    Install Dream Trainer and train your first model with advanced parallelism

    [:octicons-arrow-right-24: Getting Started](getting-started.md)

-   :material-puzzle:{ .lg .middle } __Composable by Design__

    ---

    Mix and match features with our mixin architecture - take only what you need

    [:octicons-arrow-right-24: Core Concepts](core-concepts.md)

-   :material-server-network:{ .lg .middle } __DTensor-Native__

    ---

    First-class support for FSDP2, Tensor Parallelism, Pipeline Parallelism, and more

    [:octicons-arrow-right-24: Parallelism Guide](parallelism.md)

-   :material-cog:{ .lg .middle } __Extensible Callbacks__

    ---

    Customize every aspect of training with the powerful callback system

    [:octicons-arrow-right-24: Callbacks](callbacks.md)

</div>

---

## Why Dream Trainer?

### The Problem

Modern distributed training is complex. Each parallelism scheme requires verbose, error-prone setup that must be applied in the correct order. Traditional frameworks are tightly coupled to legacy APIs, making debugging difficult and preventing adoption of new DTensor-based patterns.

### The Solution

Dream Trainer takes a different approach:

=== "Composable Architecture"

    ```python
    # Start minimal - just the essentials
    class SimpleTrainer(BaseTrainer, SetupMixin):
        def training_step(self, batch, batch_idx):
            loss = self.model(batch)
            self.backward(loss)
            return {"loss": loss}

    # Add features as you need them
    class ProductionTrainer(BaseTrainer, SetupMixin,
                           EvalMetricMixin, WandBLoggerMixin):
        pass  # Now with metrics and logging!
    ```

=== "DTensor-Native"

    ```python
    # Every parameter is a DTensor - clean, debuggable distributed code
    def apply_tensor_parallel(self, tp_mesh):
        for layer in self.model.layers:
            parallelize_module(layer, tp_mesh, {
                "attn.qkv": ColwiseParallel(),
                "attn.out": RowwiseParallel(),
            })
    ```

=== "Config as Code"

    ```python
    @dataclass
    class MyConfig(DreamTrainerConfig):
        learning_rate: float = 3e-4  # Type-checked!
        num_layers: int = 12         # IDE auto-completion!

        def validate(self):
            assert self.num_layers >= 1, "Need at least 1 layer"
    ```

---

## Feature Highlights

### Parallelism Support

| Strategy | Description | Use Case |
|----------|-------------|----------|
| **FSDP2** | Fully Sharded Data Parallel on DTensor | Memory-efficient large model training |
| **Tensor Parallelism** | Parameter sharding across devices | Very large layers (LLM attention) |
| **Pipeline Parallelism** | Layer pipelining with auto-scheduling | Models too large for single device |
| **Context Parallelism** | Sequence parallelism | Extremely long sequences |
| **HSDP** | Hybrid sharding (TP + FSDP) | Multi-node clusters |

### Training Features

- **Automatic gradient accumulation** with proper scaling
- **Mixed precision training** (FP16, BF16, FP8)
- **Gradient clipping** per optimizer
- **Learning rate scheduling** with warmup support
- **Distributed checkpointing** via PyTorch DCP
- **Fault tolerance** via torchft integration

### Developer Experience

- **Type-safe configurations** with dataclass validation
- **Comprehensive callbacks** for every training phase
- **Built-in logging** to WandB, TensorBoard, and more
- **Progress tracking** with tqdm integration
- **Debugging utilities** for distributed training

---

## Quick Example

```python
from dataclasses import dataclass
from dream_trainer import DreamTrainer, DreamTrainerConfig
from dream_trainer.configs import DeviceParameters, TrainingParameters
from dream_trainer.callbacks import CheckpointCallback, LoggerCallback

@dataclass
class MyConfig(DreamTrainerConfig):
    hidden_size: int = 768
    num_layers: int = 12
    learning_rate: float = 3e-4

class MyTrainer(DreamTrainer):
    config: MyConfig

    def configure_models(self):
        self.model = TransformerModel(
            hidden_size=self.config.hidden_size,
            num_layers=self.config.num_layers,
        )

    def configure_optimizers(self):
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate
        )

    def configure_dataloaders(self):
        return train_loader, val_loader

    def training_step(self, batch, batch_idx):
        loss = self.model(batch["input_ids"])
        self.backward(loss)

        if not self.is_accumulating_gradients:
            self.step(self.optimizer)

        return {"loss": loss}

# Configure and run
config = MyConfig(
    device_parameters=DeviceParameters.FSDP(dp_shard=4),
    training_parameters=TrainingParameters(n_epochs=10),
)

trainer = MyTrainer(config)
trainer.fit()
```

---

## Documentation

### Getting Started

<div class="grid cards" markdown>

-   :material-download:{ .lg .middle } [**Installation**](installation.md)

    System requirements, pip install, optional dependencies

-   :material-rocket-launch:{ .lg .middle } [**Quick Start**](getting-started.md)

    Your first trainer in 5 minutes

-   :material-book-open:{ .lg .middle } [**Core Concepts**](core-concepts.md)

    DTensor, mixins, and the training loop

</div>

### User Guide

<div class="grid cards" markdown>

-   :material-cog:{ .lg .middle } [**Configuration**](configuration.md)

    Training, device, and checkpoint parameters

-   :material-puzzle-outline:{ .lg .middle } [**Trainer Guide**](trainer-guide.md)

    Building custom trainers with mixins

-   :material-webhook:{ .lg .middle } [**Callbacks**](callbacks.md)

    Extend training with custom callbacks

-   :material-server-network:{ .lg .middle } [**Parallelism**](parallelism.md)

    FSDP2, TP, PP, CP, and HSDP

-   :material-bug:{ .lg .middle } [**Debugging**](debugging.md)

    Debug distributed training issues

-   :material-speedometer:{ .lg .middle } [**Performance**](performance.md)

    Optimize training throughput

</div>

### Tutorials

<div class="grid cards" markdown>

-   :material-school:{ .lg .middle } [**Your First Trainer**](tutorials/first-trainer.md)

    Step-by-step guide to building a trainer

-   :material-chip:{ .lg .middle } [**Multi-GPU Training**](tutorials/multi-gpu.md)

    Scale to multiple GPUs with FSDP2

-   :material-wrench:{ .lg .middle } [**Custom Components**](tutorials/custom-components.md)

    Write your own mixins and callbacks

-   :material-factory:{ .lg .middle } [**Production Setup**](tutorials/production.md)

    Deploy to clusters with fault tolerance

</div>

### Examples

<div class="grid cards" markdown>

-   :material-image:{ .lg .middle } [**Vision Models**](examples/vision.md)

    Image classification, segmentation, diffusion

-   :material-text:{ .lg .middle } [**Language Models**](examples/nlp.md)

    GPT, LLaMA, and transformer training

-   :material-image-text:{ .lg .middle } [**Multi-Modal**](examples/multimodal.md)

    Vision-language models and CLIP

-   :material-code-braces:{ .lg .middle } [**Advanced Patterns**](examples/advanced.md)

    EMA, knowledge distillation, curriculum learning

</div>

### API Reference

<div class="grid cards" markdown>

-   :material-api:{ .lg .middle } [**API Overview**](api/index.md)

    Complete API documentation

-   :material-school:{ .lg .middle } [**Trainers**](api/trainers/base.md)

    AbstractTrainer, BaseTrainer, DreamTrainer

-   :material-puzzle:{ .lg .middle } [**Mixins**](api/mixins/setup.md)

    SetupMixin, EvalMetricMixin, LoggerMixin

-   :material-webhook:{ .lg .middle } [**Callbacks**](api/callbacks/base.md)

    Checkpoint, logging, performance callbacks

</div>

---

## Requirements

| Requirement | Version |
|-------------|---------|
| Python | >= 3.10 |
| PyTorch | >= 2.7.0 |
| CUDA | 11.8+ (recommended) |

---

## Community

- [:fontawesome-brands-github: GitHub](https://github.com/dream3d/dream-trainer) - Source code and issues
- [:fontawesome-brands-discord: Discord](https://discord.gg/dream-trainer) - Chat with the community
- [:fontawesome-brands-twitter: Twitter](https://twitter.com/dream3d_ai) - Latest updates

---

## License

Dream Trainer is released under the [Apache 2.0 License](https://github.com/dream3d/dream-trainer/blob/main/LICENSE).
