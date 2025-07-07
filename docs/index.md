# Dream Trainer Documentation

Dream Trainer is a powerful, distributed training framework built exclusively around PyTorch's new DTensor abstractions. It provides a flexible, composable approach that makes it easy to adopt the latest PyTorch DTensor APIs.

Dream Trainer was created to address these core issues:

- **Boilerplate Overload**: Each parallelism scheme (DDP, FSDP, tensor, pipeline, etc.) requires its own verbose, error-prone setup & configuration that must be applied in the correct order.
- **Legacy Trainer Limitations**: Most trainers are tightly coupled to old DDP/FSDP APIs and "zero-config" abstractions, making debugging harder and preventing them from taking advantage of new DTensor-based distributed patterns. Being DTensor-native makes code simpler and easier to debug.
- **Complexity in Real Workflows**: Even simple training scripts become unwieldy when mixing advanced parallelism, due to scattered configuration and framework assumptions.

## 🏗️ Design Principles

Dream Trainer is built on three core principles:

1. **Native PyTorch First**

   - Designed exclusively around PyTorch's new DTensor abstractions for simple but powerful parallelism
   - Direct integration with PyTorch's ecosystem (torchao, torchft, DCP, torchrun)

2. **Minimal Assumptions**

   - Let users make their own choices
   - No automatic model wrapping or hidden behaviors
   - Assume users know what they're doing with advanced parallelism

3. **Composable Architecture**
   - Trainer is a composition of mixins
   - Take what you need, drop the rest
   - Write your own components when needed
   - Callback system for drop-in modifications to the loop

## 🌟 Key Features

### Parallelism Support

Dream Trainer provides simple configuration for all PyTorch parallelism schemes:

- **Data Parallelism**: Basic multi-GPU training with PyTorch's `replicate()` API
- **FSDP2**: Second-generation Fully Sharded Data Parallel built on DTensor
- **Tensor Parallelism (TP)**: Parameter-wise sharding via DTensor layouts; composable with FSDP2 for HSDP
- **Context Parallelism (CP)**: Sequence parallelism for extremely long contexts
- **Pipeline Parallelism (PP)**: Layer pipelining across GPUs / nodes with automatic schedule search


Unlike monolithic frameworks, Dream Trainer uses mixins to let you pick exactly what you need:

```python
# Minimal trainer for research
class SimpleTrainer(BaseTrainer, SetupMixin):
    def training_step(self, batch, batch_idx):
        loss = self.model(batch)
        self.backward(loss)
        return {"loss": loss}

# Production trainer with all the bells and whistles
class ProductionTrainer(BaseTrainer, SetupMixin, EvalMetricMixin, 
                       WandBLoggerMixin, QuantizeMixin):
    # Same training_step, but now with metrics, logging, and quantization!
```

### Other Features via Callbakcs

- **Checkpointing** DCP-based checkpointing with async checkpoint support
- **Built-in Fault Tolerance** via torchft
- **Native FP8 Quantization** via torchao
- **Custom Callbacks** for extensibility
- **Build-your-own-trainer** by composing mixin primitives

## 🤔 Why Dream Trainer vs. Other Frameworks?

While PyTorch Lightning, Accelerate and DeepSpeed simplify distributed training, they revolve around classic DDP/FSDP wrappers and hide key details behind heavyweight base classes. Dream Trainer takes a different path:

- **DTensor-native** from day one—every parameter is a `DTensor`, so new sharding layouts appear the moment they land in PyTorch nightly.
- **Parallel schemes (FSDP2, TP, PP, CP)** are first-class, composable primitives, not bolt-on "plugins".
- **Mix-and-match** – DreamTrainer is designed around mixins to maximize composability.
- **Minimal magic** – no metaclasses, no `LightningModule`; your model remains a plain `nn.Module`.

## 📚 Documentation Structure

### Core Concepts

- [Getting Started](getting-started.md) - Installation and basic usage
- [Configuration](configuration.md) - Detailed configuration options
- [Trainer Guide](trainer-guide.md) - Creating custom trainers
- [Callbacks](callbacks.md) - Extending functionality with callbacks

### Advanced Features

- [Distributed Training](distributed.md) - Multi-GPU and multi-node training
- [Mixed Precision](mixed-precision.md) - FP16, BF16, and FP8 training
- [Checkpointing](checkpointing.md) - Model saving and loading
- [Logging](logging.md) - Metrics and experiment tracking

### Examples & Tutorials

- [Basic Examples](examples/basic.md) - Simple training examples
- [Advanced Examples](examples/advanced.md) - Complex use cases
- [Best Practices](best-practices.md) - Training optimization tips

### API Reference

- [Trainer API](api/trainer.md) - Core trainer classes
- [Config API](api/config.md) - Configuration classes
- [Callback API](api/callbacks.md) - Built-in callbacks
- [Utils API](api/utils.md) - Utility functions


## 🔧 Requirements

- Python >= 3.10
- PyTorch >= 2.7.0
- CUDA-capable GPU (recommended)

## 📖 Next Steps

- Follow the [Getting Started](getting-started.md) guide to install and set up Dream Trainer
- Check out the [Examples](examples/basic.md) for complete working code
- Read the [Trainer Guide](trainer-guide.md) to create your own custom trainer

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](contributing.md) for details.
