# API Reference

Welcome to the Dream Trainer API Reference. This documentation provides detailed information about all public classes, functions, and modules in the Dream Trainer framework.

## Overview

Dream Trainer is built with a modular, composable architecture. The main components are:

- **Trainers**: Core training loop implementations
- **Mixins**: Composable functionality that can be mixed into trainers
- **Callbacks**: Hooks for extending training behavior
- **Configuration**: Type-safe configuration management
- **Utilities**: Helper functions and utilities

## Quick Links

### Core Components

- [AbstractTrainer](trainers/abstract.md) - Base interface for all trainers
- [BaseTrainer](trainers/base.md) - Default trainer implementation
- [DreamTrainer](trainers/dream.md) - Production-ready trainer with all features

### Mixins

- [Setup Mixins](mixins/setup.md) - Model, optimizer, and dataloader setup
- [Evaluation Mixins](mixins/eval_metric.md) - Metrics and evaluation
- [Logger Mixins](mixins/loggers.md) - Logging integrations (WandB, TensorBoard)
- [Quantization Mixins](mixins/quantize.md) - FP8 and INT8 quantization

### Callbacks

- [Callback Base](callbacks/base.md) - Base callback interface
- [Checkpoint Callbacks](callbacks/checkpoint.md) - Model checkpointing
- [Monitoring Callbacks](callbacks/monitoring.md) - Training monitoring
- [Performance Callbacks](callbacks/performance.md) - Performance optimization

### Configuration

- [Parameter Classes](configuration/parameters.md) - Configuration parameters
- [Device Config](configuration/device.md) - Device and parallelism settings
- [Training Config](configuration/training.md) - Training hyperparameters

### Utilities

- [World Management](utilities/world.md) - Distributed training context
- [Data Utilities](utilities/data.md) - Data loading helpers
- [Common Utilities](utilities/common.md) - General utilities

## Usage Pattern

Dream Trainer follows a composition pattern where trainers are built by combining mixins:

```python
from dream_trainer import BaseTrainer
from dream_trainer.mixins import ModelSetupMixin, OptimizerSetupMixin

class MyTrainer(BaseTrainer, ModelSetupMixin, OptimizerSetupMixin):
    def training_step(self, batch, batch_idx):
        # Your training logic here
        pass
```

## Finding What You Need

### By Task

- **Starting a new project**: See [AbstractTrainer](trainers/abstract.md)
- **Setting up distributed training**: See [World Management](utilities/world.md)
- **Adding logging**: See [Logger Mixins](mixins/loggers.md)
- **Implementing checkpointing**: See [Checkpoint Callbacks](callbacks/checkpoint.md)
- **Optimizing performance**: See [Performance Callbacks](callbacks/performance.md)

### By Integration

- **PyTorch Lightning users**: Start with [BaseTrainer](trainers/base.md)
- **HuggingFace users**: See [DreamTrainer](trainers/dream.md)
- **Custom framework users**: Start with [AbstractTrainer](trainers/abstract.md)

## API Stability

!!! note "API Stability Guidelines"
    - Classes and functions marked as **public** (not prefixed with `_`) are stable
    - Internal APIs (prefixed with `_`) may change between minor versions
    - Deprecation warnings will be provided for at least one minor version

## Type Hints

All public APIs in Dream Trainer are fully type-hinted. We recommend using a type checker like `mypy` or `pyright` for the best development experience.

## Contributing

See our [Contributing Guide](../contributing.md) for information on adding or improving API documentation. 