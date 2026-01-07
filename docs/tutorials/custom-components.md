# Tutorial: Custom Components

This tutorial teaches you how to extend Dream Trainer by creating custom mixins, callbacks, and configurations. You'll learn to build reusable components that integrate seamlessly with the framework.

## What You'll Learn

- How to create custom callbacks
- How to build custom mixins
- How to design custom configuration classes
- Best practices for component design

## Prerequisites

- Completed [Your First Trainer](first-trainer.md)
- Understanding of Python classes and inheritance
- Familiarity with Dream Trainer's mixin architecture

---

## Part 1: Custom Callbacks

Callbacks are the simplest way to extend Dream Trainer. They hook into the training lifecycle without modifying your trainer code.

### Basic Callback Structure

```python
from dream_trainer.callbacks import Callback
from dream_trainer.trainer.mixins import SetupMixin


class MyCallback(Callback[SetupMixin]):
    """A custom callback.

    The type parameter [SetupMixin] indicates this callback
    requires a trainer with SetupMixin functionality.
    """

    def __init__(self, my_param: str = "default"):
        super().__init__()
        self.my_param = my_param

    # Lifecycle hooks - implement only what you need

    def pre_fit(self):
        """Called before training starts."""
        print(f"Training starting with param: {self.my_param}")

    def post_fit(self):
        """Called after training ends."""
        print("Training complete!")

    def pre_train_epoch(self):
        """Called at the start of each training epoch."""
        print(f"Epoch {self.trainer.current_epoch} starting")

    def post_train_step(self, output: dict, batch_idx: int):
        """Called after each training step.

        Args:
            output: Dictionary returned from training_step()
            batch_idx: Index of the batch in the epoch
        """
        if batch_idx % 100 == 0:
            print(f"Step {batch_idx}: loss = {output['loss']:.4f}")
```

### Complete Callback Lifecycle

Here are all available callback hooks:

```python
class FullLifecycleCallback(Callback[SetupMixin]):
    """Demonstrates all callback hooks."""

    # === Setup Phase ===
    def pre_launch(self):
        """Before distributed backend initialization."""
        pass

    def post_launch(self):
        """After distributed backend is ready."""
        pass

    def pre_configure(self):
        """Before model configuration."""
        pass

    def post_configure(self):
        """After models are configured (on meta device)."""
        pass

    def pre_setup(self):
        """Before component setup (parallelism, weights, optimizers)."""
        pass

    def post_setup(self):
        """After all components are ready."""
        pass

    # === Training Phase ===
    def pre_fit(self):
        """Before the training loop starts."""
        pass

    def post_fit(self):
        """After training completes."""
        pass

    def pre_epoch(self):
        """At the start of each epoch."""
        pass

    def post_epoch(self):
        """At the end of each epoch."""
        pass

    def pre_train_epoch(self):
        """Before training portion of epoch."""
        pass

    def post_train_epoch(self):
        """After training portion of epoch."""
        pass

    def pre_train_step(self, batch_idx: int):
        """Before each training step."""
        pass

    def post_train_step(self, output: dict, batch_idx: int):
        """After each training step."""
        pass

    def pre_optimizer_step(self, optimizer_name: str):
        """Before optimizer.step() is called."""
        pass

    def post_optimizer_step(self, optimizer_name: str):
        """After optimizer.step() is called."""
        pass

    def pre_optimizer_zero_grad(self, optimizer_name: str):
        """Before optimizer.zero_grad() is called."""
        pass

    def post_optimizer_zero_grad(self, optimizer_name: str):
        """After optimizer.zero_grad() is called."""
        pass

    # === Validation Phase ===
    def pre_validation_epoch(self):
        """Before validation starts."""
        pass

    def post_validation_epoch(self):
        """After validation completes."""
        pass

    def pre_validation_step(self, batch_idx: int):
        """Before each validation step."""
        pass

    def post_validation_step(self, output: dict, batch_idx: int):
        """After each validation step."""
        pass

    # === Context Managers ===
    def train_context(self):
        """Context manager wrapping each training step.

        Returns:
            Context manager or None
        """
        return None

    def validation_context(self):
        """Context manager wrapping each validation step."""
        return None

    # === Error Handling ===
    def on_interrupt(self, exception: BaseException):
        """Called when training is interrupted (e.g., KeyboardInterrupt)."""
        pass
```

### Example: Learning Rate Warmup Callback

```python
from dream_trainer.callbacks import Callback
from dream_trainer.trainer.mixins import SetupMixin


class LinearWarmup(Callback[SetupMixin]):
    """Linear learning rate warmup.

    Gradually increases learning rate from 0 to target over warmup_steps.
    """

    def __init__(self, warmup_steps: int = 1000):
        super().__init__()
        self.warmup_steps = warmup_steps
        self._initial_lrs: dict[str, list[float]] = {}

    def post_setup(self):
        """Store initial learning rates after optimizers are created."""
        for name, optimizer in self.trainer.named_optimizers().items():
            self._initial_lrs[name] = [
                group['lr'] for group in optimizer.param_groups
            ]
            # Set initial LR to 0
            for group in optimizer.param_groups:
                group['lr'] = 0.0

    def post_train_step(self, output: dict, batch_idx: int):
        """Update learning rate after each step."""
        if self.trainer.global_step >= self.warmup_steps:
            return  # Warmup complete

        # Calculate warmup factor
        warmup_factor = self.trainer.global_step / self.warmup_steps

        # Update all optimizers
        for name, optimizer in self.trainer.named_optimizers().items():
            initial_lrs = self._initial_lrs[name]
            for group, initial_lr in zip(optimizer.param_groups, initial_lrs):
                group['lr'] = initial_lr * warmup_factor

    def state_dict(self) -> dict:
        """Save callback state for checkpointing."""
        return {"initial_lrs": self._initial_lrs}

    def load_state_dict(self, state_dict: dict):
        """Restore callback state from checkpoint."""
        self._initial_lrs = state_dict["initial_lrs"]
```

### Example: Early Stopping Callback

```python
from dream_trainer.callbacks import Callback
from dream_trainer.trainer.mixins import SetupMixin


class EarlyStopping(Callback[SetupMixin]):
    """Stop training when a metric stops improving.

    Args:
        monitor: Metric to monitor (e.g., "val_loss")
        patience: Number of epochs without improvement before stopping
        mode: "min" if lower is better, "max" if higher is better
        min_delta: Minimum change to qualify as an improvement
    """

    def __init__(
        self,
        monitor: str = "val_loss",
        patience: int = 3,
        mode: str = "min",
        min_delta: float = 0.0,
    ):
        super().__init__()
        self.monitor = monitor
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta

        self.best_value: float | None = None
        self.epochs_without_improvement = 0

    def post_validation_epoch(self):
        """Check if we should stop training."""
        # Get current metric value
        if not hasattr(self.trainer, '_last_val_metrics'):
            return

        current = self.trainer._last_val_metrics.get(self.monitor)
        if current is None:
            return

        # Check for improvement
        if self.best_value is None:
            self.best_value = current
            return

        if self._is_improvement(current):
            self.best_value = current
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1

        # Check if we should stop
        if self.epochs_without_improvement >= self.patience:
            print(f"Early stopping: {self.monitor} hasn't improved for {self.patience} epochs")
            self.trainer._should_stop = True

    def _is_improvement(self, current: float) -> bool:
        if self.mode == "min":
            return current < self.best_value - self.min_delta
        else:
            return current > self.best_value + self.min_delta

    def state_dict(self) -> dict:
        return {
            "best_value": self.best_value,
            "epochs_without_improvement": self.epochs_without_improvement,
        }

    def load_state_dict(self, state_dict: dict):
        self.best_value = state_dict["best_value"]
        self.epochs_without_improvement = state_dict["epochs_without_improvement"]
```

### Example: Gradient Logging Callback

```python
import torch
from dream_trainer.callbacks import Callback
from dream_trainer.trainer.mixins import SetupMixin


class GradientMonitor(Callback[SetupMixin]):
    """Monitor gradient statistics during training."""

    def __init__(self, log_every_n_steps: int = 100):
        super().__init__()
        self.log_every_n_steps = log_every_n_steps

    def pre_optimizer_step(self, optimizer_name: str):
        """Compute gradient statistics before the step."""
        if self.trainer.global_step % self.log_every_n_steps != 0:
            return

        model = self.trainer.get_model_by_optimizer(optimizer_name)
        if model is None:
            return

        # Compute gradient statistics
        grad_norms = []
        for param in model.parameters():
            if param.grad is not None:
                grad_norms.append(param.grad.norm().item())

        if grad_norms:
            stats = {
                "grad/mean_norm": sum(grad_norms) / len(grad_norms),
                "grad/max_norm": max(grad_norms),
                "grad/min_norm": min(grad_norms),
            }

            # Log if trainer has logging capability
            if hasattr(self.trainer, 'log_scalar'):
                for key, value in stats.items():
                    self.trainer.log_scalar(key, value)
            else:
                print(f"Step {self.trainer.global_step}: {stats}")
```

### Rank-Zero Callbacks

For operations that should only happen on rank 0 (e.g., logging, saving):

```python
from dream_trainer.callbacks import RankZeroCallback


class MyLoggingCallback(RankZeroCallback):
    """Only runs on rank 0 - safe for logging/saving."""

    def post_train_step(self, output: dict, batch_idx: int):
        # This only runs on rank 0
        print(f"Loss: {output['loss']:.4f}")
```

---

## Part 2: Custom Mixins

Mixins add reusable functionality to trainers. They're more powerful than callbacks but require more integration.

### Basic Mixin Structure

```python
from typing import TYPE_CHECKING
from dream_trainer import BaseTrainer

if TYPE_CHECKING:
    from dream_trainer.configs import BaseTrainerConfig


class MyMixin:
    """A custom mixin for trainers.

    Mixins can:
    - Add new methods to the trainer
    - Override existing methods (use super() carefully)
    - Add new attributes
    - Require specific configuration fields
    """

    # Type hints for IDE support
    config: "BaseTrainerConfig"

    def my_custom_method(self):
        """New method added by this mixin."""
        print("Custom functionality!")

    def setup(self):
        """Override setup to add custom initialization.

        Always call super() to maintain the mixin chain!
        """
        super().setup()  # Call next in MRO
        self._setup_my_stuff()

    def _setup_my_stuff(self):
        """Internal setup method."""
        self.my_attribute = "initialized"
```

### Example: Data Augmentation Mixin

```python
from dataclasses import dataclass
from typing import Any
import torch
import torch.nn as nn


@dataclass
class AugmentationConfigMixin:
    """Configuration for data augmentation."""
    augmentation_enabled: bool = True
    mixup_alpha: float = 0.2
    cutmix_alpha: float = 1.0
    cutmix_prob: float = 0.5


class AugmentationMixin:
    """Adds data augmentation to training.

    Implements mixup and cutmix augmentation strategies.
    """

    config: AugmentationConfigMixin

    def setup(self):
        """Initialize augmentation state."""
        super().setup()
        self._aug_enabled = self.config.augmentation_enabled

    def augment_batch(self, images: torch.Tensor, labels: torch.Tensor):
        """Apply augmentation to a batch.

        Args:
            images: Batch of images [B, C, H, W]
            labels: Batch of labels [B] or [B, num_classes]

        Returns:
            Augmented images and labels
        """
        if not self._aug_enabled or not self.training:
            return images, labels

        # Randomly choose augmentation
        if torch.rand(1).item() < self.config.cutmix_prob:
            return self._cutmix(images, labels)
        else:
            return self._mixup(images, labels)

    def _mixup(self, images: torch.Tensor, labels: torch.Tensor):
        """Apply mixup augmentation."""
        alpha = self.config.mixup_alpha
        if alpha <= 0:
            return images, labels

        lam = torch.distributions.Beta(alpha, alpha).sample().item()
        batch_size = images.size(0)
        index = torch.randperm(batch_size, device=images.device)

        mixed_images = lam * images + (1 - lam) * images[index]

        # For soft labels
        if labels.dim() == 1:
            # Convert to one-hot
            num_classes = getattr(self.config, 'num_classes', 10)
            labels = torch.nn.functional.one_hot(labels, num_classes).float()

        mixed_labels = lam * labels + (1 - lam) * labels[index]

        return mixed_images, mixed_labels

    def _cutmix(self, images: torch.Tensor, labels: torch.Tensor):
        """Apply cutmix augmentation."""
        alpha = self.config.cutmix_alpha
        if alpha <= 0:
            return images, labels

        lam = torch.distributions.Beta(alpha, alpha).sample().item()
        batch_size = images.size(0)
        index = torch.randperm(batch_size, device=images.device)

        # Get bounding box
        W, H = images.size(3), images.size(2)
        cut_rat = (1 - lam) ** 0.5
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)

        cx = torch.randint(W, (1,)).item()
        cy = torch.randint(H, (1,)).item()

        x1 = max(cx - cut_w // 2, 0)
        y1 = max(cy - cut_h // 2, 0)
        x2 = min(cx + cut_w // 2, W)
        y2 = min(cy + cut_h // 2, H)

        # Apply cutmix
        images[:, :, y1:y2, x1:x2] = images[index, :, y1:y2, x1:x2]

        # Adjust lambda based on actual area
        lam = 1 - ((x2 - x1) * (y2 - y1) / (W * H))

        # Mix labels
        if labels.dim() == 1:
            num_classes = getattr(self.config, 'num_classes', 10)
            labels = torch.nn.functional.one_hot(labels, num_classes).float()

        mixed_labels = lam * labels + (1 - lam) * labels[index]

        return images, mixed_labels
```

Usage:

```python
@dataclass
class MyConfig(BaseTrainerConfig, SetupConfigMixin, AugmentationConfigMixin):
    num_classes: int = 10
    augmentation_enabled: bool = True


class MyTrainer(BaseTrainer, SetupMixin, AugmentationMixin):
    config: MyConfig

    def training_step(self, batch, batch_idx):
        images, labels = batch
        images = images.to(self.device)
        labels = labels.to(self.device)

        # Apply augmentation
        images, labels = self.augment_batch(images, labels)

        # Forward pass with soft labels
        logits = self.model(images)

        # Cross-entropy with soft labels
        if labels.dim() == 2:
            loss = -(labels * torch.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
        else:
            loss = F.cross_entropy(logits, labels)

        self.backward(loss)
        if not self.is_accumulating_gradients:
            self.step(self.optimizer)

        return {"loss": loss}
```

### Example: Gradient Penalty Mixin

```python
from dataclasses import dataclass
import torch


@dataclass
class GradientPenaltyConfigMixin:
    """Configuration for gradient penalty."""
    gradient_penalty_weight: float = 10.0
    gradient_penalty_mode: str = "two_sided"  # "one_sided" or "two_sided"


class GradientPenaltyMixin:
    """Adds gradient penalty computation for GANs."""

    config: GradientPenaltyConfigMixin

    def compute_gradient_penalty(
        self,
        discriminator: torch.nn.Module,
        real_samples: torch.Tensor,
        fake_samples: torch.Tensor,
    ) -> torch.Tensor:
        """Compute gradient penalty for WGAN-GP.

        Args:
            discriminator: Discriminator model
            real_samples: Real data samples
            fake_samples: Generated fake samples

        Returns:
            Gradient penalty loss term
        """
        batch_size = real_samples.size(0)
        device = real_samples.device

        # Random interpolation
        alpha = torch.rand(batch_size, 1, 1, 1, device=device)
        interpolates = alpha * real_samples + (1 - alpha) * fake_samples
        interpolates.requires_grad_(True)

        # Get discriminator output
        d_interpolates = discriminator(interpolates)

        # Compute gradients
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(d_interpolates),
            create_graph=True,
            retain_graph=True,
        )[0]

        # Compute penalty
        gradients = gradients.view(batch_size, -1)
        gradient_norm = gradients.norm(2, dim=1)

        if self.config.gradient_penalty_mode == "one_sided":
            penalty = torch.relu(gradient_norm - 1).pow(2).mean()
        else:
            penalty = (gradient_norm - 1).pow(2).mean()

        return self.config.gradient_penalty_weight * penalty
```

---

## Part 3: Custom Configurations

### Composable Configuration Classes

```python
from dataclasses import dataclass, field
from typing import Literal
import torch


@dataclass
class ModelConfigMixin:
    """Configuration for model architecture."""
    model_type: Literal["small", "base", "large"] = "base"
    hidden_size: int = 768
    num_layers: int = 12
    num_heads: int = 12
    dropout: float = 0.1

    def __post_init__(self):
        """Auto-configure based on model_type."""
        if self.model_type == "small":
            self.hidden_size = 512
            self.num_layers = 6
            self.num_heads = 8
        elif self.model_type == "large":
            self.hidden_size = 1024
            self.num_layers = 24
            self.num_heads = 16


@dataclass
class OptimizerConfigMixin:
    """Configuration for optimizer."""
    optimizer_type: Literal["adam", "adamw", "sgd"] = "adamw"
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.999)
    momentum: float = 0.9  # For SGD

    def create_optimizer(self, params) -> torch.optim.Optimizer:
        """Factory method to create optimizer."""
        if self.optimizer_type == "adam":
            return torch.optim.Adam(
                params,
                lr=self.learning_rate,
                betas=self.betas,
            )
        elif self.optimizer_type == "adamw":
            return torch.optim.AdamW(
                params,
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                betas=self.betas,
            )
        elif self.optimizer_type == "sgd":
            return torch.optim.SGD(
                params,
                lr=self.learning_rate,
                momentum=self.momentum,
                weight_decay=self.weight_decay,
            )


@dataclass
class SchedulerConfigMixin:
    """Configuration for learning rate scheduler."""
    scheduler_type: Literal["cosine", "linear", "constant"] = "cosine"
    warmup_steps: int = 1000
    min_lr_ratio: float = 0.1

    def create_scheduler(
        self,
        optimizer: torch.optim.Optimizer,
        total_steps: int,
    ) -> torch.optim.lr_scheduler._LRScheduler:
        """Factory method to create scheduler."""
        if self.scheduler_type == "constant":
            return torch.optim.lr_scheduler.ConstantLR(
                optimizer,
                factor=1.0,
            )
        elif self.scheduler_type == "linear":
            return torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1.0,
                end_factor=self.min_lr_ratio,
                total_iters=total_steps,
            )
        elif self.scheduler_type == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=total_steps - self.warmup_steps,
                eta_min=optimizer.defaults['lr'] * self.min_lr_ratio,
            )
```

### Using Composed Configurations

```python
from dream_trainer import BaseTrainerConfig
from dream_trainer.trainer.mixins import SetupConfigMixin


@dataclass
class MyFullConfig(
    BaseTrainerConfig,
    SetupConfigMixin,
    ModelConfigMixin,
    OptimizerConfigMixin,
    SchedulerConfigMixin,
):
    """Complete configuration with all mixins."""
    # Additional fields
    dataset_path: str = "./data"
    num_workers: int = 4

    def validate(self):
        """Custom validation logic."""
        assert self.hidden_size % self.num_heads == 0, \
            "hidden_size must be divisible by num_heads"
        assert self.learning_rate > 0, "learning_rate must be positive"


# Usage
config = MyFullConfig(
    model_type="large",  # Auto-configures hidden_size, num_layers, num_heads
    optimizer_type="adamw",
    scheduler_type="cosine",
    training_parameters=TrainingParameters(n_epochs=10),
)

# Factory methods work automatically
optimizer = config.create_optimizer(model.parameters())
scheduler = config.create_scheduler(optimizer, total_steps=10000)
```

---

## Part 4: Best Practices

### 1. Use Type Hints

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dream_trainer import BaseTrainer


class MyMixin:
    # Enable IDE support without circular imports
    trainer: "BaseTrainer"
    config: "MyConfig"
```

### 2. Document Configuration Options

```python
@dataclass
class WellDocumentedConfig:
    """Configuration for XYZ feature.

    This config controls how XYZ behaves during training.

    Attributes:
        feature_enabled: Whether to enable the XYZ feature.
            Default: True.
        feature_strength: How strongly to apply XYZ.
            Range: 0.0 to 1.0. Default: 0.5.
        feature_schedule: When to apply XYZ during training.
            Options: "always", "warmup_only", "decay".
            Default: "always".

    Example:
        >>> config = WellDocumentedConfig(
        ...     feature_enabled=True,
        ...     feature_strength=0.7,
        ... )
    """
    feature_enabled: bool = True
    feature_strength: float = 0.5
    feature_schedule: str = "always"
```

### 3. Support Checkpointing

```python
class StatefulCallback(Callback):
    """Callback with checkpointable state."""

    def __init__(self):
        super().__init__()
        self.counter = 0
        self.history = []

    def post_train_step(self, output, batch_idx):
        self.counter += 1
        self.history.append(output['loss'])

    def state_dict(self) -> dict:
        """Return all state that should be checkpointed."""
        return {
            "counter": self.counter,
            "history": self.history,
        }

    def load_state_dict(self, state_dict: dict):
        """Restore state from checkpoint."""
        self.counter = state_dict["counter"]
        self.history = state_dict["history"]
```

### 4. Handle Distributed Training

```python
class DistributedAwareCallback(Callback):
    """Callback that works correctly in distributed training."""

    def post_train_step(self, output, batch_idx):
        # Only log on rank 0
        if self.trainer.world.rank == 0:
            print(f"Loss: {output['loss']}")

        # Synchronize across ranks when needed
        if self.trainer.world.world_size > 1:
            # Gather values from all ranks
            gathered = self.trainer.world.all_gather(output['loss'])
            mean_loss = gathered.mean()
```

### 5. Compose, Don't Inherit Deeply

```python
# Good: Flat composition
class MyTrainer(BaseTrainer, SetupMixin, MetricMixin, LoggerMixin):
    pass


# Avoid: Deep inheritance
class BaseTrainer: pass
class TrainerWithSetup(BaseTrainer): pass
class TrainerWithMetrics(TrainerWithSetup): pass
class TrainerWithLogging(TrainerWithMetrics): pass  # Hard to customize!
```

---

## Summary

You've learned how to extend Dream Trainer with:

1. **Callbacks**: Hook into training lifecycle for side effects
2. **Mixins**: Add reusable functionality to trainers
3. **Configurations**: Create composable, type-safe config classes
4. **Best Practices**: Write maintainable, distributed-aware components

## Next Steps

- [Production Setup](production.md): Deploy your custom components
- [Callbacks Reference](../callbacks.md): See all built-in callbacks
- [API Reference](../api/index.md): Detailed API documentation
