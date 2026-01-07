# Advanced Patterns Examples

This page covers advanced training patterns with Dream Trainer, including EMA (Exponential Moving Average), knowledge distillation, curriculum learning, and custom training loops.

---

## Exponential Moving Average (EMA)

### EMA Model for Stable Evaluation

```python
"""EMA model training pattern."""
from dataclasses import dataclass
import torch
import torch.nn as nn
from copy import deepcopy

from dream_trainer import DreamTrainer, DreamTrainerConfig
from dream_trainer.callbacks import Callback, EMACallback
from dream_trainer.trainer.mixins import SetupMixin


@dataclass
class EMAConfig(DreamTrainerConfig):
    # EMA settings
    ema_decay: float = 0.9999
    ema_update_every: int = 1
    ema_warmup_steps: int = 2000

    # Model
    hidden_size: int = 768


class ManualEMATrainer(DreamTrainer):
    """Trainer with manual EMA implementation."""

    config: EMAConfig

    def configure_models(self):
        self.model = MyModel(self.config)
        # EMA model is created in post_setup

    def post_setup(self):
        """Create EMA model after main model is set up."""
        super().post_setup()

        # Create EMA model as a copy
        self.ema_model = deepcopy(self.model)
        self.ema_model.requires_grad_(False)
        self.ema_model.eval()

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device)

        # Forward with training model
        logits = self.model(x)
        loss = F.cross_entropy(logits, y)

        self.backward(loss)

        if not self.is_accumulating_gradients:
            self.step(self.optimizer)

            # Update EMA model
            if self.global_step % self.config.ema_update_every == 0:
                self._update_ema()

        return {"loss": loss}

    @torch.no_grad()
    def _update_ema(self):
        """Update EMA model parameters."""
        # Linear warmup of EMA decay
        if self.global_step < self.config.ema_warmup_steps:
            decay = min(
                self.config.ema_decay,
                (1 + self.global_step) / (10 + self.global_step)
            )
        else:
            decay = self.config.ema_decay

        # Update parameters
        for ema_param, param in zip(
            self.ema_model.parameters(),
            self.model.parameters()
        ):
            ema_param.data.lerp_(param.data, 1 - decay)

        # Update buffers (e.g., batch norm running stats)
        for ema_buf, buf in zip(
            self.ema_model.buffers(),
            self.model.buffers()
        ):
            ema_buf.data.copy_(buf.data)

    def validation_step(self, batch, batch_idx):
        """Validate using EMA model."""
        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device)

        # Use EMA model for validation
        logits = self.ema_model(x)
        loss = F.cross_entropy(logits, y)

        return {"val_loss": loss}

    def state_dict(self):
        """Include EMA model in checkpoint."""
        state = super().state_dict()
        state["ema_model"] = self.ema_model.state_dict()
        return state

    def load_state_dict(self, state_dict):
        """Restore EMA model from checkpoint."""
        super().load_state_dict(state_dict)
        if "ema_model" in state_dict:
            self.ema_model.load_state_dict(state_dict["ema_model"])


# Alternative: Use the built-in EMACallback
config = EMAConfig(
    callbacks=CallbackCollection([
        EMACallback(
            decay=0.9999,
            update_every=1,
            warmup_steps=2000,
        ),
    ]),
)
```

---

## Knowledge Distillation

### Teacher-Student Training

```python
"""Knowledge distillation example."""
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

from dream_trainer import DreamTrainer, DreamTrainerConfig


@dataclass
class DistillationConfig(DreamTrainerConfig):
    # Teacher model
    teacher_checkpoint: str = "./checkpoints/teacher.pt"
    teacher_hidden_size: int = 1024
    teacher_num_layers: int = 24

    # Student model
    student_hidden_size: int = 512
    student_num_layers: int = 12

    # Distillation
    temperature: float = 4.0
    alpha: float = 0.5  # Weight for distillation loss


class DistillationTrainer(DreamTrainer):
    config: DistillationConfig

    def configure_models(self):
        """Create student model (teacher is loaded separately)."""
        # Student model (trainable)
        self.model = StudentModel(
            hidden_size=self.config.student_hidden_size,
            num_layers=self.config.student_num_layers,
        )

        # Teacher model (frozen)
        self.teacher = TeacherModel(
            hidden_size=self.config.teacher_hidden_size,
            num_layers=self.config.teacher_num_layers,
        )

    def post_setup(self):
        """Load teacher checkpoint and freeze."""
        super().post_setup()

        # Load teacher weights
        checkpoint = torch.load(
            self.config.teacher_checkpoint,
            map_location=self.device,
        )
        self.teacher.load_state_dict(checkpoint["model"])

        # Freeze teacher
        self.teacher.requires_grad_(False)
        self.teacher.eval()

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device)

        # Student forward pass
        student_logits = self.model(x)

        # Teacher forward pass (no gradients)
        with torch.no_grad():
            teacher_logits = self.teacher(x)

        # Hard label loss (cross-entropy)
        hard_loss = F.cross_entropy(student_logits, y)

        # Soft label loss (KL divergence with temperature)
        T = self.config.temperature
        soft_loss = F.kl_div(
            F.log_softmax(student_logits / T, dim=-1),
            F.softmax(teacher_logits / T, dim=-1),
            reduction='batchmean',
        ) * (T ** 2)

        # Combined loss
        alpha = self.config.alpha
        loss = alpha * soft_loss + (1 - alpha) * hard_loss

        self.backward(loss)

        if not self.is_accumulating_gradients:
            grad_norm = self.step(self.optimizer)
            return {
                "loss": loss,
                "hard_loss": hard_loss,
                "soft_loss": soft_loss,
                "grad_norm": grad_norm,
            }

        return {"loss": loss}


class FeatureDistillationTrainer(DreamTrainer):
    """Distillation with intermediate feature matching."""

    config: DistillationConfig

    def configure_models(self):
        self.model = StudentModel(self.config)
        self.teacher = TeacherModel(self.config)

        # Feature projection layers (student -> teacher dimension)
        self.feature_projectors = nn.ModuleList([
            nn.Linear(
                self.config.student_hidden_size,
                self.config.teacher_hidden_size,
            )
            for _ in range(self.config.student_num_layers)
        ])

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device)

        # Get intermediate features
        student_features, student_logits = self.model(x, return_features=True)

        with torch.no_grad():
            teacher_features, teacher_logits = self.teacher(x, return_features=True)

        # Output distillation loss
        T = self.config.temperature
        output_loss = F.kl_div(
            F.log_softmax(student_logits / T, dim=-1),
            F.softmax(teacher_logits / T, dim=-1),
            reduction='batchmean',
        ) * (T ** 2)

        # Feature matching loss
        feature_loss = 0.0
        # Map student layers to teacher layers
        layer_mapping = self._get_layer_mapping()

        for student_idx, teacher_idx in layer_mapping:
            # Project student features
            projected = self.feature_projectors[student_idx](
                student_features[student_idx]
            )
            # MSE loss with teacher features
            feature_loss += F.mse_loss(
                projected,
                teacher_features[teacher_idx],
            )

        feature_loss /= len(layer_mapping)

        # Combined loss
        loss = output_loss + 0.5 * feature_loss

        self.backward(loss)

        if not self.is_accumulating_gradients:
            self.step(self.optimizer)

        return {"loss": loss, "output_loss": output_loss, "feature_loss": feature_loss}

    def _get_layer_mapping(self):
        """Map student layers to teacher layers."""
        # Simple linear mapping
        student_layers = self.config.student_num_layers
        teacher_layers = self.config.teacher_num_layers

        mapping = []
        for i in range(student_layers):
            teacher_idx = int(i * teacher_layers / student_layers)
            mapping.append((i, teacher_idx))
        return mapping
```

---

## Curriculum Learning

### Adaptive Curriculum Training

```python
"""Curriculum learning example."""
from dataclasses import dataclass
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler
import numpy as np

from dream_trainer import DreamTrainer, DreamTrainerConfig
from dream_trainer.callbacks import Callback


@dataclass
class CurriculumConfig(DreamTrainerConfig):
    # Curriculum settings
    curriculum_type: str = "linear"  # "linear", "sqrt", "step"
    warmup_epochs: int = 5
    difficulty_metric: str = "loss"  # or "length", "complexity"


class DifficultySampler(Sampler):
    """Sampler that orders by difficulty."""

    def __init__(self, dataset, difficulties, curriculum_rate=1.0):
        self.dataset = dataset
        self.difficulties = difficulties
        self.curriculum_rate = curriculum_rate

    def __iter__(self):
        # Sort indices by difficulty
        sorted_indices = np.argsort(self.difficulties)

        # Select subset based on curriculum rate
        n_samples = int(len(sorted_indices) * self.curriculum_rate)
        selected = sorted_indices[:n_samples]

        # Shuffle within selected samples
        np.random.shuffle(selected)

        return iter(selected.tolist())

    def __len__(self):
        return int(len(self.dataset) * self.curriculum_rate)


class CurriculumCallback(Callback):
    """Callback that updates curriculum difficulty."""

    def __init__(self, config: CurriculumConfig):
        super().__init__()
        self.config = config

    def post_train_epoch(self):
        """Update curriculum rate after each epoch."""
        epoch = self.trainer.current_epoch
        total_epochs = self.config.training_parameters.n_epochs
        warmup = self.config.warmup_epochs

        if epoch < warmup:
            # Linear warmup of curriculum
            rate = (epoch + 1) / warmup
        else:
            # Full dataset after warmup
            rate = 1.0

        # Apply curriculum type
        if self.config.curriculum_type == "sqrt":
            rate = np.sqrt(rate)
        elif self.config.curriculum_type == "step":
            rate = 1.0 if epoch >= warmup else 0.5

        # Update sampler
        if hasattr(self.trainer, 'curriculum_sampler'):
            self.trainer.curriculum_sampler.curriculum_rate = rate
            print(f"Curriculum rate: {rate:.2%}")


class CurriculumTrainer(DreamTrainer):
    config: CurriculumConfig

    def configure_dataloaders(self):
        train_dataset = MyDataset("train")
        val_dataset = MyDataset("val")

        # Compute difficulties (could be pre-computed)
        difficulties = self._compute_difficulties(train_dataset)

        # Create curriculum sampler
        self.curriculum_sampler = DifficultySampler(
            train_dataset,
            difficulties,
            curriculum_rate=0.1,  # Start with 10% easiest
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.training_parameters.train_batch_size,
            sampler=self.curriculum_sampler,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.training_parameters.train_batch_size * 2,
            shuffle=False,
        )

        return train_loader, val_loader

    def _compute_difficulties(self, dataset):
        """Compute difficulty scores for each sample."""
        if self.config.difficulty_metric == "length":
            # Sequence length as proxy for difficulty
            return np.array([len(dataset[i]['text']) for i in range(len(dataset))])

        elif self.config.difficulty_metric == "loss":
            # Use loss from a simple model as difficulty
            # (Pre-compute or use running estimates)
            return np.random.rand(len(dataset))  # Placeholder

        else:
            return np.zeros(len(dataset))
```

---

## Multi-Task Learning

### Shared Backbone with Task-Specific Heads

```python
"""Multi-task learning example."""
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

from dream_trainer import DreamTrainer, DreamTrainerConfig


@dataclass
class MultiTaskConfig(DreamTrainerConfig):
    # Shared backbone
    backbone_type: str = "transformer"
    hidden_size: int = 768
    num_layers: int = 12

    # Tasks
    tasks: list = None  # e.g., ["classification", "regression", "generation"]
    task_weights: dict = None  # e.g., {"classification": 1.0, "regression": 0.5}

    def __post_init__(self):
        if self.tasks is None:
            self.tasks = ["classification", "regression"]
        if self.task_weights is None:
            self.task_weights = {task: 1.0 for task in self.tasks}


class MultiTaskModel(nn.Module):
    """Model with shared backbone and task-specific heads."""

    def __init__(self, config: MultiTaskConfig):
        super().__init__()
        self.config = config

        # Shared backbone
        self.backbone = TransformerBackbone(config)

        # Task-specific heads
        self.heads = nn.ModuleDict()

        for task in config.tasks:
            if task == "classification":
                self.heads[task] = nn.Linear(config.hidden_size, config.num_classes)
            elif task == "regression":
                self.heads[task] = nn.Linear(config.hidden_size, 1)
            elif task == "generation":
                self.heads[task] = nn.Linear(config.hidden_size, config.vocab_size)

    def forward(self, x, task=None):
        # Shared representation
        features = self.backbone(x)

        if task is not None:
            # Single task output
            return self.heads[task](features)
        else:
            # All task outputs
            return {
                task: head(features)
                for task, head in self.heads.items()
            }


class MultiTaskTrainer(DreamTrainer):
    config: MultiTaskConfig

    def configure_models(self):
        self.model = MultiTaskModel(self.config)

    def training_step(self, batch, batch_idx):
        losses = {}
        total_loss = 0.0

        # Compute loss for each task
        for task in self.config.tasks:
            if task not in batch:
                continue

            task_data = batch[task]
            x = task_data['input'].to(self.device)
            y = task_data['target'].to(self.device)

            # Forward
            logits = self.model(x, task=task)

            # Task-specific loss
            if task == "classification":
                loss = F.cross_entropy(logits, y)
            elif task == "regression":
                loss = F.mse_loss(logits.squeeze(), y)
            elif task == "generation":
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

            losses[f"{task}_loss"] = loss
            total_loss += self.config.task_weights[task] * loss

        self.backward(total_loss)

        if not self.is_accumulating_gradients:
            grad_norm = self.step(self.optimizer)
            losses["grad_norm"] = grad_norm

        losses["loss"] = total_loss
        return losses


class GradientBalancingCallback(Callback):
    """Balance gradients across tasks using GradNorm or similar."""

    def __init__(self, alpha: float = 1.5):
        super().__init__()
        self.alpha = alpha
        self.initial_losses = {}
        self.task_weights = None

    def post_train_step(self, output, batch_idx):
        # Track initial losses
        if not self.initial_losses:
            for key, value in output.items():
                if key.endswith("_loss"):
                    task = key.replace("_loss", "")
                    self.initial_losses[task] = value.item()
            return

        # Update task weights based on relative loss ratios
        current_ratios = {}
        for key, value in output.items():
            if key.endswith("_loss"):
                task = key.replace("_loss", "")
                if task in self.initial_losses:
                    current_ratios[task] = value.item() / self.initial_losses[task]

        if current_ratios:
            # Normalize
            mean_ratio = sum(current_ratios.values()) / len(current_ratios)
            for task, ratio in current_ratios.items():
                new_weight = (ratio / mean_ratio) ** self.alpha
                self.trainer.config.task_weights[task] = new_weight
```

---

## Contrastive Learning

### SimCLR-Style Self-Supervised Learning

```python
"""Self-supervised contrastive learning."""
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

from dream_trainer import DreamTrainer, DreamTrainerConfig


@dataclass
class ContrastiveConfig(DreamTrainerConfig):
    hidden_size: int = 2048
    projection_size: int = 128
    temperature: float = 0.5


class ProjectionHead(nn.Module):
    """MLP projection head for contrastive learning."""

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class SimCLRModel(nn.Module):
    def __init__(self, config: ContrastiveConfig):
        super().__init__()
        # Encoder (e.g., ResNet)
        self.encoder = ResNetEncoder(config)

        # Projection head
        self.projection = ProjectionHead(
            config.hidden_size,
            config.hidden_size,
            config.projection_size,
        )

    def forward(self, x):
        h = self.encoder(x)
        z = self.projection(h)
        return F.normalize(z, dim=-1)


def nt_xent_loss(z1, z2, temperature):
    """Normalized temperature-scaled cross entropy loss."""
    batch_size = z1.size(0)
    device = z1.device

    # Concatenate representations
    z = torch.cat([z1, z2], dim=0)  # [2B, D]

    # Compute similarity matrix
    sim = z @ z.T / temperature  # [2B, 2B]

    # Mask out self-similarity
    mask = torch.eye(2 * batch_size, device=device).bool()
    sim.masked_fill_(mask, float('-inf'))

    # Create labels (positive pairs)
    labels = torch.cat([
        torch.arange(batch_size, 2 * batch_size),
        torch.arange(batch_size),
    ], dim=0).to(device)

    # Cross-entropy loss
    loss = F.cross_entropy(sim, labels)
    return loss


class SimCLRTrainer(DreamTrainer):
    config: ContrastiveConfig

    def configure_models(self):
        self.model = SimCLRModel(self.config)

    def training_step(self, batch, batch_idx):
        # Batch contains two augmented views
        x1, x2 = batch  # Two augmented views
        x1 = x1.to(self.device)
        x2 = x2.to(self.device)

        # Get representations
        z1 = self.model(x1)
        z2 = self.model(x2)

        # Contrastive loss
        loss = nt_xent_loss(z1, z2, self.config.temperature)

        self.backward(loss)

        if not self.is_accumulating_gradients:
            grad_norm = self.step(self.optimizer)
            return {"loss": loss, "grad_norm": grad_norm}

        return {"loss": loss}

    def configure_dataloaders(self):
        # Dataset returns two augmented views per sample
        train_dataset = ContrastiveDataset(
            data_path=self.config.data_path,
            augmentation=SimCLRAugmentation(),
        )
        # ...
```

---

## Gradient Checkpointing with Custom Policies

### Fine-Grained Activation Checkpointing

```python
"""Custom gradient checkpointing patterns."""
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from dream_trainer import DreamTrainer, DreamTrainerConfig


@dataclass
class CheckpointingConfig(DreamTrainerConfig):
    # Checkpointing policy
    checkpoint_every_n_layers: int = 2
    checkpoint_attention: bool = True
    checkpoint_mlp: bool = False


class CheckpointedTransformerBlock(nn.Module):
    """Transformer block with configurable checkpointing."""

    def __init__(self, config, checkpoint_attention=True, checkpoint_mlp=False):
        super().__init__()
        self.attention = Attention(config)
        self.mlp = MLP(config)
        self.norm1 = nn.LayerNorm(config.hidden_size)
        self.norm2 = nn.LayerNorm(config.hidden_size)

        self.checkpoint_attention = checkpoint_attention
        self.checkpoint_mlp = checkpoint_mlp

    def forward(self, x):
        # Attention with optional checkpointing
        if self.checkpoint_attention and self.training:
            attn_out = checkpoint(
                self._attention_forward,
                self.norm1(x),
                use_reentrant=False,
            )
        else:
            attn_out = self._attention_forward(self.norm1(x))

        x = x + attn_out

        # MLP with optional checkpointing
        if self.checkpoint_mlp and self.training:
            mlp_out = checkpoint(
                self._mlp_forward,
                self.norm2(x),
                use_reentrant=False,
            )
        else:
            mlp_out = self._mlp_forward(self.norm2(x))

        x = x + mlp_out
        return x

    def _attention_forward(self, x):
        return self.attention(x)

    def _mlp_forward(self, x):
        return self.mlp(x)


class SelectiveCheckpointingTrainer(DreamTrainer):
    config: CheckpointingConfig

    def apply_activation_checkpointing(self):
        """Apply selective activation checkpointing."""
        for i, layer in enumerate(self.model.layers):
            # Checkpoint every N layers
            should_checkpoint = (i % self.config.checkpoint_every_n_layers == 0)

            if hasattr(layer, 'checkpoint_attention'):
                layer.checkpoint_attention = (
                    should_checkpoint and self.config.checkpoint_attention
                )
                layer.checkpoint_mlp = (
                    should_checkpoint and self.config.checkpoint_mlp
                )
```

---

## Summary

This page covered advanced training patterns:

1. **EMA**: Maintain stable model weights for evaluation
2. **Knowledge Distillation**: Transfer knowledge from large to small models
3. **Curriculum Learning**: Train on progressively harder examples
4. **Multi-Task Learning**: Share representations across tasks
5. **Contrastive Learning**: Self-supervised representation learning
6. **Gradient Checkpointing**: Memory-efficient training with selective policies

These patterns can be combined with Dream Trainer's parallelism support for large-scale training.

## Next Steps

- [Performance Guide](../performance.md): Optimize training throughput
- [Parallelism Guide](../parallelism.md): Scale advanced patterns
- [Callbacks](../callbacks.md): Implement patterns as callbacks
