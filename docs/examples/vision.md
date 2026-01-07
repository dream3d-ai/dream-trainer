# Vision Model Examples

This page provides complete examples for training vision models with Dream Trainer, including image classification, semantic segmentation, and diffusion models.

---

## Image Classification

### ResNet on ImageNet

A complete example training ResNet-50 on ImageNet with distributed data parallelism.

```python
"""ImageNet classification with ResNet-50."""
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torch.distributed._composable.fsdp import fully_shard

from dream_trainer import DreamTrainer, DreamTrainerConfig
from dream_trainer.trainer.mixins import (
    SetupMixin, SetupConfigMixin,
    EvalMetricMixin, EvalMetricConfigMixin,
)
from dream_trainer.configs import DeviceParameters, TrainingParameters
from dream_trainer.callbacks import (
    CheckpointCallback,
    LoggerCallback,
    ProgressBar,
    CallbackCollection,
)
import torchmetrics


@dataclass
class ImageNetConfig(DreamTrainerConfig, SetupConfigMixin, EvalMetricConfigMixin):
    # Model
    model_name: str = "resnet50"
    num_classes: int = 1000
    pretrained: bool = False

    # Data
    data_dir: str = "/data/imagenet"
    image_size: int = 224
    num_workers: int = 8

    # Training
    learning_rate: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 1e-4


class ImageNetTrainer(DreamTrainer, EvalMetricMixin):
    config: ImageNetConfig

    def configure_models(self):
        """Create ResNet model."""
        model_fn = getattr(models, self.config.model_name)
        self.model = model_fn(
            pretrained=self.config.pretrained,
            num_classes=self.config.num_classes,
        )

    def init_weights(self):
        """Initialize weights (skip if pretrained)."""
        if not self.config.pretrained:
            for m in self.model.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)

    def apply_fully_shard(self, fsdp_config):
        """Apply FSDP sharding."""
        # Shard each residual block
        for layer in [self.model.layer1, self.model.layer2,
                      self.model.layer3, self.model.layer4]:
            for block in layer:
                fully_shard(block, **fsdp_config)
        fully_shard(self.model, **fsdp_config)

    def configure_optimizers(self):
        """Configure SGD with momentum."""
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.config.learning_rate,
            momentum=self.config.momentum,
            weight_decay=self.config.weight_decay,
        )

    def configure_schedulers(self):
        """Cosine annealing scheduler."""
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.training_parameters.n_epochs,
        )

    def configure_dataloaders(self):
        """Create ImageNet dataloaders."""
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(self.config.image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.4, 0.4, 0.4),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(self.config.image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        train_dataset = datasets.ImageFolder(
            f"{self.config.data_dir}/train",
            transform=train_transform,
        )
        val_dataset = datasets.ImageFolder(
            f"{self.config.data_dir}/val",
            transform=val_transform,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.training_parameters.train_batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True,
            drop_last=True,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.training_parameters.train_batch_size * 2,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True,
        )

        return train_loader, val_loader

    def configure_metrics(self):
        """Configure accuracy metrics."""
        self.train_accuracy = torchmetrics.Accuracy(
            task="multiclass",
            num_classes=self.config.num_classes,
        )
        self.val_accuracy = torchmetrics.Accuracy(
            task="multiclass",
            num_classes=self.config.num_classes,
        )
        self.val_top5 = torchmetrics.Accuracy(
            task="multiclass",
            num_classes=self.config.num_classes,
            top_k=5,
        )

    def training_step(self, batch, batch_idx):
        images, labels = batch
        images = images.to(self.device, non_blocking=True)
        labels = labels.to(self.device, non_blocking=True)

        # Forward
        logits = self.model(images)
        loss = F.cross_entropy(logits, labels)

        # Backward
        self.backward(loss)

        if not self.is_accumulating_gradients:
            grad_norm = self.step(self.optimizer)
            self.train_accuracy(logits.argmax(dim=-1), labels)
            return {"loss": loss, "grad_norm": grad_norm}

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        images = images.to(self.device, non_blocking=True)
        labels = labels.to(self.device, non_blocking=True)

        logits = self.model(images)
        loss = F.cross_entropy(logits, labels)

        # Update metrics
        preds = logits.argmax(dim=-1)
        self.val_accuracy(preds, labels)
        self.val_top5(logits, labels)

        return {"val_loss": loss}


if __name__ == "__main__":
    config = ImageNetConfig(
        device_parameters=DeviceParameters.FSDP(
            dp_shard=8,
            param_dtype=torch.bfloat16,
        ),
        training_parameters=TrainingParameters(
            n_epochs=90,
            train_batch_size=64,
            gradient_clip_val=1.0,
            val_frequency=1.0,
        ),
        callbacks=CallbackCollection([
            ProgressBar(),
            LoggerCallback(log_every_n_train_batches=100),
            CheckpointCallback(),
        ]),
    )

    trainer = ImageNetTrainer(config)
    trainer.fit()
```

---

## Vision Transformer (ViT)

### ViT on ImageNet

```python
"""Vision Transformer training example."""
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    parallelize_module,
)

from dream_trainer import DreamTrainer, DreamTrainerConfig
from dream_trainer.trainer.mixins import SetupMixin, SetupConfigMixin
from dream_trainer.configs import DeviceParameters, TrainingParameters


@dataclass
class ViTConfig(DreamTrainerConfig, SetupConfigMixin):
    # Model architecture
    image_size: int = 224
    patch_size: int = 16
    hidden_size: int = 768
    num_layers: int = 12
    num_heads: int = 12
    mlp_ratio: float = 4.0
    num_classes: int = 1000

    # Training
    learning_rate: float = 3e-4
    weight_decay: float = 0.3
    warmup_epochs: int = 5


class PatchEmbedding(nn.Module):
    """Convert image to patch embeddings."""

    def __init__(self, image_size, patch_size, hidden_size):
        super().__init__()
        self.num_patches = (image_size // patch_size) ** 2
        self.proj = nn.Conv2d(
            3, hidden_size,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x):
        x = self.proj(x)  # [B, H, P, P]
        x = x.flatten(2).transpose(1, 2)  # [B, N, H]
        return x


class TransformerBlock(nn.Module):
    """Transformer encoder block."""

    def __init__(self, hidden_size, num_heads, mlp_ratio):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.attn = nn.MultiheadAttention(
            hidden_size, num_heads,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, int(hidden_size * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(hidden_size * mlp_ratio), hidden_size),
        )

    def forward(self, x):
        # Self-attention with residual
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + attn_out

        # MLP with residual
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    """Vision Transformer model."""

    def __init__(self, config: ViTConfig):
        super().__init__()
        self.patch_embed = PatchEmbedding(
            config.image_size, config.patch_size, config.hidden_size
        )

        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, config.hidden_size)
        )

        self.blocks = nn.ModuleList([
            TransformerBlock(
                config.hidden_size,
                config.num_heads,
                config.mlp_ratio,
            )
            for _ in range(config.num_layers)
        ])

        self.norm = nn.LayerNorm(config.hidden_size)
        self.head = nn.Linear(config.hidden_size, config.num_classes)

    def forward(self, x):
        B = x.size(0)

        # Patch embedding
        x = self.patch_embed(x)

        # Add CLS token and positional embedding
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        # Classification head (use CLS token)
        x = self.norm(x[:, 0])
        x = self.head(x)
        return x


class ViTTrainer(DreamTrainer):
    config: ViTConfig

    def configure_models(self):
        self.model = VisionTransformer(self.config)

    def apply_tensor_parallel(self, tp_mesh):
        """Apply tensor parallelism to transformer blocks."""
        for block in self.model.blocks:
            tp_plan = {
                "attn": ColwiseParallel(),
                "mlp.0": ColwiseParallel(),
                "mlp.2": RowwiseParallel(),
            }
            parallelize_module(block, tp_mesh, tp_plan)

    def apply_fully_shard(self, fsdp_config):
        from torch.distributed._composable.fsdp import fully_shard
        for block in self.model.blocks:
            fully_shard(block, **fsdp_config)
        fully_shard(self.model, **fsdp_config)

    # ... implement remaining methods similar to ImageNetTrainer
```

---

## Semantic Segmentation

### U-Net for Medical Image Segmentation

```python
"""U-Net segmentation example."""
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

from dream_trainer import DreamTrainer, DreamTrainerConfig
from dream_trainer.trainer.mixins import SetupMixin, SetupConfigMixin, EvalMetricMixin
import torchmetrics


@dataclass
class UNetConfig(DreamTrainerConfig, SetupConfigMixin):
    # Model
    in_channels: int = 1
    num_classes: int = 2
    base_channels: int = 64

    # Data
    image_size: int = 256


class DoubleConv(nn.Module):
    """Double convolution block."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """U-Net architecture for segmentation."""

    def __init__(self, config: UNetConfig):
        super().__init__()
        c = config.base_channels

        # Encoder
        self.enc1 = DoubleConv(config.in_channels, c)
        self.enc2 = DoubleConv(c, c * 2)
        self.enc3 = DoubleConv(c * 2, c * 4)
        self.enc4 = DoubleConv(c * 4, c * 8)

        # Bottleneck
        self.bottleneck = DoubleConv(c * 8, c * 16)

        # Decoder
        self.up4 = nn.ConvTranspose2d(c * 16, c * 8, 2, stride=2)
        self.dec4 = DoubleConv(c * 16, c * 8)
        self.up3 = nn.ConvTranspose2d(c * 8, c * 4, 2, stride=2)
        self.dec3 = DoubleConv(c * 8, c * 4)
        self.up2 = nn.ConvTranspose2d(c * 4, c * 2, 2, stride=2)
        self.dec2 = DoubleConv(c * 4, c * 2)
        self.up1 = nn.ConvTranspose2d(c * 2, c, 2, stride=2)
        self.dec1 = DoubleConv(c * 2, c)

        # Output
        self.out = nn.Conv2d(c, config.num_classes, 1)

        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Bottleneck
        b = self.bottleneck(self.pool(e4))

        # Decoder with skip connections
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.out(d1)


class UNetTrainer(DreamTrainer, EvalMetricMixin):
    config: UNetConfig

    def configure_models(self):
        self.model = UNet(self.config)

    def configure_metrics(self):
        self.dice_score = torchmetrics.Dice(
            num_classes=self.config.num_classes,
            average="macro",
        )
        self.iou = torchmetrics.JaccardIndex(
            task="multiclass",
            num_classes=self.config.num_classes,
        )

    def training_step(self, batch, batch_idx):
        images, masks = batch
        images = images.to(self.device)
        masks = masks.to(self.device)

        # Forward
        logits = self.model(images)

        # Dice loss + Cross entropy
        ce_loss = F.cross_entropy(logits, masks)
        dice_loss = self._dice_loss(logits, masks)
        loss = ce_loss + dice_loss

        self.backward(loss)

        if not self.is_accumulating_gradients:
            self.step(self.optimizer)

        return {"loss": loss, "ce_loss": ce_loss, "dice_loss": dice_loss}

    def _dice_loss(self, logits, targets):
        """Compute soft dice loss."""
        probs = F.softmax(logits, dim=1)
        targets_onehot = F.one_hot(targets, self.config.num_classes)
        targets_onehot = targets_onehot.permute(0, 3, 1, 2).float()

        intersection = (probs * targets_onehot).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + targets_onehot.sum(dim=(2, 3))

        dice = (2 * intersection + 1e-6) / (union + 1e-6)
        return 1 - dice.mean()

    def validation_step(self, batch, batch_idx):
        images, masks = batch
        images = images.to(self.device)
        masks = masks.to(self.device)

        logits = self.model(images)
        loss = F.cross_entropy(logits, masks)

        preds = logits.argmax(dim=1)
        self.dice_score(preds, masks)
        self.iou(preds, masks)

        return {"val_loss": loss}
```

---

## Diffusion Models

### DDPM for Image Generation

```python
"""Denoising Diffusion Probabilistic Model example."""
from dataclasses import dataclass
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from dream_trainer import DreamTrainer, DreamTrainerConfig
from dream_trainer.trainer.mixins import SetupMixin, SetupConfigMixin
from dream_trainer.callbacks import MediaLoggerCallback


@dataclass
class DDPMConfig(DreamTrainerConfig, SetupConfigMixin):
    # Model
    image_size: int = 64
    channels: int = 3
    hidden_channels: int = 128
    num_res_blocks: int = 2

    # Diffusion
    num_timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02

    # Training
    learning_rate: float = 2e-4


class SinusoidalPositionEmbeddings(nn.Module):
    """Timestep embeddings."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat([embeddings.sin(), embeddings.cos()], dim=-1)
        return embeddings


class ResBlock(nn.Module):
    """Residual block with time embedding."""

    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.time_mlp = nn.Linear(time_dim, out_ch)
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)

        if in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, 1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, t):
        h = F.silu(self.norm1(self.conv1(x)))
        h = h + self.time_mlp(t)[:, :, None, None]
        h = F.silu(self.norm2(self.conv2(h)))
        return h + self.shortcut(x)


class UNetDiffusion(nn.Module):
    """U-Net for diffusion models."""

    def __init__(self, config: DDPMConfig):
        super().__init__()
        c = config.hidden_channels
        time_dim = c * 4

        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(c),
            nn.Linear(c, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        # Encoder
        self.conv_in = nn.Conv2d(config.channels, c, 3, padding=1)
        self.down1 = nn.ModuleList([ResBlock(c, c, time_dim) for _ in range(config.num_res_blocks)])
        self.down2 = nn.ModuleList([ResBlock(c, c * 2, time_dim) for _ in range(config.num_res_blocks)])
        self.down3 = nn.ModuleList([ResBlock(c * 2, c * 4, time_dim) for _ in range(config.num_res_blocks)])

        # Bottleneck
        self.mid = ResBlock(c * 4, c * 4, time_dim)

        # Decoder
        self.up3 = nn.ModuleList([ResBlock(c * 8, c * 2, time_dim) for _ in range(config.num_res_blocks)])
        self.up2 = nn.ModuleList([ResBlock(c * 4, c, time_dim) for _ in range(config.num_res_blocks)])
        self.up1 = nn.ModuleList([ResBlock(c * 2, c, time_dim) for _ in range(config.num_res_blocks)])

        self.conv_out = nn.Conv2d(c, config.channels, 3, padding=1)

        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x, t):
        t = self.time_mlp(t)

        # Encoder
        x = self.conv_in(x)
        h1 = x
        for block in self.down1:
            h1 = block(h1, t)

        h2 = self.pool(h1)
        for block in self.down2:
            h2 = block(h2, t)

        h3 = self.pool(h2)
        for block in self.down3:
            h3 = block(h3, t)

        # Bottleneck
        h = self.mid(self.pool(h3), t)

        # Decoder
        h = self.upsample(h)
        h = torch.cat([h, h3], dim=1)
        for block in self.up3:
            h = block(h, t)

        h = self.upsample(h)
        h = torch.cat([h, h2], dim=1)
        for block in self.up2:
            h = block(h, t)

        h = self.upsample(h)
        h = torch.cat([h, h1], dim=1)
        for block in self.up1:
            h = block(h, t)

        return self.conv_out(h)


class DDPMTrainer(DreamTrainer):
    config: DDPMConfig

    def configure_models(self):
        self.model = UNetDiffusion(self.config)

        # Precompute diffusion schedule
        betas = torch.linspace(
            self.config.beta_start,
            self.config.beta_end,
            self.config.num_timesteps,
        )
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))

    def q_sample(self, x_0, t, noise=None):
        """Add noise to images (forward diffusion)."""
        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_alpha = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]

        return sqrt_alpha * x_0 + sqrt_one_minus_alpha * noise

    def training_step(self, batch, batch_idx):
        images = batch[0] if isinstance(batch, (list, tuple)) else batch
        images = images.to(self.device)

        # Sample random timesteps
        t = torch.randint(
            0, self.config.num_timesteps,
            (images.size(0),),
            device=self.device,
        )

        # Sample noise and create noisy images
        noise = torch.randn_like(images)
        x_t = self.q_sample(images, t, noise)

        # Predict noise
        noise_pred = self.model(x_t, t)

        # MSE loss on noise prediction
        loss = F.mse_loss(noise_pred, noise)

        self.backward(loss)

        if not self.is_accumulating_gradients:
            self.step(self.optimizer)

        return {"loss": loss}

    @torch.no_grad()
    def sample(self, batch_size: int = 16):
        """Generate samples using DDPM sampling."""
        device = self.device
        x = torch.randn(
            batch_size,
            self.config.channels,
            self.config.image_size,
            self.config.image_size,
            device=device,
        )

        for t in reversed(range(self.config.num_timesteps)):
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            noise_pred = self.model(x, t_batch)

            alpha = self.alphas_cumprod[t]
            alpha_prev = self.alphas_cumprod[t - 1] if t > 0 else torch.tensor(1.0)
            beta = self.betas[t]

            # DDPM update
            x = (1 / torch.sqrt(1 - beta)) * (
                x - beta / torch.sqrt(1 - alpha) * noise_pred
            )

            if t > 0:
                noise = torch.randn_like(x)
                x = x + torch.sqrt(beta) * noise

        return x
```

---

## Summary

These examples demonstrate:

1. **Image Classification**: ResNet and ViT training with metrics
2. **Segmentation**: U-Net with Dice loss and skip connections
3. **Generative Models**: DDPM diffusion model training

Each example follows Dream Trainer best practices:

- Composable configuration with dataclasses
- Proper FSDP sharding for distributed training
- Metrics integration with torchmetrics
- Clean separation of model and training logic

## Next Steps

- [Language Models](nlp.md): GPT, LLaMA, and transformer training
- [Multi-Modal](multimodal.md): Vision-language models
- [Advanced Patterns](advanced.md): EMA, distillation, curriculum learning
