# Multi-Modal Model Examples

This page provides examples for training vision-language models with Dream Trainer, including CLIP, image captioning, and visual question answering.

---

## CLIP: Contrastive Language-Image Pre-training

### Complete CLIP Training

```python
"""CLIP model training example."""
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from dream_trainer import DreamTrainer, DreamTrainerConfig
from dream_trainer.trainer.mixins import SetupMixin, SetupConfigMixin
from dream_trainer.configs import DeviceParameters, TrainingParameters


@dataclass
class CLIPConfig(DreamTrainerConfig, SetupConfigMixin):
    # Vision encoder
    image_size: int = 224
    patch_size: int = 16
    vision_width: int = 768
    vision_layers: int = 12
    vision_heads: int = 12

    # Text encoder
    vocab_size: int = 49408
    context_length: int = 77
    text_width: int = 512
    text_layers: int = 12
    text_heads: int = 8

    # Shared
    embed_dim: int = 512

    # Training
    learning_rate: float = 5e-4
    temperature: float = 0.07


class VisionTransformer(nn.Module):
    """Vision encoder for CLIP."""

    def __init__(self, config: CLIPConfig):
        super().__init__()
        self.patch_embed = nn.Conv2d(
            3, config.vision_width,
            kernel_size=config.patch_size,
            stride=config.patch_size,
        )

        num_patches = (config.image_size // config.patch_size) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.vision_width))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, config.vision_width)
        )

        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=config.vision_width,
                nhead=config.vision_heads,
                dim_feedforward=config.vision_width * 4,
                batch_first=True,
            )
            for _ in range(config.vision_layers)
        ])

        self.ln_final = nn.LayerNorm(config.vision_width)
        self.projection = nn.Linear(config.vision_width, config.embed_dim, bias=False)

    def forward(self, x):
        # Patch embedding
        x = self.patch_embed(x).flatten(2).transpose(1, 2)

        # Add CLS token and position embedding
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed

        # Transformer
        for block in self.blocks:
            x = block(x)

        # Get CLS token and project
        x = self.ln_final(x[:, 0])
        x = self.projection(x)

        return x


class TextTransformer(nn.Module):
    """Text encoder for CLIP."""

    def __init__(self, config: CLIPConfig):
        super().__init__()
        self.token_embedding = nn.Embedding(config.vocab_size, config.text_width)
        self.positional_embedding = nn.Parameter(
            torch.zeros(config.context_length, config.text_width)
        )

        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=config.text_width,
                nhead=config.text_heads,
                dim_feedforward=config.text_width * 4,
                batch_first=True,
            )
            for _ in range(config.text_layers)
        ])

        self.ln_final = nn.LayerNorm(config.text_width)
        self.projection = nn.Linear(config.text_width, config.embed_dim, bias=False)

        # Causal mask
        mask = torch.triu(torch.ones(config.context_length, config.context_length), diagonal=1)
        self.register_buffer("attn_mask", mask.bool())

    def forward(self, text):
        x = self.token_embedding(text) + self.positional_embedding[:text.size(1)]

        # Causal transformer
        for block in self.blocks:
            x = block(x, src_mask=self.attn_mask[:x.size(1), :x.size(1)])

        # Get EOT token embedding (last non-padded position)
        x = self.ln_final(x)

        # Use argmax to find EOT position (simplified - assumes EOT is max token)
        eot_indices = text.argmax(dim=-1)
        x = x[torch.arange(x.size(0)), eot_indices]

        x = self.projection(x)
        return x


class CLIP(nn.Module):
    """CLIP model combining vision and text encoders."""

    def __init__(self, config: CLIPConfig):
        super().__init__()
        self.visual = VisionTransformer(config)
        self.text = TextTransformer(config)
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1 / config.temperature)))

    def encode_image(self, image):
        return F.normalize(self.visual(image), dim=-1)

    def encode_text(self, text):
        return F.normalize(self.text(text), dim=-1)

    def forward(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        # Cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        return logits_per_image, logits_per_text


class CLIPTrainer(DreamTrainer):
    config: CLIPConfig

    def configure_models(self):
        self.model = CLIP(self.config)

    def init_weights(self):
        def _init(module):
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)

        self.model.apply(_init)

    def configure_optimizers(self):
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            betas=(0.9, 0.98),
            weight_decay=0.2,
        )

    def training_step(self, batch, batch_idx):
        images, texts = batch
        images = images.to(self.device)
        texts = texts.to(self.device)

        # Get logits
        logits_per_image, logits_per_text = self.model(images, texts)

        # Symmetric cross-entropy loss
        batch_size = images.size(0)
        labels = torch.arange(batch_size, device=self.device)

        loss_i = F.cross_entropy(logits_per_image, labels)
        loss_t = F.cross_entropy(logits_per_text, labels)
        loss = (loss_i + loss_t) / 2

        self.backward(loss)

        if not self.is_accumulating_gradients:
            grad_norm = self.step(self.optimizer)
            return {
                "loss": loss,
                "loss_image": loss_i,
                "loss_text": loss_t,
                "grad_norm": grad_norm,
            }

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        images, texts = batch
        images = images.to(self.device)
        texts = texts.to(self.device)

        logits_per_image, logits_per_text = self.model(images, texts)

        batch_size = images.size(0)
        labels = torch.arange(batch_size, device=self.device)

        loss_i = F.cross_entropy(logits_per_image, labels)
        loss_t = F.cross_entropy(logits_per_text, labels)
        loss = (loss_i + loss_t) / 2

        # Compute retrieval accuracy
        pred_i = logits_per_image.argmax(dim=-1)
        pred_t = logits_per_text.argmax(dim=-1)

        acc_i = (pred_i == labels).float().mean()
        acc_t = (pred_t == labels).float().mean()

        return {
            "val_loss": loss,
            "i2t_acc": acc_i,
            "t2i_acc": acc_t,
        }
```

---

## Image Captioning

### Encoder-Decoder Image Captioning

```python
"""Image captioning with encoder-decoder architecture."""
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

from dream_trainer import DreamTrainer, DreamTrainerConfig


@dataclass
class CaptioningConfig(DreamTrainerConfig):
    # Vision encoder
    image_size: int = 224
    patch_size: int = 16
    encoder_dim: int = 768
    encoder_layers: int = 12
    encoder_heads: int = 12

    # Text decoder
    vocab_size: int = 50257
    decoder_dim: int = 768
    decoder_layers: int = 6
    decoder_heads: int = 12
    max_caption_length: int = 128

    # Training
    learning_rate: float = 1e-4
    label_smoothing: float = 0.1


class ImageEncoder(nn.Module):
    """Vision Transformer encoder."""

    def __init__(self, config: CaptioningConfig):
        super().__init__()
        self.patch_embed = nn.Conv2d(
            3, config.encoder_dim,
            kernel_size=config.patch_size,
            stride=config.patch_size,
        )

        num_patches = (config.image_size // config.patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, config.encoder_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.encoder_dim,
            nhead=config.encoder_heads,
            dim_feedforward=config.encoder_dim * 4,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, config.encoder_layers)
        self.norm = nn.LayerNorm(config.encoder_dim)

    def forward(self, images):
        x = self.patch_embed(images).flatten(2).transpose(1, 2)
        x = x + self.pos_embed
        x = self.transformer(x)
        return self.norm(x)


class CaptionDecoder(nn.Module):
    """Transformer decoder for caption generation."""

    def __init__(self, config: CaptioningConfig):
        super().__init__()
        self.token_embed = nn.Embedding(config.vocab_size, config.decoder_dim)
        self.pos_embed = nn.Embedding(config.max_caption_length, config.decoder_dim)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.decoder_dim,
            nhead=config.decoder_heads,
            dim_feedforward=config.decoder_dim * 4,
            batch_first=True,
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, config.decoder_layers)

        # Cross-attention projection (if encoder_dim != decoder_dim)
        if config.encoder_dim != config.decoder_dim:
            self.encoder_proj = nn.Linear(config.encoder_dim, config.decoder_dim)
        else:
            self.encoder_proj = nn.Identity()

        self.norm = nn.LayerNorm(config.decoder_dim)
        self.output = nn.Linear(config.decoder_dim, config.vocab_size)

        # Causal mask
        mask = torch.triu(torch.ones(config.max_caption_length, config.max_caption_length), diagonal=1)
        self.register_buffer("causal_mask", mask.bool())

    def forward(self, encoder_output, captions):
        B, T = captions.size()

        # Project encoder output
        memory = self.encoder_proj(encoder_output)

        # Token + position embeddings
        positions = torch.arange(T, device=captions.device)
        x = self.token_embed(captions) + self.pos_embed(positions)

        # Causal mask
        tgt_mask = self.causal_mask[:T, :T]

        # Transformer decoder
        x = self.transformer(x, memory, tgt_mask=tgt_mask)
        x = self.norm(x)

        return self.output(x)


class CaptioningModel(nn.Module):
    """Complete image captioning model."""

    def __init__(self, config: CaptioningConfig):
        super().__init__()
        self.encoder = ImageEncoder(config)
        self.decoder = CaptionDecoder(config)

    def forward(self, images, captions):
        encoder_output = self.encoder(images)
        logits = self.decoder(encoder_output, captions)
        return logits

    @torch.no_grad()
    def generate(self, images, max_length=128, temperature=1.0, top_k=50):
        """Generate captions autoregressively."""
        B = images.size(0)
        device = images.device

        encoder_output = self.encoder(images)

        # Start with BOS token
        generated = torch.full((B, 1), 1, dtype=torch.long, device=device)  # Assume 1 is BOS

        for _ in range(max_length - 1):
            logits = self.decoder(encoder_output, generated)
            next_token_logits = logits[:, -1, :] / temperature

            # Top-k filtering
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')

            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            generated = torch.cat([generated, next_token], dim=-1)

            # Stop if all sequences have EOS
            if (next_token == 2).all():  # Assume 2 is EOS
                break

        return generated


class CaptioningTrainer(DreamTrainer):
    config: CaptioningConfig

    def configure_models(self):
        self.model = CaptioningModel(self.config)

    def training_step(self, batch, batch_idx):
        images = batch['images'].to(self.device)
        captions = batch['captions'].to(self.device)

        # Shift for teacher forcing
        inputs = captions[:, :-1]
        targets = captions[:, 1:]

        logits = self.model(images, inputs)

        loss = F.cross_entropy(
            logits.view(-1, self.config.vocab_size),
            targets.reshape(-1),
            ignore_index=0,  # Pad token
            label_smoothing=self.config.label_smoothing,
        )

        self.backward(loss)

        if not self.is_accumulating_gradients:
            self.step(self.optimizer)

        return {"loss": loss}
```

---

## Visual Question Answering (VQA)

### Multi-Modal Fusion for VQA

```python
"""Visual Question Answering example."""
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

from dream_trainer import DreamTrainer, DreamTrainerConfig


@dataclass
class VQAConfig(DreamTrainerConfig):
    # Vision
    image_size: int = 384
    vision_dim: int = 768
    vision_layers: int = 12

    # Text
    vocab_size: int = 30522
    text_dim: int = 768
    text_layers: int = 6

    # Fusion
    fusion_dim: int = 768
    fusion_layers: int = 6
    num_answers: int = 3129  # VQA v2 answer vocabulary


class VisionEncoder(nn.Module):
    def __init__(self, config: VQAConfig):
        super().__init__()
        # Simplified - use pretrained ViT in practice
        self.patch_embed = nn.Conv2d(3, config.vision_dim, 16, 16)
        num_patches = (config.image_size // 16) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, config.vision_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            config.vision_dim, 12, config.vision_dim * 4, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, config.vision_layers)

    def forward(self, images):
        x = self.patch_embed(images).flatten(2).transpose(1, 2)
        x = x + self.pos_embed
        return self.encoder(x)


class TextEncoder(nn.Module):
    def __init__(self, config: VQAConfig):
        super().__init__()
        self.embed = nn.Embedding(config.vocab_size, config.text_dim)
        self.pos_embed = nn.Embedding(512, config.text_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            config.text_dim, 12, config.text_dim * 4, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, config.text_layers)

    def forward(self, text, attention_mask=None):
        B, T = text.size()
        positions = torch.arange(T, device=text.device)

        x = self.embed(text) + self.pos_embed(positions)

        if attention_mask is not None:
            # Convert to transformer mask format
            mask = attention_mask == 0
        else:
            mask = None

        return self.encoder(x, src_key_padding_mask=mask)


class CrossAttentionFusion(nn.Module):
    """Cross-attention based multi-modal fusion."""

    def __init__(self, config: VQAConfig):
        super().__init__()
        self.vision_proj = nn.Linear(config.vision_dim, config.fusion_dim)
        self.text_proj = nn.Linear(config.text_dim, config.fusion_dim)

        decoder_layer = nn.TransformerDecoderLayer(
            config.fusion_dim, 12, config.fusion_dim * 4, batch_first=True
        )
        self.fusion = nn.TransformerDecoder(decoder_layer, config.fusion_layers)

        self.norm = nn.LayerNorm(config.fusion_dim)

    def forward(self, vision_features, text_features):
        # Project to common dimension
        v = self.vision_proj(vision_features)
        t = self.text_proj(text_features)

        # Text attends to vision (cross-attention)
        fused = self.fusion(t, v)
        fused = self.norm(fused)

        # Pool over sequence
        return fused.mean(dim=1)


class VQAModel(nn.Module):
    def __init__(self, config: VQAConfig):
        super().__init__()
        self.vision_encoder = VisionEncoder(config)
        self.text_encoder = TextEncoder(config)
        self.fusion = CrossAttentionFusion(config)
        self.classifier = nn.Sequential(
            nn.Linear(config.fusion_dim, config.fusion_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.fusion_dim * 2, config.num_answers),
        )

    def forward(self, images, questions, attention_mask=None):
        vision_features = self.vision_encoder(images)
        text_features = self.text_encoder(questions, attention_mask)
        fused = self.fusion(vision_features, text_features)
        logits = self.classifier(fused)
        return logits


class VQATrainer(DreamTrainer):
    config: VQAConfig

    def configure_models(self):
        self.model = VQAModel(self.config)

    def training_step(self, batch, batch_idx):
        images = batch['images'].to(self.device)
        questions = batch['questions'].to(self.device)
        attention_mask = batch.get('attention_mask')
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        # Soft labels for VQA (multiple valid answers)
        labels = batch['labels'].to(self.device)

        logits = self.model(images, questions, attention_mask)

        # Binary cross entropy for soft labels
        loss = F.binary_cross_entropy_with_logits(logits, labels)

        self.backward(loss)

        if not self.is_accumulating_gradients:
            self.step(self.optimizer)

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        images = batch['images'].to(self.device)
        questions = batch['questions'].to(self.device)
        attention_mask = batch.get('attention_mask')
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        labels = batch['labels'].to(self.device)

        logits = self.model(images, questions, attention_mask)
        loss = F.binary_cross_entropy_with_logits(logits, labels)

        # VQA accuracy (based on soft labels)
        preds = logits.argmax(dim=-1)
        # Get score for predicted answer
        acc = labels[torch.arange(len(preds)), preds].mean()

        return {"val_loss": loss, "val_acc": acc}
```

---

## LLaVA-Style Vision-Language Model

### Vision-Language Model with LLM Backbone

```python
"""LLaVA-style vision-language model."""
from dataclasses import dataclass
import torch
import torch.nn as nn

from dream_trainer import DreamTrainer, DreamTrainerConfig


@dataclass
class LLaVAConfig(DreamTrainerConfig):
    # Vision encoder (e.g., CLIP ViT)
    vision_hidden_size: int = 1024
    image_size: int = 336
    patch_size: int = 14

    # LLM backbone
    vocab_size: int = 32000
    hidden_size: int = 4096
    num_layers: int = 32
    num_heads: int = 32
    max_seq_length: int = 2048

    # Projector
    projector_hidden_size: int = 4096


class MLPProjector(nn.Module):
    """Projects vision features to LLM embedding space."""

    def __init__(self, config: LLaVAConfig):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(config.vision_hidden_size, config.projector_hidden_size),
            nn.GELU(),
            nn.Linear(config.projector_hidden_size, config.hidden_size),
        )

    def forward(self, x):
        return self.proj(x)


class LLaVA(nn.Module):
    """Vision-Language Model combining vision encoder with LLM."""

    def __init__(self, config: LLaVAConfig):
        super().__init__()
        self.config = config

        # Vision encoder (simplified - use pretrained CLIP in practice)
        from .vision import VisionTransformer  # Reuse from earlier examples
        self.vision_encoder = VisionTransformer(config)

        # Vision-to-language projector
        self.projector = MLPProjector(config)

        # LLM backbone (simplified - use pretrained LLaMA in practice)
        from .nlp import LLaMA  # Reuse from earlier examples
        self.llm = LLaMA(config)

        # Freeze vision encoder during fine-tuning
        for param in self.vision_encoder.parameters():
            param.requires_grad = False

    def encode_images(self, images):
        """Encode images to language model space."""
        vision_features = self.vision_encoder(images)
        return self.projector(vision_features)

    def forward(self, input_ids, images=None, image_positions=None):
        """
        Forward pass with interleaved image-text inputs.

        Args:
            input_ids: Token IDs [B, T]
            images: Image tensors [B, N_images, C, H, W] or None
            image_positions: Positions to insert image features

        Returns:
            logits: [B, T, vocab_size]
        """
        # Get text embeddings
        text_embeds = self.llm.embed_tokens(input_ids)

        if images is not None:
            B, N, C, H, W = images.size()

            # Encode all images
            images_flat = images.view(B * N, C, H, W)
            image_features = self.encode_images(images_flat)
            image_features = image_features.view(B, N, -1, self.config.hidden_size)

            # Insert image features at specified positions
            # (Simplified - production code handles this more carefully)
            for b in range(B):
                for i, pos in enumerate(image_positions[b]):
                    num_patches = image_features.size(2)
                    text_embeds[b, pos:pos+num_patches] = image_features[b, i]

        # Forward through LLM
        hidden = text_embeds
        for layer in self.llm.layers:
            hidden = layer(hidden, self.llm.freqs_cis[:hidden.size(1)], self.llm.mask[:hidden.size(1), :hidden.size(1)])

        hidden = self.llm.norm(hidden)
        logits = self.llm.lm_head(hidden)

        return logits


class LLaVATrainer(DreamTrainer):
    config: LLaVAConfig

    def configure_models(self):
        self.model = LLaVA(self.config)

    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids'].to(self.device)
        images = batch.get('images')
        if images is not None:
            images = images.to(self.device)
        image_positions = batch.get('image_positions')
        labels = batch['labels'].to(self.device)

        logits = self.model(input_ids, images, image_positions)

        # Shift for next-token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss = F.cross_entropy(
            shift_logits.view(-1, self.config.vocab_size),
            shift_labels.view(-1),
            ignore_index=-100,
        )

        self.backward(loss)

        if not self.is_accumulating_gradients:
            self.step(self.optimizer)

        return {"loss": loss}
```

---

## Summary

These examples demonstrate:

1. **CLIP**: Contrastive learning for image-text alignment
2. **Image Captioning**: Encoder-decoder architecture with cross-attention
3. **VQA**: Multi-modal fusion for question answering
4. **LLaVA**: Modern vision-language model with LLM backbone

Key techniques covered:

- Contrastive learning with symmetric loss
- Cross-attention for multi-modal fusion
- Interleaved image-text inputs
- Vision feature projection to language space

## Next Steps

- [Advanced Patterns](advanced.md): EMA, distillation, curriculum learning
- [Performance Guide](../performance.md): Optimize multi-modal training
