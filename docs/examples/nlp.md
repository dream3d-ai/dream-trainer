# Language Model Examples

This page provides complete examples for training language models with Dream Trainer, including GPT-style models, LLaMA, and encoder-decoder architectures.

---

## GPT-2 Style Language Model

### Complete GPT Training

```python
"""GPT-2 style language model training."""
from dataclasses import dataclass
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.distributed._composable.fsdp import fully_shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    parallelize_module,
)

from dream_trainer import DreamTrainer, DreamTrainerConfig
from dream_trainer.trainer.mixins import (
    SetupMixin, SetupConfigMixin,
    WandBLoggerMixin, WandBLoggerConfigMixin,
)
from dream_trainer.configs import DeviceParameters, TrainingParameters
from dream_trainer.callbacks import (
    CheckpointCallback,
    LoggerCallback,
    ProgressBar,
    OptimizeFSDP,
    CallbackCollection,
)


@dataclass
class GPTConfig(DreamTrainerConfig, SetupConfigMixin, WandBLoggerConfigMixin):
    # Model architecture
    vocab_size: int = 50257
    hidden_size: int = 768
    num_layers: int = 12
    num_heads: int = 12
    max_seq_length: int = 1024
    dropout: float = 0.1

    # Training
    learning_rate: float = 6e-4
    weight_decay: float = 0.1
    warmup_steps: int = 2000
    max_steps: int = 100000

    # Data
    data_path: str = "./data/openwebtext"


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.hidden_size % config.num_heads == 0

        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads

        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Causal mask
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.max_seq_length, config.max_seq_length))
            .view(1, 1, config.max_seq_length, config.max_seq_length)
        )

    def forward(self, x):
        B, T, C = x.size()

        # Compute Q, K, V
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention scores
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        # Apply attention to values
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        return self.resid_dropout(self.out_proj(y))


class MLP(nn.Module):
    """Feed-forward network."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, 4 * config.hidden_size)
        self.fc2 = nn.Linear(4 * config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = F.gelu(self.fc1(x))
        x = self.dropout(self.fc2(x))
        return x


class TransformerBlock(nn.Module):
    """Transformer decoder block."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.hidden_size)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.hidden_size)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    """GPT-2 style language model."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.tok_emb = nn.Embedding(config.vocab_size, config.hidden_size)
        self.pos_emb = nn.Embedding(config.max_seq_length, config.hidden_size)
        self.drop = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.num_layers)
        ])

        self.ln_f = nn.LayerNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Weight tying
        self.lm_head.weight = self.tok_emb.weight

    def forward(self, idx):
        B, T = idx.size()
        assert T <= self.config.max_seq_length

        # Token + position embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        tok_emb = self.tok_emb(idx)
        pos_emb = self.pos_emb(pos)
        x = self.drop(tok_emb + pos_emb)

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        return logits


class TextDataset(Dataset):
    """Simple text dataset for language modeling."""

    def __init__(self, data_path: str, seq_length: int):
        # Load tokenized data (assumes pre-tokenized .bin file)
        self.data = torch.load(data_path)
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        return self.data[idx:idx + self.seq_length + 1]


class GPTTrainer(DreamTrainer, WandBLoggerMixin):
    config: GPTConfig

    def configure_models(self):
        """Create GPT model."""
        self.model = GPT(self.config)

    def init_weights(self):
        """Initialize weights with small std."""
        def _init_weights(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

        self.model.apply(_init_weights)

        # Scale residual projections
        for block in self.model.blocks:
            torch.nn.init.normal_(
                block.attn.out_proj.weight,
                mean=0.0,
                std=0.02 / math.sqrt(2 * self.config.num_layers)
            )
            torch.nn.init.normal_(
                block.mlp.fc2.weight,
                mean=0.0,
                std=0.02 / math.sqrt(2 * self.config.num_layers)
            )

    def apply_tensor_parallel(self, tp_mesh):
        """Apply tensor parallelism to attention and MLP."""
        for block in self.model.blocks:
            tp_plan = {
                "attn.q_proj": ColwiseParallel(),
                "attn.k_proj": ColwiseParallel(),
                "attn.v_proj": ColwiseParallel(),
                "attn.out_proj": RowwiseParallel(),
                "mlp.fc1": ColwiseParallel(),
                "mlp.fc2": RowwiseParallel(),
            }
            parallelize_module(block, tp_mesh, tp_plan)

    def apply_fully_shard(self, fsdp_config):
        """Apply FSDP to each transformer block."""
        for block in self.model.blocks:
            fully_shard(block, **fsdp_config)
        fully_shard(self.model, **fsdp_config)

    def configure_optimizers(self):
        """Configure AdamW with weight decay."""
        # Separate weight decay groups
        decay_params = []
        no_decay_params = []

        for name, param in self.model.named_parameters():
            if 'bias' in name or 'ln' in name or 'emb' in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        self.optimizer = torch.optim.AdamW([
            {'params': decay_params, 'weight_decay': self.config.weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0},
        ], lr=self.config.learning_rate, betas=(0.9, 0.95))

    def configure_schedulers(self):
        """Cosine schedule with warmup."""
        def lr_lambda(step):
            # Linear warmup
            if step < self.config.warmup_steps:
                return step / self.config.warmup_steps

            # Cosine decay
            progress = (step - self.config.warmup_steps) / (
                self.config.max_steps - self.config.warmup_steps
            )
            return 0.1 + 0.9 * (1 + math.cos(math.pi * progress)) / 2

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda
        )

    def configure_dataloaders(self):
        """Create train and validation dataloaders."""
        train_dataset = TextDataset(
            f"{self.config.data_path}/train.bin",
            self.config.max_seq_length,
        )
        val_dataset = TextDataset(
            f"{self.config.data_path}/val.bin",
            self.config.max_seq_length,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.training_parameters.train_batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.training_parameters.train_batch_size * 2,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

        return train_loader, val_loader

    def training_step(self, batch, batch_idx):
        """Training step with next-token prediction."""
        tokens = batch.to(self.device)
        inputs = tokens[:, :-1]
        targets = tokens[:, 1:]

        # Forward pass with optional loss parallelism
        with self.loss_parallel():
            logits = self.model(inputs)
            loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                targets.reshape(-1),
            )

        self.backward(loss)

        if not self.is_accumulating_gradients:
            grad_norm = self.step(self.optimizer)

            # Log metrics
            self.log_scalar("train/loss", loss)
            self.log_scalar("train/grad_norm", grad_norm)
            self.log_scalar("train/lr", self.optimizer.param_groups[0]['lr'])
            self.log_scalar("train/perplexity", torch.exp(loss))

            return {"loss": loss, "grad_norm": grad_norm}

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        tokens = batch.to(self.device)
        inputs = tokens[:, :-1]
        targets = tokens[:, 1:]

        logits = self.model(inputs)
        loss = F.cross_entropy(
            logits.view(-1, self.config.vocab_size),
            targets.reshape(-1),
        )

        return {"val_loss": loss, "val_perplexity": torch.exp(loss)}

    @torch.no_grad()
    def generate(self, prompt: torch.Tensor, max_new_tokens: int = 100):
        """Generate text from prompt."""
        self.model.eval()
        idx = prompt

        for _ in range(max_new_tokens):
            # Crop to max sequence length
            idx_cond = idx[:, -self.config.max_seq_length:]

            # Get predictions
            logits = self.model(idx_cond)
            logits = logits[:, -1, :]

            # Sample
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)

        return idx


if __name__ == "__main__":
    callbacks = CallbackCollection([
        ProgressBar(),
        LoggerCallback(log_every_n_train_batches=10),
        OptimizeFSDP(prefetch=2),
        CheckpointCallback(),
    ])

    config = GPTConfig(
        device_parameters=DeviceParameters(
            dp_shard=8,
            tensor_parallel=1,
            param_dtype=torch.bfloat16,
            compile_model=True,
        ),
        training_parameters=TrainingParameters(
            n_epochs=1,
            train_batch_size=8,
            gradient_accumulation_steps=4,
            gradient_clip_val=1.0,
            val_frequency=0.1,
        ),
        callbacks=callbacks,
        hidden_size=768,
        num_layers=12,
        num_heads=12,
    )

    trainer = GPTTrainer(config)
    trainer.fit()
```

---

## LLaMA-Style Model

### LLaMA Architecture with RoPE and RMSNorm

```python
"""LLaMA-style model training."""
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

from dream_trainer import DreamTrainer, DreamTrainerConfig
from dream_trainer.trainer.mixins import SetupMixin, SetupConfigMixin


@dataclass
class LLaMAConfig(DreamTrainerConfig, SetupConfigMixin):
    vocab_size: int = 32000
    hidden_size: int = 4096
    intermediate_size: int = 11008
    num_layers: int = 32
    num_heads: int = 32
    num_kv_heads: int = 32  # For GQA
    max_seq_length: int = 2048
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight


def precompute_freqs_cis(dim: int, max_seq_len: int, theta: float = 10000.0):
    """Precompute rotary embedding frequencies."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_seq_len)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def apply_rotary_emb(xq, xk, freqs_cis):
    """Apply rotary embeddings to queries and keys."""
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))

    freqs_cis = freqs_cis[:xq.shape[1]]
    freqs_cis = freqs_cis[None, :, None, :]

    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)

    return xq_out.type_as(xq), xk_out.type_as(xk)


class LLaMAAttention(nn.Module):
    """Multi-head attention with rotary embeddings and GQA support."""

    def __init__(self, config: LLaMAConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.num_kv_groups = config.num_heads // config.num_kv_heads

        self.q_proj = nn.Linear(config.hidden_size, config.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.num_heads * self.head_dim, config.hidden_size, bias=False)

    def forward(self, x, freqs_cis, mask=None):
        B, T, _ = x.size()

        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(B, T, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(B, T, self.num_kv_heads, self.head_dim)

        # Apply rotary embeddings
        q, k = apply_rotary_emb(q, k, freqs_cis)

        # Expand KV heads for GQA
        if self.num_kv_groups > 1:
            k = k.repeat_interleave(self.num_kv_groups, dim=2)
            v = v.repeat_interleave(self.num_kv_groups, dim=2)

        # Transpose for attention
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Scaled dot-product attention
        scale = 1.0 / (self.head_dim ** 0.5)
        attn = (q @ k.transpose(-2, -1)) * scale

        if mask is not None:
            attn = attn + mask

        attn = F.softmax(attn, dim=-1)
        out = attn @ v

        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.o_proj(out)


class LLaMAMLP(nn.Module):
    """SwiGLU-style MLP."""

    def __init__(self, config: LLaMAConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class LLaMABlock(nn.Module):
    """LLaMA transformer block."""

    def __init__(self, config: LLaMAConfig):
        super().__init__()
        self.attention_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.attention = LLaMAAttention(config)
        self.ffn_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = LLaMAMLP(config)

    def forward(self, x, freqs_cis, mask=None):
        x = x + self.attention(self.attention_norm(x), freqs_cis, mask)
        x = x + self.mlp(self.ffn_norm(x))
        return x


class LLaMA(nn.Module):
    """LLaMA model."""

    def __init__(self, config: LLaMAConfig):
        super().__init__()
        self.config = config

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            LLaMABlock(config) for _ in range(config.num_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Precompute rotary embeddings
        freqs_cis = precompute_freqs_cis(
            config.hidden_size // config.num_heads,
            config.max_seq_length,
            config.rope_theta,
        )
        self.register_buffer("freqs_cis", freqs_cis)

        # Causal mask
        mask = torch.full((config.max_seq_length, config.max_seq_length), float('-inf'))
        mask = torch.triu(mask, diagonal=1)
        self.register_buffer("mask", mask)

    def forward(self, tokens):
        B, T = tokens.size()

        x = self.embed_tokens(tokens)
        freqs_cis = self.freqs_cis[:T]
        mask = self.mask[:T, :T]

        for layer in self.layers:
            x = layer(x, freqs_cis, mask)

        x = self.norm(x)
        logits = self.lm_head(x)

        return logits


class LLaMATrainer(DreamTrainer):
    config: LLaMAConfig

    def configure_models(self):
        self.model = LLaMA(self.config)

    def apply_tensor_parallel(self, tp_mesh):
        from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel, parallelize_module

        for layer in self.model.layers:
            tp_plan = {
                "attention.q_proj": ColwiseParallel(),
                "attention.k_proj": ColwiseParallel(),
                "attention.v_proj": ColwiseParallel(),
                "attention.o_proj": RowwiseParallel(),
                "mlp.gate_proj": ColwiseParallel(),
                "mlp.up_proj": ColwiseParallel(),
                "mlp.down_proj": RowwiseParallel(),
            }
            parallelize_module(layer, tp_mesh, tp_plan)

    def apply_fully_shard(self, fsdp_config):
        from torch.distributed._composable.fsdp import fully_shard

        fully_shard(self.model.embed_tokens, **fsdp_config)
        for layer in self.model.layers:
            fully_shard(layer, **fsdp_config)
        fully_shard(self.model, **fsdp_config)

    # ... implement remaining methods
```

---

## BERT-Style Encoder

### Masked Language Modeling

```python
"""BERT-style masked language modeling."""
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

from dream_trainer import DreamTrainer, DreamTrainerConfig


@dataclass
class BERTConfig(DreamTrainerConfig):
    vocab_size: int = 30522
    hidden_size: int = 768
    num_layers: int = 12
    num_heads: int = 12
    intermediate_size: int = 3072
    max_seq_length: int = 512
    dropout: float = 0.1
    mask_prob: float = 0.15


class BERTEmbedding(nn.Module):
    def __init__(self, config: BERTConfig):
        super().__init__()
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embedding = nn.Embedding(config.max_seq_length, config.hidden_size)
        self.segment_embedding = nn.Embedding(2, config.hidden_size)
        self.norm = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, input_ids, segment_ids=None):
        B, T = input_ids.size()
        positions = torch.arange(T, device=input_ids.device).expand(B, -1)

        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        if segment_ids is not None:
            x = x + self.segment_embedding(segment_ids)

        return self.dropout(self.norm(x))


class BERTAttention(nn.Module):
    def __init__(self, config: BERTConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads

        self.query = nn.Linear(config.hidden_size, config.hidden_size)
        self.key = nn.Linear(config.hidden_size, config.hidden_size)
        self.value = nn.Linear(config.hidden_size, config.hidden_size)
        self.out = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, attention_mask=None):
        B, T, C = x.size()

        q = self.query(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        if attention_mask is not None:
            attn = attn + attention_mask

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.out(out)


class BERTBlock(nn.Module):
    def __init__(self, config: BERTConfig):
        super().__init__()
        self.attention = BERTAttention(config)
        self.norm1 = nn.LayerNorm(config.hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.GELU(),
            nn.Linear(config.intermediate_size, config.hidden_size),
            nn.Dropout(config.dropout),
        )
        self.norm2 = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, attention_mask=None):
        x = x + self.dropout(self.attention(self.norm1(x), attention_mask))
        x = x + self.mlp(self.norm2(x))
        return x


class BERT(nn.Module):
    def __init__(self, config: BERTConfig):
        super().__init__()
        self.embedding = BERTEmbedding(config)
        self.layers = nn.ModuleList([
            BERTBlock(config) for _ in range(config.num_layers)
        ])
        self.mlm_head = nn.Linear(config.hidden_size, config.vocab_size)

    def forward(self, input_ids, attention_mask=None, segment_ids=None):
        x = self.embedding(input_ids, segment_ids)

        # Convert attention mask
        if attention_mask is not None:
            attention_mask = attention_mask[:, None, None, :]
            attention_mask = (1.0 - attention_mask) * -10000.0

        for layer in self.layers:
            x = layer(x, attention_mask)

        return self.mlm_head(x)


class BERTTrainer(DreamTrainer):
    config: BERTConfig

    def configure_models(self):
        self.model = BERT(self.config)

    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        labels = batch['labels'].to(self.device)

        # Forward pass
        logits = self.model(input_ids, attention_mask)

        # MLM loss (only on masked positions)
        loss = F.cross_entropy(
            logits.view(-1, self.config.vocab_size),
            labels.view(-1),
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

1. **GPT-2**: Decoder-only autoregressive language model
2. **LLaMA**: Modern architecture with RoPE, RMSNorm, and SwiGLU
3. **BERT**: Encoder model for masked language modeling

Key techniques covered:

- Rotary position embeddings (RoPE)
- Grouped-query attention (GQA)
- RMSNorm vs LayerNorm
- SwiGLU activation
- Tensor and FSDP parallelism for large models

## Next Steps

- [Multi-Modal](multimodal.md): Vision-language models
- [Advanced Patterns](advanced.md): EMA, distillation, curriculum learning
- [Parallelism Guide](../parallelism.md): Scale to billions of parameters
