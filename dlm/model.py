"""Phase 3: the bidirectional Transformer that predicts the original token at masked positions.

LLaDA-style (docs/adr/0001-llada-style-architecture.md): RMSNorm, rotary positions (RoPE), SwiGLU, bidirectional
attention (no causal mask), tied input/output embeddings, no time embedding, no biases.

  * Input  : input_ids [B, L] (with [MASK] at noised positions), L <= config.seq_len.
  * Output : logits [B, L, vocab_size].
  * [PAD] positions are masked out as attention KEYS (nothing attends to padding), consistent with [PAD] never
    being predicted. Queries at [PAD] positions still produce (ignored) outputs. Consequence for Phase 6:
    the sampler must decide what fills the window after a generated [EOS] (grill it there).
  * Init: normal(0, 0.02) for every Linear and the embedding (no residual rescaling); recorded here on purpose.
  * Sizes: "base" = width 512, 10 layers, 8 heads, SwiGLU hidden 1376 (~2/3 of 2048, multiple of 32) ~35.8M
    parameters with the 8,192-token vocabulary; "tiny" is for tests and smoke runs.
Saved as <dir>/config.json + <dir>/model.pt (see save_pretrained / load_model).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

PRESETS = {
    "base": dict(d_model=512, n_layers=10, n_heads=8, d_ff=1376),
    "tiny": dict(d_model=64, n_layers=2, n_heads=4, d_ff=176),
}


@dataclass
class DLMConfig:
    """Architecture settings, saved as config.json next to each checkpoint."""

    vocab_size: int = 8192
    seq_len: int = 512          # maximum positions (RoPE table length)
    d_model: int = 512
    n_layers: int = 10
    n_heads: int = 8
    d_ff: int = 1376
    rope_theta: float = 10000.0
    norm_eps: float = 1e-6
    init_std: float = 0.02
    pad_id: int = 0
    mask_id: int = 4

    def __post_init__(self):
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if (self.d_model // self.n_heads) % 2:
            raise ValueError("head dimension must be even for RoPE")

    @classmethod
    def from_preset(cls, name: str, **overrides) -> "DLMConfig":
        return cls(**{**PRESETS[name], **overrides})

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "DLMConfig":
        return cls(**d)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        xf = x.float()
        return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)).to(x.dtype) * self.weight


def rope_tables(seq_len: int, head_dim: int, theta: float):
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    freqs = torch.outer(torch.arange(seq_len, dtype=torch.float32), inv)  # [L, head_dim/2]
    return freqs.cos(), freqs.sin()


def apply_rope(x, cos, sin):
    """x [B, H, L, head_dim]; rotate the two halves of the head dimension by position-dependent angles."""
    x1, x2 = x.float().chunk(2, dim=-1)
    out = torch.cat((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)
    return out.to(x.dtype)


class Attention(nn.Module):
    def __init__(self, cfg: DLMConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.out = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x, cos, sin, key_mask):
        B, L, D = x.shape
        q, k, v = self.qkv(x).view(B, L, 3, self.n_heads, D // self.n_heads).permute(2, 0, 3, 1, 4)
        q, k = apply_rope(q, cos[:L], sin[:L]), apply_rope(k, cos[:L], sin[:L])
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=key_mask, is_causal=False)
        return self.out(y.transpose(1, 2).reshape(B, L, D))


class SwiGLU(nn.Module):
    def __init__(self, cfg: DLMConfig):
        super().__init__()
        self.w1 = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.w3 = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.w2 = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    def __init__(self, cfg: DLMConfig):
        super().__init__()
        self.norm1, self.norm2 = RMSNorm(cfg.d_model, cfg.norm_eps), RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn, self.mlp = Attention(cfg), SwiGLU(cfg)

    def forward(self, x, cos, sin, key_mask):
        x = x + self.attn(self.norm1(x), cos, sin, key_mask)
        return x + self.mlp(self.norm2(x))


class DiffusionTransformerLM(nn.Module):
    """forward(input_ids [B, L]) -> logits [B, L, vocab]. Bidirectional (no causal mask)."""

    def __init__(self, config: DLMConfig):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList(Block(config) for _ in range(config.n_layers))
        self.norm = RMSNorm(config.d_model, config.norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        cos, sin = rope_tables(config.seq_len, config.d_model // config.n_heads, config.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.apply(self._init)
        self.lm_head.weight = self.tok_emb.weight  # tied

    def _init(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=self.config.init_std)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())  # tied weights are counted once

    def forward(self, input_ids, attention_mask=None):
        L = input_ids.shape[1]
        if L > self.config.seq_len:
            raise ValueError(f"sequence length {L} exceeds config.seq_len {self.config.seq_len}")
        if attention_mask is None:
            attention_mask = input_ids != self.config.pad_id  # True = real token
        key_mask = attention_mask[:, None, None, :]
        x = self.tok_emb(input_ids)
        for block in self.blocks:
            x = block(x, self.rope_cos, self.rope_sin, key_mask)
        return self.lm_head(self.norm(x))

    def save_pretrained(self, directory) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "config.json").write_text(json.dumps(self.config.to_dict(), indent=2) + "\n")
        torch.save(self.state_dict(), directory / "model.pt")


def load_model(directory, device="cpu") -> DiffusionTransformerLM:
    directory = Path(directory)
    cfg = DLMConfig.from_dict(json.loads((directory / "config.json").read_text()))
    model = DiffusionTransformerLM(cfg)
    model.load_state_dict(torch.load(directory / "model.pt", map_location="cpu"))
    return model.to(device)


def describe(model: DiffusionTransformerLM) -> str:
    c = model.config
    return (f"{model.num_parameters() / 1e6:.2f}M parameters (width {c.d_model}, {c.n_layers} layers, "
            f"{c.n_heads} heads, SwiGLU {c.d_ff}, vocab {c.vocab_size}, max length {c.seq_len})")
