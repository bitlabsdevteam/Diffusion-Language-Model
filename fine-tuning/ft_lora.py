"""Optional LoRA (low-rank adaptation) for the ablation arm. Built in (no `peft` dependency): about 40 lines, works with
the tied embedding, and merges back into a plain DiffusionTransformerLM so the published checkpoint format is unchanged.

LoRA freezes each pre-trained weight W (out x in) and learns W' = W + (alpha / r) * B @ A with A (r x in) random-init and
B (out x r) zero-init, so training starts exactly at the base model. Only A and B train; r is the rank bottleneck.

Targets: attn.qkv, attn.out, mlp.w1, mlp.w3, mlp.w2 in every block. NOT adapted: tok_emb and lm_head (they are one tied
tensor: adapting either would break the tie), and the RMSNorm gains (frozen).
"""
from __future__ import annotations

import _path  # noqa: F401

import copy
import math

import torch
import torch.nn as nn

TARGETS = ("attn.qkv", "attn.out", "mlp.w1", "mlp.w3", "mlp.w2")


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        if r < 1:
            raise ValueError("LoRA rank must be >= 1")
        self.base, self.r, self.scale = base, r, alpha / r
        self.A = nn.Parameter(torch.empty(r, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        return self.base(x) + (self.drop(x) @ self.A.t() @ self.B.t()) * self.scale

    def merged_weight(self):
        return self.base.weight + self.scale * (self.B @ self.A)


def _get(root, dotted):
    parent = root
    *path, leaf = dotted.split(".")
    for p in path:
        parent = getattr(parent, p)
    return parent, leaf


def apply_lora(model, r: int, alpha: float | None = None, dropout: float = 0.0) -> int:
    """Freeze every parameter, then wrap the target linears with trainable adapters. Returns the trainable parameter count."""
    alpha = float(alpha if alpha is not None else 2 * r)
    for p in model.parameters():
        p.requires_grad_(False)
    for block in model.blocks:
        for name in TARGETS:
            parent, leaf = _get(block, name)
            setattr(parent, leaf, LoRALinear(getattr(parent, leaf), r, alpha, dropout))
    return count_trainable(model)


def count_trainable(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def has_lora(model) -> bool:
    return any(isinstance(m, LoRALinear) for m in model.modules())


def merge_lora(model):
    """A plain copy of `model` with every adapter folded into its base weight (the original is left untouched)."""
    out = copy.deepcopy(model)  # deepcopy keeps the tok_emb / lm_head tie
    for block in out.blocks:
        for name in TARGETS:
            parent, leaf = _get(block, name)
            mod = getattr(parent, leaf)
            if isinstance(mod, LoRALinear):
                lin = nn.Linear(mod.base.in_features, mod.base.out_features, bias=False)
                lin.weight = nn.Parameter(mod.merged_weight().detach().clone())
                setattr(parent, leaf, lin)
    for p in out.parameters():
        p.requires_grad_(True)
    return out
