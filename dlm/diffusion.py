"""Phase 4: the masked-diffusion objective (forward noising and ELBO-weighted loss).

Decisions (grilled 2026-11-22): continuous time t in (0, 1]; linear mask rate (cosine sin(pi*t/2) behind a flag
for the Phase 7 ablation); stratified t; no time input to the model.

  * Mask rate  p(t) = (1 - EPS) * s(t) + EPS with s linear (s = t) or cosine, EPS = 1e-3 (LLaDA bounds t away from 0
    so the 1/p weight cannot explode).
  * ELBO weight w(t) = p'(t) / p(t): (1 - EPS) / p for linear. Then E_t[w * p] = 1 - EPS (a test checks this).
  * Corruption: every token that is not protected and not [PAD] ("maskable") is replaced by [MASK] with
    probability p(t), independently. [BOS]-header tokens (protected) and [PAD] are never masked.
  * Loss per block = sum over masked tokens of w * cross-entropy, divided by the number of maskable tokens in the
    block; training loss = mean over blocks. Evaluation reports the per-token bound
    sum(w * CE over masked) / sum(maskable) over the whole evaluation set (see `loss_sums`).
  * All randomness uses an explicit CPU torch.Generator, so evaluation never touches the training RNG and a
    resumed run is exact.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

EPS = 1e-3


def mask_prob(t, schedule: str = "linear"):
    """Probability that a maskable token is masked at time t in [0, 1] (tensor in, tensor out)."""
    if schedule == "linear":
        s = t
    elif schedule == "cosine":
        s = torch.sin(0.5 * math.pi * t)
    else:
        raise ValueError(f"unknown schedule {schedule!r}")
    return (1.0 - EPS) * s + EPS


def elbo_weight(t, schedule: str = "linear"):
    """Per-block loss weight p'(t) / p(t)."""
    if schedule == "linear":
        ds = torch.ones_like(t)
    elif schedule == "cosine":
        ds = 0.5 * math.pi * torch.cos(0.5 * math.pi * t)
    else:
        raise ValueError(f"unknown schedule {schedule!r}")
    return (1.0 - EPS) * ds / mask_prob(t, schedule)


def sample_t(batch_size: int, generator: torch.Generator, stratified: bool = True):
    """t in (0, 1] per block. Stratified: one draw from each of B equal intervals, then shuffled."""
    u = torch.rand(batch_size, generator=generator)
    if not stratified:
        return 1.0 - u  # (0, 1]
    t = (torch.arange(batch_size, dtype=torch.float32) + u) / batch_size
    return t[torch.randperm(batch_size, generator=generator)]


def corrupt(input_ids, protect, mask_id: int, pad_id: int, t, schedule: str = "linear",
            generator: torch.Generator | None = None):
    """Return (noised_ids, masked, maskable). `t` [B] is given; protected and [PAD] positions are never masked."""
    maskable = (~protect) & (input_ids != pad_id)
    p = mask_prob(t, schedule).to(input_ids.device)[:, None]
    noise = torch.rand(input_ids.shape, generator=generator).to(input_ids.device)
    masked = (noise < p) & maskable
    return torch.where(masked, torch.full_like(input_ids, mask_id), input_ids), masked, maskable


def loss_sums(model, input_ids, protect, mask_id: int, pad_id: int, t, schedule: str = "linear",
              generator: torch.Generator | None = None):
    """Per-block (weighted CE sum over masked tokens, number of maskable tokens, number masked); each [B]."""
    noised, masked, maskable = corrupt(input_ids, protect, mask_id, pad_id, t, schedule, generator)
    logits = model(noised)
    ce = F.cross_entropy(logits[masked].float(), input_ids[masked], reduction="none")  # only masked positions
    w = elbo_weight(t, schedule).to(input_ids.device)
    block = torch.arange(input_ids.shape[0], device=input_ids.device)[:, None].expand_as(masked)[masked]
    num = torch.zeros(input_ids.shape[0], device=input_ids.device).index_add_(0, block, ce * w[block])
    return num, maskable.sum(1), masked.sum(1)


def diffusion_loss(model, input_ids, protect, mask_id: int, pad_id: int, schedule: str = "linear",
                   generator: torch.Generator | None = None, t=None, stratified: bool = True):
    """Training loss: mean over blocks of (weighted CE over masked tokens / maskable tokens in the block)."""
    if t is None:
        t = sample_t(input_ids.shape[0], generator, stratified)
    num, den, _ = loss_sums(model, input_ids, protect, mask_id, pad_id, t, schedule, generator)
    return (num / den.clamp_min(1)).mean()
