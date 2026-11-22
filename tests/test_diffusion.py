"""Offline logic tests for dlm/diffusion.py: python run.py test  (or python tests/test_diffusion.py)"""
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dlm import diffusion as df  # noqa: E402

V, PAD, MASK = 64, 0, 4


class Uniform(torch.nn.Module):
    """Stub model: uniform logits, so every masked token costs exactly ln V."""

    def forward(self, ids):
        return torch.zeros(*ids.shape, V)


class Oracle(torch.nn.Module):
    """Cheating stub: logits peaked on the true tokens (needs them), so the loss should be ~0."""

    def __init__(self, truth):
        super().__init__()
        self.truth = truth

    def forward(self, ids):
        return torch.nn.functional.one_hot(self.truth, V).float() * 60.0


def blocks(B, L, seed, n_protect=5, n_pad=20):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(5, V, (B, L), generator=g)
    ids[:, L - n_pad:] = PAD
    protect = torch.zeros(B, L, dtype=torch.bool)
    protect[:, :n_protect] = True
    return ids, protect


def main():
    checks = 0

    def ok(cond, msg):
        nonlocal checks
        assert cond, msg
        checks += 1

    t = torch.linspace(0, 1, 11)
    ok(abs(float(df.mask_prob(torch.tensor(0.0))) - df.EPS) < 1e-9 and abs(float(df.mask_prob(torch.tensor(1.0))) - 1) < 1e-9,
       "linear mask rate runs from EPS to 1")
    for sch in ("linear", "cosine"):
        p = df.mask_prob(t, sch)
        ok(bool((p[1:] > p[:-1]).all()) and float(p.min()) >= df.EPS - 1e-9 and float(p.max()) <= 1 + 1e-9, f"{sch}: monotone in [EPS, 1]")
        # weight equals p'/p numerically
        tt = torch.linspace(0.05, 0.95, 19, dtype=torch.float64)
        h = 1e-6
        dp = (df.mask_prob(tt + h, sch) - df.mask_prob(tt - h, sch)) / (2 * h)
        ok(torch.allclose(df.elbo_weight(tt, sch), dp / df.mask_prob(tt, sch), rtol=1e-4), f"{sch}: weight = p'/p")
        # E_t[w * p] = 1 - EPS (midpoint rule)
        grid = (torch.arange(200000, dtype=torch.float64) + 0.5) / 200000
        e = float((df.elbo_weight(grid, sch) * df.mask_prob(grid, sch)).mean())
        ok(abs(e - (1 - df.EPS)) < 1e-4, f"{sch}: E[w*p] = 1-EPS (got {e:.5f})")
    ok(abs(float(df.mask_prob(torch.tensor(0.5), "cosine")) - ((1 - df.EPS) * math.sin(math.pi / 4) + df.EPS)) < 1e-6,
       "cosine mask rate formula")
    try:
        df.mask_prob(t, "bogus")
        bad = False
    except ValueError:
        bad = True
    ok(bad, "unknown schedule rejected")

    # stratified t
    g = torch.Generator().manual_seed(0)
    B = 64
    allt = []
    for _ in range(200):
        tb = df.sample_t(B, g)
        ok(bool(((tb > 0) & (tb <= 1)).all()), "t in (0, 1]") if _ == 0 else None
        ok(torch.equal(torch.floor(tb.sort().values * B).long(), torch.arange(B)), "exactly one t per stratum") if _ < 3 else None
        allt.append(tb)
    allt = torch.cat(allt)
    ok(abs(float(allt.mean()) - 0.5) < 0.005, "stratified marginal mean ~ 0.5")
    hist = torch.histc(allt, bins=10, min=0, max=1) / allt.numel()
    ok(float((hist - 0.1).abs().max()) < 0.005, "stratified marginal is uniform across deciles")
    ok(not torch.equal(df.sample_t(B, g), df.sample_t(B, g)), "draws differ between calls")
    u = df.sample_t(5000, g, stratified=False)
    ok(bool(((u > 0) & (u <= 1)).all()), "unstratified t in (0, 1]")
    g1, g2 = torch.Generator().manual_seed(7), torch.Generator().manual_seed(7)
    ok(torch.equal(df.sample_t(32, g1), df.sample_t(32, g2)), "same seed -> same t")

    # corruption
    ids, protect = blocks(16, 128, 0)
    g = torch.Generator().manual_seed(1)
    tt = df.sample_t(16, g)
    noised, masked, maskable = df.corrupt(ids, protect, MASK, PAD, tt, "linear", g)
    ok(not bool((masked & protect).any()), "protected tokens are never masked")
    ok(not bool((masked & (ids == PAD)).any()), "[PAD] is never masked")
    ok(bool((noised[~masked] == ids[~masked]).all()) and bool((noised[masked] == MASK).all()), "only masked positions change, to [MASK]")
    ok(int(maskable.sum()) == 16 * (128 - 5 - 20), "maskable = not protected and not [PAD]")
    # empirical mask rate ~ p(t) per block
    many_ids, many_prot = blocks(2000, 128, 3)
    tfix = torch.full((2000,), 0.3)
    _, m2, mk2 = df.corrupt(many_ids, many_prot, MASK, PAD, tfix, "linear", torch.Generator().manual_seed(2))
    rate = float(m2.sum() / mk2.sum())
    ok(abs(rate - float(df.mask_prob(torch.tensor(0.3)))) < 0.005, f"empirical mask rate matches p(t) (got {rate:.4f})")
    ga, gb = torch.Generator().manual_seed(5), torch.Generator().manual_seed(5)
    ok(torch.equal(df.corrupt(ids, protect, MASK, PAD, tt, "linear", ga)[1], df.corrupt(ids, protect, MASK, PAD, tt, "linear", gb)[1]),
       "corruption is reproducible from the generator")

    # uniform model: loss -> (1 - EPS) * ln V for both schedules
    for sch in ("linear", "cosine"):
        g = torch.Generator().manual_seed(11)
        vals = []
        for i in range(60):
            ids, protect = blocks(64, 96, 100 + i, n_protect=11, n_pad=30)
            with torch.no_grad():
                vals.append(float(df.diffusion_loss(Uniform(), ids, protect, MASK, PAD, sch, g)))
        got, want = sum(vals) / len(vals), (1 - df.EPS) * math.log(V)
        ok(abs(got - want) < 0.05, f"{sch}: uniform model loss ~ (1-EPS) ln V = {want:.3f} (got {got:.3f})")

    # oracle model: loss ~ 0; gradient flows only when something is masked
    ids, protect = blocks(8, 64, 4)
    g = torch.Generator().manual_seed(3)
    with torch.no_grad():
        lo = float(df.diffusion_loss(Oracle(ids), ids, protect, MASK, PAD, "linear", g))
    ok(lo < 1e-6, f"a perfect predictor has ~0 loss (got {lo})")

    # a block with nothing maskable contributes 0 and no NaN
    ids, protect = blocks(4, 32, 6)
    protect[0, :] = True
    num, den, nm = df.loss_sums(Uniform(), ids, protect, MASK, PAD, torch.full((4,), 0.9), "linear", torch.Generator().manual_seed(8))
    ok(float(num[0]) == 0 and int(den[0]) == 0 and int(nm[0]) == 0, "fully protected block: zero loss, zero maskable")
    ok(bool(torch.isfinite(df.diffusion_loss(Uniform(), ids, protect, MASK, PAD, "linear", torch.Generator().manual_seed(9)))), "no NaN with an empty block")

    # randomness is isolated in the generator (global RNG untouched)
    ids, protect = blocks(8, 64, 12)
    before = torch.get_rng_state()
    with torch.no_grad():
        df.diffusion_loss(Uniform(), ids, protect, MASK, PAD, "linear", torch.Generator().manual_seed(1))
    ok(torch.equal(before, torch.get_rng_state()), "loss consumes only the explicit generator, not the global RNG")

    # gradient reaches a real model through masked positions only
    from dlm.model import DLMConfig, DiffusionTransformerLM
    m = DiffusionTransformerLM(DLMConfig.from_preset("tiny", vocab_size=V, seq_len=64))
    df.diffusion_loss(m, ids, protect, MASK, PAD, "linear", torch.Generator().manual_seed(2)).backward()
    ok(all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()), "loss backpropagates through the model")
    print(f"All {checks} diffusion checks passed.")


if __name__ == "__main__":
    main()
