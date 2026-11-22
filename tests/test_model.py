"""Offline logic tests for dlm/model.py: python run.py test  (or python tests/test_model.py)"""
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dlm.model import DLMConfig, DiffusionTransformerLM, apply_rope, describe, load_model, rope_tables  # noqa: E402


def main():
    checks = 0

    def ok(cond, msg):
        nonlocal checks
        assert cond, msg
        checks += 1

    torch.manual_seed(0)
    base = DiffusionTransformerLM(DLMConfig())
    n = base.num_parameters()
    expect = 8192 * 512 + 10 * (4 * 512 * 512 + 3 * 512 * 1376 + 2 * 512) + 512
    ok(n == expect, f"base parameter count {n} == analytic {expect}")
    ok(35_000_000 < n < 37_000_000, "base is about 36M parameters")
    print("  base:", describe(base))
    ok(base.lm_head.weight is base.tok_emb.weight, "input/output embeddings are tied")
    ok(len(list(base.parameters())) == len({id(p) for p in base.parameters()}), "tied weight listed once")
    emb_std = float(base.tok_emb.weight.detach().std())
    ok(abs(emb_std - 0.02) < 0.002, f"embedding init std ~0.02 (got {emb_std:.4f})")
    ok(abs(float(base.blocks[0].attn.qkv.weight.detach().std()) - 0.02) < 0.002, "linear init std ~0.02")
    del base

    cfg = DLMConfig.from_preset("tiny", vocab_size=50, seq_len=32)
    model = DiffusionTransformerLM(cfg).eval()
    ids = torch.randint(5, 50, (3, 16))
    with torch.no_grad():
        logits = model(ids)
    ok(logits.shape == (3, 16, 50) and torch.isfinite(logits).all(), "logits shape and finite")

    # bidirectional: a later token changes earlier logits (a causal model would not)
    ids2 = ids.clone()
    ids2[:, -1] = (ids2[:, -1] % 40) + 6
    with torch.no_grad():
        l2 = model(ids2)
    ok(not torch.allclose(logits[:, 0], l2[:, 0], atol=1e-6), "attention is bidirectional (no causal mask)")

    # [PAD] keys are masked: extra padding does not change the logits at real positions
    real = torch.randint(5, 50, (2, 10))
    short = torch.cat([real, torch.zeros(2, 4, dtype=torch.long)], 1)
    long_ = torch.cat([real, torch.zeros(2, 12, dtype=torch.long)], 1)
    with torch.no_grad():
        a, b = model(short)[:, :10], model(long_)[:, :10]
    ok(torch.allclose(a, b, atol=1e-5), "[PAD] positions are ignored as attention keys")
    with torch.no_grad():  # sanity: not masking pads WOULD change them
        am = torch.ones_like(long_, dtype=torch.bool)
        c = model(long_, attention_mask=am)[:, :10]
    ok(not torch.allclose(a, c, atol=1e-5), "without the pad mask the logits would differ (the mask matters)")

    # RoPE: relative-position property of the dot product, and unit rotation
    cos, sin = rope_tables(16, 8, 10000.0)
    q = torch.randn(1, 1, 16, 8)
    r = apply_rope(q, cos, sin)
    ok(torch.allclose(r.norm(dim=-1), q.norm(dim=-1), atol=1e-5), "RoPE preserves vector norms")
    v = torch.randn(1, 1, 1, 8)
    qq, kk = v.expand(1, 1, 16, 8).contiguous(), v.expand(1, 1, 16, 8).contiguous()
    rq, rk = apply_rope(qq, cos, sin), apply_rope(kk, cos, sin)
    d1 = (rq[0, 0, 3] * rk[0, 0, 1]).sum()
    d2 = (rq[0, 0, 9] * rk[0, 0, 7]).sum()
    ok(torch.allclose(d1, d2, atol=1e-5), "RoPE score depends only on the position difference")

    # length limits, config validation, save/load
    try:
        model(torch.randint(5, 50, (1, 33)))
        raised = False
    except ValueError:
        raised = True
    ok(raised, "length beyond config.seq_len raises")
    ok(model(torch.randint(5, 50, (1, 5))).shape == (1, 5, 50), "shorter sequences work")
    try:
        DLMConfig(d_model=100, n_heads=8)
        bad = False
    except ValueError:
        bad = True
    ok(bad, "d_model not divisible by n_heads rejected")
    try:
        DLMConfig(d_model=60, n_heads=4)  # head dim 15 (odd)
        bad = False
    except ValueError:
        bad = True
    ok(bad, "odd head dimension rejected")
    with tempfile.TemporaryDirectory() as d:
        model.save_pretrained(d)
        loaded = load_model(d).eval()
        with torch.no_grad():
            ok(torch.equal(model(ids), loaded(ids)), "save/load round trip is exact")
        ok(DLMConfig.from_dict(loaded.config.to_dict()) == cfg, "config round trip")
        ok(loaded.lm_head.weight is loaded.tok_emb.weight, "tying survives loading")

    # gradient reaches every parameter
    model.train()
    model(ids).sum().backward()
    ok(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()), "all parameters get gradients")
    print(f"All {checks} model checks passed.")


if __name__ == "__main__":
    main()
