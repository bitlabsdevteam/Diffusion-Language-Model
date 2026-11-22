"""Offline logic tests for dlm/train.py (tiny model, synthetic blocks, CPU): python run.py test  (or python tests/test_train.py)"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dlm import train as tr  # noqa: E402
from dlm.model import DLMConfig, DiffusionTransformerLM, load_model  # noqa: E402

V, L = 40, 24


class FakeBlocks:
    """Memorisable synthetic blocks: block i is pattern i % 16; 3 protected tokens, 4 trailing [PAD]."""

    def __init__(self, n, offset=0):
        self.n, self.offset = n, offset
        rng = np.random.RandomState(0)
        self.patterns = rng.randint(5, V, size=(16, L)).astype(np.int64)

    def __len__(self):
        return self.n

    def arrays(self, i):
        ids = self.patterns[(i + self.offset) % 16].copy()
        ids[L - 4:] = 0
        protect = np.zeros(L, dtype=bool)
        protect[:3] = True
        return ids, protect, 1


def args(**kw):
    a = tr.build_parser().parse_args([])
    base = dict(size="tiny", steps=8, batch_size=8, lr=3e-3, warmup=4, seq_len=L, eval_every=4, ckpt_every=100,
                log_every=2, device="cpu", precision="fp32")
    for k, v in {**base, **kw}.items():
        setattr(a, k, v)
    return a


def cfg():
    return DLMConfig.from_preset("tiny", vocab_size=V, seq_len=L)


def run(out, **kw):
    return tr.run_training(args(**kw), cfg(), FakeBlocks(64), FakeBlocks(32, 3), FakeBlocks(16), Path(out))


def main():
    checks = 0

    def ok(cond, msg):
        nonlocal checks
        assert cond, msg
        checks += 1

    # learning-rate schedule
    f = lambda s: tr.lr_at(s, 2e-4, 1000, 40000, 0.1)  # noqa: E731
    ok(abs(f(0) - 2e-7) < 1e-12 and abs(f(499) - 1e-4) < 1e-12, "linear warmup")
    ok(abs(f(1000) - 2e-4) < 1e-12, "peak right after warmup")
    ok(abs(f(40000) - 2e-5) < 1e-12, "decays to 10% of peak")
    ok(all(f(s) >= f(s + 1000) for s in range(1000, 39000, 1000)), "monotone decay after warmup")
    ok(abs(f(500000) - 2e-5) < 1e-12, "stays at the floor past the end")

    # parameter groups: matrices decay, norms do not, tied weight once
    m = DiffusionTransformerLM(cfg())
    groups = tr.param_groups(m, 0.1)
    flat = [p for g in groups for p in g["params"]]
    ok(len(flat) == len({id(p) for p in flat}) == len(list(m.parameters())), "every parameter in exactly one group (tied weight once)")
    ok(all(p.ndim >= 2 for p in groups[0]["params"]) and all(p.ndim < 2 for p in groups[1]["params"]), "decay only on matrices")
    ok(groups[0]["weight_decay"] == 0.1 and groups[1]["weight_decay"] == 0.0, "weight decay values")
    ok(any(p is m.tok_emb.weight for p in groups[0]["params"]), "tied embedding is in the decay group")

    # data order: a pure function of (step, seed)
    n, B = 10, 4
    seq = [i for s in range(5) for i in tr.batch_indices(s, B, n, 0)]  # 20 slots = 2 epochs
    ok(sorted(seq[:10]) == list(range(10)) and sorted(seq[10:]) == list(range(10)), "each block exactly once per epoch")
    ok(tr.batch_indices(2, B, n, 0) == seq[8:12], "batch crossing an epoch boundary")
    ok(tr.batch_indices(3, B, n, 0) == tr.batch_indices(3, B, n, 0), "deterministic")
    ok(tr.batch_indices(0, B, n, 0) != tr.batch_indices(0, B, n, 1), "seed changes the order")
    ok(tr.evenly(100, 4) == [0, 25, 50, 75] and tr.evenly(3, 10) == [0, 1, 2], "evenly spaced evaluation indices")

    # evaluation: deterministic, and never touches the global RNG
    m = DiffusionTransformerLM(cfg())
    vb = FakeBlocks(32, 3)
    before = torch.get_rng_state()
    e1 = tr.evaluate(m, vb, list(range(32)), cfg(), "cpu", tr.make_amp("cpu", "fp32")[0])
    e2 = tr.evaluate(m, vb, list(range(32)), cfg(), "cpu", tr.make_amp("cpu", "fp32")[0])
    ok(e1 == e2, "evaluation is deterministic")
    ok(torch.equal(before, torch.get_rng_state()), "evaluation leaves the global RNG alone")
    # one evaluation is a noisy estimate (the 1/p weight has a heavy tail at small t: std ~0.1 nats at 600 tiny
    # blocks); the mean over evaluation seeds must match the theory (1-EPS) ln V for a uniform model
    class Uniform(torch.nn.Module):
        def forward(self, ids):
            return torch.zeros(*ids.shape, V)

    big = FakeBlocks(600, 3)
    amp0 = tr.make_amp("cpu", "fp32")[0]
    vals = [tr.evaluate(Uniform(), big, list(range(600)), cfg(), "cpu", amp0, seed=s) for s in range(40)]
    ok(abs(float(np.mean(vals)) - (1 - 1e-3) * np.log(V)) < 0.06,
       f"uniform model: mean evaluation over 40 seeds ~ (1-EPS) ln V (got {np.mean(vals):.3f})")
    ok(len(set(vals)) > 30, "different evaluation seeds give different draws")
    untrained = [tr.evaluate(m, big, list(range(600)), cfg(), "cpu", amp0, seed=s) for s in range(10)]
    ok(abs(float(np.mean(untrained)) - np.log(V)) < 0.25, f"untrained tiny model scores near ln V (got {np.mean(untrained):.3f})")
    m.train()
    tr.evaluate(m, vb, list(range(32)), cfg(), "cpu", tr.make_amp("cpu", "fp32")[0])
    ok(m.training, "evaluate restores train mode")

    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        # training learns the memorisable task; outputs exist; reload reproduces the evaluation
        res = run(d / "a", steps=400, eval_every=100, warmup=20)
        ev = [h for h in res["history"] if "val" in h]
        ok(len(ev) == 4 and ev[0]["step"] == 100, "evaluated every 100 steps")
        ok(ev[-1]["val"] < 0.5 * np.log(V), f"validation bound falls well below ln V ({ev[-1]['val']:.3f}; a plateau of ~100 steps first)")
        ok(ev[-1]["val"] < ev[0]["val"], "validation bound keeps falling")
        # monitoring: epoch, live files, TensorBoard, plot command
        trs = [h for h in res["history"] if "loss" in h]
        ok(all(abs(h["epoch"] - h["step"] * 8 / 64) < 1e-9 for h in trs), "epoch = step * batch / train blocks, on every log entry")
        ok(all(h["epoch"] == trs[i]["epoch"] or h["epoch"] > trs[i]["epoch"] for i, h in enumerate(trs[1:], 0)), "epoch never decreases")
        ok(abs(trs[-1]["epoch"] - 50.0) < 1e-9, "400 steps of 8 blocks over 64 blocks = 50 epochs")
        ok(all("epoch" in h for h in res["history"] if "val" in h), "evaluation entries carry the epoch")
        ok(all({"lr", "grad_norm", "loss"} <= h.keys() for h in trs), "log entries hold loss, lr and grad norm")
        a_dir = d / "a"
        ok((a_dir / "history.json").is_file() and (a_dir / "training_curves.png").is_file(), "live history.json + curves in the run folder")
        png = (a_dir / "training_curves.png").read_bytes()
        ok(png[:8] == b"\x89PNG\r\n\x1a\n" and len(png) > 20_000, "training_curves.png is a real PNG with content")
        ok(len(json.loads((a_dir / "history.json").read_text())) == len(res["history"]), "live history.json is complete after the run")
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        acc = EventAccumulator(str(a_dir / "tb"))
        acc.Reload()
        tags = set(acc.Tags()["scalars"])
        want = {"train/cross_entropy_loss", "train/learning_rate", "train/epoch", "train/grad_norm", "eval/validation", "eval/train_eval", "eval/gap"}
        ok(want <= tags, f"TensorBoard has loss, learning rate, epoch, grad norm and eval scalars (missing {want - tags})")
        ok(len(acc.Scalars("train/epoch")) == len(trs) and acc.Scalars("train/epoch")[-1].value == trs[-1]["epoch"], "TensorBoard epoch matches history")
        png_before = (a_dir / "training_curves.png").stat().st_mtime_ns
        ok(tr.plot_main(["--out", str(a_dir)]) == 0 and (a_dir / "training_curves.png").stat().st_mtime_ns >= png_before, "`plot` redraws the curves from history.json")
        ok(tr.plot_main(["--out", str(d / "does_not_exist")]) == 1, "`plot` on an empty folder exits 1")
        ok(tr.moving_average([1, 2, 3, 4], 2) == [1.0, 1.5, 2.5, 3.5], "moving average")
        ok(tr.fmt_time(75) == "1m15s" and tr.fmt_time(3725) == "1h02m", "ETA formatting")
        quiet = run(d / "quiet", steps=4, eval_every=4, no_tensorboard=True)
        ok(not (d / "quiet" / "tb").exists() and (d / "quiet" / "training_curves.png").is_file(), "--no-tensorboard skips tb/, curves still written")

        fin = d / "a" / "final"
        ok(all((fin / f).exists() for f in ("model.pt", "config.json", "history.json", "run_info.json")), "final/ contents")
        info = json.loads((fin / "run_info.json").read_text())
        ok(info["parameters"] > 0 and info["steps_per_second"] > 0 and info["recipe"]["lr"] == 3e-3, "run_info.json")
        ok(len(json.loads((fin / "history.json").read_text())) == len(res["history"]), "history.json saved")
        amp = tr.make_amp("cpu", "fp32")[0]
        idx = list(range(len(FakeBlocks(32, 3))))
        trained = tr.evaluate(res["model"], FakeBlocks(32, 3), idx, cfg(), "cpu", amp)
        reloaded = tr.evaluate(load_model(fin), FakeBlocks(32, 3), idx, cfg(), "cpu", amp)
        ok(trained == reloaded == res["full_val"], "a reloaded checkpoint reproduces the evaluation loss exactly")

        # resume is exact: 8 uninterrupted steps == 4 steps + resume + 4 steps
        full = run(d / "full", steps=8, eval_every=4)
        part = run(d / "part", steps=8, eval_every=4, stop_at=4)
        ok(part["final"] is False and (d / "part" / "resume.pt").exists(), "--stop-at saves resume.pt and no final")
        ok((d / "part" / "history.json").is_file() and (d / "part" / "training_curves.png").is_file(), "curves are available before the run finishes")
        done = run(d / "part", steps=8, eval_every=4, resume=True)
        same = all(torch.equal(p, q) for p, q in zip(full["model"].state_dict().values(), done["model"].state_dict().values()))
        ok(same, "resumed run matches the uninterrupted run bit for bit")
        ok(full["history"] == done["history"], "identical loss / evaluation history after resume")
        ok(full["full_val"] == done["full_val"], "identical final evaluation after resume")

        # resume refuses a changed recipe
        try:
            run(d / "part", steps=8, eval_every=4, resume=True, lr=1e-3)
            refused = False
        except SystemExit as e:
            refused = "lr" in str(e)
        ok(refused, "resume refuses when the recipe differs")
        run(d / "fresh", steps=4, eval_every=4, resume=True)  # --resume with nothing to resume just starts
        ok((d / "fresh" / "final" / "model.pt").exists(), "--resume without resume.pt starts from scratch")

    # best/ checkpoint (no early stopping: the run always reaches --steps)
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        tok = d / "tok"
        tok.mkdir()
        (tok / "tokenizer.json").write_text("{}")
        res = tr.run_training(args(steps=12, eval_every=4), cfg(), FakeBlocks(64), FakeBlocks(32, 3), FakeBlocks(16),
                              d / "b", tokenizer_dir=tok)
        ok(res["final"] and res["step"] == 12, "the run is never cut short by the best-checkpoint logic (no early stopping)")
        bdir = d / "b" / "best"
        evs = [h for h in res["history"] if "val" in h]
        top = min(evs, key=lambda h: h["val"])
        info = json.loads((bdir / "best.json").read_text())
        ok(all((bdir / f).exists() for f in ("model.pt", "config.json", "best.json", "tokenizer/tokenizer.json")), "best/ has the final/ layout plus best.json")
        ok(info["step"] == top["step"] and info["val"] == top["val"], "best.json = the lowest validation evaluation of the run")
        ok(not (d / "b" / "best.tmp").exists() and not (d / "b" / "best.old").exists(), "no temporary folders are left behind")
        ok(json.loads((d / "b" / "final" / "run_info.json").read_text())["best"]["step"] == top["step"], "run_info.json records the best step")
        idx = list(range(len(FakeBlocks(32, 3))))
        amp = tr.make_amp("cpu", "fp32")[0]
        bm = load_model(bdir)
        ok(tr.evaluate(bm, FakeBlocks(32, 3), idx, cfg(), "cpu", amp) > 0, "best/ loads with load_model")
        # the saved weights are the weights AT the best step: re-evaluating on the 256-block protocol reproduces `val`
        val_idx = tr.evenly(32, tr.EVAL_BLOCKS)
        ok(tr.evaluate(bm, FakeBlocks(32, 3), val_idx, cfg(), "cpu", amp) == top["val"], "best/ weights reproduce the best validation number exactly")

        # a worse evaluation never replaces best/: stop at 4, resume with a poisoned best that is unbeatable
        run(d / "keep", steps=8, eval_every=4, stop_at=4)
        keep_info = tr.read_best(d / "keep")
        (d / "keep" / "best" / "best.json").write_text(json.dumps({**keep_info, "val": -1.0, "step": 4}))
        mt = (d / "keep" / "best" / "model.pt").stat().st_mtime_ns
        run(d / "keep", steps=8, eval_every=4, resume=True)
        ok((d / "keep" / "best" / "model.pt").stat().st_mtime_ns == mt and tr.read_best(d / "keep")["val"] == -1.0, "resume restores best from best.json and does not overwrite it with a worse evaluation")
        # an incomplete best/ (no model.pt) counts as no best
        (d / "keep" / "best" / "model.pt").unlink()
        ok(tr.read_best(d / "keep") is None, "a best/ without model.pt is ignored")
        # an interrupted save leaves the old best/ intact
        class Boom:
            def save_pretrained(self, _):
                raise OSError("disk full")
        before = (d / "b" / "best" / "model.pt").read_bytes()
        try:
            tr.save_best(Boom(), d / "b", None, {"step": 0, "val": 0.0})
        except OSError:
            pass
        ok((d / "b" / "best" / "model.pt").read_bytes() == before, "a failed write keeps the previous best/")

    # export: an unfinished run's resume.pt becomes a loadable folder
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        tok = d / "tok"
        tok.mkdir()
        (tok / "tokenizer.json").write_text("{}")
        part = run(d / "p", steps=8, eval_every=4, stop_at=4)
        exp = tr.export_checkpoint(d / "p" / "resume.pt", tokenizer_dir=tok)
        ok(exp == d / "p" / "export_step4" and all((exp / f).exists() for f in ("model.pt", "config.json", "history.json", "export.json", "tokenizer/tokenizer.json")), "export writes a final/-style folder next to resume.pt")
        info = json.loads((exp / "export.json").read_text())
        ok(info["step"] == 4 and info["steps_planned"] == 8 and info["finished"] is False, "export.json says it is unfinished")
        amp = tr.make_amp("cpu", "fp32")[0]
        val_idx = tr.evenly(32, tr.EVAL_BLOCKS)
        v = tr.evaluate(load_model(exp), FakeBlocks(32, 3), val_idx, cfg(), "cpu", amp)
        ok(v == [h for h in part["history"] if "val" in h][-1]["val"] == info["last_val_bound"], "the exported weights reproduce the evaluation logged at that step exactly")
        ri = json.loads((exp / "run_info.json").read_text())
        ok(ri["stopped_at_step"] == 4 and ri["recipe"]["steps"] == 8, "export writes run_info.json marking the run as stopped early")
        ok(tr.export_main(["--resume", str(d / "p" / "resume.pt"), "--out", str(d / "o"), "--tokenizer-dir", str(tok)]) == 0 and (d / "o" / "model.pt").is_file(), "export CLI with --out")
        ok(tr.export_main(["--resume", str(d / "missing.pt")]) == 1, "export CLI: missing file -> exit 1")

    # CLI behaviour
    try:
        tr.main(["--task", "sft", "--init-from", "x"])
        nb = False
    except NotImplementedError as e:
        nb = "Phase 8" in str(e)
    ok(nb, "--task sft still reports Phase 8 as not built")
    try:
        tr.main(["--task", "sft"])
        usage = False
    except SystemExit as e:
        usage = e.code == 2
    ok(usage, "--task sft without --init-from is a usage error")
    ok(tr.main(["--cache", "/nonexistent/cache"]) == 1, "missing token cache -> exit 1")
    print(f"All {checks} training checks passed.")


if __name__ == "__main__":
    main()
