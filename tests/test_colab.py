"""Tests for dlm/colab.py (fake google.colab module) and the structure of run_colab.ipynb:
    python run.py test  (or python tests/test_colab.py)"""
import json
import sys
import tempfile
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dlm import colab  # noqa: E402


def fake_colab(events, flush_fails=False):
    """Install a fake google.colab; every drive/runtime call is appended to `events`."""
    google, mod = types.ModuleType("google"), types.ModuleType("google.colab")
    drive, runtime = types.SimpleNamespace(), types.SimpleNamespace()

    def flush():
        if flush_fails:
            raise OSError("drive busy")
        events.append("flush_and_unmount")

    drive.flush_and_unmount = flush
    runtime.unassign = lambda: events.append("unassign")
    mod.drive, mod.runtime = drive, runtime
    google.colab = mod
    sys.modules["google"], sys.modules["google.colab"] = google, mod


def remove_colab():
    sys.modules.pop("google.colab", None)
    sys.modules.pop("google", None)


def main():
    checks = 0

    def ok(cond, msg):
        nonlocal checks
        assert cond, msg
        checks += 1

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "MyDrive" / "diffusion_lm_outputs"
        run = out / "checkpoints" / "pretrain_base"
        gif_dir = Path(tmp) / "content"
        gif_dir.mkdir()
        (gif_dir / "unmask_1.gif").write_bytes(b"GIF89a-one")
        (gif_dir / "unmask_2.gif").write_bytes(b"GIF89a-two")
        globs = (str(gif_dir / "unmask*.gif"),)
        logs, events, sleeps = [], [], []
        kw = dict(copy_globs=globs, sleep=sleeps.append, log=logs.append)

        # nothing trained yet: never end the session
        fake_colab(events)
        r = colab.finish_session(out, **kw)
        ok(not r["ended"] and events == [] and "model.pt" in r["reason"], "no final model: refuses, touches nothing")
        ok(any("NOT ending the session" in m for m in logs), "the refusal is printed")
        ok(sleeps == [], "no countdown when refusing")

        # model saved but no evaluation report
        (run / "final").mkdir(parents=True)
        (run / "final" / "model.pt").write_bytes(b"weights")
        r = colab.finish_session(out, **kw)
        ok(not r["ended"] and events == [] and "report.json" in r["reason"], "model but no report: refuses")
        ok(not (run / "eval").exists(), "a refusal copies nothing")

        # the report is optional when asked
        r = colab.finish_session(out, require_eval=False, grace_seconds=3, **kw)
        ok(r["ended"] and events == ["flush_and_unmount", "unassign"], "require_eval=False ends the session")
        ok(sleeps == [1, 1, 1], "a 3-second countdown sleeps 3 x 1 s")

        # everything in place: copies GIFs, countdown, flush BEFORE unassign
        events.clear(); sleeps.clear(); logs.clear()
        (run / "eval").mkdir(parents=True, exist_ok=True)
        (run / "eval" / "report.json").write_text("{}")
        r = colab.finish_session(out, grace_seconds=5, **kw)
        ok(r["ended"] and r["reason"] == "unassigned", "all saved: the session is ended")
        ok(events == ["flush_and_unmount", "unassign"], "Drive is flushed before the runtime is released")
        ok(sleeps == [1] * 5 and sum("Ending the session in" in m for m in logs) == 5, "countdown of grace_seconds one-second ticks")
        ok((run / "eval" / "unmask_1.gif").read_bytes() == b"GIF89a-one" and len(r["copied"]) == 2, "GIFs from /content are copied to Drive first")
        ok(any("Saved on Drive" in m for m in logs), "tells you what is saved")

        # disabled
        events.clear()
        r = colab.finish_session(out, enabled=False, **kw)
        ok(not r["ended"] and events == [] and "QUIT_WHEN_DONE" in r["reason"], "enabled=False never ends the session")

        # a cancelled countdown (the user stops the cell) never reaches unassign
        events.clear()

        def interrupted(_):
            raise KeyboardInterrupt

        try:
            colab.finish_session(out, grace_seconds=10, copy_globs=globs, sleep=interrupted, log=lambda m: None)
            cancelled = False
        except KeyboardInterrupt:
            cancelled = True
        ok(cancelled and events == [], "stopping the cell during the countdown cancels everything (no flush, no unassign)")

        # flush failure: keep the session so nothing is lost
        events.clear()
        fake_colab(events, flush_fails=True)
        r = colab.finish_session(out, **dict(kw, log=lambda m: None))
        ok(not r["ended"] and "flush" in r["reason"] and "unassign" not in events, "a failed Drive flush does not end the session")

        # not on Colab
        remove_colab()
        events.clear()
        r = colab.finish_session(out, **dict(kw, log=lambda m: None))
        ok(not r["ended"] and "not running on Colab" in r["reason"], "outside Colab it does nothing")

    # ---- the notebook itself
    nb = json.loads((ROOT / "run_colab.ipynb").read_text())
    cells = nb["cells"]
    code = [c for c in cells if c["cell_type"] == "code"]
    src = lambda c: "".join(c["source"])  # noqa: E731
    ok(nb["nbformat"] == 4 and all(isinstance(c["source"], list) for c in cells), "valid notebook structure")
    last = src(code[-1])
    ok("finish_session" in last and "QUIT_WHEN_DONE" in last and "GRACE_SECONDS" in last, "the LAST code cell ends the session")
    ok(cells[-1] is code[-1] or cells[-1]["cell_type"] == "markdown", "the quit cell comes after the evaluation and tests cells")
    order = [src(c) for c in code]
    real = next(i for i, s in enumerate(order) if "run.py pretrain --cache /content/tokens --out $OUT/checkpoints/pretrain_base" in s)
    evalc = next(i for i, s in enumerate(order) if s.startswith("# 10d"))
    quit_i = next(i for i, s in enumerate(order) if "finish_session" in s)
    ok(real < evalc < quit_i, "order: real run, then evaluation, then end of session")
    ok("--resume" in order[real], "the real-run cell always passes --resume")
    ok(any("RUN_SWEEP = False" in s for s in order) and any("if RUN_SWEEP" in s and "--sweep" in s for s in order),
       "the long sweep is opt-in (RUN_SWEEP = False by default)")
    ok(any("QUIT_WHEN_DONE = True" in s for s in order), "session ending is on by default")
    ok(not any("--sweep" in s and "RUN_SWEEP" not in s for s in order), "no cell runs the sweep unconditionally")
    ok("try:" in last and "except" in last, "an error in the quit cell keeps the session (never ends it by accident)")
    print(f"All {checks} Colab checks passed.")


if __name__ == "__main__":
    main()
