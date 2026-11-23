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


_MISSING = object()
# On a real Colab the real `google` / `google.colab` are installed: remember them so the tests can put them back.
_ORIGINAL = {k: sys.modules.get(k, _MISSING) for k in ("google", "google.colab")}


def restore_modules():
    for name, mod in _ORIGINAL.items():
        if mod is _MISSING:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = mod


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
    """Make `from google.colab import ...` raise ImportError, even where the real google.colab is installed.

    Popping the entries is not enough: on Colab the import would then find the REAL module, and `unassign()` would
    be called for real (it fails outside a notebook kernel, and inside one it would end the user's session)."""
    restore_modules()
    sys.modules["google.colab"] = None  # None in sys.modules = "import blocked"


def main():
    try:
        _main()
    finally:
        restore_modules()  # never leave a fake (or a block) behind for later code in this process


def _main():
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
    real = next(i for i, s in enumerate(order) if "run.py pretrain --cache /content/tokens --out $RUN" in s)
    evalc = next(i for i, s in enumerate(order) if s.startswith("# 10d"))
    quit_i = next(i for i, s in enumerate(order) if "finish_session" in s)
    ok(real < evalc < quit_i, "order: real run, then evaluation, then end of session")
    ok("--resume" in order[real], "the real-run cell always passes --resume")
    ok("--steps $STEPS" in order[real] and "--schedule $NOISE_SCHEDULE" in order[real], "the real run takes STEPS and NOISE_SCHEDULE from cell 1")
    ok(all(f"{n} = " in order[0] for n in ("STEPS", "NOISE_SCHEDULE", "RUN_NAME")) and "STEPS = 25_000" in order[0] and 'NOISE_SCHEDULE = "cosine"' in order[0], "cell 1 has STEPS / NOISE_SCHEDULE / RUN_NAME (25,000 steps, cosine)")
    ok(not any("pretrain_base" in s_ for s_ in order), "no cell hard-codes the old run folder: everything uses RUN")
    ok("run=RUN_NAME" in last, "the end-of-session cell checks the same run folder")
    gen = next(s_ for s_ in order if s_.startswith("# 10b"))
    ok(gen.count("--schedule $NOISE_SCHEDULE") == 2, "generation uses the training noise schedule")
    ok(any("RUN_SWEEP = False" in s for s in order) and any("if RUN_SWEEP" in s and "--sweep" in s for s in order),
       "the long sweep is opt-in (RUN_SWEEP = False by default)")
    ok(any("QUIT_WHEN_DONE = True" in s for s in order), "session ending is on by default")
    ok(not any("--sweep" in s and "RUN_SWEEP" not in s for s in order), "no cell runs the sweep unconditionally")
    ok("try:" in last and "except" in last, "an error in the quit cell keeps the session (never ends it by accident)")
    # ---- test_pretrained_colab.ipynb (the inference-test notebook)
    import contextlib
    import io
    import os
    import re
    nb2 = json.loads((ROOT / "test_pretrained_colab.ipynb").read_text())
    c2 = [c for c in nb2["cells"] if c["cell_type"] == "code"]
    s2 = ["".join(c["source"]) for c in c2]
    ok(nb2["nbformat"] == 4 and all(isinstance(c["source"], list) for c in nb2["cells"]), "test notebook: valid structure")
    for i, s_ in enumerate(s2):
        compile(re.sub(r"^(\s*)[!%].*$", r"\1pass", s_, flags=re.M), f"cell{i}", "exec")
    ok(True, "test notebook: every code cell compiles")
    ok("finish_session" in s2[-1] and "try:" in s2[-1] and "except" in s2[-1], "test notebook: the LAST cell ends the session, guarded")
    ok("QUIT_WHEN_DONE = True" in s2[0] and "PUSH_TO_HUB = False" in s2[0] and "HF_PRIVATE = True" in s2[0] and "HF_DRY_RUN = True" in s2[0], "test notebook: safe defaults (quits when done, no upload, private, dry run)")
    ok(not any(re.search(r"hf_[A-Za-z0-9]{20,}", s_) for s_ in s2) and "token=" not in "".join(s2), "test notebook: no token inside")
    ok(not any("pretrain" in s_ and "run.py pretrain" in s_ for s_ in s2), "test notebook: never trains")
    pub = next(s_ for s_ in s2 if "publish(" in s_)
    ok("if not PUSH_TO_HUB" in pub and "dry_run=HF_DRY_RUN" in pub and "except" in pub, "test notebook: publishing is skipped by default, honours the dry run, never raises")
    ok("hf_" + "bitlabsdb" not in "".join(s2) and "bitlabsdb/" not in "".join(s2), "test notebook: no personal Hub name committed")

    # cell 4 for real: a stopped run (resume.pt only) is exported to final/, a finished one is left alone
    sys.path.insert(0, str(ROOT / "tests"))
    import test_train as tt
    cell3, cell4 = next(s for s in s2 if "def sh(" in s), next(s for s in s2 if s.startswith("# 4."))
    here = os.getcwd()
    os.chdir(ROOT)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "pretrain_base"
            tt.run(run_dir, steps=8, eval_every=4, stop_at=4)
            ns = {"RUN": str(run_dir), "CKPT": str(run_dir / "final")}
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                exec("import subprocess, sys\ndef sh(" + cell3.split("def sh(")[1], ns)  # only the helper: the pip step is not run here
                exec(cell4, ns)
            out = buf.getvalue()
            ok((run_dir / "final" / "model.pt").is_file() and "exporting resume.pt" in out and "stopped_at_step" in out, "test notebook cell 4: resume.pt only -> final/ is made and the stop step is shown")
            before = (run_dir / "final" / "model.pt").stat().st_mtime_ns
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                exec(cell4, ns)
            ok("Using the model in" in buf.getvalue() and (run_dir / "final" / "model.pt").stat().st_mtime_ns == before, "test notebook cell 4: an existing final/ is used, not rewritten")
            empty = {"RUN": str(Path(tmp) / "nothing"), "CKPT": str(Path(tmp) / "nothing" / "final")}
            try:
                exec(cell4, {**ns, **empty})
                stopped = False
            except SystemExit as e:
                stopped = "Nothing to test" in str(e)
            ok(stopped, "test notebook cell 4: nothing to test -> a clear message")
    finally:
        os.chdir(here)
    print(f"All {checks} Colab checks passed.")


if __name__ == "__main__":
    main()
