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

        # a notebook that only tests a model on the Hub has nothing to save: require_model=False
        events.clear(); sleeps.clear(); logs.clear()
        fake_colab(events)
        empty_out = Path(tmp) / "MyDrive2" / "diffusion_lm_outputs"
        r = colab.finish_session(empty_out, **kw)
        ok(not r["ended"] and "model.pt" in r["reason"] and events == [], "default: no trained model on Drive still refuses")
        r = colab.finish_session(empty_out, require_model=False, require_eval=False, grace_seconds=2, **kw)
        ok(r["ended"] and events == ["flush_and_unmount", "unassign"], "require_model=False + require_eval=False ends the session without a model on Drive")
        ok(any("Nothing to save" in m for m in logs) and not any("Saved on Drive" in m for m in logs), "it says there is nothing to save")
        ok((empty_out / "checkpoints" / "pretrain_base" / "eval" / "unmask_1.gif").is_file() and len(r["copied"]) == 2, "GIFs are still copied to Drive first")
        events.clear()
        no_gifs = dict(kw, copy_globs=(str(gif_dir / "nothing*.gif"),))
        other_out = Path(tmp) / "MyDrive3" / "diffusion_lm_outputs"
        r = colab.finish_session(other_out, require_model=False, require_eval=False, grace_seconds=0, **no_gifs)
        ok(r["ended"] and not (other_out / "checkpoints").exists(), "nothing to copy: no folders are created on Drive")
        r = colab.finish_session(empty_out, require_model=False, grace_seconds=0, run="no_such_run", **kw)
        ok(not r["ended"] and "report.json" in r["reason"], "require_model=False still honours require_eval")

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
    # ---- test_pretrained_colab.ipynb (tests the model published on Hugging Face)
    import contextlib
    import io
    import os
    import random
    import re
    import shutil

    import pyarrow as pa
    import pyarrow.parquet as pq
    import torch

    nb2 = json.loads((ROOT / "test_pretrained_colab.ipynb").read_text())
    c2 = [c for c in nb2["cells"] if c["cell_type"] == "code"]
    s2 = ["".join(c["source"]) for c in c2]
    all2 = "".join(s2) + "".join("".join(c["source"]) for c in nb2["cells"])
    ok(nb2["nbformat"] == 4 and all(isinstance(c["source"], list) for c in nb2["cells"]), "test notebook: valid structure")
    for i, s_ in enumerate(s2):
        compile(re.sub(r"^(\s*)[!%].*$", r"\1pass", s_, flags=re.M), f"cell{i}", "exec")
    ok(True, "test notebook: every code cell compiles")
    ok("finish_session" in s2[-1] and "try:" in s2[-1] and "except" in s2[-1], "test notebook: the LAST cell ends the session, guarded")
    ok("require_model=False" in s2[-1] and "require_eval=False" in s2[-1], "test notebook: ending the session needs no model on Drive (it is on the Hub)")
    ok('HF_REPO_ID = ""' in s2[0] and "QUIT_WHEN_DONE = True" in s2[0] and 'SCHEDULE = "cosine"' in s2[0], "test notebook: repo left blank, quits when done, cosine by default")
    ok(not any(re.search(r"hf_[A-Za-z0-9]{20,}", s_) for s_ in s2) and "token=" not in "".join(s2), "test notebook: no token inside")
    ok(not any("run.py pretrain" in s_ for s_ in s2), "test notebook: never trains")
    ok(not any(w in "".join(s2) for w in ("publish(", "upload_folder", "create_repo", "PUSH_TO_HUB")), "test notebook: never uploads or changes anything on the Hub")
    ok("bitlabsdb" not in all2, "test notebook: no personal Hub name committed")
    gens = [s_ for s_ in s2 if 'sh("generate"' in s_ or "--ckpt" in s_]
    ok(gens and all("LOCAL" not in s_ or "COMPARE_WITH_DRIVE" in s_ for s_ in gens if "run.py eval" in s_ or 'sh("generate"' in s_), "test notebook: generation and eval use the Hub download, not the Drive copy")
    ok(all('"--ckpt", CKPT' in s_ for s_ in s2 if 'sh("generate"' in s_) and not any("--hf" in s_ and "sh(" in s_ for s_ in s2), "test notebook: every generate call reads CKPT (the downloaded Hub folder)")
    ok(any("--out /content/hf_eval" in s_ for s_ in s2), "test notebook: eval writes outside the Hub download folder")

    # cells 4 and 6 for real, on a synthetic model: "uploaded" to a fake Hub (prepare_folder builds the exact Hub layout)
    sys.path.insert(0, str(ROOT / "tests"))
    import test_hub as th
    import dlm.download as p0
    import dlm.hub as hub
    import dlm.tokenizer as p1
    from dlm.model import DLMConfig, DiffusionTransformerLM
    cell3 = next(s_ for s_ in s2 if "def sh(" in s_)
    cell4 = next(s_ for s_ in s2 if s_.startswith("# 4."))
    cell6 = next(s_ for s_ in s2 if s_.startswith("# 6."))
    here = os.getcwd()
    os.chdir(ROOT)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rng = random.Random(0)
            words = "the a cat dog sun ran big small happy sad tree house lily tom said went home".split()
            train = [" ".join(rng.choice(words) for _ in range(rng.randint(8, 40))) + "." for _ in range(300)]
            data = root / "data"
            data.mkdir()
            pq.write_table(pa.table({"text": train}), data / "train.parquet")
            pq.write_table(pa.table({"text": train[:20]}), data / "validation.parquet")
            (data / p0.MANIFEST).write_text(json.dumps({
                "dataset": p0.DATASET, "revision": "abc123rev", "downloaded_at": "x",
                "splits": {k: {"stories": n, "empty": 0, "avg_chars": 1.0, "avg_words": 1.0, "file_bytes": 1}
                           for k, n in (("train", 300), ("validation", 20))}}))
            with contextlib.redirect_stdout(io.StringIO()):
                p1.build(data, root / "tokenizer", 320, 300, 0)
            V = json.loads((root / "tokenizer" / "tokenizer_manifest.json").read_text())["params"]["vocab_size"]
            run = root / "run"
            final = run / "final"
            torch.manual_seed(0)
            model = DiffusionTransformerLM(DLMConfig.from_preset("tiny", vocab_size=V, seq_len=96)).eval()
            model.save_pretrained(final)
            shutil.copytree(root / "tokenizer", final / "tokenizer")
            (final / "run_info.json").write_text(json.dumps({"recipe": dict(th.RECIPE, steps=25000, schedule="cosine"), "device": "cuda",
                                                             "precision": "bf16", "seconds": 1.0, "steps_per_second": 3.0}))
            (run / "eval").mkdir()
            (run / "eval" / "report.json").write_text(json.dumps(th.REPORT))
            (run / "eval" / "report.md").write_text("# report\n")
            (run / "eval" / "samples.md").write_text("# samples\n")
            with contextlib.redirect_stdout(io.StringIO()):
                hub_dir = hub.prepare_folder(run, root / "hub_download", "someone/tiny-model")["out"]
            ns = {"HF_REPO_ID": "someone/tiny-model", "SCHEDULE": "cosine", "COMPARE_WITH_DRIVE": True, "RUN": str(run), "LOCAL": str(final)}
            real_download = hub.download_snapshot
            asked = []
            hub.download_snapshot = lambda repo: asked.append(repo) or hub_dir  # the fake Hub: no network, no token
            try:
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    exec("import subprocess, sys\ndef sh(" + cell3.split("def sh(")[1], ns)
                    exec(cell4, ns)
                out4 = buf.getvalue()
                ok(asked == ["someone/tiny-model"] and ns["CKPT"] == hub_dir, "test notebook cell 4: downloads HF_REPO_ID and sets CKPT to that folder")
                ok("All required files are on the Hub" in out4 and "model.safetensors" in out4 and "Trained for 25000 steps" in out4, "test notebook cell 4: lists the files, confirms the upload is complete, shows the steps")
                ok("WARNING" not in out4 and "NOTE: the repo's generation_config" not in out4, "test notebook cell 4: a Hub repo whose generation_config.json matches SCHEDULE gets no warning")
                gc_path = Path(hub_dir) / "generation_config.json"
                good_gc = gc_path.read_text()
                gc_path.write_text(json.dumps({**json.loads(good_gc), "schedule": "linear"}))  # what an upload made before the fix says
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    exec(cell4, ns)
                ok("NOTE: the repo's generation_config.json says schedule = 'linear'" in buf.getvalue(), "test notebook cell 4: warns when the repo's generation_config.json says a different schedule")
                gc_path.write_text(good_gc)
                os.remove(Path(hub_dir) / "model.safetensors")
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    exec(cell4, ns)
                ok("MISSING from the Hub repo: ['model.safetensors']" in buf.getvalue(), "test notebook cell 4: a file missing on the Hub is reported")
                hub.prepare_folder(run, root / "hub_download", "someone/tiny-model")  # rebuild the complete folder
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    exec(cell6, ns)
                out6 = buf.getvalue()
                ok("largest difference = 0" in out6 and out6.count("IDENTICAL") == 2 and "RESULT: the Hub model is identical" in out6, "test notebook cell 6: Hub weights and greedy text equal the Drive model")
                # a Hub copy with different weights must be caught
                other = DiffusionTransformerLM(DLMConfig.from_preset("tiny", vocab_size=V, seq_len=96)).eval()
                torch.manual_seed(1)
                for prm in other.parameters():
                    torch.nn.init.normal_(prm, std=0.05) if prm.ndim >= 2 else None
                other.save_pretrained(root / "other")
                shutil.copytree(root / "tokenizer", root / "other" / "tokenizer")
                from dlm.model import save_safetensors
                save_safetensors(other, Path(hub_dir) / "model.safetensors")
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    exec(cell6, ns)
                ok("MISMATCH" in buf.getvalue() and "RESULT: the Hub model is identical" not in buf.getvalue(), "test notebook cell 6: a Hub model with different weights is flagged")
                with contextlib.redirect_stdout(io.StringIO()) as skip_buf:
                    exec(cell6, {**ns, "COMPARE_WITH_DRIVE": False})
                ok("Skipped" in skip_buf.getvalue(), "test notebook cell 6: skipped when COMPARE_WITH_DRIVE = False")
                with contextlib.redirect_stdout(io.StringIO()) as nodrive_buf:
                    exec(cell6, {**ns, "LOCAL": str(root / "no_final")})
                ok("Skipped: no trained model" in nodrive_buf.getvalue(), "test notebook cell 6: skipped when Drive has no final/")
            finally:
                hub.download_snapshot = real_download
            try:
                exec(cell4, {**ns, "HF_REPO_ID": ""})
                blank = False
            except SystemExit as e:
                blank = "Set HF_REPO_ID" in str(e)
            ok(blank, "test notebook cell 4: a blank HF_REPO_ID gives a clear message")

            def failing(repo):
                raise hub.HubError("could not download 'x/y' from the Hugging Face Hub (RepositoryNotFoundError)")

            hub.download_snapshot = failing
            try:
                exec(cell4, ns)
                failed = False
            except SystemExit as e:
                failed = "HF_TOKEN" in str(e) and "RepositoryNotFoundError" in str(e)
            finally:
                hub.download_snapshot = real_download
            ok(failed, "test notebook cell 4: a failed download explains the HF_TOKEN secret")
    finally:
        os.chdir(here)
    print(f"All {checks} Colab checks passed.")


if __name__ == "__main__":
    main()
