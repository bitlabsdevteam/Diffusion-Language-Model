"""Offline tests for dlm/hub.py: the Hugging Face upload folder, model card, token handling and a fake Hub API:
    python run.py test  (or python tests/test_hub.py)"""
import contextlib
import io
import json
import os
import random
import shutil
import sys
import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import dlm.download as p0  # noqa: E402
import dlm.hub as hub  # noqa: E402
import dlm.sample as sm  # noqa: E402
import dlm.tokenizer as p1  # noqa: E402
from dlm.model import DLMConfig, DiffusionTransformerLM, load_model  # noqa: E402

REPORT = {
    "settings": {"max_val_blocks": None},
    "likelihood": {"nats_per_token": 2.345, "bits_per_char": 0.789, "blocks": 11451, "draws": 4,
                   "unigram_baseline_nats_per_token": 5.923},
    "headline": {"unconditional": {"stats": {"n": 200, "finished_rate": 0.93, "length": {"median": 190, "p90": 340},
                                            "distinct_2": 0.51, "repetition_rate": 0.04}, "gen_ppl": 21.5},
                 "reference_real_stories": {"stats": {"n": 200, "finished_rate": 1.0, "length": {"median": 188, "p90": 376},
                                                      "distinct_2": 0.536, "repetition_rate": 0.15}, "gen_ppl": 9.8}}}
RECIPE = {"size": "base", "steps": 20000, "batch_size": 32, "lr": 0.0002, "warmup": 1000, "min_lr_ratio": 0.1,
          "weight_decay": 0.1, "beta1": 0.9, "beta2": 0.999, "clip": 1.0, "seq_len": 512, "seed": 42, "schedule": "linear"}


class FakeApi:
    def __init__(self, calls, fail=False):
        self.calls, self.fail = calls, fail

    def create_repo(self, **kw):
        self.calls.append(("create_repo", kw))

    def upload_folder(self, **kw):
        if self.fail:
            raise ConnectionError("network down")
        kw["files"] = sorted(str(p.relative_to(kw["folder_path"])) for p in Path(kw["folder_path"]).rglob("*") if p.is_file())
        self.calls.append(("upload_folder", kw))


def main():
    checks = 0

    def ok(cond, msg):
        nonlocal checks
        assert cond, msg
        checks += 1

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        # a synthetic tokenizer
        rng = random.Random(0)
        words = "the a cat dog sun ran big small happy sad tree house lily tom said went home".split()
        train = [" ".join(rng.choice(words) for _ in range(rng.randint(8, 40))) + "." for _ in range(300)]
        data = root / "data"
        data.mkdir()
        pq.write_table(pa.table({"text": train}), data / "train.parquet")
        pq.write_table(pa.table({"text": train[:20]}), data / "validation.parquet")
        (data / p0.MANIFEST).write_text(json.dumps({
            "dataset": p0.DATASET, "revision": "abc123rev", "downloaded_at": "x",
            "splits": {s: {"stories": n, "empty": 0, "avg_chars": 1.0, "avg_words": 1.0, "file_bytes": 1}
                       for s, n in (("train", 300), ("validation", 20))}}))
        p1.build(data, root / "tokenizer", 320, 300, 0)
        V = json.loads((root / "tokenizer" / "tokenizer_manifest.json").read_text())["params"]["vocab_size"]

        # a synthetic finished run
        run = root / "run"
        final = run / "final"
        torch.manual_seed(0)
        model = DiffusionTransformerLM(DLMConfig.from_preset("tiny", vocab_size=V, seq_len=96)).eval()
        model.save_pretrained(final)
        shutil.copytree(root / "tokenizer", final / "tokenizer")
        (final / "run_info.json").write_text(json.dumps({"recipe": RECIPE, "device": "cuda", "precision": "bf16",
                                                         "seconds": 6300.0, "steps_per_second": 3.17}))
        (final / "history.json").write_text(json.dumps([{"step": 100, "epoch": 0.003, "loss": 5.0}, {"step": 20000, "epoch": 0.57, "loss": 2.0}]))
        (final / "training_curves.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 64)
        (run / "eval").mkdir()
        (run / "eval" / "report.json").write_text(json.dumps(REPORT))
        (run / "eval" / "report.md").write_text("# report\n")
        (run / "eval" / "samples.md").write_text("# samples\n")

        # ---- the upload folder
        out = root / "hf"
        info = hub.prepare_folder(run, out, "someone/dlm-tinystories-36m", "apache-2.0")
        names = {rel for rel, _ in info["files"]}
        want = {"README.md", "config.json", "model.safetensors", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
                "tokenizer_manifest.json", "training_curves.png", "run_info.json", "eval/report.md", "eval/samples.md"}
        ok(want <= names, f"upload folder has the standard files (missing {want - names})")
        ok("model.pt" not in names and hub.MARKER not in names, "no pickle weights and no marker file are listed for upload")

        # ---- weights: safetensors == the original pt, tied weights, loadable from the Hub layout
        from safetensors.torch import load_file
        st = load_file(str(out / "model.safetensors"))
        sd = torch.load(final / "model.pt")
        shared = [k for k in sd if k not in st]
        ok(len(shared) <= 1 and all(torch.equal(st[k], sd[k]) for k in st), "model.safetensors holds exactly the trained weights (tied weight stored once)")
        loaded = load_model(out).eval()
        ids = torch.randint(5, V, (2, 20))
        with torch.no_grad():
            ok(torch.equal(model(ids), loaded(ids)), "load_model reads the Hub layout: identical logits")
        ok(loaded.lm_head.weight is loaded.tok_emb.weight, "tied embeddings survive safetensors")
        m2, tok2 = sm.load_checkpoint(out)
        ok(tok2.get_vocab_size() == V and m2.config.vocab_size == V, "load_checkpoint reads the Hub layout (tokenizer.json at the top level)")
        res = sm.generate_text(m2, tok2, "", seed=0, max_new_tokens=12, temperature=0)
        ok(res["forward_calls"] > 0, "generation works from the Hub layout")

        # ---- config.json and generation_config.json
        cfg = json.loads((out / "config.json").read_text())
        ok(cfg["model_type"] == "dlm" and cfg["architectures"] == ["DiffusionTransformerLM"], "config.json has model_type and architectures")
        ok(all(cfg[k] == v for k, v in model.config.to_dict().items()), "config.json keeps every architecture field")
        gc = json.loads((out / "generation_config.json").read_text())
        ok(gc["max_new_tokens"] == 480 and gc["block_length"] == 32 and gc["temperature"] == 0.8 and "<|end|>" in gc["stop_tokens"], "generation_config.json = the sampler defaults")
        ok(gc["schedule"] == "linear", "generation_config.json: a linear-trained run gets schedule linear")
        # the sampler's default schedule follows the schedule the model was trained with
        run_c = root / "run_cosine"
        shutil.copytree(run, run_c)
        (run_c / "final" / "run_info.json").write_text(json.dumps({"recipe": dict(RECIPE, schedule="cosine"), "device": "cuda", "precision": "bf16",
                                                                   "seconds": 6300.0, "steps_per_second": 3.17}))
        out_c = Path(hub.prepare_folder(run_c, root / "hub_cosine", "u/cosine-model")["out"])
        gc_c = json.loads((out_c / "generation_config.json").read_text())
        ok(gc_c["schedule"] == "cosine" and {k: v for k, v in gc_c.items() if k != "schedule"} == {k: v for k, v in gc.items() if k != "schedule"},
           "generation_config.json: a cosine-trained run gets schedule cosine and every other default unchanged")
        ok("`cosine` unmasking schedule" in (out_c / "README.md").read_text() and "`linear` unmasking schedule" in (out / "README.md").read_text(),
           "the model card names the default unmasking schedule")
        ok(hub.sampling_schedule(None) == hub.sampling_schedule({}) == hub.sampling_schedule({"recipe": {}}) == hub.sampling_schedule({"recipe": {"schedule": "weird"}}) == "linear",
           "no run_info, no recipe or an unknown schedule falls back to linear")

        # ---- the model card
        from huggingface_hub import ModelCard
        card = ModelCard.load(out / "README.md")
        d = card.data
        ok(d.license == "apache-2.0" and d.language == "en" and d.library_name == "dlm" and d.pipeline_tag == "text-generation",
           "card metadata: license, language, library_name, pipeline_tag")
        ok("roneneldan/TinyStories" in d.datasets and "diffusion" in d.tags and "safetensors" in d.tags, "card metadata: datasets and tags")
        body = (out / "README.md").read_text()
        ok(body.startswith("---\n") and "\n---\n" in body[4:], "the YAML header is delimited by ---")
        for section in ("## Model description", "## Intended uses & limitations", "## Training data", "## Training procedure",
                        "## Evaluation results", "## How to use", "## References"):
            ok(section in body, f"card section {section!r}")
        ok("2.345" in body and "0.789" in body and "5.923" in body and "9.8" in body, "card reports the evaluation numbers and the baseline")
        ok("20,000" in body and "0.57" in body and "bf16" in body and "1.75 h" in body, "card reports steps, epochs, precision and training time")
        resumed = root / "run_resumed"
        shutil.copytree(run, resumed)
        ri = json.loads((resumed / "final" / "run_info.json").read_text())
        ri["seconds"] = 1000.0  # a run resumed after a disconnect only records its LAST session
        (resumed / "final" / "run_info.json").write_text(json.dumps(ri))
        hub.prepare_folder(resumed, root / "hf_resumed", "someone/r", "apache-2.0")
        rb = (root / "hf_resumed" / "README.md").read_text()
        ok("about 1.75 h" in rb and "0.28 h" not in rb and "estimate" in rb, "the training time is estimated from steps / speed, not the last session's seconds")
        ok("abc123rev" in body and "roneneldan/TinyStories" in body and "CDLA-Sharing-1.0" in body, "card names the dataset revision and its licence")
        ok("someone/dlm-tinystories-36m" in body and "run.py generate --hf someone/dlm-tinystories-36m" in body, "usage uses the repo id")
        ok("not a `transformers` model" in body and "AutoModel" in body, "card says AutoModel does not apply")
        ok("synthetic" in body and "GPT-3.5" in body, "card states the data is synthetic text from GPT-3.5/4")
        ok("arxiv.org/abs/2502.09992" in body and "2305.07759" in body, "card cites LLaDA and TinyStories")
        ok(not any(s in body for s in ("/Users/", "/content/", "/private/", "hf_", "gho_")), "no local paths or token-like strings in the card")
        ok(len(d.tags) == len(set(d.tags)), "no duplicate tags")
        rows = [ln for ln in body.splitlines() if ln.startswith("|")]
        ok(rows and all(ln.rstrip().endswith("|") for ln in rows), "every Markdown table row is closed (no text after the last |)")
        ok(any("20,000 / 32 blocks of 512 tokens (about 0.57 epochs)" in ln for ln in rows), "steps, batch size and epochs share one table row")
        ok("all 11,451 validation blocks" in body, "a full evaluation is described as the whole validation set")
        no_eval = root / "run2"
        shutil.copytree(run, no_eval)
        shutil.rmtree(no_eval / "eval")
        hub.prepare_folder(no_eval, root / "hf2", "someone/x", "mit")
        b2 = (root / "hf2" / "README.md").read_text()
        ok("No evaluation report was available" in b2 and "nats per token" not in b2 and ModelCard.load(root / "hf2" / "README.md").data.license == "mit",
           "without a report the card says so (no invented numbers) and uses the chosen licence")
        ok(not (root / "hf2" / "eval").exists(), "no eval folder when there is no report")
        sub = root / "run3"
        shutil.copytree(run, sub)
        (sub / "eval" / "report.json").write_text(json.dumps({**REPORT, "settings": {"max_val_blocks": 400}}))
        hub.prepare_folder(sub, root / "hf3", "someone/y", "apache-2.0")
        b3 = (root / "hf3" / "README.md").read_text()
        ok("A subset of the TinyStories validation set" in b3 and "all 11,451" not in b3, "a subset evaluation is not described as the whole set")

        # ---- refusals
        for bad, why in ((lambda: hub.prepare_folder(root / "nope", root / "h3", "a/b"), "an unfinished run"),
                         (lambda: hub.publish(run, "noslash", dry_run=True), "a repo id without a slash"),
                         (lambda: hub.publish(run, "a/b c", dry_run=True), "a repo id with a space"),
                         (lambda: hub.publish(run, "a/b", license="", dry_run=True), "an empty licence")):
            try:
                bad()
                raised = False
            except hub.HubError:
                raised = True
            ok(raised, f"HubError for {why}")
        stray = root / "stray"
        stray.mkdir()
        (stray / "important.txt").write_text("keep me")
        try:
            hub.prepare_folder(run, stray, "a/b")
            refused = False
        except hub.HubError:
            refused = True
        ok(refused and (stray / "important.txt").read_text() == "keep me", "a foreign non-empty --out folder is never cleared")
        hub.prepare_folder(run, out, "someone/again")  # our own folder (marker) is rebuilt in place
        ok("someone/again" in (out / "README.md").read_text(), "a folder created by this tool is rebuilt")

        # ---- publishing with a fake Hub API
        calls = []
        saved_env = {k: os.environ.pop(k, None) for k in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")}
        import huggingface_hub
        saved_get_token = huggingface_hub.get_token
        try:
            huggingface_hub.get_token = lambda: None
            ok(hub.get_token() is None, "no token anywhere -> None")
            r = hub.publish(run, "someone/m", dry_run=True, out_dir=root / "d1", api_factory=lambda t: FakeApi(calls), log=lambda m: None)
            ok(calls == [] and not r["uploaded"] and r["url"] is None, "dry run: no API call, no token needed")
            try:
                hub.publish(run, "someone/m", out_dir=root / "d2", api_factory=lambda t: FakeApi(calls), log=lambda m: None)
                no_token = None
            except hub.HubError as e:
                no_token = str(e)
            ok(no_token and "HF_TOKEN" in no_token and "WRITE" in no_token and calls == [], "no token: a clear message about the Colab secret, no API call")
            os.environ["HF_TOKEN"] = "hf_FAKE_TOKEN_FOR_TEST"
            ok(hub.get_token() == "hf_FAKE_TOKEN_FOR_TEST", "HF_TOKEN environment variable is used")
            seen_tokens = []
            r = hub.publish(run, "someone/m", out_dir=root / "d3", api_factory=lambda t: (seen_tokens.append(t), FakeApi(calls))[1], log=lambda m: None)
            ok(seen_tokens == ["hf_FAKE_TOKEN_FOR_TEST"], "the token reaches the API client")
            create, upload = calls
            ok(create[0] == "create_repo" and create[1]["private"] is True and create[1]["repo_type"] == "model" and create[1]["exist_ok"] is True,
               "the repo is created PRIVATE by default (model repo, exist_ok)")
            ok(upload[0] == "upload_folder" and upload[1]["repo_id"] == "someone/m" and upload[1]["ignore_patterns"] == [hub.MARKER],
               "the folder is uploaded to that repo, without the marker file")
            ok("model.safetensors" in upload[1]["files"] and "README.md" in upload[1]["files"], "the uploaded folder holds the weights and the card")
            ok(r["uploaded"] and r["url"] == "https://huggingface.co/someone/m" and r["private"] is True, "publish returns the URL")
            calls.clear()
            hub.publish(run, "someone/m", private=False, out_dir=root / "d4", api_factory=lambda t: FakeApi(calls), log=lambda m: None)
            ok(calls[0][1]["private"] is False, "private=False makes it public only when asked")
            logs = []
            hub.publish(run, "someone/m", out_dir=root / "d5", api_factory=lambda t: FakeApi([]), log=logs.append)
            ok(not any("hf_FAKE" in m for m in logs), "the token is never printed")
            try:
                hub.publish(run, "someone/m", out_dir=root / "d6", api_factory=lambda t: FakeApi([], fail=True), log=lambda m: None)
                wrapped = None
            except hub.HubError as e:
                wrapped = str(e)
            ok(wrapped and "upload to someone/m failed" in wrapped and "hf_FAKE" not in wrapped, "an upload failure becomes a HubError without leaking the token")

            # ---- CLI
            def cli(*a):
                o, e = io.StringIO(), io.StringIO()
                with contextlib.redirect_stdout(o), contextlib.redirect_stderr(e):
                    code = hub.main(list(a))
                return code, o.getvalue(), e.getvalue()

            code, out_txt, _ = cli("--run", str(run), "--repo-id", "someone/m", "--dry-run", "--out", str(root / "d7"))
            ok(code == 0 and "model.safetensors" in out_txt and "Dry run" in out_txt, "CLI --dry-run lists the files")
            ok(cli("--run", str(root / "nope"), "--repo-id", "someone/m", "--dry-run")[0] == 1, "CLI: missing run -> exit 1")
            ok(cli("--run", str(run), "--repo-id", "bad id", "--dry-run")[0] == 1, "CLI: bad repo id -> exit 1")

            # ---- generate --hf (download patched), and mutually exclusive sources
            hub_dl = hub.download_snapshot
            hub.download_snapshot = lambda repo: str(out)
            try:
                o, e = io.StringIO(), io.StringIO()
                with contextlib.redirect_stdout(o), contextlib.redirect_stderr(e):
                    code = sm.main(["--hf", "someone/m", "--max-new-tokens", "10", "--device", "cpu", "--temperature", "0"])
                ok(code == 0 and "sample 1/1" in o.getvalue(), "generate --hf downloads and generates")
            finally:
                hub.download_snapshot = hub_dl
            for argv in (["--hf", "a/b", "--ckpt", str(out)], []):
                try:
                    with contextlib.redirect_stderr(io.StringIO()):
                        sm.main(argv)
                    exited = None
                except SystemExit as ex:
                    exited = ex.code
                ok(exited == 2, f"generate needs exactly one of --ckpt / --hf ({argv})")
            ok(sm.main(["--ckpt", str(root / "none"), "--device", "cpu"]) == 1, "a missing checkpoint is still exit 1")
        finally:
            huggingface_hub.get_token = saved_get_token
            for k, v in saved_env.items():
                if v is not None:
                    os.environ[k] = v
                else:
                    os.environ.pop(k, None)

    # ---- the notebook: publishing is off and private by default, and comes before the end of the session
    nb = json.loads((Path(__file__).resolve().parents[1] / "run_colab.ipynb").read_text())
    code_cells = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    first = code_cells[0]
    ok("PUSH_TO_HUB = False" in first and "HF_PRIVATE = True" in first and "HF_REPO_ID" in first and "HF_LICENSE" in first,
       "cell 1: publishing is OFF and PRIVATE by default")
    pub = next(i for i, s in enumerate(code_cells) if "publish(" in s)
    quit_i = next(i for i, s in enumerate(code_cells) if "finish_session" in s)
    ok(pub < quit_i and quit_i == len(code_cells) - 1, "the publish cell comes before the final end-of-session cell")
    cell13 = code_cells[pub]
    ok("assert " not in cell13 and "try:" in cell13 and "except Exception" in cell13, "the publish cell has no bare assert and catches every error")
    ok("if not PUSH_TO_HUB" in cell13, "the publish cell is skipped unless PUSH_TO_HUB is True")
    called = []
    real_publish = hub.publish
    hub.publish = lambda *a_, **k_: called.append((a_, k_)) or {"url": "u"}
    try:
        def run_cell(**ns):
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                exec(cell13, {"OUT": "/x", "RUN": "/x/checkpoints/pretrain_cosine", "HF_PRIVATE": True, "HF_LICENSE": "mit", **ns})
            return buf.getvalue()
        out_ = run_cell(PUSH_TO_HUB=False, HF_REPO_ID="u/m")
        ok(called == [] and "Skipping" in out_, "PUSH_TO_HUB = False: nothing is called, a message is printed")
        out_ = run_cell(PUSH_TO_HUB=True, HF_REPO_ID="")
        ok(called == [] and "set HF_REPO_ID" in out_, "PUSH_TO_HUB = True without a repo id: a message, no call, no exception")
        run_cell(PUSH_TO_HUB=True, HF_REPO_ID="u/m")
        ok(len(called) == 1 and called[0][1]["private"] is True and called[0][1]["repo_id"] == "u/m" and "dry_run" not in called[0][1],
           "PUSH_TO_HUB = True publishes to that repo, private by default")

        def boom(*a_, **k_):
            raise hub.HubError("no token")
        hub.publish = boom
        out_ = run_cell(PUSH_TO_HUB=True, HF_REPO_ID="u/m")
        ok("NOT uploaded" in out_ and "no token" in out_, "an upload error is printed and never raised (the next cell still runs)")
    finally:
        hub.publish = real_publish
    # an exported (stopped early) run builds a card that says so
    import tempfile as _tf
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import test_train as _tt
    from dlm import train as _tr
    with _tf.TemporaryDirectory() as _d:
        _d = Path(_d)
        _tt.run(_d / "run", steps=8, eval_every=4, stop_at=4)
        _tr.export_main(["--resume", str(_d / "run" / "resume.pt"), "--out", str(_d / "run" / "final")])
        hub.prepare_folder(_d / "run", _d / "hubdir", "u/m")
        card = (_d / "hubdir" / "README.md").read_text()
        ok("Intermediate checkpoint" in card and "stopped at step 4" in card and "planned for 8 steps" in card, "card of a stopped run says it is an intermediate checkpoint")
        ok("| Steps / batch size | 4 /" in card and "Training time" not in card, "card shows the steps actually trained and no invented training time")
        ok((_d / "hubdir" / "model.safetensors").is_file(), "Hub folder is built from an exported run")
    import re
    ok(re.search(r"hf_[A-Za-z0-9]{20,}", "".join(code_cells)) is None and "token=" not in code_cells[pub], "no token in the notebook")
    print(f"All {checks} Hugging Face Hub checks passed.")


if __name__ == "__main__":
    main()
