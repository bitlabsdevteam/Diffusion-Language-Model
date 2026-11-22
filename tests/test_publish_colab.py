"""Execute the publication notebook offline: real export and standalone inference, fake Drive/Hub boundaries."""
import contextlib
import hashlib
import io
import json
import shutil
import subprocess
import sys
import tempfile
import types
from pathlib import Path
from unittest.mock import patch

import torch
from tokenizers import Tokenizer, models, pre_tokenizers

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dlm.model import DLMConfig, DiffusionTransformerLM  # noqa: E402
from dlm.sample import generate_text, load_checkpoint  # noqa: E402
from dlm.train import build_parser, recipe_of  # noqa: E402


class FakeHub:
    """Only the network boundary is fake; uploaded/downloaded files are real independent copies."""

    def __init__(self, root):
        self.root = root
        self.creates, self.uploads, self.downloads = [], [], []
        self.fail = False
        self.oid = "a" * 40

    def create_repo(self, **kw):
        self.creates.append(kw)

    def upload_folder(self, **kw):
        if self.fail:
            raise ConnectionError("simulated upload failure")
        self.uploads.append(kw)
        self.remote = self.root / "remote" / self.oid
        shutil.copytree(kw["folder_path"], self.remote, ignore=shutil.ignore_patterns(*kw["ignore_patterns"]))
        return types.SimpleNamespace(oid=self.oid)

    def download(self, **kw):
        assert kw["revision"] in (self.oid, "main")
        self.downloads.append(kw)
        snapshot = self.root / f"download-{len(self.downloads)}" / "snapshots" / self.oid
        blobs = snapshot.parent.parent / "blobs"
        blobs.mkdir(parents=True)
        # HF's real cache uses symlinks. Resolving infer.py would lose the snapshot root.
        for file in self.remote.rglob("*"):
            if file.is_file():
                blob = blobs / hashlib.sha256(file.read_bytes()).hexdigest()
                shutil.copy2(file, blob)
                link = snapshot / file.relative_to(self.remote)
                link.parent.mkdir(parents=True, exist_ok=True)
                link.symlink_to(blob)
        return str(snapshot)


def fixture(root, schedule):
    """A small checkpoint with the real trainer's schema and a tokenizer whose IDs match its embeddings."""
    specials = ["[PAD]", "[UNK]", "[BOS]", "[EOS]", "[MASK]", "<|user|>", "<|assistant|>", "<|system|>", "<|end|>"]
    words = "Write a short story. Once upon time, there was little rabbit cat dog went home happy".split()
    vocab = {word: i for i, word in enumerate(specials + words)}
    tok = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tok.add_special_tokens(specials)
    tokenizer_dir = root / "tokenizer"
    tokenizer_dir.mkdir(parents=True)
    tok.save(str(tokenizer_dir / "tokenizer.json"))
    cfg = DLMConfig(vocab_size=len(vocab), seq_len=64, d_model=32, n_layers=1, n_heads=4, d_ff=48)
    torch.manual_seed(1)
    model = DiffusionTransformerLM(cfg).eval()
    args = build_parser().parse_args([])
    args.schedule, args.steps, args.seq_len = schedule, 100, cfg.seq_len
    source = root / "resume.pt"
    torch.save({"model": model.state_dict(), "optimizer": {"sentinel": 123}, "rng": {"sentinel": 456},
                "step": 50, "history": [{"step": 50, "epoch": 0.1, "loss": 3.5, "val": 3.4}],
                "recipe": recipe_of(args, cfg)}, source)
    return source, tokenizer_dir, model, tok


def main():
    torch.set_num_threads(1)
    nb = json.loads((ROOT / "publish_hf_colab.ipynb").read_text())
    cells = {c["id"]: "".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"}
    checks = 0

    def ok(condition, message):
        nonlocal checks
        assert condition, message
        checks += 1

    def execute(name, ns):
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(cells[name], f"publish_hf_colab:{name}", "exec"), ns)

    def raises(name, ns, exception, text):
        try:
            execute(name, ns)
        except exception as e:
            ok(text in str(e), f"{name}: actionable error: {text}")
            return str(e)
        else:
            raise AssertionError(f"{name} should raise {exception.__name__}")

    ok(nb["nbformat"] == 4 and len({c["id"] for c in nb["cells"]}) == len(nb["cells"]), "valid notebook IDs")
    for name, code in cells.items():
        compile(code, name, "exec")
    ok(all(c.get("outputs", []) == [] and c.get("execution_count") is None for c in nb["cells"]), "no saved outputs")
    raises("settings", {}, ValueError, "Set HF_REPO_ID")

    mounts = []
    colab = types.ModuleType("google.colab")
    colab.drive = types.SimpleNamespace(mount=lambda path: mounts.append(path))
    colab.userdata = types.SimpleNamespace(get=lambda key: None)
    with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules, {"google.colab": colab}):
        root = Path(tmp)
        # The notebook's temporary export folders are cleaned up with this test's root.
        with patch.object(tempfile, "tempdir", str(root)):
            # A guessed cosine-run path must identify the checkpoint actually saved on Drive.
            drive_root = root / "Drive files" / "diffusion_lm_outputs"
            available = drive_root / "checkpoints" / "pretrain_base" / "resume.pt"
            available.parent.mkdir(parents=True)
            available.touch()
            missing = drive_root / "checkpoints" / "pretrain_cosine" / "resume.pt"
            discovery = {"MODE": "publish", "Path": Path, "RESUME_PATH": str(missing),
                         "CHECKPOINT_SEARCH_ROOT": str(drive_root), "TOKENIZER_DIR": "", "REPO_DIR": ROOT}
            with patch("dlm.train.export_checkpoint") as exporter, patch.object(tempfile, "mkdtemp") as stage:
                message = raises("export", discovery, FileNotFoundError, f"RESUME_PATH = {str(available)!r}")
                ok(str(missing) in message and "cell 1" in message, "missing path and recovery instructions are shown")
                raises("export", {**discovery, "RESUME_PATH": "  "}, FileNotFoundError, str(available))
                other = drive_root / "checkpoints" / "another_run" / "resume.pt"
                other.parent.mkdir(parents=True)
                other.touch()
                # Folders named resume.pt are not valid candidates.
                decoy = drive_root / "not_a_checkpoint" / "resume.pt"
                decoy.mkdir(parents=True)
                message = raises("export", discovery, FileNotFoundError, str(available))
                ok(str(other) in message and str(decoy) not in message, "all candidate files, and only files, are listed")
                empty_root = root / "empty-output"
                empty_root.mkdir()
                message = raises("export", {**discovery, "CHECKPOINT_SEARCH_ROOT": str(empty_root)},
                                 FileNotFoundError, "No resume.pt files found")
                ok("CHECKPOINT_SEARCH_ROOT" in message, "empty search explains how to search a different folder")
                raises("export", {**discovery, "CHECKPOINT_SEARCH_ROOT": str(root / "missing-output")},
                       FileNotFoundError, "Search folder does not exist")
                ok(not exporter.called and not stage.called, "discovery never exports an unselected run or creates staging files")

            for schedule in ("cosine", "linear"):
                case = root / schedule
                source, tokenizer_dir, original, tok = fixture(case, schedule)
                original_bytes = source.read_bytes()
                # An existing final/ is not used in place of the explicitly selected resume.pt.
                (case / "final").mkdir()
                (case / "final" / "model.pt").write_bytes(b"existing final sentinel")
                fake = FakeHub(case)
                ns = {"__name__": "__main__", "MODE": "publish", "HF_REPO_ID": "tester/dlm",
                       "HF_PRIVATE": True, "HF_LICENSE": "mit", "RESUME_PATH": str(source),
                       "CHECKPOINT_SEARCH_ROOT": str(case),
                      "TOKENIZER_DIR": str(tokenizer_dir), "HF_REVISION": "main", "REPO_DIR": ROOT,
                      "CODE_COMMIT": "b" * 40, "SOURCE_REPO": "https://example.com/dlm.git",
                      "Path": Path, "json": json, "sys": sys, "subprocess": subprocess,
                      "snapshot_download": fake.download, "api": fake, "_hf_token": "offline-test-token"}
                execute("export", ns)
                ok(ns["SCHEDULE"] == schedule and ns["export_info"]["step"] == 50, "checkpoint schedule and step preserved")
                execute("inference-script", ns)
                compile(ns["INFERENCE_SCRIPT"], "infer.py", "exec")
                execute("upload", ns)
                ok(fake.creates[0]["private"] is True and fake.uploads[0]["repo_id"] == "tester/dlm", "correct upload destination")
                ok(ns["UPLOADED_REVISION"] == fake.oid, "upload commit is retained")
                remote = fake.remote
                ok(not (remote / "resume.pt").exists() and not (remote / ".dlm_hf_upload").exists(), "training state and staging marker excluded")
                generation = json.loads((remote / "generation_config.json").read_text())
                ok(generation["schedule"] == schedule, "HF generation config matches checkpoint, including cosine")
                from huggingface_hub import ModelCard
                card = ModelCard.load(remote / "README.md")
                ok(card.data.license == "mit" and "infer.py" in card.text and f"**{schedule}**" in card.text,
                   "model card has chosen license and portable inference instructions")
                ok("stopped at step 50" in card.text and "training_curves.png" not in card.text, "card reports actual checkpoint and has no broken plot")
                execute("download", ns)
                ok(fake.downloads[-1]["revision"] == fake.oid and fake.downloads[-1]["token"] == "offline-test-token",
                   "verification downloads the uploaded commit with authentication")
                ok(ns["SNAPSHOT_DIR"] != ns["UPLOAD_DIR"] and (ns["SNAPSHOT_DIR"] / "infer.py").is_symlink(),
                   "inference source is the independent HF-style symlink cache")
                loaded, loaded_tok = load_checkpoint(ns["SNAPSHOT_DIR"])
                ok(all(torch.equal(v, loaded.state_dict()[k]) for k, v in original.state_dict().items()), "all exported weights round-trip exactly")
                ok(loaded_tok.to_str() == tok.to_str() and loaded.lm_head.weight is loaded.tok_emb.weight,
                   "tokenizer and tied embeddings round-trip")

                # Run the uploaded entry point in a fresh process, outside the project directory.
                prompt = "Once upon a time, there was a little rabbit"
                expected = generate_text(original, tok, prompt, max_new_tokens=8, temperature=0, schedule=schedule)["text"]
                command = [sys.executable, str(ns["SNAPSHOT_DIR"] / "infer.py"), "--prompt", prompt,
                           "--max-new-tokens", "8", "--temperature", "0", "--device", "cpu"]
                result = subprocess.run(command, cwd=case, check=True, capture_output=True, text=True)
                ok(result.stdout.startswith(expected + "\n") and f"schedule={schedule}" in result.stdout,
                   "standalone downloaded code generates the same greedy text and uses saved schedule")

                # Run the actual notebook inference cell; write the receipt only after successful inference.
                receipt_path = case / "hf_publication.json"
                with patch.object(subprocess, "run", side_effect=subprocess.CalledProcessError(1, "infer.py")):
                    raises("generate", ns, subprocess.CalledProcessError, "infer.py")
                ok(not receipt_path.exists(), "failed inference does not report a verified publication")
                execute("generate", ns)
                receipt = json.loads(receipt_path.read_text())
                ok(receipt["revision"] == fake.oid and receipt["download_and_inference_verified"], "receipt records verified HF commit")
                ok(source.read_bytes() == original_bytes and (case / "final" / "model.pt").read_bytes() == b"existing final sentinel",
                   "source checkpoint and existing final weights are untouched")

                # The "load the deployed DLM" test cell: loads the downloaded HF snapshot in a fresh process.
                def run_cell(name, namespace):
                    out = io.StringIO()
                    with contextlib.redirect_stdout(out):
                        exec(compile(cells[name], f"publish_hf_colab:{name}", "exec"), namespace)
                    return out.getvalue()

                text = run_cell("hf-load-test", {**ns, "TEST_MAX_NEW_TOKENS": 8})
                ok("PASSED" in text and "loaded " in text and f"schedule {schedule}" in text, "deployed model loads from the HF snapshot and passes the test cell")
                ok(text.count("--- prompt") == 3, "the test cell generates from every test prompt")
                raises("hf-load-test", {**ns, "SNAPSHOT_DIR": case / "no-snapshot"}, RuntimeError, "download cell")
                broken = shutil.copytree(ns["SNAPSHOT_DIR"], case / "broken-snapshot")
                (broken / "model.safetensors").write_bytes(b"corrupted")
                raises("hf-load-test", {**ns, "SNAPSHOT_DIR": broken, "TEST_MAX_NEW_TOKENS": 8}, RuntimeError, "failed the load test")

                # Fresh-runtime mode: no Drive, exporter, or local model variables are available.
                inference = {"MODE": "inference", "HF_REPO_ID": "tester/dlm", "HF_REVISION": "main"}
                with patch("huggingface_hub.get_token", return_value=None), patch("huggingface_hub.HfApi", return_value=fake), \
                     patch("huggingface_hub.snapshot_download", side_effect=fake.download):
                    execute("setup", inference)
                    execute("authentication", inference)
                before_mounts, before_uploads = len(mounts), len(fake.uploads)
                for name in ("export", "inference-script", "upload", "download", "generate"):
                    execute(name, inference)
                ok(len(mounts) == before_mounts and len(fake.uploads) == before_uploads, "inference-only mode never mounts Drive or uploads")
                ok(fake.downloads[-1]["token"] is None, "public inference needs no token")
                ok("PASSED" in run_cell("hf-load-test", {**inference, "TEST_MAX_NEW_TOKENS": 8}), "inference-only mode passes the deployed-model test")
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    exec(cells["reuse-snippet"], inference)
                compile(output.getvalue(), "copyable-inference.py", "exec")
                ok(fake.oid in output.getvalue(), "copyable Python snippet pins the resolved HF commit")

                # A corrupted remote payload is detected before invoking its inference script.
                (fake.remote / "model.safetensors").write_bytes(b"corrupted")
                raises("download", ns, RuntimeError, "model.safetensors")
                fake.fail = True
                raises("upload", ns, ConnectionError, "simulated upload failure")
                ok(ns["UPLOADED_REVISION"] is None, "failed upload clears a previous success state")
                raises("download", ns, RuntimeError, "Complete the upload")

                bad_ns = {**ns, "RESUME_PATH": str(case / "missing.pt")}
                raises("export", bad_ns, FileNotFoundError, str(source))
                execute("export", {**ns, "RESUME_PATH": f"  {source}  "})
                ok(True, "a selected existing path with surrounding whitespace exports successfully")
                raises("export", {**ns, "TOKENIZER_DIR": str(case / "wrong-tokenizer")}, FileNotFoundError, "TOKENIZER_DIR")
                old = case / "old.pt"
                torch.save({"config": {}, "model": {}}, old)
                raises("export", {**ns, "RESUME_PATH": str(old)}, ValueError, "Older standalone notebooks")
                wrong_tok = case / "wrong-vocab"
                wrong_tok.mkdir()
                smaller = Tokenizer(models.WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
                smaller.save(str(wrong_tok / "tokenizer.json"))
                raises("export", {**ns, "TOKENIZER_DIR": str(wrong_tok)}, ValueError, "vocabulary size differs")

            with patch("huggingface_hub.get_token", return_value=None):
                raises("authentication", {"MODE": "publish"}, RuntimeError, "write-access HF_TOKEN")

    print(f"All {checks} publication notebook checks passed.")


if __name__ == "__main__":
    main()
