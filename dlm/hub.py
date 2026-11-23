"""Publish a trained run to the Hugging Face Hub in the standard layout.

`python run.py hub --run <run dir> --repo-id user/name [--public] [--license apache-2.0] [--dry-run]`
or `from dlm.hub import publish`. The upload folder follows Hub practice:

    README.md                   model card: YAML metadata (license, language, library_name, pipeline_tag, tags,
                                datasets) + description, intended uses and limitations, training data, training
                                procedure, evaluation results, how to use, references
    config.json                 architecture settings (+ model_type, architectures)
    model.safetensors           the weights (safetensors, tied embeddings stored once)
    generation_config.json      the sampler's default settings
    tokenizer.json, tokenizer_config.json, tokenizer_manifest.json
    training_curves.png, run_info.json, eval/report.md, eval/samples.md   (whatever exists)

Safety defaults: the repo is created PRIVATE, `dry_run` builds and checks the folder without any network call or token,
and the token is read from the environment / a Colab secret named HF_TOKEN / the cached login and never printed.
This model is not a `transformers` model, so AutoModel cannot load it; the card explains how to load it with this
project's code (`run.py generate --hf user/name`, or `load_checkpoint(snapshot_download(...))`).
"""
from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from pathlib import Path

GITHUB_URL = "https://github.com/bitlabsdevteam/Diffusion-Language-Model"
DATASET = "roneneldan/TinyStories"
MARKER = ".dlm_hf_upload"  # marks a folder this module created, so it may be cleared and rebuilt
REPO_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")
GENERATION_CONFIG = {"max_new_tokens": 480, "block_length": 32, "tokens_per_step": 1, "temperature": 0.8,
                     "schedule": "linear", "stop_tokens": ["<|end|>", "[EOS]"]}
TAGS = ["diffusion", "masked-diffusion", "text-diffusion", "language-model", "tinystories", "from-scratch", "pytorch",
        "safetensors"]


class HubError(RuntimeError):
    """Raised for anything that should stop a publication (and is safe to show to the user)."""


def get_token() -> str | None:
    """A Hugging Face token from the environment, a Colab secret named HF_TOKEN, or the cached login (never logged)."""
    for var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        if os.environ.get(var):
            return os.environ[var]
    try:
        from google.colab import userdata

        tok = userdata.get("HF_TOKEN")
        if tok:
            return tok
    except Exception:
        pass
    try:
        from huggingface_hub import get_token as cached

        return cached()
    except Exception:
        return None


def download_snapshot(repo_id: str) -> str:
    """Download a model repo from the Hub (public, or private if a token is available); returns the local folder."""
    from huggingface_hub import snapshot_download

    try:
        return snapshot_download(repo_id=repo_id, repo_type="model", token=get_token())
    except Exception as e:
        raise HubError(f"could not download {repo_id!r} from the Hugging Face Hub ({type(e).__name__}: {str(e)[:120]})") from e


# ---------------------------------------------------------------- the model card
def sampling_schedule(run_info: dict | None) -> str:
    """The sampler's default noise schedule: the one the model was trained with (run_info's recipe), else linear."""
    schedule = ((run_info or {}).get("recipe") or {}).get("schedule")
    return schedule if schedule in ("linear", "cosine") else GENERATION_CONFIG["schedule"]


def _read_json(path: Path):
    return json.loads(path.read_text()) if path.is_file() else None


def render_card_body(repo_id: str, name: str, model_info: dict, cfg: dict, run_info: dict | None, report: dict | None,
                     tok_manifest: dict | None, epochs: float | None) -> str:
    n_params = model_info["parameters"]
    lines = [
        f"# {name}", "",
        f"A small **masked diffusion language model** ({n_params / 1e6:.1f}M parameters) trained from scratch on the "
        f"[TinyStories](https://huggingface.co/datasets/{DATASET}) dataset. Instead of writing text left to right, it starts "
        "from a fully masked response and fills in tokens over several denoising steps, committing the most confident "
        "predictions first (the approach of [LLaDA](https://arxiv.org/abs/2502.09992)). It is a research and learning "
        f"model, built with [this code]({GITHUB_URL}).", "",
        "## Model description", "",
        "| | |", "|---|---|",
        f"| Type | masked diffusion language model, bidirectional Transformer (no causal mask), no time input |",
        f"| Parameters | {n_params:,} (input and output embeddings tied) |",
        f"| Layers / width / heads | {cfg['n_layers']} / {cfg['d_model']} / {cfg['n_heads']} |",
        f"| Blocks | RMSNorm, rotary position embeddings (RoPE), SwiGLU feed-forward (hidden size {cfg['d_ff']}), no biases |",
        f"| Vocabulary | {cfg['vocab_size']:,} byte-level BPE tokens, trained from scratch on the same corpus |",
        f"| Context length | {cfg['seq_len']} tokens |",
        "", "Special tokens (fixed ids): `[PAD]`=0 `[UNK]`=1 `[BOS]`=2 `[EOS]`=3 `[MASK]`=4 `<|user|>`=5 `<|assistant|>`=6 "
        "`<|system|>`=7 `<|end|>`=8.",
        "", "## Intended uses & limitations", "",
        "**Intended for:** studying masked diffusion language models, and generating short children's-story-style English text.", "",
        "**Not suitable for:** question answering, following instructions or chat (the model only ever saw the fixed prompt "
        "\"Write a short story.\"; free-form instructions need fine-tuning, which has not been done), factual tasks, other "
        "languages, or anything safety-critical.", "",
        "**Limitations:** the model is tiny and its vocabulary and style are those of simple children's stories. It can "
        "repeat or loop and may not finish a Story inside the window. TinyStories is synthetic text written by GPT-3.5 and "
        "GPT-4, so the model inherits that text's style and biases. Stories longer than 512 tokens (about 3.4% of the "
        "training Stories) were dropped, so long outputs are untested.", "",
        "## Training data", "",
        f"[{DATASET}](https://huggingface.co/datasets/{DATASET})"
        + (f", revision `{tok_manifest['params']['phase0_revision']}`" if tok_manifest else "")
        + " (about 2.1M training Stories). Preprocessing: empty Stories dropped; about 6% of Stories contained garbled "
        "quotes (mojibake such as `Donâ€™t`), which were repaired, and curly quotes were folded to plain ones; each Story "
        "was wrapped in a fixed chat template (`<|user|>`, \"Write a short story.\", `<|assistant|>`, the Story, `<|end|>`) "
        "and packed with other whole Stories into blocks of 512 tokens; the prompt part of every Story is never masked. "
        "The dataset is released under CDLA-Sharing-1.0.", "",
        "## Training procedure", "",
    ]
    if run_info:
        r = run_info["recipe"]
        stopped = run_info.get("stopped_at_step")
        if stopped:
            lines += [f"> **Intermediate checkpoint.** Training was planned for {r['steps']:,} steps and was stopped at step "
                      f"{stopped:,}, before the learning-rate decay finished (the rate was still well above its final value). "
                      "A completed run would be expected to score better.", ""]
        lines += [
            "Objective: masked-diffusion ELBO. Each block gets a random noise level, every maskable token is masked with that "
            "probability, and the loss is the cross-entropy on the masked tokens weighted by the ELBO factor.", "",
            "| | |", "|---|---|",
            f"| Steps / batch size | {(stopped or r['steps']):,} / {r['batch_size']} blocks of {r['seq_len']} tokens"
            + (f" (about {epochs:.2f} epochs)" if epochs else "") + " |",
            f"| Optimizer | AdamW, peak learning rate {r['lr']}, betas ({r['beta1']}, {r['beta2']}), weight decay "
            f"{r['weight_decay']} on matrices only, gradient clipping {r['clip']} |",
            f"| Learning-rate schedule | {r['warmup']:,} warmup steps, then cosine decay to {r['min_lr_ratio']:.0%} of the peak |",
            f"| Noise schedule | {r['schedule']} |",
            f"| Precision / device | {run_info.get('precision', '?')} / {run_info.get('device', '?')} |",
            f"| Seed | {r['seed']} |"]
        if run_info.get("steps_per_second") and not stopped:
            # run_info only knows the LAST process (a run resumed after a disconnect reports just its final session),
            # so the total is estimated as steps / speed instead of being taken from the recorded seconds
            lines.append(f"| Training time | about {r['steps'] / run_info['steps_per_second'] / 3600:.2f} h (steps / speed of the last "
                         f"session, {run_info['steps_per_second']} steps/s; an estimate if the run was resumed) |")
    else:
        lines.append("Training settings were not recorded with this checkpoint.")
    lines += ["", "![training curves](training_curves.png)", "", "## Evaluation results", ""]
    if report:
        L, h = report["likelihood"], report["headline"]
        lines += [
            (f"Validation set of TinyStories (all {L['blocks']:,} validation blocks, {L['draws']} noise draws per block). "
             if not (report.get("settings") or {}).get("max_val_blocks") else
             f"A subset of the TinyStories validation set ({L['blocks']:,} evenly spaced blocks, {L['draws']} noise draws per block). ")
            +
            "The likelihood is an upper bound (the diffusion ELBO), not an exact likelihood.", "",
            "| metric | value |", "|---|---|",
            f"| Validation ELBO bound (nats per token) | **{L['nats_per_token']:.3f}** |",
            f"| Validation ELBO bound (bits per character) | **{L['bits_per_char']:.3f}** |",
            f"| Baseline: a model that knows only token frequencies (nats per token) | {L['unigram_baseline_nats_per_token']:.3f} |", "",
            f"Samples ({h['unconditional']['stats']['n']} unconditional Stories, fixed seeds; default decoding: one token per "
            "step, decoding blocks of 32, temperature 0.8):", "",
            "| set | finished | length median | distinct-2 | repetition | GPT-2 perplexity |", "|---|---|---|---|---|---|"]
        for label, row in (("generated", h["unconditional"]), ("real validation Stories", h["reference_real_stories"])):
            s = row["stats"]
            ppl = row.get("gen_ppl")
            lines.append(f"| {label} | {s['finished_rate']:.0%} | {s['length']['median']} | {s['distinct_2']:.3f} | "
                         f"{s['repetition_rate']:.1%} | {'n/a' if ppl is None else f'{ppl:.1f}'} |")
        lines += ["", "The GPT-2 perplexity is only meaningful relative to the real-Stories row. The full report and fixed-seed "
                  "samples are in `eval/`."]
    else:
        lines.append("No evaluation report was available when this card was written.")
    lines += [
        "", "## How to use", "",
        "This is **not a `transformers` model**: `AutoModel` cannot load it. Use the project's code:", "",
        "```bash", f"git clone {GITHUB_URL}", "cd Diffusion-Language-Model",
        "pip install torch tokenizers safetensors huggingface_hub",
        f"python run.py generate --hf {repo_id} --n 3 --seed 0                       # unconditional Stories",
        f"python run.py generate --hf {repo_id} --prompt \"Once upon a time, there was a little\"   # continue your opening words",
        "```", "", "```python",
        "from huggingface_hub import snapshot_download",
        "from dlm.sample import load_checkpoint, generate_text", "",
        f"model, tok = load_checkpoint(snapshot_download(\"{repo_id}\"), device=\"cpu\")",
        "print(generate_text(model, tok, \"Once upon a time, there was a little\", seed=0)[\"text\"])", "```", "",
        "Defaults are in `generation_config.json`: up to 480 new tokens, decoding blocks of 32 filled left to right, one token "
        f"per step, temperature 0.8, the `{sampling_schedule(run_info)}` unmasking schedule (the noise schedule the model was trained "
        "with; `python run.py generate` needs `--schedule` to match), the Story cut at the first `<|end|>` or `[EOS]`.", "",
        "## References", "",
        "- Nie et al., *Large Language Diffusion Models* (LLaDA), 2025. https://arxiv.org/abs/2502.09992",
        "- Eldan and Li, *TinyStories: How Small Can Language Models Be and Still Speak Coherent English?*, 2023. "
        "https://arxiv.org/abs/2305.07759",
        f"- Training and evaluation code: {GITHUB_URL}", ""]
    return "\n".join(lines)


def render_card(repo_id: str, license: str, **kw) -> str:
    from huggingface_hub import ModelCardData

    data = ModelCardData(language="en", license=license, library_name="dlm", pipeline_tag="text-generation", tags=TAGS,
                         datasets=[DATASET])
    return f"---\n{data.to_yaml()}\n---\n\n{render_card_body(repo_id, **kw)}"


# ---------------------------------------------------------------- the upload folder
def prepare_folder(run_dir, out_dir, repo_id: str, license: str = "apache-2.0") -> dict:
    """Build the Hub folder from <run_dir>/final (and eval/). Returns {"out": path, "files": [(relpath, bytes)]}."""
    import torch
    from dlm.model import DLMConfig, DiffusionTransformerLM, save_safetensors

    run_dir, out = Path(run_dir), Path(out_dir)
    final = run_dir / "final"
    for need in (final / "model.pt", final / "config.json", final / "tokenizer" / "tokenizer.json"):
        if not need.is_file():
            raise HubError(f"{need} does not exist: the run is not finished (no final/ folder)")
    if out.exists():
        if not (out / MARKER).is_file() and any(out.iterdir()):
            raise HubError(f"{out} exists and was not created by this tool; choose another --out or delete it yourself")
        shutil.rmtree(out)
    out.mkdir(parents=True)
    (out / MARKER).write_text("created by dlm.hub; safe to delete\n")

    cfg = json.loads((final / "config.json").read_text())
    model = DiffusionTransformerLM(DLMConfig.from_dict(cfg))
    model.load_state_dict(torch.load(final / "model.pt", map_location="cpu"))
    save_safetensors(model, out / "model.safetensors")
    (out / "config.json").write_text(json.dumps({"model_type": "dlm", "architectures": ["DiffusionTransformerLM"], **cfg}, indent=2) + "\n")
    for f in sorted((final / "tokenizer").iterdir()):
        if f.is_file() and not f.name.startswith("."):
            shutil.copy2(f, out / f.name)
    run_info = _read_json(final / "run_info.json")
    if run_info:
        shutil.copy2(final / "run_info.json", out / "run_info.json")
    schedule = sampling_schedule(run_info)
    (out / "generation_config.json").write_text(json.dumps({**GENERATION_CONFIG, "schedule": schedule}, indent=2) + "\n")
    for src in (final / "training_curves.png", run_dir / "training_curves.png"):
        if src.is_file():
            shutil.copy2(src, out / "training_curves.png")
            break
    report = _read_json(run_dir / "eval" / "report.json")
    for f in ("report.md", "samples.md"):
        if (run_dir / "eval" / f).is_file():
            (out / "eval").mkdir(exist_ok=True)
            shutil.copy2(run_dir / "eval" / f, out / "eval" / f)
    history = _read_json(final / "history.json") or []
    epochs = next((h["epoch"] for h in reversed(history) if "epoch" in h), None)
    card = render_card(repo_id, license, name=repo_id.split("/")[1], model_info={"parameters": model.num_parameters()},
                       cfg=cfg, run_info=run_info, report=report, tok_manifest=_read_json(final / "tokenizer" / "tokenizer_manifest.json"),
                       epochs=epochs)
    (out / "README.md").write_text(card)
    from huggingface_hub import ModelCard

    parsed = ModelCard.load(out / "README.md")  # the YAML header must parse
    if parsed.data.license != license or "text-generation" != parsed.data.pipeline_tag:
        raise HubError("model card metadata did not round-trip")
    files = sorted((str(p.relative_to(out)), p.stat().st_size) for p in out.rglob("*") if p.is_file() and p.name != MARKER)
    return {"out": str(out), "files": files}


def _default_api(token):
    from huggingface_hub import HfApi

    return HfApi(token=token)


def publish(run_dir, repo_id: str, private: bool = True, license: str = "apache-2.0", token: str | None = None,
            out_dir=None, dry_run: bool = False, api_factory=None, log=print) -> dict:
    """Build the folder and (unless dry_run) create the repo and upload it. Private by default."""
    if not REPO_ID_RE.match(repo_id or ""):
        raise HubError(f"repo id {repo_id!r} must look like 'username/model-name' (letters, digits, '.', '_', '-')")
    if not license:
        raise HubError("a license is required for the model card (for example 'apache-2.0' or 'mit')")
    out_dir = out_dir or tempfile.mkdtemp(prefix="hf_upload_")
    info = prepare_folder(run_dir, out_dir, repo_id, license)
    total = sum(s for _, s in info["files"])
    log(f"Prepared {info['out']}: {len(info['files'])} files, {total / 1e6:.1f} MB")
    for rel, size in info["files"]:
        log(f"  {rel:<28} {size / 1e6:8.2f} MB")
    if dry_run:
        log("Dry run: nothing was uploaded. Set PUSH_TO_HUB = True (or drop --dry-run) to upload.")
        return {**info, "uploaded": False, "url": None, "private": private}
    token = token or get_token()
    if not token:
        raise HubError("no Hugging Face token found. In Colab: click the key icon in the left sidebar, add a secret named "
                       "HF_TOKEN with a token that has WRITE access (huggingface.co/settings/tokens), and turn on "
                       "'Notebook access'. Elsewhere: set the HF_TOKEN environment variable or run `huggingface-cli login`.")
    api = (api_factory or _default_api)(token)
    try:
        api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
        api.upload_folder(folder_path=info["out"], repo_id=repo_id, repo_type="model", ignore_patterns=[MARKER],
                          commit_message="Upload masked diffusion language model (weights, tokenizer, model card)")
    except Exception as e:
        raise HubError(f"upload to {repo_id} failed ({type(e).__name__}: {str(e)[:200]})") from e
    url = f"https://huggingface.co/{repo_id}"
    log(f"Uploaded: {url} ({'private' if private else 'PUBLIC'})")
    return {**info, "uploaded": True, "url": url, "private": private}


def main(argv: list[str] | None = None) -> int:
    import argparse
    import sys

    p = argparse.ArgumentParser(description="Publish a trained run to the Hugging Face Hub (python run.py hub ...).")
    p.add_argument("--run", type=Path, required=True, help="the run folder that contains final/ (and eval/)")
    p.add_argument("--repo-id", required=True, help="user/model-name on the Hub")
    p.add_argument("--public", action="store_true", help="make the repo public (default: private)")
    p.add_argument("--license", default="apache-2.0", help="license for the model card (default: apache-2.0)")
    p.add_argument("--out", type=Path, default=None, help="where to build the upload folder (default: a temp folder)")
    p.add_argument("--dry-run", action="store_true", help="build and check the folder only; no token, no network")
    a = p.parse_args(argv)
    try:
        publish(a.run, a.repo_id, private=not a.public, license=a.license, out_dir=a.out, dry_run=a.dry_run)
    except HubError as e:
        print(f"hub: {e}", file=sys.stderr)
        return 1
    return 0
