"""Validate a trained checkpoint and write the versioned upload folder (weights, tokenizer, deployment.json).

    python aws/prepare_artifact.py --source <run>/final --target <empty dir> --model-version cosine-step25000-v1 \
        --code-ref <git sha> [--schedule cosine|linear]

`deployment.json` records the release name, the code revision the container must run, the sampling schedule and a
SHA-256 for every file. The schedule defaults to the one recorded in `run_info.json` and must match it.
"""
import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dlm.sample import load_checkpoint  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, type=Path)
    ap.add_argument("--target", required=True, type=Path)
    ap.add_argument("--model-version", required=True)
    ap.add_argument("--code-ref", required=True)
    ap.add_argument("--schedule", choices=["linear", "cosine"])
    args = ap.parse_args()

    source, target = args.source.resolve(), args.target
    if any(target.iterdir()):
        sys.exit(f"{target} is not empty")
    recorded = None
    info = source / "run_info.json"
    if info.is_file():
        recorded = json.loads(info.read_text()).get("recipe", {}).get("schedule")
    schedule = args.schedule or recorded
    if schedule is None:
        sys.exit("No schedule given and run_info.json has none: pass --schedule linear|cosine")
    if recorded is not None and recorded != schedule:
        sys.exit(f"Schedule {schedule!r} differs from the training run's {recorded!r}")

    model, tok = load_checkpoint(source, "cpu")
    assert tok.get_vocab_size() == model.config.vocab_size, "Tokenizer vocabulary mismatch"
    assert tok.token_to_id("[MASK]") == model.config.mask_id, "MASK ID mismatch"
    assert tok.token_to_id("[PAD]") == model.config.pad_id, "PAD ID mismatch"

    weights = "model.pt" if (source / "model.pt").is_file() else "model.safetensors"
    tokenizer_source = source / "tokenizer" if (source / "tokenizer" / "tokenizer.json").is_file() else source
    (target / "tokenizer").mkdir()
    for name in ("tokenizer.json", "tokenizer_config.json", "tokenizer_manifest.json"):
        if (tokenizer_source / name).is_file():
            shutil.copy2(tokenizer_source / name, target / "tokenizer" / name)
    for name in ("config.json", weights, "run_info.json", "export.json", "generation_config.json"):
        if (source / name).is_file():
            shutil.copy2(source / name, target / name)

    hashes = {str(p.relative_to(target)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(target.rglob("*")) if p.is_file()}
    release = {"model_version": args.model_version, "code_ref": args.code_ref,
               "schedule": schedule, "sha256": hashes}
    (target / "deployment.json").write_text(json.dumps(release, indent=2) + "\n")
    print(f"Validated artifact: {target} ({len(hashes)} files, schedule {schedule})")


if __name__ == "__main__":
    main()
