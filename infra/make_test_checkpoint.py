"""Write a tiny RANDOM-WEIGHT checkpoint (real committed tokenizer) for testing the deployment plumbing.

    python aws/make_test_checkpoint.py --out <dir> [--schedule cosine|linear]

Its output text is noise. It exists to exercise prepare_artifact, the container, /healthz and /generate
without a trained model.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dlm.model import DLMConfig, DiffusionTransformerLM  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--schedule", choices=["linear", "cosine"], default="cosine")
    args = ap.parse_args()
    torch.manual_seed(0)
    model = DiffusionTransformerLM(DLMConfig.from_preset("tiny"))
    model.save_pretrained(args.out)
    shutil.copytree(ROOT / "tokenizer", args.out / "tokenizer", dirs_exist_ok=True)
    (args.out / "run_info.json").write_text(json.dumps({"recipe": {"schedule": args.schedule}, "synthetic": True}))
    print(f"wrote {args.out} ({model.num_parameters() / 1e6:.2f}M random parameters)")


if __name__ == "__main__":
    main()
