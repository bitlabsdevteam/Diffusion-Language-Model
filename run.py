#!/usr/bin/env python3
"""Single entry point for the DLM project.

    python run.py phase0 [args]    download TinyStories to data/            (dlm/download.py)
    python run.py phase1 [args]    train + verify the tokenizer             (dlm/tokenizer.py)
    python run.py test             run all offline tests                    (tests/)
    python run.py phase2 [args]    tokenize + block the Stories             (dlm/data.py)
    python run.py pretrain [args]  pre-train the model (Phase 5)            (dlm/train.py; --help for options)
    python run.py plot [--out DIR] [--watch SECONDS]   redraw the training curves from a run's history.json
    python run.py export --resume <run>/resume.pt [--out DIR]   make an unfinished run's checkpoint loadable (generate/eval/hub)
    python run.py hub --run DIR --repo-id user/name [--dry-run]   publish a trained run to the Hugging Face Hub (dlm/hub.py)
    python run.py generate [args]  generate Stories from a trained run (Phase 6)  (dlm/sample.py; --help for options)
    python run.py eval [args]      evaluate a trained run: likelihood, sample metrics, decoding sweep (Phase 7)  (dlm/evaluate.py)
    python run.py sft              interface skeleton (Phase 8): exits 2 and names what to grill first

Everything after the command is passed to that command: `python run.py phase1 --help`.
The same code runs on a laptop and in Colab (see run_colab.ipynb).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

TESTS = ("test_download.py", "test_tokenizer.py", "test_data.py", "test_model.py", "test_diffusion.py", "test_train.py",
         "test_sample.py", "test_evaluate.py", "test_hub.py", "test_colab.py", "test_publish_colab.py", "test_layout.py")


def run_tests() -> int:
    import subprocess

    code = 0
    for t in TESTS:
        print(f"== {t}", flush=True)
        code |= subprocess.call([sys.executable, str(ROOT / "tests" / t)])
    return code


def main(argv: list[str]) -> int:
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    cmd, rest = argv[0], argv[1:]
    if cmd == "phase0":
        from dlm.download import main as run
        return run(rest)
    if cmd == "phase1":
        from dlm.tokenizer import main as run
        return run(rest)
    if cmd == "eval":
        from dlm.evaluate import main as run
        return run(rest)
    if cmd == "hub":
        from dlm.hub import main as run
        return run(rest)
    if cmd == "export":
        from dlm.train import export_main
        return export_main(rest)
    if cmd == "plot":
        from dlm.train import plot_main
        return plot_main(rest)
    if cmd == "test":
        return run_tests()
    if cmd in ("phase2", "pretrain", "sft", "generate"):
        try:
            if cmd == "phase2":
                from dlm.data import main as run
                return run(rest)
            if cmd == "generate":
                from dlm.sample import main as run
                return run(rest)
            from dlm.train import main as run
            return run(["--task", cmd] + rest)
        except NotImplementedError as e:
            print(f"`{cmd}`: {e}", file=sys.stderr)
            return 2
    print(f"Unknown command {cmd!r}.\n{__doc__}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
