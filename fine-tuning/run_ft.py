#!/usr/bin/env python3
"""Entry point for fine-tuning (everything here is separate from the pre-training code in ../dlm and ../run.py).

    python fine-tuning/run_ft.py prep [--keep-prob 0.2]            parse + tokenize TinyStoriesInstruct into fine-tuning/data/cache
    python fine-tuning/run_ft.py train --init-from <pretrained> [--steps 3000 --lr 3e-5 --lora-r 0 ...]
    python fine-tuning/run_ft.py generate --ckpt <run>/final --words cat,ball --features Dialogue --n 3
    python fine-tuning/run_ft.py eval --base <pretrained> --sft <run>/final      base vs SFT control metrics
    python fine-tuning/run_ft.py test                              offline tests (fine-tuning/tests)

Run from DLM/ with the venv, e.g. `.venv/bin/python fine-tuning/run_ft.py test`. Everything after the command is passed on:
`run_ft.py train --help`.
"""
from __future__ import annotations

import _path  # noqa: F401

import argparse
import json
import subprocess
import sys
from pathlib import Path

FT_DIR = _path.FT_DIR
DLM_ROOT = _path.DLM_ROOT
HF_REPO = "roneneldan/TinyStoriesInstruct"
RAW = {"train": "TinyStories-Instruct-train.txt", "valid": "TinyStories-Instruct-valid.txt"}
TESTS = ("test_ft_parse.py", "test_ft_data.py", "test_ft_train.py", "test_ft_eval.py", "test_ft_layout.py")


def fetch_raw(split: str, override: Path | None) -> Path:
    if override:
        return Path(override).expanduser()
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(repo_id=HF_REPO, filename=RAW[split], repo_type="dataset"))


def prep_main(argv) -> int:
    import ft_data as D
    import ft_parse as P

    p = argparse.ArgumentParser(description="Build the SFT cache from TinyStoriesInstruct.")
    p.add_argument("--out", type=Path, default=FT_DIR / "data" / "cache")
    p.add_argument("--tokenizer-dir", type=Path, default=DLM_ROOT / "tokenizer")
    p.add_argument("--keep-prob", type=float, default=0.2, help="share of TRAIN records kept (0.2 = about 420k examples); valid keeps all")
    p.add_argument("--limit", type=int, default=None, help="stop after this many kept train records (smoke tests)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--seq-len", type=int, default=D.DEFAULT_SEQ_LEN)
    p.add_argument("--raw-train", type=Path, default=None, help="local TinyStories-Instruct-train.txt (default: Hugging Face cache/download)")
    p.add_argument("--raw-valid", type=Path, default=None)
    p.add_argument("--force", action="store_true")
    p.add_argument("--stats", action="store_true", help="only print parser statistics of the valid file, then exit")
    a = p.parse_args(argv)
    tok_json = a.tokenizer_dir / "tokenizer.json"
    if a.stats:
        print(json.dumps(P.scan_stats(fetch_raw("valid", a.raw_valid)), indent=2))
        return 0
    mf = a.out / D.CACHE_MANIFEST
    old = json.loads(mf.read_text()) if mf.is_file() else {"splits": {}}
    for split, kp in (("valid", 1.0), ("train", a.keep_prob)):
        have = old["splits"].get(split)
        want_lim = a.limit if split == "train" else None
        if (have and not a.force and have["keep_prob"] == kp and have["seed"] == a.seed and have.get("limit") == want_lim
                and (a.out / f"{split}_resp_tokens.npy").is_file()):
            print(f"[{split}] cache already built ({have['examples']:,} examples); use --force to rebuild")
            continue
        raw = fetch_raw(split, a.raw_train if split == "train" else a.raw_valid)
        print(f"[{split}] parsing {raw}")
        m = D.build_cache(raw, a.out, split, tok_json, keep_prob=kp, seed=a.seed, limit=want_lim, seq_len=a.seq_len)
        m["splits"][split]["limit"] = want_lim
        (a.out / D.CACHE_MANIFEST).write_text(json.dumps(m, indent=2) + "\n")
        s = m["splits"][split]
        print(f"[{split}] {s['examples']:,} examples ({s['dropped_response_too_long']:,} too long), "
              f"{s['response_tokens'] / 1e6:.1f}M response tokens, mean {s['response_tokens_mean']}, "
              f"trailing fields dropped in {s['trailing_fields_dropped_share']:.1%} of records")
    return 0


def generate_main(argv) -> int:
    import torch
    from tokenizers import Tokenizer

    import ft_data as D
    import ft_sample as S
    from dlm.model import load_model
    from dlm.sample import render_trace
    from dlm.train import pick_device

    p = argparse.ArgumentParser(description="Generate stories from an instruction with a fine-tuned (or base) checkpoint.")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--tokenizer-dir", type=Path, default=DLM_ROOT / "tokenizer")
    p.add_argument("--words", default="", help="comma-separated, e.g. cat,ball,happy")
    p.add_argument("--features", default="", help="comma-separated, e.g. Dialogue,BadEnding")
    p.add_argument("--summary", default=None)
    p.add_argument("--sentence", default=None, help="a sentence that must appear in the story")
    p.add_argument("--instruction", default=None, help="full instruction text (overrides the field flags)")
    p.add_argument("--n", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--max-new-tokens", type=int, default=480)
    p.add_argument("--schedule", choices=("auto", "linear", "cosine"), default="auto", help="auto = the noise schedule the checkpoint was trained with")
    p.add_argument("--trace", action="store_true")
    p.add_argument("--device", default="auto")
    a = p.parse_args(argv)
    device = pick_device(a.device)
    tok = Tokenizer.from_file(str(a.tokenizer_dir / "tokenizer.json"))
    model = load_model(a.ckpt, device)
    schedule = S.infer_schedule(a.ckpt) if a.schedule == "auto" else a.schedule
    instr = a.instruction or S.make_instruction([w.strip() for w in a.words.split(",") if w.strip()],
                                                [f.strip() for f in a.features.split(",") if f.strip()], a.summary, a.sentence)
    print(f"--- instruction ---\n{instr}\n")
    for k in range(a.n):
        r = S.generate_story(model, tok, instr, seed=a.seed + k, max_new_tokens=a.max_new_tokens, temperature=a.temperature,
                             schedule=schedule, return_trace=a.trace)
        print(f"--- story {k} (seed {a.seed + k}, {'finished' if r['end_index'] is not None else 'TRUNCATED'}) ---\n{r['text'].strip()}")
        for w in r["warnings"]:
            print(f"[warning] {w}")
        if a.trace:
            print(render_trace(r, tok))
        print()
    return 0


def run_tests() -> int:
    code = 0
    for t in TESTS:
        f = FT_DIR / "tests" / t
        if not f.is_file():
            continue
        print(f"== {t}", flush=True)
        code |= subprocess.call([sys.executable, str(f)])
    return code


def main(argv: list[str]) -> int:
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    cmd, rest = argv[0], argv[1:]
    if cmd == "prep":
        return prep_main(rest)
    if cmd == "train":
        from ft_train import main as run
        return run(rest)
    if cmd == "generate":
        return generate_main(rest)
    if cmd == "eval":
        from ft_eval import main as run
        return run(rest)
    if cmd == "test":
        return run_tests()
    print(f"Unknown command {cmd!r}.\n{__doc__}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
