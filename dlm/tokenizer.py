#!/usr/bin/env python3
"""Phase 1: train the byte-level BPE tokenizer from the Phase 0 TinyStories Parquet files.

Reads  data/train.parquet (Phase 0). Writes tokenizer/ (next to run.py):
    tokenizer.json, tokenizer_config.json                             (Hugging Face PreTrainedTokenizerFast)
    tokenizer_manifest.json                                           (written last; presence = complete)

Design (see PRD.md Phase 1):
  * trained on a seeded random sample of non-empty train Stories, never on validation;
  * each Story goes through dlm.cleaning.clean_story (mojibake repair) and the chat template;
  * byte-level BPE, full 256-byte alphabet (so [UNK] cannot occur), no normalizer (lossless),
    no automatic [BOS]/[EOS] (the Phase 2 data pipeline adds them);
  * fixed special-token IDs: [PAD]=0 [UNK]=1 [BOS]=2 [EOS]=3 [MASK]=4 <|user|>=5 <|assistant|>=6 <|system|>=7 <|end|>=8.
A finished tokenizer is verified on every validation Story (exact round trip, no [UNK]) before it is kept.
Re-running is a no-op when the manifest matches the requested settings and the Phase 0 revision.

Usage (from DLM/; `python run.py phase1 ...` is the same as `python -m dlm.tokenizer ...`):
    python run.py phase1
    python run.py phase1 --vocab-size 8192 --n-stories 500000 --seed 0
    python run.py phase1 --force
    python run.py phase1 --encode "Once upon a time"     # show tokens (no training)
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import random
import shutil
import statistics
import sys
from pathlib import Path

from dlm.cleaning import CLEANING_VERSION, clean_story, has_mojibake

ROOT = Path(__file__).resolve().parents[1]  # DLM/
DEFAULT_VOCAB_SIZE = 8192
DEFAULT_N_STORIES = 500_000
DEFAULT_SEED = 0
MANIFEST = "tokenizer_manifest.json"
SEQ_LEN = 256  # the notebooks' block length; only used for the "fits in SEQ_LEN" statistic

SPECIAL_TOKENS = [
    "[PAD]", "[UNK]", "[BOS]", "[EOS]", "[MASK]",
    "<|user|>", "<|assistant|>", "<|system|>", "<|end|>",
]
EXPECTED_IDS = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
CHAT_TOKENS = ["<|user|>", "<|assistant|>", "<|system|>", "<|end|>"]
PRETRAIN_INSTRUCTION = "Write a short story."


class Phase1Error(RuntimeError):
    """Raised when the tokenizer cannot be trusted."""


def format_story(story: str) -> str:
    """Cleaned Story in the pre-training chat template (no [BOS]/[EOS]; Phase 2 adds those)."""
    return f"<|user|>\n{PRETRAIN_INSTRUCTION}\n<|assistant|>\n{clean_story(story)}\n<|end|>\n"


def read_split(data_dir: Path, split: str) -> list[str]:
    import pyarrow.parquet as pq

    return pq.read_table(data_dir / f"{split}.parquet", columns=["text"]).column("text").to_pylist()


def sample_training_stories(stories: list[str], n: int, seed: int) -> tuple[list[str], dict]:
    """Seeded random sample of non-empty Stories (empty = empty after cleaning). Returns (sample, stats)."""
    eligible = [i for i, s in enumerate(stories) if s.strip()]
    if len(eligible) < n:
        n = len(eligible)
    idx = random.Random(seed).sample(eligible, n)
    sample = [stories[i] for i in idx]
    stats = {
        "eligible_stories": len(eligible),
        "sampled_stories": len(sample),
        "sampled_with_mojibake": sum(1 for s in sample if has_mojibake(s)),
    }
    return sample, stats


def build_tokenizer():
    from tokenizers import Tokenizer
    from tokenizers.decoders import ByteLevel as ByteLevelDecoder
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import ByteLevel

    tok = Tokenizer(BPE(unk_token="[UNK]"))
    tok.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tok.decoder = ByteLevelDecoder()
    # No normalizer (lossless) and no post-processor (no automatic [BOS]/[EOS]).
    return tok


def train_tokenizer(texts, vocab_size: int):
    from tokenizers.pre_tokenizers import ByteLevel
    from tokenizers.trainers import BpeTrainer

    tok = build_tokenizer()
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=ByteLevel.alphabet(),  # all 256 bytes: every input is encodable
        show_progress=True,
    )
    tok.train_from_iterator(texts, trainer=trainer)
    return tok


def wrap_hf(tok):
    from transformers import PreTrainedTokenizerFast

    return PreTrainedTokenizerFast(
        tokenizer_object=tok,
        pad_token="[PAD]", unk_token="[UNK]", bos_token="[BOS]", eos_token="[EOS]", mask_token="[MASK]",
        additional_special_tokens=CHAT_TOKENS,
    )


def load_tokenizer(tok_dir: Path):
    from transformers import PreTrainedTokenizerFast

    return PreTrainedTokenizerFast.from_pretrained(str(tok_dir))


def check_special_ids(tok, vocab_size: int) -> None:
    got = {t: tok.token_to_id(t) for t in SPECIAL_TOKENS}
    if got != EXPECTED_IDS:
        raise Phase1Error(f"special-token IDs differ from the contract: {got} != {EXPECTED_IDS}")
    if tok.get_vocab_size() != vocab_size:
        raise Phase1Error(f"vocabulary has {tok.get_vocab_size()} tokens, requested {vocab_size}")


def verify(tok, val_stories: list[str]) -> dict:
    """Acceptance checks on every non-empty validation Story. Raises Phase1Error on any failure."""
    texts = [format_story(s) for s in val_stories if s.strip()]
    if any("â€" in t for t in texts):
        raise Phase1Error("mojibake marker 'â€' still present after cleaning")
    enc = tok.encode_batch(texts, add_special_tokens=False)
    unk = tok.token_to_id("[UNK]")
    reserved = {tok.token_to_id(t) for t in ("[PAD]", "[UNK]", "[BOS]", "[EOS]", "[MASK]")}
    n_unk = n_reserved = bad = 0
    lens: list[int] = []
    chars = 0
    for text, e in zip(texts, enc):
        ids = e.ids
        n_unk += ids.count(unk)
        n_reserved += sum(1 for i in ids if i in reserved)
        lens.append(len(ids))
        chars += len(text)
        if tok.decode(ids, skip_special_tokens=False) != text:
            bad += 1
    if bad:
        raise Phase1Error(f"round trip failed on {bad:,} of {len(texts):,} validation Stories")
    if n_unk:
        raise Phase1Error(f"[UNK] occurred {n_unk:,} times on validation Stories")
    if n_reserved:
        raise Phase1Error(f"reserved tokens ([PAD]/[UNK]/[BOS]/[EOS]/[MASK]) appeared {n_reserved:,} times in plain text")
    probe = tok.encode("Once upon a time", add_special_tokens=False).ids
    if tok.token_to_id("[BOS]") in probe or tok.token_to_id("[EOS]") in probe:
        raise Phase1Error("the tokenizer adds [BOS]/[EOS] automatically")
    lens_sorted = sorted(lens)
    return {
        "validation_stories_checked": len(texts),
        "round_trip_exact": True,
        "unk_tokens": 0,
        "tokens_per_story_mean": round(statistics.fmean(lens), 1),
        "tokens_per_story_p50": lens_sorted[len(lens) // 2],
        "tokens_per_story_p90": lens_sorted[int(0.9 * len(lens))],
        "tokens_per_story_p99": lens_sorted[int(0.99 * len(lens))],
        "chars_per_token": round(chars / sum(lens), 3),
        f"share_fitting_{SEQ_LEN}_tokens": round(sum(1 for n in lens if n <= SEQ_LEN) / len(lens), 4),
    }


def request_params(vocab_size: int, n_stories: int, seed: int, data_revision: str) -> dict:
    return {
        "vocab_size": vocab_size,
        "n_stories": n_stories,
        "seed": seed,
        "cleaning_version": CLEANING_VERSION,
        "phase0_revision": data_revision,
    }


def is_complete(tok_dir: Path, params: dict) -> bool:
    mf = tok_dir / MANIFEST
    if not mf.is_file() or not (tok_dir / "tokenizer.json").is_file():
        return False
    try:
        m = json.loads(mf.read_text())
    except json.JSONDecodeError:
        return False
    return m.get("params") == params and m.get("special_tokens") == SPECIAL_TOKENS


def build(data_dir: Path, out: Path, vocab_size: int, n_stories: int, seed: int) -> dict:
    import tokenizers
    import transformers

    from dlm.download import is_complete as phase0_complete
    from dlm.download import read_manifest

    if not phase0_complete(data_dir):
        raise Phase1Error(f"No complete Phase 0 data in {data_dir}. Run: python run.py phase0")
    data_revision = read_manifest(data_dir)["revision"]
    params = request_params(vocab_size, n_stories, seed, data_revision)

    print("Reading Stories ...", flush=True)
    train, val = read_split(data_dir, "train"), read_split(data_dir, "validation")
    sample, sample_stats = sample_training_stories(train, n_stories, seed)
    if len(sample) < n_stories:
        print(f"WARNING: only {len(sample):,} eligible Stories (requested {n_stories:,})", flush=True)
    del train
    print(f"Training {vocab_size:,}-token BPE on {len(sample):,} Stories "
          f"({sample_stats['sampled_with_mojibake']:,} repaired by clean_story) ...", flush=True)
    tok = train_tokenizer((format_story(s) for s in sample), vocab_size)
    check_special_ids(tok, vocab_size)

    tmp = out.parent / (out.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    wrap_hf(tok).save_pretrained(str(tmp))
    loaded = load_tokenizer(tmp)  # verify what was actually saved, not the in-memory object
    stats = verify(loaded._tokenizer, val)

    manifest = {
        "params": params,
        "special_tokens": SPECIAL_TOKENS,
        "template": f"<|user|>\n{PRETRAIN_INSTRUCTION}\n<|assistant|>\n{{story}}\n<|end|>\n",
        "training_sample": sample_stats,
        "verification": stats,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "tokenizers_version": tokenizers.__version__,
        "transformers_version": transformers.__version__,
    }
    (tmp / MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n")  # last: marks the folder complete
    if out.exists():
        shutil.rmtree(out)
    tmp.rename(out)
    return manifest


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=None, help="Phase 0 output (default: DLM/data, or Drive on Colab)")
    p.add_argument("--out", type=Path, default=ROOT / "tokenizer", help="output folder (default: DLM/tokenizer)")
    p.add_argument("--vocab-size", type=int, default=DEFAULT_VOCAB_SIZE)
    p.add_argument("--n-stories", type=int, default=DEFAULT_N_STORIES, help="training sample size")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--force", action="store_true", help="retrain even if the tokenizer is already complete")
    p.add_argument("--encode", metavar="TEXT", help="print the tokens of TEXT with the saved tokenizer, then exit")
    args = p.parse_args(argv)
    out = args.out.expanduser().resolve()

    if args.encode is not None:
        if not (out / MANIFEST).is_file():
            print(f"No tokenizer in {out}. Run: python run.py phase1", file=sys.stderr)
            return 1
        tok = load_tokenizer(out)
        ids = tok.encode(args.encode, add_special_tokens=False)
        print(len(ids), "tokens")
        for i in ids:
            print(f"{i:>6}  {tok.convert_ids_to_tokens(i)!r}")
        return 0

    from dlm.download import default_out_dir
    from dlm.download import is_complete as phase0_complete
    from dlm.download import read_manifest

    data_dir = (args.data_dir or default_out_dir()).expanduser().resolve()
    try:
        if not phase0_complete(data_dir):
            raise Phase1Error(f"No complete Phase 0 data in {data_dir}. Run: python run.py phase0")
        params = request_params(args.vocab_size, args.n_stories, args.seed, read_manifest(data_dir)["revision"])
        if not args.force and is_complete(out, params):
            m = json.loads((out / MANIFEST).read_text())
            print(f"Already complete in {out} (vocab {params['vocab_size']:,}); nothing to do.")
            print("  verification:", json.dumps(m["verification"]))
            return 0
        manifest = build(data_dir, out, args.vocab_size, args.n_stories, args.seed)
    except Phase1Error as e:
        print(f"PHASE 1 FAILED: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"PHASE 1 FAILED ({type(e).__name__}): {e}", file=sys.stderr)
        return 1

    print(f"\nDone. Wrote {out}")
    for k, v in manifest["verification"].items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
