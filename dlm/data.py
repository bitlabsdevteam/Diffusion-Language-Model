"""Phase 2: turn the Phase 0 Parquet Stories into training blocks.

Pipeline (decisions grilled 2026-11-21, see PRD.md Phase 2):
  1. Drop empty Stories; clean each with dlm.cleaning.clean_story; wrap in the chat template
     (dlm.tokenizer.format_story); tokenize with the Phase 1 tokenizer; add [BOS] / [EOS] explicitly
     (the tokenizer adds none). One tokenized Story = [BOS] <|user|>...<|assistant|>\\n story \\n<|end|>\\n [EOS].
  2. Cache all non-empty tokenized Stories as uint16 arrays under data/tokens/ (independent of block length,
     so changing --seq-len never re-tokenizes), with a manifest; idempotent.
  3. StoryBlocks cuts fixed-length blocks (default 512 tokens): Stories longer than a block are DROPPED (train and
     validation alike); each block starts at a Story's [BOS] and is filled with the following whole Stories while
     they fit (greedy, no overlap); the rest of the block is [PAD]. No Story is ever split across blocks.
  4. Protected tokens (never masked): [BOS] plus the 10-token header of EVERY Story in the block (they are
     always given at generation time). <|end|> and [EOS] stay maskable so the model learns where Stories end.
     [PAD] positions are not protected here; the Phase 4 corruption never masks [PAD].
     No random-length crops.

Block item: {"input_ids": LongTensor[seq_len], "protect": BoolTensor[seq_len]}.
Validation uses the dataset's own validation Split; a fixed seeded 5,000-Story slice of train ("train-eval",
`sample_stories=5000`) gives the train-side loss for the train/validation gap.

Still open (Phase 8): build_sft_example (which instruction dataset). CLI: python run.py phase2 [--seq-len 512 ...]
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import random
import shutil
import sys
from pathlib import Path

from dlm import not_built
from dlm.cleaning import CLEANING_VERSION
from dlm.download import ROOT, default_out_dir, read_manifest
from dlm.download import is_complete as phase0_complete
from dlm.tokenizer import PRETRAIN_INSTRUCTION, format_story, read_split

DEFAULT_SEQ_LEN = 512
TRAIN_EVAL_STORIES = 5_000
FORMAT_VERSION = 1  # bump when the Story token layout changes
CACHE_NAME = "tokens"
CACHE_MANIFEST = "tokens_manifest.json"
SPLITS = ("train", "validation")
CHUNK = 20_000
HEADER = f"<|user|>\n{PRETRAIN_INSTRUCTION}\n<|assistant|>\n"


class Phase2Error(RuntimeError):
    """Raised when the token cache cannot be trusted."""


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cache_params(data_dir: Path, tokenizer_dir: Path) -> dict:
    return {
        "phase0_revision": read_manifest(data_dir)["revision"],
        "tokenizer_sha256": file_sha256(tokenizer_dir / "tokenizer.json"),
        "cleaning_version": CLEANING_VERSION,
        "format_version": FORMAT_VERSION,
    }


def cache_is_complete(cache_dir: Path, params: dict) -> bool:
    mf = cache_dir / CACHE_MANIFEST
    if not mf.is_file():
        return False
    try:
        m = json.loads(mf.read_text())
    except json.JSONDecodeError:
        return False
    if m.get("params") != params:
        return False
    return all((cache_dir / f"{s}_{k}.npy").is_file() for s in SPLITS for k in ("tokens", "offsets"))


def tokenize_split(tok, stories: list[str], bos: int, eos: int, header_ids: list[int]):
    """Tokenize all non-empty Stories of a split. Returns (tokens uint16, offsets int64, stats)."""
    import numpy as np

    texts = [s for s in stories if s.strip()]
    empty_dropped = len(stories) - len(texts)
    parts: list = []
    lengths = np.empty(len(texts), dtype=np.int64)
    header = np.asarray(header_ids, dtype=np.uint16)
    bad_header = 0
    for a in range(0, len(texts), CHUNK):
        chunk = texts[a:a + CHUNK]
        enc = tok.encode_batch([format_story(s) for s in chunk], add_special_tokens=False)
        for k, e in enumerate(enc):
            arr = np.empty(len(e.ids) + 2, dtype=np.uint16)
            arr[0], arr[-1] = bos, eos
            arr[1:-1] = e.ids
            if len(arr) < len(header) + 3 or not np.array_equal(arr[1:1 + len(header)], header):
                bad_header += 1
            lengths[a + k] = len(arr)
            parts.append(arr)
        print(f"  tokenized {min(a + CHUNK, len(texts)):,} / {len(texts):,}", end="\r", flush=True)
    print()
    if bad_header:
        raise Phase2Error(f"{bad_header:,} Stories do not start with the expected header tokens")
    tokens = np.concatenate(parts) if parts else np.empty(0, dtype=np.uint16)
    offsets = np.zeros(len(texts) + 1, dtype=np.int64)
    np.cumsum(lengths, out=offsets[1:])
    stats = {
        "stories_total": len(stories),
        "empty_dropped": empty_dropped,
        "stories": len(texts),
        "tokens": int(tokens.size),
        "tokens_per_story_mean": round(float(lengths.mean()), 1) if len(texts) else 0.0,
    }
    return tokens, offsets, stats


def build_token_cache(data_dir, tokenizer_dir, out_dir, force: bool = False) -> dict:
    """Tokenize the Parquet Stories once and cache them (idempotent, with a manifest written last)."""
    import numpy as np
    from tokenizers import Tokenizer

    data_dir, tokenizer_dir, out_dir = Path(data_dir), Path(tokenizer_dir), Path(out_dir)
    if not phase0_complete(data_dir):
        raise Phase2Error(f"No complete Phase 0 data in {data_dir}. Run: python run.py phase0")
    if not (tokenizer_dir / "tokenizer.json").is_file():
        raise Phase2Error(f"No tokenizer in {tokenizer_dir}. Run: python run.py phase1")
    params = cache_params(data_dir, tokenizer_dir)
    if not force and cache_is_complete(out_dir, params):
        return json.loads((out_dir / CACHE_MANIFEST).read_text())

    tok = Tokenizer.from_file(str(tokenizer_dir / "tokenizer.json"))
    special_ids = {t: tok.token_to_id(t) for t in ("[PAD]", "[UNK]", "[BOS]", "[EOS]", "[MASK]")}
    if None in special_ids.values() or tok.get_vocab_size() > 65536:
        raise Phase2Error("tokenizer is missing special tokens or has a vocabulary too large for uint16")
    header_ids = tok.encode(HEADER, add_special_tokens=False).ids

    tmp = out_dir.parent / (out_dir.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    splits: dict[str, dict] = {}
    for split in SPLITS:
        print(f"[{split}] reading and tokenizing ...", flush=True)
        tokens, offsets, stats = tokenize_split(tok, read_split(data_dir, split), special_ids["[BOS]"],
                                                special_ids["[EOS]"], header_ids)
        if int(tokens.max(initial=0)) >= tok.get_vocab_size():
            raise Phase2Error(f"{split}: token id outside the vocabulary")
        np.save(tmp / f"{split}_tokens.npy", tokens)
        np.save(tmp / f"{split}_offsets.npy", offsets)
        splits[split] = stats
        print(f"[{split}] {stats['stories']:,} Stories ({stats['empty_dropped']:,} empty dropped), "
              f"{stats['tokens'] / 1e6:.1f}M tokens", flush=True)
    manifest = {
        "params": params,
        "special_ids": special_ids,
        "header_len": len(header_ids),
        "protected_per_story": 1 + len(header_ids),  # [BOS] + header
        "vocab_size": tok.get_vocab_size(),
        "splits": splits,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    }
    (tmp / CACHE_MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n")  # last: marks the folder complete
    if out_dir.exists():
        shutil.rmtree(out_dir)
    tmp.rename(out_dir)
    return manifest


class StoryBlocks:
    """Dataset of fixed-length blocks; items are {"input_ids": LongTensor[L], "protect": BoolTensor[L]}.

    Stories longer than `seq_len` are dropped; blocks start at a Story's [BOS] and are filled greedily with the
    following whole Stories; the remainder is [PAD]. `sample_stories` keeps a seeded random subset of the eligible
    Stories (used for the fixed "train-eval" slice). Works with torch's DataLoader (len + getitem).
    """

    def __init__(self, cache_dir, split: str, seq_len: int = DEFAULT_SEQ_LEN,
                 sample_stories: int | None = None, seed: int = 0):
        import numpy as np

        cache_dir = Path(cache_dir)
        m = json.loads((cache_dir / CACHE_MANIFEST).read_text())
        self.seq_len = seq_len
        self.split = split
        self.pad_id = m["special_ids"]["[PAD]"]
        self.protected_per_story = m["protected_per_story"]
        self.tokens = np.load(cache_dir / f"{split}_tokens.npy", mmap_mode="r")
        self.offsets = np.load(cache_dir / f"{split}_offsets.npy")
        lengths = np.diff(self.offsets)
        eligible = np.flatnonzero(lengths <= seq_len)
        self.too_long = int(len(lengths) - len(eligible))
        if sample_stories is not None and sample_stories < len(eligible):
            eligible = np.array(sorted(random.Random(seed).sample(eligible.tolist(), sample_stories)), dtype=np.int64)
        self.eligible = eligible
        elig_len = lengths[eligible].tolist()
        bounds: list[tuple[int, int]] = []
        i, n = 0, len(elig_len)
        while i < n:
            j, used = i, 0
            while j < n and used + elig_len[j] <= seq_len:
                used += elig_len[j]
                j += 1
            bounds.append((i, j))  # j > i: every eligible Story fits an empty block
            i = j
        self.bounds = bounds
        self.story_tokens = int(sum(elig_len))

    def __len__(self) -> int:
        return len(self.bounds)

    def arrays(self, index: int):
        """(input_ids int64[L], protect bool[L], n_stories) for block `index` (numpy, no torch)."""
        import numpy as np

        a, b = self.bounds[index]
        ids = np.full(self.seq_len, self.pad_id, dtype=np.int64)
        protect = np.zeros(self.seq_len, dtype=bool)
        pos = 0
        for k in self.eligible[a:b]:
            s, e = int(self.offsets[k]), int(self.offsets[k + 1])
            ids[pos:pos + e - s] = self.tokens[s:e]
            protect[pos:pos + self.protected_per_story] = True
            pos += e - s
        return ids, protect, b - a

    def __getitem__(self, index: int):
        import torch

        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        ids, protect, _ = self.arrays(index)
        return {"input_ids": torch.from_numpy(ids), "protect": torch.from_numpy(protect)}

    def stats(self) -> dict:
        n = len(self)
        return {
            "split": self.split,
            "seq_len": self.seq_len,
            "stories_kept": len(self.eligible),
            "dropped_too_long": self.too_long,  # for the whole split, even when sample_stories is used
            "blocks": n,
            "fill": round(self.story_tokens / (n * self.seq_len), 4) if n else 0.0,
            "stories_per_block": round(len(self.eligible) / n, 2) if n else 0.0,
            "protected_share": round(len(self.eligible) * self.protected_per_story / (n * self.seq_len), 4) if n else 0.0,
        }


def build_sft_example(tokenizer, prompt: str, response: str, seq_len: int):
    """Phase 8 helper: prompt tokens protected, response padded with [EOS] (LLaDA SFT)."""
    not_built("Phase 8 (fine-tuning)", "which instruction dataset to use")


def inspect_block(blocks: StoryBlocks, tok, index: int) -> str:
    """Decode a block with protected runs in [[...]] and padding summarised."""
    ids, protect, n = blocks.arrays(index)
    out: list[str] = []
    run: list[int] = []
    prot_run = False
    for pid, pr in zip(ids.tolist(), protect.tolist()):
        if pid == blocks.pad_id:
            break
        if pr != prot_run and run:
            txt = tok.decode(run, skip_special_tokens=False)
            out.append(f"[[{txt}]]" if prot_run else txt)
            run = []
        prot_run = pr
        run.append(pid)
    if run:
        txt = tok.decode(run, skip_special_tokens=False)
        out.append(f"[[{txt}]]" if prot_run else txt)
    n_pad = int((ids == blocks.pad_id).sum())
    return f"block {index}: {n} Stories, {blocks.seq_len - n_pad} tokens, {n_pad} [PAD]\n" + "".join(out)


def main(argv: list[str] | None = None) -> int:
    from tokenizers import Tokenizer

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=None, help="Phase 0 output (default: DLM/data, or Drive on Colab)")
    p.add_argument("--tokenizer-dir", type=Path, default=ROOT / "tokenizer", help="Phase 1 output (default: DLM/tokenizer)")
    p.add_argument("--out", type=Path, default=None, help="token cache folder (default: <data-dir>/tokens)")
    p.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN, help="block length for the statistics / --inspect")
    p.add_argument("--force", action="store_true", help="re-tokenize even if the cache is complete")
    p.add_argument("--inspect", type=int, default=None, metavar="N", help="print N decoded train blocks (protected runs in [[ ]])")
    args = p.parse_args(argv)
    data_dir = (args.data_dir or default_out_dir()).expanduser().resolve()
    out = (args.out or data_dir / CACHE_NAME).expanduser().resolve()
    tok_dir = args.tokenizer_dir.expanduser().resolve()
    try:
        m = build_token_cache(data_dir, tok_dir, out, force=args.force)
        print(f"Token cache ready in {out}")
        for s, st in m["splits"].items():
            print(f"  {s}: {st['stories']:,} Stories, {st['tokens'] / 1e6:.1f}M tokens "
                  f"({st['empty_dropped']:,} empty dropped, mean {st['tokens_per_story_mean']} tokens/Story)")
        sets = {
            "train": StoryBlocks(out, "train", args.seq_len),
            "validation": StoryBlocks(out, "validation", args.seq_len),
            "train-eval": StoryBlocks(out, "train", args.seq_len, sample_stories=TRAIN_EVAL_STORIES, seed=0),
        }
        print(f"\nBlocks at seq_len={args.seq_len}:")
        for name, ds in sets.items():
            s = ds.stats()
            dropped = "n/a (slice)" if name == "train-eval" else f"{s['dropped_too_long']:,} too long dropped"
            print(f"  {name:<10} {s['blocks']:>9,} blocks | {s['stories_kept']:>9,} Stories kept, "
                  f"{dropped} | fill {s['fill']:.1%} | "
                  f"{s['stories_per_block']} Stories/block | protected {s['protected_share']:.1%}")
        if args.inspect:
            tok = Tokenizer.from_file(str(tok_dir / "tokenizer.json"))
            for i in range(min(args.inspect, len(sets["train"]))):
                print("\n" + inspect_block(sets["train"], tok, i)[:1500])
    except Phase2Error as e:
        print(f"PHASE 2 FAILED: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"PHASE 2 FAILED ({type(e).__name__}): {e}", file=sys.stderr)
        return 1
    return 0
