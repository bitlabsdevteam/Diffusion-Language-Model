#!/usr/bin/env python3
"""Phase 0: download TinyStories from Hugging Face into Parquet files.

Writes, into the output directory:
    train.parquet, validation.parquet   one `text` column, one Story per row
    manifest.json                       dataset, revision, counts, date (written last)

Re-running is a no-op once manifest.json and both Parquet files agree.
A download that is interrupted or has implausible counts exits non-zero and leaves no
manifest.json, so an incomplete folder never looks complete.

Usage (from DLM/; `python run.py phase0 ...` is the same as `python -m dlm.download ...`):
    python run.py phase0                  # -> DLM/data/ (or Google Drive on Colab)
    python run.py phase0 --out /some/dir
    python run.py phase0 --revision <commit>   # pin a dataset revision
    python run.py phase0 --force               # re-download even if complete
    python run.py phase0 --inspect             # look inside the Parquet files (no download)
    python run.py phase0 --inspect --split validation --start 100 --n 5
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import random
import sys
from pathlib import Path

DATASET = "roneneldan/TinyStories"
SPLITS = ("train", "validation")
# Sanity floors, not exact counts: TinyStories has ~2.1M train / ~22k validation Stories.
MIN_STORIES = {"train": 2_000_000, "validation": 20_000}
# The raw TinyStories train split contains a few hundred empty rows (~0.01%). Phase 0 keeps the
# data unchanged and records them; only an implausible share is treated as a broken download.
MAX_EMPTY_FRACTION = 0.001
ROOT = Path(__file__).resolve().parents[1]  # DLM/
COLAB_DRIVE = Path("/content/drive/MyDrive")
MANIFEST = "manifest.json"
SAMPLE_SEED = 0


class Phase0Error(RuntimeError):
    """Raised when the download cannot be trusted."""


def default_out_dir() -> Path:
    """Colab with Drive mounted -> Drive folder; otherwise DLM/data next to this script."""
    if COLAB_DRIVE.is_dir():
        return COLAB_DRIVE / "diffusion_lm_outputs" / "data"
    return ROOT / "data"


def parquet_rows(path: Path) -> int:
    import pyarrow.parquet as pq

    return pq.ParquetFile(path).metadata.num_rows


def read_manifest(out: Path) -> dict | None:
    path = out / MANIFEST
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def is_complete(out: Path, revision: str | None = None) -> bool:
    """True when the manifest and the Parquet files agree (and match `revision` if given)."""
    m = read_manifest(out)
    if not m or m.get("dataset") != DATASET:
        return False
    if revision is not None and m.get("revision") != revision:
        return False
    for split in SPLITS:
        f = out / f"{split}.parquet"
        info = m.get("splits", {}).get(split)
        if not f.is_file() or not info:
            return False
        try:
            if parquet_rows(f) != info["stories"]:
                return False
        except Exception:  # corrupt / truncated file
            return False
    return True


def check_counts(split: str, n: int, empty: int) -> None:
    if n < MIN_STORIES[split]:
        raise Phase0Error(
            f"{split}: only {n:,} Stories (expected at least {MIN_STORIES[split]:,}); "
            "the download looks incomplete."
        )
    if empty / n > MAX_EMPTY_FRACTION:
        raise Phase0Error(
            f"{split}: {empty:,} of {n:,} Stories are empty (more than {MAX_EMPTY_FRACTION:.1%}); "
            "the download looks broken."
        )


def split_stats(texts: list[str] | object) -> dict:
    """Counts over a `datasets` column (or list) of Stories."""
    n = len(texts)
    chars = words = empty = 0
    for t in texts:
        chars += len(t)
        words += len(t.split())
        if not t.strip():
            empty += 1
    return {
        "stories": n,
        "empty": empty,  # kept as downloaded; the data pipeline (Phase 2) must drop these
        "avg_chars": round(chars / n, 1) if n else 0.0,
        "avg_words": round(words / n, 1) if n else 0.0,
    }


def pick_samples(texts, k: int = 3, seed: int = SAMPLE_SEED) -> list[str]:
    idx = random.Random(seed).sample(range(len(texts)), min(k, len(texts)))
    return [texts[i] for i in sorted(idx)]


def read_stories(path: Path, start: int, n: int) -> list[str]:
    """Read `n` Stories from row `start` without loading the whole file (reads whole row groups only)."""
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    out: list[str] = []
    offset = 0
    for g in range(pf.num_row_groups):
        rows = pf.metadata.row_group(g).num_rows
        if offset + rows > start:
            col = pf.read_row_group(g, columns=["text"]).column("text").to_pylist()
            out.extend(col[max(0, start - offset):])
            if len(out) >= n:
                break
        offset += rows
    return out[:n]


def inspect_data(out: Path, split: str, start: int, n: int) -> int:
    m = read_manifest(out)
    if not m or not is_complete(out):
        print(f"No complete Phase 0 data in {out}. Run: python run.py phase0", file=sys.stderr)
        return 1
    print(f"{m['dataset']} @ {m['revision'][:8]}, downloaded {m['downloaded_at']}")
    for s, info in m["splits"].items():
        print(f"  {s}: {info['stories']:,} Stories, {info['empty']:,} empty, "
              f"avg {info['avg_words']} words, {info['file_bytes'] / 1e6:.0f} MB")
    total = m["splits"][split]["stories"]
    if not 0 <= start < total:
        print(f"--start must be between 0 and {total - 1}", file=sys.stderr)
        return 1
    print(f"\n--- {split}: Stories {start} to {min(start + n, total) - 1} ---")
    for i, t in enumerate(read_stories(out / f"{split}.parquet", start, n), start):
        print(f"[{i}] ({len(t)} chars) {t.strip()[:400] or '<EMPTY>'}{' ...' if len(t) > 400 else ''}\n")
    return 0


def resolve_revision(requested: str | None) -> str:
    from huggingface_hub import HfApi

    return HfApi().dataset_info(DATASET, revision=requested).sha


def download(out: Path, revision: str) -> dict:
    from datasets import load_dataset

    out.mkdir(parents=True, exist_ok=True)
    (out / MANIFEST).unlink(missing_ok=True)  # never leave a stale manifest next to new data
    splits: dict[str, dict] = {}
    samples: dict[str, list[str]] = {}
    for split in SPLITS:
        print(f"[{split}] downloading {DATASET}@{revision[:8]} ...", flush=True)
        ds = load_dataset(DATASET, split=split, revision=revision)
        if ds.column_names != ["text"]:
            raise Phase0Error(f"{split}: unexpected columns {ds.column_names}, expected ['text']")
        stats = split_stats(ds["text"])
        check_counts(split, stats["stories"], stats["empty"])
        tmp = out / f"{split}.parquet.tmp"
        ds.to_parquet(str(tmp))
        if parquet_rows(tmp) != stats["stories"]:
            tmp.unlink(missing_ok=True)
            raise Phase0Error(f"{split}: Parquet row count differs from the downloaded Stories.")
        os.replace(tmp, out / f"{split}.parquet")
        stats["file_bytes"] = (out / f"{split}.parquet").stat().st_size
        splits[split] = stats
        samples[split] = pick_samples(ds["text"])
        print(
            f"[{split}] {stats['stories']:,} Stories, avg {stats['avg_words']} words / "
            f"{stats['avg_chars']} chars, {stats['file_bytes'] / 1e6:.0f} MB",
            flush=True,
        )
        if stats["empty"]:
            print(
                f"[{split}] WARNING: {stats['empty']:,} empty Stories kept as downloaded "
                "(recorded in the manifest; drop them in Phase 2).",
                flush=True,
            )
    manifest = {
        "dataset": DATASET,
        "revision": revision,
        "downloaded_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "format": "parquet, single `text` column, one Story per row",
        "splits": splits,
    }
    tmp = out / (MANIFEST + ".tmp")
    tmp.write_text(json.dumps(manifest, indent=2) + "\n")
    os.replace(tmp, out / MANIFEST)  # written last: its presence means the folder is complete
    return {"manifest": manifest, "samples": samples}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=None, help="output directory (default: DLM/data, or Drive on Colab)")
    p.add_argument("--revision", default=None, help="dataset commit/branch to pin (default: latest main)")
    p.add_argument("--force", action="store_true", help="re-download even if the folder is already complete")
    p.add_argument("--inspect", action="store_true", help="show the manifest and sample Stories from the Parquet files, then exit")
    p.add_argument("--split", choices=SPLITS, default="train", help="with --inspect: which Split (default: train)")
    p.add_argument("--start", type=int, default=0, help="with --inspect: first Story index (default: 0)")
    p.add_argument("--n", type=int, default=3, help="with --inspect: how many Stories to print (default: 3)")
    args = p.parse_args(argv)
    out = (args.out or default_out_dir()).expanduser().resolve()
    if args.inspect:
        return inspect_data(out, args.split, args.start, args.n)

    try:
        if not args.force and is_complete(out, args.revision):
            m = read_manifest(out)
            print(f"Already complete in {out} (revision {m['revision'][:8]}); nothing to download.")
            for s, info in m["splits"].items():
                print(f"  {s}: {info['stories']:,} Stories")
            return 0
        revision = resolve_revision(args.revision)
        result = download(out, revision)
    except Phase0Error as e:
        print(f"PHASE 0 FAILED: {e}", file=sys.stderr)
        return 1
    except Exception as e:  # network error, interrupted download, disk full, ...
        print(f"PHASE 0 FAILED ({type(e).__name__}): {e}", file=sys.stderr)
        return 1

    print(f"\nDone. Wrote {out}")
    for split, texts in result["samples"].items():
        print(f"\n--- 3 sample Stories from {split} ---")
        for i, t in enumerate(texts, 1):
            print(f"[{i}] {t[:300].strip()}{' ...' if len(t) > 300 else ''}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
