"""Offline logic tests for dlm/download.py (no network): python run.py test  (or python tests/test_download.py)"""
import json
import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

import sys
from pathlib import Path as _P

sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
import dlm.download as p0  # noqa: E402


def write_parquet(path: Path, n: int):
    pq.write_table(pa.table({"text": [f"story {i}" for i in range(n)]}), path)


def make_complete(out: Path, n_train=5, n_val=3, revision="abc123"):
    write_parquet(out / "train.parquet", n_train)
    write_parquet(out / "validation.parquet", n_val)
    (out / p0.MANIFEST).write_text(json.dumps({
        "dataset": p0.DATASET, "revision": revision, "downloaded_at": "2026-01-01T00:00:00+00:00",
        "splits": {
            s: {"stories": n, "empty": 0, "avg_chars": 7.0, "avg_words": 2.0, "file_bytes": 1000}
            for s, n in (("train", n_train), ("validation", n_val))
        },
    }))


def expect_error(fn, *a):
    try:
        fn(*a)
    except p0.Phase0Error:
        return True
    return False


def main():
    checks = 0

    def ok(cond, msg):
        nonlocal checks
        assert cond, msg
        checks += 1

    with tempfile.TemporaryDirectory() as d:
        out = Path(d)
        ok(not p0.is_complete(out), "empty folder is not complete")
        make_complete(out)
        ok(p0.is_complete(out), "matching manifest + parquet is complete")
        ok(p0.is_complete(out, "abc123"), "same revision is complete")
        ok(not p0.is_complete(out, "other"), "different revision is not complete")

        (out / p0.MANIFEST).unlink()
        ok(not p0.is_complete(out), "missing manifest is not complete")

        make_complete(out)
        write_parquet(out / "train.parquet", 4)  # row count no longer matches manifest
        ok(not p0.is_complete(out), "row-count mismatch is not complete")

        make_complete(out)
        (out / "validation.parquet").write_bytes(b"truncated")
        ok(not p0.is_complete(out), "corrupt parquet is not complete")

        make_complete(out)
        (out / p0.MANIFEST).write_text("{not json")
        ok(not p0.is_complete(out), "corrupt manifest is not complete")

        make_complete(out)
        (out / "validation.parquet").unlink()
        ok(not p0.is_complete(out), "missing split file is not complete")

    ok(expect_error(p0.check_counts, "train", 10, 0), "too few train Stories fails")
    ok(expect_error(p0.check_counts, "validation", 5, 0), "too few validation Stories fails")
    ok(expect_error(p0.check_counts, "train", 2_100_000, 5_000), "a large share of empty Stories fails")
    p0.check_counts("train", 2_119_719, 230)  # the real train split: tolerated and recorded
    checks += 1
    p0.check_counts("train", 2_100_000, 0)
    p0.check_counts("validation", 22_000, 0)
    checks += 2

    s = p0.split_stats(["a b c", "d e", "  "])
    ok(s["stories"] == 3 and s["empty"] == 1, "stats count Stories and empties")
    ok(s["avg_words"] == round(5 / 3, 1), "avg words")

    texts = [f"s{i}" for i in range(100)]
    ok(p0.pick_samples(texts) == p0.pick_samples(texts), "samples are deterministic")
    ok(len(p0.pick_samples(["only"], k=3)) == 1, "samples capped at corpus size")

    with tempfile.TemporaryDirectory() as d:
        out = Path(d)
        pq.write_table(pa.table({"text": [f"story {i}" for i in range(10)]}), out / "train.parquet", row_group_size=3)
        ok(p0.read_stories(out / "train.parquet", 0, 2) == ["story 0", "story 1"], "read from the start")
        ok(p0.read_stories(out / "train.parquet", 4, 3) == ["story 4", "story 5", "story 6"], "read across row groups")
        ok(p0.read_stories(out / "train.parquet", 8, 5) == ["story 8", "story 9"], "read stops at the end")
        make_complete(out, n_train=10, n_val=3)
        ok(p0.inspect_data(out, "train", 0, 2) == 0, "inspect succeeds on complete data")
        ok(p0.inspect_data(out, "train", 99, 2) == 1, "inspect rejects an out-of-range start")
    with tempfile.TemporaryDirectory() as d:
        ok(p0.inspect_data(Path(d), "train", 0, 2) == 1, "inspect fails when nothing is downloaded")

    ok(p0.default_out_dir().name in ("data", "diffusion_lm_outputs") or p0.default_out_dir().parts[-1] == "data",
       "default out dir is a data folder")
    print(f"All {checks} Phase 0 logic checks passed.")


if __name__ == "__main__":
    main()
