"""Offline logic tests for dlm/data.py (synthetic corpus, no network): python run.py test  (or python tests/test_data.py)"""
import json
import random
import sys
import tempfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import dlm.data as d  # noqa: E402
import dlm.download as p0  # noqa: E402
import dlm.tokenizer as p1  # noqa: E402

SEQ = 128
VOCAB = 320


def stories(n, seed, lo=8, hi=40):
    rng = random.Random(seed)
    words = "the a cat dog sun ran big small happy sad tree house lily tom said went home".split()
    return [" ".join(rng.choice(words) for _ in range(rng.randint(lo, hi))) + "." for _ in range(n)]


def write_phase0(data_dir, train, val, rev="rev1"):
    data_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({"text": train}), data_dir / "train.parquet", row_group_size=50)
    pq.write_table(pa.table({"text": val}), data_dir / "validation.parquet")
    (data_dir / p0.MANIFEST).write_text(json.dumps({
        "dataset": p0.DATASET, "revision": rev, "downloaded_at": "2026-01-01T00:00:00+00:00",
        "splits": {s: {"stories": len(x), "empty": 0, "avg_chars": 1.0, "avg_words": 1.0, "file_bytes": 1}
                   for s, x in (("train", train), ("validation", val))}}))


def main():
    checks = 0

    def ok(cond, msg):
        nonlocal checks
        assert cond, msg
        checks += 1

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        data_dir, tok_dir, cache = root / "data", root / "tokenizer", root / "data" / "tokens"
        short = stories(400, 0)
        long_ = stories(5, 9, lo=300, hi=320)  # ~340+ tokens: longer than SEQ
        moji = ["Donâ€™t go â€œthereâ€ now, said Tom."]
        train = short[:200] + ["", "   "] + long_ + moji + short[200:]
        val = stories(60, 1) + ["", long_[0]]
        write_phase0(data_dir, train, val)
        p1.build(data_dir, tok_dir, VOCAB, 300, 0)

        m = d.build_token_cache(data_dir, tok_dir, cache)
        ok((cache / d.CACHE_MANIFEST).is_file() and not (root / "data" / "tokens.tmp").exists(), "cache + manifest written")
        tr = m["splits"]["train"]
        ok(tr["stories_total"] == len(train) and tr["empty_dropped"] == 2 and tr["stories"] == len(train) - 2,
           "empty Stories dropped and counted")
        ok(m["splits"]["validation"]["empty_dropped"] == 1, "validation empties counted")
        sid = m["special_ids"]
        ok(sid == {"[PAD]": 0, "[UNK]": 1, "[BOS]": 2, "[EOS]": 3, "[MASK]": 4}, "special ids")
        ok(m["protected_per_story"] == 1 + m["header_len"], "protected = [BOS] + header")

        tokens = np.load(cache / "train_tokens.npy")
        offs = np.load(cache / "train_offsets.npy")
        ok(tokens.dtype == np.uint16 and len(offs) == tr["stories"] + 1 and offs[-1] == len(tokens), "arrays consistent")
        ok(all(tokens[offs[i]] == 2 and tokens[offs[i + 1] - 1] == 3 for i in range(len(offs) - 1)),
           "every Story starts with [BOS] and ends with [EOS]")
        ok(int((tokens == 2).sum()) == tr["stories"] and int((tokens == 3).sum()) == tr["stories"],
           "[BOS]/[EOS] occur only at Story boundaries")
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(str(tok_dir / "tokenizer.json"))
        kept = [s for s in train if s.strip()]
        for i in (0, 100, 205, len(kept) - 1):
            got = tok.decode(tokens[offs[i] + 1:offs[i + 1] - 1].tolist(), skip_special_tokens=False)
            ok(got == p1.format_story(kept[i]), f"Story {i} round-trips through the cache")
        mi = kept.index(moji[0])
        got = tok.decode(tokens[offs[mi] + 1:offs[mi + 1] - 1].tolist(), skip_special_tokens=False)
        ok("â€" not in got and "Don't go \"there\" now" in got, "mojibake repaired in the cache")
        hl = m["header_len"]
        ok(all(tok.decode(tokens[offs[i] + 1:offs[i] + 1 + hl].tolist(), skip_special_tokens=False) == d.HEADER
               for i in range(0, len(offs) - 1, 37)), "header tokens decode to the header")

        # idempotence / staleness / force
        t0 = (cache / d.CACHE_MANIFEST).stat().st_mtime_ns
        d.build_token_cache(data_dir, tok_dir, cache)
        ok((cache / d.CACHE_MANIFEST).stat().st_mtime_ns == t0, "second run does not rewrite the cache")
        params = d.cache_params(data_dir, tok_dir)
        ok(d.cache_is_complete(cache, params), "complete with matching params")
        ok(not d.cache_is_complete(cache, dict(params, phase0_revision="other")), "Phase 0 revision change -> stale")
        ok(not d.cache_is_complete(cache, dict(params, tokenizer_sha256="x")), "tokenizer change -> stale")
        ok(not d.cache_is_complete(cache, dict(params, cleaning_version=99)), "cleaning version change -> stale")
        ok(not d.cache_is_complete(cache, dict(params, format_version=99)), "format version change -> stale")
        d.build_token_cache(data_dir, tok_dir, cache, force=True)
        ok((cache / d.CACHE_MANIFEST).stat().st_mtime_ns != t0, "--force rebuilds")
        ok(np.array_equal(np.load(cache / "train_tokens.npy"), tokens), "rebuild is reproducible")

        # blocks
        lengths = np.diff(offs)
        blocks = d.StoryBlocks(cache, "train", SEQ)
        ok(blocks.too_long == int((lengths > SEQ).sum()) and blocks.too_long >= 5, "long Stories counted as dropped")
        ok(len(blocks.eligible) == int((lengths <= SEQ).sum()), "eligible = Stories that fit")
        seen = []
        for i in range(len(blocks)):
            ids, protect, n = blocks.arrays(i)
            if i == 0:
                ok(len(ids) == SEQ and ids.dtype == np.int64 and protect.dtype == bool, "block shape/dtype")
            real = ids != 0
            n_real = int(real.sum())
            ok(real[:n_real].all() and not real[n_real:].any(), f"block {i}: PAD only in the tail")
            ok(ids[0] == 2, f"block {i} starts at a Story's [BOS]")
            ok(int((ids == 2).sum()) == n == int((ids == 3).sum()), f"block {i}: whole Stories only")
            a, b = blocks.bounds[i]
            seen.extend(blocks.eligible[a:b].tolist())
            pos, exp = 0, np.zeros(SEQ, dtype=bool)
            for k in blocks.eligible[a:b]:
                L = int(lengths[k])
                ok(np.array_equal(ids[pos:pos + L], tokens[offs[k]:offs[k + 1]]), "Story tokens copied exactly")
                exp[pos:pos + 1 + hl] = True
                pos += L
            ok(np.array_equal(protect, exp), f"block {i}: protect = [BOS]+header of every Story, nothing else")
            ok(not protect[~real].any(), "PAD is never protected")
            if i + 1 < len(blocks):  # greedy: the next Story would not have fit
                nxt = blocks.eligible[blocks.bounds[i + 1][0]]
                ok(n_real + int(lengths[nxt]) > SEQ, f"block {i} is greedily full")
        ok(seen == blocks.eligible.tolist(), "every eligible Story appears exactly once, in order (no overlap)")
        ok(all(lengths[k] <= SEQ for k in seen), "no dropped Story in any block")
        st = blocks.stats()
        ok(st["blocks"] == len(blocks) and 0.5 < st["fill"] <= 1.0 and st["dropped_too_long"] == blocks.too_long, "stats")

        item = blocks[0]
        ok(item["input_ids"].dtype == torch.long and item["protect"].dtype == torch.bool, "torch item dtypes")
        ok(int(item["input_ids"].max()) < VOCAB, "ids inside the vocabulary")
        ok(blocks[-1]["input_ids"].shape == (SEQ,), "negative index")
        try:
            blocks[len(blocks)]
            raised = False
        except IndexError:
            raised = True
        ok(raised, "out-of-range index raises")
        batch = next(iter(torch.utils.data.DataLoader(blocks, batch_size=4, shuffle=True)))
        ok(batch["input_ids"].shape == (4, SEQ) and batch["protect"].shape == (4, SEQ), "works with DataLoader")

        # other seq_len needs no re-tokenize and changes the drop count
        small = d.StoryBlocks(cache, "train", 64)
        ok(small.too_long > blocks.too_long, "smaller block drops more Stories")
        big = d.StoryBlocks(cache, "train", 1024)
        ok(big.too_long == 0, "a large block drops nothing")

        # train-eval slice + validation
        a1 = d.StoryBlocks(cache, "train", SEQ, sample_stories=50, seed=0)
        a2 = d.StoryBlocks(cache, "train", SEQ, sample_stories=50, seed=0)
        a3 = d.StoryBlocks(cache, "train", SEQ, sample_stories=50, seed=1)
        ok(len(a1.eligible) == 50 and a1.eligible.tolist() == a2.eligible.tolist(), "train-eval slice deterministic")
        ok(a1.eligible.tolist() != a3.eligible.tolist(), "different seed -> different slice")
        v = d.StoryBlocks(cache, "validation", SEQ)
        ok(len(v) > 0 and v.too_long >= 1, "validation blocks, long Story dropped there too")

        # inspect + CLI
        text = d.inspect_block(blocks, tok, 0)
        ok("[[" in text and "<|user|>" in text and "Stories" in text.splitlines()[0], "inspect_block output")
        ok(d.main(["--data-dir", str(data_dir), "--tokenizer-dir", str(tok_dir), "--seq-len", str(SEQ),
                   "--inspect", "1"]) == 0, "CLI succeeds")
        ok(d.main(["--data-dir", str(data_dir), "--tokenizer-dir", str(root / "none")]) == 1, "missing tokenizer -> exit 1")
        ok(d.main(["--data-dir", str(root / "nope"), "--tokenizer-dir", str(tok_dir)]) == 1, "missing Phase 0 -> exit 1")

        # failure is loud and leaves no cache: a Phase 0 folder whose manifest disagrees with its files
        bad = root / "badcache"
        pq.write_table(pa.table({"text": ["x y z"] * 3}), data_dir / "validation.parquet")
        try:
            d.build_token_cache(data_dir, tok_dir, bad)
            loud = False
        except d.Phase2Error:
            loud = True
        ok(loud and not (bad / d.CACHE_MANIFEST).exists(), "inconsistent Phase 0 data fails without a cache")

    try:
        d.build_sft_example(None, "p", "r", 8)
        sft = False
    except NotImplementedError:
        sft = True
    ok(sft, "SFT example builder is still an explicit skeleton")
    print(f"All {checks} Phase 2 data checks passed.")


if __name__ == "__main__":
    main()
