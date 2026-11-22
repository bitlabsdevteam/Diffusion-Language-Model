"""Offline logic tests for dlm/cleaning.py and dlm/tokenizer.py (synthetic data, no network):
    python run.py test   (or python tests/test_tokenizer.py)
"""
import json
import random
import sys
import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import dlm.download as p0  # noqa: E402
import dlm.tokenizer as p1  # noqa: E402
from dlm.cleaning import clean_story  # noqa: E402


def synthetic_stories(n, seed=0):
    rng = random.Random(seed)
    words = "the a cat dog sun ran big small happy sad tree house lily tom said went home".split()
    return [" ".join(rng.choice(words) for _ in range(rng.randint(8, 40))) + "." for _ in range(n)]


def write_phase0(data_dir: Path, train, val, revision="rev123"):
    data_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({"text": train}), data_dir / "train.parquet", row_group_size=50)
    pq.write_table(pa.table({"text": val}), data_dir / "validation.parquet")
    (data_dir / p0.MANIFEST).write_text(json.dumps({
        "dataset": p0.DATASET, "revision": revision, "downloaded_at": "2026-01-01T00:00:00+00:00",
        "splits": {s: {"stories": len(x), "empty": 0, "avg_chars": 1.0, "avg_words": 1.0, "file_bytes": 1}
                   for s, x in (("train", train), ("validation", val))},
    }))


def main():
    checks = 0

    def ok(cond, msg):
        nonlocal checks
        assert cond, msg
        checks += 1

    # ---- cleaning ----
    ok(clean_story("Donâ€™t") == "Don't", "apostrophe repaired")
    ok(clean_story("â€œHi!â€ he said") == '"Hi!" he said', "open/close double quotes (lost last byte) repaired")
    ok(clean_story("said,â€Itâ€™s fine") == 'said,"It\'s fine', "opening quote whose last byte was lost")
    ok(clean_story("â€˜x") == "'x", "left single quote")
    ok(clean_story("aâ€“b") == "a-b" and clean_story("aâ€”b") == "a-b", "dashes")
    ok(clean_story("waitâ€¦") == "wait...", "ellipsis")
    ok(clean_story("aâ€‹b") == "ab" and clean_story("aâ€Šb") == "a b", "zero-width / hair space")
    ok(clean_story("“Hi” ‘x’") == "\"Hi\" 'x'", "real curly quotes fold to ASCII")
    ok(clean_story("  hello \n") == "hello", "surrounding whitespace stripped")
    ok(clean_story("") == "" and clean_story("  \n") == "", "empty stays empty")
    ok(clean_story("plain text, no change.") == "plain text, no change.", "clean text untouched")
    ok(clean_story("café Ã© naïve") == "café Ã© naïve", "unrelated non-ASCII is untouched")
    for s in ["â€œA,â€ B â€™", "x", "â€"]:
        ok(clean_story(clean_story(s)) == clean_story(s), f"cleaning is idempotent: {s!r}")
    ok("â€" not in clean_story("â€œa â€™ b â€ c â€I d"), "no marker left")

    # ---- template ----
    ok(p1.format_story(" Once. ") == "<|user|>\nWrite a short story.\n<|assistant|>\nOnce.\n<|end|>\n", "chat template")
    ok(p1.format_story("Donâ€™t") .endswith("Don't\n<|end|>\n"), "template uses cleaned text")

    # ---- sampling ----
    stories = ["a", "", "  ", "b", "câ€™", "d", "e", "f"]
    s1, st = p1.sample_training_stories(stories, 3, seed=0)
    s2, _ = p1.sample_training_stories(stories, 3, seed=0)
    s3, _ = p1.sample_training_stories(stories, 3, seed=1)
    ok(s1 == s2, "sampling is deterministic")
    ok(all(x.strip() for x in s1) and len(s1) == 3, "no empty Stories sampled")
    ok(st["eligible_stories"] == 6, "eligible count excludes empty/blank")
    ok(len(p1.sample_training_stories(stories, 100, seed=0)[0]) == 6, "sample capped at eligible Stories")
    many = [f"story {i}" for i in range(1000)]
    ok(p1.sample_training_stories(many, 50, seed=0)[0] != p1.sample_training_stories(many, 50, seed=1)[0],
       "different seeds give different samples")

    # ---- tokenizer on synthetic data ----
    VOCAB = 320
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        data_dir, out = root / "data", root / "tokenizer"
        train = synthetic_stories(400, 0) + ["", "   "]
        val = synthetic_stories(60, 1) + ["Donâ€™t go â€œthereâ€ now ðŸ˜€ café"]
        write_phase0(data_dir, train, val)

        m = p1.build(data_dir, out, VOCAB, 300, 0)
        ok((out / "tokenizer.json").is_file() and (out / "tokenizer_config.json").is_file(), "HF files written")
        ok((out / p1.MANIFEST).is_file() and not (root / "tokenizer.tmp").exists(), "manifest written, tmp dir gone")
        ok(m["verification"]["round_trip_exact"] and m["verification"]["unk_tokens"] == 0, "verification passed")

        tok = p1.load_tokenizer(out)
        ok(len(tok) == VOCAB, "vocab size exact")
        ids = {t: tok.convert_tokens_to_ids(t) for t in p1.SPECIAL_TOKENS}
        ok(ids == p1.EXPECTED_IDS, "special IDs fixed")
        ok(tok.pad_token_id == 0 and tok.bos_token_id == 2 and tok.eos_token_id == 3 and tok.mask_token_id == 4,
           "HF special-token attributes")

        from tokenizers import Tokenizer
        raw = Tokenizer.from_file(str(out / "tokenizer.json"))  # loads without transformers (Phase 2 fallback)
        ok(raw.encode("the cat", add_special_tokens=False).ids == tok.encode("the cat", add_special_tokens=False),
           "tokenizer.json loads through `tokenizers` alone and agrees")
        mfj = json.loads((out / p1.MANIFEST).read_text())
        ok(mfj["template"] == "<|user|>\nWrite a short story.\n<|assistant|>\n{story}\n<|end|>\n",
           "manifest template holds real newlines")
        ok("transformers_version" in mfj, "manifest records the transformers version")

        weird = "emoji 😀 café ÿ \x00 ünïcödé 日本語 â€ [MASK] <|end|>\n\ttab"
        enc = tok.encode(weird, add_special_tokens=False)
        ok(tok.decode(enc, skip_special_tokens=False) == weird, "lossless round trip for arbitrary text")
        plain = "emoji 😀 café ÿ \x00 ünïcödé 日本語 â€ \n\ttab"
        pids = tok.encode(plain, add_special_tokens=False)
        ok(tok.unk_token_id not in pids, "no [UNK] for any byte sequence")
        ok(tok.decode(pids, skip_special_tokens=False) == plain, "round trip with all kinds of characters")
        ok(tok.encode("hello", add_special_tokens=False) == tok("hello", add_special_tokens=False)["input_ids"],
           "encode paths agree")
        ok(tok.bos_token_id not in tok.encode("hello") and tok.eos_token_id not in tok.encode("hello"),
           "no automatic [BOS]/[EOS], even with add_special_tokens=True")

        # idempotence / staleness
        params = p1.request_params(VOCAB, 300, 0, "rev123")
        ok(p1.is_complete(out, params), "complete with matching params")
        ok(not p1.is_complete(out, p1.request_params(VOCAB + 1, 300, 0, "rev123")), "vocab change -> stale")
        ok(not p1.is_complete(out, p1.request_params(VOCAB, 301, 0, "rev123")), "sample size change -> stale")
        ok(not p1.is_complete(out, p1.request_params(VOCAB, 300, 1, "rev123")), "seed change -> stale")
        ok(not p1.is_complete(out, p1.request_params(VOCAB, 300, 0, "rev999")), "Phase 0 revision change -> stale")
        stale = dict(params, cleaning_version=params["cleaning_version"] + 1)
        ok(not p1.is_complete(out, stale), "cleaning version change -> stale")
        ok(p1.main(["--data-dir", str(data_dir), "--out", str(out), "--vocab-size", str(VOCAB),
                    "--n-stories", "300"]) == 0, "second run is a no-op that succeeds")
        t0 = (out / p1.MANIFEST).stat().st_mtime_ns
        p1.main(["--data-dir", str(data_dir), "--out", str(out), "--vocab-size", str(VOCAB), "--n-stories", "300"])
        ok((out / p1.MANIFEST).stat().st_mtime_ns == t0, "no-op did not rewrite the manifest")

        (out / p1.MANIFEST).write_text("{broken")
        ok(not p1.is_complete(out, params), "corrupt manifest -> not complete")
        (out / p1.MANIFEST).unlink()
        ok(not p1.is_complete(out, params), "missing manifest -> not complete")

        # same inputs, same tokenizer (reproducible)
        out2 = root / "tokenizer2"
        p1.build(data_dir, out2, VOCAB, 300, 0)
        ok((out / "tokenizer.json").read_text() == (out2 / "tokenizer.json").read_text(), "training is reproducible")

        # failures are loud
        ok(p1.main(["--data-dir", str(root / "nope"), "--out", str(root / "x")]) == 1, "missing Phase 0 data -> exit 1")
        bad = root / "tokenizer_bad"
        ok(p1.main(["--data-dir", str(data_dir), "--out", str(bad), "--vocab-size", "100", "--n-stories", "300"]) == 1
           and not (bad / p1.MANIFEST).exists(), "impossible vocab size fails without writing a manifest")

        # verify() catches a broken tokenizer: one with a normalizer is not lossless
        from tokenizers.normalizers import NFKC
        broken = p1.train_tokenizer((p1.format_story(s) for s in synthetic_stories(100)), VOCAB)
        broken.normalizer = NFKC()
        try:
            p1.verify(broken, ["™ symbol"] + val)
            caught = False
        except p1.Phase1Error:
            caught = True
        ok(caught, "verify() rejects a lossy tokenizer")

        # --encode
        p1.main(["--out", str(out2), "--encode", "the cat"])
        ok(p1.main(["--out", str(root / "none"), "--encode", "x"]) == 1, "--encode without a tokenizer -> exit 1")

    print(f"All {checks} Phase 1 logic checks passed.")


if __name__ == "__main__":
    main()
