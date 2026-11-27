"""SFT data for controllable story generation (TinyStoriesInstruct -> instruction/response token blocks).

Layout of one example, identical to the pre-training Story layout (dlm/data.py) so nothing in the model changes:

    [BOS] <|user|>\\n{instruction}\\n<|assistant|>\\n  {story}\\n<|end|>\\n  [EOS]
    |<---------- protected: never masked ---------->|<------- maskable (the response) ------->|

  * Instruction = "Write a short story." plus a random SUBSET (1 to 4) of the record's fields (Words / Features /
    Summary / Random sentence) in RANDOM order, re-drawn every epoch (TinyStories paper recipe). `plain_prob` of the
    examples (default 15%) get the plain pre-training prompt so unconditional generation does not drift.
  * Padding: whole examples are packed into 512-token blocks like pre-training (greedy, no example split across
    blocks); the rest of a block is [PAD], which is excluded from the loss and from attention keys. We deliberately do
    NOT pad with [EOS] (LLaDA's recipe): that makes [EOS] the dominant target and causes early termination ("EOS
    overflow", Rainbow Padding / BackPlay). See README.md.
  * Determinism: the block at global index g is a pure function of (seed, g), so a resumed run sees exactly the data
    an uninterrupted run would have seen (same property as dlm.train.batch_indices).

Cache (built once by `build_cache`, under fine-tuning/data/): per split `<split>_resp_tokens.npy` (uint16, story +
end tokens, no BOS/EOS), `<split>_resp_offsets.npy`, `<split>_fields.jsonl`, `cache_manifest.json` (written last).
"""
from __future__ import annotations

import _path  # noqa: F401  (puts the DLM repo root on sys.path)

import datetime as dt
import functools
import json
import random
from pathlib import Path

import numpy as np

from dlm.cleaning import CLEANING_VERSION, clean_story
from dlm.tokenizer import PRETRAIN_INSTRUCTION
from ft_parse import Record, iter_records

PLAIN_INSTRUCTION = PRETRAIN_INSTRUCTION  # "Write a short story."
END_SUFFIX = "\n<|end|>\n"
FIELD_ORDER = ("Words", "Features", "Summary", "Random sentence")
CACHE_MANIFEST = "cache_manifest.json"
CACHE_FORMAT_VERSION = 1
CHUNK_EXAMPLES = 4096  # packing granularity of the training stream (blocks never span chunks)
DEFAULT_PLAIN_PROB = 0.15
DEFAULT_SEQ_LEN = 512


def chat_prompt(instruction: str) -> str:
    """The protected text: identical to dlm.data.HEADER when instruction == PLAIN_INSTRUCTION."""
    return f"<|user|>\n{instruction}\n<|assistant|>\n"


# ------------------------------------------------------------------ instructions
def field_texts(fields: dict) -> list[tuple[str, str]]:
    """[(name, text)] for the fields a record really has. `fields` = {"w": [...], "f": [...], "s": str|None, "r": str|None}."""
    out = []
    if fields.get("w"):
        out.append(("Words", ", ".join(fields["w"])))
    if fields.get("f"):
        out.append(("Features", ", ".join(fields["f"])))
    if fields.get("s"):
        out.append(("Summary", fields["s"]))
    if fields.get("r"):
        out.append(("Random sentence", fields["r"]))
    return out


def build_instruction(fields: dict, rng: random.Random, plain_prob: float = DEFAULT_PLAIN_PROB,
                      max_fields: int = 4) -> tuple[str, list[str]]:
    """Random subset (1..max_fields) of the available fields in random order. Returns (instruction, names used).
    With probability `plain_prob`, or when a record has no fields, the plain pre-training instruction is returned."""
    avail = field_texts(fields)
    if not avail or rng.random() < plain_prob:
        return PLAIN_INSTRUCTION, []
    k = rng.randint(1, min(max_fields, len(avail)))
    chosen = rng.sample(avail, k)  # sample() returns a random ORDER too
    lines = [PLAIN_INSTRUCTION] + [f"{name}: {text}" for name, text in chosen]
    return "\n".join(lines), [name for name, _ in chosen]


def record_fields(rec: Record) -> dict:
    return {"w": [clean_story(w) for w in rec.words], "f": list(rec.features),
            "s": clean_story(rec.summary) if rec.summary else None,
            "r": clean_story(rec.sentence) if rec.sentence else None}


# ------------------------------------------------------------------ single example (the Phase 8 stub's contract)
def special_ids(tok) -> dict:
    ids = {n: tok.token_to_id(n) for n in ("[PAD]", "[BOS]", "[EOS]", "[MASK]", "<|end|>")}
    if None in ids.values():
        raise ValueError(f"tokenizer is missing special tokens: {[n for n, i in ids.items() if i is None]}")
    return ids


def encode_prompt(tok, instruction: str) -> list[int]:
    """Token ids of the protected prompt WITHOUT [BOS]."""
    return tok.encode(chat_prompt(clean_story(instruction)), add_special_tokens=False).ids


def encode_response(tok, story: str) -> list[int]:
    return tok.encode(clean_story(story) + END_SUFFIX, add_special_tokens=False).ids


def build_sft_example(tok, prompt: str, response: str, seq_len: int | None = None, pad: bool = False):
    """One example as {"input_ids": int64[n], "protect": bool[n]}, or None when it does not fit `seq_len`.

    `prompt` is the instruction text (the chat tokens are added here); `response` the story. protect = [BOS] + the whole
    prompt; the response, its `<|end|>` and the final [EOS] are maskable. With pad=True the arrays are padded to
    seq_len with [PAD] (never protected, never masked by dlm.diffusion.corrupt)."""
    sp = special_ids(tok)
    p, r = encode_prompt(tok, prompt), encode_response(tok, response)
    n = 1 + len(p) + len(r) + 1
    if seq_len is not None and n > seq_len:
        return None
    ids = np.array([sp["[BOS]"]] + p + r + [sp["[EOS]"]], dtype=np.int64)
    protect = np.zeros(n, dtype=bool)
    protect[:1 + len(p)] = True
    if pad and seq_len is not None:
        out = np.full(seq_len, sp["[PAD]"], dtype=np.int64)
        out[:n] = ids
        pr = np.zeros(seq_len, dtype=bool)
        pr[:n] = protect
        return {"input_ids": out, "protect": pr}
    return {"input_ids": ids, "protect": protect}


# ------------------------------------------------------------------ cache
class CacheError(RuntimeError):
    pass


def build_cache(raw_path, out_dir, split: str, tokenizer_json, keep_prob: float = 1.0, seed: int = 0,
                limit: int | None = None, seq_len: int = DEFAULT_SEQ_LEN, chunk: int = 20_000) -> dict:
    """Parse a TinyStoriesInstruct text file and cache tokenized responses + fields for `split`.

    keep_prob < 1 keeps a seeded random share of the records (streaming, so the 2.66 GB train file never has to fit in
    memory). Records whose response alone cannot fit a block together with the SHORTEST prompt are dropped and counted.
    The manifest is updated after each split; the file set of a split is replaced atomically per file."""
    from tokenizers import Tokenizer

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tok = Tokenizer.from_file(str(tokenizer_json))
    sp = special_ids(tok)
    min_prompt = len(encode_prompt(tok, PLAIN_INSTRUCTION))
    rng = random.Random(seed)
    parts: list[np.ndarray] = []
    lens: list[int] = []
    fields_rows: list[str] = []
    seen = kept = too_long = trailing = 0
    buf_texts: list[str] = []
    buf_fields: list[dict] = []

    def flush():
        nonlocal too_long
        if not buf_texts:
            return
        enc = tok.encode_batch([clean_story(t) + END_SUFFIX for t in buf_texts], add_special_tokens=False)
        for e, fl in zip(enc, buf_fields):
            if 1 + min_prompt + len(e.ids) + 1 > seq_len:
                too_long += 1
                continue
            parts.append(np.asarray(e.ids, dtype=np.uint16))
            lens.append(len(e.ids))
            fields_rows.append(json.dumps(fl, ensure_ascii=False))
        buf_texts.clear()
        buf_fields.clear()

    for rec in iter_records(raw_path):
        seen += 1
        trailing += rec.trailing_dropped
        if keep_prob < 1.0 and rng.random() >= keep_prob:
            continue
        kept += 1
        buf_texts.append(rec.story)
        buf_fields.append(record_fields(rec))
        if len(buf_texts) >= chunk:
            flush()
            print(f"  [{split}] parsed {seen:,} kept {kept:,}", end="\r", flush=True)
        if limit and kept >= limit:
            break
    flush()
    print()
    if not parts:
        raise CacheError(f"no usable records in {raw_path}")
    tokens = np.concatenate(parts)
    offsets = np.zeros(len(lens) + 1, dtype=np.int64)
    np.cumsum(np.asarray(lens, dtype=np.int64), out=offsets[1:])
    if int(tokens.max()) >= tok.get_vocab_size():
        raise CacheError("token id outside the vocabulary")
    np.save(out_dir / f"{split}_resp_tokens.npy", tokens)
    np.save(out_dir / f"{split}_resp_offsets.npy", offsets)
    (out_dir / f"{split}_fields.jsonl").write_text("\n".join(fields_rows) + "\n", encoding="utf-8")
    manifest_path = out_dir / CACHE_MANIFEST
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {"splits": {}}
    manifest.update({"format_version": CACHE_FORMAT_VERSION, "cleaning_version": CLEANING_VERSION,
                     "special_ids": sp, "vocab_size": tok.get_vocab_size(), "seq_len": seq_len,
                     "min_prompt_tokens": min_prompt})
    manifest["splits"][split] = {
        "source": str(raw_path), "keep_prob": keep_prob, "seed": seed, "records_seen": seen, "records_kept": kept,
        "examples": len(lens), "dropped_response_too_long": too_long,
        "trailing_fields_dropped_share": round(trailing / max(seen, 1), 4),
        "response_tokens": int(tokens.size), "response_tokens_mean": round(float(np.mean(lens)), 1),
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")  # last
    return manifest


class SftCache:
    """Read access to a cached split."""

    def __init__(self, cache_dir, split: str):
        cache_dir = Path(cache_dir)
        mf = cache_dir / CACHE_MANIFEST
        if not mf.is_file():
            raise CacheError(f"No fine-tuning cache in {cache_dir}. Run: python fine-tuning/run_ft.py prep")
        self.manifest = json.loads(mf.read_text())
        if split not in self.manifest["splits"]:
            raise CacheError(f"split {split!r} not in the cache ({list(self.manifest['splits'])})")
        self.split = split
        self.sp = self.manifest["special_ids"]
        self.tokens = np.load(cache_dir / f"{split}_resp_tokens.npy", mmap_mode="r")
        self.offsets = np.load(cache_dir / f"{split}_resp_offsets.npy")
        self.fields = [json.loads(line) for line in (cache_dir / f"{split}_fields.jsonl").read_text(encoding="utf-8").splitlines() if line]
        if len(self.fields) != len(self.offsets) - 1:
            raise CacheError("fields file and token offsets disagree: rebuild with `prep --force`")

    def __len__(self) -> int:
        return len(self.fields)

    def response(self, i: int) -> np.ndarray:
        return np.asarray(self.tokens[int(self.offsets[i]):int(self.offsets[i + 1])])


# ------------------------------------------------------------------ packing
def pack(examples: list[tuple[list[int], np.ndarray]], seq_len: int, sp: dict):
    """Greedy-pack (prompt_ids, response_ids) examples, in the given order, into [n, seq_len] blocks.

    Returns (ids uint16 [n, L], protect bool [n, L], n_examples int [n], dropped_too_long). Blocks start at an
    example's [BOS]; an example that does not fit the current block opens a new one; an example longer than a whole
    block is dropped."""
    blocks_ids: list[np.ndarray] = []
    blocks_pr: list[np.ndarray] = []
    counts: list[int] = []
    cur_ids = cur_pr = None
    used = 0
    dropped = 0
    for prompt_ids, resp in examples:
        n = 1 + len(prompt_ids) + len(resp) + 1
        if n > seq_len:
            dropped += 1
            continue
        if cur_ids is None or used + n > seq_len:
            cur_ids = np.full(seq_len, sp["[PAD]"], dtype=np.uint16)
            cur_pr = np.zeros(seq_len, dtype=bool)
            blocks_ids.append(cur_ids)
            blocks_pr.append(cur_pr)
            counts.append(0)
            used = 0
        cur_ids[used] = sp["[BOS]"]
        cur_ids[used + 1:used + 1 + len(prompt_ids)] = prompt_ids
        cur_ids[used + 1 + len(prompt_ids):used + n - 1] = resp
        cur_ids[used + n - 1] = sp["[EOS]"]
        cur_pr[used:used + 1 + len(prompt_ids)] = True
        used += n
        counts[-1] += 1
    if not blocks_ids:
        return (np.empty((0, seq_len), np.uint16), np.empty((0, seq_len), bool), np.empty(0, np.int64), dropped)
    return np.stack(blocks_ids), np.stack(blocks_pr), np.asarray(counts, dtype=np.int64), dropped


def _prompts_for(cache: SftCache, tok, order: list[int], rng: random.Random, plain_prob: float):
    instrs = [build_instruction(cache.fields[i], rng, plain_prob)[0] for i in order]
    enc = tok.encode_batch([chat_prompt(x) for x in instrs], add_special_tokens=False)
    return [e.ids for e in enc], instrs


class PackedSet:
    """A FIXED packed split (validation): prompts drawn once from `seed`. Mirrors dlm.data.StoryBlocks' interface
    (`__len__`, `arrays(i) -> (ids int64[L], protect bool[L], n_examples)`), so dlm.train.evaluate works on it."""

    def __init__(self, cache: SftCache, tok, seq_len: int = DEFAULT_SEQ_LEN, seed: int = 0,
                 plain_prob: float = DEFAULT_PLAIN_PROB, limit: int | None = None):
        n = len(cache) if limit is None else min(limit, len(cache))
        order = list(range(n))
        rng = random.Random(f"fixed-{seed}")
        prompts, self.instructions = _prompts_for(cache, tok, order, rng, plain_prob)
        examples = [(p, cache.response(i)) for p, i in zip(prompts, order)]
        self.ids, self.protect, self.counts, self.dropped = pack(examples, seq_len, cache.sp)
        self.seq_len = seq_len
        self.pad_id = cache.sp["[PAD]"]
        self.examples = n

    def __len__(self) -> int:
        return len(self.counts)

    def arrays(self, index: int):
        return self.ids[index].astype(np.int64), self.protect[index].copy(), int(self.counts[index])

    def stats(self) -> dict:
        n = len(self)
        return {"blocks": n, "examples": self.examples, "dropped_too_long": self.dropped,
                "fill": round(float((self.ids != self.pad_id).sum()) / max(n * self.seq_len, 1), 4),
                "protected_share": round(float(self.protect.sum()) / max(n * self.seq_len, 1), 4)}


class SftStream:
    """The training stream: blocks for epoch e come from a seeded permutation of the cached examples, with prompts
    re-drawn per (epoch, chunk) and packed per chunk of CHUNK_EXAMPLES. `arrays(g)` for a global block index g is a pure
    function of (seed, g), so batch(step) is reproducible and resume needs no data-loader state. Also usable with
    dlm.train.make_batch(stream, indices, device)."""

    def __init__(self, cache: SftCache, tok, seq_len: int = DEFAULT_SEQ_LEN, seed: int = 0,
                 plain_prob: float = DEFAULT_PLAIN_PROB, chunk_examples: int = CHUNK_EXAMPLES):
        self.cache, self.tok = cache, tok
        self.seq_len, self.seed, self.plain_prob, self.chunk = seq_len, seed, plain_prob, chunk_examples
        self.pad_id = cache.sp["[PAD]"]
        self.n_chunks = -(-len(cache) // chunk_examples)
        self._counts: dict[tuple[int, int], int] = {}
        self.dropped = 0
        self._chunk_cached = functools.lru_cache(maxsize=6)(self._build_chunk)
        self._perm_cached = functools.lru_cache(maxsize=2)(self._perm)

    def _perm(self, epoch: int) -> list[int]:
        order = list(range(len(self.cache)))
        random.Random(f"perm-{self.seed}-{epoch}").shuffle(order)
        return order

    def _build_chunk(self, epoch: int, c: int):
        order = self._perm_cached(epoch)[c * self.chunk:(c + 1) * self.chunk]
        rng = random.Random(f"prompt-{self.seed}-{epoch}-{c}")
        prompts, _ = _prompts_for(self.cache, self.tok, order, rng, self.plain_prob)
        ex = [(p, self.cache.response(i)) for p, i in zip(prompts, order)]
        ids, pr, counts, dropped = pack(ex, self.seq_len, self.cache.sp)
        if (epoch, c) not in self._counts:
            self.dropped += dropped
        self._counts[(epoch, c)] = len(counts)
        return ids, pr, counts

    def _count(self, epoch: int, c: int) -> int:
        """Blocks in chunk (epoch, c); packs the chunk only the first time (counts are remembered, chunks are LRU-cached)."""
        if (epoch, c) not in self._counts:
            self._chunk_cached(epoch, c)
        return self._counts[(epoch, c)]

    def locate(self, g: int) -> tuple[int, int, int]:
        """Global block index -> (epoch, chunk, offset in chunk)."""
        epoch, c, left = 0, 0, g
        while True:
            n = self._count(epoch, c)
            if left < n:
                return epoch, c, left
            left -= n
            c += 1
            if c == self.n_chunks:
                epoch, c = epoch + 1, 0

    def arrays(self, g: int):
        epoch, c, off = self.locate(g)
        ids, pr, counts = self._chunk_cached(epoch, c)
        return ids[off].astype(np.int64), pr[off].copy(), int(counts[off])

    def __len__(self) -> int:  # one epoch's block count (computes every chunk once)
        return sum(self._count(0, c) for c in range(self.n_chunks))
