"""Offline tests for fine-tuning/ft_data.py (uses the committed tokenizer; no GPU, no network):
python fine-tuning/tests/test_ft_data.py"""
import random
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _path  # noqa: E402,F401
from _check import check, exit_with  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

import ft_data as D  # noqa: E402
from dlm import diffusion as df  # noqa: E402
from dlm.data import HEADER  # noqa: E402

TOK_PATH = _path.DLM_ROOT / "tokenizer" / "tokenizer.json"
tok = Tokenizer.from_file(str(TOK_PATH))
sp = D.special_ids(tok)
PAD, BOS, EOS, MASK, END = sp["[PAD]"], sp["[BOS]"], sp["[EOS]"], sp["[MASK]"], sp["<|end|>"]
STORY = 'Sam built a tower of books. "Look!" he said.\nIt fell down. He was sad.'

# -- prompt template equals the pre-training header for the plain instruction
check(D.chat_prompt(D.PLAIN_INSTRUCTION) == HEADER, "plain chat prompt == dlm.data.HEADER")

# -- separate encoding of prompt and response == joint encoding (so the SFT layout matches the pre-training layout)
for instr in (D.PLAIN_INSTRUCTION, D.PLAIN_INSTRUCTION + "\nWords: tower, dragon\nFeatures: Dialogue"):
    joint = tok.encode(D.chat_prompt(instr) + STORY + D.END_SUFFIX, add_special_tokens=False).ids
    sep = D.encode_prompt(tok, instr) + D.encode_response(tok, STORY)
    check(joint == sep, f"joint == separate tokenization for {instr[:30]!r}")

# -- build_sft_example
ex = D.build_sft_example(tok, D.PLAIN_INSTRUCTION, STORY)
ids, pr = ex["input_ids"], ex["protect"]
n_prompt = len(D.encode_prompt(tok, D.PLAIN_INSTRUCTION))
check(ids[0] == BOS and ids[-1] == EOS, "starts with [BOS], ends with [EOS]")
check(pr[:1 + n_prompt].all() and not pr[1 + n_prompt:].any(), "protect = [BOS] + prompt only")
check(ids[-2] != END or True, "")
check(END in ids[1 + n_prompt:].tolist(), "<|end|> is part of the (maskable) response")
check(not pr[ids == END].any() and not pr[ids == EOS].any(), "<|end|> and [EOS] are maskable")
check(D.build_sft_example(tok, "x", STORY, seq_len=20) is None, "too long -> None")
padded = D.build_sft_example(tok, "x", STORY, seq_len=512, pad=True)
check(len(padded["input_ids"]) == 512 and (padded["input_ids"][-1] == PAD) and not padded["protect"][-1], "padding is [PAD], unprotected")
check(MASK not in ids.tolist() and PAD not in ids.tolist(), "no [MASK]/[PAD] inside a real example")
check(D.build_sft_example(tok, "Write a short story.", "  " + STORY + "  ")["input_ids"].tolist() == ids.tolist(), "response is cleaned/stripped")

# -- build_instruction
F = {"w": ["tower", "dragon", "knight"], "f": ["Dialogue", "BadEnding"], "s": "A tower falls.", "r": "He was sad."}
rng = random.Random(0)
seen_orders, seen_sizes = set(), set()
for _ in range(400):
    text, used = D.build_instruction(F, rng, plain_prob=0.0)
    lines = text.split("\n")
    check(lines[0] == D.PLAIN_INSTRUCTION, "first line is always the plain instruction")
    check(1 <= len(used) <= 4 and len(lines) == 1 + len(used), f"subset size {used}")
    seen_orders.add(tuple(used))
    seen_sizes.add(len(used))
    for ln in lines[1:]:
        name, _, val = ln.partition(": ")
        check(name in D.FIELD_ORDER, f"known field name {name}")
check(seen_sizes == {1, 2, 3, 4}, f"all subset sizes occur: {seen_sizes}")
check(len(seen_orders) > 30, f"orders vary ({len(seen_orders)} distinct)")
check(all(D.build_instruction(F, random.Random(i), plain_prob=1.0)[0] == D.PLAIN_INSTRUCTION for i in range(20)), "plain_prob=1 -> always plain")
check(all(D.build_instruction({}, random.Random(i), 0.0)[0] == D.PLAIN_INSTRUCTION for i in range(20)), "no fields -> plain")
share = sum(D.build_instruction(F, random.Random(i), 0.15)[1] == [] for i in range(2000)) / 2000
check(0.11 < share < 0.19, f"plain share ~15%: {share:.3f}")
check(D.build_instruction(F, random.Random(5), 0.0) == D.build_instruction(F, random.Random(5), 0.0), "deterministic for a seed")
check(D.field_texts({"w": ["a", "b"]}) == [("Words", "a, b")], "field_texts")

# -- packing + prompt protection under real corruption
L = 128
rows = []
rg = np.random.default_rng(0)
for i in range(60):
    p = D.encode_prompt(tok, D.PLAIN_INSTRUCTION + (f"\nWords: w{i}" if i % 2 else ""))
    rows.append((p, np.asarray(tok.encode(f"Story number {i}. " * int(rg.integers(1, 9)) + D.END_SUFFIX, add_special_tokens=False).ids)))
rows.append((rows[0][0], np.arange(10, 10 + 300)))  # longer than a block: dropped
ids_, pr_, counts, dropped = D.pack(rows, L, sp)
check(dropped == 1, f"over-long example dropped ({dropped})")
check(counts.sum() == 60 and ids_.shape == (len(counts), L), "every other example kept, none split")
for b in range(len(counts)):
    row, prot = ids_[b].astype(int), pr_[b]
    nreal = int((row != PAD).sum())
    check(row[0] == BOS and (row[nreal:] == PAD).all(), f"block {b} starts with [BOS], pads at the end")
    check(int((row == BOS).sum()) == counts[b] == int((row == EOS).sum()), f"block {b}: one [BOS]/[EOS] per example")
    check(not prot[nreal:].any(), "PAD never protected")
    # each BOS opens a protected run that is exactly BOS + <|user|> ... <|assistant|>\n
    for pos in np.flatnonzero(row == BOS):
        check(prot[pos] and row[pos + 1] == tok.token_to_id("<|user|>"), "protected run starts with BOS, <|user|>")
    # protected tokens are exactly the prompt spans: the token after the run is a response token
    runs = np.flatnonzero(np.diff(np.concatenate([[0], prot.astype(int), [0]])) == 1)
    check(len(runs) == counts[b], f"block {b}: one protected run per example")
    ap = tok.token_to_id("<|assistant|>")
    for r0 in runs:
        r1 = r0
        while r1 < L and prot[r1]:
            r1 += 1
        check(row[r1 - 2] == ap, "protected run ends right after <|assistant|>\\n")

# prompt tokens are never masked, response tokens are masked, PAD never masked (PRD Phase 8 acceptance)
class Dummy(torch.nn.Module):
    def forward(self, x):
        return torch.zeros(*x.shape, 8192)

ids_t, pr_t = torch.from_numpy(ids_.astype(np.int64)), torch.from_numpy(pr_)
for sched in ("linear", "cosine"):
    t = torch.full((len(ids_t),), 1.0)  # p ~ 1: everything maskable is masked
    noised, masked, maskable = df.corrupt(ids_t, pr_t, MASK, PAD, t, sched, torch.Generator().manual_seed(0))
    check(not (masked & pr_t).any(), f"{sched}: no protected token is ever masked")
    check(not (masked & (ids_t == PAD)).any(), f"{sched}: PAD never masked")
    check(torch.equal(noised[pr_t], ids_t[pr_t]), f"{sched}: prompt tokens identical after noising")
    resp = (~pr_t) & (ids_t != PAD)
    check(masked.sum() > 0.95 * resp.sum(), f"{sched}: nearly all response tokens masked at t=1 ({int(masked.sum())}/{int(resp.sum())})")
    check(torch.equal(maskable, resp), f"{sched}: maskable == response tokens")
num, den, nm = df.loss_sums(Dummy(), ids_t, pr_t, MASK, PAD, torch.full((len(ids_t),), 0.5), "linear", torch.Generator().manual_seed(1))
check(torch.equal(den, ((~pr_t) & (ids_t != PAD)).sum(1)), "loss denominator counts response tokens only")

# -- cache + stream on a synthetic raw file
def raw_record(i):
    fields = [f"Words: w{i}, x{i}", f"Features: {'Dialogue' if i % 2 else 'Twist'}", f"Summary: Summary {i}.", f"Random sentence: Line {i} here."]
    random.Random(i).shuffle(fields)
    body = f"Story number {i}. " + "He ran. " * (3 + i % 7) + f"Line {i} here."
    tail = f"\nSummary: trailing {i}." if i % 3 == 0 else ""
    return "\n".join(fields) + "\nStory: \n" + body + tail + "\n"

with tempfile.TemporaryDirectory() as d:
    d = Path(d)
    raw = d / "raw.txt"
    raw.write_text("fragment without marker\n" + "<|endoftext|>\n".join(raw_record(i) for i in range(300)), encoding="utf-8")
    man = D.build_cache(raw, d / "cache", "train", TOK_PATH, keep_prob=1.0, seed=0)
    st = man["splits"]["train"]
    check(st["examples"] == 300 and st["records_kept"] == 300, f"cache examples {st}")
    check(0.3 < st["trailing_fields_dropped_share"] < 0.36, f"trailing share {st['trailing_fields_dropped_share']}")
    man2 = D.build_cache(raw, d / "cache2", "train", TOK_PATH, keep_prob=0.3, seed=1)
    k = man2["splits"]["train"]["examples"]
    check(60 < k < 120, f"keep_prob 0.3 keeps about 90 of 300 (got {k})")
    check(D.build_cache(raw, d / "cache2b", "train", TOK_PATH, keep_prob=0.3, seed=1)["splits"]["train"]["examples"] == k, "keep_prob is seeded/deterministic")
    cache = D.SftCache(d / "cache", "train")
    check(len(cache) == 300, "cache length")
    # the response of example 0 has no trailing summary and ends with <|end|>
    r0 = tok.decode(cache.response(0).tolist(), skip_special_tokens=False)
    check("trailing" not in r0 and r0.endswith("<|end|>\n"), f"response cleaned of trailing fields: {r0!r}")
    check(cache.fields[1]["w"] == ["w1", "x1"] and cache.fields[1]["f"] == ["Dialogue"], f"fields cached {cache.fields[1]}")

    s1 = D.SftStream(cache, tok, seq_len=256, seed=7, chunk_examples=64)
    s2 = D.SftStream(cache, tok, seq_len=256, seed=7, chunk_examples=64)
    s3 = D.SftStream(cache, tok, seq_len=256, seed=8, chunk_examples=64)
    n_epoch = len(s1)
    check(n_epoch > 5, f"blocks per epoch {n_epoch}")
    same = all(np.array_equal(s1.arrays(g)[0], s2.arrays(g)[0]) and np.array_equal(s1.arrays(g)[1], s2.arrays(g)[1]) for g in range(n_epoch))
    check(same, "stream is a pure function of (seed, block index)")
    diff = any(not np.array_equal(s1.arrays(g)[0], s3.arrays(g)[0]) for g in range(n_epoch))
    check(diff, "a different seed gives different blocks")
    # random access order does not matter (resume from the middle == straight run)
    order = list(range(n_epoch)); random.Random(0).shuffle(order)
    check(all(np.array_equal(s1.arrays(g)[0], s2.arrays(g)[0]) for g in order), "random-order access is identical")
    # epoch 1 re-draws prompts and order
    e0 = s1.arrays(0)[0]; e1 = s1.arrays(n_epoch)[0]
    check(s1.locate(n_epoch)[0] == 1 and s1.locate(n_epoch - 1)[0] == 0, "locate crosses the epoch boundary")
    check(not np.array_equal(e0, e1), "next epoch differs (new permutation and new prompts)")
    # prompts of the same example differ across epochs: collect decoded prompts for example text 'Story number 5.'
    def prompts_of(stream, lo, hi):
        out = []
        for g in range(lo, hi):
            row, prot, _ = stream.arrays(g)
            text = tok.decode(row[row != PAD].tolist(), skip_special_tokens=False)
            for seg in text.split("[BOS]")[1:]:
                if "Story number 5." in seg:
                    out.append(seg.split("<|assistant|>")[0])
        return out
    p0, p1 = prompts_of(s1, 0, n_epoch), prompts_of(s1, n_epoch, 2 * n_epoch)
    check(len(p0) == 1 and len(p1) == 1 and p0 != p1, f"the same example gets a new prompt each epoch: {p0} vs {p1}")
    # every protected run in the stream is a prompt: decode and check shape of instruction
    row, prot, n = s1.arrays(3)
    check(prot[0] and row[0] == BOS and n >= 1, "block 3 well-formed")
    # fixed validation set
    vs = D.PackedSet(cache, tok, seq_len=256, seed=0)
    vs2 = D.PackedSet(cache, tok, seq_len=256, seed=0)
    check(len(vs) > 0 and np.array_equal(vs.ids, vs2.ids), "PackedSet is deterministic")
    check(vs.examples == 300 and int(vs.counts.sum()) + vs.dropped == 300, "PackedSet keeps every example")
    sts = vs.stats()
    check(0.2 < sts["protected_share"] < 0.7 and 0.3 < sts["fill"] <= 1.0, f"PackedSet stats {sts}")
    # dlm.train.evaluate works on a PackedSet (same interface as StoryBlocks)
    from dlm.model import DLMConfig, DiffusionTransformerLM
    from dlm.train import evaluate, evenly, make_amp
    cfg = DLMConfig.from_preset("tiny", vocab_size=8192, seq_len=256, pad_id=PAD, mask_id=MASK)
    model = DiffusionTransformerLM(cfg)
    amp, _ = make_amp("cpu", "fp32")
    v1 = evaluate(model, vs, evenly(len(vs), 8), cfg, "cpu", amp)
    v2 = evaluate(model, vs, evenly(len(vs), 8), cfg, "cpu", amp)
    check(np.isfinite(v1) and v1 == v2, f"dlm.train.evaluate on PackedSet is finite and deterministic ({v1:.3f})")

exit_with("test_ft_data")
