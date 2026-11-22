"""Offline logic tests for dlm/sample.py (synthetic tokenizer, stub and tiny models, CPU):
    python run.py test  (or python tests/test_sample.py)"""
import contextlib
import io
import json
import random
import shutil
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import dlm.download as p0  # noqa: E402
import dlm.sample as sm  # noqa: E402
import dlm.tokenizer as p1  # noqa: E402
from dlm.data import HEADER  # noqa: E402
from dlm.model import DLMConfig, DiffusionTransformerLM  # noqa: E402


def make_tokenizer(root: Path):
    rng = random.Random(0)
    words = "the a cat dog sun ran big small happy sad tree house lily tom said went home".split()
    train = [" ".join(rng.choice(words) for _ in range(rng.randint(8, 40))) + "." for _ in range(300)]
    data = root / "data"
    data.mkdir()
    pq.write_table(pa.table({"text": train}), data / "train.parquet")
    pq.write_table(pa.table({"text": train[:20]}), data / "validation.parquet")
    (data / p0.MANIFEST).write_text(json.dumps({
        "dataset": p0.DATASET, "revision": "r", "downloaded_at": "x",
        "splits": {s: {"stories": n, "empty": 0, "avg_chars": 1.0, "avg_words": 1.0, "file_bytes": 1}
                   for s, n in (("train", 300), ("validation", 20))}}))
    p1.build(data, root / "tokenizer", 320, 300, 0)
    return Tokenizer.from_file(str(root / "tokenizer" / "tokenizer.json"))


class Stub(torch.nn.Module):
    """A 'model' whose logits are written by fn(out [B, L, V], ids) so tests control the predictions."""

    def __init__(self, V, seq_len, fn):
        super().__init__()
        self.config = SimpleNamespace(seq_len=seq_len, vocab_size=V)
        self.p = torch.nn.Parameter(torch.zeros(1))
        self.fn = fn
        self.calls = 0

    def forward(self, ids):
        self.calls += 1
        out = torch.zeros(ids.shape[0], ids.shape[1], self.config.vocab_size)
        self.fn(out, ids)
        return out


def main():
    checks = 0

    def ok(cond, msg):
        nonlocal checks
        assert cond, msg
        checks += 1

    # ---- the unmasking plan
    ok(sm.commit_plan(32, 8) == [4] * 8, "linear plan: equal numbers per step")
    for L, S in ((30, 8), (7, 3), (32, 32), (1, 1), (100, 7)):
        plan = sm.commit_plan(L, S)
        ok(sum(plan) == L and all(c >= 1 for c in plan), f"linear plan L={L} S={S} sums to L, every step reveals something")
    for L, S in ((32, 8), (30, 8), (32, 32), (50, 64)):
        plan = sm.commit_plan(L, S, "cosine")
        ok(sum(plan) == L and plan[-1] >= 1, f"cosine plan L={L} S={S} sums to L")
    cp = sm.commit_plan(64, 16, "cosine")
    ok(cp != [4] * 16 and sum(cp) == 64, "the cosine plan differs from the linear plan (and still sums to the block)")
    ok(sm.plan_blocks(100, 32, None) == ([(0, 32), (32, 64), (64, 96), (96, 100)], 32), "blocks of 32, last one short, 1 token/step")
    ok(sm.plan_blocks(100, 0, None) == ([(0, 100)], 100), "block_length 0 = one global block")
    ok(sm.plan_blocks(100, 500, None)[0] == [(0, 100)], "block longer than the window = one block")
    ok(sm.plan_blocks(128, 32, 8)[1] == 2, "total steps are spread over the blocks")
    ok(sm.plan_blocks(128, 32, 100000)[1] == 32, "steps per block clamp to the block length")
    ok(sm.plan_blocks(128, 32, 1)[1] == 1, "at least one step per block")

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        tok = make_tokenizer(root)
        sp = sm.specials(tok)
        V = tok.get_vocab_size()
        header = tok.encode(HEADER, add_special_tokens=False).ids
        P = 1 + len(header)
        normal = [tok.token_to_id(w) for w in ("Ġthe", "Ġcat", "Ġdog", "Ġsun", "Ġran") if tok.token_to_id(w) is not None]
        ok(len(normal) >= 4, "tokenizer has ordinary word tokens for the stubs")
        END, MASK, PAD = sp["<|end|>"], sp["[MASK]"], sp["[PAD]"]
        cfg = DLMConfig.from_preset("tiny", vocab_size=V, seq_len=P + 80)
        torch.manual_seed(0)
        model = DiffusionTransformerLM(cfg)

        # ---- invariants on a random (untrained) model, over several seeds and settings
        results = []
        for seed in range(6):
            r = sm.generate_text(model, tok, "", seed=seed, max_new_tokens=40, block_length=8, temperature=1.0, return_trace=True)
            results.append(r)
            ids, Pf = r["ids"], r["prefix_len"]
            ok(Pf == P and ids[:P] == [sp["[BOS]"]] + header, "the prefix ([BOS] + header) is never touched")
            cov = ids[P:r["covered"]]
            ok(MASK not in cov, "no [MASK] left in the processed region")
            ok(MASK not in ids[:r["covered"]] and PAD not in ids, "[MASK]/[PAD] never sampled")
            ok(all(t == MASK for t in ids[r["covered"]:]), "blocks after the stop are still [MASK] (never run)")
            seen = set()
            for ev in r["trace"]:
                lo, hi = P + 8 * ev["block"], P + min(8 * (ev["block"] + 1), r["gen_len"])
                ok(all(lo <= q < hi for q in ev["positions"]), "a step only commits inside its block")
                ok(not (seen & set(ev["positions"])), "no position is committed twice (committed tokens never change)")
                seen |= set(ev["positions"])
                ok(all(ids[q] == t for q, t in zip(ev["positions"], ev["ids"])), "the recorded token is the final token")
            ok(len(seen) == r["covered"] - P, "every processed position was committed exactly once")
            ok(r["forward_calls"] == len(r["trace"]), "one model call per committing step")
        ok(len({r["text"] for r in results}) > 1, "different seeds give different samples at temperature 1")
        a = sm.generate_text(model, tok, "", seed=5, max_new_tokens=40, block_length=8)
        b = sm.generate_text(model, tok, "", seed=5, max_new_tokens=40, block_length=8)
        ok(a["ids"] == b["ids"], "same seed -> same sample")
        g1 = sm.generate_text(model, tok, "", seed=1, max_new_tokens=40, block_length=8, temperature=0)
        g2 = sm.generate_text(model, tok, "", seed=2, max_new_tokens=40, block_length=8, temperature=0)
        ok(g1["ids"] == g2["ids"], "greedy decoding is deterministic and ignores the seed")
        glob = sm.generate_text(model, tok, "", max_new_tokens=40, block_length=0, return_trace=True)
        ok(glob["n_blocks"] == 1 and MASK not in glob["ids"], "block_length 0 = one global pass")
        cosr = sm.generate_text(model, tok, "", max_new_tokens=40, block_length=8, schedule="cosine", return_trace=True)
        ok(MASK not in cosr["ids"][:cosr["covered"]], "the cosine schedule also completes")
        few = sm.generate_text(model, tok, "", max_new_tokens=40, block_length=8, steps=10, return_trace=True)
        ok(few["forward_calls"] < 40 and MASK not in few["ids"][:few["covered"]], "fewer steps = fewer model calls, still complete")
        full = sm.generate_text(model, tok, "", max_new_tokens=40, block_length=0, return_trace=True)
        ok(full["forward_calls"] == 40, "default = one token per step")

        # ---- window handling and modes
        long_prompt = " ".join(["the cat"] * 60)
        for bad, why in ((lambda: sm.generate_text(model, tok, long_prompt), "a prompt longer than the window"),
                         (lambda: sm.generate_text(model, tok, "hello", mode="story"), "story mode with a prompt"),
                         (lambda: sm.generate_text(model, tok, "", mode="continue"), "continue mode without a prompt"),
                         (lambda: sm.generate_text(model, tok, "x", mode="chat"), "an unknown mode"),
                         (lambda: sm.generate_text(model, tok, "", max_new_tokens=0), "no room to generate")):
            try:
                bad()
                raised = False
            except ValueError:
                raised = True
            ok(raised, f"ValueError for {why}")
        clamp = sm.generate_text(model, tok, "", max_new_tokens=480, block_length=0, steps=4)
        ok(clamp["gen_len"] == cfg.seq_len - P, "max_new_tokens is clamped to the window")
        ok(sm.generate_text(model, tok, "the cat", max_new_tokens=5)["prefix_len"] > P, "continue-mode prompt tokens extend the fixed prefix")

        # ---- stopping at the end of the Story (stub predicting <|end|> everywhere)
        def all_end(out, ids):
            out[..., END] = 30.0
        stub = Stub(V, P + 100, all_end)
        r = sm.generate_text(stub, tok, "", max_new_tokens=96, block_length=32, temperature=0, return_trace=True)
        ok(r["blocks_run"] == 1 and r["n_blocks"] == 3, "generation stops after the block that contains the end token")
        ok(r["forward_calls"] == 32 and stub.calls == 32, "later blocks never call the model")
        ok(r["end_index"] == 0 and r["text"] == "" and not r["truncated"], "empty Story when the first token is the end token")
        ok(all(t == MASK for t in r["ids"][P + 32:]), "blocks after the stop stay [MASK]")
        r2 = sm.generate_text(stub, tok, '“the cat”', max_new_tokens=96, temperature=0)
        ok(r2["text"] == '"the cat"', "continue mode: opening words kept, curly quotes folded, nothing after the end token")

        # ---- the cut: tokens after the first end token are dropped; position-specific stub
        target = {P + j: t for j, t in enumerate([normal[0], normal[1], normal[2], END, normal[3], normal[0], END, normal[1]] * 3)}

        def scripted(out, ids):
            for pos, t in target.items():
                if pos < out.shape[1]:
                    out[:, pos, t] = 30.0
        stub2 = Stub(V, P + 24, scripted)
        r = sm.generate_text(stub2, tok, "", max_new_tokens=24, block_length=8, temperature=0)
        want = tok.decode([normal[0], normal[1], normal[2]], skip_special_tokens=True)
        ok(r["text"] == want and r["end_index"] == 3, "Story is cut at the first <|end|>")
        ok(r["blocks_run"] == 1 and len(r["ids"]) == P + 24, "the stop happens at the end of the block containing it")
        ok(r["ids"][P + 4] == normal[3], "tokens after the end inside the finished block are kept in ids, not in the text")

        # ---- never an end token -> all blocks run, truncated, collapse warning
        def only_word(out, ids):
            out[..., normal[0]] = 30.0
        r = sm.generate_text(Stub(V, P + 64, only_word), tok, "", max_new_tokens=64, block_length=16, temperature=0)
        ok(r["blocks_run"] == 4 and r["truncated"] and r["end_index"] is None, "no end token: every block runs, Story marked unfinished")
        ok(any("collapse" in w for w in r["warnings"]), "collapse warning when one token dominates")
        ok(not any("collapse" in w for w in results[0]["warnings"]), "no collapse warning for varied output")

        # ---- most confident first (confidence rises with position -> commits from the right)
        def sharper_to_the_right(out, ids):
            for pos in range(out.shape[1]):
                out[:, pos, normal[0]] = 1.0 + 0.5 * pos
        r = sm.generate_text(Stub(V, P + 8, sharper_to_the_right), tok, "", max_new_tokens=8, block_length=0, steps=4,
                             temperature=0, return_trace=True)
        order = [sorted(ev["positions"]) for ev in r["trace"]]
        ok(order == [[P + 6, P + 7], [P + 4, P + 5], [P + 2, P + 3], [P + 0, P + 1]], f"most confident positions are committed first ({order})")
        ok([len(ev["positions"]) for ev in r["trace"]] == [2, 2, 2, 2], "linear schedule: equal numbers per step")

        # ---- temperature: sampled frequencies follow softmax(logits / T); T=0 is argmax
        def two_way(out, ids):
            out[..., :] = -30.0
            out[..., normal[0]] = 2.0
            out[..., normal[1]] = 0.0
        two = Stub(V, P + 400, two_way)
        for T, want_p in ((1.0, 1 / (1 + 2.718281828 ** -2)), (0.5, 1 / (1 + 2.718281828 ** -4))):
            hits = n = 0
            for s in range(5):
                r = sm.generate_text(two, tok, "", seed=s, max_new_tokens=400, block_length=0, steps=1, temperature=T)
                g = r["ids"][P:]
                hits += sum(1 for t in g if t == normal[0])
                n += len(g)
            ok(abs(hits / n - want_p) < 0.03, f"T={T}: P(top token) = {hits / n:.3f}, expected {want_p:.3f}")
        r = sm.generate_text(two, tok, "", max_new_tokens=400, block_length=0, steps=1, temperature=0)
        ok(all(t == normal[0] for t in r["ids"][P:]), "T=0 always picks the argmax")

        # ---- seeing the process
        r = sm.generate_text(model, tok, "", max_new_tokens=40, block_length=8, return_trace=True, seed=1)
        txt = sm.render_trace(r, tok, max_rows=10, color=False)
        lines = txt.splitlines()
        ok(len(lines) <= 11 and "█" in lines[1] and "masked   0" in lines[-1], "trace: header + <= 10 rows, first row masked, last row complete")
        ok("\033[" not in txt, "no ANSI codes when color is off")
        ok("\033[32m" in sm.render_trace(r, tok, max_rows=10, color=True), "green highlight when color is on")
        ok("no trace" in sm.render_trace(sm.generate_text(model, tok, "", max_new_tokens=8), tok), "trace explains when it was not recorded")
        gif = root / "u.gif"
        nframes = sm.write_gif(r, tok, gif, max_frames=12)
        from PIL import Image
        im = Image.open(gif)
        ok(gif.read_bytes()[:3] == b"GIF" and 2 <= nframes <= 12 and im.n_frames == nframes, f"GIF written with {nframes} frames")

        # ---- the CLI on a saved run folder
        ckpt = root / "run" / "final"
        model.save_pretrained(ckpt)
        shutil.copytree(root / "tokenizer", ckpt / "tokenizer")
        m2, t2 = sm.load_checkpoint(ckpt)
        ok(torch.equal(m2.tok_emb.weight, model.tok_emb.weight) and t2.get_vocab_size() == V, "load_checkpoint restores model and tokenizer")

        def cli(*args):
            out, err = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                code = sm.main(list(args))
            return code, out.getvalue(), err.getvalue()

        gifs = root / "g.gif"
        code, out, _ = cli("--ckpt", str(ckpt), "--n", "2", "--max-new-tokens", "24", "--block-length", "8", "--device", "cpu",
                           "--seed", "3", "--trace", "--gif", str(gifs))
        ok(code == 0 and "sample 1/2" in out and "sample 2/2" in out and "unmasking trace" in out, "CLI prints samples and the trace")
        ok((root / "g_1.gif").is_file() and (root / "g_2.gif").is_file(), "CLI writes one GIF per sample")
        code2, out2, _ = cli("--ckpt", str(ckpt), "--n", "2", "--max-new-tokens", "24", "--block-length", "8", "--device", "cpu",
                             "--seed", "3", "--trace", "--gif", str(gifs))
        ok(out == out2, "CLI output is reproducible for a fixed seed")
        code, out, _ = cli("--ckpt", str(ckpt), "--prompt", "the cat", "--max-new-tokens", "10", "--device", "cpu", "--temperature", "0")
        ok(code == 0 and "the cat" in out, "CLI continue mode shows the opening words")
        ok(cli("--ckpt", str(root / "nope"))[0] == 1, "missing checkpoint -> exit 1")
        ok(cli("--ckpt", str(ckpt), "--mode", "story", "--prompt", "x", "--device", "cpu")[0] == 2, "story mode with a prompt -> exit 2")
        ok(cli("--ckpt", str(ckpt), "--mode", "continue", "--device", "cpu")[0] == 2, "continue mode without a prompt -> exit 2")

    print(f"All {checks} sampler checks passed.")


if __name__ == "__main__":
    main()
