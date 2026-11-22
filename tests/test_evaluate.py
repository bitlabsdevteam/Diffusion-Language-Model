"""Offline logic tests for dlm/evaluate.py (synthetic data, stub/tiny models, CPU): python run.py test  (or python tests/test_evaluate.py)"""
import contextlib
import io
import json
import math
import random
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import dlm.data as d  # noqa: E402
import dlm.download as p0  # noqa: E402
import dlm.evaluate as ev  # noqa: E402
import dlm.tokenizer as p1  # noqa: E402
from dlm.data import StoryBlocks  # noqa: E402
from dlm.model import DLMConfig, DiffusionTransformerLM  # noqa: E402
from dlm.train import make_amp  # noqa: E402

SEQ = 128


def stories(n, seed):
    rng = random.Random(seed)
    words = "the a cat dog sun ran big small happy sad tree house lily tom said went home".split()
    return [" ".join(rng.choice(words) for _ in range(rng.randint(8, 40))) + "." for _ in range(n)]


class Uniform(torch.nn.Module):
    def __init__(self, V):
        super().__init__()
        self.V = V

    def forward(self, ids):
        return torch.zeros(*ids.shape, self.V)


def main():
    checks = 0

    def ok(cond, msg):
        nonlocal checks
        assert cond, msg
        checks += 1

    # ---- pure metrics on hand-made inputs (known answers)
    ok(ev.distinct_n([[1, 2, 3], [1, 2, 3]], 1) == 0.5, "distinct-1 of two identical sequences")
    ok(ev.distinct_n([[1, 2, 3], [1, 2, 3]], 2) == 0.5, "distinct-2: 2 unique of 4")
    ok(ev.distinct_n([[1, 1, 1, 1]], 1) == 0.25 and ev.distinct_n([[1, 2, 3, 4]], 3) == 1.0, "distinct-n extremes")
    ok(ev.distinct_n([[1, 2]], 3) == 0.0 and ev.distinct_n([], 1) == 0.0, "no n-grams -> 0")
    ok(abs(ev.repetition_rate([[1, 2, 3, 4, 1, 2, 3, 4, 5]], 4) - 8 / 9) < 1e-12, "repetition: 8 of 9 tokens in a repeated 4-gram")
    ok(ev.repetition_rate([list(range(10))], 4) == 0.0, "no repeated 4-gram -> 0")
    ok(ev.repetition_rate([[1, 2, 3]], 4) == 0.0, "too short for a 4-gram -> 0")
    ok(ev.repetition_rate([[7] * 10], 4) == 1.0, "a loop of one token is 100% repetition")
    ok(abs(ev.repetition_rate([[1, 2, 3, 4, 1, 2, 3, 4], list(range(10, 18))], 4) - 0.5) < 1e-12, "repetition is pooled over tokens")
    ok(ev.length_stats([5, 1, 3, 2, 4]) == {"median": 3, "p90": 5, "mean": 3.0}, "length median / p90 / mean")
    ok(ev.length_stats([]) == {"median": 0, "p90": 0, "mean": 0.0}, "empty length stats")
    samples = [{"tokens": [1, 2, 3], "finished": True, "warnings": []}, {"tokens": [4] * 5, "finished": False, "warnings": ["collapse"]},
               {"tokens": [1, 2, 3, 4], "finished": True, "warnings": []}, {"tokens": [9], "finished": False, "warnings": []}]
    st = ev.story_stats(samples)
    ok(st["n"] == 4 and st["finished_rate"] == 0.5 and st["collapse_rate"] == 0.25, "finished and collapse rates")
    ok(abs(ev.bits_per_char(math.log(2) * 100, 100) - 1.0) < 1e-12 and math.isnan(ev.bits_per_char(1.0, 0)), "bits per character")
    scorer = lambda texts: (2.0 * len(texts), 4 * len(texts))  # noqa: E731  (NLL 0.5 per token)
    ok(abs(ev.gen_ppl(["a b", "c d", "  "], scorer) - math.exp(0.5)) < 1e-12, "GPT-2-style pooled perplexity (empty texts skipped)")
    ok(ev.gen_ppl(["a"], None) is None and ev.gen_ppl([" "], scorer) is None, "no scorer / no text -> None")
    ok([ev.steps_for(k, 480, 32) for k in (1, 2, 4, 8, 16, 32)] == [480, 240, 120, 60, 30, 15], "steps for blocks of 32: 32/k per block x 15 blocks")
    ok([ev.steps_for(k, 480, 0) for k in (1, 2, 4, 8, 16, 32)] == [480, 240, 120, 60, 30, 15], "steps for one global pass: 480/k")
    ok(ev.steps_for(1000, 480, 32) == 15 and ev.steps_for(1, 100, 32) == 128, "at least one step per block; a short last block is rounded up")
    sw = ev.sweep_settings()
    ok(len(sw) == 24 and len({tuple(sorted(s.items())) for s in sw}) == 24, "the sweep has 24 distinct settings")
    ok(ev.HEADLINE in sw, "the headline setting is part of the sweep")
    ok("global" in ev.setting_label({"tokens_per_step": 4, "block_length": 0, "schedule": "cosine"}), "setting labels")

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        data_dir, tok_dir, cache = root / "data", root / "tokenizer", root / "data" / "tokens"
        train, val = stories(400, 0), stories(400, 1)
        data_dir.mkdir()
        pq.write_table(pa.table({"text": train}), data_dir / "train.parquet")
        pq.write_table(pa.table({"text": val}), data_dir / "validation.parquet")
        (data_dir / p0.MANIFEST).write_text(json.dumps({
            "dataset": p0.DATASET, "revision": "r", "downloaded_at": "x",
            "splits": {s: {"stories": 400, "empty": 0, "avg_chars": 1.0, "avg_words": 1.0, "file_bytes": 1} for s in ("train", "validation")}}))
        p1.build(data_dir, tok_dir, 320, 300, 0)
        d.build_token_cache(data_dir, tok_dir, cache)
        tok = Tokenizer.from_file(str(tok_dir / "tokenizer.json"))
        V = tok.get_vocab_size()
        vb = StoryBlocks(cache, "validation", SEQ)
        cfg = SimpleNamespace(mask_id=4, pad_id=0)
        amp = make_amp("cpu", "fp32")[0]
        all_idx = list(range(len(vb)))

        # ---- likelihood
        r1 = ev.elbo_bound(Uniform(V), vb, cfg, "cpu", amp, all_idx, draws=4)
        want = (1 - df_eps()) * math.log(V)
        ok(abs(r1["nats_per_token"] - want) < 0.2, f"uniform model: bound ~ (1-EPS) ln V (got {r1['nats_per_token']:.3f}, want {want:.3f})")
        independent = 0
        for i in all_idx:
            ids, prot, _ = vb.arrays(i)
            independent += int(((ids != 0) & ~prot).sum())
        ok(abs(r1["maskable_tokens"] - independent) < 1e-6, "maskable token count matches an independent count")
        ok(ev.elbo_bound(Uniform(V), vb, cfg, "cpu", amp, all_idx, draws=4)["nats_per_token"] == r1["nats_per_token"], "the bound is reproducible")
        ok(len(set(round(x, 6) for x in r1["per_draw"])) == 4, "the draws are different")
        sd = lambda draws: float(np.std([ev.elbo_bound(Uniform(V), vb, cfg, "cpu", amp, all_idx, draws=draws, seed=s)["nats_per_token"]  # noqa: E731
                                         for s in range(16)]))
        ok(sd(4) < 0.8 * sd(1), "more draws per block reduce the noise")
        small = ev.elbo_bound(Uniform(V), vb, cfg, "cpu", amp, all_idx[:5], draws=2)
        ok(small["blocks"] == 5 and small["maskable_tokens"] < r1["maskable_tokens"], "block_indices limits the evaluation")
        before = torch.get_rng_state()
        ev.elbo_bound(Uniform(V), vb, cfg, "cpu", amp, all_idx[:5], draws=1)
        ok(torch.equal(before, torch.get_rng_state()), "the bound does not touch the global RNG")
        tiny = DiffusionTransformerLM(DLMConfig.from_preset("tiny", vocab_size=V, seq_len=SEQ)).eval()
        a1 = ev.elbo_bound(tiny, vb, tiny.config, "cpu", amp, all_idx[:20], draws=2)["nats_per_token"]
        ok(abs(a1 - math.log(V)) < 0.5, "an untrained model scores near ln V")

        # unigram baseline == an independent pure-python computation
        m = json.loads((cache / d.CACHE_MANIFEST).read_text())
        pps = m["protected_per_story"]

        def maskable(split):
            tk, of = np.load(cache / f"{split}_tokens.npy"), np.load(cache / f"{split}_offsets.npy")
            return [int(t) for i in range(len(of) - 1) for t in tk[of[i] + pps:of[i + 1]]]
        c = Counter(int(t) for t in np.load(cache / "train_tokens.npy"))
        pre = Counter(int(t) for i, o in enumerate(np.load(cache / "train_offsets.npy")[:-1])
                      for t in np.load(cache / "train_tokens.npy")[o:o + pps])
        c.subtract(pre)
        tot = sum(c.values()) + V
        vals = maskable("validation")
        indep = -sum(math.log((c.get(t, 0) + 1) / tot) for t in vals) / len(vals)
        ok(abs(ev.unigram_baseline(cache) - indep) < 1e-9, "unigram baseline equals an independent computation")
        ok(0 < ev.unigram_baseline(cache) < math.log(V), "unigram baseline is below the uniform ln V")

        # ---- real Stories and prompts
        info = ev.real_story_info(vb, tok, [int(k) for k in vb.eligible[:5]])
        for k, toks, text, nch in zip(vb.eligible[:5], info["tokens"], info["texts"], info["chars"]):
            ok(text == tok.decode(toks, skip_special_tokens=True) and nch == len(text), "real Story text and character count")
            raw = np.asarray(vb.tokens[int(vb.offsets[k]) + pps:int(vb.offsets[k + 1])])
            ok(len(toks) == int(np.flatnonzero(raw == tok.token_to_id("<|end|>"))[0]), "real Story length = tokens before the first end token")
        ok("<|end|>" not in " ".join(info["texts"]), "end markers are not part of the Story text")
        pr = ev.pick_prompts(vb, tok, 20)
        ok(pr == ev.pick_prompts(vb, tok, 20) and len(set(pr)) > 10, "prompts are deterministic and mostly distinct")
        ok(all(1 <= len(p.split()) <= ev.PROMPT_WORDS for p in pr), "prompts are at most 8 words")

        # ---- run_setting
        def only_word(out, ids):
            out[..., tok.token_to_id("Ġthe")] = 30.0

        class WordStub(torch.nn.Module):
            def __init__(self, V, seq_len):
                super().__init__()
                self.config = SimpleNamespace(seq_len=seq_len, vocab_size=V)
                self.p = torch.nn.Parameter(torch.zeros(1))

            def forward(self, ids):
                out = torch.zeros(ids.shape[0], ids.shape[1], self.config.vocab_size)
                only_word(out, ids)
                return out

        ws = WordStub(V, 120)
        hlen = len(tok.encode(d.HEADER, add_special_tokens=False).ids)
        G = 120 - (1 + hlen)
        for k, block, want_calls in ((1, 0, G), (8, 0, math.ceil(G / 8)), (1, 32, G), (4, 32, 8 * (G // 32) + min(8, G % 32))):
            res = ev.run_setting(ws, tok, {"tokens_per_step": k, "block_length": block, "schedule": "linear"}, [0, 1],
                                 max_new_tokens=480, temperature=0)
            ok(all(s["calls"] == want_calls for s in res["samples"]), f"{k} tokens/step, block {block}: {want_calls} model calls (got {[s['calls'] for s in res['samples']]})")
        res = ev.run_setting(ws, tok, {"tokens_per_step": 1, "block_length": 0, "schedule": "linear"}, [0, 1], temperature=0)
        ok(res["throughput"]["model_calls"] == 2 * G and res["throughput"]["ms_per_call"] > 0, "throughput bookkeeping")
        ok(all(not s["finished"] and len(s["tokens"]) == G for s in res["samples"]), "unfinished Stories keep all generated tokens")
        a = ev.run_setting(tiny, tok, ev.HEADLINE, [3, 4], max_new_tokens=30)
        b = ev.run_setting(tiny, tok, ev.HEADLINE, [3, 4], max_new_tokens=30)
        ok([s["tokens"] for s in a["samples"]] == [s["tokens"] for s in b["samples"]], "fixed seeds reproduce the samples")
        c2 = ev.run_setting(tiny, tok, ev.HEADLINE, [0, 1], prompts=["the cat", "a dog"], max_new_tokens=30)
        ok([s["prompt"] for s in c2["samples"]] == ["the cat", "a dog"] and c2["samples"][0]["text"].startswith("the cat"), "continue mode keeps the prompts")

        # ---- the full report on a saved run folder (tiny model, tiny sizes)
        run = root / "run"
        ckpt = run / "final"
        tiny.save_pretrained(ckpt)
        shutil.copytree(tok_dir, ckpt / "tokenizer")
        base = ["--ckpt", str(ckpt), "--cache", str(cache), "--n-stories", "6", "--n-prompts", "3", "--draws", "2",
                "--max-val-blocks", "20", "--max-new-tokens", "30", "--no-gpt2", "--device", "cpu"]

        def cli(*args):
            out_, err_ = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(out_), contextlib.redirect_stderr(err_):
                code = ev.main(list(args))
            return code, out_.getvalue(), err_.getvalue()

        def strip(o):
            if isinstance(o, dict):
                return {k: strip(v) for k, v in o.items() if k not in ("throughput", "created_at")}
            if isinstance(o, list):
                return [strip(v) for v in o]
            return o

        code, out, _ = cli(*base)
        ok(code == 0, "eval runs end to end")
        rep1 = json.loads((run / "eval" / "report.json").read_text())
        ok(all((run / "eval" / f).is_file() for f in ("report.json", "report.md", "samples.md")), "report.json, report.md, samples.md written")
        ok(rep1["likelihood"]["nats_per_token"] > 0 and rep1["likelihood"]["bits_per_char"] > 0, "likelihood in nats/token and bits/char")
        ok(rep1["likelihood"]["unigram_baseline_nats_per_token"] > 0 and "own_schedule" not in rep1["likelihood"], "baseline reported; linear model has no second schedule")
        ok(rep1["headline"]["unconditional"]["stats"]["n"] == 6 and rep1["headline"]["continue"]["stats"]["n"] == 3
           and rep1["headline"]["reference_real_stories"]["stats"]["n"] == 6, "sample counts")
        ok(rep1["headline"]["reference_real_stories"]["stats"]["finished_rate"] == 1.0, "real Stories all 'finish'")
        ok(rep1["sweep"] == [] and rep1["settings"]["gpt2_perplexity"] is False, "no sweep / GPT-2 unless asked")
        md = (run / "eval" / "report.md").read_text()
        ok(all(s in md for s in ("## Likelihood", "bits per character", "unigram baseline", "real validation Stories", "## Throughput")), "report.md sections")
        sm = (run / "eval" / "samples.md").read_text()
        ok("Real validation Stories" in sm and "seed 0." in sm and "**prompt:**" in sm, "samples.md sections")
        first_md = (run / "eval" / "report.md").read_text()
        code, _, _ = cli(*base)
        ok(code == 0 and strip(json.loads((run / "eval" / "report.json").read_text())) == strip(rep1), "same checkpoint + seeds -> identical report (timings excepted)")
        ok((run / "eval" / "samples.md").read_text() == sm, "samples.md is reproducible")
        ok(first_md.splitlines()[3:20] == (run / "eval" / "report.md").read_text().splitlines()[3:20], "likelihood table text is reproducible")

        # a cosine-trained run is also reported under its own schedule; sweep runs 24 settings
        (ckpt / "run_info.json").write_text(json.dumps({"recipe": {"schedule": "cosine"}}))
        seen = []
        real_run_setting = ev.run_setting
        ev.run_setting = lambda *a_, **k_: (seen.append((a_[2], len(a_[3]), k_.get("prompts") is not None)), real_run_setting(*a_, **k_))[1]
        try:
            code, _, _ = cli(*base, "--sweep", "--sweep-n", "2", "--out", str(run / "eval2"))
        finally:
            ev.run_setting = real_run_setting
        ok(len(seen) == 2 + 23, f"the headline sweep row reuses the headline Stories: {len(seen)} generation runs, not 26")
        ok(sum(1 for s_, n_, c_ in seen if s_ == ev.HEADLINE) == 2 and (ev.HEADLINE, 2, False) not in seen,
           "the headline setting is generated only for the unconditional and continue sets")
        rep2 = json.loads((run / "eval2" / "report.json").read_text())
        ok(code == 0 and rep2["likelihood"]["own_schedule"]["schedule"] == "cosine", "a cosine-trained run is reported under its own schedule too")
        ok(len(rep2["sweep"]) == 24 and all(r["stats"]["n"] == 2 for r in rep2["sweep"]), "--sweep runs the 24 settings")
        ok("## Decoding sweep" in (run / "eval2" / "report.md").read_text() and "tok/step" in (run / "eval2" / "samples.md").read_text(), "sweep tables and samples")
        head_row = next(r for r in rep2["sweep"] if r["setting"] == ev.HEADLINE)
        ok(head_row["stats"]["n"] == 2 and head_row["throughput"]["model_calls"] > 0, "the headline sweep row has the same n as the others")
        thr = ev.throughput_of([{"calls": 10, "seconds": 0.5, "tokens": [1] * 20}, {"calls": 30, "seconds": 1.5, "tokens": [1] * 60}])
        ok(thr["model_calls"] == 40 and thr["ms_per_call"] == 50.0 and thr["seconds_per_story"] == 1.0 and thr["story_tokens_per_second"] == 40.0,
           "throughput_of on hand-made numbers")
        calls = {ev.setting_label(r["setting"]): r["throughput"]["model_calls"] for r in rep2["sweep"]}
        ok(calls["1 tok/step, global, linear"] > calls["8 tok/step, global, linear"] > calls["32 tok/step, global, linear"],
           "fewer model calls with more tokens per step")

        # errors
        ok(cli("--ckpt", str(root / "nope"), "--cache", str(cache))[0] == 1, "missing checkpoint -> exit 1")
        ok(cli("--ckpt", str(ckpt), "--cache", str(root / "nocache"))[0] == 1, "missing token cache -> exit 1")

    print(f"All {checks} evaluation checks passed.")


def df_eps():
    from dlm import diffusion
    return diffusion.EPS


if __name__ == "__main__":
    main()
