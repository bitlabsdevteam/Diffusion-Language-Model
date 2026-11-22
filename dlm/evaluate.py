"""Phase 7: evaluate a trained run (likelihood, sample quality, decoding sweep, throughput).

Decisions (grilled 2026-11-22, see PRD.md Phase 7):
  * `python run.py eval --ckpt <run>/final` writes <run>/eval/{report.json, report.md, samples.md}. samples.md holds
    fixed-seed Stories to READ: numbers alone cannot say whether a Story is good.
  * Likelihood: validation ELBO bound under the LINEAR schedule over the full validation set with 4 stratified noise
    draws per block, in nats per token and bits per character; a model trained with another schedule is also reported
    under its own. Printed next to the unigram baseline (a model that knows only token frequencies).
    bits/char = total weighted nats / (ln 2 * characters of the evaluated Stories' text up to the end marker).
  * Sample metrics (no external model): finished-Story rate, Story length (median / 90th percentile) against real
    validation Stories, pooled distinct 1/2/3-gram ratios, repetition rate (share of tokens inside a 4-gram that
    occurs at least twice in the same Story), collapse-warning rate. Real Stories get the same statistics (reference).
  * Reference-model perplexity: pretrained GPT-2 small scores the samples and real validation Stories; only the
    RELATIVE number means anything (GPT-2 knows web text, not TinyStories). Skipped with a message when unavailable.
    No LLM judge (it would send the Stories to an external service).
  * Samples: 200 unconditional Stories (seeds 0-199) + 50 continue prompts (the first 8 words of 50 seeded validation
    Stories, seeds 1000+i); all seeds are fixed so a re-run on CPU reproduces the report (timings excepted).
  * --sweep: tokens per step {1, 2, 4, 8, 16, 32} x decoding {blocks of 32, one global pass} x schedule {linear,
    cosine} = 24 settings on the one trained model, 50 Stories each (the headline setting keeps its full 200 + 50, and
    its sweep row reuses the first 50 of those Stories instead of generating them again).
    k tokens per step = every decoding block takes ceil(block / k) steps (one global pass: ceil(gen_len / k)); with
    15 decoding blocks the minimum is therefore 15 model calls.
  * Throughput (one Story at a time): milliseconds per model call, seconds per Story, Story tokens per second.
No AR baseline, no LLM judge, no target numbers yet: you set targets after the first real report.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import random
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from dlm import diffusion as df
from dlm.data import CACHE_MANIFEST, HEADER, StoryBlocks
from dlm.download import default_out_dir
from dlm.sample import DEFAULT_BLOCK, DEFAULT_MAX_NEW, DEFAULT_TEMPERATURE, generate_text, load_checkpoint, specials
from dlm.train import evenly, make_amp, make_batch, pick_device

TOKENS_PER_STEP = (1, 2, 4, 8, 16, 32)
DECODINGS = (DEFAULT_BLOCK, 0)  # decoding-block length; 0 = one global pass
SCHEDULES = ("linear", "cosine")
HEADLINE = {"tokens_per_step": 1, "block_length": DEFAULT_BLOCK, "schedule": "linear"}
PROMPT_WORDS = 8


# ---------------------------------------------------------------- metrics on token sequences (pure, unit-tested)
def distinct_n(seqs: list[list[int]], n: int) -> float:
    """Pooled distinct-n: unique n-grams / all n-grams over all sequences (1.0 = no repetition at all)."""
    total, seen = 0, set()
    for s in seqs:
        grams = list(zip(*[s[i:] for i in range(n)]))
        total += len(grams)
        seen.update(grams)
    return len(seen) / total if total else 0.0


def repetition_rate(seqs: list[list[int]], n: int = 4) -> float:
    """Share of tokens that lie inside an n-gram occurring at least twice in the same sequence."""
    covered = total = 0
    for s in seqs:
        total += len(s)
        if len(s) < n:
            continue
        grams = [tuple(s[i:i + n]) for i in range(len(s) - n + 1)]
        counts = Counter(grams)
        mark = [False] * len(s)
        for i, g in enumerate(grams):
            if counts[g] >= 2:
                for j in range(i, i + n):
                    mark[j] = True
        covered += sum(mark)
    return covered / total if total else 0.0


def length_stats(lengths: list[int]) -> dict:
    if not lengths:
        return {"median": 0, "p90": 0, "mean": 0.0}
    xs = sorted(lengths)
    return {"median": xs[len(xs) // 2], "p90": xs[min(len(xs) - 1, int(0.9 * len(xs)))], "mean": round(statistics.fmean(xs), 1)}


def story_stats(samples: list[dict]) -> dict:
    """Automatic quality statistics of a list of samples ({"tokens", "finished", "warnings"})."""
    seqs = [s["tokens"] for s in samples]
    n = len(samples)
    return {
        "n": n,
        "finished_rate": round(sum(1 for s in samples if s["finished"]) / n, 4) if n else 0.0,
        "length": length_stats([len(t) for t in seqs]),
        "distinct_1": round(distinct_n(seqs, 1), 4),
        "distinct_2": round(distinct_n(seqs, 2), 4),
        "distinct_3": round(distinct_n(seqs, 3), 4),
        "repetition_rate": round(repetition_rate(seqs, 4), 4),
        "collapse_rate": round(sum(1 for s in samples if s.get("warnings")) / n, 4) if n else 0.0,
    }


def bits_per_char(total_nats: float, chars: int) -> float:
    return total_nats / (math.log(2) * chars) if chars else float("nan")


def gen_ppl(texts: list[str], scorer) -> float | None:
    """Pooled perplexity under a reference model: exp(total NLL / total tokens). `scorer(texts) -> (nll, tokens)`."""
    texts = [t for t in texts if t.strip()]
    if scorer is None or not texts:
        return None
    nll, n = scorer(texts)
    return math.exp(nll / n) if n else None


def gpt2_scorer(device: str):
    """Scorer using pretrained GPT-2 small, or None (with a message) if it cannot be loaded."""
    try:
        from transformers import GPT2LMHeadModel, GPT2TokenizerFast

        tk = GPT2TokenizerFast.from_pretrained("gpt2")
        lm = GPT2LMHeadModel.from_pretrained("gpt2").to(device).eval()
    except Exception as e:  # offline, missing package, ...
        print(f"GPT-2 perplexity skipped ({type(e).__name__}: {str(e)[:80]})", file=sys.stderr)
        return None

    @torch.no_grad()
    def score(texts: list[str]):
        total, count = 0.0, 0
        for t in texts:
            ids = tk(tk.bos_token + t, return_tensors="pt", truncation=True, max_length=1024).input_ids.to(device)
            if ids.shape[1] < 2:
                continue
            total += float(lm(ids, labels=ids).loss) * (ids.shape[1] - 1)
            count += ids.shape[1] - 1
        return total, count

    return score


# ---------------------------------------------------------------- likelihood
def elbo_bound(model, blocks: StoryBlocks, cfg, device, amp, block_indices: list[int], draws: int = 4,
               schedule: str = "linear", seed: int = 777, batch_size: int = 32) -> dict:
    """Per-token ELBO bound (sum of weighted CE over masked tokens / maskable tokens) over `block_indices`, averaged
    over `draws` independent stratified noise draws per block. Uses its own generator (global RNG untouched)."""
    was_training = model.training
    model.eval()
    num_tot = den_tot = 0.0
    per_draw = []
    with torch.no_grad():
        for d in range(draws):
            gen = torch.Generator().manual_seed(seed + 1000 * d)
            num_d = den_d = 0.0
            for a in range(0, len(block_indices), batch_size):
                ids, protect = make_batch(blocks, block_indices[a:a + batch_size], device)
                t = df.sample_t(ids.shape[0], gen, stratified=True)
                with amp():
                    num, den, _ = df.loss_sums(model, ids, protect, cfg.mask_id, cfg.pad_id, t, schedule, gen)
                num_d += float(num.sum())
                den_d += float(den.sum())
            per_draw.append(num_d / max(den_d, 1.0))
            num_tot += num_d
            den_tot += den_d
    model.train(was_training)
    return {"nats_per_token": num_tot / max(den_tot, 1.0), "total_nats": num_tot / draws, "maskable_tokens": den_tot / draws,
            "blocks": len(block_indices), "draws": draws, "per_draw": per_draw}


def unigram_baseline(cache_dir) -> float:
    """Cross-entropy (nats/token) of a model that knows only train token frequencies, on the validation tokens that
    are not protected prompt tokens. Add-one smoothing; every validation Story counts."""
    cache_dir = Path(cache_dir)
    m = json.loads((cache_dir / CACHE_MANIFEST).read_text())
    V, pps = m["vocab_size"], m["protected_per_story"]

    def prefix_idx(offs):
        return (offs[:-1, None] + np.arange(pps)[None, :]).ravel()

    tr = np.load(cache_dir / "train_tokens.npy", mmap_mode="r")
    tr_offs = np.load(cache_dir / "train_offsets.npy")
    counts = np.zeros(V, dtype=np.int64)
    for a in range(0, len(tr), 50_000_000):
        counts += np.bincount(np.asarray(tr[a:a + 50_000_000], dtype=np.int64), minlength=V)
    counts -= np.bincount(np.asarray(tr[prefix_idx(tr_offs)], dtype=np.int64), minlength=V)
    p = (counts + 1) / (counts + 1).sum()
    va = np.load(cache_dir / "validation_tokens.npy")
    va_offs = np.load(cache_dir / "validation_offsets.npy")
    keep = np.ones(len(va), dtype=bool)
    keep[prefix_idx(va_offs)] = False
    return float(-np.log(p[va[keep].astype(np.int64)]).mean())


# ---------------------------------------------------------------- real Stories (reference rows, prompts)
def real_story_info(blocks: StoryBlocks, tok, story_indices) -> dict:
    """For eligible validation Stories: token ids between the header and the first end token, their lengths,
    decoded texts and character counts."""
    end_id = specials(tok)["<|end|>"]
    pps = blocks.protected_per_story
    toks, texts = [], []
    for k in story_indices:
        arr = np.asarray(blocks.tokens[int(blocks.offsets[k]) + pps:int(blocks.offsets[k + 1])], dtype=np.int64)
        hits = np.flatnonzero(arr == end_id)
        story = arr[:hits[0]] if len(hits) else arr
        toks.append(story.tolist())
        texts.append(tok.decode(story.tolist(), skip_special_tokens=True))
    return {"tokens": toks, "texts": texts, "chars": [len(t) for t in texts]}


def pick_prompts(blocks: StoryBlocks, tok, n: int, words: int = PROMPT_WORDS, seed: int = 0) -> list[str]:
    """First `words` words of `n` seeded validation Stories (distinct Stories)."""
    picks = random.Random(seed).sample(range(len(blocks.eligible)), min(n, len(blocks.eligible)))
    info = real_story_info(blocks, tok, [int(blocks.eligible[i]) for i in picks])
    return [" ".join(t.split()[:words]) for t in info["texts"]]


# ---------------------------------------------------------------- sampling a setting
def _sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def steps_for(tokens_per_step: int, gen_len: int, block_length: int | None = DEFAULT_BLOCK) -> int:
    """Total `steps` to hand the sampler so that every decoding block takes ceil(block / k) steps (k tokens per step);
    one global pass takes ceil(gen_len / k). (An explicit total would otherwise be spread evenly over the blocks.)
    Exact when gen_len is a multiple of the block, as with the default 480 = 15 x 32; a shorter last block is capped
    at one token per step by the sampler, so it takes min(steps per block, its length) steps."""
    block = gen_len if not block_length or block_length >= gen_len else block_length
    n_blocks = math.ceil(gen_len / block)
    return max(1, math.ceil(block / tokens_per_step)) * n_blocks


def run_setting(model, tok, setting: dict, seeds: list[int], prompts: list[str] | None = None,
                max_new_tokens: int = DEFAULT_MAX_NEW, temperature: float = DEFAULT_TEMPERATURE,
                device: str = "cpu") -> dict:
    """Generate one Story per seed (or per prompt) under `setting`; returns samples + timing."""
    header_len = len(tok.encode(HEADER, add_special_tokens=False).ids)
    samples = []
    cfg = model.config
    for i, seed in enumerate(seeds):
        prompt = prompts[i] if prompts is not None else ""
        prompt_len = len(tok.encode(prompt, add_special_tokens=False).ids) if prompt else 0
        gen_len = min(max_new_tokens, cfg.seq_len - (1 + header_len + prompt_len))
        _sync(device)
        t0 = time.perf_counter()
        r = generate_text(model, tok, prompt, seed=seed, max_new_tokens=max_new_tokens,
                          steps=steps_for(setting["tokens_per_step"], gen_len, setting["block_length"]),
                          block_length=setting["block_length"],
                          temperature=temperature, schedule=setting["schedule"])
        _sync(device)
        elapsed = time.perf_counter() - t0
        gen = r["ids"][r["prefix_len"]:r["covered"]]
        story = gen if r["end_index"] is None else gen[:r["end_index"]]
        samples.append({"seed": seed, "prompt": prompt, "text": r["text"], "tokens": story, "finished": not r["truncated"],
                        "warnings": r["warnings"], "calls": r["forward_calls"], "seconds": elapsed})
    return {"setting": setting, "samples": samples, "throughput": throughput_of(samples)}


def throughput_of(samples: list[dict]) -> dict:
    """Model calls, seconds and rates over a list of samples (each carries its own `calls` and `seconds`)."""
    calls = sum(s["calls"] for s in samples)
    secs = sum(s["seconds"] for s in samples)
    n_tok = sum(len(s["tokens"]) for s in samples)
    return {"model_calls": calls, "seconds": round(secs, 3), "ms_per_call": round(1000 * secs / max(calls, 1), 2),
            "seconds_per_story": round(secs / max(len(samples), 1), 3),
            "story_tokens_per_second": round(n_tok / max(secs, 1e-9), 1)}


def setting_label(s: dict) -> str:
    dec = "global" if s["block_length"] == 0 else f"blocks of {s['block_length']}"
    return f"{s['tokens_per_step']} tok/step, {dec}, {s['schedule']}"


def sweep_settings() -> list[dict]:
    return [{"tokens_per_step": k, "block_length": b, "schedule": sch}
            for sch in SCHEDULES for b in DECODINGS for k in TOKENS_PER_STEP]


# ---------------------------------------------------------------- the report
def summarize(res: dict, scorer, with_text: int = 0) -> dict:
    out = {"setting": res["setting"], "stats": story_stats(res["samples"]),
           "gen_ppl": gen_ppl([s["text"] for s in res["samples"]], scorer), "throughput": res["throughput"]}
    if with_text:
        out["first_texts"] = [s["text"] for s in res["samples"][:with_text]]
    return out


def build_report(model, tok, cache: Path, ckpt: Path, a, device: str) -> tuple[dict, dict]:
    """Returns (report dict, samples dict for samples.md)."""
    cfg = model.config
    amp, precision = make_amp(device, "auto")
    info_path = ckpt / "run_info.json"
    info = json.loads(info_path.read_text()) if info_path.is_file() else {}
    train_schedule = info.get("recipe", {}).get("schedule", "linear")
    val = StoryBlocks(cache, "validation", cfg.seq_len)
    idx = evenly(len(val), a.max_val_blocks) if a.max_val_blocks else list(range(len(val)))
    print(f"likelihood: {len(idx):,} validation blocks x {a.draws} draws ...", flush=True)
    lin = elbo_bound(model, val, cfg, device, amp, idx, a.draws, "linear")
    chars = sum(len(t) for t in real_story_info(
        val, tok, [int(k) for b in idx for k in val.eligible[val.bounds[b][0]:val.bounds[b][1]]])["texts"])
    like = {"schedule_reported": "linear", "nats_per_token": lin["nats_per_token"],
            "bits_per_char": bits_per_char(lin["total_nats"], chars), "chars": chars, "blocks": lin["blocks"],
            "draws": a.draws, "draw_spread_nats": max(lin["per_draw"]) - min(lin["per_draw"]),
            "unigram_baseline_nats_per_token": unigram_baseline(cache), "train_schedule": train_schedule}
    if train_schedule != "linear":
        own = elbo_bound(model, val, cfg, device, amp, idx, a.draws, train_schedule)
        like["own_schedule"] = {"schedule": train_schedule, "nats_per_token": own["nats_per_token"],
                                "bits_per_char": bits_per_char(own["total_nats"], chars)}
    scorer = None if a.no_gpt2 else gpt2_scorer(device)

    rng = random.Random(0)
    ref_picks = [int(val.eligible[i]) for i in rng.sample(range(len(val.eligible)), min(a.n_stories, len(val.eligible)))]
    ref = real_story_info(val, tok, ref_picks)
    ref_samples = [{"tokens": t, "finished": True, "warnings": []} for t in ref["tokens"]]
    reference = {"stats": story_stats(ref_samples), "gen_ppl": gen_ppl(ref["texts"], scorer), "n": len(ref_samples)}
    texts_for_md = {"real": ref["texts"][:3]}

    print(f"headline: {a.n_stories} unconditional + {a.n_prompts} continue Stories ...", flush=True)
    kw = dict(max_new_tokens=a.max_new_tokens, device=device)
    uncond = run_setting(model, tok, HEADLINE, list(range(a.n_stories)), **kw)
    prompts = pick_prompts(val, tok, a.n_prompts)
    cont = run_setting(model, tok, HEADLINE, [1000 + i for i in range(len(prompts))], prompts=prompts, **kw)
    headline = {"setting": HEADLINE, "unconditional": summarize(uncond, scorer), "continue": summarize(cont, scorer),
                "reference_real_stories": reference}
    texts_for_md["unconditional"] = [s["text"] for s in uncond["samples"][:10]]
    texts_for_md["continue"] = [(s["prompt"], s["text"]) for s in cont["samples"][:5]]

    sweep = []
    if a.sweep:
        todo = sweep_settings()
        for n, s in enumerate(todo, 1):
            if s == HEADLINE:  # already generated (seeds 0..199): reuse its first sweep_n Stories, no extra model calls
                print(f"sweep {n}/{len(todo)}: {setting_label(s)} (reusing the headline Stories)", flush=True)
                part = uncond["samples"][:a.sweep_n]
                res = {"setting": s, "samples": part, "throughput": throughput_of(part)}
            else:
                print(f"sweep {n}/{len(todo)}: {setting_label(s)} ({a.sweep_n} Stories)", flush=True)
                res = run_setting(model, tok, s, list(range(a.sweep_n)), **kw)
            sweep.append(summarize(res, scorer, with_text=2))
    report = {"created_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "ckpt": str(ckpt),
              "model_parameters": sum(p.numel() for p in model.parameters()), "device": device, "precision": precision,
              "settings": {"n_stories": a.n_stories, "n_prompts": len(prompts), "draws": a.draws,
                           "max_new_tokens": a.max_new_tokens, "temperature": DEFAULT_TEMPERATURE,
                           "sweep": a.sweep, "sweep_n": a.sweep_n, "max_val_blocks": a.max_val_blocks,
                           "gpt2_perplexity": scorer is not None},
              "likelihood": like, "headline": headline, "sweep": sweep}
    return report, {**texts_for_md, "sweep": [(setting_label(r["setting"]), r.get("first_texts", [])) for r in sweep]}


def _f(x, nd=3):
    return "n/a" if x is None else f"{x:.{nd}f}"


def render_markdown(rep: dict) -> str:
    L = rep["likelihood"]
    out = [f"# Evaluation report\n\nCheckpoint: `{rep['ckpt']}` | {rep['model_parameters'] / 1e6:.2f}M parameters | "
           f"device {rep['device']} ({rep['precision']}) | {rep['created_at']}\n",
           "## Likelihood (validation ELBO bound)\n",
           "| metric | value |\n|---|---|",
           f"| nats per token (linear schedule) | **{L['nats_per_token']:.4f}** |",
           f"| bits per character | **{_f(L['bits_per_char'], 4)}** |",
           f"| unigram baseline (nats/token) | {L['unigram_baseline_nats_per_token']:.4f} |",
           f"| blocks x noise draws | {L['blocks']:,} x {L['draws']} (draw-to-draw spread {L['draw_spread_nats']:.4f} nats) |"]
    if "own_schedule" in L:
        o = L["own_schedule"]
        out.append(f"| under its own schedule ({o['schedule']}) | {o['nats_per_token']:.4f} nats/token, {_f(o['bits_per_char'], 4)} bits/char |")
    h = rep["headline"]
    out += ["", f"## Samples at the headline setting ({setting_label(h['setting'])})\n",
            "| set | n | finished | length median / p90 | distinct-1/2/3 | repetition | collapse | GPT-2 ppl |",
            "|---|---|---|---|---|---|---|---|"]
    for name, row in (("unconditional", h["unconditional"]), ("continue", h["continue"]),
                      ("real validation Stories", h["reference_real_stories"])):
        s = row["stats"]
        out.append(f"| {name} | {s['n']} | {s['finished_rate']:.0%} | {s['length']['median']} / {s['length']['p90']} | "
                   f"{s['distinct_1']:.3f} / {s['distinct_2']:.3f} / {s['distinct_3']:.3f} | {s['repetition_rate']:.1%} | "
                   f"{s['collapse_rate']:.0%} | {_f(row.get('gen_ppl'), 1)} |")
    out.append("\nGPT-2 perplexity is meaningful only relative to the real-Stories row (GPT-2 knows web text, not TinyStories).")
    if rep["sweep"]:
        out += ["", "## Decoding sweep\n",
                "| setting | n | finished | length median | distinct-2 | repetition | collapse | GPT-2 ppl | ms/call | s/Story |",
                "|---|---|---|---|---|---|---|---|---|---|"]
        for r in rep["sweep"]:
            s, t = r["stats"], r["throughput"]
            out.append(f"| {setting_label(r['setting'])} | {s['n']} | {s['finished_rate']:.0%} | {s['length']['median']} | "
                       f"{s['distinct_2']:.3f} | {s['repetition_rate']:.1%} | {s['collapse_rate']:.0%} | {_f(r['gen_ppl'], 1)} | "
                       f"{t['ms_per_call']} | {t['seconds_per_story']} |")
    t = h["unconditional"]["throughput"]
    out += ["", "## Throughput (headline setting, one Story at a time; timings vary from run to run)\n",
            f"{t['ms_per_call']} ms per model call, {t['seconds_per_story']} s per Story, "
            f"{t['story_tokens_per_second']} Story tokens per second ({t['model_calls']} model calls)."]
    return "\n".join(out) + "\n"


def render_samples(texts: dict) -> str:
    out = ["# Samples to read\n\nFixed seeds: re-running reproduces these. Read them: numbers cannot tell you whether they are good.\n",
           "## Real validation Stories (what good looks like)\n"]
    out += [f"**{i + 1}.** {t}\n" for i, t in enumerate(texts["real"])]
    out += ["## Unconditional, seeds 0-9 (headline setting)\n"]
    out += [f"**seed {i}.** {t}\n" for i, t in enumerate(texts["unconditional"])]
    out += ["## Continue (opening words are given)\n"]
    out += [f"**prompt:** {p}\n\n**result:** {t}\n" for p, t in texts["continue"]]
    if texts.get("sweep"):
        out += ["## Decoding sweep (first 2 Stories per setting, seeds 0-1)\n"]
        for label, ts in texts["sweep"]:
            out.append(f"### {label}\n")
            out += [f"{i}. {t}\n" for i, t in enumerate(ts)]
    return "\n".join(out) + "\n"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate a trained run (python run.py eval --ckpt <run>/final).")
    p.add_argument("--ckpt", type=Path, required=True, help="a run's final/ folder")
    p.add_argument("--out", type=Path, default=None, help="output folder (default: <run>/eval)")
    p.add_argument("--cache", type=Path, default=None, help="Phase 2 token cache (default: <data dir>/tokens)")
    p.add_argument("--n-stories", type=int, default=200, help="unconditional Stories (and real reference Stories)")
    p.add_argument("--n-prompts", type=int, default=50, help="continue prompts")
    p.add_argument("--draws", type=int, default=4, help="noise draws per validation block")
    p.add_argument("--max-val-blocks", type=int, default=None, help="evaluate only this many evenly spaced validation blocks")
    p.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW)
    p.add_argument("--sweep", action="store_true", help="also run the 24-setting decoding sweep (long)")
    p.add_argument("--sweep-n", type=int, default=50, help="Stories per sweep setting")
    p.add_argument("--no-gpt2", action="store_true", help="skip the GPT-2 reference perplexity")
    p.add_argument("--device", default="auto", help="auto | cuda | mps | cpu")
    return p


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    cache = (a.cache or default_out_dir() / "tokens").expanduser().resolve()
    ckpt = a.ckpt.expanduser().resolve()
    if not (cache / CACHE_MANIFEST).is_file():
        print(f"eval: no token cache in {cache}. Run: python run.py phase2", file=sys.stderr)
        return 1
    device = pick_device(a.device)
    try:
        model, tok = load_checkpoint(ckpt, device)
    except FileNotFoundError as e:
        print(f"eval: {e}", file=sys.stderr)
        return 1
    out = (a.out or ckpt.parent / "eval").expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    report, texts = build_report(model, tok, cache, ckpt, a, device)
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (out / "report.md").write_text(render_markdown(report))
    (out / "samples.md").write_text(render_samples(texts))
    print(render_markdown(report))
    print(f"wrote {out / 'report.json'}, report.md and samples.md")
    return 0
