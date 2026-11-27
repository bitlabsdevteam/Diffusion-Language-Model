"""Control evaluation: does the fine-tuned model follow the instruction better than the base model?

Same held-out prompts (built from TinyStories-Instruct-valid, fixed seeds) go to BOTH models; the base model sees the
instruction as the same protected prefix, it just was never trained to use it. Conditions (n prompts each):

  plain     "Write a short story."                      reference: quote rate, length, repetition without any control
  words     + Words: a, b, c                            metric: share of the required words present in the story
  sentence  + Random sentence: ...                      metric: the sentence occurs verbatim (and its word recall)
  dialogue  + Features: Dialogue                        metric: the story contains a quoted passage (>= 2 quote marks)
  summary   + Summary: ...                              metric: recall of the summary's content words (a weak proxy)
  all       every available field                       all of the above that apply

Also per condition: finished rate, length, distinct-n and repetition rate (reused from dlm.evaluate), and the
unconditional validation bound (forgetting guard) when the pre-training token cache is available. BadEnding / MoralValue /
Twist adherence needs a judge (TinyStories "GPT-Eval"); none is called here (it would send stories to an external service).
Outputs: <out>/control_report.json, control_report.md, control_samples.md.
"""
from __future__ import annotations

import _path  # noqa: F401

import argparse
import json
import random
import re
import statistics
import sys
from pathlib import Path

import torch

import ft_data as D
import ft_sample as S
from dlm.evaluate import distinct_n, length_stats, repetition_rate
from dlm.model import load_model
from dlm.train import evenly, make_amp, pick_device

CONDITIONS = ("plain", "words", "sentence", "dialogue", "summary", "all")
_STOP = set("the a an and or but of to in on at with for from by as is was were are be been it its he she they them his her their "
            "this that then there when what who which into out up down over very just so not no".split())
_WORD = re.compile(r"[a-z']+")


# ------------------------------------------------------------------ prompts
def select_prompts(cache: D.SftCache, n: int, seed: int = 0) -> dict[str, list[dict]]:
    """{condition: [{"instruction", "fields"(what was asked), "index"}]} from the held-out cache; deterministic."""
    rng = random.Random(f"eval-{seed}")
    order = list(range(len(cache)))
    rng.shuffle(order)
    out: dict[str, list[dict]] = {c: [] for c in CONDITIONS}
    for i in order:
        f = cache.fields[i]
        cands = {
            "plain": {},
            "words": {"w": f["w"]} if f.get("w") else None,
            "sentence": {"r": f["r"]} if f.get("r") else None,
            "dialogue": {"f": ["Dialogue"]} if "Dialogue" in f.get("f", []) else None,
            "summary": {"s": f["s"]} if f.get("s") else None,
            "all": {k: v for k, v in f.items() if v} if len([k for k, v in f.items() if v]) >= 3 else None,
        }
        for cond, req in cands.items():
            if req is None or len(out[cond]) >= n:
                continue
            out[cond].append({"index": i, "fields": req,
                              "instruction": S.make_instruction(req.get("w"), req.get("f"), req.get("s"), req.get("r"))})
        if all(len(v) >= n for v in out.values()):
            break
    return out


# ------------------------------------------------------------------ per-sample metrics
def _norm(s: str) -> str:
    return " ".join(s.lower().replace("’", "'").split())


def content_words(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if len(w) > 3 and w not in _STOP}


def score(text: str, req: dict) -> dict:
    """Metrics that apply to the requested fields (keys absent when not asked)."""
    t = _norm(text)
    words_in = set(_WORD.findall(t))
    m: dict = {"quotes": text.count('"') >= 2}
    if req.get("w"):
        m["words"] = sum(1 for w in req["w"] if _norm(w) in words_in) / len(req["w"])
    if req.get("r"):
        sent = _norm(req["r"])
        m["sentence_exact"] = sent in t
        sw = _WORD.findall(sent)
        m["sentence_recall"] = sum(1 for w in sw if w in words_in) / max(len(sw), 1)
    if "Dialogue" in req.get("f", []):
        m["dialogue"] = m["quotes"]
    if req.get("s"):
        cw = content_words(req["s"])
        m["summary_recall"] = len(cw & content_words(text)) / max(len(cw), 1)
    return m


def mean(xs):
    xs = [float(x) for x in xs]
    return round(sum(xs) / len(xs), 4) if xs else None


# ------------------------------------------------------------------ generation + aggregation
def run_model(model, tok, prompts: dict[str, list[dict]], seed: int, gen_kwargs: dict, log=print) -> dict:
    res: dict = {}
    for cond, items in prompts.items():
        samples, texts = [], []
        for k, it in enumerate(items):
            r = S.generate_story(model, tok, it["instruction"], seed=seed + 1000 * CONDITIONS.index(cond) + k, **gen_kwargs)
            ids = S.story_token_ids(r)
            samples.append({"tokens": ids, "finished": r["end_index"] is not None, "warnings": r["warnings"],
                            "metrics": score(r["text"], it["fields"])})
            texts.append(r["text"])
        agg = {"n": len(samples),
               "finished_rate": mean(s["finished"] for s in samples),
               "length": length_stats([len(s["tokens"]) for s in samples]),
               "distinct_2": round(distinct_n([s["tokens"] for s in samples], 2), 4),
               "repetition_rate": round(repetition_rate([s["tokens"] for s in samples], 4), 4),
               "collapse_rate": mean(bool(s["warnings"]) for s in samples)}
        for key in ("quotes", "words", "sentence_exact", "sentence_recall", "dialogue", "summary_recall"):
            vals = [s["metrics"][key] for s in samples if key in s["metrics"]]
            if vals:
                agg[key] = mean(vals)
        res[cond] = {"agg": agg, "texts": texts}
        log(f"  {cond:<9} n={len(samples)} " + " ".join(f"{k}={v}" for k, v in agg.items() if k in ("words", "sentence_exact", "dialogue", "summary_recall", "finished_rate")))
    return res


HEADLINE = {"words": ("words", "required words present"), "sentence": ("sentence_exact", "random sentence verbatim"),
            "dialogue": ("dialogue", "dialogue (quoted speech) present"), "summary": ("summary_recall", "summary content-word recall")}


def render_markdown(rep: dict) -> str:
    L = ["# Control evaluation: base vs fine-tuned", "",
         f"base: `{rep['base_ckpt']}`  \nSFT: `{rep['sft_ckpt']}`  \nprompts per condition: {rep['n_per_condition']} (held-out TinyStories-Instruct valid, seed {rep['seed']}); "
         f"max_new_tokens {rep['gen']['max_new_tokens']}, temperature {rep['gen']['temperature']}", "",
         "## Headline (higher is better)", "", "| condition | metric | base | SFT | delta |", "|---|---|---|---|---|"]
    for cond, (key, label) in HEADLINE.items():
        b, s = rep["base"][cond]["agg"].get(key), rep["sft"][cond]["agg"].get(key)
        d = f"{s - b:+.3f}" if b is not None and s is not None else "n/a"
        L.append(f"| {cond} | {label} | {b} | {s} | {d} |")
    L += ["", "## Quality and length (per condition)", "", "| condition | model | finished | median len | p90 len | distinct-2 | repetition | collapse |", "|---|---|---|---|---|---|---|---|"]
    for cond in rep["base"]:
        for name in ("base", "sft"):
            a = rep[name][cond]["agg"]
            L.append(f"| {cond} | {name} | {a['finished_rate']} | {a['length']['median']} | {a['length']['p90']} | {a['distinct_2']} | {a['repetition_rate']} | {a['collapse_rate']} |")
    if rep.get("uncond"):
        u = rep["uncond"]
        L += ["", "## Forgetting guard (plain-prompt validation bound, nats/token, lower is better)", "",
              f"base {u['base']:.4f} | SFT {u['sft']:.4f} | drift {u['sft'] - u['base']:+.4f}"]
    L += ["", "BadEnding / MoralValue / Twist adherence is not measured here (needs a judge). The summary metric is a weak lexical proxy.",
          "Read `control_samples.md` before trusting any number."]
    return "\n".join(L) + "\n"


def render_samples(rep: dict, prompts: dict, k: int = 3) -> str:
    out = ["# Samples (same prompt, both models)", ""]
    for cond, items in prompts.items():
        for j, it in enumerate(items[:k]):
            out += [f"## {cond} #{j}", "", "```", it["instruction"], "```", "",
                    "**base**", "", rep["_texts"]["base"][cond][j].strip(), "", "**SFT**", "", rep["_texts"]["sft"][cond][j].strip(), ""]
    return "\n".join(out)


def uncond_bounds(models: dict, pre_cache: Path | None, seq_len: int, device: str, blocks: int = 512):
    if pre_cache is None or not (Path(pre_cache) / "tokens_manifest.json").is_file():
        return None
    from dlm.data import StoryBlocks
    from dlm.evaluate import elbo_bound
    sb = StoryBlocks(pre_cache, "validation", seq_len)
    amp, _ = make_amp(device, "auto")
    out = {}
    for name, m in models.items():
        out[name] = elbo_bound(m, sb, m.config, device, amp, evenly(len(sb), blocks))["nats_per_token"]
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Compare the base and the fine-tuned model on instruction control.")
    p.add_argument("--base", required=True, help="pre-trained checkpoint folder")
    p.add_argument("--sft", required=True, help="fine-tuned checkpoint folder (final/ or best/)")
    p.add_argument("--cache", type=Path, default=_path.FT_DIR / "data" / "cache")
    p.add_argument("--pretrain-cache", type=Path, default=_path.DLM_ROOT / "data" / "tokens")
    p.add_argument("--tokenizer-dir", type=Path, default=_path.DLM_ROOT / "tokenizer")
    p.add_argument("--out", type=Path, default=None, help="report folder (default: <sft>/../control_eval)")
    p.add_argument("-n", "--n", type=int, default=30, help="prompts per condition")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-new-tokens", type=int, default=400)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--steps", type=int, default=None, help="total sampler steps per story (default: 1 token per step)")
    p.add_argument("--schedule", choices=("auto", "linear", "cosine"), default="auto", help="sampling noise schedule; auto = the one the model was trained with")
    p.add_argument("--device", default="auto")
    return p


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    from tokenizers import Tokenizer
    device = pick_device(a.device)
    tok = Tokenizer.from_file(str(Path(a.tokenizer_dir) / "tokenizer.json"))
    cache = D.SftCache(a.cache, "valid")
    prompts = select_prompts(cache, a.n, a.seed)
    short = [c for c, v in prompts.items() if len(v) < a.n]
    if short:
        print(f"note: fewer than {a.n} prompts for {short}: {[len(prompts[c]) for c in short]}", file=sys.stderr)
    if a.schedule == "auto":
        a.schedule = S.infer_schedule(a.sft) if (Path(a.sft) / "run_info.json").is_file() or (Path(a.sft).parent / "run_info.json").is_file() else S.infer_schedule(a.base)
    gen_kwargs = {"max_new_tokens": a.max_new_tokens, "temperature": a.temperature, "schedule": a.schedule}
    if a.steps:
        gen_kwargs["steps"] = a.steps
    models = {"base": load_model(a.base, device), "sft": load_model(a.sft, device)}
    rep: dict = {"base_ckpt": str(a.base), "sft_ckpt": str(a.sft), "n_per_condition": a.n, "seed": a.seed,
                 "gen": {"max_new_tokens": a.max_new_tokens, "temperature": a.temperature, "schedule": a.schedule, "steps": a.steps}}
    texts = {}
    for name, m in models.items():
        print(f"[{name}] generating ...", flush=True)
        r = run_model(m, tok, prompts, a.seed, gen_kwargs)
        rep[name] = {c: {"agg": v["agg"]} for c, v in r.items()}
        texts[name] = {c: v["texts"] for c, v in r.items()}
    u = uncond_bounds(models, a.pretrain_cache, models["base"].config.seq_len, device)
    rep["uncond"] = u
    out = (a.out or Path(a.sft).parent / "control_eval").expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "control_report.json").write_text(json.dumps(rep, indent=2) + "\n")
    (out / "control_report.md").write_text(render_markdown(rep))
    rep["_texts"] = texts
    (out / "control_samples.md").write_text(render_samples(rep, prompts))
    print(render_markdown(rep))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
