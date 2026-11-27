"""Supervised fine-tuning (SFT) of the pre-trained DLM for controllable story generation.

Same loop as dlm/train.py (plain PyTorch, AdamW, warmup + cosine, bf16 on CUDA, bit-exact resume), but with `--init-from`
and the SFT data from ft_data.py. dlm/ is imported, never modified (that is why this file exists instead of wiring
`--task sft` into dlm/train.py).

  * Loss: dlm.diffusion.diffusion_loss with the prompt in `protect`: the prompt is never masked and never in the loss;
    the loss is the ELBO-weighted CE over masked RESPONSE tokens (response = story + <|end|> + [EOS]).
  * Recipe (starting points, sweep lr in {2e-5, 5e-5}): peak LR 3e-5 (about 7x below pre-training), 200 warmup steps,
    cosine to 10%, weight decay 0.1 on matrices, clip 1.0, batch 32 blocks, 3,000 steps.
  * Evaluation every --eval-every steps, all with fixed noise (comparable across steps and with the base model):
      val        SFT validation bound (fixed prompts)   <- picks best/
      train_eval SFT bound on a fixed slice of training examples
      uncond     the PRE-TRAINING validation bound on plain "Write a short story." Stories (forgetting guard)
    A step-0 evaluation of the untouched base model is recorded first, so the improvement is read off history.json.
  * --lora-r R > 0 trains LoRA adapters only (ablation arm; ft_lora.py); best/ and final/ are always merged, plain checkpoints.
  * Resume: resume.pt (model, optimizer, step, history, RNG). The data stream is a pure function of (seed, block index),
    so no data-loader state is needed. A changed recipe refuses to resume.
Outputs under --out: best/ final/ (model.pt, config.json, tokenizer/, ...), history.json, training_curves.png, run_info.json.
"""
from __future__ import annotations

import _path  # noqa: F401

import argparse
import datetime as dt
import json
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

from dlm import diffusion as df
from dlm.download import COLAB_DRIVE
from dlm.model import DiffusionTransformerLM, describe, load_model
from dlm.train import (atomic_save, evenly, evaluate, fmt_time, lr_at, make_amp, make_batch, pick_device, read_best,
                       rng_state, save_best, set_rng_state)
import ft_data as D
import ft_lora as L
from ft_sample import infer_schedule  # noqa: F401  (re-exported: the base model's noise schedule)

FT_DIR = _path.FT_DIR
DLM_ROOT = _path.DLM_ROOT
RECIPE_KEYS = ("init_from", "steps", "batch_size", "lr", "warmup", "min_lr_ratio", "weight_decay", "beta1", "beta2", "clip",
               "seq_len", "seed", "schedule", "plain_prob", "lora_r", "lora_alpha", "lora_dropout", "train_examples")
EVAL_BLOCKS = 256


class SftError(RuntimeError):
    pass


def trainable_groups(model, weight_decay: float):
    """Decay on matrices only (ndim >= 2); none on 1-D tensors; frozen parameters are left out."""
    params = [p for p in model.parameters() if p.requires_grad]
    return [{"params": [p for p in params if p.ndim >= 2], "weight_decay": weight_decay},
            {"params": [p for p in params if p.ndim < 2], "weight_decay": 0.0}]


def recipe_of(a, cfg) -> dict:
    return {**{k: getattr(a, k) for k in RECIPE_KEYS}, "model": cfg.to_dict()}


def export_model(model):
    """The plain DiffusionTransformerLM to save/evaluate (adapters merged when LoRA is on)."""
    return L.merge_lora(model) if L.has_lora(model) else model


def plot_curves(history: list, path: Path) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed: skipping training_curves.png (history.json has the data)")
        return False
    tr = [h for h in history if "loss" in h]
    ev = [h for h in history if "val" in h]
    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True, gridspec_kw={"height_ratios": [3, 2, 1.2]})
    if tr:
        axes[0].plot([h["step"] for h in tr], [h["loss"] for h in tr], color="tab:blue", alpha=0.3, label="train (per log interval)")
    if ev:
        axes[0].plot([h["step"] for h in ev], [h["val"] for h in ev], "o-", color="tab:red", label="SFT validation")
        axes[0].plot([h["step"] for h in ev], [h["train_eval"] for h in ev], "s--", color="tab:green", label="SFT train-eval")
        un = [h for h in ev if "uncond" in h]
        if un:
            axes[1].plot([h["step"] for h in un], [h["uncond"] for h in un], "o-", color="tab:purple", label="unconditional validation bound")
            axes[1].axhline(un[0]["uncond"], color="gray", ls=":", label="base model (step 0)")
            axes[1].legend(fontsize=8)
    axes[0].set_ylabel("ELBO bound (nats/token)")
    axes[0].legend(fontsize=8)
    axes[1].set_ylabel("forgetting guard\n(plain-prompt bound)")
    if tr:
        axes[2].plot([h["step"] for h in tr], [h["lr"] for h in tr], color="tab:orange")
    axes[2].set_ylabel("learning rate")
    axes[2].set_xlabel("step")
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return True


def write_live(out: Path, history: list) -> None:
    tmp = out / "history.json.tmp"
    tmp.write_text(json.dumps(history, indent=1) + "\n")
    os.replace(tmp, out / "history.json")
    plot_curves(history, out / "training_curves.png")


def eval_all(model, sets: dict, cfg, device, amp) -> dict:
    """{"val", "train_eval", "uncond"?}: fixed-noise per-token bounds (linear schedule, so they are comparable)."""
    out = {"val": evaluate(model, sets["val"], evenly(len(sets["val"]), EVAL_BLOCKS), cfg, device, amp),
           "train_eval": evaluate(model, sets["train_eval"], evenly(len(sets["train_eval"]), EVAL_BLOCKS), cfg, device, amp)}
    if sets.get("uncond") is not None:
        out["uncond"] = evaluate(model, sets["uncond"], evenly(len(sets["uncond"]), EVAL_BLOCKS), cfg, device, amp)
    return out


def load_sets(a, tok, cfg) -> tuple[D.SftStream, dict]:
    cache_tr = D.SftCache(a.cache, "train")
    cache_va = D.SftCache(a.cache, "valid")
    if cache_tr.sp["[MASK]"] != cfg.mask_id or cache_tr.sp["[PAD]"] != cfg.pad_id:
        raise SftError("special-token ids in the cache differ from the model config: rebuild the cache with the model's tokenizer")
    stream = D.SftStream(cache_tr, tok, a.seq_len, a.seed, a.plain_prob)
    sets = {"val": D.PackedSet(cache_va, tok, a.seq_len, seed=0, plain_prob=a.plain_prob, limit=a.val_examples),
            "train_eval": D.PackedSet(cache_tr, tok, a.seq_len, seed=0, plain_prob=a.plain_prob, limit=a.val_examples),
            "uncond": None}
    pre = Path(a.pretrain_cache).expanduser() if a.pretrain_cache else None
    if pre is not None and (pre / "tokens_manifest.json").is_file():
        from dlm.data import StoryBlocks
        sets["uncond"] = StoryBlocks(pre, "validation", a.seq_len)
    else:
        print("forgetting guard OFF: no pre-training token cache (pass --pretrain-cache DIR, default DLM/data/tokens)", flush=True)
    return stream, sets


def run_sft(a, out: Path, tok) -> dict:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    init = Path(a.init_from).expanduser().resolve()
    device = pick_device(a.device)
    amp, precision = make_amp(device, a.precision)
    torch.manual_seed(a.seed)
    random.seed(a.seed)
    np.random.seed(a.seed)
    model = load_model(init, device)
    cfg = model.config
    if a.seq_len > cfg.seq_len:
        raise SftError(f"--seq-len {a.seq_len} exceeds the model's window {cfg.seq_len} (RoPE table)")
    if a.schedule == "auto":
        a.schedule = infer_schedule(init)
    stream, sets = load_sets(a, tok, cfg)
    n_trainable = None
    if a.lora_r:
        n_trainable = L.apply_lora(model, a.lora_r, a.lora_alpha, a.lora_dropout)
    print(f"model: {describe(model)} | device {device} | precision {precision} | schedule {a.schedule}"
          + (f" | LoRA r={a.lora_r}: {n_trainable:,} trainable ({n_trainable / model.num_parameters():.1%})" if a.lora_r else " | full fine-tune"),
          flush=True)
    opt = torch.optim.AdamW(trainable_groups(model, a.weight_decay), lr=a.lr, betas=(a.beta1, a.beta2))
    gen = torch.Generator().manual_seed(a.seed + 1)
    recipe = recipe_of(a, cfg)
    history: list[dict] = []
    best = None
    start = 0
    resume_path = out / "resume.pt"
    if a.resume and resume_path.exists():
        ck = torch.load(resume_path, map_location="cpu", weights_only=False)
        diff = {k for k in recipe if ck["recipe"].get(k) != recipe[k]}
        if diff:
            raise SystemExit(f"Refusing to resume: recipe differs from the checkpoint in {sorted(diff)}")
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        history, start = ck["history"], ck["step"]
        best = read_best(out)
        set_rng_state(ck["rng"], gen)
        print(f"resumed from step {start}", flush=True)
    elif a.resume:
        print("no resume.pt found: starting from scratch", flush=True)

    ex_model = export_model(model)  # a no-op alias unless LoRA is on
    if start == 0:
        base = eval_all(ex_model, sets, cfg, device, amp)
        history.append({"step": 0, "epoch": 0.0, "base": True, **base, "gap": base["val"] - base["train_eval"]})
        print("step 0 (base model) | " + " | ".join(f"{k} {v:.4f}" for k, v in base.items()), flush=True)
        best = None
        if a.gate_uncond_max > 0 and base.get("uncond") is not None and base["uncond"] > a.gate_uncond_max:
            raise SftError(f"STEP 0 GATE: the base model's plain-prompt validation bound is {base['uncond']:.3f} nats/token, above "
                           f"{a.gate_uncond_max} (the unigram baseline is 5.92). It has not learned much beyond token frequencies, so "
                           f"fine-tuning is wasted: fix pre-training first, or pass --gate-uncond-max 0 to override.")
    n_ex = len(stream.cache)
    end = min(a.steps, a.stop_at) if a.stop_at else a.steps
    model.train()
    run_sum, run_n, gn = torch.zeros((), device=device), 0, torch.zeros(())
    t_start, steps_done = time.time(), 0
    ex_per_block = max(float(np.mean([stream.arrays(i)[2] for i in range(8)])), 1.0)
    for step in range(start, end):
        lr = lr_at(step, a.lr, a.warmup, a.steps, a.min_lr_ratio)
        for g in opt.param_groups:
            g["lr"] = lr
        ids, protect = make_batch(stream, list(range(step * a.batch_size, (step + 1) * a.batch_size)), device)
        with amp():
            loss = df.diffusion_loss(model, ids, protect, cfg.mask_id, cfg.pad_id, a.schedule, gen)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], a.clip)
        opt.step()
        run_sum += loss.detach()
        run_n += 1
        steps_done += 1
        done = step + 1
        if done % a.log_every == 0 or done == end:
            avg = float(run_sum) / run_n
            if not math.isfinite(avg):
                raise RuntimeError(f"loss is {avg} at step {done}: stopping")
            epoch = done * a.batch_size * ex_per_block / max(n_ex, 1)
            history.append({"step": done, "epoch": epoch, "loss": avg, "lr": lr, "grad_norm": float(gn)})
            run_sum, run_n = torch.zeros((), device=device), 0
            if done % a.print_every == 0 or done == end:
                sps = steps_done / max(time.time() - t_start, 1e-9)
                print(f"step {done:>6}/{a.steps} | ~epoch {epoch:.3f} | loss {avg:.4f} | lr {lr:.2e} | grad {float(gn):.2f} | "
                      f"{sps:.2f} steps/s | eta {fmt_time((a.steps - done) / sps)}", flush=True)
        if done % a.eval_every == 0 or done == a.steps:
            ex_model = export_model(model)
            ev = eval_all(ex_model, sets, cfg, device, amp)
            history.append({"step": done, "epoch": done * a.batch_size * ex_per_block / max(n_ex, 1), **ev,
                            "gap": ev["val"] - ev["train_eval"]})
            is_best = best is None or ev["val"] < best["val"]
            if is_best:
                best = {"step": done, **ev}
                save_best(ex_model, out, a.tokenizer_dir, best)
            write_live(out, history)
            tag = "new best, saved to best/" if is_best else f"best {best['val']:.4f} @ step {best['step']}"
            base_u = next((h["uncond"] for h in history if h.get("base") and "uncond" in h), None)
            drift = f" ({ev['uncond'] - base_u:+.4f} vs base)" if base_u is not None and "uncond" in ev else ""
            print(f"EVAL step {done:>6}/{a.steps} | " + " | ".join(f"{k} {v:.4f}" for k, v in ev.items()) + drift + f" | {tag}", flush=True)
        if done % a.ckpt_every == 0 and done < a.steps:
            atomic_save({"model": model.state_dict(), "optimizer": opt.state_dict(), "step": done, "history": history,
                         "rng": rng_state(gen), "recipe": recipe}, resume_path)
    elapsed = time.time() - t_start
    write_live(out, history)
    if end < a.steps:
        atomic_save({"model": model.state_dict(), "optimizer": opt.state_dict(), "step": end, "history": history,
                     "rng": rng_state(gen), "recipe": recipe}, resume_path)
        print(f"stopped at step {end} (--stop-at); resume with --resume", flush=True)
        return {"step": end, "final": False, "history": history}

    ex_model = export_model(model)
    full_val = evaluate(ex_model, sets["val"], list(range(len(sets["val"]))), cfg, device, amp)
    final_u = None
    if sets.get("uncond") is not None:
        final_u = evaluate(ex_model, sets["uncond"], evenly(len(sets["uncond"]), a.final_uncond_blocks), cfg, device, amp)
    print(f"final | full SFT validation bound {full_val:.4f}" + (f" | unconditional bound {final_u:.4f}" if final_u is not None else ""), flush=True)
    final = out / "final"
    ex_model.save_pretrained(final)
    if (Path(a.tokenizer_dir) / "tokenizer.json").is_file():
        shutil.copytree(a.tokenizer_dir, final / "tokenizer", dirs_exist_ok=True)
    (final / "history.json").write_text(json.dumps(history, indent=1) + "\n")
    base0 = next((h for h in history if h.get("base")), None)
    info = {"task": "sft", "recipe": recipe, "init_from": str(init), "device": device, "precision": precision,
            "parameters": ex_model.num_parameters(), "trainable_parameters": n_trainable or ex_model.num_parameters(),
            "seconds": round(elapsed, 1), "final_val_bound": full_val, "final_uncond_bound": final_u, "base_eval": base0,
            "best": best, "torch": torch.__version__,
            "finished_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}
    (final / "run_info.json").write_text(json.dumps(info, indent=2) + "\n")
    plot_curves(history, final / "training_curves.png")
    return {"step": end, "final": True, "history": history, "info": info, "final_dir": final}


def default_out(name: str = "sft") -> Path:
    if COLAB_DRIVE.is_dir():
        return COLAB_DRIVE / "diffusion_lm_outputs" / "checkpoints" / name
    return FT_DIR / "checkpoints" / name


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="SFT of the pre-trained DLM (python fine-tuning/run_ft.py train ...).")
    p.add_argument("--init-from", required=True, help="pre-trained checkpoint folder (final/, best/ or a Hub snapshot: config.json + model.pt|safetensors)")
    p.add_argument("--out", type=Path, default=None, help="run folder (default: fine-tuning/checkpoints/sft, Drive on Colab)")
    p.add_argument("--cache", type=Path, default=FT_DIR / "data" / "cache", help="SFT cache from `prep`")
    p.add_argument("--pretrain-cache", type=Path, default=DLM_ROOT / "data" / "tokens", help="pre-training token cache (forgetting guard)")
    p.add_argument("--tokenizer-dir", type=Path, default=DLM_ROOT / "tokenizer")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--stop-at", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--min-lr-ratio", type=float, default=0.1)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.999)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--schedule", choices=("auto", "linear", "cosine"), default="auto", help="noise schedule; auto = the base model's")
    p.add_argument("--plain-prob", type=float, default=D.DEFAULT_PLAIN_PROB, help="share of examples with the plain pre-training prompt")
    p.add_argument("--val-examples", type=int, default=5000, help="examples in the fixed SFT validation / train-eval sets")
    p.add_argument("--final-uncond-blocks", type=int, default=2048)
    p.add_argument("--gate-uncond-max", type=float, default=5.5,
                   help="step-0 gate: abort when the base model's plain-prompt validation bound (needs --pretrain-cache) is above this; 0 = off")
    p.add_argument("--lora-r", type=int, default=0, help="0 = full fine-tune; R > 0 = LoRA adapters of rank R (ablation)")
    p.add_argument("--lora-alpha", type=float, default=None, help="default 2 * r")
    p.add_argument("--lora-dropout", type=float, default=0.0)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--ckpt-every", type=int, default=500)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--print-every", type=int, default=100)
    p.add_argument("--device", default="auto")
    p.add_argument("--precision", choices=("auto", "bf16", "fp32"), default="auto")
    return p


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    from tokenizers import Tokenizer
    try:
        tok = Tokenizer.from_file(str(Path(a.tokenizer_dir) / "tokenizer.json"))
        a.train_examples = D.SftCache(a.cache, "train").manifest["splits"]["train"]["examples"]  # part of the recipe
        out = (a.out or default_out()).expanduser().resolve()
        res = run_sft(a, out, tok)
    except (SftError, D.CacheError, FileNotFoundError) as e:
        print(f"SFT FAILED: {e}", file=sys.stderr)
        return 1
    if res["final"]:
        print(f"Done. Wrote {res['final_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
