"""Phase 5: pre-train the DLM. (Phase 8, fine-tuning, will be the same loop with --init-from and --task sft; it is
still `not_built`: the instruction dataset has not been chosen.)

Decisions (grilled 2026-11-22): plain PyTorch, single device, bf16 autocast when the CUDA GPU supports it and fp32
otherwise (a Colab T4 has no bf16: use an L4 or A100; MPS/CPU runs are for smoke tests); AdamW, peak LR 2e-4, warmup
1,000 steps, cosine decay to 10% of peak, weight decay 0.1 on matrices only (ndim >= 2), beta2 0.999, gradient
clipping 1.0, batch 32 blocks, 40,000 steps. These values come from the old notebooks' proxy trials (hyper.md) on a
DIFFERENT architecture: starting points, not tuned for this model.

  * Data order: block order is a pure function of (seed, step) (per-epoch permutation from seed + epoch), so resuming
    needs no data-loader state.
  * Evaluation every `eval_every` steps on 256 evenly spaced validation blocks and 256 train-eval blocks, with a
    FIXED t per block and masks drawn from a separate generator seeded identically each time: deterministic, and it
    never consumes the training RNG. At the end: the full validation set (11,451 blocks) and full train-eval slice.
    Reported loss = per-token ELBO bound under the LINEAR schedule (comparable across runs and schedules).
  * resume.pt (every `ckpt_every` steps) holds model, optimizer, step, history and all RNG states; a resumed run is
    bit-identical to an uninterrupted one on CPU (tested). It refuses to resume if the recipe or model config differs.
  * final/ = model.pt, config.json, tokenizer/, history.json, run_info.json (+ training_curves.png if matplotlib).

Monitoring: a progress line every --print-every steps (step, epoch, loss, lr, grad norm, steps/s, ETA); an EVAL line
every --eval-every; <out>/training_curves.png (loss, learning rate, epoch, grad norm) and <out>/history.json are
refreshed at every evaluation; TensorBoard scalars go to <out>/tb (tensorboard --logdir <out>/tb); redraw anytime with
`python run.py plot --out <out> [--watch 60]`.

CLI: python run.py pretrain [--size base|tiny --steps N --out DIR --resume ...]
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import functools
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
from dlm import not_built
from dlm.download import COLAB_DRIVE, ROOT, default_out_dir
from dlm.model import DLMConfig, DiffusionTransformerLM, describe

PHASE = "Phase 5 (pre-training) / Phase 8 (fine-tuning)"
RECIPE_KEYS = ("size", "steps", "batch_size", "lr", "warmup", "min_lr_ratio", "weight_decay", "beta1", "beta2",
               "clip", "seq_len", "seed", "schedule")
EVAL_BLOCKS = 256


# ---------------------------------------------------------------- pieces (all unit-tested)
def lr_at(step: int, lr: float, warmup: int, steps: int, min_ratio: float) -> float:
    """Linear warmup to `lr`, then cosine decay to lr * min_ratio at `steps`."""
    if step < warmup:
        return lr * (step + 1) / warmup
    prog = min((step - warmup) / max(1, steps - warmup), 1.0)
    return lr * (min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * prog)))


def param_groups(model, weight_decay: float):
    """Weight decay on matrices only (ndim >= 2, incl. the tied embedding); none on norms. Each parameter once."""
    params = list(model.parameters())  # tied weights appear once
    return [{"params": [p for p in params if p.ndim >= 2], "weight_decay": weight_decay},
            {"params": [p for p in params if p.ndim < 2], "weight_decay": 0.0}]


@functools.lru_cache(maxsize=4)
def _perm(epoch: int, n: int, seed: int):
    g = torch.Generator().manual_seed(seed + epoch)
    return torch.randperm(n, generator=g).tolist()


def batch_indices(step: int, batch_size: int, n_blocks: int, seed: int) -> list[int]:
    """Block indices of training step `step`: a pure function of (step, seed); epochs are consecutive permutations."""
    out: list[int] = []
    pos = step * batch_size
    while len(out) < batch_size:
        epoch, off = divmod(pos, n_blocks)
        take = min(batch_size - len(out), n_blocks - off)
        out += _perm(epoch, n_blocks, seed)[off:off + take]
        pos += take
    return out


def make_batch(blocks, idx, device):
    arrs = [blocks.arrays(i) for i in idx]
    ids = torch.from_numpy(np.stack([a[0] for a in arrs])).to(device)
    protect = torch.from_numpy(np.stack([a[1] for a in arrs])).to(device)
    return ids, protect


def evenly(n: int, k: int) -> list[int]:
    return list(range(n)) if k >= n else [int(i * n / k) for i in range(k)]


def pick_device(name: str) -> str:
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def make_amp(device: str, precision: str):
    """Returns (context-manager factory, precision actually used)."""
    if precision == "auto":
        precision = "bf16" if device == "cuda" and torch.cuda.is_bf16_supported() else "fp32"
    if precision == "bf16":
        return (lambda: torch.autocast(device_type=device, dtype=torch.bfloat16)), "bf16"
    return contextlib.nullcontext, "fp32"


def evaluate(model, blocks, indices, cfg: DLMConfig, device, amp, schedule: str = "linear", seed: int = 12345,
             batch_size: int = 32) -> float:
    """Deterministic per-token ELBO bound over the given blocks: fixed t per block, own generator."""
    was_training = model.training
    model.eval()
    gen = torch.Generator().manual_seed(seed)
    n = len(indices)
    total_num = total_den = 0.0
    with torch.no_grad():
        for a in range(0, n, batch_size):
            idx = indices[a:a + batch_size]
            ids, protect = make_batch(blocks, idx, device)
            t = (torch.arange(a, a + len(idx), dtype=torch.float32) + 0.5) / n
            with amp():
                num, den, _ = df.loss_sums(model, ids, protect, cfg.mask_id, cfg.pad_id, t, schedule, gen)
            total_num += float(num.sum())
            total_den += float(den.sum())
    model.train(was_training)
    return total_num / max(total_den, 1.0)


# ---------------------------------------------------------------- checkpoints
def rng_state(gen: torch.Generator) -> dict:
    s = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
         "gen": gen.get_state()}
    if torch.cuda.is_available():
        s["cuda"] = torch.cuda.get_rng_state_all()
    return s


def set_rng_state(s: dict, gen: torch.Generator) -> None:
    random.setstate(s["python"])
    np.random.set_state(s["numpy"])
    torch.set_rng_state(s["torch"])
    gen.set_state(s["gen"])
    if "cuda" in s and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(s["cuda"])


def recipe_of(a, cfg: DLMConfig) -> dict:
    return {**{k: getattr(a, k) for k in RECIPE_KEYS}, "model": cfg.to_dict()}


def atomic_save(obj, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


UNIGRAM_BASELINE = 5.92  # nats/token of a model that only knows token frequencies, on the tokens the model predicts
#                          (excludes the always-given [BOS] + header; 5.87 if they are counted). Exact value: run.py eval


def moving_average(xs: list[float], k: int) -> list[float]:
    out, run = [], []
    for x in xs:
        run.append(x)
        if len(run) > k:
            run.pop(0)
        out.append(sum(run) / len(run))
    return out


def plot_curves(history: list, path: Path) -> bool:
    """Standard training trace in four stacked panels sharing the step axis: cross-entropy loss (train raw + moving
    average, train-eval, validation), learning rate, epoch, gradient norm. Dotted vertical lines mark epoch
    boundaries. Needs matplotlib (skipped with a message if it is missing)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed: skipping training_curves.png (history.json has the data)")
        return False
    tr = [h for h in history if "loss" in h]
    ev = [h for h in history if "val" in h]
    fig, axes = plt.subplots(4, 1, figsize=(9, 11), sharex=True, gridspec_kw={"height_ratios": [3, 1.3, 1.2, 1.2]})
    ax = axes[0]
    if tr:
        steps, loss = [h["step"] for h in tr], [h["loss"] for h in tr]
        ax.plot(steps, loss, color="tab:blue", alpha=0.25, label="train (per log interval)")
        ax.plot(steps, moving_average(loss, 10), color="tab:blue", label="train (moving average of 10)")
    if ev:
        ax.plot([h["step"] for h in ev], [h["val"] for h in ev], "o-", color="tab:red", label="validation")
        ax.plot([h["step"] for h in ev], [h["train_eval"] for h in ev], "s--", color="tab:green", label="train-eval")
    ax.axhline(UNIGRAM_BASELINE, color="gray", ls=":", label=f"token-frequency baseline ({UNIGRAM_BASELINE})")
    ax.set_ylabel("cross-entropy loss\n(ELBO-weighted, nats/token)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)
    if tr:
        last = tr[-1]
        v = f" | val {ev[-1]['val']:.3f}" if ev else ""
        ax.set_title(f"step {last['step']:,} | epoch {last.get('epoch', 0):.3f} | loss {last['loss']:.3f}{v} | lr {last['lr']:.2e}")
        axes[1].plot([h["step"] for h in tr], [h["lr"] for h in tr], color="tab:orange")
        axes[2].plot([h["step"] for h in tr], [h.get("epoch", 0) for h in tr], color="tab:purple")
        axes[3].plot([h["step"] for h in tr], [h["grad_norm"] for h in tr], color="tab:brown")
        e_max = int(tr[-1].get("epoch", 0))
        for e in range(1, e_max + 1):  # epoch boundaries
            first = next((h["step"] for h in tr if h.get("epoch", 0) >= e), None)
            if first is not None:
                for a_ in axes:
                    a_.axvline(first, color="gray", ls=":", alpha=0.6)
    axes[1].set_ylabel("learning rate")
    axes[2].set_ylabel("epoch")
    axes[3].set_ylabel("grad norm\n(before clipping)")
    axes[3].set_xlabel("step")
    for a_ in axes[1:]:
        a_.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return True


def write_live(out: Path, history: list) -> None:
    """Refresh history.json and training_curves.png in the output folder (called at every evaluation)."""
    tmp = out / "history.json.tmp"
    tmp.write_text(json.dumps(history, indent=1) + "\n")
    os.replace(tmp, out / "history.json")
    plot_curves(history, out / "training_curves.png")


def open_writer(out: Path):
    """TensorBoard writer (<out>/tb) if tensorboard is installed, else None."""
    try:
        from torch.utils.tensorboard import SummaryWriter
        return SummaryWriter(str(out / "tb"))
    except Exception as e:  # tensorboard missing or broken: the PNG and history.json still work
        print(f"TensorBoard logging off ({type(e).__name__}); using training_curves.png / history.json only", flush=True)
        return None


def fmt_time(sec: float) -> str:
    sec = int(max(sec, 0))
    return f"{sec // 3600}h{sec % 3600 // 60:02d}m" if sec >= 3600 else f"{sec // 60}m{sec % 60:02d}s"


def plot_main(argv: list[str] | None = None) -> int:
    """python run.py plot [--out DIR] [--watch SECONDS]: redraw training_curves.png from history.json (works while
    a run is in progress in another terminal)."""
    p = argparse.ArgumentParser(description="Redraw the training curves from a run's history.json.")
    p.add_argument("--out", type=Path, default=None, help="run folder (default: the pretrain_base folder)")
    p.add_argument("--watch", type=float, default=None, metavar="SECONDS", help="keep redrawing every SECONDS")
    args = p.parse_args(argv)
    out = (args.out or default_checkpoint_dir("base")).expanduser().resolve()
    while True:
        src = next((f for f in (out / "history.json", out / "final" / "history.json") if f.is_file()), None)
        if src is None:
            print(f"No history.json in {out} yet.", file=sys.stderr)
            if args.watch is None:
                return 1
        else:
            hist = json.loads(src.read_text())
            if plot_curves(hist, out / "training_curves.png"):
                print(f"wrote {out / 'training_curves.png'} ({len(hist)} history entries, last step "
                      f"{max((h['step'] for h in hist), default=0):,})", flush=True)
            elif args.watch is None:
                return 1
        if args.watch is None:
            return 0
        time.sleep(args.watch)


# ---------------------------------------------------------------- the loop
def run_training(a, cfg: DLMConfig, train_blocks, val_blocks, traineval_blocks, out: Path,
                 tokenizer_dir: Path | None = None) -> dict:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    device = pick_device(a.device)
    amp, precision = make_amp(device, a.precision)
    torch.manual_seed(a.seed)
    random.seed(a.seed)
    np.random.seed(a.seed)
    model = DiffusionTransformerLM(cfg).to(device)
    print(f"model: {describe(model)} | device {device} | precision {precision}", flush=True)
    opt = torch.optim.AdamW(param_groups(model, a.weight_decay), lr=a.lr, betas=(a.beta1, a.beta2))
    gen = torch.Generator().manual_seed(a.seed + 1)  # training noise only
    n_train = len(train_blocks)
    val_idx = evenly(len(val_blocks), EVAL_BLOCKS)
    tre_idx = evenly(len(traineval_blocks), EVAL_BLOCKS)
    recipe = recipe_of(a, cfg)
    history: list[dict] = []
    writer = None if a.no_tensorboard else open_writer(out)
    epoch_of = lambda s_: s_ * a.batch_size / n_train  # noqa: E731  (float epochs: 1.0 = every train block seen once)
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
        set_rng_state(ck["rng"], gen)
        print(f"resumed from step {start}", flush=True)
    elif a.resume:
        print("no resume.pt found: starting from scratch", flush=True)

    end = min(a.steps, a.stop_at) if a.stop_at else a.steps
    model.train()
    run_sum, run_n, gn = torch.zeros((), device=device), 0, torch.zeros(())
    t_start, steps_done = time.time(), 0
    for step in range(start, end):
        lr = lr_at(step, a.lr, a.warmup, a.steps, a.min_lr_ratio)
        for g in opt.param_groups:
            g["lr"] = lr
        ids, protect = make_batch(train_blocks, batch_indices(step, a.batch_size, n_train, a.seed), device)
        with amp():
            loss = df.diffusion_loss(model, ids, protect, cfg.mask_id, cfg.pad_id, a.schedule, gen)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip)
        opt.step()
        run_sum += loss.detach()
        run_n += 1
        steps_done += 1
        done = step + 1
        if done % a.log_every == 0 or done == end:
            avg = float(run_sum) / run_n
            if not math.isfinite(avg):
                raise RuntimeError(f"loss is {avg} at step {done}: stopping")
            epoch = epoch_of(done)
            history.append({"step": done, "epoch": epoch, "loss": avg, "lr": lr, "grad_norm": float(gn)})
            run_sum, run_n = torch.zeros((), device=device), 0
            if writer:
                writer.add_scalar("train/cross_entropy_loss", avg, done)
                writer.add_scalar("train/learning_rate", lr, done)
                writer.add_scalar("train/epoch", epoch, done)
                writer.add_scalar("train/grad_norm", float(gn), done)
            if done % a.print_every == 0 or done == end:
                sps = steps_done / max(time.time() - t_start, 1e-9)
                print(f"step {done:>6}/{a.steps} | epoch {epoch:.3f} | loss {avg:.4f} | lr {lr:.2e} | "
                      f"grad {float(gn):.2f} | {sps:.2f} steps/s | eta {fmt_time((a.steps - done) / sps)}", flush=True)
        if done % a.eval_every == 0 or done == a.steps:
            v = evaluate(model, val_blocks, val_idx, cfg, device, amp)
            e = evaluate(model, traineval_blocks, tre_idx, cfg, device, amp)
            entry = {"step": done, "epoch": epoch_of(done), "val": v, "train_eval": e, "gap": v - e}
            if a.schedule != "linear":
                entry["val_own"] = evaluate(model, val_blocks, val_idx, cfg, device, amp, a.schedule)
            history.append(entry)
            if writer:
                writer.add_scalar("eval/validation", v, done)
                writer.add_scalar("eval/train_eval", e, done)
                writer.add_scalar("eval/gap", v - e, done)
            write_live(out, history)  # history.json + training_curves.png are refreshed at every evaluation
            el = time.time() - t_start
            last = next((h for h in reversed(history) if "loss" in h), {"loss": float("nan")})
            print(f"EVAL step {done:>6}/{a.steps} | epoch {epoch_of(done):.3f} | train {last['loss']:.4f} | "
                  f"train-eval {e:.4f} | val {v:.4f} | gap {v - e:+.4f} | {steps_done / el:.2f} steps/s", flush=True)
        if done % a.ckpt_every == 0 and done < a.steps:
            atomic_save({"model": model.state_dict(), "optimizer": opt.state_dict(), "step": done, "history": history,
                         "rng": rng_state(gen), "recipe": recipe}, resume_path)
    elapsed = time.time() - t_start
    if writer:
        writer.close()
    write_live(out, history)

    if end < a.steps:  # stopped early on purpose (--stop-at): save a resume point, no final
        atomic_save({"model": model.state_dict(), "optimizer": opt.state_dict(), "step": end, "history": history,
                     "rng": rng_state(gen), "recipe": recipe}, resume_path)
        print(f"stopped at step {end} (--stop-at) after {elapsed:.0f}s = {steps_done / max(elapsed, 1e-9):.2f} steps/s; "
              f"resume with --resume", flush=True)
        return {"step": end, "final": False, "model": model, "history": history}

    full_val = evaluate(model, val_blocks, list(range(len(val_blocks))), cfg, device, amp)
    full_tre = evaluate(model, traineval_blocks, list(range(len(traineval_blocks))), cfg, device, amp)
    print(f"final | full validation bound {full_val:.4f} | full train-eval {full_tre:.4f} | gap {full_val - full_tre:+.4f}",
          flush=True)
    final = out / "final"
    model.save_pretrained(final)
    if tokenizer_dir and (Path(tokenizer_dir) / "tokenizer.json").is_file():
        shutil.copytree(tokenizer_dir, final / "tokenizer", dirs_exist_ok=True)
    (final / "history.json").write_text(json.dumps(history, indent=1) + "\n")
    info = {
        "recipe": recipe, "device": device, "precision": precision, "parameters": model.num_parameters(),
        "steps_per_second": round(steps_done / max(elapsed, 1e-9), 3),
        "block_slots_per_second": round(steps_done * a.batch_size * a.seq_len / max(elapsed, 1e-9), 1),
        "seconds": round(elapsed, 1), "final_val_bound": full_val, "final_train_eval_bound": full_tre,
        "torch": torch.__version__, "finished_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    }
    (final / "run_info.json").write_text(json.dumps(info, indent=2) + "\n")
    plot_curves(history, final / "training_curves.png")
    return {"step": end, "final": True, "model": model, "history": history, "info": info,
            "full_val": full_val, "full_train_eval": full_tre, "final_dir": final}


# ---------------------------------------------------------------- CLI
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Pre-train the DLM (python run.py pretrain ...).")
    p.add_argument("--task", choices=("pretrain", "sft"), default="pretrain")
    p.add_argument("--init-from", default=None, help="checkpoint folder to start from (required for sft)")
    p.add_argument("--out", type=Path, default=None, help="output folder (default: DLM/checkpoints/pretrain_<size>, Drive on Colab)")
    p.add_argument("--cache", type=Path, default=None, help="Phase 2 token cache (default: <data dir>/tokens)")
    p.add_argument("--tokenizer-dir", type=Path, default=ROOT / "tokenizer")
    p.add_argument("--resume", action="store_true", help="continue from resume.pt in --out")
    p.add_argument("--size", choices=("base", "tiny"), default="base")
    p.add_argument("--steps", type=int, default=40_000)
    p.add_argument("--stop-at", type=int, default=None, help="stop early at this step and save resume.pt (for time-limited sessions)")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--warmup", type=int, default=1000)
    p.add_argument("--min-lr-ratio", type=float, default=0.1)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.999)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--schedule", choices=("linear", "cosine"), default="linear")
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--ckpt-every", type=int, default=1000)
    p.add_argument("--log-every", type=int, default=25, help="record loss/lr/epoch/grad-norm every N steps (the curves)")
    p.add_argument("--print-every", type=int, default=100, help="print a progress line every N steps (use a multiple of --log-every)")
    p.add_argument("--no-tensorboard", action="store_true", help="do not write TensorBoard logs to <out>/tb")
    p.add_argument("--device", default="auto", help="auto | cuda | mps | cpu")
    p.add_argument("--precision", choices=("auto", "bf16", "fp32"), default="auto")
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    a = parser.parse_args(argv)
    if a.task == "sft":
        if not a.init_from:
            parser.error("--task sft needs --init-from <pre-trained checkpoint>")
        not_built("Phase 8 (fine-tuning)", "which instruction dataset to use")
    from dlm.data import CACHE_MANIFEST, TRAIN_EVAL_STORIES, StoryBlocks

    cache = (a.cache or default_out_dir() / "tokens").expanduser().resolve()
    if not (cache / CACHE_MANIFEST).is_file():
        print(f"No token cache in {cache}. Run: python run.py phase2", file=sys.stderr)
        return 1
    m = json.loads((cache / CACHE_MANIFEST).read_text())
    print("building blocks ...", flush=True)
    train = StoryBlocks(cache, "train", a.seq_len)
    val = StoryBlocks(cache, "validation", a.seq_len)
    trev = StoryBlocks(cache, "train", a.seq_len, sample_stories=TRAIN_EVAL_STORIES, seed=0)
    print(f"blocks: train {len(train):,} | validation {len(val):,} | train-eval {len(trev):,} "
          f"(seq_len {a.seq_len})", flush=True)
    base = DLMConfig.from_preset(a.size)
    cfg = DLMConfig.from_dict({**base.to_dict(), "vocab_size": m["vocab_size"], "seq_len": a.seq_len,
                               "pad_id": m["special_ids"]["[PAD]"], "mask_id": m["special_ids"]["[MASK]"]})
    out = (a.out or default_checkpoint_dir(a.size)).expanduser().resolve()
    res = run_training(a, cfg, train, val, trev, out, a.tokenizer_dir)
    if res["final"]:
        print(f"Done. Wrote {res['final_dir']}")
    return 0


def default_checkpoint_dir(size: str) -> Path:
    if COLAB_DRIVE.is_dir():
        return COLAB_DRIVE / "diffusion_lm_outputs" / "checkpoints" / f"pretrain_{size}"
    return ROOT / "checkpoints" / f"pretrain_{size}"
