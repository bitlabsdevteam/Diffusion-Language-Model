"""Phase 6: generate text by iterative unmasking.

Decisions (grilled 2026-11-22, see PRD.md Phase 6):
  * Two modes for the base model: "story" (unconditional: [BOS] + header, then generate a whole Story) and
    "continue" (the opening words you give follow the header and are fixed; the rest is generated). Free-form chat
    instructions wait for fine-tuning (Phase 8): pre-training only ever saw "Write a short story.".
  * The window is [BOS] + header + prompt (all given) + `max_new_tokens` [MASK] tokens (default 480, clamped so the
    total never exceeds the model's seq_len). No [PAD] at all: the model never attends to [PAD] in training either.
  * The Story is cut at the first <|end|> or [EOS] of the final text; whatever follows is discarded (the model never
    learned to fill the tail with [EOS]: after [EOS] it starts another Story, as in packed training blocks).
  * Semi-autoregressive by default: decoding blocks of 32 tokens filled left to right (not the 512-token training
    Block); inside a decoding block the most confident predictions are committed first; generation stops after the
    decoding block that contains an end token (later ones are never run). `block_length=0` is one global pass.
  * Commit rule: per step the k most confident still-masked positions of the current decoding block are committed
    (confidence = probability of the chosen token under the UNtempered softmax); committed tokens never change;
    [MASK] and [PAD] are never sampled. k follows the linear unmasking schedule that matches training (equal numbers
    per step); "cosine" is a flag for the Phase 7 ablation. Steps that commit nothing skip the model call.
  * steps = one token per step by default (so `max_new_tokens` model calls at most); an explicit `steps` is spread
    evenly over the blocks and clamped to the block length. temperature 0.8 by default (0 = greedy, deterministic); randomness comes from an
    explicit CPU generator seeded by --seed.
  * A collapse warning fires when >= 60% of the generated tokens are the same token.
  * --trace prints the unmasking as text rows; --gif writes an animation (Pillow).

CLI: python run.py generate --ckpt <run>/final [--prompt "..."] [--mode story|continue] [--n 3] [--seed 0] ...
"""
from __future__ import annotations

import argparse
import math
import sys
from collections import Counter
from pathlib import Path

import torch

from dlm.cleaning import clean_story
from dlm.data import HEADER
from dlm.model import load_model

PHASE = "Phase 6 (sampling)"
DEFAULT_MAX_NEW = 480
DEFAULT_BLOCK = 32
DEFAULT_TEMPERATURE = 0.8
COLLAPSE_SHARE = 0.6
END_NAMES = ("<|end|>", "[EOS]")


def specials(tok) -> dict:
    """Ids of the special tokens the sampler needs, from a `tokenizers.Tokenizer`."""
    names = ("[PAD]", "[MASK]", "[BOS]", "[EOS]", "<|end|>")
    ids = {n: tok.token_to_id(n) for n in names}
    if None in ids.values():
        raise ValueError(f"tokenizer is missing special tokens: {[n for n, i in ids.items() if i is None]}")
    return ids


def _raw(u: float, schedule: str) -> float:
    if schedule == "linear":
        return u
    if schedule == "cosine":
        return math.sin(0.5 * math.pi * u)
    raise ValueError(f"unknown schedule {schedule!r}")


def commit_plan(length: int, steps: int, schedule: str = "linear") -> list[int]:
    """How many of `length` masked tokens each of `steps` steps commits (sums to `length`)."""
    masked, plan = length, []
    for k in range(1, steps + 1):
        remaining = 0 if k == steps else min(masked, math.ceil(length * _raw((steps - k) / steps, schedule) - 1e-9))
        plan.append(masked - remaining)
        masked = remaining
    return plan


def plan_blocks(gen_len: int, block_length: int | None, steps: int | None):
    """(block bounds relative to the generated region, steps per block)."""
    block = gen_len if not block_length or block_length >= gen_len else block_length
    blocks = [(a, min(a + block, gen_len)) for a in range(0, gen_len, block)]
    if steps is None:  # default: one token per step in every block (a short last block just takes fewer steps)
        return blocks, block
    return blocks, max(1, min(block, math.ceil(max(1, steps) / len(blocks))))


@torch.no_grad()
def diffusion_generate(model, tok, header_ids: list[int], prompt_ids: list[int], max_new_tokens: int = DEFAULT_MAX_NEW,
                       steps: int | None = None, block_length: int | None = DEFAULT_BLOCK,
                       temperature: float = DEFAULT_TEMPERATURE, schedule: str = "linear",
                       generator: torch.Generator | None = None, return_trace: bool = False) -> dict:
    """Fill the response region (starts all [MASK]) and return the cut Story plus bookkeeping (see module doc)."""
    sp = specials(tok)
    cfg = model.config
    device = next(model.parameters()).device
    prefix = [sp["[BOS]"]] + list(header_ids) + list(prompt_ids)
    if len(prefix) >= cfg.seq_len:
        raise ValueError(f"prompt too long: {len(prefix)} prefix tokens, the model's window is {cfg.seq_len}")
    gen_len = min(max_new_tokens, cfg.seq_len - len(prefix))
    if gen_len < 1:
        raise ValueError("max_new_tokens must be at least 1")
    if temperature > 0 and generator is None:
        generator = torch.Generator().manual_seed(0)
    blocks, steps_per_block = plan_blocks(gen_len, block_length, steps)
    P = len(prefix)
    ids = torch.tensor(prefix + [sp["[MASK]"]] * gen_len, device=device)[None]
    end_ids = {sp[n] for n in END_NAMES}
    mask_id, pad_id = sp["[MASK]"], sp["[PAD]"]
    was_training = model.training
    model.eval()
    trace, forward_calls, blocks_run, covered = [], 0, 0, 0
    for bi, (a, b) in enumerate(blocks):
        lo, hi = P + a, P + b
        for n_commit in commit_plan(b - a, min(steps_per_block, b - a), schedule):
            if n_commit == 0:
                continue
            logits = model(ids)[0, lo:hi].float()
            forward_calls += 1
            logits[:, mask_id] = float("-inf")
            logits[:, pad_id] = float("-inf")
            probs = torch.softmax(logits, dim=-1)  # untempered: used for the confidence
            if temperature > 0:
                x0 = torch.multinomial(torch.softmax(logits / temperature, dim=-1).cpu(), 1, generator=generator)
                x0 = x0.squeeze(-1).to(device)
            else:
                x0 = logits.argmax(dim=-1)
            conf = probs.gather(-1, x0[:, None]).squeeze(-1)
            conf = torch.where(ids[0, lo:hi] == mask_id, conf, torch.full_like(conf, float("-inf")))
            top = conf.topk(n_commit).indices
            ids[0, lo + top] = x0[top]
            if return_trace:
                trace.append({"step": forward_calls, "block": bi, "positions": (lo + top).tolist(), "ids": x0[top].tolist()})
        blocks_run, covered = bi + 1, hi
        if any(t in end_ids for t in ids[0, P:hi].tolist()):  # an end token is committed: later blocks are not run
            break
    model.train(was_training)
    gen = ids[0, P:covered].tolist()
    end_index = next((i for i, t in enumerate(gen) if t in end_ids), None)
    story = gen if end_index is None else gen[:end_index]
    text = tok.decode(list(prompt_ids) + story, skip_special_tokens=True)
    warnings = []
    if len(story) >= 20:
        top_tok, n = Counter(story).most_common(1)[0]
        if n / len(story) >= COLLAPSE_SHARE:
            warnings.append(f"collapse: {n / len(story):.0%} of the generated tokens are {tok.id_to_token(top_tok)!r}")
    return {"text": text, "ids": ids[0].tolist(), "prefix_len": P, "gen_len": gen_len, "end_index": end_index,
            "truncated": end_index is None, "forward_calls": forward_calls, "steps_per_block": steps_per_block,
            "n_blocks": len(blocks), "blocks_run": blocks_run, "covered": covered, "trace": trace,
            "warnings": warnings, "mask_id": mask_id}


def generate_text(model, tok, prompt: str = "", mode: str | None = None, seed: int = 0, **kwargs) -> dict:
    """story: unconditional; continue: `prompt` are the opening words of the Story. mode=None picks by `prompt`."""
    mode = mode or ("continue" if prompt.strip() else "story")
    if mode not in ("story", "continue"):
        raise ValueError(f"mode must be 'story' or 'continue', got {mode!r}")
    if mode == "story" and prompt.strip():
        raise ValueError("mode 'story' takes no prompt; use mode 'continue' to give the opening words")
    if mode == "continue" and not prompt.strip():
        raise ValueError("mode 'continue' needs a prompt (the opening words of the Story)")
    header_ids = tok.encode(HEADER, add_special_tokens=False).ids
    prompt_ids = tok.encode(clean_story(prompt), add_special_tokens=False).ids if mode == "continue" else []
    gen = torch.Generator().manual_seed(seed)
    return diffusion_generate(model, tok, header_ids, prompt_ids, generator=gen, **kwargs)


# ---------------------------------------------------------------- seeing the process
def _states(result, tok):
    """Yield (step, tokens_text_per_position, new_flags) for the generated region: masked positions are None."""
    P, covered = result["prefix_len"], result["covered"]
    n = covered - P
    cur: list = [None] * n
    yield 0, list(cur), [False] * n
    for ev in result["trace"]:
        new = [False] * n
        for pos, tid in zip(ev["positions"], ev["ids"]):
            cur[pos - P] = tid
            new[pos - P] = True
        yield ev["step"], list(cur), new


def render_trace(result, tok, max_rows: int = 24, color: bool | None = None) -> str:
    """Text rows of the unmasking: █ = still masked; revealed tokens in green (ANSI) when `color`."""
    if not result["trace"]:
        return "(no trace: generate with return_trace=True)"
    color = sys.stdout.isatty() if color is None else color
    states = list(_states(result, tok))
    keep = sorted({round(i * (len(states) - 1) / max(max_rows - 1, 1)) for i in range(min(max_rows, len(states)))})
    rows = [f"unmasking trace: {len(states) - 1} steps, {result['blocks_run']}/{result['n_blocks']} decoding blocks run"
            + (" (green = revealed in that step)" if color else "")]
    for i in keep:
        step, cur, new = states[i]
        parts = []
        for tid, is_new in zip(cur, new):
            if tid is None:
                parts.append("█")
            else:
                t = tok.decode([tid], skip_special_tokens=False).replace("\n", "↵")
                parts.append(f"\033[32m{t}\033[0m" if (color and is_new) else t)
        masked = sum(1 for t in cur if t is None)
        rows.append(f"step {step:>4} | masked {masked:>3} | " + "".join(parts))
    return "\n".join(rows)


def write_gif(result, tok, path, fps: int = 8, max_frames: int = 60, width: int = 960, height: int = 540) -> int:
    """Animate the unmasking with Pillow (masked = grey blocks, revealed since the previous frame = green). Returns the number of frames."""
    from PIL import Image, ImageDraw, ImageFont

    states = list(_states(result, tok))
    keep = sorted({round(i * (len(states) - 1) / max(max_frames - 1, 1)) for i in range(min(max_frames, len(states)))})
    try:
        font = ImageFont.load_default(size=16)
    except TypeError:  # old Pillow
        font = ImageFont.load_default()
    pad, line_h = 20, 22
    frames, prev = [], None
    for i in keep:
        step, cur, _ = states[i]
        # highlight everything revealed since the previous DRAWN frame (frames skip steps when there are many)
        new = [t is not None and (prev is None or prev[j] is None) and i > 0 for j, t in enumerate(cur)]
        prev = cur
        img = Image.new("RGB", (width, height), (16, 18, 24))
        d = ImageDraw.Draw(img)
        masked = sum(1 for t in cur if t is None)
        d.text((pad, 8), f"step {step}   masked {masked}/{len(cur)}   (green = just revealed)", fill=(150, 160, 180), font=font)
        x, y = pad, 40
        cell = d.textlength("n", font=font)
        for tid, is_new in zip(cur, new):
            if tid is None:  # a filled block (the default font has no U+2588 glyph)
                if x + cell > width - pad:
                    x, y = pad, y + line_h
                if y < height - line_h:
                    d.rectangle([x, y + 3, x + cell, y + line_h - 6], fill=(60, 64, 78))
                x += cell + 2
                continue
            text = tok.decode([tid], skip_special_tokens=False)
            fill = (80, 220, 120) if is_new else (225, 228, 235)
            for k, piece in enumerate(text.split("\n")):
                if k > 0:
                    x, y = pad, y + line_h
                w = d.textlength(piece, font=font)
                if x + w > width - pad:
                    x, y = pad, y + line_h
                if y < height - line_h:
                    d.text((x, y), piece, fill=fill, font=font)
                x += w
        frames.append(img)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=int(1000 / fps), loop=0)
    return len(frames)


# ---------------------------------------------------------------- CLI
def load_checkpoint(ckpt, device: str = "cpu"):
    """(model, tokenizer) from a run's final/ folder (config.json, model.pt, tokenizer/tokenizer.json)."""
    from tokenizers import Tokenizer

    ckpt = Path(ckpt)
    tok_file = ckpt / "tokenizer" / "tokenizer.json"
    if not (ckpt / "model.pt").is_file() or not tok_file.is_file():
        raise FileNotFoundError(f"{ckpt} needs model.pt, config.json and tokenizer/tokenizer.json (a run's final/ folder)")
    return load_model(ckpt, device), Tokenizer.from_file(str(tok_file))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate Stories from a trained DLM (python run.py generate ...).")
    p.add_argument("--ckpt", type=Path, required=True, help="a run's final/ folder")
    p.add_argument("--prompt", default="", help="opening words of the Story (mode continue)")
    p.add_argument("--mode", choices=("story", "continue"), default=None, help="default: continue if --prompt is given")
    p.add_argument("--n", type=int, default=1, help="how many samples")
    p.add_argument("--seed", type=int, default=0, help="sample i uses seed + i")
    p.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW)
    p.add_argument("--steps", type=int, default=None, help="total steps (default: one per new token)")
    p.add_argument("--block-length", type=int, default=DEFAULT_BLOCK, help="size of a decoding block (semi-autoregressive, filled left to right); 0 = one global pass")
    p.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE, help="0 = greedy (deterministic)")
    p.add_argument("--schedule", choices=("linear", "cosine"), default="linear")
    p.add_argument("--trace", action="store_true", help="print the unmasking as text rows")
    p.add_argument("--gif", type=Path, default=None, help="write an unmasking animation to this .gif (needs Pillow)")
    p.add_argument("--device", default="auto", help="auto | cuda | mps | cpu")
    return p


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    from dlm.train import pick_device

    try:
        model, tok = load_checkpoint(a.ckpt, pick_device(a.device))
    except FileNotFoundError as e:
        print(f"generate: {e}", file=sys.stderr)
        return 1
    status = 0
    for i in range(a.n):
        try:
            r = generate_text(model, tok, a.prompt, a.mode, seed=a.seed + i, max_new_tokens=a.max_new_tokens,
                              steps=a.steps, block_length=a.block_length, temperature=a.temperature,
                              schedule=a.schedule, return_trace=bool(a.trace or a.gif))
        except ValueError as e:
            print(f"generate: {e}", file=sys.stderr)
            return 2
        print(f"--- sample {i + 1}/{a.n} (seed {a.seed + i}; {r['forward_calls']} model calls, "
              f"{r['blocks_run']}/{r['n_blocks']} decoding blocks{'; Story not finished in the window' if r['truncated'] else ''}) ---")
        print(r["text"])
        for w in r["warnings"]:
            print(f"warning: {w}", file=sys.stderr)
        if a.trace:
            print(render_trace(r, tok))
        if a.gif:
            path = a.gif if a.n == 1 else a.gif.with_name(f"{a.gif.stem}_{i + 1}{a.gif.suffix}")
            try:
                print(f"wrote {path} ({write_gif(r, tok, path)} frames)")
            except ImportError:
                print("generate: Pillow is not installed; skipping --gif", file=sys.stderr)
                status = 1
    return status
