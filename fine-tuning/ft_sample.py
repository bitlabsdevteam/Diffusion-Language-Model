"""Instruction-conditioned generation with the fine-tuned model.

The pre-training sampler (dlm.sample.diffusion_generate) already takes an arbitrary protected header, so this is a thin
wrapper: header = the chat prompt built from the instruction, nothing else is fixed, and the response is generated
block by block (confidence remasking), cut at the first <|end|>/[EOS].
"""
from __future__ import annotations

import _path  # noqa: F401

import json
from pathlib import Path

import torch

import ft_data as D
from dlm.sample import diffusion_generate


def make_instruction(words=None, features=None, summary=None, sentence=None) -> str:
    """The same text ft_data.build_instruction produces, with every given field included in canonical order."""
    fields = {"w": list(words or []), "f": list(features or []), "s": summary, "r": sentence}
    texts = D.field_texts(fields)
    return "\n".join([D.PLAIN_INSTRUCTION] + [f"{n}: {t}" for n, t in texts])


def generate_story(model, tok, instruction: str = D.PLAIN_INSTRUCTION, seed: int = 0, max_new_tokens: int = 480, **kwargs) -> dict:
    """Returns dlm.sample.diffusion_generate's dict: "text" (the story), "ids", "end_index", "warnings", ...
    kwargs: steps, block_length, temperature, schedule, return_trace."""
    header_ids = D.encode_prompt(tok, instruction)
    gen = torch.Generator().manual_seed(seed)
    return diffusion_generate(model, tok, header_ids, [], max_new_tokens=max_new_tokens, generator=gen, **kwargs)


def story_token_ids(res: dict) -> list[int]:
    """Generated story tokens (up to the first end token when finished, else everything generated)."""
    P = res["prefix_len"]
    stop = res["end_index"] if res["end_index"] is not None else res["covered"] - P
    return res["ids"][P:P + stop]


def infer_schedule(ckpt_dir) -> str:
    """The noise schedule a checkpoint was trained with (recipe in its run_info.json, here or one folder up), else linear.
    Sampling should use the training schedule (the Hub generation_config.json does the same)."""
    for name in ("run_info.json", "../run_info.json"):
        f = Path(ckpt_dir) / name
        if f.is_file():
            try:
                sch = json.loads(f.read_text())["recipe"].get("schedule")
            except (ValueError, KeyError):
                continue
            if sch in ("linear", "cosine"):
                return sch
    return "linear"
