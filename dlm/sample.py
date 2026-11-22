"""Phase 6 (NOT BUILT): generate text by iterative unmasking.

Owns:       diffusion_generate (the sampler) and generate_text (prompt string in, text out); shared by
            evaluation, fine-tuning checks and the AWS serving code (Phase 10).
Consumes:   a trained model, the tokenizer, a prompt.
Produces:   generated text (and, optionally, a trace of which positions were revealed at each step).
Open decisions to grill before building:
  - sampler: low-confidence remasking schedule matching the training schedule, steps, block (semi-autoregressive)
    length, temperature / greedy, how end-of-story is handled (boundary tokens committed last?);
  - LLaDA2.1-style token editing (Phase 11, optional).
Invariants the notebooks established and tests must keep: no [MASK] left at the end, committed tokens never
change, greedy decoding is deterministic, [MASK]/[PAD] are never sampled.
CLI: python run.py generate
"""
from __future__ import annotations

from dlm import not_built

PHASE = "Phase 6 (sampling)"
DECISION = "remasking schedule, steps, block length, temperature, end-of-story handling"


def diffusion_generate(model, tokenizer, prompt_ids, max_new_tokens: int, steps: int | None = None,
                       block_length: int | None = None, temperature: float = 0.0):
    """Fill the response region (starts all [MASK]); returns the final token ids."""
    not_built(PHASE, DECISION)


def generate_text(model, tokenizer, prompt: str, **kwargs) -> str:
    not_built(PHASE, DECISION)


def main(argv: list[str] | None = None) -> int:
    not_built(PHASE, DECISION)
