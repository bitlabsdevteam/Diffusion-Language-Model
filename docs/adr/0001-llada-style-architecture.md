---
status: accepted
---
# The model is a LLaDA-style bidirectional Transformer, not the notebooks' nn.TransformerEncoder

The model uses RMSNorm, rotary positions (RoPE), SwiGLU, bidirectional attention, tied input/output embeddings and no time embedding, at about 40M parameters. We chose it over the prototype `nn.TransformerEncoder` (learned positions, optional time embedding) because serving support is per architecture: only a LLaDA-compatible model can later be exported to LLaDA's Hugging Face format and served with SGLang or dInfer on GPU, or converted to GGUF for llama.cpp, whereas the prototype can only be served by our own PyTorch server. Changing the architecture later would invalidate every checkpoint and the loss comparisons built on it.

## Considered options

- **Keep the notebooks' `nn.TransformerEncoder`.** Least work and matches earlier runs, but locks serving to a custom PyTorch server (AWS plan Phase 10a only).
- **Time-conditioned model.** Rejected: `USE_TIME_EMB=True` gave the same loss as `False` in the proxy trials (`hyper.md`), and masked diffusion models can infer the noise level from the number of `[MASK]` tokens.

## Consequences

Results from the earlier notebooks (including `hyper.md`) were measured on a different architecture, so hyperparameters carry over only as starting points. The serving research that motivates this is in `CLAUDE.md` ("Deployment plan"), dated 2026-11-21 and unverified on real hardware.
