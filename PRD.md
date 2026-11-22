# PRD: Diffusion Language Model (DLM) from Scratch

Status date: 2026-11-21. Main project folder: `/Users/davidbong/Documents/DiffusionLanguageModel/DLM/`.
Prototype code (notebooks, scripts, research notes) still lives in the parent folder `/Users/davidbong/Documents/DiffusionLanguageModel/` and is referenced below by file name.

## 1. What this project is

A **discrete (masked) diffusion language model** built from scratch: we pre-train our own tokenizer and weights, deploy the model for inference on AWS, and fine-tune it.

### How a DLM differs from a normal (autoregressive) LLM

| | Autoregressive LLM (GPT-style) | Masked diffusion LM (LLaDA-style) |
|---|---|---|
| Attention | causal (left to right) | bidirectional (no causal mask) |
| Training | predict next token | randomly mask tokens, predict the masked ones |
| Noise | none | `[MASK]` token; ratio `t` drawn at random per sequence |
| Generation | one token per forward pass, in order | start fully masked, then fill in tokens over several denoising steps, in any order |
| Loss | cross-entropy on next token | cross-entropy on masked positions only, weighted `1/t` (an ELBO bound on likelihood) |

**Forward process (training).** Take a clean sequence, draw `t` in (0,1], replace each token by `[MASK]` with probability `t` (the *noise schedule* decides how mask rate depends on time). **Reverse process (inference).** Begin with the response all `[MASK]`; at each step the model predicts every masked position, we commit the most confident predictions and leave the rest masked, until nothing is masked.

### Reference papers (`/Users/davidbong/Documents/DiffusionLanguageModel/PDF/`)

| File | Paper | Used for |
|---|---|---|
| `LLDM_DiffusionModel.pdf` | LLaDA, *Large Language Diffusion Models* (arXiv 2502.09992) | Pre-training + SFT recipe, masked-diffusion objective, low-confidence remasking sampler |
| `LlaDA2_1.pdf` | LLaDA2.1, *Speeding Up Text Diffusion via Token Editing* (arXiv 2602.08676) | Block diffusion, Mask-to-Token + Token-to-Token editing, threshold decoding, RL for dLLMs |
| `Dream7BDiffusionModel.pdf` | Dream 7B (arXiv 2508.15487) | AR-initialised diffusion LM, token-level noise rescheduling (alternative to from-scratch) |
| `FairSteer- Inference Time Debiasing ... .pdf` | FairSteer | Inference-time debiasing; only for the optional Phase 11 |

## 2. Goals and non-goals

**Goals**
1. **Pre-train a DLM from scratch** (own tokenizer, no pretrained starting checkpoint).
2. **Deploy for inference on AWS** with a suitable inference framework (Section 5).
3. **Fine-tune the pre-trained DLM** (LLaDA-style SFT: prompt never masked, loss on the response only).

**Non-goals (for now)**
- Matching 8B-scale LLaDA/Dream quality. Early phases use TinyStories at ~45M parameters to validate the pipeline.
- Starting from an autoregressive checkpoint (Dream's approach); it is the alternative path, not this one.

## 3. Phases

Each phase has an objective, deliverables, and acceptance criteria. "Status" is what exists today; **none of the notebooks has been completed on Colab/real TinyStories end to end**, and results marked "CPU/synthetic" were checked only on a synthetic corpus.

### Phase 0: Environment and dataset download (TinyStories from Hugging Face)

- **Objective:** a reproducible environment and the training corpus on disk.
- **Work:**
  - Colab GPU runtime (`Runtime -> GPU`), or local MPS/CPU for smoke tests. Install `datasets`, `tokenizers`, `accelerate`, `transformers`, `einops`, `imageio`, `pillow` (do not reinstall numpy/pillow on Colab: it breaks the kernel).
  - Download **`roneneldan/TinyStories`** with `datasets.load_dataset` (train and validation splits; stream or cache). Mount Google Drive so data and checkpoints survive a Colab disconnect.
  - Set `RUN_MODE` (`quick` smoke test vs `budget_100`) and a fixed `SEED`.
- **Deliverables:** cached dataset, config cell, Drive-backed output folder.
- **Acceptance:** `load_dataset` returns train/validation splits; story count and a few sample stories printed; re-running does not re-download.
- **Status: built and run (2026-11-21).** `DLM/dlm/download.py` via `python run.py phase0` (+ `tests/test_download.py`, 26 offline checks) downloaded the full dataset at revision `f54c09fd` to `DLM/data/`: train 2,119,719 Stories (974 MB Parquet), validation 21,990; `manifest.json` records revision, counts and averages (about 175 words per Story). A second run does nothing. Finding: the train split has **230 empty Stories** (kept as downloaded, recorded in the manifest); Phase 2 must drop them.

### Phase 1: Tokenizer

- **Objective:** a tokenizer trained from scratch on the corpus, with the special tokens a DLM needs.
- **Decisions (grilled 2026-11-21):**
  - Script `DLM/dlm/tokenizer.py` (`python run.py phase1`) reading the Phase 0 Parquet files; shared `DLM/dlm/cleaning.py` (`clean_story`) so Phase 1 and Phase 2 clean text identically.
  - **Vocabulary 8,192** (not the notebooks' 26,000: that would be about 30% of a 45M model in embeddings; 8k matches the `hyper.md` proxy trials). `--vocab-size` changes it; fix it before Phase 5 because it sets the loss metric.
  - **Mojibake repair:** about 6% of Stories contain UTF-8 punctuation decoded as cp1252 (`Donâ€™t`). `clean_story` repairs the known sequences, folds curly quotes to ASCII and strips whitespace.
  - Trained on a seeded random sample of 500,000 non-empty **train** Stories, never validation.
  - Byte-level BPE with the full 256-byte alphabet (`[UNK]` cannot occur), **no normalizer** (lossless; the notebooks' NFKC is dropped) and **no automatic `[BOS]`/`[EOS]`** (Phase 2 adds them).
  - Fixed special-token IDs: `[PAD]`=0 `[UNK]`=1 `[BOS]`=2 `[EOS]`=3 `[MASK]`=4 `<|user|>`=5 `<|assistant|>`=6 `<|system|>`=7 (reserved for SFT) `<|end|>`=8. Pre-training template `<|user|>\nWrite a short story.\n<|assistant|>\n{story}\n<|end|>\n`.
- **Deliverables:** `DLM/tokenizer/` (`tokenizer.json`, `tokenizer_config.json`, `tokenizer_manifest.json`), committed to git (about 0.5 MB).
- **Acceptance (all checked by the script before it keeps the result):** decode(encode(x)) equals x for every cleaned validation Story; zero `[UNK]`; no `â€` left; special IDs and vocabulary size exact; no reserved token in plain text; no automatic `[BOS]`/`[EOS]`; a second run does nothing; a failed run writes no manifest.
- **Status: built and run (2026-11-21).** `dlm/tokenizer.py` + `dlm/cleaning.py` + `tests/test_tokenizer.py` (52 offline checks). Real run: 8,192-token vocabulary trained on 500,000 Stories (30,509 repaired), verified on all 21,990 validation Stories. Formatted Stories average **224.8 tokens** (median 198, p90 333, p99 635), 4.1 characters per token, and **81.7% fit in 256 tokens** (before Phase 2 adds `[BOS]`/`[EOS]`, so slightly less in practice), so a block of 256 holds a whole Story only about 4 times in 5 (input for Phase 2).

### Phase 2: Data pipeline

- **Objective:** fixed-length training blocks the diffusion objective can consume.
- **Decisions (grilled 2026-11-21):**
  - Code in `DLM/dlm/data.py` (`python run.py phase2`). Empty Stories dropped; every Story goes through `dlm.cleaning.clean_story` and the chat template, is tokenized, and gets `[BOS]`/`[EOS]` added explicitly.
  - **Block length 512** (`--seq-len`), not 256: 97.1% of Stories fit whole (256 fits only 81.3%).
  - **Packing:** each block starts at a Story's `[BOS]` and is filled greedily with the following whole Stories; the rest is `[PAD]`; no Story is split, no overlap.
  - **Stories longer than a block are dropped**, from train and validation alike.
  - **Protected (never masked) tokens:** `[BOS]` plus the 10-token header of *every* Story in a block (11 per Story; `[BOS]` was added to the protected set because generation always starts with it). `<|end|>` and `[EOS]` stay maskable so the model learns where Stories end. `[PAD]` is excluded from masking by the Phase 4 corruption.
  - No random-length crops. Validation = the dataset's own 21,990-Story validation Split; "train-eval" = a fixed seeded 5,000-Story slice of train (for the train/validation gap).
  - Tokenized Stories cached as `uint16` arrays in `data/tokens/` (independent of block length, so `--seq-len` never re-tokenizes), manifest written last, idempotent, stale when the Phase 0 revision, tokenizer file, cleaning version or format version changes.
- **Deliverables:** `data/tokens/`, `StoryBlocks` dataset: items `{"input_ids": LongTensor[512], "protect": BoolTensor[512]}`.
- **Status: built and run (2026-11-21).** `tests/test_data.py` has 1,590 checks (cache contents, Story round trip, mojibake repaired, greedy packing, whole Stories only, protect mask, no overlap, determinism, staleness, DataLoader). Real run (about 4 minutes): train 2,119,489 Stories / 495.2M tokens (230 empty dropped, mean 233.6 tokens/Story); validation 21,990 Stories / 5.0M tokens; cache 970 MB. At 512: train 1,123,259 blocks (2,046,597 Stories kept, **72,892 = 3.4% too long, dropped**), validation 11,451 blocks (635 dropped), train-eval 2,774 blocks; 1.82 Stories per block; protected tokens are 3.9% of each block.
- **Observation to decide on:** whole-Story greedy packing leaves **22% of every block as `[PAD]`** (fill 78.1%), so about a fifth of the compute and memory is spent on padding. A look-ahead / best-fit packing that fills the gap with a later, shorter Story keeps Stories whole and would raise fill to roughly 95%+, at the cost of Story order; not built, not decided.
- **Still open (Phase 8):** `build_sft_example` and the instruction dataset.

### Phase 3: Model architecture

- **Objective:** a bidirectional Transformer that predicts the original token at every masked position.
- **Work (current prototype):** `DiffusionTransformerLM`: `nn.TransformerEncoder`, pre-norm, GELU, learned position embeddings, optional timestep embedding (`USE_TIME_EMB`, off by default), tied input/output embeddings, **no causal mask**. Base size 512 x 10 layers x 8 heads, FFN 2048 (~45M parameters).
- **Open decision (blocks Phase 9/10):** the prototype is custom, so no serving framework loads it. For a standard serving path the full-scale model should use LLaDA-style blocks (RMSNorm, RoPE, SwiGLU, no time embedding). See Section 5.
- **Acceptance:** forward pass returns logits of shape `[batch, seq, vocab]`; parameter count matches config; config round-trips through `config.json`.
- **Status:** prototype implemented; LLaDA-compatible architecture **not started**.

### Phase 4: Diffusion objective (forward noising and loss)

- **Objective:** a correct masked-diffusion training signal.
- **Work:** per sequence draw `t`, mask non-protected tokens with probability from the noise schedule (`linear`, or MaskGIT/MD4 cosine `sin(pi*u/2)`; the cosine notebook also uses stratified `t`). Cross-entropy on masked positions only, weighted by the ELBO factor `1/t` (`LOSS_WEIGHTING="elbo"`), so validation loss is a per-token likelihood bound. `[BOS]`/`[EOS]` are maskable so the model learns where stories end.
- **Acceptance:** Monte Carlo check that a uniform-random model gives loss ~ `ln(V)`; `E[w*p] = 1`; the prompt header is never masked. (`python test_cosine_notebook.py`, 70 checks, passes on CPU/synthetic.)
- **Open decision (to grill at the start of Phase 4):** noise schedule. Proposed default: linear for training and sampling, with cosine tested as a Phase 7 ablation (2x2 of training x sampling schedule), and whether training and sampling schedules should match.
- **Status:** implemented and unit-tested; schedule choice (linear vs cosine) is still an open ablation. Cosine is not expected to lower the val bound, only to help sample quality per step.

### Phase 5: Pre-training

- **Objective:** a trained base model whose validation loss is clearly below the token-frequency plateau.
- **Work:** HF `accelerate` (bf16 where supported), AdamW with cosine-warmup LR, peak LR 2e-4, weight decay 0.1, batch 32, `TRAIN_STEPS=40,000` in `budget_100`. Resumable checkpoints (`resume.pt` every `CKPT_EVERY` steps, final to `checkpoints/final/{model.pt,config.json,tokenizer/,history.json,run_info.json}`), loss logging (train, train-eval, val, gap), training-curve plot. Staged hyperparameter search exists but is off by default.
- **Known risk:** the loss sits on the ~5.7-5.8 "token frequency" plateau for the first thousands of steps. In 12 proxy trials (`hyper.md`) LR, `token_mean`, and the time embedding did not matter; turning random-length crops off helped (5.33 vs 5.62 at 1,000 steps). The full 40,000-step run has not been done.
- **Acceptance:** val loss falls well below ~5.7 and keeps improving; a reloaded checkpoint reproduces eval loss; a resumed run matches an uninterrupted one.
- **Status:** training code complete and tested on CPU/synthetic; **real TinyStories pre-training not completed**.

### Phase 6: Sampling and inference (notebook level)

- **Objective:** generate text by iterative unmasking.
- **Work:** `diffusion_generate`: prompt fixed, response starts all `[MASK]`; each step predicts only masked positions, commits the most confident ones (low-confidence remasking, LLaDA-style), never edits committed tokens, forbids `[MASK]`/`[PAD]` in logits. Options: `temperature<=0` = argmax, any number of steps `N`, semi-autoregressive `block_length` (default 32), `boundary_conf` for end tokens. `generate_text(prompt, mode="continue"|"chat")`, unmasking trace/animation (`show_unmasking`, `animate_unmasking`, GIF).
- **Acceptance:** no `[MASK]` left at the end; greedy decoding is deterministic; committed tokens never change; steps 1/8/16/64 all run; warning when >=60% of generated tokens are one token (collapse detector).
- **Status:** implemented and tested on CPU/synthetic; quality on a trained TinyStories model is unverified.

### Phase 7: Evaluation

- **Objective:** know whether the model is good, and whether a change helped.
- **Work:**
  - Likelihood: validation ELBO bound (reported under the linear schedule so runs are comparable).
  - Sample quality: coherence of generated stories at several step counts (e.g. 8/16/32/64), with and without block decoding; 2x2 ablation of training x sampling schedule (linear/cosine).
  - Throughput: tokens/sec and latency vs number of steps.
  - For the scaled model: standard benchmarks following the LLaDA paper.
- **Acceptance:** a written results table per run (config, val bound, sample examples, speed).
- **Status:** val-bound reporting implemented; ablations and sample-quality study **not run**.

### Phase 8: Fine-tuning (SFT)

- **Objective:** turn the pre-trained base model into an instruction-following model.
- **Work:** LLaDA-style SFT: build prompt/response pairs in the chat template; **never mask the prompt**, mask only response tokens, compute loss on the response only, pad short responses with `[EOS]` so the model learns when to stop. Lower LR than pre-training, small number of epochs, held-out instruction set for evaluation. TinyStories only teaches "Write a short story.", so SFT needs a broader instruction dataset (to be chosen; Hugging Face instruction datasets are candidates). Optional later step: RL / preference tuning as in LLaDA2.1.
- **Acceptance:** SFT model follows held-out instructions better than the base model on a fixed prompt set; prompt tokens are provably never masked; response length behaviour is sane.
- **Status:** the prompt-protection mechanism exists in the pre-training code; the SFT dataset, training script and evaluation are **not started**.

### Phase 9: Scale-up pre-training (beyond TinyStories)

- **Objective:** a model that is useful beyond story text.
- **Work:** choose the final architecture first (Phase 3 decision), a general text corpus and a larger tokenizer, longer context, multi-GPU training (AWS or other), and a scaling plan. Compute budget and data are to be decided.
- **Acceptance:** scaled model trains stably; val bound and benchmarks improve with scale.
- **Status:** not started; depends on the Phase 3 decision and the compute budget.

### Phase 10: Deployment on AWS

- **Objective:** an inference endpoint for the trained (and fine-tuned) model on AWS. Framework research is in Section 5.
- **Work:**
  - **Phase 10a (works with the current model):** custom PyTorch inference server (FastAPI wrapping `load_trained_model` + `diffusion_generate`), packaged in Docker, run on an EC2 GPU instance or a SageMaker bring-your-own-container endpoint; weights in S3; request schema with prompt, `max_new_tokens`, steps, temperature, block length; health check, timeouts, logging, load test.
  - **Phase 10b (optional, standard engine):** if the architecture is LLaDA-compatible, export in LLaDA's Hugging Face format and serve with SGLang's dLLM mode or dInfer on GPU, or llama.cpp GGUF for CPU/edge.
- **Acceptance:** endpoint returns generations matching local output for the same seed/greedy; documented latency and cost per request at a target concurrency; infra described as code (CloudFormation/Terraform/CDK).
- **Status:** not started; no AWS resources exist.

### Phase 11 (optional): Improvements

- Token editing / threshold decoding (LLaDA2.1 Speedy/Quality modes), block diffusion at scale, RL for dLLMs.
- FairSteer-style inference-time debiasing on pretrained DLMs (`FAIRSTEER_DLM_PLAN.md`, `fairsteer_dlm.py`, `FairSteer_DLM_LLaDA_Dream.ipynb`: prototype for LLaDA-8B and Dream-7B; real-model wiring untested).

## 3a. Code layout

One entry point, shared modules, and a thin Colab runner, so pre-training (Phase 5) and fine-tuning (Phase 8) reuse the same model, loss and sampler instead of copying notebooks:

```
DLM/
  run.py              python run.py phase0 | phase1 | pretrain | sft | generate | test
  run_colab.ipynb     thin Colab runner (mount Drive, clone, pip install, call run.py)
  dlm/                download.py, cleaning.py, tokenizer.py   (built, Phases 0-1)
                      data.py, model.py, diffusion.py, train.py, sample.py   (data.py built, Phase 2; model, diffusion, train, sample are interface skeletons: Phases 3-6 and 8)
  tests/              offline tests (1,687 checks); `python run.py test`
```

The five skeleton modules each state what they own, consume and produce, and the open decisions to grill before building them; until then `python run.py pretrain|sft|generate|phase2` exits 2 and names the phase. Fine-tuning is `train.py` with `--init-from <pre-trained checkpoint> --task sft`: same loop, prompt never masked, loss on the response only.

## 4. Phase dependency summary

```
0 Dataset -> 1 Tokenizer -> 2 Data -> 3 Model -> 4 Objective -> 5 Pre-train -> 6 Sampling -> 7 Eval
                                          |                                        |
                                          +--(architecture decision)--> 9 Scale-up +--> 8 SFT --> 10 AWS deploy --> 11 Extras
```

Phases 0-7 are the TinyStories proof of concept. Phase 3's architecture decision gates Phase 9 and the Phase 10b serving options.

## 5. Inference framework for AWS (researched 2026-11-21; not run or deployed)

| Option | Fit | Notes |
|---|---|---|
| **vLLM** | No for the custom model | No native diffusion decoding in mainline as far as found; only used as a backend inside dInfer for LLaDA / LLaDA-MoE |
| **SGLang (dLLM mode)** | Good if LLaDA2-compatible | Official block-diffusion support (`--dllm-algorithm`, e.g. `LowConfidence`, `JointThreshold`; flag names vary by version); documented models are the LLaDA2.0 family |
| **dInfer (inclusionAI)** | Good if LLaDA-compatible | Diffusion-specific engine; vLLM backend for LLaDA / LLaDA-MoE, SGLang backend for LLaDA2 |
| **GGUF / llama.cpp** | CPU/edge only | `llama-diffusion-cli` runs Dream and LLaDA 1.0; `llama-server` has no diffusion-decoding mode as far as found; open PRs (#17454, #24423) unverified |
| **Custom PyTorch server** | Yes, now | Works with any architecture; lowest throughput but no porting work |

**Recommendation:** start with the custom PyTorch server (Phase 10a). If a standard engine is wanted, make the full-scale model LLaDA-compatible (RMSNorm, RoPE, SwiGLU, no time embedding, optionally block diffusion) before the real pre-training run, then use SGLang or dInfer on GPU (Phase 10b). Decide before Phase 9, because it determines the architecture.

## 6. Risks and open questions

- **Loss plateau:** the model may stall at the token-frequency plateau (~5.7-5.8); mitigations in `hyper.md` (crops off, larger embedding init not yet implemented).
- **Unverified on real data:** everything is tested on CPU/synthetic corpora, not on Colab with the real TinyStories run.
- **Serving compatibility:** custom architecture cannot use vLLM/SGLang/GGUF as is (Section 5).
- **Instruction data for SFT** and the **corpus/compute budget for scale-up** are undecided.
- **Noise schedule:** linear vs cosine still an open ablation; the two "cosines" in the literature are mirror images and do not transfer.
- **Time-sensitive facts:** framework support (vLLM, SGLang, llama.cpp) changes quickly; re-check before Phase 10.

## 7. Source files (parent folder)

- Main notebook: `Final_Diffusion_LLM_from_scratch_TinyStories_Inference_Fixed_Cosine.ipynb` (and `_v2`); older clones kept for comparison.
- `hyper.md`, `hyper_trials.jsonl`: hyperparameter study. `COSINE_SCHEDULE_RESEARCH.md`: schedule math. `DLM_INFERENCE_REVIEW.md`: review of the inference code.
- `test_cosine_notebook.py`: 70 logic checks on CPU. `fairsteer_dlm.py`, `test_fairsteer_dlm.py`, `build_fairsteer_notebook.py`: FairSteer work.
- `CLAUDE.md` and `GLOSSARY.md` (now in `DLM/`): working notes for Claude Code and the project's domain terms.
