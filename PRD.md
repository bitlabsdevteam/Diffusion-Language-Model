# PRD: Diffusion Language Model (DLM) from Scratch

Status date: 2026-11-22. Main project folder: `DLM/` (the git repo root; remote `bitlabsdevteam/Diffusion-Language-Model`).
Prototype code (notebooks, scripts, research notes) still lives in the parent folder `../` (outside the repo) and is referenced below by file name.

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

### Reference papers (`../PDF/`, outside the repo)

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
- **Observation to decide on:** whole-Story greedy packing leaves **22% of every block as `[PAD]`** (fill 78.1%), so about a fifth of the compute and memory is spent on padding. Look-ahead packing (fill the gap with a later, shorter Story from the next W Stories; Stories stay whole) was simulated on the first 400,000 eligible train Stories: greedy (current) 78.1% fill; window 8 -> 82.8% (5.7% fewer blocks); window 32 -> 85.9% (9.1% fewer); window 128 -> 88.9% (12.2% fewer). So the earlier estimate of "roughly 95%+" was wrong: the realistic gain is about 11 points of fill, i.e. about 12% less padding compute, and it reorders Stories within a window of W. Not built, not decided.
- **Still open (Phase 8):** `build_sft_example` and the instruction dataset.

### Phase 3: Model architecture

- **Decision (grilled 2026-11-22, ADR `docs/adr/0001-llada-style-architecture.md`):** a LLaDA-style bidirectional Transformer in `dlm/model.py`: RMSNorm, RoPE, SwiGLU, no biases, tied input/output embeddings, **no time embedding**, init normal(0, 0.02), `F.scaled_dot_product_attention` with `is_causal=False`. This keeps the SGLang / dInfer / GGUF serving options open (Phase 10b). `[PAD]` is masked out as an attention key (consistent with never being predicted); what fills the window after a generated `[EOS]` is a Phase 6 question.
- **Size:** width 512, 10 layers, 8 heads, SwiGLU hidden 1376 (about 2/3 of 2048): **35.83M parameters** with the 8,192-token vocabulary (the grill said "about 40M"; this is the exact figure). A "tiny" preset (0.62M) exists for tests and smoke runs.
- **Acceptance:** logits `[B, L, vocab]`; parameter count equals the analytic count; tied weights counted once; attention is bidirectional; extra `[PAD]` does not change the logits at real positions; RoPE preserves norms and scores depend only on relative position; save/load is exact; all parameters receive gradients.
- **Status: built (2026-11-22).** `tests/test_model.py`, 20 checks. Not yet exported to LLaDA's Hugging Face format (Phase 10b).

### Phase 4: Diffusion objective (forward noising and loss)

- **Decisions (grilled 2026-11-22):** `dlm/diffusion.py`. Continuous time `t` in (0, 1], linear mask rate `p(t) = (1 - 1e-3) t + 1e-3` (the bound keeps `1/p` finite, as in LLaDA's code), stratified `t`, cosine `sin(pi t / 2)` behind `--schedule` for the Phase 7 ablation, no time input. ELBO weight `p'(t)/p(t)`, so `E[w p] = 1 - 1e-3`. Protected tokens and `[PAD]` are never masked. Loss per block = sum of weighted cross-entropy over masked tokens / number of maskable tokens in the block; training loss = mean over blocks. Validation is reported under the linear schedule as a per-token bound (token-weighted over the evaluation set).
- **Acceptance:** `E[w p]` matches `1 - EPS` by numerical integration for both schedules; the stratified `t` has exactly one draw per stratum and a uniform marginal; protected and `[PAD]` tokens are never masked; the empirical mask rate matches `p(t)`; **a uniform-logits model scores (1 - EPS) * ln V over many batches for both schedules (tested with V = 64, not 8,192)**; a perfect predictor scores about 0; a fully protected block gives 0 and no NaN; the loss uses only its own generator (the global RNG is untouched).
- **Status: built (2026-11-22).** `tests/test_diffusion.py`, 31 checks. The cosine-schedule ablation itself (Phase 7) is not run.

### Phase 5: Pre-training

- **Decisions (grilled 2026-11-22):** `dlm/train.py` (`python run.py pretrain`), plain PyTorch on one device; bf16 autocast only on a CUDA GPU that supports it (otherwise fp32; a Colab T4 would be slow, use an L4 or A100). AdamW, peak LR 2e-4, warmup 1,000 steps, cosine decay to 10% of peak, weight decay 0.1 on matrices only, beta2 0.999, gradient clipping 1.0, batch 32 blocks (16,384 token slots), 40,000 steps (about 1.14 passes over the 1,123,259 train blocks). **These values come from the old notebooks' proxy trials (`hyper.md`) on a different architecture; they are starting points, not tuned for this model.** Evaluation every 500 steps on 256 evenly spaced validation blocks and 256 train-eval blocks with fixed `t` and a separately seeded generator; full validation (11,451 blocks) and train-eval (2,774) at the end; `resume.pt` every 1,000 steps; `final/{model.pt, config.json, tokenizer/, history.json, run_info.json, training_curves.png}`; **training trace:** progress line (step, epoch, loss, lr, grad norm, steps/s, ETA) every 100 steps, and the loss / learning rate / epoch / grad norm recorded every 25 steps and drawn in `training_curves.png` (4 panels, with validation and train-eval and the 5.87 baseline), refreshed at every evaluation together with `history.json`; TensorBoard scalars in `<out>/tb`; `python run.py plot --out DIR [--watch 60]` redraws it during a run; `--stop-at` for time-limited sessions; resume refuses a changed recipe.
- **Reference numbers:** uniform guessing = 9.01 nats; a model that knows only token frequencies = **5.87 nats** on the validation tokens (slightly higher on masked tokens only, because the always-given header is included in that count).
- **Acceptance (Q8):** (a) validation loss well below 5.87 and still falling; (b) a reloaded checkpoint reproduces its evaluation loss; (c) a resumed run matches an uninterrupted one; (d) loss curves saved. No target number is set before the first real curve.
- **Status: built and smoke-tested; the real run has NOT been done.** `tests/test_train.py`, 52 checks (including the monitoring: epoch values, live files, TensorBoard scalars, the `plot` command): the tiny model learns a memorisable task (3.69 to 0.003 over 1,500 steps, after a plateau of roughly 100 steps); on CPU, a reloaded checkpoint reproduces its evaluation exactly (b) and resume is bit-for-bit identical to an uninterrupted run, including history and final evaluation (c); on CUDA with bf16 these will be close, not identical; `history.json` and the curves are written (d); resume refuses a changed recipe; `--task sft` still ends in `not_built`. Local smoke runs on this Mac (MPS, fp32, real data): the **tiny** model (0.62M parameters, batch 16) went from 7.53 to 6.26 validation bound in 400 steps at 9.7 steps/s, which is still above the 5.87 baseline, so a run that short says **nothing** about whether the plateau is escaped; the **base** model (35.83M parameters, batch 32) runs at **0.46 steps/s**, so 40,000 steps would take about 24 hours on this Mac: the real run needs a GPU. Acceptance (a) is open until that run. Not tested: the bf16 path (no CUDA here) and Colab.
- **Known risk:** the loss may sit on the ~5.87 token-frequency plateau for thousands of steps (seen in the old notebooks with a different model); watch the first evaluations. One evaluation on 256 blocks is accurate to a few hundredths of a nat because the `1/p` weight is heavy-tailed at small `t` (the mean over evaluation seeds is unbiased, tested); compare steps within a run and use the full-validation number at the end.

### Phase 6: Sampling and inference

- **Decisions (grilled 2026-11-22):** `dlm/sample.py` (`python run.py generate --ckpt <run>/final ...`).
  - **Two modes** for the base model: `story` (unconditional: `[BOS]` + header, generate a whole Story) and `continue` (the opening words you give follow the header and are fixed). Chat instructions wait for fine-tuning (Phase 8): pre-training only saw "Write a short story.".
  - **Window:** `[BOS]` + header + prompt (all given) + up to **480** `[MASK]` tokens, clamped so the total never exceeds `seq_len` (512); no `[PAD]` (the model never attends to it in training either). The Story is **cut at the first `<|end|>` or `[EOS]`** of the final text; the model never learned to fill the tail, and after `[EOS]` it tends to start another Story (packed training blocks), which is discarded.
  - **Semi-autoregressive by default:** *decoding blocks* of 32 tokens (not the 512-token training Block) filled left to right, most-confident-first inside a decoding block, **stop after the decoding block that contains an end token** (later ones are never run). `--block-length 0` = one global pass. The Phase 7 ablation compares them.
  - **Commit rule:** per step the k most confident still-masked positions of the current block are committed (confidence = probability of the chosen token under the **untempered** softmax); committed tokens never change; `[MASK]`/`[PAD]` are never sampled; k follows the linear schedule matching training (equal numbers per step), `--schedule cosine` for the ablation; steps that commit nothing skip the model call.
  - **Steps and temperature:** one token per step by default (an explicit `--steps` is spread over the blocks), **temperature 0.8**, `--temperature 0` = greedy and deterministic; randomness from a seeded CPU generator (`--seed`, sample i uses seed + i).
  - **Seeing the process:** `--trace` prints the unmasking as text rows; `--gif path` writes an animation (Pillow); a **collapse warning** fires when 60% or more of the generated tokens are the same token.
  - Packing decision (Q8, Phase 2): keep the simple greedy packing.
- **Acceptance (tested):** the prefix is never touched; no `[MASK]` left in the processed region; `[MASK]`/`[PAD]` never sampled; committed tokens never change and each position is committed exactly once; a step commits only inside its decoding block; most-confident positions are committed first (stub with rising confidence commits from the right); linear plan = equal numbers per step, plans sum to the block, cosine plan differs; stops after the block containing the end token and later blocks never call the model; Story cut at the first end token; same seed gives the same sample, greedy ignores the seed, different seeds differ; sampled token frequencies follow softmax(logits / T) (T = 1 and 0.5, tolerance 0.03); the collapse warning fires for a degenerate stub and not for varied output; window clamping and mode errors; trace and GIF output; the CLI on a saved run folder (reproducible output, exit codes).
- **Status: built (2026-11-22).** `tests/test_sample.py`, 817 checks, on stub models, a tiny untrained model and a synthetic tokenizer. **No trained model has been sampled yet.** A smoke check on the tiny 600-step real-data model (0.62M parameters, validation 5.96 on the token-frequency plateau) showed the machinery working (left-to-right blocks, confident-first reveal, `Story not finished in the window`, collapse warning at 77% of tokens being `.`); its text is token-frequency soup, which says nothing about the base model. What fills the window after the generated end token is cut, not generated (decided above).
- **Not decided / not built:** token editing (LLaDA2.1), sampling-quality metrics (Phase 7), the batched / served variant for AWS (Phase 10).

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
                      data.py, model.py, diffusion.py, train.py, sample.py   (built: Phases 0-6 incl. model/diffusion/train/sample; train.py --task sft is the only interface skeleton left, Phase 8)
  tests/              offline tests (2,605 checks); `python run.py test`
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
