# Fine-tuning plan: controllable story generation for dlm-tinystories-36m

## Context

The 36M masked-diffusion LM (LLaDA/MDLM family, d512 x 10 layers, 8k byte-level BPE, 512-token blocks) is pre-trained on TinyStories and only knows one prompt: `<|user|>\nWrite a short story.\n<|assistant|>\n`. Phase 8 of `DLM/PRD.md` (SFT) is unstarted: `--task sft` in `DLM/dlm/train.py:533` and `build_sft_example` in `DLM/dlm/data.py:255` are stubs, and the instruction dataset was left open.

Decision (confirmed with user): **SFT goal = controllable story generation**, run on **Colab L4/A100**.

**Hard constraint (user): everything new lives in one separate folder, `DLM/fine-tuning/`. No pre-training file is edited** (not `dlm/*.py`, `run.py`, `tests/`, `requirements.txt`, `.gitignore`, `PRD.md`, ADR, notebooks). The `dlm` package is imported read-only. This means the `--init-from` / `--task sft` stubs in `dlm/train.py` and `dlm/data.py` stay as they are, and fine-tuning gets its own trainer.
Rationale: a 36M model with a TinyStories-only vocabulary and knowledge cannot become a general chat assistant. Alpaca-style SFT would mostly demonstrate failure (a 50M diffusion model SFT'd on Alpaca-cleaned collapses into repetition loops). In-domain conditioning is learnable and measurable.

## Research summary (what other DLMs do, and what transfers)

| Source | Finding | Use here |
|---|---|---|
| [LLaDA](https://arxiv.org/abs/2502.09992) | SFT = prompt never masked, mask only response tokens, CE on masked positions, short responses padded with `\|EOS\|` | Prompt protection: yes. EOS padding: deliberately not (below) |
| [Rainbow Padding](https://arxiv.org/abs/2510.03680) (ICLR 2026), [BackPlay](https://arxiv.org/pdf/2601.06428), [LLaDA-MoE](https://arxiv.org/pdf/2509.24389) | EOS used as both stop and padding makes mass pile on EOS: early termination ("EOS overflow"). Fixes: distinct pad tokens, cap maskable EOS at ~16, cap length | Avoid EOS-as-padding |
| [Dream 7B](https://arxiv.org/abs/2508.15487) | AR init plus context-adaptive token-level noise rescheduling (CART) | Not applicable at 36M from scratch; CART is an optional later ablation |
| [d1](https://arxiv.org/abs/2504.12216), [LLaDA 1.5 / VRPO](https://arxiv.org/abs/2505.19223) | Post-SFT RL (diffu-GRPO) and variance-reduced preference optimisation | Out of scope for v1. Verifiable story-control rewards (words/sentence/dialogue present) make a later diffu-GRPO step plausible |
| [dLLM library](https://github.com/ZHZisZZ/dllm) | Open SFT/eval recipes for LLaDA, Dream, MDLM, BD3LM | Reference only; our loop is custom PyTorch with a custom 8k vocab, so it cannot be dropped in |
| [nanoDiff 50M/150M/350M SFT](https://huggingface.co/Sebasdi/nanodiff-50m-sft-alpaca) | Small diffusion LM SFT'd on Alpaca-cleaned, response-only masking; repetition loops at small scale | Evidence for choosing in-domain data |
| [TinyStories paper](https://arxiv.org/abs/2305.07759) | TinyStories-Instruct = random subset of {words, features, summary, random sentence} prepended to the story; GPT-4 "GPT-Eval" grading | Data design and eval protocol |
| [SimpleStories](https://huggingface.co/datasets/lennart-finke/SimpleStories) (NeurIPS 2025) | Labelled (theme, topic, style) successor to TinyStories, less formulaic (59% of TinyStories open "Once upon a time") | Optional stage 2 |
| [LoRA Learns Less and Forgets Less](https://arxiv.org/abs/2405.09673) | LoRA forgets less but learns less | Full fine-tuning at 36M; no LoRA needed |

Caveat: no published study combines TinyStories pre-training, masked-diffusion SFT and control metrics, so the recipe below is assembled from adjacent work, not copied.

## Dataset

**Primary: [`roneneldan/TinyStoriesInstruct`](https://huggingface.co/datasets/roneneldan/TinyStoriesInstruct)** (already in `~/.cache/huggingface`; `TinyStories-Instruct-train.txt` 2.66 GB, `-valid.txt` 27 MB). Same story distribution the base model saw, so SFT teaches only conditioning, with minimal forgetting.

Verified on the valid split (25,027 records, `<|endoftext|>`-separated):
- Fields: `Words:` 67%, `Features:` 54% (Dialogue, BadEnding, Foreshadowing, Conflict, MoralValue, Twist...), `Random sentence:` 30%, `Summary:` 100%. Order varies.
- **~30% of records (7,560) carry extra fields AFTER the story body** (e.g. a trailing `Summary:`). A naive split at `Story:` would put summaries inside the response and teach the model to append them.

**Optional stage 2: SimpleStories** (labelled theme/topic/style), only if v1 controls work and more diversity is wanted.
**Rejected:** Alpaca-cleaned / SmolTalk / Dolly (out-of-domain for this vocabulary; revisit only as a deliberate failure-mode study); Alpaca is cached locally, ignore it.
**Licence:** check TinyStoriesInstruct's licence before publishing the SFT model (licence question from the earlier publish work is still open).

## Prerequisite (Step 0): gate on the checkpoint

Weights are not on this machine (`DLM/checkpoints/` is git-ignored; no final validation bound is recorded anywhere). In Colab, load `bitlabsdb/dlm-tinystories-36m` (or the Drive `cosine-step25000-v1` export) via `load_model` (`dlm/model.py:177`) and run `python run.py eval`. Proceed only if the validation ELBO is well below the 5.92 nats/token unigram baseline. If it is on the plateau, SFT is wasted; fix pre-training first.

## Design decisions

1. **Prompt template (reuse existing tokens, no tokenizer change):** `<|user|>\n{instruction}\n<|assistant|>\n{story}\n<|end|>\n`, with `<|system|>` unused. Instruction = a random subset (1-4) of the available fields among Words / Features / Summary / Random sentence, shuffled, e.g. `Write a short story. Words: tower, dragon. Features: Dialogue. Summary: ...`. **Regenerate the subset and order every epoch.** Keep ~15% of examples as the plain "Write a short story." so the unconditional mode does not drift.
2. **Parser:** extract fields before `Story:`, the story body, and discard any post-story fields (v1 drops them; a summarisation task is out of scope). Unit test must cover the trailing-field case and the 1 record with no `Story:` marker and the empty records.
3. **Prompt never masked; loss on response only.** Reuse `diffusion_loss` / `loss_sums` (`dlm/diffusion.py:68-86`), which already take a `protect` mask. PRD acceptance requires a test that prompt tokens are never masked.
4. **Padding: packed SFT blocks, not `[EOS]` padding.** Pack whole examples into 512-token blocks like pre-training (`StoryBlocks`, `dlm/data.py:174`), `[PAD]` stays excluded from the loss and attention keys, and each example ends with `<|end|>` as in pre-training. This matches the pre-training distribution, so no flood of EOS targets. This deliberately departs from the PRD/stub wording ("pad with `[EOS]`"); the PRD is left untouched, and the deviation and its rationale are recorded in `fine-tuning/README.md`. Fallback if length control is poor: one example per block with at most ~16 maskable trailing `[EOS]`, rest `[PAD]`.
5. **Length check:** pre-training already dropped 3.4% of stories over 512 tokens; the prompt adds roughly 40-90 tokens. Measure the new drop rate after tokenizing; if above ~6%, raise to 640 only if RoPE table allows (`DLMConfig.seq_len`), otherwise accept the drop.
6. **Recipe:** full fine-tune from the pre-trained checkpoint; AdamW, peak LR ~2e-5 to 5e-5 (about 5-10x below the 2e-4 pre-training; sweep 2e-5/5e-5), 200-step warmup, cosine decay, wd 0.1, clip 1.0, batch 32 blocks, bf16. Train on a **subset (about 300k-500k examples, a few thousand steps)**; the Instruct train file is larger than the whole pre-training corpus and is not needed in full. Evaluate every 250 steps; keep the best checkpoint (own small saver in `fine-tuning/`, same on-disk format as `save_pretrained`).
7. **Forgetting guard:** track unconditional TinyStories validation ELBO alongside SFT validation ELBO every eval.

## Tooling: Hugging Face, Unsloth, LoRA / low-rank

**What LoRA is.** Freeze each pre-trained weight `W` (out x in) and learn a low-rank update `W' = W + (alpha/r) * B @ A`, with `A` (r x in) random-init and `B` (out x r) zero-init, so training starts exactly at the base model. Only `A` and `B` train; `r` (the rank, typically 8-64) is the bottleneck that limits how much the layer can change. The bet is that task adaptation has low intrinsic rank. Payoff on big LLMs: far fewer trainable parameters and optimizer states, so SFT fits on small GPUs (QLoRA adds 4-bit base weights). Trade-off from [LoRA Learns Less and Forgets Less](https://arxiv.org/abs/2405.09673): lower forgetting, but lower ceiling when the task shift is large.

**Why it does not pay off here.** The model is 35.8M parameters: weights plus AdamW states are about 0.6 GB, so full fine-tuning is trivial on a Colab L4. LoRA saves nothing meaningful. Conditioning on prompt words/sentences also needs real changes to attention (copying from the prompt), which is where low rank can bind.

**Per-tool verdict**

| Tool | Fit | Verdict |
|---|---|---|
| **Custom PyTorch loop (existing `train.py`)** | Already has the diffusion loss, prompt protection, deterministic batches, bit-exact resume, best-checkpoint, tests | **Primary.** Full fine-tune |
| **Hugging Face Transformers `Trainer` / TRL `SFTTrainer`** | Assume a `PreTrainedModel` with an autoregressive `labels` loss. Our model is a plain `nn.Module` with a random-mask ELBO loss (1/t weighted); TRL has no masked-diffusion path | **Skip.** A `compute_loss` subclass could work but would discard the resume/determinism guarantees for no gain |
| **Hugging Face PEFT** | LoRA wraps any `nn.Module` by layer name, so no port is needed | **Use only for the LoRA ablation arm** (below) |
| **Unsloth** | Hand-optimised (Triton) kernels for specific HF architectures (Llama, Qwen, Gemma...) built for multi-billion-parameter LLMs on small GPUs | **Not applicable**: custom architecture, bidirectional diffusion, 36M parameters |
| **[dLLM library](https://github.com/ZHZisZZ/dllm)** | HF-Trainer-based MDLM/LLaDA/Dream SFT recipes | Reference only. Using it would mean porting to its model format and our 8k vocab; its details were not verified |
| **Hugging Face Hub (`dlm/hub.py`)** | Model and dataset hosting | Keep for publishing the SFT model, private first |

**LoRA ablation arm (optional, after the full-FT baseline).** Add `peft` to `fine-tuning/requirements-ft.txt` (not the shared `requirements.txt`; `peft` is not installed in `.venv` now).
- Targets (names from `dlm/model.py`): `attn.qkv`, `attn.out`, `mlp.w1`, `mlp.w3`, `mlp.w2` in all 10 blocks. **Exclude `lm_head`**: it is tied to `tok_emb`, and adapting one would break the tie. Do not adapt `tok_emb`.
- Rank sweep r in {8, 32} (alpha = 2r, dropout 0.0-0.05): r=16 adds about 1.4M trainable parameters (about 4% of the model); r=64 about 5.6M (about 16%).
- LoRA LR is about 10x the full-FT LR (1e-4 to 5e-4), same warmup/cosine, same data and steps so the comparison is clean.
- Export: `merge_and_unload()` then the model's existing `save_pretrained` (`dlm/model.py:163`, called, not modified), so the published checkpoint keeps the existing format.
- Decision rule: adopt LoRA only if it matches full-FT control metrics with less unconditional-ELBO drift; otherwise report it as a negative result. Expected outcome: full FT wins or ties.

## Folder layout: `DLM/fine-tuning/` (all new files; nothing outside it is edited)

```
fine-tuning/
  README.md              purpose, how to run, padding deviation from PRD, licence note
  PLAN.md                copy of this plan (research summary, decisions)
  requirements-ft.txt    peft (LoRA ablation only)
  .gitignore             data/, checkpoints/, runs/ (so DLM/.gitignore stays untouched)
  _path.py               puts the DLM repo root on sys.path so `import dlm` works (read-only reuse)
  ft_parse.py            TinyStoriesInstruct parser (pre-story fields, story, drop trailing fields)
  ft_data.py             prompt builder (random field subset/order per epoch, ~15% plain prompt),
                         build_sft_example, packer into 512-token blocks with prompt `protect` mask,
                         token-cache builder -> fine-tuning/data/
  ft_train.py            SFT loop with --init-from, LR/warmup flags, eval + best-checkpoint, resume
  ft_lora.py             optional PEFT wrapper (targets qkv/out/w1/w3/w2, merge_and_unload export)
  ft_sample.py           instruct-mode generation (fixed prompt, generate response)
  ft_eval.py             control metrics, base vs SFT, forgetting ELBO, report.md/json
  run_ft.py              entry point: prep | train | generate | eval | test
  sft_colab.ipynb        Colab runner (L4/A100), mirrors the existing run_colab.ipynb pattern
  tests/                 test_parse.py, test_data.py, test_train.py (own runner via run_ft.py test)
  data/ checkpoints/ runs/   generated, git-ignored
```

**Read-only reuse from `dlm/` (import, never modify):** `model.load_model` (`model.py:177`) and `save_pretrained` (`:163`); `diffusion.diffusion_loss` / `loss_sums` / `sample_t` / `mask_prob` (`diffusion.py:27-86`, already take a `protect` mask); `tokenizer` (special-token ids and `<|user|>/<|assistant|>/<|end|>` template); `sample.diffusion_generate` (`:85`, confidence remasking, block 32); `train.lr_at`, `param_groups`, `make_amp`, `pick_device` (`train.py:62,70,106,116`); `StoryBlocks` (`data.py:174`) as the packing pattern to mirror.
**Why a separate trainer:** `dlm.train.run_training` (`:373`) always builds a fresh model and `--task sft` ends in `not_built`; wiring `--init-from` there would edit pre-training code, so `ft_train.py` re-implements only the small loop and keeps determinism (batch order a pure function of `(seed, step)`), resume and best-checkpoint behaviour in its own checkpoints dir. Signatures of the reused helpers are checked when implementing; if one is not importable cleanly, copy the few lines into `fine-tuning/` instead of editing `dlm/`.
Do not touch or copy `DLM/AWS_credential.md` (plaintext token, git-ignored).

## Evaluation (control, not just loss)

Same held-out prompts built from `TinyStories-Instruct-valid.txt` (fixed seed), base model vs SFT model, e.g. 500 prompts:
- **Words:** fraction of the 3 required words present in the output.
- **Random sentence:** verbatim string match rate.
- **Dialogue:** quotation marks present when requested.
- **Summary / BadEnding / MoralValue / Twist:** optional LLM-judge using the TinyStories GPT-Eval protocol.
- **Quality / collapse:** existing distinct-n, repetition rate, finished-rate, collapse-rate (`dlm/evaluate.py`); response-length distribution (EOS-overflow check).
- **Forgetting:** unconditional validation ELBO vs the pre-trained model.
Note: TinyStories train/valid overlap is known (~30% of validation stories appear in train per third-party analysis), so use the Instruct valid file only for relative comparison.

Acceptance (adapts PRD Phase 8): SFT beats base on word-inclusion and sentence-match by a clear margin on the fixed prompt set, prompt tokens provably never masked, response length sane, unconditional ELBO within a small tolerance of base.

## Staged run order

0. Checkpoint gate (above).
1. Create `fine-tuning/`; parser + tests (`ft_parse.py`); count trailing-field and over-length rates; build token cache into `fine-tuning/data/`.
2. `ft_data.py` (`build_sft_example`, packer), `ft_train.py` (`--init-from`); CPU smoke test on the tiny preset, then a short Mac/MPS overfit check.
3. Colab (`sft_colab.ipynb`): LR sweep (2e-5 vs 5e-5, ~1.5k steps each), then the full SFT run.
4. Eval vs base with `ft_eval.py`; results and the padding rationale go in `fine-tuning/README.md` (PRD/ADR untouched).
5. Optional LoRA ablation (PEFT, r in {8, 32}) against the full-FT baseline.
6. Optional: SimpleStories stage 2; verifiable-reward diffu-GRPO; CART noise rescheduling ablation.

## Verification

- `python fine-tuning/run_ft.py test` passes (parser incl. trailing fields, prompt-never-masked, packing/length, `--init-from` round trip).
- Existing `python run.py test` still passes unchanged, and `git status` shows changes only under `DLM/fine-tuning/` (proves no pre-training file was touched).
- Tiny-preset smoke SFT loss falls on a memorisable set; reloaded `--init-from` checkpoint reproduces pre-trained eval exactly on CPU.
- `python fine-tuning/run_ft.py eval` on base vs SFT writes comparable `report.md`; control metrics table shows the improvement.
- Spot-read `samples.md` for 20 prompts across field combinations.

## Implementation notes (added after building; deviations from the plan above)

- **LoRA is built in (`ft_lora.py`), not `peft`.** About 40 lines, testable offline, keeps the tok_emb/lm_head tie, merges into a plain checkpoint. `requirements-ft.txt` is informational.
- **Summary coverage is 70%, not 100%**: the plan counted every `Summary:` line; 30% of records have it AFTER the story, where the parser strips it (verified: 7,560 of 25,026 valid records).
- **Over-length**: on a 20k-record smoke prep, 2.3% of responses cannot fit a 512 block even with the shortest prompt, plus 0.8% once real prompts (mean 37 tokens) are added: about 3.1% in total (pre-training: 3.4%).
- **Eval conditions** are plain / words / sentence / dialogue / summary / all; BadEnding / MoralValue / Twist need a judge and are not measured.
- **Status**: code and offline tests complete (2,182 fine-tuning checks; the 2,913 pre-training checks still pass); nothing trained on the real checkpoint yet (weights are not on this machine).
