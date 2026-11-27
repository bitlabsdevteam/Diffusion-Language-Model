# Fine-tuning the pre-trained DLM (controllable story generation)

Everything in this folder is **new and separate**: the pre-training code (`../dlm`, `../run.py`, `../tests`,
`../PRD.md`, the notebooks) is imported read-only and never edited. `PLAN.md` has the research summary and the reasoning
behind the choices below.

**Goal.** The 36M masked-diffusion model only knows `Write a short story.`. Fine-tuning teaches it to follow
TinyStories-Instruct-style instructions: required **words**, **features** (Dialogue, BadEnding, Twist, ...), a
**summary**, or a **sentence that must appear**. A 36M model with a TinyStories-only vocabulary cannot become a general
chat assistant, so Alpaca/SmolTalk-style data was rejected (see PLAN.md).

## Quick start (from `DLM/`)

```bash
.venv/bin/python fine-tuning/run_ft.py test                      # offline tests (about a minute)
.venv/bin/python fine-tuning/run_ft.py prep                      # parse + tokenize (valid + 20% of train) -> fine-tuning/data/cache
.venv/bin/python fine-tuning/run_ft.py train --init-from <pre-trained folder> --out fine-tuning/checkpoints/sft_lr3e-5
.venv/bin/python fine-tuning/run_ft.py generate --ckpt fine-tuning/checkpoints/sft_lr3e-5/final --words cat,ball --features Dialogue
.venv/bin/python fine-tuning/run_ft.py eval --base <pre-trained folder> --sft fine-tuning/checkpoints/sft_lr3e-5/final
```

Real training belongs on a Colab L4/A100 (`sft_colab.ipynb`); the Mac (MPS) is for smoke tests only.
**Step 0 (gate):** the pre-trained weights are not in this repo. Before any SFT, load them
(`bitlabsdb/dlm-tinystories-36m` or the Drive `cosine-step25000-v1` export) and run `python run.py eval --ckpt <folder>`:
the validation bound must be well below the 5.92 nats/token unigram baseline. If it is on that plateau, fine-tuning is wasted.
`train` enforces this: with `--pretrain-cache` it aborts at step 0 when the base model's plain-prompt bound is above
`--gate-uncond-max` (default 5.5; `0` turns the gate off). Generation and `eval` sample with the noise schedule the model was
trained with (`--schedule auto`, read from `run_info.json`).

## Files

| File | Role |
|---|---|
| `_path.py` | puts the DLM root on `sys.path` (read-only `import dlm`) |
| `ft_parse.py` | TinyStoriesInstruct parser; strips fields that appear AFTER the story |
| `ft_data.py` | instruction builder, `build_sft_example`, packer, cache, per-epoch deterministic stream, fixed validation set |
| `ft_train.py` | SFT loop (`--init-from`, resume, best/final, forgetting guard, optional LoRA) |
| `ft_lora.py` | built-in LoRA (no `peft` dependency), merges into a plain checkpoint |
| `ft_sample.py` | instruction-conditioned generation (wraps `dlm.sample.diffusion_generate`) |
| `ft_eval.py` | base-vs-SFT control metrics |
| `run_ft.py` | entry point: `prep`, `train`, `generate`, `eval`, `test` |
| `sft_colab.ipynb` | Colab runner |
| `tests/` | `test_ft_parse`, `test_ft_data`, `test_ft_train`, `test_ft_eval`, `test_ft_layout` |

## Data

[`roneneldan/TinyStoriesInstruct`](https://huggingface.co/datasets/roneneldan/TinyStoriesInstruct) (same stories the base
model saw, so SFT teaches conditioning, not new content). Measured on the valid split (25,026 usable records):
`Words:` 67%, `Features:` 54%, `Random sentence:` 30%; **30% of records carry fields after the story body**. Those trailing
fields (mostly `Summary:`) are stripped from the response, which leaves a usable `Summary:` in 70% of records, not 100%.
`prep` keeps 20% of train (about 420k examples) by default: a few thousand steps never need more.
On a 20k-record smoke prep, 2.3% of responses are too long to fit a 512-token block even with the shortest prompt; the
real prompts (mean 37 tokens) drop 0.8% more (about 3.1% in total, versus 3.4% in pre-training).

## Design decisions (and where they depart from the PRD)

1. **Layout** identical to pre-training: `[BOS] <|user|>\n{instruction}\n<|assistant|>\n {story}\n<|end|>\n [EOS]`; the
   `<|system|>` token stays unused. No tokenizer change.
2. **Instruction** = `Write a short story.` + a random subset (1-4) of the available fields in random order, re-drawn
   every epoch; 15% of examples keep the plain prompt so unconditional generation does not drift.
3. **Prompt never masked, loss on the response only**: the prompt (and `[BOS]`) goes in `protect`, which
   `dlm.diffusion.corrupt` never masks. `tests/test_ft_data.py` proves it under both noise schedules at t=1.
4. **Padding: packed blocks, NOT `[EOS]` padding.** `PRD.md` and the `build_sft_example` stub say "pad with `[EOS]`" (LLaDA's
   SFT recipe). Here whole examples are packed into 512-token blocks like pre-training, `[PAD]` stays out of the loss and
   out of attention keys, and each example ends with `<|end|>`. Reason: using `[EOS]` as both stop token and padding makes
   `[EOS]` the dominant target and causes early termination ("EOS overflow": Rainbow Padding, BackPlay, LLaDA-MoE). The PRD
   is left untouched; this paragraph is the record. Fallback if length control is poor: one example per block with at
   most about 16 maskable trailing `[EOS]`.
5. **Recipe** (starting points): LR 3e-5 (sweep 2e-5 / 5e-5), 200 warmup steps, cosine to 10%, weight decay 0.1, clip 1.0,
   batch 32 blocks, 3,000 steps, bf16 on CUDA, noise schedule = the base model's (read from its `run_info.json`).
6. **Forgetting guard:** every evaluation also reports the plain-prompt pre-training validation bound; a step-0 evaluation of
   the untouched base model is recorded first, so `history.json` and the log show the drift directly.
7. **Own trainer:** `dlm.train.run_training` always builds a fresh model and `--task sft` is a stub; changing it would edit
   pre-training code, so `ft_train.py` re-implements the small loop and reuses `dlm.train`'s helpers (`lr_at`, `evaluate`,
   `make_batch`, `save_best`, ...). Resume is bit-exact on CPU (tested).

## LoRA / low-rank (ablation arm, off by default)

`train --lora-r 8` trains adapters only: `W' = W + (alpha/r) B A` on `attn.qkv`, `attn.out`, `mlp.w1/w3/w2` (alpha = 2r by
default). The embedding / `lm_head` (one tied tensor) and the norm gains stay frozen. `best/` and `final/` are always merged,
so the checkpoint format is unchanged. At 35.8M parameters full fine-tuning is trivial on an L4, so LoRA saves nothing;
it is run only to measure whether it forgets less at equal control quality. Hugging Face Trainer/TRL (autoregressive
`labels` loss, `PreTrainedModel`) and Unsloth (kernels for specific HF architectures) do not fit this custom diffusion model.
The built-in LoRA replaced the planned `peft` dependency (about 40 lines, testable offline, no tie handling surprises);
`requirements-ft.txt` is therefore informational.

## Evaluation

`eval` sends the same held-out prompts (plain / words / sentence / dialogue / summary / all) to the base and the SFT model and
reports required-word presence, verbatim sentence rate, quoted-speech rate, a lexical summary proxy, finished rate, length,
distinct-n, repetition and collapse rates, plus the forgetting bound. BadEnding / MoralValue / Twist adherence needs a judge
(TinyStories "GPT-Eval"); none is called (it would send stories to an external service). Always read `control_samples.md`.
Acceptance: SFT beats base on word and sentence control by a clear margin, response lengths are sane, the unconditional
bound stays within a small tolerance of the base.

**Status:** code and offline tests are done; nothing has been trained on the real checkpoint. Results will be added here.

## Before publishing a fine-tuned model

Check the TinyStoriesInstruct licence and settle the model licence (the earlier publish work left the licence open; the
model card defaults to apache-2.0 and the GitHub repo has no LICENSE). `../AWS_credential.md` holds a plaintext token and
must never be copied into a Hub upload or Colab notebook.
