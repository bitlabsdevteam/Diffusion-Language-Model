# Diffusion Language Model

A project that builds, pre-trains, fine-tunes and deploys a masked diffusion language model from scratch, starting from the TinyStories corpus.

## Language

**Phase**:
A numbered stage of the project plan in `DLM/PRD.md`, from Phase 0 (get the data) onward. Each phase has deliverables and acceptance criteria.
_Avoid_: Step, milestone, stage

**TinyStories**:
The corpus of short, simple children's stories (Hugging Face `roneneldan/TinyStories`) used to train the first model.
_Avoid_: the dataset (too generic), stories file

**Story**:
One complete text example in TinyStories; the unit that is counted, sampled and later wrapped in the chat template.
_Avoid_: document, sample, row

**Split**:
A named, disjoint part of TinyStories: `train` or `validation`. Splits are downloaded whole; choosing a subset of a split is a later-phase decision.
_Avoid_: shard, partition, slice

**Manifest**:
A small record kept beside a downloaded or built artifact that states exactly what it was made from, its counts, and when, so later runs are reproducible.
_Avoid_: metadata file, log

**Mojibake**:
Garbled text where UTF-8 punctuation was decoded with the wrong encoding, such as `Donâ€™t` for `Don't`. Roughly 6% of TinyStories Stories contain it.
_Avoid_: corrupted text, encoding errors

**Special token**:
A token with a reserved meaning and a fixed ID that never comes from ordinary text: `[PAD]`, `[UNK]`, `[BOS]`, `[EOS]`, `[MASK]` (the diffusion noise token) and the chat markers `<|user|>`, `<|assistant|>`, `<|system|>`, `<|end|>`.
_Avoid_: control token, reserved word

**Chat template**:
The fixed wrapping `<|user|>`, an instruction, `<|assistant|>`, the Story, `<|end|>` that marks which part is the prompt and which is the response. Pre-training always uses the instruction "Write a short story."
_Avoid_: prompt format, prompt template

**Block**:
A fixed-length run of tokens fed to the model, made of whole Stories laid end to end from a Story's start, with [PAD] filling the remainder. A Story is never split across Blocks.
_Avoid_: chunk, window, sequence (too generic), batch

**Protected token**:
A token that is never masked during training because it is always given at generation time: the [BOS] and the header of every Story in a Block (the prompt part of the chat template).
_Avoid_: frozen token, fixed token, prompt mask

**Decoding block**:
A run of consecutive positions of the generated response (32 by default) that the sampler fills completely before moving on to the next one, so the response is built left to right while tokens inside the block are revealed most-confident first. Not the training Block.
_Avoid_: an unqualified "block" (it means the training Block), chunk, window
