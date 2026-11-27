"""Offline tests for fine-tuning/ft_eval.py metrics and prompt selection: python fine-tuning/tests/test_ft_eval.py"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _path  # noqa: E402,F401
from _check import check, exit_with  # noqa: E402

import ft_data as D  # noqa: E402
import ft_eval as E  # noqa: E402
import ft_sample as S  # noqa: E402

# -- scoring
req = {"w": ["cat", "Ball", "happy"], "r": "The cat was happy.", "f": ["Dialogue"], "s": "A little cat finds a red ball in the garden."}
good = 'The cat was happy. "Look at my ball!" said the cat. She found a red ball in the garden.'
m = E.score(good, req)
check(abs(m["words"] - 1.0) < 1e-9, f"all required words present: {m['words']}")
check(m["sentence_exact"] and m["sentence_recall"] == 1.0, "sentence found verbatim (case/space-insensitive)")
check(m["dialogue"] is True and m["quotes"] is True, "quoted speech detected")
check(abs(m["summary_recall"] - 0.5) < 1e-9, f"summary recall {m['summary_recall']}")
bad = "A dog ran away. Nothing else happened."
m2 = E.score(bad, req)
check(m2["words"] == 0.0 and not m2["sentence_exact"] and m2["dialogue"] is False and m2["quotes"] is False, "unrelated story scores zero")
check(E.score("The Cat sat. A happy ball!", {"w": ["cat", "ball", "happy"]})["words"] == 1.0, "word match is case-insensitive")
check(E.score("The cats sat.", {"w": ["cat"]})["words"] == 0.0, "whole words only (no substring credit)")
check(E.score("It’s fine. he said it.", {"r": "It's fine."})["sentence_exact"], "curly apostrophe normalised")
check("words" not in E.score("x", {"f": ["BadEnding"]}) and "dialogue" not in E.score('"a" "b"', {"f": ["BadEnding"]}), "metrics only for requested fields")
check(E.score('one " quote', {"f": ["Dialogue"]})["dialogue"] is False, "a single quote mark is not dialogue")
check(E.mean([]) is None and E.mean([True, False]) == 0.5, "mean helper")

# -- instruction builder used for eval/generation matches ft_data's format
instr = S.make_instruction(words=["a", "b"], features=["Dialogue"], summary="S.", sentence="X.")
check(instr.split("\n") == [D.PLAIN_INSTRUCTION, "Words: a, b", "Features: Dialogue", "Summary: S.", "Random sentence: X."], instr.split("\n"))
check(S.make_instruction() == D.PLAIN_INSTRUCTION, "no fields -> plain instruction")

# -- prompt selection on a synthetic cache
def raw_record(i):
    lines = [f"Words: w{i}, x{i}, y{i}", f"Summary: Sum {i} garden.", f"Random sentence: Line {i} here."]
    if i % 2:
        lines.append("Features: Dialogue, Twist")
    return "\n".join(lines) + "\nStory: \n" + f"Story {i}. Line {i} here.\n"

with tempfile.TemporaryDirectory() as d:
    d = Path(d)
    raw = d / "raw.txt"
    raw.write_text("<|endoftext|>\n".join(raw_record(i) for i in range(120)), encoding="utf-8")
    tok_json = _path.DLM_ROOT / "tokenizer" / "tokenizer.json"
    D.build_cache(raw, d / "c", "valid", tok_json)
    cache = D.SftCache(d / "c", "valid")
    p1 = E.select_prompts(cache, 10, seed=0)
    p2 = E.select_prompts(cache, 10, seed=0)
    p3 = E.select_prompts(cache, 10, seed=1)
    check(p1 == p2, "prompt selection is deterministic")
    check(p1 != p3, "a different seed selects different prompts")
    check(set(p1) == set(E.CONDITIONS) and all(len(v) == 10 for v in p1.values()), {k: len(v) for k, v in p1.items()})
    check(all(it["instruction"] == D.PLAIN_INSTRUCTION for it in p1["plain"]), "plain condition = plain instruction")
    check(all(it["instruction"].count("\n") == 1 and it["instruction"].split("\n")[1].startswith("Words: ") for it in p1["words"]), "words condition asks for words only")
    check(all(it["instruction"].endswith("Features: Dialogue") for it in p1["dialogue"]), "dialogue condition asks for Dialogue only")
    check(all(cache.fields[it["index"]]["f"] and "Dialogue" in cache.fields[it["index"]]["f"] for it in p1["dialogue"]), "dialogue prompts come from records that have dialogue")
    check(all(len(it["fields"]) >= 3 for it in p1["all"]), "'all' uses 3+ fields")
    # held-out prompts of one condition never repeat an example
    check(len({it["index"] for it in p1["words"]}) == 10, "no duplicate examples within a condition")

# -- markdown rendering does not crash and carries the headline rows
agg = {"n": 2, "finished_rate": 1.0, "length": {"median": 100, "p90": 120, "mean": 100.0}, "distinct_2": 0.9,
       "repetition_rate": 0.01, "collapse_rate": 0.0, "words": 0.5, "sentence_exact": 0.1, "dialogue": 0.4, "summary_recall": 0.3}
side = {c: {"agg": dict(agg)} for c in E.CONDITIONS}
rep = {"base_ckpt": "b", "sft_ckpt": "s", "n_per_condition": 2, "seed": 0, "gen": {"max_new_tokens": 400, "temperature": 0.8},
       "base": side, "sft": {c: {"agg": dict(agg, words=0.9)} for c in E.CONDITIONS}, "uncond": {"base": 1.0, "sft": 1.05}}
md = E.render_markdown(rep)
check("required words present | 0.5 | 0.9 | +0.400" in md and "drift +0.0500" in md, "markdown has headline delta and drift")

exit_with("test_ft_eval")
