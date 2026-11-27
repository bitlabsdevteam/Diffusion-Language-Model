"""Offline tests for fine-tuning/ft_parse.py: python fine-tuning/tests/test_ft_parse.py"""
import glob
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _path  # noqa: E402,F401
from _check import check, exit_with  # noqa: E402

import ft_parse as P  # noqa: E402

SIMPLE = """Words: tower, dragon, knight
Features: Dialogue, BadEnding
Summary: A tower falls.
Random sentence: He was sad.
Story: 
Sam built a tower. "Look!" he said.
It fell. He was sad.
"""
FIELDS_AFTER = """Features: Dialogue
Story: 
Tom had a ball.
He threw it.
Summary: Tom plays with a ball.
"""
ORDER_VARIES = """Summary: Mia finds a cat.
Random sentence: The cat purred.
Story: 
Mia found a cat. The cat purred.
"""
NO_MARKER = "ooks. He said, \"Yes, please.\"\nSummary: broken fragment\n"
MULTI_TRAILING = """Words: a
Story: 
One line.
Summary: s
Features: Twist
Random sentence: x
"""

# -- basic record
r = P.parse_record(SIMPLE)
check(r is not None, "simple record parses")
check(r.words == ["tower", "dragon", "knight"], f"words {r.words}")
check(r.features == ["Dialogue", "BadEnding"], f"features {r.features}")
check(r.summary == "A tower falls." and r.sentence == "He was sad.", "summary / sentence")
check(r.story.startswith("Sam built a tower.") and r.story.endswith("He was sad."), f"story {r.story!r}")
check(not r.trailing_dropped, "no trailing fields in simple record")
check("Summary" not in r.story and "Words:" not in r.story, "no field text inside story")

# -- fields AFTER the story must not leak into the response (the 30% case)
r = P.parse_record(FIELDS_AFTER)
check(r is not None and r.trailing_dropped, "trailing field detected")
check(r.story == "Tom had a ball.\nHe threw it.", f"trailing summary stripped: {r.story!r}")
check(r.summary is None, "a trailing Summary is not turned into an instruction field in v1")
check(r.features == ["Dialogue"], "leading Features still parsed")

r = P.parse_record(MULTI_TRAILING)
check(r.story == "One line." and r.trailing_dropped, f"several trailing fields stripped: {r.story!r}")

# -- order variation
r = P.parse_record(ORDER_VARIES)
check(r.summary == "Mia finds a cat." and r.sentence == "The cat purred." and not r.words, "field order does not matter")
check(r.fields_present() == ["Summary", "Random sentence"], r.fields_present())

# -- degenerate records
check(P.parse_record(NO_MARKER) is None, "record without Story: marker is skipped")
check(P.parse_record("") is None and P.parse_record("\n\n") is None, "empty record is skipped")
check(P.parse_record("Words: a\nStory: \n\n") is None, "record with an empty story is skipped")

# -- a story line that merely contains a field name mid-line is kept
r = P.parse_record("Story: \nHe said Summary: none.\nThe end.\n")
check(r.story == "He said Summary: none.\nThe end.", f"mid-line field name kept: {r.story!r}")

# -- streaming reader: separators, chunk boundaries, last record without separator
text = P.SEPARATOR.join([SIMPLE, FIELDS_AFTER, NO_MARKER, ORDER_VARIES]) 
with tempfile.TemporaryDirectory() as d:
    f = Path(d) / "x.txt"
    f.write_text(text, encoding="utf-8")
    for chunk in (7, 64, 1 << 20):  # tiny chunks split separators and records mid-way
        recs = list(P.iter_records(f, chunk_bytes=chunk))
        check(len(recs) == 3, f"chunk {chunk}: {len(recs)} records (expected 3, fragment skipped)")
        check([x.story[:8] for x in recs] == ["Sam buil", "Tom had ", "Mia foun"], f"chunk {chunk}: order/content")
    st = P.scan_stats(f)
    check(st["records"] == 3 and st["trailing_fields_dropped"] == 1, st)

# -- real data (skipped when the HF cache is absent): the numbers the plan relies on
hits = glob.glob(str(Path.home() / ".cache/huggingface/hub/datasets--roneneldan--TinyStoriesInstruct/snapshots/*/TinyStories-Instruct-valid.txt"))
if hits:
    st = P.scan_stats(hits[0])
    check(24_900 <= st["records"] <= 25_100, f"valid records {st['records']}")
    check(0.25 <= st["trailing_share"] <= 0.35, f"trailing share {st['trailing_share']} (measured ~0.30)")
    check(st["field_share"]["Summary"] < 1.0 + 1e-9, "summary share is a share")
    recs = list(P.iter_records(hits[0]))
    leaked = sum(1 for x in recs if any(P._FIELD_LINE.match(l) for l in x.story.split("\n")[-1:]))
    check(leaked == 0, f"{leaked} stories still end in a field line")
else:
    print("  (real valid file not in the HF cache: real-data checks skipped)")

exit_with("test_ft_parse")
