"""Parser for roneneldan/TinyStoriesInstruct (TinyStories-Instruct-{train,valid}.txt).

File layout: records separated by the literal text `<|endoftext|>`. Inside a record, instruction fields come in a
varying order and subset (`Words:`, `Features:`, `Summary:`, `Random sentence:`), then a `Story:` marker line, then
the story body. IMPORTANT (measured on the valid split: 7,560 of 25,026 records, about 30%): some records carry
fields AFTER the story body (typically a trailing `Summary:`). A naive "split at Story:" would leave those lines in
the response and teach the model to append a summary after every story. This parser therefore:

  * takes the fields before `Story:` as the instruction,
  * strips any run of field lines at the END of the body (and counts them: `trailing_dropped`),
  * skips records without a `Story:` marker (the first record of the valid file starts mid-story) and empty ones.

Pure text processing: no torch, no dlm import. Cleaning (mojibake, curly quotes) happens in ft_data.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

SEPARATOR = "<|endoftext|>"
FIELD_NAMES = ("Words", "Features", "Summary", "Random sentence")
_FIELD_LINE = re.compile(r"^(Words|Features|Summary|Random sentence):[ \t]*(.*)$")
_STORY_MARKER = re.compile(r"(?m)^Story:[ \t]*\n?")


@dataclass
class Record:
    words: list[str] = field(default_factory=list)
    features: list[str] = field(default_factory=list)
    summary: str | None = None
    sentence: str | None = None  # the "Random sentence:" field: a sentence that occurs in the story
    story: str = ""
    trailing_dropped: bool = False  # True when field lines after the story body were removed

    def fields_present(self) -> list[str]:
        out = []
        if self.words:
            out.append("Words")
        if self.features:
            out.append("Features")
        if self.summary:
            out.append("Summary")
        if self.sentence:
            out.append("Random sentence")
        return out


def _split_list(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def _apply_field(rec: Record, name: str, value: str) -> None:
    value = value.strip()
    if name == "Words":
        rec.words = _split_list(value)
    elif name == "Features":
        rec.features = _split_list(value)
    elif name == "Summary":
        rec.summary = value or None
    elif name == "Random sentence":
        rec.sentence = value or None


def strip_trailing_fields(body: str) -> tuple[str, bool]:
    """Remove the run of field lines at the very end of a story body. Returns (story, removed_any)."""
    lines = body.rstrip("\n").split("\n")
    end = len(lines)
    while end > 0 and (not lines[end - 1].strip() or _FIELD_LINE.match(lines[end - 1])):
        end -= 1
    removed = any(_FIELD_LINE.match(x) for x in lines[end:])
    return "\n".join(lines[:end]).strip(), removed


def parse_record(text: str) -> Record | None:
    """One record (text between `<|endoftext|>` separators) -> Record, or None if it has no usable story."""
    text = text.strip("\n")
    if not text.strip():
        return None
    m = _STORY_MARKER.search(text)
    if m is None:
        return None
    rec = Record()
    for line in text[:m.start()].split("\n"):
        fm = _FIELD_LINE.match(line)
        if fm:
            _apply_field(rec, fm.group(1), fm.group(2))
    rec.story, rec.trailing_dropped = strip_trailing_fields(text[m.end():])
    return rec if rec.story else None


def iter_records(path: str | Path, chunk_bytes: int = 1 << 24) -> Iterator[Record]:
    """Stream a TinyStoriesInstruct text file (2.66 GB for train) without loading it whole."""
    buf = ""
    with open(path, encoding="utf-8", errors="replace") as f:
        while True:
            chunk = f.read(chunk_bytes)
            if not chunk:
                break
            buf += chunk
            parts = buf.split(SEPARATOR)
            buf = parts.pop()  # the last piece may be incomplete
            for p in parts:
                rec = parse_record(p)
                if rec is not None:
                    yield rec
    rec = parse_record(buf)
    if rec is not None:
        yield rec


def scan_stats(path: str | Path, limit: int | None = None) -> dict:
    """Counts used to justify the parser (field coverage, trailing-field rate, skipped records)."""
    n = trailing = 0
    present = {k: 0 for k in FIELD_NAMES}
    words_len = 0
    for rec in iter_records(path):
        n += 1
        trailing += rec.trailing_dropped
        for name in rec.fields_present():
            present[name] += 1
        words_len += len(rec.story.split())
        if limit and n >= limit:
            break
    return {"records": n, "trailing_fields_dropped": trailing,
            "trailing_share": round(trailing / n, 4) if n else 0.0,
            "field_share": {k: round(v / n, 4) if n else 0.0 for k, v in present.items()},
            "mean_story_words": round(words_len / n, 1) if n else 0.0}
