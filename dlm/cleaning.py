"""Shared Story cleaning. Used by Phase 1 (tokenizer training) and Phase 2 (training text) so the two never disagree.

TinyStories contains mojibake: UTF-8 punctuation that was decoded as cp1252, e.g. ``Donâ€™t`` and
``â€œHi!â€``. About 6% of Stories are affected. The closing double quote is E2 80 9D and the last byte has
no cp1252 character, so it was lost and only a bare ``â€`` remains.

`clean_story` repairs the known sequences (no encoding guessing), folds curly quotes to ASCII, and strips
surrounding whitespace. Everything else passes through unchanged. Bump CLEANING_VERSION when it changes:
the tokenizer manifest records it, so a stale tokenizer is retrained.
"""
from __future__ import annotations

CLEANING_VERSION = 1

# Known mojibake sequences -> ASCII. Order matters only in that the bare "â€" fallback runs last.
_MOJIBAKE = (
    ("â€œ", '"'),      # “ opening double quote
    ("â€™", "'"),      # ’ apostrophe / right single quote
    ("â€˜", "'"),      # ‘ left single quote
    ("â€“", "-"),      # – en dash
    ("â€”", "-"),      # — em dash
    ("â€¦", "..."),    # … ellipsis
    ("â€‹", ""),  # zero-width space
    ("â€Š", " "),  # hair space
)
_MOJIBAKE_MARKER = "â€"  # a bare marker left after the table is a double quote whose last byte was lost
_FOLD = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"'})


def has_mojibake(text: str) -> bool:
    return _MOJIBAKE_MARKER in text


def clean_story(text: str) -> str:
    """Repair known mojibake, fold curly quotes to ASCII, strip surrounding whitespace."""
    text = text.strip()
    if _MOJIBAKE_MARKER in text:
        for bad, good in _MOJIBAKE:
            text = text.replace(bad, good)
        text = text.replace(_MOJIBAKE_MARKER, '"')
    return text.translate(_FOLD)
