"""Put the DLM repo root on sys.path so `import dlm` works from fine-tuning/ (read-only reuse of the pre-training code).

Import this module first in every fine-tuning script: `import _path  # noqa: F401`.
"""
import sys
from pathlib import Path

FT_DIR = Path(__file__).resolve().parent
DLM_ROOT = FT_DIR.parent
for p in (str(FT_DIR), str(DLM_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)
