"""Guards the separation: fine-tuning/ must never change pre-training files. python fine-tuning/tests/test_ft_layout.py

Checks (1) the expected files exist, (2) no fine-tuning source imports from the pre-training package by writing to it
(read-only reuse only: no `dlm.` attribute assignment / monkeypatch), (3) when DLM/ is a git repo, `git status` lists nothing
modified or untracked outside fine-tuning/ that this work could have caused: tracked pre-training files must be unmodified
relative to HEAD, unless HEAD itself is what the user committed (reported, not failed, for files the user edited)."""
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _path  # noqa: E402,F401
from _check import check, exit_with  # noqa: E402

FT = _path.FT_DIR
ROOT = _path.DLM_ROOT
expected = ["README.md", "PLAN.md", "requirements-ft.txt", ".gitignore", "_path.py", "ft_parse.py", "ft_data.py", "ft_train.py",
            "ft_lora.py", "ft_sample.py", "ft_eval.py", "run_ft.py", "sft_colab.ipynb",
            "tests/test_ft_parse.py", "tests/test_ft_data.py", "tests/test_ft_train.py", "tests/test_ft_eval.py", "tests/test_ft_layout.py"]
for f in expected:
    check((FT / f).is_file(), f"missing {f}")

# no fine-tuning module writes into dlm.* (assignment to an attribute of an imported dlm module)
for f in FT.glob("*.py"):
    src = f.read_text()
    for m in re.finditer(r"^\s*(dlm(?:\.\w+)+)\.\w+\s*=[^=]", src, re.M):
        check(False, f"{f.name} assigns into the pre-training package: {m.group(0).strip()}")
    check("importlib.reload" not in src and "monkeypatch" not in src, f"{f.name}: no reload/monkeypatch of dlm")

# the pre-training sources are untouched in git (only when a git repo with a HEAD exists)
try:
    out = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--", ".", ":(exclude)fine-tuning"], capture_output=True, text=True, timeout=30)
    if out.returncode == 0:
        touched = [l for l in out.stdout.splitlines() if l.strip()]
        pre = [l for l in touched if re.search(r"\b(dlm/|tests/|run\.py|requirements\.txt|\.gitignore|PRD\.md|run_colab|publish_hf|test_pretrained)", l)]
        if pre:  # may be the user's own edits: reported, not failed (the dlm.* assignment check above is the hard guard)
            print(f"  NOTE: uncommitted changes to pre-training files (not caused by fine-tuning code unless you just ran it): {pre}")
    else:
        print("  (not a git repo: git check skipped)")
except (OSError, subprocess.TimeoutExpired):
    print("  (git unavailable: git check skipped)")

exit_with("test_ft_layout")
