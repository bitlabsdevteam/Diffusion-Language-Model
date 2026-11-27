"""Tiny assertion helper shared by the fine-tuning tests (same style as ../tests: counted checks, exit code)."""
import sys

_n = 0
_fail = 0


def check(cond, msg=""):
    global _n, _fail
    _n += 1
    if not cond:
        _fail += 1
        print(f"  FAIL: {msg}", flush=True)


def finish(name):
    print(f"{name}: {_n - _fail}/{_n} checks passed", flush=True)
    return 1 if _fail else 0


def exit_with(name):
    sys.exit(finish(name))
