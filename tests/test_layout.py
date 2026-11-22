"""Layout tests: every module named in the project layout exists, imports cheaply, and the unbuilt phases
fail the same way (exit 2, naming the phase). python run.py test  (or python tests/test_layout.py)"""
import contextlib
import importlib
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

LIGHT = ("download", "cleaning", "tokenizer", "data", "colab", "hub")   # must import without torch
HEAVY = ("model", "diffusion", "train", "sample")        # may import torch
# command -> text that must appear in the message (the phase and the thing to grill first)
UNBUILT = {
    "sft": (["--init-from", "ckpt"], ("Phase 8", "instruction dataset")),
}


def main():
    checks = 0

    def ok(cond, msg):
        nonlocal checks
        assert cond, msg
        checks += 1

    had_torch = "torch" in sys.modules
    for m in LIGHT:
        mod = importlib.import_module(f"dlm.{m}")
        ok(mod.__doc__ and len(mod.__doc__) > 40, f"dlm.{m} has a docstring")
    ok(had_torch or "torch" not in sys.modules, "importing download/cleaning/tokenizer/data does not import torch")
    for m in HEAVY:
        mod = importlib.import_module(f"dlm.{m}")
        ok(mod.__doc__ and len(mod.__doc__) > 40, f"dlm.{m} has a docstring")

    import run
    for cmd, (extra, needles) in UNBUILT.items():
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            code = run.main([cmd] + extra)
        msg = err.getvalue()
        ok(code == 2, f"`{cmd}` exits 2")
        ok(all(n in msg for n in needles), f"`{cmd}` names {needles}: {msg.strip()!r}")
    err = io.StringIO()
    try:
        with contextlib.redirect_stderr(err):
            run.main(["sft"])
        code = 0
    except SystemExit as e:  # argparse usage error
        code = e.code
    ok(code == 2 and "--init-from" in err.getvalue(), "sft without --init-from is a usage error naming --init-from")
    ok(run.main(["--help"]) == 0, "run.py --help works")
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            run.main(["pretrain", "--help"])
        code = None
    except SystemExit as e:
        code = e.code
    ok(code == 0, "`run.py pretrain --help` shows the training options (pretrain is built)")
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            run.main(["generate", "--help"])
        code = None
    except SystemExit as e:
        code = e.code
    ok(code == 0, "`run.py generate --help` shows the sampler options (generate is built)")
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        ok(run.main(["nonsense"]) == 2, "unknown command exits 2")

    from dlm.train import build_parser
    ns = build_parser().parse_args(["--task", "sft", "--init-from", "x"])
    ok(ns.task == "sft" and ns.init_from == "x", "train parser accepts --task sft --init-from")
    print(f"All {checks} layout checks passed.")


if __name__ == "__main__":
    main()
