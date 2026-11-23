"""Colab helpers: end the session when the work is saved, so no more compute units are spent.

`finish_session` is the last cell of run_colab.ipynb. It only ends the session when it is SAFE to do so:
  1. the trained model <OUT>/checkpoints/<run>/final/model.pt exists on Drive (and, by default, the evaluation report
     <run>/eval/report.json too, because that report is the point of the run). `require_model=False` skips the model
     check, and `out_root=None` skips every Drive step (no checks, no copying, no flush), for a notebook that only TESTS a
     model published on the Hugging Face Hub and never mounts Drive;
  2. GIFs produced under /content (not on Drive) are copied to <run>/eval/ first;
  3. a countdown gives you `grace_seconds` to cancel (stop the cell);
  4. Drive is flushed and unmounted so every write reaches it, and only then the runtime is released with
     google.colab.runtime.unassign().
If any check fails it prints why and does NOT end the session (the session then keeps running and spending credits
until you end it yourself: Runtime -> Disconnect and delete runtime). Outside Colab it does nothing.
"""
from __future__ import annotations

import glob
import shutil
import time
from pathlib import Path


def finish_session(out_root, run: str = "pretrain_base", grace_seconds: int = 60, require_eval: bool = True,
                   require_model: bool = True, enabled: bool = True, copy_globs: tuple[str, ...] = ("/content/unmask*.gif",),
                   sleep=time.sleep, log=print) -> dict:
    """Return {"ended": bool, "reason": str, "copied": [paths]}; "ended" means unassign() was called."""
    def refuse(reason):
        log(f"NOT ending the session: {reason}")
        return {"ended": False, "reason": reason, "copied": copied}

    copied: list[str] = []
    if not enabled:
        return refuse("QUIT_WHEN_DONE is False (the session stays connected)")
    use_drive = out_root is not None  # out_root=None: a notebook that never mounted Drive (it only tests a model from the Hub)
    if use_drive:
        run_dir = Path(out_root) / "checkpoints" / run
        model = run_dir / "final" / "model.pt"
        if require_model and not model.is_file():
            return refuse(f"{model} does not exist, so training has not finished (or did not save). "
                          "Nothing is lost by staying connected; fix the problem, or disconnect yourself")
        if require_eval and not (run_dir / "eval" / "report.json").is_file():
            return refuse(f"{run_dir / 'eval' / 'report.json'} does not exist: run the evaluation cell first "
                          "(or call finish_session(..., require_eval=False))")
    try:
        from google.colab import runtime
    except ImportError:
        return refuse("not running on Colab (google.colab is unavailable)")

    if use_drive:
        dest = run_dir / "eval"
        for pattern in copy_globs:
            for f in sorted(glob.glob(pattern)):
                dest.mkdir(parents=True, exist_ok=True)
                target = dest / Path(f).name
                shutil.copy2(f, target)
                copied.append(str(target))
        saved = f"Saved on Drive: {model}" if model.is_file() else "Nothing to save (the model is on the Hub)"
        log(saved + (f"; copied {len(copied)} file(s) to {dest}" if copied else ""))
    else:
        log("Nothing to save: the model is on the Hub and Drive is not used.")
    for s in range(int(grace_seconds), 0, -1):
        log(f"Ending the session in {s} s (stop this cell to cancel) ...")
        sleep(1)
    if use_drive:
        try:
            from google.colab import drive

            drive.flush_and_unmount()  # make sure every write has reached Drive before the disk disappears
        except Exception as e:
            return refuse(f"could not flush Drive ({type(e).__name__}: {e}); not ending the session so nothing is lost")
        log("Drive flushed. Releasing the Colab runtime now.")
    else:
        log("Releasing the Colab runtime now.")
    runtime.unassign()
    return {"ended": True, "reason": "unassigned", "copied": copied}
