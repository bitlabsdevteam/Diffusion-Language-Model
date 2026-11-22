"""DLM: a discrete (masked) diffusion language model built from scratch. Entry point: python run.py --help"""


def not_built(phase: str, decision: str = "") -> None:
    """Raise for a module that exists as an interface skeleton but is not implemented yet."""
    hint = f" Grill first: {decision}." if decision else ""
    raise NotImplementedError(f"{phase} is not built yet (see PRD.md).{hint}")
