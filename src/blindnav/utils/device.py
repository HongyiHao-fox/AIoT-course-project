from __future__ import annotations

import gc


def resolve_device(preference: str = "auto") -> str:
    """'auto' picks cuda:0, then Apple mps, then cpu."""
    if preference != "auto":
        return preference
    import torch

    if torch.cuda.is_available():
        return "cuda:0"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def free_memory() -> None:
    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
