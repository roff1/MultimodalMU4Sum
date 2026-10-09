"""Seeding, determinism and RNG-state capture for exact checkpoint resumption."""
import hashlib
import json
import os
import random
from typing import Any, Dict

import numpy as np


def set_seed(seed: int = 42, deterministic: bool = True) -> int:
    """Seed python / numpy / torch (CPU+CUDA) and optionally force deterministic kernels."""
    random.seed(seed)
    # Only affects subprocesses (e.g. DataLoader workers): hash seed is fixed at interpreter start.
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return seed

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
    return seed


def seed_worker(worker_id: int) -> None:
    """DataLoader worker_init_fn: derive python/numpy seeds from torch's per-worker seed."""
    import torch

    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_rng_state() -> Dict[str, Any]:
    """Capture python, numpy and (if available) torch CPU/CUDA RNG states."""
    state: Dict[str, Any] = {"python": random.getstate(), "numpy": np.random.get_state()}
    try:
        import torch

        state["torch"] = torch.get_rng_state()
        if torch.cuda.is_available():
            state["cuda"] = torch.cuda.get_rng_state_all()
    except ImportError:
        pass
    return state


def set_rng_state(state: Dict[str, Any]) -> None:
    """Restore RNG states captured by get_rng_state."""
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    if "torch" in state:
        import torch

        torch.set_rng_state(state["torch"].cpu())
        if "cuda" in state and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def fingerprint(obj: Any) -> str:
    """Stable short hash of any JSON-serialisable structure."""
    blob = json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]
