"""Checkpointing that restores *everything* needed to continue a run as if it never stopped:
weights, optimizer, LR scheduler, progress counters and all RNG states.

Layout under <run_dir>/checkpoints/:
    step-0000200/    periodic resumable checkpoints (adapter + trainer_state.pt), last N kept
    final/           the finished model (adapter only) + DONE marker; this is what the next stage loads
"""
import os
import shutil
from pathlib import Path
from typing import Optional

from mu4sum.utils.reproducibility import get_rng_state, set_rng_state

STATE_FILE = "trainer_state.pt"
DONE_MARKER = "DONE"


class PeftCheckpointIO:
    """Saves/loads only the LoRA adapter (a few MB) instead of the multi-GB base model."""

    def save(self, model, path: Path) -> None:
        """Write the adapter weights and config to `path`."""
        model.save_pretrained(str(path))

    def load(self, model, path: Path) -> None:
        """Load adapter weights from `path` (safetensors, else .bin) into the model in place."""
        from peft import set_peft_model_state_dict

        safetensors_file = Path(path) / "adapter_model.safetensors"
        if safetensors_file.exists():
            from safetensors.torch import load_file

            state = load_file(str(safetensors_file))
        else:
            import torch

            state = torch.load(Path(path) / "adapter_model.bin", map_location="cpu")
        set_peft_model_state_dict(model, state)


def _step_dirs(root: Path):
    """Step checkpoint dirs that contain a trainer state file, oldest first."""
    return sorted(p for p in Path(root).glob("step-*") if (p / STATE_FILE).exists())


def latest_checkpoint(root: Path) -> Optional[Path]:
    """Return the newest resumable step checkpoint under `root`, or None."""
    dirs = _step_dirs(root)
    return dirs[-1] if dirs else None


def final_checkpoint(root: Path) -> Optional[Path]:
    """Return the `final` checkpoint dir if its DONE marker exists, else None."""
    path = Path(root) / "final"
    return path if (path / DONE_MARKER).exists() else None


def save_checkpoint(root: Path, step: int, model, optimizer, scheduler, progress: dict, io, keep_last: int) -> Path:
    """Atomically write a step checkpoint (adapter, optimizer, scheduler, progress, RNG) and return it.

    Only the newest `keep_last` step checkpoints are retained; `keep_last <= 0` keeps all."""
    import torch

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    final_path, tmp = root / f"step-{step:07d}", root / f".tmp-step-{step:07d}"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir()
    io.save(model, tmp)
    torch.save({"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "progress": progress, "rng": get_rng_state()}, tmp / STATE_FILE)
    if final_path.exists():
        shutil.rmtree(final_path)
    os.replace(tmp, final_path)  # atomic: a half-written checkpoint is never picked up
    for old in _step_dirs(root)[:-keep_last] if keep_last > 0 else []:
        shutil.rmtree(old)
    return final_path


def load_checkpoint(path: Path, model, optimizer, scheduler, io) -> dict:
    """Restore adapter, optimizer, scheduler and RNG states in place; return the saved progress dict."""
    import torch

    io.load(model, path)
    state = torch.load(Path(path) / STATE_FILE, map_location="cpu", weights_only=False)
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    set_rng_state(state["rng"])
    return state["progress"]


def save_final(root: Path, model, io) -> Path:
    """Replace <root>/final with the adapter plus a DONE marker and return its path."""
    path = Path(root) / "final"
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)
    io.save(model, path)
    (path / DONE_MARKER).write_text("ok")
    return path
