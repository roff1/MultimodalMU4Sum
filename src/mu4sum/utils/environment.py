"""Hardware / platform decoupling: environment detection, root paths, device selection."""
import os
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[3]
ENVIRONMENTS = ("LOCAL", "COLAB", "KAGGLE")


def detect_environment() -> str:
    """Return KAGGLE, COLAB or LOCAL based on environment variables and well-known paths."""
    if "KAGGLE_KERNEL_RUN_TYPE" in os.environ or Path("/kaggle/working").exists():
        return "KAGGLE"
    if "COLAB_GPU" in os.environ or "COLAB_RELEASE_TAG" in os.environ or Path("/content/drive").exists() \
            or Path("/content/sample_data").exists():
        return "COLAB"
    return "LOCAL"


def _default_roots(environment: str):
    """Return the default (data_root, output_root) for an environment; Colab prefers Drive when mounted."""
    if environment == "KAGGLE":
        base = Path("/kaggle/working")
        return base / "data", base / "outputs"
    if environment == "COLAB":
        drive = Path("/content/drive/MyDrive/MU4Sum")
        if drive.parent.exists():
            return drive / "data", drive / "outputs"
        return Path("/content/data"), Path("/content/outputs")
    return PROJECT_ROOT / "data", PROJECT_ROOT / "outputs"


def resolve_runtime(runtime_cfg: dict) -> dict:
    """Turn the `runtime` config section into concrete values (priority: explicit > env var > default)."""
    environment = str(runtime_cfg.get("environment") or "AUTO").upper()
    if environment == "AUTO":
        environment = detect_environment()
    if environment not in ENVIRONMENTS:
        raise ValueError(f"runtime.environment must be AUTO or one of {ENVIRONMENTS}, got '{environment}'")

    default_data, default_out = _default_roots(environment)
    data_root = runtime_cfg.get("data_root") or os.environ.get("MU4SUM_DATA_ROOT") or default_data
    output_root = runtime_cfg.get("output_root") or os.environ.get("MU4SUM_OUTPUT_ROOT") or default_out
    return {
        "environment": environment,
        "data_root": str(Path(data_root).expanduser().resolve()),
        "output_root": str(Path(output_root).expanduser().resolve()),
    }


def get_device(preferred: Optional[str] = None) -> str:
    """cuda > mps > cpu unless a device is requested explicitly."""
    if preferred and preferred != "auto":
        return preferred
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
