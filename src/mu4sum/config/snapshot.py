"""Immutable per-run configuration snapshots plus provenance (git commit, library versions)."""
import os
import platform
import stat
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from mu4sum.config.loader import config_fingerprint
from mu4sum.utils.environment import PROJECT_ROOT
from mu4sum.utils.io import write_json_atomic


class ImmutableConfigError(RuntimeError):
    """Raised when a run folder already holds a snapshot with a different configuration."""
    pass


def _comparable(cfg: DictConfig) -> dict:
    data = OmegaConf.to_container(cfg, resolve=True)
    data.pop("runtime", None)
    return data


def save_run_config(cfg: DictConfig, directory, operation: str, versioned: bool = False) -> Path:
    """Write a read-only snapshot of the resolved config; never overwrite an existing one.

    Default: `config_<operation>.yaml`. Re-running with the same config is a no-op (resume); a
    different config in the same folder is refused, since it would silently change a run's meaning.
    versioned=True (incremental, resumable stages like data generation): the file name carries the
    config fingerprint, so every distinct config gets its own immutable snapshot side by side.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    name = f"config_{operation}_{config_fingerprint(cfg)}.yaml" if versioned else f"config_{operation}.yaml"
    path = directory / name
    if path.exists():
        if _comparable(OmegaConf.load(path)) != _comparable(cfg):
            raise ImmutableConfigError(
                f"{path} already exists with a different configuration. Runs are immutable: "
                f"use a new --tag (or remove the folder) instead of overwriting."
            )
        return path
    OmegaConf.save(cfg, path)
    os.chmod(path, stat.S_IREAD | stat.S_IRGRP | stat.S_IROTH)
    save_provenance(cfg, directory, f"{operation}_{config_fingerprint(cfg)}" if versioned else operation)
    return path


def _git(*args) -> str:
    try:
        out = subprocess.run(["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=10)
        return out.stdout.strip() if out.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


def _version(pkg: str):
    try:
        from importlib.metadata import version

        return version(pkg)
    except Exception:
        return None


def save_provenance(cfg: DictConfig, directory, operation: str) -> None:
    """Write provenance_<operation>.json (git commit, environment, versions) unless it already exists."""
    path = Path(directory) / f"provenance_{operation}.json"
    if path.exists():
        return
    write_json_atomic(path, {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config_fingerprint": config_fingerprint(cfg),
        "seed": cfg.seed,
        "environment": cfg.runtime.environment,
        "git_commit": _git("rev-parse", "HEAD") or None,
        "git_dirty": bool(_git("status", "--porcelain")),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {p: _version(p) for p in
                     ("torch", "transformers", "peft", "datasets", "omegaconf", "bitsandbytes")},
    })
