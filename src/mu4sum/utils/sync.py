"""Pull lightweight run artifacts (configs, metrics, predictions, logs) from the remote/Drive output root
into the local repo so they can be tracked with Git. Checkpoints and caches are never copied."""
import filecmp
import shutil
from pathlib import Path
from typing import List, Sequence

DEFAULT_SUFFIXES = (".yaml", ".yml", ".json", ".jsonl", ".csv", ".txt", ".md")
EXCLUDED_DIRS = {"checkpoints", "_kagglehub", "__pycache__"}


def sync_artifacts(src: Path, dst: Path, suffixes: Sequence[str] = DEFAULT_SUFFIXES,
                   dry_run: bool = False) -> List[Path]:
    """Copy new/changed light files from src to dst. Immutable config snapshots that already exist
    locally are never overwritten. Returns the destination paths that were (or would be) written."""
    src, dst = Path(src), Path(dst)
    if not src.is_dir():
        raise FileNotFoundError(f"Source folder not found: {src}")
    copied = []
    for path in sorted(src.rglob("*")):
        rel = path.relative_to(src)
        if not path.is_file() or path.suffix.lower() not in suffixes or EXCLUDED_DIRS & set(rel.parts):
            continue
        target = dst / rel
        if target.exists():
            if filecmp.cmp(path, target, shallow=False):
                continue
            if rel.name.startswith(("config_", "provenance_")):
                print(f"[Sync] WARNING: immutable file differs, not overwriting: {target}")
                continue
        copied.append(target)
        if not dry_run:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
    return copied
