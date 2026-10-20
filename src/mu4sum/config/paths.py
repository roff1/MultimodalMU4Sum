"""Single source of truth for every on-disk location, derived from the resolved config."""
from pathlib import Path

from omegaconf import DictConfig


def raw_dir(cfg: DictConfig) -> Path:
    """Folder with the raw downloaded dataset."""
    return Path(cfg.runtime.data_root) / "raw" / cfg.dataset.name


RECORDS_FILE = "records.jsonl"
FAILURES_FILE = "failures.jsonl"
MANIFEST = "manifest.json"
PROGRESS = "progress.json"


def processed_dir(cfg: DictConfig) -> Path:
    """Folder holding every stage's output for the dataset: intermediate files and the final data."""
    return Path(cfg.runtime.data_root) / "processed" / cfg.dataset.name


def final_stage(cfg: DictConfig) -> str:
    """Last stage of the dataset's pipeline; its output is what training and evaluation read."""
    return "augment" if cfg.augmentation.enabled else "process"


def processed_file(cfg: DictConfig, stage: str, name: str) -> Path:
    """File `name` (RECORDS_FILE, MANIFEST, ...) written by `stage` in the processed folder.

    The final stage uses the standard name; an intermediate stage prefixes its own (process_records.jsonl),
    so training and evaluation always read the standard names whatever stages produced them."""
    prefix = "" if stage == final_stage(cfg) else f"{stage}_"
    return processed_dir(cfg) / f"{prefix}{name}"


def run_dir(cfg: DictConfig, method: str = None, tag: str = "__cfg__") -> Path:
    """outputs/<model>/<dataset>/<method>[/<tag>]."""
    method = method or cfg.run.method
    tag = cfg.run.tag if tag == "__cfg__" else tag
    path = Path(cfg.runtime.output_root) / cfg.run.model / cfg.run.dataset / method
    return path / tag if tag else path


def original_dir(cfg: DictConfig) -> Path:
    """Run folder of the Original model that finetune / default / unlearning methods start from."""
    return run_dir(cfg, "original", cfg.run.original_tag)


def checkpoints_dir(directory) -> Path:
    """Checkpoints sub-folder of a run directory."""
    return Path(directory) / "checkpoints"


def eval_dir(cfg: DictConfig, method: str = None, tag: str = "__cfg__") -> Path:
    """<run_dir>/evaluation/<eval_tag>; pass method/tag to address another method's evaluation."""
    return run_dir(cfg, method, tag) / "evaluation" / str(cfg.run.eval_tag)
