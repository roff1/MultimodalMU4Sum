"""Single source of truth for every on-disk location, derived from the resolved config."""
from pathlib import Path

from omegaconf import DictConfig


def _slug(name) -> str:
    return str(name).replace("/", "__")


def raw_dir(cfg: DictConfig) -> Path:
    """Folder with the raw downloaded dataset."""
    return Path(cfg.runtime.data_root) / "raw" / cfg.dataset.name


def processed_dir(cfg: DictConfig) -> Path:
    """Folder with the processed (normalized) dataset."""
    return Path(cfg.runtime.data_root) / "processed" / cfg.dataset.name


def hybrid_dir(cfg: DictConfig) -> Path:
    """Teacher-generated aspect summaries (one folder per teacher preset)."""
    return processed_dir(cfg) / f"hybrid__{_slug(cfg.run.model)}"


def run_dir(cfg: DictConfig, method: str = None, tag: str = "__cfg__") -> Path:
    """outputs/<model>/<dataset>/<method>[/<tag>]"""
    method = method or cfg.run.method
    tag = cfg.run.tag if tag == "__cfg__" else tag
    path = Path(cfg.runtime.output_root) / cfg.run.model / cfg.run.dataset / method
    return path / tag if tag else path


def finetune_dir(cfg: DictConfig) -> Path:
    """Run folder of the fine-tuned model that unlearning starts from."""
    return run_dir(cfg, "finetune", cfg.run.finetune_tag)


def checkpoints_dir(directory) -> Path:
    """Checkpoints sub-folder of a run directory."""
    return Path(directory) / "checkpoints"


def eval_dir(cfg: DictConfig, method: str = None, tag: str = "__cfg__") -> Path:
    """<run_dir>/evaluation/<eval_tag>; pass method/tag to address another method's evaluation."""
    return run_dir(cfg, method, tag) / "evaluation" / str(cfg.run.eval_tag)
