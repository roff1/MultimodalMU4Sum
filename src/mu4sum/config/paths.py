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
    """Teacher-generated aspect summaries (one folder per teacher preset, whatever student uses them)."""
    return processed_dir(cfg) / f"hybrid__{_slug(cfg.run.teacher)}"


def run_dir(cfg: DictConfig, method: str = None, tag: str = "__cfg__") -> Path:
    """outputs/<model>/<dataset>[__teacher-<teacher>]/<method>[/<tag>]; the teacher is named only when it is
    not the model itself, so runs on data from different teachers never share a folder."""
    method = method or cfg.run.method
    tag = cfg.run.tag if tag == "__cfg__" else tag
    dataset = cfg.run.dataset
    if cfg.run.teacher != cfg.run.model:
        dataset = f"{dataset}__teacher-{_slug(cfg.run.teacher)}"
    path = Path(cfg.runtime.output_root) / cfg.run.model / dataset / method
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
