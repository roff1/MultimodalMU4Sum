"""Student-side views of the hybrid dataset (training and evaluation): one sample per study."""
from pathlib import Path
from typing import Optional, Sequence, Tuple

from omegaconf import DictConfig

from mu4sum.config.paths import hybrid_dir
from mu4sum.preprocessing.answers import format_answer
from mu4sum.preprocessing.augment import load_images
from mu4sum.preprocessing.pipeline import HF_DIRNAME, MANIFEST
from mu4sum.preprocessing.prompts import aspect_keys, build_student_prompt
from mu4sum.utils.io import read_json


def load_hybrid(cfg: DictConfig) -> Tuple["DatasetDict", Path]:  # noqa: F821
    """Load the teacher-generated dataset and the root its relative image paths refer to."""
    from datasets import load_from_disk

    directory = hybrid_dir(cfg)
    manifest_path = directory / MANIFEST
    if not manifest_path.exists() or not (directory / HF_DIRNAME).exists():
        raise FileNotFoundError(f"Hybrid dataset not found in {directory}. Run scripts/run_preprocess.py first.")
    manifest = read_json(manifest_path)
    if not manifest.get("complete"):
        print(f"[Data] WARNING: hybrid dataset in {directory} is incomplete "
              f"({manifest['num_records']} records, {manifest['num_failed']} failed).")
    image_root = Path(manifest["image_root"])
    if not image_root.is_absolute():
        image_root = Path(cfg.runtime.data_root) / image_root
    return load_from_disk(str(directory / HF_DIRNAME)), image_root


class StudyDataset:
    """Map-style dataset (works with torch DataLoader): one sample per study, MU4Sum style.

    The prompt asks for the `aspects` listed (all of them when None) and the target is one aspect-summary
    block per aspect (see mu4sum.preprocessing.answers). Retain and forget sets are two views of the same
    studies with disjoint aspects. `target_override` replaces every summary with a fixed text. Images are
    opened lazily.
    """

    def __init__(self, split, image_root: Path, cfg: DictConfig,
                 aspects: Optional[Sequence[str]] = None, limit: Optional[int] = None,
                 target_override: Optional[str] = None):
        keys = aspect_keys(cfg)
        self.aspects = list(keys if aspects is None else aspects)
        unknown = [a for a in self.aspects if a not in keys]
        if unknown:
            raise ValueError(f"Unknown aspect(s) {unknown}; dataset aspects are {keys}")
        self.split, self.image_root, self.target_override = split, Path(image_root), target_override
        self.prompt = build_student_prompt(cfg, self.aspects)
        self.n = len(split) if limit is None else min(len(split), int(limit))

    def __len__(self) -> int:
        return self.n

    def meta(self, i: int) -> dict:
        """study_id, aspects, reference summaries, target answer text and its aspect spans, without images."""
        record = self.split[i]
        by_aspect = dict(zip(record["aspects"], record["aspect_summaries"]))
        if self.target_override is not None:
            by_aspect = {a: self.target_override for a in self.aspects}
        target, spans = format_answer(by_aspect, self.aspects)
        return {"study_id": record["study_id"], "aspects": self.aspects,
                "references": {a: by_aspect[a] for a in self.aspects}, "target": target, "spans": spans}

    def __getitem__(self, i: int) -> dict:
        """Return `meta(i)` plus the prompt and the loaded RGB images of study i."""
        record = self.split[i]
        return {**self.meta(i), "prompt": self.prompt,
                "images": load_images(self.image_root, record["image_paths"])}


def split_aspects(cfg: DictConfig) -> Tuple[list, list]:
    """(forget_aspects, retain_aspects) from the dataset preset: retain = every aspect not forgotten."""
    keys = aspect_keys(cfg)
    forget = list(cfg.get("forget_aspects") or [])
    unknown = [a for a in forget if a not in keys]
    if unknown:
        raise ValueError(f"forget_aspects {unknown} not in dataset aspects {keys}")
    retain = [a for a in keys if a not in forget]
    if not forget or not retain:
        raise ValueError("Unlearning needs at least one forget aspect and one retain aspect: "
                         "set `forget_aspects` in the dataset preset to a proper subset of its aspects")
    return forget, retain
