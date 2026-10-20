"""Student-side views of the final processed dataset (training and evaluation): one sample per record."""
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from omegaconf import DictConfig

from mu4sum.config.paths import MANIFEST, RECORDS_FILE, processed_dir
from mu4sum.preprocessing.answers import format_answer
from mu4sum.preprocessing.augment import load_images
from mu4sum.preprocessing.download import image_root
from mu4sum.preprocessing.prompts import aspect_keys, build_student_prompt
from mu4sum.utils.io import read_json, read_jsonl


def load_processed(cfg: DictConfig) -> Tuple[Dict[str, List[dict]], Path]:
    """Load the final records.jsonl grouped by split, and the root their relative image paths refer to.

    Which stages produced it (process only, or process + augment) makes no difference here. The split is the
    key of the result, so it is not repeated inside the records."""
    directory = processed_dir(cfg)
    manifest_path, records_path = directory / MANIFEST, directory / RECORDS_FILE
    if not manifest_path.exists() or not records_path.exists():
        raise FileNotFoundError(f"Processed dataset not found in {directory}. Run scripts/run_preprocess.py first.")
    manifest = read_json(manifest_path)
    if not manifest.get("complete"):
        print(f"[Data] WARNING: processed dataset in {directory} is incomplete "
              f"({manifest['num_records']} records, {manifest['num_failed']} failed).")
    splits: Dict[str, List[dict]] = {}
    for r in read_jsonl(records_path):
        splits.setdefault(r["split"], []).append({k: v for k, v in r.items() if k != "split"})
    return splits, image_root(cfg)


class RecordDataset:
    """Map-style dataset (works with torch DataLoader): one sample per record, MU4Sum style.

    The input is the prompt built from (text, image_types, image_paths) asking for the `aspects` listed (all of
    them when None) plus the images; the target is one aspect-summary block per aspect (see
    mu4sum.preprocessing.answers). Retain and forget sets are two views of the same records with disjoint
    aspects. `target_override` replaces every summary with a fixed text. Images are opened lazily.
    """

    def __init__(self, split, image_root: Path, cfg: DictConfig,
                 aspects: Optional[Sequence[str]] = None, limit: Optional[int] = None,
                 target_override: Optional[str] = None):
        keys = aspect_keys(cfg)
        self.aspects = list(keys if aspects is None else aspects)
        unknown = [a for a in self.aspects if a not in keys]
        if unknown:
            raise ValueError(f"Unknown aspect(s) {unknown}; dataset aspects are {keys}")
        self.split, self.image_root, self.cfg, self.target_override = split, Path(image_root), cfg, target_override
        self.n = len(split) if limit is None else min(len(split), int(limit))

    def __len__(self) -> int:
        return self.n

    def meta(self, i: int) -> dict:
        """id, aspects, reference summaries, target answer text and its aspect spans, without images."""
        record = self.split[i]
        by_aspect = dict(record["aspect_summaries"])
        if self.target_override is not None:
            by_aspect = {a: self.target_override for a in self.aspects}
        target, spans = format_answer(by_aspect, self.aspects)
        return {"id": record["id"], "aspects": self.aspects,
                "references": {a: by_aspect[a] for a in self.aspects}, "target": target, "spans": spans}

    def __getitem__(self, i: int) -> dict:
        """Return `meta(i)` plus the prompt and the loaded RGB images of record i."""
        record = self.split[i]
        prompt = build_student_prompt(self.cfg, record["text"], record["image_types"], record["image_paths"],
                                      self.aspects)
        return {**self.meta(i), "prompt": prompt, "images": load_images(self.image_root, record["image_paths"])}


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
