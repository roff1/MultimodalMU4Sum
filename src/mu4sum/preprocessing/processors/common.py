"""Pieces shared by every dataset processor: the registry, the record contract, record filters and splits."""
import random
from collections import Counter
from typing import List, Optional

from omegaconf import DictConfig

from mu4sum.utils.registry import Registry

PROCESSORS = Registry("processor")
RECORD_KEYS = ("id", "image_types", "image_paths", "text")
IMAGE_FILTER_MODES = ("none", "count", "count_and_type")


def _images_allowed(image_types: List[str], rule: DictConfig) -> bool:
    """Apply `dataset.filtering.images`: no filter, exact number of images, or exact number of images per type."""
    mode = rule.mode
    if mode not in IMAGE_FILTER_MODES:
        raise ValueError(f"Unknown dataset.filtering.images.mode '{mode}'; choose from {IMAGE_FILTER_MODES}")
    if mode == "none":
        return True
    if mode == "count":
        if rule.get("count") is None:
            raise ValueError("dataset.filtering.images.count is required when the mode is 'count'")
        return len(image_types) == int(rule.count)
    if not rule.get("types"):
        raise ValueError("dataset.filtering.images.types (image type -> how many) is required when the mode "
                         "is 'count_and_type'")
    return Counter(image_types) == Counter({t: int(n) for t, n in rule.types.items() if int(n) > 0})


def rejection_reason(record: dict, filtering: DictConfig) -> Optional[str]:
    """Why a record must be dropped according to the dataset's `filtering` section (None = keep it)."""
    if len(record["text"]) < int(filtering.min_text_chars):
        return "text_too_short"
    if not _images_allowed(record["image_types"], filtering.images):
        return "images_filtered_out"
    return None


def check_records(records: List[dict], processor: str) -> None:
    """Fail loudly when a processor breaks the record contract (keys, one type per image, unique ids)."""
    seen = set()
    for r in records:
        missing = [k for k in RECORD_KEYS if k not in r]
        if missing:
            raise ValueError(f"Processor '{processor}' produced a record without {missing}")
        if len(r["image_types"]) != len(r["image_paths"]):
            raise ValueError(f"Record {r['id']}: image_types and image_paths must have the same length")
        if r["id"] in seen:
            raise ValueError(f"Processor '{processor}' produced the id {r['id']} twice")
        seen.add(r["id"])


def assign_splits(records: List[dict], fractions: DictConfig, seed: int) -> List[dict]:
    """Deterministic record-level split (a record's images never end up in different splits)."""
    names = list(fractions.keys())
    total = sum(float(fractions[n]) for n in names)
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"dataset.splits must sum to 1.0, got {total}")
    ids = sorted(r["id"] for r in records)
    random.Random(seed).shuffle(ids)
    n, start, owner = len(ids), 0, {}
    for i, name in enumerate(names):
        size = n - start if i == len(names) - 1 else int(n * float(fractions[name]))
        for rid in ids[start:start + size]:
            owner[rid] = name
        start += size
    return [{**r, "split": owner[r["id"]]} for r in sorted(records, key=lambda r: r["id"])]
