"""Raw dataset -> list of *study* records (a study groups a variable number of images + its report).

A processor is a function `(dataset_cfg, raw_root) -> list[dict]` registered in PROCESSORS. The
generic `grouped_csv` processor covers any dataset described by a CSV with one row per image, so
a new dataset of that shape only needs a YAML preset. Exotic formats register their own processor.

Study record:
    study_id: str
    image_paths: list[str]   # relative to the raw root, ordered by view_order
    image_views: list[str]   # same length as image_paths ("" when unknown)
    report: dict[str, str]   # report field -> text
    report_text: str
"""
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List

import pandas as pd
from omegaconf import DictConfig

from mu4sum.utils.registry import Registry

PROCESSORS = Registry("processor")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _is_empty(value) -> bool:
    return value is None or (isinstance(value, float) and pd.isna(value)) or not str(value).strip()


def _find_csv(raw_root: Path, filename: str) -> Path:
    matches = sorted(raw_root.rglob(filename))
    if not matches:
        raise FileNotFoundError(f"'{filename}' not found under {raw_root}")
    return matches[0]


def _build_image_index(raw_root: Path, images_cfg: DictConfig) -> Dict[str, Path]:
    search_dir = raw_root
    subdir = images_cfg.get("search_subdir")
    if subdir:
        candidate = raw_root / str(subdir).format(resolution=images_cfg.get("resolution"))
        if candidate.is_dir():
            search_dir = candidate
        else:
            print(f"[Processor] WARNING: image sub-folder {candidate} not found; searching all of {raw_root} "
                  f"(several resolutions may be mixed up).")
    index: Dict[str, Path] = {}
    for path in sorted(search_dir.rglob("*")):
        if path.suffix.lower() in IMAGE_EXTENSIONS:
            index.setdefault(path.name, path)
            index.setdefault(path.stem, path)
    return index


def _resolve_image(raw_root: Path, name: str, index: Dict[str, Path]):
    direct = raw_root / name
    if direct.is_file():
        return direct
    return index.get(Path(name).name) or index.get(Path(name).stem)


@PROCESSORS.register("grouped_csv")
def grouped_csv(ds: DictConfig, raw_root: Path) -> List[dict]:
    """Build studies from a one-row-per-image CSV, dropping missing images and too-short reports."""
    csv_path = _find_csv(raw_root, ds.csv_filename)
    df = pd.read_csv(csv_path)
    cols = ds.columns

    wanted = [cols.study_id, cols.image, *cols.report.values()]
    missing = [c for c in wanted if c not in df.columns]
    if missing:
        raise KeyError(f"Columns {missing} not found in {csv_path}. Actual columns: {list(df.columns)}. "
                       f"Fix `dataset.columns` in the dataset preset.")
    projection_col = cols.get("projection")
    if projection_col and projection_col not in df.columns:
        print(f"[Processor] WARNING: projection column '{projection_col}' not found; views will be unknown.")
        projection_col = None

    index = _build_image_index(raw_root, ds.images)
    view_order = list(ds.filtering.view_order)
    min_chars = int(ds.filtering.min_report_chars)
    dropped: Counter = Counter()
    studies: List[dict] = []

    for study_id, group in df.groupby(cols.study_id, sort=True):
        found, seen = [], set()
        for _, row in group.iterrows():
            if _is_empty(row[cols.image]):
                continue
            path = _resolve_image(raw_root, str(row[cols.image]), index)
            if path is None:
                dropped["image_file_missing"] += 1
                continue
            rel = path.relative_to(raw_root).as_posix()
            if rel in seen:
                continue
            seen.add(rel)
            view = "" if not projection_col or _is_empty(row[projection_col]) else str(row[projection_col])
            found.append((rel, view))
        if not found:
            dropped["study_without_images"] += 1
            continue
        found.sort(key=lambda rv: (view_order.index(rv[1]) if rv[1] in view_order else len(view_order), rv[0]))

        report = {}
        for field, column in cols.report.items():
            values = [str(v).strip() for v in group[column] if not _is_empty(v)]
            report[field] = values[0] if values else ""
        report_text = "\n".join(f"{k.upper()}: {v}" for k, v in report.items() if v)
        if len(report_text) < min_chars:
            dropped["report_too_short"] += 1
            continue

        studies.append({
            "study_id": str(study_id),
            "image_paths": [rv[0] for rv in found],
            "image_views": [rv[1] for rv in found],
            "report": report,
            "report_text": report_text,
        })

    if dropped:
        print(f"[Processor] Dropped: {dict(dropped)}")
    if not studies:
        raise RuntimeError(f"No valid study produced from {csv_path}; check dataset.columns / dataset.images.")
    return studies


def assign_splits(studies: List[dict], fractions: DictConfig, seed: int) -> List[dict]:
    """Deterministic study-level split (a study's views never end up in different splits)."""
    names = list(fractions.keys())
    total = sum(float(fractions[n]) for n in names)
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"dataset.splits must sum to 1.0, got {total}")
    ids = sorted(s["study_id"] for s in studies)
    random.Random(seed).shuffle(ids)
    n, start, owner = len(ids), 0, {}
    for i, name in enumerate(names):
        size = n - start if i == len(names) - 1 else int(n * float(fractions[name]))
        for sid in ids[start:start + size]:
            owner[sid] = name
        start += size
    return [{**s, "split": owner[s["study_id"]]} for s in sorted(studies, key=lambda s: s["study_id"])]


def build_studies(cfg: DictConfig, raw_root: Path) -> List[dict]:
    """Run the dataset's registered processor, then assign each study to a split."""
    studies = PROCESSORS.get(cfg.dataset.processor)(cfg.dataset, raw_root)
    return assign_splits(studies, cfg.dataset.splits, cfg.seed)
