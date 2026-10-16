"""Processor of the IU X-Ray dataset (chest radiographs with their reports).

The raw data is a CSV with one row per image (`image_id`, `projection`, `org_caption`). The exam id is the part of
the image file name before the first "_"; rows sharing it form one record holding every image of the exam (e.g.
Frontal + Lateral, the projection being its image type); the text is the report caption.
"""
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from omegaconf import DictConfig

from mu4sum.preprocessing.processors.common import PROCESSORS, rejection_reason

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _is_empty(value) -> bool:
    return value is None or (isinstance(value, float) and pd.isna(value)) or not str(value).strip()


def _exam_id(image_name) -> str:
    """Exam id: the part of the image file name before the first underscore."""
    return str(image_name).strip().split("_", 1)[0]


def _image_index(images_dir: Path) -> Dict[str, Path]:
    """Image files under images_dir by file name and by stem (the CSV may omit or change the extension)."""
    index: Dict[str, Path] = {}
    for path in sorted(images_dir.rglob("*")):
        if path.suffix.lower() in IMAGE_EXTENSIONS:
            index.setdefault(path.name, path)
            index.setdefault(path.stem, path)
    return index


def _resolve_image(images_dir: Path, name: str, index: Dict[str, Path]) -> Optional[Path]:
    direct = images_dir / name
    if direct.is_file():
        return direct
    return index.get(Path(name).name) or index.get(Path(name).stem)


@PROCESSORS.register("iu_xray")
def iu_xray(ds: DictConfig, raw_root: Path) -> List[dict]:
    """One record per exam; exams with a missing image file, or rejected by `dataset.filtering`, are dropped."""
    csv_path = raw_root / ds.csv_path
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV not found: {csv_path}. Fix `dataset.csv_path` (relative to {raw_root}).")
    images_dir = raw_root / str(ds.images.path).format(resolution=ds.images.resolution)
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Image folder not found: {images_dir}. Fix `dataset.images.path` / "
                                f"`dataset.images.resolution` (path relative to {raw_root}).")

    df = pd.read_csv(csv_path)
    cols = ds.columns
    wanted = [cols.image, cols.text]
    missing = [c for c in wanted if c not in df.columns]
    if missing:
        raise KeyError(f"Columns {missing} not found in {csv_path}. Actual columns: {list(df.columns)}. "
                       f"Fix `dataset.columns` in the dataset preset.")
    type_col = cols.get("image_type")
    if type_col and type_col not in df.columns:
        print(f"[Processor] WARNING: image type column '{type_col}' not found; image types will be unknown.")
        type_col = None

    index = _image_index(images_dir) # index all image files by filename and stem (filename without extension)
    dropped: Counter = Counter() # track the count of dropped records for reporting
    records: List[dict] = [] # store all successfully processed exam records
    
    df = df.assign(_exam=[_exam_id(name) for name in df[cols.image]])
    for exam_id, group in df.groupby("_exam", sort=True):
        found, seen = [], set()
        for _, row in group.iterrows():
            if _is_empty(row[cols.image]):
                continue
            path = _resolve_image(images_dir, str(row[cols.image]), index)
            if path is None:
                dropped["image_file_missing"] += 1
                continue
            rel = path.relative_to(raw_root).as_posix()
            if rel in seen:
                continue
            seen.add(rel)
            image_type = "" if not type_col or _is_empty(row[type_col]) else str(row[type_col])
            found.append((rel, image_type))
        if not found:
            dropped["record_without_images"] += 1
            continue

        captions = [str(v).strip() for v in group[cols.text] if not _is_empty(v)]
        record = {
            "id": str(exam_id),
            "image_types": [t for _, t in found],
            "image_paths": [p for p, _ in found],
            "text": captions[0] if captions else "",
        }
        reason = rejection_reason(record, ds.filtering)
        if reason:
            dropped[reason] += 1
            continue
        records.append(record)

    if dropped:
        print(f"[Processor] Dropped: {dict(dropped)}")
    if not records:
        raise RuntimeError(f"No valid record produced from {csv_path}; check dataset.columns / dataset.images / "
                           f"dataset.filtering.")
    return records
