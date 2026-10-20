"""Hybrid-data generation: run the teacher over every record and persist its aspect summaries.

Each hybrid record is `{id, image_types, image_paths, text, aspect_summaries}`, where `text` is the original record
text (the teacher never rewrites it). Records are appended to a JSONL file one by one
(plus the `split` they belong to), so an interrupted run (Colab disconnect) resumes exactly where it stopped.
Records whose JSON stays invalid after all retries go to failures.jsonl.
"""
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from omegaconf import DictConfig
from tqdm import tqdm

from mu4sum.config.paths import FAILURES_FILE, RECORDS_FILE
from mu4sum.preprocessing.prompts import aspect_keys, build_teacher_prompt
from mu4sum.preprocessing.teacher import TeacherBackend, TeacherFailure, generate_aspect_summaries
from mu4sum.utils.io import append_jsonl, read_jsonl


def load_images(image_root: Path, rel_paths: List[str]) -> list:
    """Open images (paths relative to image_root) and return them as RGB PIL images."""
    from PIL import Image

    images = []
    for rel in rel_paths:
        with Image.open(Path(image_root) / rel) as img:
            images.append(img.convert("RGB"))
    return images


def _counters(total: int, done: int, failed: int, new: int = 0) -> Dict[str, int]:
    return {"done": done, "failed": failed, "new": new, "pending": total - done - failed}


def plan_augmentation(cfg: DictConfig, records: List[dict], out_dir: Path) -> Tuple[List[dict], Dict[str, int]]:
    """Records the teacher still has to process in this run (`augmentation.limit` applied), plus the current counters.

    Cheap (reads only the ids already stored), so callers can tell whether the run has work left before loading
    the model. An empty list means the augmentation is finished for the current limit, even if the dataset is not
    complete (`counters["pending"]` > 0)."""
    out_dir = Path(out_dir)
    done = {r["id"] for r in read_jsonl(out_dir / RECORDS_FILE)}
    failed = {r["id"] for r in read_jsonl(out_dir / FAILURES_FILE)}
    pending = [r for r in records if r["id"] not in done and r["id"] not in failed]
    limit: Optional[int] = cfg.augmentation.limit
    if limit is not None:
        pending = pending[: max(int(limit) - len(done), 0)]
    return pending, _counters(len(records), len(done), len(failed))


def augment_records(
    cfg: DictConfig,
    records: List[dict],
    image_root: Path,
    out_dir: Path,
    backend: TeacherBackend,
    image_loader: Callable[[Path, List[str]], list] = load_images,
) -> Dict[str, int]:
    """Process pending records. Returns counters; `pending` > 0 means the dataset is not complete."""
    out_dir = Path(out_dir)
    records_path, failures_path = out_dir / RECORDS_FILE, out_dir / FAILURES_FILE
    pending, before = plan_augmentation(cfg, records, out_dir)
    keys = aspect_keys(cfg)

    n_new = n_failed = 0
    for record in tqdm(pending, desc="Teacher", unit="record"):
        images = image_loader(image_root, record["image_paths"])
        prompt = build_teacher_prompt(cfg, record["text"], record["image_types"], record["image_paths"])
        try:
            summaries, _ = generate_aspect_summaries(backend, images, prompt, keys, cfg.generation)
        except TeacherFailure as e:
            append_jsonl(failures_path, {"id": record["id"], "errors": e.errors})
            n_failed += 1
            continue
        append_jsonl(records_path, {
            "id": record["id"],
            "image_types": record["image_types"],
            "image_paths": record["image_paths"],
            "text": record["text"],
            "aspect_summaries": {k: summaries[k] for k in keys},
            "split": record["split"],
        })
        n_new += 1

    return _counters(len(records), before["done"] + n_new, before["failed"] + n_failed, n_new)
