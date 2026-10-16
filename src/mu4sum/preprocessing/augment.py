"""Hybrid-data generation: run the teacher over every record and persist its aspect summaries.

Each hybrid record is `{id, image_types, image_paths, text, aspect_summaries}`, where `text` is the original record
text (the teacher never rewrites it). Records are appended to a JSONL file one by one
(plus the `split` they belong to), so an interrupted run (Colab disconnect) resumes exactly where it stopped.
Records whose JSON stays invalid after all retries go to failures.jsonl.
"""
from pathlib import Path
from typing import Callable, Dict, List, Optional

from omegaconf import DictConfig
from tqdm import tqdm

from mu4sum.preprocessing.prompts import aspect_keys, build_teacher_prompt
from mu4sum.preprocessing.teacher import TeacherBackend, TeacherFailure, generate_aspect_summaries
from mu4sum.utils.io import append_jsonl, read_jsonl

RECORDS_FILE = "records.jsonl"
FAILURES_FILE = "failures.jsonl"


def load_images(image_root: Path, rel_paths: List[str]) -> list:
    """Open images (paths relative to image_root) and return them as RGB PIL images."""
    from PIL import Image

    images = []
    for rel in rel_paths:
        with Image.open(Path(image_root) / rel) as img:
            images.append(img.convert("RGB"))
    return images


def augment_records(
    cfg: DictConfig,
    records: List[dict],
    image_root: Path,
    out_dir: Path,
    backend: TeacherBackend,
    image_loader: Callable[[Path, List[str]], list] = load_images,
) -> Dict[str, int]:
    """Process pending records. Returns counters; `pending` > 0 means the run is not complete."""
    out_dir = Path(out_dir)
    records_path, failures_path = out_dir / RECORDS_FILE, out_dir / FAILURES_FILE
    done = {r["id"] for r in read_jsonl(records_path)}
    failed = {r["id"] for r in read_jsonl(failures_path)}

    keys = aspect_keys(cfg)
    limit: Optional[int] = cfg.augmentation.limit

    pending = [r for r in records if r["id"] not in done and r["id"] not in failed]
    if limit is not None:
        pending = pending[: max(int(limit) - len(done), 0)]

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

    total_done = len(done) + n_new
    total_failed = len(failed) + n_failed
    return {"done": total_done, "failed": total_failed, "new": n_new,
            "pending": len(records) - total_done - total_failed}
