"""Hybrid-data generation: run the teacher over every study and persist aspect summaries.

Records are appended to a JSONL file one by one, so an interrupted run (Colab disconnect) resumes
exactly where it stopped. Studies whose JSON stays invalid after all retries go to failures.jsonl.
"""
from pathlib import Path
from typing import Callable, Dict, List, Optional

from omegaconf import DictConfig
from tqdm import tqdm

from mu4sum.preprocessing.prompts import (EXPANDED_TEXT_KEY, aspect_keys, build_student_prompt, build_teacher_prompt,
                                 expansion_enabled)
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


def augment_studies(
    cfg: DictConfig,
    studies: List[dict],
    image_root: Path,
    out_dir: Path,
    backend: TeacherBackend,
    image_loader: Callable[[Path, List[str]], list] = load_images,
) -> Dict[str, int]:
    """Process pending studies. Returns counters; `pending` > 0 means the run is not complete."""
    out_dir = Path(out_dir)
    records_path, failures_path = out_dir / RECORDS_FILE, out_dir / FAILURES_FILE
    done = {r["study_id"] for r in read_jsonl(records_path)}
    failed = {r["study_id"] for r in read_jsonl(failures_path)}

    keys = aspect_keys(cfg)
    expand = expansion_enabled(cfg)
    student_prompt = build_student_prompt(cfg, keys)
    system = cfg.prompt.teacher_system
    limit: Optional[int] = cfg.augmentation.limit

    pending = [s for s in studies if s["study_id"] not in done and s["study_id"] not in failed]
    if limit is not None:
        pending = pending[: max(int(limit) - len(done), 0)]

    n_new = n_failed = 0
    for study in tqdm(pending, desc="Teacher", unit="study"):
        images = image_loader(image_root, study["image_paths"])
        teacher_prompt = build_teacher_prompt(cfg, study["report_text"], study["image_views"])
        try:
            summaries, expanded, meta = generate_aspect_summaries(
                backend, images, system, teacher_prompt, keys, cfg.generation, expand)
        except TeacherFailure as e:
            append_jsonl(failures_path, {"study_id": study["study_id"], "errors": e.errors})
            n_failed += 1
            continue
        record = {
            "study_id": study["study_id"],
            "split": study["split"],
            "image_paths": study["image_paths"],
            "image_views": study["image_views"],
            "prompt": student_prompt,
            "aspects": keys,
            "aspect_summaries": [summaries[k] for k in keys],
            "report": study["report"],
            "teacher": {"attempts": meta["attempts"], "temperature": meta["temperature"]},
        }
        if expand:
            record[EXPANDED_TEXT_KEY] = expanded
        append_jsonl(records_path, record)
        n_new += 1

    total_done = len(done) + n_new
    total_failed = len(failed) + n_failed
    return {"done": total_done, "failed": total_failed, "new": n_new,
            "pending": len(studies) - total_done - total_failed}
