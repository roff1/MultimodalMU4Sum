"""Stage orchestration: download -> process -> augment, each skipped when its artifacts are valid.

Every stage writes into the same processed folder. The last stage of the dataset's pipeline (`augment`, or
`process` when `augmentation.enabled` is false) writes the final data under the standard names (records.jsonl,
manifest.json, progress.json) that training and evaluation read; earlier stages prefix theirs
(process_records.jsonl, process_manifest.json), see `config.paths.processed_file`.

An artifact is valid when its manifest fingerprint matches the current config. If it exists but the
config changed, we refuse to silently reuse (or overwrite) it: pass force=True to recompute.
"""
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from omegaconf import DictConfig

from mu4sum.config import start_run
from mu4sum.config.loader import config_fingerprint
from mu4sum.config.paths import (FAILURES_FILE, MANIFEST, PROGRESS, RECORDS_FILE, processed_dir, processed_file,
                                 raw_dir)
from mu4sum.preprocessing.augment import augment_records, plan_augmentation
from mu4sum.preprocessing.download import download_dataset, image_root
from mu4sum.preprocessing.processors import build_records
from mu4sum.preprocessing.teacher import TeacherBackend
from mu4sum.utils.io import read_json, read_jsonl, write_json_atomic, write_jsonl
from mu4sum.utils.reproducibility import fingerprint

STAGES = ("download", "process", "augment")


class StaleArtifactError(RuntimeError):
    """Raised when an existing artifact was built with a different config and force is not set."""
    pass


def default_stages(cfg: DictConfig) -> List[str]:
    """Every stage the dataset needs to reach its final data (augment only when `augmentation.enabled`)."""
    return [*STAGES[:2], *(["augment"] if cfg.augmentation.enabled else [])]


def process_fingerprint(cfg: DictConfig) -> str:
    """Fingerprint of the config sections that determine the processed records."""
    # Without augmentation the processed records are the final data, so the aspects they carry matter too.
    sections = ("dataset", "seed") if cfg.augmentation.enabled else ("dataset", "seed", "aspects")
    return config_fingerprint(cfg, *sections)


def augment_fingerprint(cfg: DictConfig) -> str:
    """Fingerprint of everything that determines the teacher-generated data, upstream processed records included."""
    # `augmentation.limit` is deliberately excluded: it only caps how much work a run does.
    core = config_fingerprint(cfg, "aspects", "prompt", "model", "generation")
    return fingerprint([core, process_fingerprint(cfg)])


def _check_fresh(manifest_path: Path, expected: str, what: str, force: bool) -> bool:
    """True if a valid artifact exists. Raises if one exists but was built with another config."""
    if force or not manifest_path.exists():
        return False
    if read_json(manifest_path).get("fingerprint") != expected:
        raise StaleArtifactError(f"{what} in {manifest_path.parent} was built with a different configuration. "
                                 f"Re-run with --force to rebuild it.")
    return True


def stage_process(cfg: DictConfig, force: bool = False) -> Path:
    """Build the processed records (with their split) from the raw data; skipped if current.

    When `process` is the final stage the output is the final data (standard names),
    otherwise it is the intermediate `process_records.jsonl` that `augment` reads."""
    out = processed_dir(cfg)
    manifest_path = processed_file(cfg, "process", MANIFEST)
    records_path = processed_file(cfg, "process", RECORDS_FILE)
    fp = process_fingerprint(cfg)
    if _check_fresh(manifest_path, fp, "Processed records", force):
        print(f"[Process] Up to date ({records_path}), skipping.")
        return out

    records = build_records(cfg, image_root(cfg))
    write_jsonl(records_path, records)
    split_counts: Dict[str, int] = {}
    for r in records:
        split_counts[r["split"]] = split_counts.get(r["split"], 0) + 1
    if cfg.augmentation.enabled:
        write_json_atomic(manifest_path, {
            "fingerprint": fp,
            "num_records": len(records),
            "num_images": sum(len(r["image_paths"]) for r in records),
            "splits": split_counts,
        })
    else:
        _finalize(cfg, fp, {"failed": 0, "pending": 0})
    print(f"[Process] {len(records)} records {split_counts} -> {records_path}")
    return out


def stage_augment(cfg: DictConfig, backend: Optional[TeacherBackend] = None, force: bool = False) -> Path:
    """Generate teacher summaries from the processed records into the final data, resuming partial runs.

    The stage is skipped (model not even loaded) when no record is left to process under the current
    `augmentation.limit` and the manifest already reflects the stored records. This is independent of the
    manifest's `complete` flag, which only says whether the whole dataset has been augmented.
    Writes the manifest at the end; force=True deletes the previous augmented data first
    (the intermediate process files are kept).
    """
    if not cfg.augmentation.enabled:
        raise ValueError("The 'augment' stage is disabled for this dataset (`augmentation.enabled` is false).")
    if cfg.get("model") is None:
        raise ValueError("The 'augment' stage needs a teacher model preset (--model).")
    out = processed_dir(cfg)
    manifest_path = processed_file(cfg, "augment", MANIFEST)
    progress_path = processed_file(cfg, "augment", PROGRESS)
    fp = augment_fingerprint(cfg)

    if force:
        _clear_final(out)
    has_manifest = manifest_path.exists() and _check_fresh(manifest_path, fp, "Augmented dataset", False)

    if progress_path.exists() and read_json(progress_path)["fingerprint"] != fp:
        raise StaleArtifactError(f"Partial augmented data in {out} comes from a different configuration. "
                                 f"Re-run with --force to start over.")
    write_json_atomic(progress_path, {"fingerprint": fp})

    if not processed_file(cfg, "process", MANIFEST).exists():
        raise FileNotFoundError("Dataset not processed yet: run the 'process' stage first.")
    records = list(read_jsonl(processed_file(cfg, "process", RECORDS_FILE)))

    todo, counters = plan_augmentation(cfg, records, out)
    if not todo:
        if has_manifest:
            m = read_json(manifest_path)
            if m.get("num_records") == counters["done"] and m.get("num_failed") == counters["failed"]:
                print(f"[Augment] Nothing left to do ({counters['done']} done, {counters['failed']} failed, "
                      f"{counters['pending']} beyond the limit), skipping.")
                return out
        print(f"[Augment] {counters}")
    else:
        if backend is None:
            from mu4sum.preprocessing.teacher import HFTeacher

            backend = HFTeacher(cfg.model)
        counters = augment_records(cfg, records, image_root(cfg), out, backend)
        print(f"[Augment] {counters}")

    _finalize(cfg, fp, counters)
    return out


def _clear_final(out: Path) -> None:
    """Delete the final data (standard names) of the processed folder, leaving the intermediate files alone."""
    for name in (RECORDS_FILE, FAILURES_FILE, MANIFEST, PROGRESS):
        (out / name).unlink(missing_ok=True)


def _finalize(cfg: DictConfig, fp: str, counters: Dict[str, int]) -> None:
    """Write the manifest of the final records.jsonl (complete if no record is pending)."""
    out = processed_dir(cfg)
    splits: Dict[str, int] = {}
    for r in read_jsonl(out / RECORDS_FILE):
        splits[r["split"]] = splits.get(r["split"], 0) + 1
    write_json_atomic(out / MANIFEST, {
        "fingerprint": fp,
        "teacher": cfg.run.model if cfg.augmentation.enabled else None,
        "complete": counters["pending"] == 0,
        "num_records": sum(splits.values()),
        "num_failed": counters["failed"],
        "splits": splits,
    })


def run_preprocess(cfg: DictConfig, stages: Optional[Iterable[str]] = None, force: bool = False,
                   backend: Optional[TeacherBackend] = None) -> None:
    """Run the requested stages (default: all the dataset needs) in the order download, process, augment."""
    stages = default_stages(cfg) if stages is None else list(stages)
    unknown = [s for s in stages if s not in STAGES]
    if unknown:
        raise ValueError(f"Unknown stage(s) {unknown}; choose from {STAGES}")
    start_run(cfg, processed_dir(cfg), "preprocess", versioned=True)
    if "download" in stages:
        download_dataset(cfg, raw_dir(cfg), force=force)
    if "process" in stages:
        stage_process(cfg, force=force)
    if "augment" in stages:
        stage_augment(cfg, backend=backend, force=force)
