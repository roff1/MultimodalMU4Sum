"""Stage orchestration: download -> process -> augment, each skipped when its artifacts are valid.

An artifact is valid when its manifest fingerprint matches the current config. If it exists but the
config changed, we refuse to silently reuse (or overwrite) it: pass force=True to recompute.
"""
import shutil
from pathlib import Path
from typing import Dict, Iterable, Optional

from omegaconf import DictConfig

from mu4sum.config.loader import config_fingerprint
from mu4sum.config.paths import hybrid_dir, processed_dir, raw_dir
from mu4sum.preprocessing.augment import FAILURES_FILE, RECORDS_FILE, augment_studies
from mu4sum.preprocessing.download import _to_stored, download_dataset, raw_root
from mu4sum.preprocessing.processors import build_studies
from mu4sum.preprocessing.teacher import TeacherBackend
from mu4sum.utils.io import read_json, read_jsonl, write_json_atomic, write_jsonl
from mu4sum.utils.reproducibility import fingerprint

STAGES = ("download", "process", "augment")
STUDIES_FILE = "studies.jsonl"
MANIFEST = "manifest.json"
PROGRESS = "progress.json"
HF_DIRNAME = "hf"


class StaleArtifactError(RuntimeError):
    """Raised when an existing artifact was built with a different config and force is not set."""
    pass


def process_fingerprint(cfg: DictConfig) -> str:
    """Fingerprint of the config sections (dataset, seed) that determine the processed studies."""
    return config_fingerprint(cfg, "dataset", "seed")


def augment_fingerprint(cfg: DictConfig) -> str:
    """Fingerprint of everything that determines the teacher-generated data, upstream studies included."""
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
    """Build split studies.jsonl and manifest from the raw data; skipped if the manifest is current."""
    out = processed_dir(cfg)
    manifest_path = out / MANIFEST
    fp = process_fingerprint(cfg)
    if _check_fresh(manifest_path, fp, "Processed studies", force):
        print(f"[Process] Up to date ({out / STUDIES_FILE}), skipping.")
        return out

    root = raw_root(raw_dir(cfg), Path(cfg.runtime.data_root))
    if root is None:
        raise FileNotFoundError("Raw data not downloaded yet: run the 'download' stage first.")
    studies = build_studies(cfg, root)
    write_jsonl(out / STUDIES_FILE, studies)
    split_counts: Dict[str, int] = {}
    for s in studies:
        split_counts[s["split"]] = split_counts.get(s["split"], 0) + 1
    write_json_atomic(manifest_path, {
        "fingerprint": fp,
        "image_root": _to_stored(root, Path(cfg.runtime.data_root)),
        "num_studies": len(studies),
        "num_images": sum(len(s["image_paths"]) for s in studies),
        "splits": split_counts,
    })
    print(f"[Process] {len(studies)} studies {split_counts} -> {out / STUDIES_FILE}")
    return out


def stage_augment(cfg: DictConfig, backend: Optional[TeacherBackend] = None, force: bool = False) -> Path:
    """Generate teacher summaries into the hybrid folder, resuming partial runs; skipped once complete.

    Builds the HF dataset and manifest at the end; force=True deletes the hybrid folder first.
    """
    if cfg.get("model") is None:
        raise ValueError("The 'augment' stage needs a teacher model preset (--model).")
    out = hybrid_dir(cfg)
    manifest_path = out / MANIFEST
    fp = augment_fingerprint(cfg)

    if force and out.exists():
        shutil.rmtree(out)
    if manifest_path.exists() and _check_fresh(manifest_path, fp, "Hybrid dataset", False) \
            and read_json(manifest_path).get("complete"):
        print(f"[Augment] Up to date ({out / HF_DIRNAME}), skipping.")
        return out

    progress_path = out / PROGRESS
    if progress_path.exists() and read_json(progress_path)["fingerprint"] != fp:
        raise StaleArtifactError(f"Partial hybrid data in {out} comes from a different configuration. "
                                 f"Re-run with --force to start over.")
    out.mkdir(parents=True, exist_ok=True)
    write_json_atomic(progress_path, {"fingerprint": fp})

    proc_dir = processed_dir(cfg)
    if not (proc_dir / MANIFEST).exists():
        raise FileNotFoundError("Studies not processed yet: run the 'process' stage first.")
    proc_manifest = read_json(proc_dir / MANIFEST)
    image_root = Path(proc_manifest["image_root"])
    if not image_root.is_absolute():
        image_root = Path(cfg.runtime.data_root) / image_root
    studies = list(read_jsonl(proc_dir / STUDIES_FILE))

    if backend is None:
        from mu4sum.preprocessing.teacher import HFTeacher

        backend = HFTeacher(cfg.model)
    counters = augment_studies(cfg, studies, image_root, out, backend)
    print(f"[Augment] {counters}")

    _finalize(out, fp, counters, proc_manifest["image_root"])
    return out


def _finalize(out: Path, fp: str, counters: Dict[str, int], image_root: str) -> None:
    """Convert records.jsonl into a per-split HF dataset and write the manifest (complete if none pending)."""
    from datasets import Dataset, DatasetDict

    records = list(read_jsonl(out / RECORDS_FILE))
    by_split: Dict[str, list] = {}
    for r in records:
        by_split.setdefault(r["split"], []).append(r)
    if by_split:
        dd = DatasetDict({name: Dataset.from_list(rows) for name, rows in sorted(by_split.items())})
        hf_path = out / HF_DIRNAME
        if hf_path.exists():
            shutil.rmtree(hf_path)
        dd.save_to_disk(str(hf_path))
    write_json_atomic(out / MANIFEST, {
        "fingerprint": fp,
        "complete": counters["pending"] == 0,
        "image_root": image_root,
        "num_records": len(records),
        "num_failed": counters["failed"],
        "splits": {k: len(v) for k, v in by_split.items()},
    })


def run_preprocess(cfg: DictConfig, stages: Iterable[str] = STAGES, force: bool = False,
                   backend: Optional[TeacherBackend] = None) -> None:
    """Run the requested stages in the order download, process, augment."""
    stages = list(stages)
    unknown = [s for s in stages if s not in STAGES]
    if unknown:
        raise ValueError(f"Unknown stage(s) {unknown}; choose from {STAGES}")
    if "download" in stages:
        download_dataset(cfg, raw_dir(cfg), force=force)
    if "process" in stages:
        stage_process(cfg, force=force)
    if "augment" in stages:
        stage_augment(cfg, backend=backend, force=force)
