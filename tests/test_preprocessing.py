import json

import pytest

from mu4sum.config.paths import hybrid_dir, processed_dir, raw_dir
from mu4sum.preprocessing.answers import display_name, parse_answer
from mu4sum.preprocessing.datasets import StudyDataset, load_hybrid, split_aspects
from mu4sum.preprocessing.pipeline import StaleArtifactError, run_preprocess
from mu4sum.preprocessing.prompts import (ASPECT_SUMMARIES_KEY, EXPANDED_TEXT_KEY, aspect_keys,
                                           expansion_enabled)
from mu4sum.utils.io import read_jsonl


class FakeTeacher:
    """Deterministic backend; optionally fails (returns garbage) for chosen call numbers."""

    def __init__(self, cfg, garbage_for_studies=()):
        self.keys, self.calls, self.garbage = list(cfg.aspects.names), 0, set(garbage_for_studies)
        self.expand = expansion_enabled(cfg)

    def generate(self, images, system, prompt, *, temperature, top_p, max_new_tokens):
        self.calls += 1
        if any(f"study {s}." in prompt for s in self.garbage):
            return "not json"
        out = {ASPECT_SUMMARIES_KEY: {k: f"summary of {k} ({len(images)} imgs)" for k in self.keys}}
        if self.expand:
            out[EXPANDED_TEXT_KEY] = f"expanded text ({len(images)} imgs)"
        return json.dumps(out)


def test_full_pipeline_groups_views_and_splits_by_study(make_cfg):
    cfg = make_cfg()
    teacher = FakeTeacher(cfg)
    run_preprocess(cfg, backend=teacher)

    studies = list(read_jsonl(processed_dir(cfg) / "studies.jsonl"))
    assert len(studies) == 20
    by_id = {s["study_id"]: s for s in studies}
    assert by_id["0"]["image_views"] == ["Frontal", "Lateral"]  # reordered, variable count per study
    assert by_id["1"]["image_views"] == ["Frontal"]
    assert by_id["0"]["report"]["impression"] == "No acute disease 0."
    assert "FINDINGS:" in by_id["0"]["report_text"]
    assert {s["split"] for s in studies} == {"train", "validation", "test"}

    splits, image_root = load_hybrid(cfg)
    ids = {name: {r["study_id"] for r in splits[name]} for name in splits}
    assert sum(len(v) for v in ids.values()) == 20
    assert not (ids["train"] & ids["test"]) and not (ids["train"] & ids["validation"])

    record = splits["train"][0]
    assert record["aspects"] == list(cfg.aspects.names)
    assert len(record["aspect_summaries"]) == 4
    assert record[EXPANDED_TEXT_KEY].startswith("expanded text")
    assert "Aspect 4/4: impression" in record["prompt"]
    assert not any(p.startswith("/") for p in record["image_paths"])  # relative paths
    assert (image_root / record["image_paths"][0]).exists()


def test_split_is_deterministic_across_runs(make_cfg, tmp_path):
    run_preprocess(make_cfg(), stages=["download", "process"])
    first = (processed_dir(make_cfg()) / "studies.jsonl").read_text()
    run_preprocess(make_cfg(), stages=["process"], force=True)
    assert (processed_dir(make_cfg()) / "studies.jsonl").read_text() == first


def test_existing_artifacts_are_not_recomputed(make_cfg, capsys):
    cfg = make_cfg()
    run_preprocess(cfg, backend=FakeTeacher(cfg))
    teacher = FakeTeacher(cfg)
    run_preprocess(cfg, backend=teacher)
    assert teacher.calls == 0
    assert "skipping" in capsys.readouterr().out


def test_stale_artifacts_are_refused_unless_forced(make_cfg):
    cfg = make_cfg()
    run_preprocess(cfg, backend=FakeTeacher(cfg))

    changed = make_cfg(overrides=["aspects.descriptions.impression=Something else entirely."])
    with pytest.raises(StaleArtifactError):
        run_preprocess(changed, stages=["augment"], backend=FakeTeacher(changed))
    teacher = FakeTeacher(changed)
    run_preprocess(changed, stages=["augment"], backend=teacher, force=True)
    assert teacher.calls == 20

    changed_split = make_cfg(overrides=["dataset.splits.train=0.7", "dataset.splits.test=0.2"])
    with pytest.raises(StaleArtifactError):
        run_preprocess(changed_split, stages=["process"])


def test_resume_after_partial_run_continues_where_it_stopped(make_cfg):
    limited = make_cfg(overrides=["augmentation.limit=5"])
    run_preprocess(limited, backend=FakeTeacher(limited))
    manifest = json.loads((hybrid_dir(limited) / "manifest.json").read_text())
    assert manifest["num_records"] == 5 and manifest["complete"] is False

    cfg = make_cfg()  # same artifact fingerprint: `limit` is not part of it
    teacher = FakeTeacher(cfg)
    run_preprocess(cfg, backend=teacher)
    assert teacher.calls == 15
    manifest = json.loads((hybrid_dir(cfg) / "manifest.json").read_text())
    assert manifest["num_records"] == 20 and manifest["complete"] is True


def test_teacher_failures_are_recorded_and_skipped(make_cfg):
    cfg = make_cfg(overrides=["generation.max_retries=1"])
    teacher = FakeTeacher(cfg, garbage_for_studies=[3])
    run_preprocess(cfg, backend=teacher)
    failures = list(read_jsonl(hybrid_dir(cfg) / "failures.jsonl"))
    assert [f["study_id"] for f in failures] == ["3"] and len(failures[0]["errors"]) == 2
    manifest = json.loads((hybrid_dir(cfg) / "manifest.json").read_text())
    assert manifest["num_records"] == 19 and manifest["num_failed"] == 1 and manifest["complete"] is True


def test_unknown_column_error_lists_actual_columns(make_cfg):
    cfg = make_cfg(overrides=["dataset.columns.study_id=patient"])
    with pytest.raises(KeyError, match="Actual columns"):
        run_preprocess(cfg, stages=["download", "process"])


def test_process_before_download_fails_clearly(make_cfg):
    with pytest.raises(FileNotFoundError, match="download"):
        run_preprocess(make_cfg(), stages=["process"])


def test_study_dataset_builds_one_multi_aspect_sample_per_study(make_cfg):
    cfg = make_cfg("train", training="ga_gd")
    run_preprocess(cfg, backend=FakeTeacher(cfg))
    splits, image_root = load_hybrid(cfg)
    train = splits["train"]

    full = StudyDataset(train, image_root, cfg)
    assert len(full) == len(train)
    sample = full[0]
    assert set(sample) == {"study_id", "aspects", "references", "target", "spans", "prompt", "images"}
    assert sample["aspects"] == aspect_keys(cfg) and sample["images"][0].mode == "RGB"
    n = len(sample["images"])
    assert sample["references"]["lungs_and_pleura"] == f"summary of lungs_and_pleura ({n} imgs)"
    assert sample["target"].startswith(f"heart and mediastinum\nsummary of heart_and_mediastinum ({n} imgs)\n\n")
    assert parse_answer(sample["target"], sample["aspects"]) == sample["references"]
    for aspect, start, end in sample["spans"]:
        assert sample["target"][start:end] == f"{display_name(aspect)}\n{sample['references'][aspect]}"
    assert full.meta(0)["target"] == sample["target"]

    forget_aspects, retain_aspects = split_aspects(cfg)
    forget = StudyDataset(train, image_root, cfg, aspects=forget_aspects)
    retain = StudyDataset(train, image_root, cfg, aspects=retain_aspects)
    assert forget[0]["aspects"] == ["impression"] and "impression" not in retain[0]["aspects"]
    assert forget[0]["study_id"] == retain[0]["study_id"] == sample["study_id"]
    assert forget[0]["prompt"] != retain[0]["prompt"] and "impression" in forget[0]["prompt"]
    assert forget[0]["target"] == f"impression\nsummary of impression ({n} imgs)"

    default = StudyDataset(train, image_root, cfg, aspects=forget_aspects, target_override="I can't assist you on this")
    assert default[0]["target"] == "impression\nI can't assist you on this"
    assert default[0]["prompt"] == forget[0]["prompt"]

    assert len(StudyDataset(train, image_root, cfg, limit=3)) == 3
    with pytest.raises(ValueError, match="Unknown aspect"):
        StudyDataset(train, image_root, cfg, aspects=["nope"])


def test_a_student_trains_on_data_generated_by_a_different_teacher(make_cfg):
    run_preprocess(make_cfg(model="qwen2_vl_7b"), backend=FakeTeacher(make_cfg(model="qwen2_vl_7b")))
    student = dict(model="qwen2_5_vl_3b", training="ga_gd")

    splits, _ = load_hybrid(make_cfg("train", teacher="qwen2_vl_7b", **student))
    assert len(splits["train"]) > 0
    with pytest.raises(FileNotFoundError, match="hybrid__qwen2_5_vl_3b"):
        load_hybrid(make_cfg("train", **student))  # no teacher named: it looks for the student's own data

    cfg = make_cfg("train", teacher="qwen2_vl_7b", **student)
    assert StudyDataset(splits["train"], load_hybrid(cfg)[1], cfg)[0]["target"]


def test_split_aspects_validation(make_cfg):
    with pytest.raises(ValueError, match="not in dataset aspects"):
        split_aspects(make_cfg("train", training="ga_gd", overrides=["forget_aspects=[nope]"]))
    with pytest.raises(ValueError, match="at least one"):
        split_aspects(make_cfg("train", training="ga_gd", overrides=[
            "forget_aspects=[heart_and_mediastinum,lungs_and_pleura,bones_soft_tissue_and_devices,impression]"]))
    with pytest.raises(ValueError, match="at least one"):
        split_aspects(make_cfg("train", training="ga_gd", overrides=["forget_aspects=[]"]))


def test_forget_aspects_belong_to_the_dataset_not_to_the_method(make_cfg):
    for method in ("ga_gd", "ga"):
        assert "forget" not in make_cfg("train", training=method).training
        assert split_aspects(make_cfg("train", training=method))[0] == ["impression"]
    cfg = make_cfg("train", training="ga", overrides=["forget_aspects=[lungs_and_pleura]"])
    forget, retain = split_aspects(cfg)
    assert forget == ["lungs_and_pleura"] and "lungs_and_pleura" not in retain and len(retain) == 3


def test_changing_forget_aspects_does_not_invalidate_hybrid_data(make_cfg):
    cfg = make_cfg()
    run_preprocess(cfg, backend=FakeTeacher(cfg))
    other = make_cfg(overrides=["forget_aspects=[lungs_and_pleura]"])
    teacher = FakeTeacher(other)
    run_preprocess(other, backend=teacher)
    assert teacher.calls == 0


def test_expansion_disabled_stores_no_expanded_text(make_cfg):
    cfg = make_cfg(overrides=["prompt.expansion.enabled=false"])
    run_preprocess(cfg, backend=FakeTeacher(cfg))
    splits, _ = load_hybrid(cfg)
    assert EXPANDED_TEXT_KEY not in splits["train"].column_names
