import json
from collections import defaultdict

import pytest

from mu4sum.config.paths import hybrid_dir, processed_dir, raw_dir
from mu4sum.data.datasets import AspectDataset, load_hybrid, split_aspects
from mu4sum.data.pipeline import StaleArtifactError, run_data_pipeline
from mu4sum.data.prompts import ASPECT_SUMMARIES_KEY, EXPANDED_TEXT_KEY, expansion_enabled
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
    run_data_pipeline(cfg, backend=teacher)

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
    run_data_pipeline(make_cfg(), stages=["download", "process"])
    first = (processed_dir(make_cfg()) / "studies.jsonl").read_text()
    run_data_pipeline(make_cfg(), stages=["process"], force=True)
    assert (processed_dir(make_cfg()) / "studies.jsonl").read_text() == first


def test_existing_artifacts_are_not_recomputed(make_cfg, capsys):
    cfg = make_cfg()
    run_data_pipeline(cfg, backend=FakeTeacher(cfg))
    teacher = FakeTeacher(cfg)
    run_data_pipeline(cfg, backend=teacher)
    assert teacher.calls == 0
    assert "skipping" in capsys.readouterr().out


def test_stale_artifacts_are_refused_unless_forced(make_cfg):
    cfg = make_cfg()
    run_data_pipeline(cfg, backend=FakeTeacher(cfg))

    changed = make_cfg(overrides=["aspects.descriptions.impression=Something else entirely."])
    with pytest.raises(StaleArtifactError):
        run_data_pipeline(changed, stages=["augment"], backend=FakeTeacher(changed))
    teacher = FakeTeacher(changed)
    run_data_pipeline(changed, stages=["augment"], backend=teacher, force=True)
    assert teacher.calls == 20

    changed_split = make_cfg(overrides=["dataset.splits.train=0.7", "dataset.splits.test=0.2"])
    with pytest.raises(StaleArtifactError):
        run_data_pipeline(changed_split, stages=["process"])


def test_resume_after_partial_run_continues_where_it_stopped(make_cfg):
    limited = make_cfg(overrides=["augmentation.limit=5"])
    run_data_pipeline(limited, backend=FakeTeacher(limited))
    manifest = json.loads((hybrid_dir(limited) / "manifest.json").read_text())
    assert manifest["num_records"] == 5 and manifest["complete"] is False

    cfg = make_cfg()  # same artifact fingerprint: `limit` is not part of it
    teacher = FakeTeacher(cfg)
    run_data_pipeline(cfg, backend=teacher)
    assert teacher.calls == 15
    manifest = json.loads((hybrid_dir(cfg) / "manifest.json").read_text())
    assert manifest["num_records"] == 20 and manifest["complete"] is True


def test_teacher_failures_are_recorded_and_skipped(make_cfg):
    cfg = make_cfg(overrides=["generation.max_retries=1"])
    teacher = FakeTeacher(cfg, garbage_for_studies=[3])
    run_data_pipeline(cfg, backend=teacher)
    failures = list(read_jsonl(hybrid_dir(cfg) / "failures.jsonl"))
    assert [f["study_id"] for f in failures] == ["3"] and len(failures[0]["errors"]) == 2
    manifest = json.loads((hybrid_dir(cfg) / "manifest.json").read_text())
    assert manifest["num_records"] == 19 and manifest["num_failed"] == 1 and manifest["complete"] is True


def test_unknown_column_error_lists_actual_columns(make_cfg):
    cfg = make_cfg(overrides=["dataset.columns.study_id=patient"])
    with pytest.raises(KeyError, match="Actual columns"):
        run_data_pipeline(cfg, stages=["download", "process"])


def test_process_before_download_fails_clearly(make_cfg):
    with pytest.raises(FileNotFoundError, match="download"):
        run_data_pipeline(make_cfg(), stages=["process"])


def test_aspect_dataset_expands_study_aspect_pairs(make_cfg):
    cfg = make_cfg("unlearn", unlearning="grad_diff")
    run_data_pipeline(cfg, backend=FakeTeacher(cfg))
    splits, image_root = load_hybrid(cfg)
    train = splits["train"]

    full = AspectDataset(train, image_root, cfg)
    assert len(full) == len(train) * 4
    sample = full[0]
    assert set(sample) == {"study_id", "aspect", "prompt", "target", "images"}
    assert sample["target"] == f"summary of {sample['aspect']} ({len(sample['images'])} imgs)"
    assert sample["images"][0].mode == "RGB"

    forget_aspects, retain_aspects = split_aspects(cfg)
    forget = AspectDataset(train, image_root, cfg, aspects=forget_aspects)
    retain = AspectDataset(train, image_root, cfg, aspects=retain_aspects)
    assert {forget[i]["aspect"] for i in range(len(forget))} == {"impression"}
    assert "impression" not in {retain[i]["aspect"] for i in range(len(retain))}
    assert len(forget) + len(retain) == len(full)
    counts = defaultdict(int)
    for i in range(len(full)):
        counts[full.index[i][1]] += 1
    assert set(counts.values()) == {len(train)}


def test_split_aspects_validation(make_cfg):
    with pytest.raises(ValueError, match="not in dataset aspects"):
        split_aspects(make_cfg("unlearn", unlearning="grad_diff", overrides=["forget_aspects=[nope]"]))
    with pytest.raises(ValueError, match="at least one"):
        split_aspects(make_cfg("unlearn", unlearning="grad_diff", overrides=[
            "forget_aspects=[heart_and_mediastinum,lungs_and_pleura,bones_soft_tissue_and_devices,impression]"]))
    with pytest.raises(ValueError, match="at least one"):
        split_aspects(make_cfg("unlearn", unlearning="grad_diff", overrides=["forget_aspects=[]"]))


def test_forget_aspects_belong_to_the_dataset_not_to_the_method(make_cfg):
    for method in ("grad_diff", "grad_ascent"):
        assert "forget" not in make_cfg("unlearn", unlearning=method).unlearning
        assert split_aspects(make_cfg("unlearn", unlearning=method))[0] == ["impression"]
    cfg = make_cfg("unlearn", unlearning="grad_ascent", overrides=["forget_aspects=[lungs_and_pleura]"])
    forget, retain = split_aspects(cfg)
    assert forget == ["lungs_and_pleura"] and "lungs_and_pleura" not in retain and len(retain) == 3


def test_changing_forget_aspects_does_not_invalidate_hybrid_data(make_cfg):
    cfg = make_cfg()
    run_data_pipeline(cfg, backend=FakeTeacher(cfg))
    other = make_cfg(overrides=["forget_aspects=[lungs_and_pleura]"])
    teacher = FakeTeacher(other)
    run_data_pipeline(other, backend=teacher)
    assert teacher.calls == 0


def test_expansion_disabled_stores_no_expanded_text(make_cfg):
    cfg = make_cfg(overrides=["prompt.expansion.enabled=false"])
    run_data_pipeline(cfg, backend=FakeTeacher(cfg))
    splits, _ = load_hybrid(cfg)
    assert EXPANDED_TEXT_KEY not in splits["train"].column_names
