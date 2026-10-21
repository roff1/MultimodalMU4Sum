import json

import pytest

from mu4sum.config.paths import processed_dir, raw_dir
from mu4sum.preprocessing.answers import parse_answer
from mu4sum.preprocessing.datasets import RecordDataset, load_processed, split_aspects
from mu4sum.preprocessing.pipeline import StaleArtifactError, default_stages, run_preprocess
from mu4sum.preprocessing.processors.common import PROCESSORS
from mu4sum.preprocessing.prompts import ASPECT_SUMMARIES_KEY, aspect_keys
from mu4sum.utils.io import read_jsonl


class FakeTeacher:
    """Deterministic backend; optionally fails (returns garbage) for chosen call numbers."""

    def __init__(self, cfg, garbage_for_records=()):
        self.keys, self.calls, self.garbage = list(cfg.aspects.names), 0, set(garbage_for_records)

    def generate(self, images, prompt, *, temperature, top_p, max_new_tokens):
        self.calls += 1
        if any(f"study {s}." in prompt for s in self.garbage):
            return "not json"
        out = {ASPECT_SUMMARIES_KEY: {k: f"summary of {k} ({len(images)} imgs)" for k in self.keys}}
        return json.dumps(out)


def test_full_pipeline_groups_images_and_splits_by_record(make_cfg):
    cfg = make_cfg()
    teacher = FakeTeacher(cfg)
    run_preprocess(cfg, backend=teacher)

    records = list(read_jsonl(processed_dir(cfg) / "process_records.jsonl"))
    assert len(records) == 20
    by_id = {r["id"]: r for r in records}
    assert by_id["0"]["image_types"] == ["Frontal", "Lateral"]  # variable number of images per record
    assert by_id["1"]["image_types"] == ["Frontal"]
    assert len(by_id["0"]["image_paths"]) == 2
    assert by_id["0"]["text"] == "findings of study 0. heart normal. lungs clear. no acute disease 0."
    assert {r["split"] for r in records} == {"train", "validation", "test"}

    splits, image_root = load_processed(cfg)
    ids = {name: {r["id"] for r in splits[name]} for name in splits}
    assert sum(len(v) for v in ids.values()) == 20
    assert not (ids["train"] & ids["test"]) and not (ids["train"] & ids["validation"])

    record = splits["train"][0]
    assert list(record["aspect_summaries"]) == list(cfg.aspects.names)
    assert record["text"] == by_id[record["id"]]["text"]  # the teacher never rewrites the text
    assert set(record) == {"id", "image_types", "image_paths", "text", "aspect_summaries"}  # no split, no prompt
    assert not any(p.startswith("/") for p in record["image_paths"])  # relative paths
    assert (image_root / record["image_paths"][0]).exists()


def test_split_is_deterministic_across_runs(make_cfg, tmp_path):
    run_preprocess(make_cfg(), stages=["download", "process"])
    first = (processed_dir(make_cfg()) / "process_records.jsonl").read_text()
    run_preprocess(make_cfg(), stages=["process"], force=True)
    assert (processed_dir(make_cfg()) / "process_records.jsonl").read_text() == first


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
    manifest = json.loads((processed_dir(limited) / "manifest.json").read_text())
    assert manifest["num_records"] == 5 and manifest["complete"] is False

    cfg = make_cfg()  # same artifact fingerprint: `limit` is not part of it
    teacher = FakeTeacher(cfg)
    run_preprocess(cfg, backend=teacher)
    assert teacher.calls == 15
    manifest = json.loads((processed_dir(cfg) / "manifest.json").read_text())
    assert manifest["num_records"] == 20 and manifest["complete"] is True


def test_rerun_with_limit_reached_does_nothing(make_cfg, monkeypatch):
    limited = make_cfg(overrides=["augmentation.limit=5"])
    run_preprocess(limited, backend=FakeTeacher(limited))
    manifest_path = processed_dir(limited) / "manifest.json"
    before = manifest_path.read_text()

    def no_model(*args, **kwargs):
        raise AssertionError("the teacher model must not be loaded when nothing is left to do")

    monkeypatch.setattr("mu4sum.preprocessing.teacher.HFTeacher", no_model)
    run_preprocess(limited, stages=["augment"])
    assert manifest_path.read_text() == before
    assert json.loads(before)["complete"] is False


def test_teacher_failures_are_recorded_and_skipped(make_cfg):
    cfg = make_cfg(overrides=["generation.max_retries=1"])
    teacher = FakeTeacher(cfg, garbage_for_records=[3])
    run_preprocess(cfg, backend=teacher)
    failures = list(read_jsonl(processed_dir(cfg) / "failures.jsonl"))
    assert [f["id"] for f in failures] == ["3"] and len(failures[0]["errors"]) == 2
    manifest = json.loads((processed_dir(cfg) / "manifest.json").read_text())
    assert manifest["num_records"] == 19 and manifest["num_failed"] == 1 and manifest["complete"] is True


def test_unknown_column_error_lists_actual_columns(make_cfg):
    cfg = make_cfg(overrides=["dataset.columns.image=patient"])
    with pytest.raises(KeyError, match="Actual columns"):
        run_preprocess(cfg, stages=["download", "process"])


def test_process_before_download_fails_clearly(make_cfg):
    with pytest.raises(FileNotFoundError, match="download"):
        run_preprocess(make_cfg(), stages=["process"])


def test_record_dataset_builds_one_multi_aspect_sample_per_record(make_cfg):
    cfg = make_cfg("train", training="ga_gd")
    run_preprocess(cfg, backend=FakeTeacher(cfg))
    splits, image_root = load_processed(cfg)
    train = splits["train"]

    full = RecordDataset(train, image_root, cfg)
    assert len(full) == len(train)
    sample = full[0]
    assert set(sample) == {"id", "aspects", "references", "target", "spans", "prompt", "images"}
    assert sample["aspects"] == aspect_keys(cfg) and sample["images"][0].mode == "RGB"
    n = len(sample["images"])
    assert sample["references"]["lungs_and_pleura"] == f"summary of lungs_and_pleura ({n} imgs)"
    assert json.loads(sample["target"]) == {ASPECT_SUMMARIES_KEY: sample["references"]}
    assert '"heart_and_mediastinum"' in sample["prompt"] and "Aspect 1/" not in sample["prompt"]
    assert parse_answer(sample["target"], sample["aspects"]) == sample["references"]
    for aspect, start, end in sample["spans"]:
        assert sample["target"][start:end].startswith(f'"{aspect}": "{sample["references"][aspect]}"')
    assert full.meta(0)["target"] == sample["target"]

    forget_aspects, retain_aspects = split_aspects(cfg)
    forget = RecordDataset(train, image_root, cfg, aspects=forget_aspects)
    retain = RecordDataset(train, image_root, cfg, aspects=retain_aspects)
    assert forget[0]["aspects"] == ["impression"] and "impression" not in retain[0]["aspects"]
    assert forget[0]["id"] == retain[0]["id"] == sample["id"]
    assert forget[0]["prompt"] != retain[0]["prompt"] and "impression" in forget[0]["prompt"]
    assert json.loads(forget[0]["target"]) == {ASPECT_SUMMARIES_KEY: {"impression": f"summary of impression ({n} imgs)"}}

    default = RecordDataset(train, image_root, cfg, aspects=forget_aspects, target_override="I can't assist you on this")
    assert json.loads(default[0]["target"]) == {ASPECT_SUMMARIES_KEY: {"impression": "I can't assist you on this"}}
    assert default[0]["prompt"] == forget[0]["prompt"]

    assert len(RecordDataset(train, image_root, cfg, limit=3)) == 3
    with pytest.raises(ValueError, match="Unknown aspect"):
        RecordDataset(train, image_root, cfg, aspects=["nope"])


def test_a_student_trains_on_data_generated_by_a_different_teacher(make_cfg):
    run_preprocess(make_cfg(model="qwen2_5_vl_7b"), backend=FakeTeacher(make_cfg(model="qwen2_5_vl_7b")))
    cfg = make_cfg("train", model="qwen2_5_vl_3b", training="ga_gd")

    splits, image_root = load_processed(cfg)  # same folder whatever the student: no teacher in the path
    assert len(splits["train"]) > 0
    assert json.loads((processed_dir(cfg) / "manifest.json").read_text())["teacher"] == "qwen2_5_vl_7b"
    assert RecordDataset(splits["train"], image_root, cfg)[0]["target"]


def test_a_different_teacher_is_stale_until_forced(make_cfg):
    run_preprocess(make_cfg(model="qwen2_5_vl_7b"), backend=FakeTeacher(make_cfg(model="qwen2_5_vl_7b")))
    other = make_cfg(model="qwen2_5_vl_3b")
    with pytest.raises(StaleArtifactError):
        run_preprocess(other, stages=["augment"], backend=FakeTeacher(other))
    run_preprocess(other, stages=["augment"], backend=FakeTeacher(other), force=True)
    assert json.loads((processed_dir(other) / "manifest.json").read_text())["teacher"] == "qwen2_5_vl_3b"


def test_intermediate_files_are_prefixed_and_final_files_use_standard_names(make_cfg):
    cfg = make_cfg()
    run_preprocess(cfg, backend=FakeTeacher(cfg))
    out = processed_dir(cfg)
    assert {"process_records.jsonl", "process_manifest.json"} <= {p.name for p in out.iterdir()}
    assert {"records.jsonl", "manifest.json", "progress.json"} <= {p.name for p in out.iterdir()}
    assert "num_images" in json.loads((out / "process_manifest.json").read_text())
    assert json.loads((out / "manifest.json").read_text())["complete"] is True
    assert not any(p.name.startswith("hybrid") for p in out.iterdir())


def test_preprocessing_saves_its_config_snapshot_like_the_other_stages(make_cfg):
    cfg = make_cfg()
    run_preprocess(cfg, stages=["download"])
    names = {p.name for p in processed_dir(cfg).iterdir()}
    assert any(n.startswith("config_preprocess_") for n in names)
    assert any(n.startswith("provenance_preprocess_") for n in names)


def test_forcing_augment_keeps_the_intermediate_process_files(make_cfg):
    cfg = make_cfg()
    run_preprocess(cfg, backend=FakeTeacher(cfg))
    intermediate = (processed_dir(cfg) / "process_records.jsonl").read_text()
    teacher = FakeTeacher(cfg)
    run_preprocess(cfg, stages=["augment"], backend=teacher, force=True)
    assert teacher.calls == 20
    assert (processed_dir(cfg) / "process_records.jsonl").read_text() == intermediate
    assert len(load_processed(cfg)[0]["train"]) > 0


def test_missing_process_stage_before_augment_fails_clearly(make_cfg):
    cfg = make_cfg()
    run_preprocess(cfg, stages=["download"])
    with pytest.raises(FileNotFoundError, match="process"):
        run_preprocess(cfg, stages=["augment"], backend=FakeTeacher(cfg))


ASPECTS = ["heart_and_mediastinum", "lungs_and_pleura", "bones_soft_tissue_and_devices", "impression"]


@PROCESSORS.register("iu_xray_labeled")
def _iu_xray_labeled(ds, raw_root):
    """iu_xray records that already carry their aspect summaries (a dataset that needs no augmentation)."""
    return [{**r, "aspect_summaries": {a: f"label of {a}" for a in ASPECTS}} for r in PROCESSORS.get("iu_xray")(ds, raw_root)]


def _no_augmentation(make_cfg, *extra, **kwargs):
    return make_cfg(overrides=["augmentation.enabled=false", "dataset.processor=iu_xray_labeled", *extra], **kwargs)


def test_dataset_without_augmentation_writes_final_files_straight_from_process(make_cfg):
    cfg = _no_augmentation(make_cfg)
    assert default_stages(cfg) == ["download", "process"]
    run_preprocess(cfg)
    out = processed_dir(cfg)
    names = {p.name for p in out.iterdir()}
    assert {"records.jsonl", "manifest.json"} <= names
    assert not any(n.startswith("process_") for n in names)
    assert json.loads((out / "manifest.json").read_text())["complete"] is True

    train_cfg = _no_augmentation(make_cfg, operation="train", training="ga_gd")
    splits, image_root = load_processed(train_cfg)  # training reads it exactly like augmented data
    assert RecordDataset(splits["train"], image_root, train_cfg)[0]["references"]["impression"] == "label of impression"


def test_dataset_without_augmentation_rejects_the_augment_stage(make_cfg):
    with pytest.raises(ValueError, match="augmentation.enabled"):
        run_preprocess(_no_augmentation(make_cfg), stages=["download", "process", "augment"])


def test_processor_without_aspect_summaries_is_rejected_when_there_is_no_augmentation(make_cfg):
    plain = make_cfg(overrides=["augmentation.enabled=false"])  # plain iu_xray processor: no summaries
    with pytest.raises(ValueError, match="aspect_summaries"):
        run_preprocess(plain, stages=["download", "process"])


def test_default_stages_follow_the_augmentation_flag(make_cfg):
    assert default_stages(make_cfg()) == ["download", "process", "augment"]
    assert default_stages(make_cfg(overrides=["augmentation.enabled=false"])) == ["download", "process"]


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


def _processed(cfg):
    run_preprocess(cfg, stages=["download", "process"], force=True)
    return list(read_jsonl(processed_dir(cfg) / "process_records.jsonl"))


def test_image_filter_modes(make_cfg):
    assert len(_processed(make_cfg())) == 20  # mode "none": every record
    both = _processed(make_cfg(overrides=["dataset.filtering.images.mode=count", "dataset.filtering.images.count=2"]))
    assert len(both) == 10 and all(len(r["image_paths"]) == 2 for r in both)
    one = _processed(make_cfg(overrides=["dataset.filtering.images.mode=count", "dataset.filtering.images.count=1"]))
    assert len(one) == 10 and all(r["image_types"] == ["Frontal"] for r in one)
    typed = _processed(make_cfg(overrides=["dataset.filtering.images.mode=count_and_type"]))
    assert len(typed) == 10 and all(sorted(r["image_types"]) == ["Frontal", "Lateral"] for r in typed)
    frontal = _processed(make_cfg(overrides=["dataset.filtering.images.mode=count_and_type",
                                             "dataset.filtering.images.types.Lateral=0"]))
    assert len(frontal) == 10 and all(r["image_types"] == ["Frontal"] for r in frontal)


def test_invalid_image_filter_is_rejected(make_cfg):
    with pytest.raises(ValueError, match="images.mode"):
        _processed(make_cfg(overrides=["dataset.filtering.images.mode=bogus"]))
    with pytest.raises(ValueError, match="count"):
        _processed(make_cfg(overrides=["dataset.filtering.images.mode=count", "dataset.filtering.images.count=null"]))


def test_missing_csv_or_image_folder_fails_clearly(make_cfg):
    with pytest.raises(FileNotFoundError, match="CSV not found"):
        _processed(make_cfg(overrides=["dataset.csv_path=nope.csv"]))
    with pytest.raises(FileNotFoundError, match="Image folder not found"):
        _processed(make_cfg(overrides=["dataset.images.resolution=999"]))
