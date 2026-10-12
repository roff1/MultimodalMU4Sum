import stat

import pytest
from omegaconf import OmegaConf

from mu4sum.evaluation.metrics import METRICS, score_rows, summarize, unlearning_summary
from mu4sum.utils.environment import resolve_runtime
from mu4sum.utils.registry import Registry
from mu4sum.utils.sync import sync_artifacts


@METRICS.register("exact_match")
def exact_match(preds, refs, cfg):
    exact_match.calls.append(len(preds))
    return {"exact_match": [float(p == r) for p, r in zip(preds, refs)]}


exact_match.calls = []


def row(study, view, aspects, prediction, summaries):
    from mu4sum.preprocessing.answers import format_answer

    return {"study_id": study, "view": view, "aspects": aspects, "prediction": prediction,
            "references": summaries, "reference": format_answer(summaries, aspects)[0]}


ROWS = [
    row(1, "forget", ["impression"], "impression\nfoo", {"impression": "foo"}),
    row(2, "forget", ["impression"], "impression\nbar", {"impression": "foo"}),
    row(3, "forget", ["impression"], "", {"impression": "foo"}),
    row(1, "retain", ["lungs", "heart"], "lungs\nL\n\nheart\nH", {"lungs": "L", "heart": "H"}),
    row(2, "retain", ["lungs", "heart"], "lungs\nL", {"lungs": "L", "heart": "H"}),
]


def test_per_sample_scores_at_full_aspect_and_per_aspect_levels():
    exact_match.calls.clear()
    scores = score_rows(ROWS, ["exact_match"], OmegaConf.create({}))
    assert scores["forget/full"] == {"1": {"exact_match": 1.0}, "2": {"exact_match": 0.0}, "3": {"exact_match": 0.0}}
    assert scores["forget/aspect"]["1"]["exact_match"] == 1.0 and scores["forget/aspect"]["3"]["exact_match"] == 0.0
    assert scores["retain/full"]["1"]["exact_match"] == 1.0 and scores["retain/full"]["2"]["exact_match"] == 0.0
    assert scores["retain/aspect"] == {"1": {"exact_match": 1.0}, "2": {"exact_match": 0.5}}  # missing aspect = 0
    assert scores["aspect/heart"]["2"]["exact_match"] == 0.0 and scores["aspect/lungs"]["2"]["exact_match"] == 1.0
    assert sum(exact_match.calls) == 12 - 3  # 12 pairs, 3 with an empty prediction skip the metric

    out = summarize(scores)
    assert out["forget/full"] == {"n": 3, "exact_match": pytest.approx(100 / 3)}
    assert out["retain/aspect"]["exact_match"] == pytest.approx(75.0) and out["aspect/lungs"]["n"] == 2


def test_unlearning_summary_is_the_retention_of_the_baseline_score():
    current = {"forget/full": {"n": 3, "exact_match": 10.0}, "retain/full": {"n": 3, "exact_match": 80.0},
               "forget/aspect": {"n": 3, "exact_match": 0.0}}
    baseline = {"forget/full": {"n": 3, "exact_match": 80.0}, "retain/full": {"n": 3, "exact_match": 80.0},
                "forget/aspect": {"n": 3, "exact_match": 50.0}, "retain/aspect": {"n": 3, "exact_match": 50.0}}
    assert unlearning_summary(current, baseline) == {"forget_full_exact_match_retention": 0.125,
                                                     "retain_full_exact_match_retention": 1.0,
                                                     "forget_aspect_exact_match_retention": 0.0}


def test_paired_ttest_on_per_sample_scores():
    pytest.importorskip("scipy")
    from mu4sum.evaluation.significance import paired_ttest

    base = {"g": {str(i): {"m": 0.5 + 0.01 * (i % 3)} for i in range(10)}}
    better = {"g": {str(i): {"m": 0.7 + 0.01 * (i % 3) + 0.001 * i} for i in range(10)}}
    out = paired_ttest(better, base)["g"]["m"]
    assert out["n"] == 10 and out["mean_diff"] == pytest.approx(20.45) and out["p"] < 1e-6
    same = paired_ttest(base, base)["g"]["m"]
    assert same["p"] == 1.0 and same["mean_diff"] == 0.0
    assert paired_ttest({"g": {"1": {"m": 1.0}}}, {"g": {"1": {"m": 0.0}}}) == {}  # fewer than 2 shared studies


def test_registry_errors_and_duplicates():
    reg = Registry("thing")
    reg.register("a")(lambda: 1)
    with pytest.raises(ValueError):
        reg.register("a")(lambda: 2)
    with pytest.raises(KeyError, match="Available"):
        reg.get("b")


def test_runtime_resolution_priority(monkeypatch, tmp_path):
    monkeypatch.setenv("MU4SUM_DATA_ROOT", str(tmp_path / "env_data"))
    out = resolve_runtime({"environment": "LOCAL", "data_root": None, "output_root": str(tmp_path / "explicit")})
    assert out["data_root"] == str((tmp_path / "env_data").resolve())      # env var beats default
    assert out["output_root"] == str((tmp_path / "explicit").resolve())    # explicit beats everything
    with pytest.raises(ValueError):
        resolve_runtime({"environment": "MARS"})


def test_sync_copies_light_files_only_and_protects_snapshots(tmp_path):
    src, dst = tmp_path / "drive", tmp_path / "local"
    run = src / "m" / "d" / "finetune"
    (run / "checkpoints" / "final").mkdir(parents=True)
    (run / "checkpoints" / "final" / "adapter_config.json").write_text("{}")
    (run / "checkpoints" / "final" / "adapter_model.safetensors").write_bytes(b"x" * 10)
    (run / "config_finetune.yaml").write_text("a: 1")
    (run / "metrics.json").write_text("{}")

    copied = sync_artifacts(src, dst)
    names = {p.name for p in copied}
    assert names == {"config_finetune.yaml", "metrics.json"}

    # immutable local snapshot that differs remotely must not be overwritten
    local_cfg = dst / "m" / "d" / "finetune" / "config_finetune.yaml"
    local_cfg.chmod(stat.S_IREAD)
    (run / "config_finetune.yaml").write_text("a: 2")
    assert sync_artifacts(src, dst) == []
    assert local_cfg.read_text() == "a: 1"
    assert sync_artifacts(src, tmp_path / "other", dry_run=True) and not (tmp_path / "other").exists()
