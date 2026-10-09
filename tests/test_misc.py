import stat

import pytest
from omegaconf import OmegaConf

from mu4sum.evaluation.metrics import METRICS, compute_all, unlearning_summary
from mu4sum.utils.environment import resolve_runtime
from mu4sum.utils.registry import Registry
from mu4sum.utils.sync import sync_artifacts


@METRICS.register("exact_match")
def exact_match(preds, refs, cfg):
    return {"exact_match": sum(p == r for p, r in zip(preds, refs)) / len(preds)}


def test_metrics_per_aspect_and_forget_retain_groups():
    rows = [
        {"aspect": "impression", "prediction": "a", "reference": "a"},
        {"aspect": "impression", "prediction": "a", "reference": "b"},
        {"aspect": "lungs", "prediction": "c", "reference": "c"},
    ]
    out = compute_all(rows, ["exact_match"], OmegaConf.create({}), forget_aspects=["impression"])
    assert out["overall"]["exact_match"] == pytest.approx(2 / 3)
    assert out["aspect/impression"]["exact_match"] == 0.5 and out["aspect/lungs"]["n"] == 1
    assert out["forget"]["n"] == 2 and out["retain"]["exact_match"] == 1.0

    base = {"forget": {"exact_match": 1.0}, "retain": {"exact_match": 1.0}}
    summary = unlearning_summary(out, base, "exact_match")
    assert summary == {"forget_exact_match_retention": 0.5, "retain_exact_match_retention": 1.0}


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
