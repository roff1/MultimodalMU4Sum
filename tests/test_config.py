import stat

import pytest
from omegaconf import OmegaConf

from mu4sum.config import ImmutableConfigError, load_config, save_run_config
from mu4sum.config.loader import config_fingerprint
from mu4sum.config.paths import eval_dir, finetune_dir, run_dir


def test_operation_selects_groups():
    cfg = load_config("unlearn", model="qwen2_vl_7b", dataset="iu_xray", unlearning="grad_diff")
    assert "unlearning" in cfg and "model" in cfg and "dataset" in cfg
    assert "evaluation" not in cfg and "finetune" not in cfg

    cfg = load_config("evaluate", model="qwen2_vl_7b", dataset="iu_xray")
    assert cfg.evaluation.split == "test"  # default evaluation preset
    assert "unlearning" not in cfg

    cfg = load_config("preprocess", dataset="iu_xray")
    assert "model" not in cfg


def test_missing_required_preset_lists_available():
    with pytest.raises(ValueError, match="requires a model"):
        load_config("finetune", dataset="iu_xray")
    with pytest.raises(FileNotFoundError, match="Available"):
        load_config("finetune", model="nope", dataset="iu_xray")
    with pytest.raises(ValueError):
        load_config("finetune", model="../base", dataset="iu_xray")


def test_overrides_strict_and_plus():
    cfg = load_config("finetune", model="qwen2_vl_7b", dataset="iu_xray",
                      overrides=["finetune.lr=1e-3", "finetune.lora.r=4", "+finetune.extra=1"])
    assert cfg.finetune.lr == 1e-3 and cfg.finetune.lora.r == 4 and cfg.finetune.extra == 1
    with pytest.raises(ValueError, match="Invalid override"):
        load_config("finetune", model="qwen2_vl_7b", dataset="iu_xray", overrides=["finetune.lerning_rate=1"])


def test_config_is_readonly():
    cfg = load_config("finetune", model="qwen2_vl_7b", dataset="iu_xray")
    with pytest.raises(Exception):
        cfg.seed = 1


def test_hierarchical_output_paths():
    cfg = load_config("unlearn", model="qwen2_vl_7b", dataset="iu_xray", unlearning="grad_diff",
                      tag="a0.5", finetune_tag="ft1", overrides=["runtime.output_root=/out"])
    assert run_dir(cfg).as_posix().endswith("out/qwen2_vl_7b/iu_xray/grad_diff/a0.5")
    assert finetune_dir(cfg).as_posix().endswith("out/qwen2_vl_7b/iu_xray/finetune/ft1")

    cfg = load_config("evaluate", model="qwen2_vl_7b", dataset="iu_xray", unlearning="grad_diff",
                      eval_target="grad_diff", overrides=["runtime.output_root=/out"])
    assert eval_dir(cfg).as_posix().endswith("grad_diff/evaluation/default_eval")


def test_snapshot_is_immutable(tmp_path):
    cfg = load_config("finetune", model="qwen2_vl_7b", dataset="iu_xray")
    path = save_run_config(cfg, tmp_path, "finetune")
    assert not path.stat().st_mode & stat.S_IWUSR  # read-only
    assert (tmp_path / "provenance_finetune.json").exists()
    assert save_run_config(cfg, tmp_path, "finetune") == path  # same config: resume is a no-op

    other = load_config("finetune", model="qwen2_vl_7b", dataset="iu_xray", overrides=["finetune.lr=1e-3"])
    with pytest.raises(ImmutableConfigError):
        save_run_config(other, tmp_path, "finetune")


def test_runtime_paths_do_not_break_resume(tmp_path):
    """Same experiment started locally and resumed on Colab: only `runtime` differs."""
    a = load_config("finetune", model="qwen2_vl_7b", dataset="iu_xray", overrides=["runtime.data_root=/a"])
    b = load_config("finetune", model="qwen2_vl_7b", dataset="iu_xray", overrides=["runtime.data_root=/b"])
    assert config_fingerprint(a) == config_fingerprint(b)
    save_run_config(a, tmp_path, "finetune")
    save_run_config(b, tmp_path, "finetune")


def test_versioned_snapshots_coexist(tmp_path):
    a = load_config("preprocess", dataset="iu_xray")
    b = load_config("preprocess", dataset="iu_xray", overrides=["augmentation.limit=5"])
    pa = save_run_config(a, tmp_path, "preprocess", versioned=True)
    pb = save_run_config(b, tmp_path, "preprocess", versioned=True)
    assert pa != pb and pa.exists() and pb.exists()
    assert OmegaConf.load(pb).augmentation.limit == 5
