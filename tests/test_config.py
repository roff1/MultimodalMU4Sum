import stat

import pytest
from omegaconf import OmegaConf

from mu4sum.config import ImmutableConfigError, load_config, save_run_config
from mu4sum.config.loader import config_fingerprint
from mu4sum.config.paths import eval_dir, original_dir, run_dir


def test_operation_selects_groups():
    cfg = load_config("train", model="qwen2_vl_7b", dataset="iu_xray", training="ga_gd")
    assert cfg.training.method == "ga_gd" and "model" in cfg and "dataset" in cfg
    assert "evaluation" not in cfg and cfg.run.method == "ga_gd"

    cfg = load_config("evaluate", model="qwen2_vl_7b", dataset="iu_xray")
    assert "split" not in cfg.evaluation  # evaluation always covers every split in EVAL_SPLITS
    assert cfg.run.method == "original" and cfg.training.method == "original"  # default evaluation target

    cfg = load_config("evaluate", model="qwen2_vl_7b", dataset="iu_xray", training="base")
    assert cfg.run.method == "base" and "training" not in cfg  # pretrained model: no training preset

    cfg = load_config("preprocess", dataset="iu_xray")
    assert "model" not in cfg


def test_missing_required_preset_lists_available():
    with pytest.raises(ValueError, match="requires a model"):
        load_config("train", dataset="iu_xray", training="original")
    with pytest.raises(FileNotFoundError, match="Available"):
        load_config("train", model="nope", dataset="iu_xray", training="original")
    with pytest.raises(ValueError):
        load_config("train", model="../base", dataset="iu_xray", training="original")
    with pytest.raises(ValueError, match="requires a training"):
        load_config("train", model="qwen2_vl_7b", dataset="iu_xray")


def test_overrides_strict_and_plus():
    cfg = load_config("train", model="qwen2_vl_7b", dataset="iu_xray", training="original",
                      overrides=["training.lr=1e-3", "lora.r=4", "+training.extra=1"])
    assert cfg.training.lr == 1e-3 and cfg.lora.r == 4 and cfg.training.extra == 1
    with pytest.raises(ValueError, match="Invalid override"):
        load_config("train", model="qwen2_vl_7b", dataset="iu_xray", training="original",
                    overrides=["training.lerning_rate=1"])


def test_config_is_readonly():
    cfg = load_config("train", model="qwen2_vl_7b", dataset="iu_xray", training="original")
    with pytest.raises(Exception):
        cfg.seed = 1


def test_hierarchical_output_paths():
    cfg = load_config("train", model="qwen2_vl_7b", dataset="iu_xray", training="ga_gd",
                      tag="a0.5", original_tag="o1", overrides=["runtime.output_root=/out"])
    assert run_dir(cfg).as_posix().endswith("out/qwen2_vl_7b/iu_xray/ga_gd/a0.5")
    assert original_dir(cfg).as_posix().endswith("out/qwen2_vl_7b/iu_xray/original/o1")

    cfg = load_config("evaluate", model="qwen2_vl_7b", dataset="iu_xray", training="ga_gd",
                      overrides=["runtime.output_root=/out"])
    assert eval_dir(cfg).as_posix().endswith("ga_gd/evaluation/default_eval")


def test_every_method_has_a_preset_and_the_other_way_round():
    from mu4sum.config import available_presets
    from mu4sum.training.methods import TRAINING_METHODS

    assert available_presets("training") == TRAINING_METHODS.names()
    for name in TRAINING_METHODS.names():
        cfg = load_config("train", model="qwen2_vl_7b", dataset="iu_xray", training=name)
        assert cfg.training.method == name


def test_snapshot_is_immutable(tmp_path):
    cfg = load_config("train", model="qwen2_vl_7b", dataset="iu_xray", training="original")
    path = save_run_config(cfg, tmp_path, "train")
    assert not path.stat().st_mode & stat.S_IWUSR  # read-only
    assert (tmp_path / "provenance_train.json").exists()
    assert save_run_config(cfg, tmp_path, "train") == path  # same config: resume is a no-op

    other = load_config("train", model="qwen2_vl_7b", dataset="iu_xray", training="original", overrides=["training.lr=1e-3"])
    with pytest.raises(ImmutableConfigError):
        save_run_config(other, tmp_path, "train")


def test_runtime_paths_do_not_break_resume(tmp_path):
    """Same experiment started locally and resumed on Colab: only `runtime` differs."""
    a = load_config("train", model="qwen2_vl_7b", dataset="iu_xray", training="original", overrides=["runtime.data_root=/a"])
    b = load_config("train", model="qwen2_vl_7b", dataset="iu_xray", training="original", overrides=["runtime.data_root=/b"])
    assert config_fingerprint(a) == config_fingerprint(b)
    save_run_config(a, tmp_path, "train")
    save_run_config(b, tmp_path, "train")


def test_versioned_snapshots_coexist(tmp_path):
    a = load_config("preprocess", dataset="iu_xray")
    b = load_config("preprocess", dataset="iu_xray", overrides=["augmentation.limit=5"])
    pa = save_run_config(a, tmp_path, "preprocess", versioned=True)
    pb = save_run_config(b, tmp_path, "preprocess", versioned=True)
    assert pa != pb and pa.exists() and pb.exists()
    assert OmegaConf.load(pb).augmentation.limit == 5


def test_teacher_defaults_to_the_model_and_changes_data_and_output_paths():
    from mu4sum.config.paths import hybrid_dir

    same = load_config("train", model="qwen2_5_vl_3b", dataset="iu_xray", training="ga_gd",
                       overrides=["runtime.output_root=/out", "runtime.data_root=/data"])
    assert same.run.teacher == "qwen2_5_vl_3b"
    assert hybrid_dir(same).as_posix().endswith("data/processed/iu_xray/hybrid__qwen2_5_vl_3b")
    assert run_dir(same).as_posix().endswith("out/qwen2_5_vl_3b/iu_xray/ga_gd")

    other = load_config("train", model="qwen2_5_vl_3b", teacher="qwen2_vl_7b", dataset="iu_xray", training="ga_gd",
                        overrides=["runtime.output_root=/out", "runtime.data_root=/data"])
    assert other.run.model == "qwen2_5_vl_3b" and other.run.teacher == "qwen2_vl_7b"
    assert hybrid_dir(other).as_posix().endswith("data/processed/iu_xray/hybrid__qwen2_vl_7b")
    assert run_dir(other).as_posix().endswith("out/qwen2_5_vl_3b/iu_xray__teacher-qwen2_vl_7b/ga_gd")
    assert original_dir(other).as_posix().endswith("iu_xray__teacher-qwen2_vl_7b/original")

    evaluation = load_config("evaluate", model="qwen2_5_vl_3b", teacher="qwen2_vl_7b", dataset="iu_xray")
    assert evaluation.run.teacher == "qwen2_vl_7b"
    with pytest.raises(FileNotFoundError, match="Available"):
        load_config("train", model="qwen2_5_vl_3b", teacher="nope", dataset="iu_xray", training="original")
    with pytest.raises(ValueError, match="teacher"):
        load_config("preprocess", model="qwen2_vl_7b", teacher="qwen2_vl_7b", dataset="iu_xray")


def test_lora_settings_belong_to_the_model_and_do_not_touch_the_teacher_fingerprint():
    from mu4sum.config import available_presets
    from mu4sum.preprocessing.pipeline import augment_fingerprint

    for name in available_presets("model"):
        cfg = load_config("train", model=name, dataset="iu_xray", training="original")
        assert "lora" not in cfg.training and cfg.lora.target_modules and cfg.lora.r > 0
    a = load_config("preprocess", model="qwen2_vl_7b", dataset="iu_xray")
    b = load_config("preprocess", model="qwen2_vl_7b", dataset="iu_xray", overrides=["lora.r=4"])
    assert augment_fingerprint(a) == augment_fingerprint(b)
