"""Hierarchical, operation-aware configuration loading.

The user only names presets (model, dataset, ...). Which preset *groups* get merged is
decided by the operation (the three pipeline stages: `preprocess`, `train`, `evaluate`), so e.g.
`preprocess` never needs a model and `train` always pulls in a training-method preset. Groups live in
`configs/<dir>/<name>.yaml`.
"""
import re
from typing import Dict, Iterable, Optional

from omegaconf import DictConfig, OmegaConf

from mu4sum.utils.environment import PROJECT_ROOT, resolve_runtime
from mu4sum.utils.reproducibility import fingerprint

CONFIGS_DIR = PROJECT_ROOT / "configs"

GROUP_DIRS = {
    "dataset": "datasets",
    "model": "models",
    "training": "training",
    "evaluation": "evaluation",
}

# group -> "required" | "optional" | "default:<preset>"
OPERATIONS: Dict[str, Dict[str, str]] = {
    "preprocess": {"dataset": "required", "model": "optional"},  # model = the teacher (augment stage only)
    "train": {"model": "required", "dataset": "required", "training": "required"},  # training = the method
    "evaluate": {"model": "required", "dataset": "required", "evaluation": "default:default_eval",
                 "training": "default:original"},
}

# `evaluate` target that scores the pretrained model without any adapter (no training preset is loaded).
BASE_TARGET = "base"

_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


def available_presets(group: str) -> list:
    """Sorted names of the YAML presets found for a config group (empty if the folder is missing)."""
    folder = CONFIGS_DIR / GROUP_DIRS[group]
    return sorted(p.stem for p in folder.glob("*.yaml")) if folder.exists() else []


def _load_preset(group: str, name: str) -> DictConfig:
    if not _NAME_RE.match(name):
        raise ValueError(f"Invalid {group} preset name '{name}'")
    path = CONFIGS_DIR / GROUP_DIRS[group] / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"No {group} preset '{name}' ({path}). Available: {available_presets(group)}")
    return OmegaConf.load(path)


def _apply_overrides(cfg: DictConfig, overrides: Iterable[str]) -> DictConfig:
    """`a.b=1` must hit an existing key (typos fail loudly); `+a.b=1` may create a new one."""
    strict, loose = [], []
    for item in overrides or []:
        (loose if item.startswith("+") else strict).append(item)
    loose = [item[1:] for item in loose]
    OmegaConf.set_struct(cfg, True)
    if strict:
        try:
            cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(strict))
        except Exception as e:
            raise ValueError(f"Invalid override(s) {strict}: {e}. Use '+key=value' to add a new key.") from e
    if loose:
        OmegaConf.set_struct(cfg, False)
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(loose))
        OmegaConf.set_struct(cfg, True)
    return cfg


def load_config(
    operation: str,
    *,
    model: Optional[str] = None,
    dataset: Optional[str] = None,
    training: Optional[str] = None,
    evaluation: Optional[str] = None,
    tag: Optional[str] = None,
    original_tag: Optional[str] = None,
    eval_tag: Optional[str] = None,
    overrides: Optional[Iterable[str]] = None,
) -> DictConfig:
    """Build the fully-resolved, read-only config for one operation.

    training: the training method preset (original, retrain, finetune, default, ga, ga_gd, ga_kd, npo).
        For `evaluate` it names the method whose model is scored (default "original"; "base" = no adapter).
    tag: sub-folder to keep several runs of the same method (e.g. a hyper-parameter sweep).
    original_tag: tag of the `original` run that finetune / default / unlearning methods start from and
        that evaluations compare against.
    In `preprocess` the `model` preset is the teacher of the augmentation stage.
    """
    if operation not in OPERATIONS:
        raise ValueError(f"Unknown operation '{operation}'. Available: {sorted(OPERATIONS)}")
    requested = {"model": model, "dataset": dataset, "training": training, "evaluation": evaluation}
    base_target = operation == "evaluate" and training == BASE_TARGET

    parts = [OmegaConf.load(CONFIGS_DIR / "base.yaml")]
    chosen: Dict[str, str] = {}
    for group, rule in OPERATIONS[operation].items():
        if base_target and group == "training": # No training params needed here (base model, no fine-tuning/unlearning)
            continue
        name = requested[group]
        if name is None and rule.startswith("default:"):
            name = rule.split(":", 1)[1]
        if name is None:
            if rule == "required":
                raise ValueError(f"Operation '{operation}' requires a {group} preset. "
                                 f"Available: {available_presets(group)}")
            continue
        parts.append(_load_preset(group, name))
        chosen[group] = name

    method = BASE_TARGET if base_target else chosen.get("training")
    cfg = OmegaConf.merge(*parts)
    cfg = OmegaConf.merge(cfg, OmegaConf.create({"run": {
        "operation": operation,
        "model": chosen.get("model"),
        "dataset": chosen.get("dataset"),
        "method": method,
        "tag": tag,
        "original_tag": original_tag,
        "eval_tag": eval_tag or chosen.get("evaluation"),
    }}))
    cfg = _apply_overrides(cfg, overrides)

    cfg.runtime = OmegaConf.create(resolve_runtime(OmegaConf.to_container(cfg.runtime)))
    OmegaConf.resolve(cfg)
    OmegaConf.set_readonly(cfg, True)
    return cfg


def config_fingerprint(cfg: DictConfig, *sections: str) -> str:
    """Hash of the config minus the machine-specific `runtime` section (or of `sections` only)."""
    data = OmegaConf.to_container(cfg, resolve=True)
    if sections:
        data = {s: data.get(s) for s in sections}
    else:
        data.pop("runtime", None)
    return fingerprint(data)
