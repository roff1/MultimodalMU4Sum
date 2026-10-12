"""Command-line entry points. Every script takes the minimum (model, dataset, ...) plus free-form
`key=value` overrides of any config value, e.g.:  training.lr=1e-4  forget_aspects=[impression]

The pipeline has three stages: preprocess -> train -> evaluate.
"""
import argparse
from typing import List, Optional

from mu4sum.config import available_presets, load_config, save_run_config
from mu4sum.config.loader import BASE_TARGET
from mu4sum.config.paths import hybrid_dir, processed_dir
from mu4sum.utils.reproducibility import set_seed


def _parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--seed", type=int, default=None, help="Override the config seed")
    p.add_argument("--tag", default=None, help="Sub-folder to keep several runs of the same method apart")
    p.add_argument("overrides", nargs="*", help="Config overrides: key.path=value ('+key=value' adds a new key)")
    return p


def _add_teacher(p: argparse.ArgumentParser) -> None:
    p.add_argument("--teacher", default=None, choices=available_presets("model"),
                   help="Model preset that generated the hybrid data (default: same as --model)")


def _with_seed(overrides: List[str], seed: Optional[int]) -> List[str]:
    return [*overrides, f"seed={seed}"] if seed is not None else list(overrides)


def preprocess_main(argv=None) -> None:
    """Stage 1: run the download/process/augment stages for a dataset (augment needs a teacher --model)."""
    p = _parser("Preprocessing: download, process and augment (teacher) a dataset")
    p.add_argument("--dataset", required=True, choices=available_presets("dataset"))
    p.add_argument("--model", default=None, choices=available_presets("model"),
                   help="Teacher model preset; without it only download+process run")
    p.add_argument("--stages", nargs="+", default=None, choices=["download", "process", "augment"])
    p.add_argument("--force", action="store_true", help="Recompute the selected stages even if artifacts exist")
    a = p.parse_args(argv)

    from mu4sum.preprocessing.pipeline import run_preprocess

    stages = a.stages or (["download", "process", "augment"] if a.model else ["download", "process"])
    if "augment" in stages and not a.model:
        p.error("the 'augment' stage needs --model (teacher preset)")
    cfg = load_config("preprocess", dataset=a.dataset, model=a.model, overrides=_with_seed(a.overrides, a.seed))
    set_seed(cfg.seed, cfg.deterministic)
    save_run_config(cfg, hybrid_dir(cfg) if a.model else processed_dir(cfg), "preprocess", versioned=True)
    run_preprocess(cfg, stages, force=a.force)


def train_main(argv=None) -> None:
    """Stage 2: train one method (standard baseline or unlearning baseline) on the chosen model and dataset."""
    p = _parser("Train a VLM variant: original, retrain, finetune, default, ga, ga_gd, ga_kd, npo, tw_ga_gd or tw_ga_kd")
    p.add_argument("--model", required=True, choices=available_presets("model"))
    p.add_argument("--dataset", required=True, choices=available_presets("dataset"))
    p.add_argument("--method", required=True, choices=available_presets("training"), help="Training method preset")
    _add_teacher(p)
    p.add_argument("--original-tag", default=None,
                   help="Tag of the 'original' run that finetune / default / unlearning methods start from")
    a = p.parse_args(argv)

    from mu4sum.training.run import run_train

    cfg = load_config("train", model=a.model, dataset=a.dataset, training=a.method, tag=a.tag,
                      original_tag=a.original_tag, teacher=a.teacher, overrides=_with_seed(a.overrides, a.seed))
    run_train(cfg)


def evaluate_main(argv=None) -> None:
    """Stage 3: evaluate the model of a training method (or the pretrained base) on train and test, per aspect."""
    p = _parser("Evaluate the model of a training method (or the pretrained base), per aspect")
    p.add_argument("--model", required=True, choices=available_presets("model"))
    p.add_argument("--dataset", required=True, choices=available_presets("dataset"))
    p.add_argument("--method", default="original", choices=[BASE_TARGET, *available_presets("training")],
                   help="Training method whose model to score ('base' = pretrained model, no adapter)")
    _add_teacher(p)
    p.add_argument("--eval", default=None, choices=available_presets("evaluation"), help="Evaluation preset")
    p.add_argument("--eval-tag", default=None, help="Output sub-folder name (default: the evaluation preset)")
    p.add_argument("--original-tag", default=None, help="Tag of the 'original' run used as baseline")
    a = p.parse_args(argv)

    from mu4sum.evaluation.evaluator import run_evaluate

    cfg = load_config("evaluate", model=a.model, dataset=a.dataset, evaluation=a.eval, training=a.method,
                      tag=a.tag, original_tag=a.original_tag, teacher=a.teacher, eval_tag=a.eval_tag,
                      overrides=_with_seed(a.overrides, a.seed))
    run_evaluate(cfg)


def sync_main(argv=None) -> None:
    """Copy light run artifacts from a remote outputs folder into the local outputs folder."""
    from pathlib import Path

    from mu4sum.utils.environment import PROJECT_ROOT
    from mu4sum.utils.sync import sync_artifacts

    p = argparse.ArgumentParser(description="Copy light run artifacts from a remote/Drive outputs folder "
                                            "into the local repo (for Git tracking)")
    p.add_argument("--src", required=True, help="Remote outputs root (e.g. Drive/MU4Sum/outputs)")
    p.add_argument("--dst", default=str(PROJECT_ROOT / "outputs"))
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args(argv)
    files = sync_artifacts(Path(a.src), Path(a.dst), dry_run=a.dry_run)
    print(f"[Sync] {'Would copy' if a.dry_run else 'Copied'} {len(files)} file(s) to {a.dst}")
