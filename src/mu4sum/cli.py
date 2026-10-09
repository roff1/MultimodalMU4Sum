"""Command-line entry points. Every script takes the minimum (model, dataset, ...) plus free-form
`key=value` overrides of any config value, e.g.:  finetune.lr=1e-4  forget_aspects=[impression]
"""
import argparse
from typing import List, Optional

from mu4sum.config import available_presets, load_config, save_run_config
from mu4sum.config.paths import hybrid_dir, processed_dir
from mu4sum.utils.reproducibility import set_seed


def _parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--seed", type=int, default=None, help="Override the config seed")
    p.add_argument("--tag", default=None, help="Sub-folder to keep several runs of the same method apart")
    p.add_argument("overrides", nargs="*", help="Config overrides: key.path=value ('+key=value' adds a new key)")
    return p


def _with_seed(overrides: List[str], seed: Optional[int]) -> List[str]:
    return [*overrides, f"seed={seed}"] if seed is not None else list(overrides)


def data_pipeline_main(argv=None) -> None:
    """Run the download/process/augment stages for a dataset (augment needs a teacher --model)."""
    p = _parser("Download, process and augment (teacher) a dataset")
    p.add_argument("--dataset", required=True, choices=available_presets("dataset"))
    p.add_argument("--model", default=None, choices=available_presets("model"),
                   help="Teacher model preset; without it only download+process run")
    p.add_argument("--stages", nargs="+", default=None, choices=["download", "process", "augment"])
    p.add_argument("--force", action="store_true", help="Recompute the selected stages even if artifacts exist")
    a = p.parse_args(argv)

    from mu4sum.data.pipeline import run_data_pipeline

    stages = a.stages or (["download", "process", "augment"] if a.model else ["download", "process"])
    if "augment" in stages and not a.model:
        p.error("the 'augment' stage needs --model (teacher preset)")
    cfg = load_config("preprocess", dataset=a.dataset, model=a.model, overrides=_with_seed(a.overrides, a.seed))
    set_seed(cfg.seed, cfg.deterministic)
    save_run_config(cfg, hybrid_dir(cfg) if a.model else processed_dir(cfg), "preprocess", versioned=True)
    run_data_pipeline(cfg, stages, force=a.force)


def finetune_main(argv=None) -> None:
    """Parse arguments and launch LoRA fine-tuning of the chosen model on the chosen dataset."""
    p = _parser("Fine-tune a VLM (LoRA) on aspect-based summaries")
    p.add_argument("--model", required=True, choices=available_presets("model"))
    p.add_argument("--dataset", required=True, choices=available_presets("dataset"))
    p.add_argument("--finetune", default=None, choices=available_presets("finetune"), help="Training preset")
    a = p.parse_args(argv)

    from mu4sum.pipelines import run_finetune

    cfg = load_config("finetune", model=a.model, dataset=a.dataset, finetune=a.finetune, tag=a.tag,
                      overrides=_with_seed(a.overrides, a.seed))
    run_finetune(cfg)


def unlearn_main(argv=None) -> None:
    """Parse arguments and run the chosen unlearning method starting from a fine-tuned model."""
    p = _parser("Unlearn selected aspects from a fine-tuned model")
    p.add_argument("--model", required=True, choices=available_presets("model"))
    p.add_argument("--dataset", required=True, choices=available_presets("dataset"))
    p.add_argument("--method", required=True, choices=available_presets("unlearning"), help="Unlearning preset")
    p.add_argument("--finetune-tag", default=None, help="Tag of the fine-tuning run to start from")
    a = p.parse_args(argv)

    from mu4sum.pipelines import run_unlearn

    cfg = load_config("unlearn", model=a.model, dataset=a.dataset, unlearning=a.method, tag=a.tag,
                      finetune_tag=a.finetune_tag, overrides=_with_seed(a.overrides, a.seed))
    run_unlearn(cfg)


def evaluate_main(argv=None) -> None:
    """Parse arguments and evaluate a base, fine-tuned or unlearned model per aspect."""
    p = _parser("Evaluate a base / fine-tuned / unlearned model, per aspect")
    p.add_argument("--model", required=True, choices=available_presets("model"))
    p.add_argument("--dataset", required=True, choices=available_presets("dataset"))
    p.add_argument("--target", default="finetune",
                   help="'base', 'finetune' or the name of an unlearning preset whose model to score")
    p.add_argument("--unlearning", default=None, choices=available_presets("unlearning"),
                   help="Unlearning preset: adds forget/retain metric groups, with aspects from the dataset preset (implied when --target is a method)")
    p.add_argument("--eval", default=None, choices=available_presets("evaluation"), help="Evaluation preset")
    p.add_argument("--eval-tag", default=None, help="Output sub-folder name (default: the evaluation preset)")
    p.add_argument("--finetune-tag", default=None, help="Tag of the fine-tuning run used as baseline")
    a = p.parse_args(argv)

    from mu4sum.evaluation.evaluator import run_evaluate

    unlearning = a.unlearning or (a.target if a.target in available_presets("unlearning") else None)
    cfg = load_config("evaluate", model=a.model, dataset=a.dataset, evaluation=a.eval, unlearning=unlearning,
                      eval_target=a.target, tag=a.tag, finetune_tag=a.finetune_tag, eval_tag=a.eval_tag,
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
