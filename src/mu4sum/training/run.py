"""Training stage: one entry point for every method in TRAINING_METHODS (thin glue around the shared Trainer)."""
from pathlib import Path
from typing import Optional

from omegaconf import DictConfig

from mu4sum.config import save_run_config
from mu4sum.config.paths import checkpoints_dir, original_dir, run_dir
from mu4sum.models.loader import attach_lora, load_model, load_processor
from mu4sum.preprocessing.datasets import RecordDataset, load_processed, split_aspects
from mu4sum.training.batching import PairedDataset, mixed_plan, shuffled_plan
from mu4sum.training.checkpoint import final_checkpoint
from mu4sum.training.collate import PairedCollator, SFTCollator, TokenWiseCollator
from mu4sum.training.methods import ALL, FROM_BASE, TRAINING_METHODS, TrainingMethod
from mu4sum.training.trainer import EvalData, Trainer, TrainSpec
from mu4sum.utils.io import read_json, write_json_atomic
from mu4sum.utils.reproducibility import set_seed


def build_method(cfg: DictConfig) -> TrainingMethod:
    """The training method selected by the `training` preset of the config."""
    return TRAINING_METHODS.get(cfg.training.method)(cfg.training)


def model_checkpoint(cfg: DictConfig) -> Optional[Path]:
    """Final adapter of the model this run's method produced (None while it is unfinished)."""
    return final_checkpoint(checkpoints_dir(run_dir(cfg)))


def run_train(cfg: DictConfig) -> Optional[Path]:
    """Train the configured method on the train split and return its final checkpoint path.

    Methods starting from Original raise FileNotFoundError if it has not finished; writes lineage.json."""
    out = run_dir(cfg)
    save_run_config(cfg, out, "train")
    set_seed(cfg.seed, cfg.deterministic)

    method, tcfg = build_method(cfg), cfg.training
    start = None
    if method.init != FROM_BASE:
        parent = original_dir(cfg)
        start = final_checkpoint(checkpoints_dir(parent))
        if start is None:
            raise FileNotFoundError(f"No finished 'original' model in {parent}. Run "
                                    f"`scripts/run_train.py --method original` first (or point --original-tag "
                                    f"at the right run).")
    write_json_atomic(out / "lineage.json", {
        "method": tcfg.method,
        "starts_from": str(start) if start else None,
        "original_config_fingerprint": (read_json(parent / "provenance_train.json")["config_fingerprint"]
                                        if start and (parent / "provenance_train.json").exists() else None),
        "forget_aspects": list(cfg.get("forget_aspects") or []),
    })
    forget_aspects, retain_aspects = split_aspects(cfg) if method.data != ALL else ([], None)
    splits, image_root = load_processed(cfg)
    sft = SFTCollator(load_processor(cfg.model), bool(cfg.model.processor.nested_images))

    def inputs(split: str):
        """(dataset, collate_fn, batch plan) of this method on one split of the processed dataset."""
        def view(aspects, target=None):
            return RecordDataset(splits[split], image_root, cfg, aspects=aspects, target_override=target)

        if method.paired:
            forget = view(forget_aspects, method.forget_target)
            retain = view(retain_aspects) if method.needs_retain else None
            print(f"[Train] {tcfg.method} {split}: forget={forget_aspects} ({len(forget)} records)"
                  + (f", retain={retain_aspects} ({len(retain)} records)" if retain else ""))
            return (PairedDataset(forget, retain), PairedCollator(sft),
                    mixed_plan(len(forget), len(retain) if retain else 0, tcfg.batch_size, cfg.seed))
        if method.token_wise:
            dataset, collate = view(None), TokenWiseCollator(sft, forget_aspects)
        else:
            dataset, collate = view(None if method.data == ALL else retain_aspects), sft
        print(f"[Train] {tcfg.method} {split}: {len(dataset)} records, aspects={dataset.aspects}")
        return dataset, collate, shuffled_plan(len(dataset), tcfg.batch_size, cfg.seed)

    dataset, collate, plan = inputs("train")
    evaluation = EvalData(*inputs("validation"))

    if method.init == FROM_BASE:
        model = attach_lora(load_model(cfg.model), cfg.lora)
    else:
        model = load_model(cfg.model, adapter_dir=start, trainable_adapter=True)
    reference = None
    if method.needs_reference:
        reference = load_model(cfg.model, adapter_dir=start).eval().requires_grad_(False)

    trainer = Trainer(model, dataset, collate, plan, lambda m, b: method.step(m, b, reference),
                      TrainSpec.from_cfg(tcfg), out, cfg.seed, cfg.deterministic, evaluation=evaluation)
    return trainer.fit()
