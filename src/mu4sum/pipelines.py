"""Stage entry points for fine-tuning and unlearning (thin glue around the shared Trainer)."""
from pathlib import Path

from omegaconf import DictConfig

from mu4sum.config import save_run_config
from mu4sum.config.paths import checkpoints_dir, finetune_dir, run_dir
from mu4sum.data.collate import PairedCollator, SFTCollator
from mu4sum.data.datasets import AspectDataset, load_hybrid, split_aspects
from mu4sum.models.loader import attach_lora, load_model, load_processor
from mu4sum.training.batching import PairedDataset, paired_plan, shuffled_plan
from mu4sum.training.checkpoint import final_checkpoint
from mu4sum.training.trainer import Trainer, TrainSpec
from mu4sum.unlearning.methods import UNLEARNING_METHODS
from mu4sum.utils.io import read_json, write_json_atomic
from mu4sum.utils.reproducibility import set_seed


def run_finetune(cfg: DictConfig) -> Path:
    """Fine-tune a new LoRA adapter on the training split and return the final checkpoint path."""
    out = run_dir(cfg)
    save_run_config(cfg, out, "finetune")
    set_seed(cfg.seed, cfg.deterministic)

    splits, image_root = load_hybrid(cfg)
    train = AspectDataset(splits["train"], image_root, cfg)
    print(f"[Finetune] {len(train)} (study, aspect) training samples")

    processor = load_processor(cfg.model)
    model = attach_lora(load_model(cfg.model), cfg.finetune.lora)
    collate = SFTCollator(processor, bool(cfg.model.processor.nested_images))

    trainer = Trainer(model, train, collate, shuffled_plan(len(train), cfg.finetune.batch_size, cfg.seed),
                      lambda m, b: (m(**b).loss, {}), TrainSpec.from_cfg(cfg.finetune), out,
                      cfg.seed, cfg.deterministic)
    return trainer.fit()


def run_unlearn(cfg: DictConfig) -> Path:
    """Unlearn the forget aspects from the fine-tuned adapter and return the final checkpoint path.

    Raises FileNotFoundError if the parent fine-tune has not finished; writes lineage.json."""
    out = run_dir(cfg)
    save_run_config(cfg, out, "unlearn")
    set_seed(cfg.seed, cfg.deterministic)

    parent = finetune_dir(cfg)
    start = final_checkpoint(checkpoints_dir(parent))
    if start is None:
        raise FileNotFoundError(f"No finished fine-tuned model in {parent}. Run scripts/run_finetune.py first "
                                f"(or point --finetune-tag at the right run).")
    ucfg = cfg.unlearning
    method = UNLEARNING_METHODS.get(ucfg.method)(ucfg)
    forget_aspects, retain_aspects = split_aspects(cfg)
    write_json_atomic(out / "lineage.json", {
        "starts_from": str(start),
        "finetune_config_fingerprint": (read_json(parent / "provenance_finetune.json")["config_fingerprint"]
                                        if (parent / "provenance_finetune.json").exists() else None),
        "forget_aspects": forget_aspects,
    })

    splits, image_root = load_hybrid(cfg)
    forget = AspectDataset(splits["train"], image_root, cfg, aspects=forget_aspects)
    retain = AspectDataset(splits["train"], image_root, cfg, aspects=retain_aspects) if method.needs_retain else None
    print(f"[Unlearn] forget={forget_aspects} ({len(forget)} samples)"
          + (f", retain={retain_aspects} ({len(retain)} samples)" if retain else ""))

    processor = load_processor(cfg.model)
    model = load_model(cfg.model, adapter_dir=start, trainable_adapter=True)
    collate = PairedCollator(SFTCollator(processor, bool(cfg.model.processor.nested_images)))
    plan = paired_plan(len(forget), len(retain) if retain else 0, ucfg.batch_size, cfg.seed)

    trainer = Trainer(model, PairedDataset(forget, retain), collate, plan, method.step_fn,
                      TrainSpec.from_cfg(ucfg), out, cfg.seed, cfg.deterministic)
    return trainer.fit()
