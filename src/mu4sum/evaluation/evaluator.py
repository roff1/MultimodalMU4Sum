"""Evaluation stage: generate answers for every (study, aspect) of a split, then score per aspect."""
from pathlib import Path

from omegaconf import DictConfig

from mu4sum.config import save_run_config
from mu4sum.config.paths import checkpoints_dir, eval_dir, run_dir
from mu4sum.data.datasets import AspectDataset, load_hybrid
from mu4sum.evaluation.metrics import compute_all, unlearning_summary
from mu4sum.models.chat import processor_inputs, render_prompt
from mu4sum.models.loader import load_model, load_processor
from mu4sum.training.checkpoint import final_checkpoint
from mu4sum.utils.io import read_json, read_jsonl, write_json_atomic, write_jsonl
from mu4sum.utils.reproducibility import seed_worker, set_seed

PREDICTIONS_FILE = "predictions.jsonl"
METRICS_FILE = "metrics.json"


class GenerationCollator:
    """Left-padded prompt-only batches; keeps the references for scoring."""

    def __init__(self, processor, nested_images: bool):
        self.processor, self.nested = processor, nested_images
        processor.tokenizer.padding_side = "left"

    def __call__(self, samples):
        """Render and tokenize prompts for a batch; return (model inputs, per-sample reference metadata)."""
        texts = [render_prompt(self.processor, s["prompt"], len(s["images"])) for s in samples]
        inputs = processor_inputs(self.processor, texts, [s["images"] for s in samples], self.nested)
        meta = [{k: s[k] for k in ("study_id", "aspect", "target")} for s in samples]
        return inputs, meta


def _generate(cfg: DictConfig, model, processor, dataset, out_path: Path) -> None:
    """Generate a prediction per sample (greedy if temperature is 0) and write them to out_path as JSONL."""
    import torch
    from torch.utils.data import DataLoader

    ev = cfg.evaluation
    loader = DataLoader(dataset, batch_size=ev.batch_size, shuffle=False, num_workers=ev.num_workers,
                        collate_fn=GenerationCollator(processor, bool(cfg.model.processor.nested_images)),
                        worker_init_fn=seed_worker)
    temperature = float(ev.generation.temperature)
    sampling = {"do_sample": True, "temperature": temperature} if temperature > 0 else {"do_sample": False}
    device = next(model.parameters()).device
    model.eval()
    rows = []
    with torch.no_grad():
        for inputs, meta in loader:
            inputs = inputs.to(device)
            out = model.generate(**inputs, max_new_tokens=int(ev.generation.max_new_tokens), **sampling)
            texts = processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            rows += [{"study_id": m["study_id"], "aspect": m["aspect"], "reference": m["target"],
                      "prediction": t.strip()} for m, t in zip(meta, texts)]
    write_jsonl(out_path, rows)


def run_evaluate(cfg: DictConfig) -> dict:
    """Generate (or reuse cached) predictions, compute metrics, save metrics.json and return the metrics."""
    target = cfg.run.method
    out = eval_dir(cfg)
    save_run_config(cfg, out, "evaluate")
    set_seed(cfg.seed, cfg.deterministic)

    pred_path = out / PREDICTIONS_FILE
    if not pred_path.exists():
        adapter = None
        if target != "base":
            adapter = final_checkpoint(checkpoints_dir(run_dir(cfg)))
            if adapter is None:
                raise FileNotFoundError(f"No finished '{target}' model in {run_dir(cfg)}; train it first.")
        splits, image_root = load_hybrid(cfg)
        dataset = AspectDataset(splits[cfg.evaluation.split], image_root, cfg, limit=cfg.evaluation.limit)
        print(f"[Evaluate] '{target}' on {len(dataset)} (study, aspect) samples of '{cfg.evaluation.split}'")
        _generate(cfg, load_model(cfg.model, adapter_dir=adapter), load_processor(cfg.model), dataset, pred_path)
    else:
        print(f"[Evaluate] Reusing predictions in {pred_path}")

    forget = list(cfg.unlearning.forget.aspects) if cfg.get("unlearning") else []
    metrics = compute_all(list(read_jsonl(pred_path)), list(cfg.evaluation.metrics), cfg.evaluation, forget)

    baseline_path = eval_dir(cfg, "finetune", cfg.run.finetune_tag) / METRICS_FILE
    if forget and baseline_path.exists():
        names = [m for m in ("rougeL", "bleu", "bertscore_f1") if m in metrics["overall"]]
        if names:
            metrics["unlearning_summary"] = unlearning_summary(metrics, read_json(baseline_path), names[0])
    write_json_atomic(out / METRICS_FILE, metrics)
    print(f"[Evaluate] overall: {metrics['overall']}")
    return metrics

