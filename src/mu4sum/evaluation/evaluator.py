"""Evaluation stage: generate multi-aspect answers per view for every record of a split, then score them.

Views: with `forget_aspects` set, every model (pretrained base, original, retrain, unlearned...) is asked once for
the forget aspects ("forget") and once for the retain aspects ("retain") of each record, exactly the two prompts
sample-wise unlearning trains on; without it a single "all" view asks for every aspect. Works for every training
method (and the pretrained `base` model).
"""

from omegaconf import DictConfig

from mu4sum.config import start_run
from mu4sum.config.loader import BASE_TARGET
from mu4sum.config.paths import eval_dir, train_dir
from mu4sum.evaluation.metrics import score_rows, summarize
from mu4sum.models.chat import processor_inputs, render_prompt
from mu4sum.models.loader import load_model, load_processor
from mu4sum.preprocessing.datasets import RecordDataset, load_processed, split_aspects
from mu4sum.training.run import build_method, model_checkpoint
from mu4sum.utils.io import read_jsonl, write_json_atomic, write_jsonl
from mu4sum.utils.reproducibility import seed_worker

EVAL_SPLITS = ("train", "test")  # train = what unlearning acted on, test = generalisation to unseen records
METRICS_FILE = "metrics.json"


def predictions_file(split: str) -> str:
    """File name of the predictions of one split inside the evaluation folder."""
    return f"predictions_{split}.jsonl"


def eval_views(cfg: DictConfig) -> dict:
    """{view name: aspects asked in that view}; None stands for every aspect."""
    if cfg.get("forget_aspects"):
        forget, retain = split_aspects(cfg)
        return {"forget": forget, "retain": retain}
    return {"all": None}


class GenerationCollator:
    """Left-padded prompt-only batches; keeps the references for scoring."""

    def __init__(self, processor, nested_images: bool):
        self.processor, self.nested = processor, nested_images
        processor.tokenizer.padding_side = "left"

    def __call__(self, samples):
        """Render and tokenize prompts for a batch; return (model inputs, per-sample reference metadata)."""
        texts = [render_prompt(self.processor, s["prompt"], len(s["images"])) for s in samples]
        inputs = processor_inputs(self.processor, texts, [s["images"] for s in samples], self.nested)
        meta = [{k: s[k] for k in ("id", "aspects", "references", "target")} for s in samples]
        return inputs, meta


def _generate(cfg: DictConfig, model, processor, dataset, view: str) -> list:
    """Generate an answer per record (greedy if temperature is 0) and return them as prediction rows."""
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
            rows += [{"id": m["id"], "view": view, "aspects": m["aspects"], "reference": m["target"],
                      "references": m["references"], "prediction": t.strip()} for m, t in zip(meta, texts)]
    return rows


def _cached(path, views: dict) -> bool:
    """True when `path` holds predictions in the current format for exactly the expected views."""
    if not path.exists():
        return False
    rows = list(read_jsonl(path))
    ok = bool(rows) and all("id" in r and "view" in r and "references" in r for r in rows) and {r["view"] for r in rows} == set(views)
    if not ok:
        print(f"[Evaluate] Ignoring {path}: not in the current (view, multi-aspect) format; regenerating")
    return ok


def run_evaluate(cfg: DictConfig) -> dict:
    """Generate (or reuse cached) predictions for every split in EVAL_SPLITS and score them.

    Saves metrics.json ({split: {group: {n, metric: mean x100}}}) and returns it."""
    target = cfg.run.method
    method = None if target == BASE_TARGET else build_method(cfg)
    views = eval_views(cfg)
    out = eval_dir(cfg)
    start_run(cfg, out, "evaluate")

    missing = [s for s in EVAL_SPLITS if not _cached(out / predictions_file(s), views)]
    if missing:
        adapter = None
        if method:
            adapter = model_checkpoint(cfg)
            if adapter is None:
                raise FileNotFoundError(f"No finished '{target}' model in {train_dir(cfg)}; run "
                                        f"`scripts/run_train.py --method {target}` first.")
        splits, image_root = load_processed(cfg)
        model, processor = load_model(cfg.model, adapter_dir=adapter), load_processor(cfg.model)
        for split in missing:
            rows = []
            for view, aspects in views.items():
                dataset = RecordDataset(splits[split], image_root, cfg, aspects=aspects, limit=cfg.evaluation.limit)
                print(f"[Evaluate] '{target}' on {len(dataset)} records of '{split}', view '{view}'")
                rows += _generate(cfg, model, processor, dataset, view)
            write_jsonl(out / predictions_file(split), rows)
    for split in set(EVAL_SPLITS) - set(missing):
        print(f"[Evaluate] Reusing predictions in {out / predictions_file(split)}")

    names = list(cfg.evaluation.metrics)
    metrics = {s: summarize(score_rows(list(read_jsonl(out / predictions_file(s))), names, cfg.evaluation))
               for s in EVAL_SPLITS}
    write_json_atomic(out / METRICS_FILE, metrics)
    for split in EVAL_SPLITS:
        print(f"[Evaluate] {split}: " + "; ".join(
            f"{g}: " + " ".join(f"{k}={v:.2f}" for k, v in m.items() if k != "n") for g, m in metrics[split].items()
            if not g.startswith("aspect/")))
    return metrics
