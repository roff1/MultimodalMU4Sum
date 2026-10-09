"""Metric registry. A metric is `fn(predictions, references, cfg) -> {metric_name: score}`; heavy
libraries are imported lazily so only the metrics listed in the evaluation preset need installing."""
from typing import Dict, List

from omegaconf import DictConfig

from mu4sum.utils.registry import Registry

METRICS = Registry("metric")


@METRICS.register("rouge")
def rouge(preds: List[str], refs: List[str], cfg: DictConfig) -> Dict[str, float]:
    """Mean ROUGE-1/2/L F-measure (with stemming) over the prediction/reference pairs."""
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = [scorer.score(r, p) for p, r in zip(preds, refs)]
    return {name: sum(s[name].fmeasure for s in scores) / len(scores) for name in ("rouge1", "rouge2", "rougeL")}


@METRICS.register("bleu")
def bleu(preds: List[str], refs: List[str], cfg: DictConfig) -> Dict[str, float]:
    """Corpus-level BLEU score via sacrebleu."""
    import sacrebleu

    return {"bleu": sacrebleu.corpus_bleu(preds, [refs]).score}


@METRICS.register("bertscore")
def bertscore(preds: List[str], refs: List[str], cfg: DictConfig) -> Dict[str, float]:
    """Mean BERTScore F1 using the model configured in cfg.bertscore.model_type."""
    from bert_score import score

    _, _, f1 = score(preds, refs, model_type=cfg.bertscore.model_type, lang="en")
    return {"bertscore_f1": float(f1.mean())}


def compute_group_metrics(records: List[dict], metric_names: List[str], cfg: DictConfig) -> Dict[str, float]:
    """Run the named metrics on one group of records and return their scores plus the sample count `n`."""
    preds = [r["prediction"] for r in records]
    refs = [r["reference"] for r in records]
    out: Dict[str, float] = {"n": len(records)}
    for name in metric_names:
        out.update(METRICS.get(name)(preds, refs, cfg))
    return out


def compute_all(records: List[dict], metric_names: List[str], cfg: DictConfig,
                forget_aspects: List[str] = ()) -> Dict[str, Dict[str, float]]:
    """Metrics overall, per aspect and (for unlearning runs) aggregated over forget / retain aspects."""
    groups: Dict[str, List[dict]] = {"overall": records}
    for aspect in sorted({r["aspect"] for r in records}):
        groups[f"aspect/{aspect}"] = [r for r in records if r["aspect"] == aspect]
    if forget_aspects:
        groups["forget"] = [r for r in records if r["aspect"] in forget_aspects]
        groups["retain"] = [r for r in records if r["aspect"] not in forget_aspects]
    return {name: compute_group_metrics(rows, metric_names, cfg) for name, rows in groups.items() if rows}


def unlearning_summary(current: dict, baseline: dict, metric: str) -> Dict[str, float]:
    """How much of the baseline (fine-tuned) score survives on each side: forget should drop towards 0,
    retain should stay near 1."""
    out = {}
    for side in ("forget", "retain"):
        if side in current and side in baseline and baseline[side].get(metric):
            out[f"{side}_{metric}_retention"] = current[side][metric] / baseline[side][metric]
    return out
