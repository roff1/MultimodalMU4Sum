"""Metric registry and per-sample scoring, following the MU4Sum evaluation.

A metric is `fn(predictions, references, cfg) -> {metric_name: [score of each pair, in [0, 1]]}`; heavy libraries
are imported lazily so only the metrics listed in the evaluation preset need installing. Scores are kept per
sample (needed for the paired significance test) and averaged x100 only when summarised.

Every generated answer is scored at two levels:
  full    the whole answer against the whole reference answer
  aspect  each requested aspect's summary (read back from the answer, see preprocessing.answers) against its
          reference summary; an aspect the model did not write scores 0; per sample it is the mean over aspects
and per aspect ("aspect/<name>") on the aspect-level pairs, for each view ("forget", "retain", or "all").
"""
from typing import Dict, List

from omegaconf import DictConfig

from mu4sum.preprocessing.answers import parse_answer
from mu4sum.utils.registry import Registry

METRICS = Registry("metric")

LEVELS = ("full", "aspect")


@METRICS.register("rouge")
def rouge(preds: List[str], refs: List[str], cfg: DictConfig) -> Dict[str, List[float]]:
    """ROUGE-1/2/L F-measure per pair (no stemming, as in the reference evaluation)."""
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)
    scores = [scorer.score(r, p) for p, r in zip(preds, refs)]
    return {name: [s[name].fmeasure for s in scores] for name in ("rouge1", "rouge2", "rougeL")}


@METRICS.register("bleu")
def bleu(preds: List[str], refs: List[str], cfg: DictConfig) -> Dict[str, List[float]]:
    """Sentence-level BLEU per pair via sacrebleu, rescaled to [0, 1]."""
    import sacrebleu

    return {"bleu": [sacrebleu.sentence_bleu(p, [r]).score / 100 for p, r in zip(preds, refs)]}


@METRICS.register("bertscore")
def bertscore(preds: List[str], refs: List[str], cfg: DictConfig) -> Dict[str, List[float]]:
    """BERTScore F1 per pair with the model configured in cfg.bertscore.model_type (English)."""
    from bert_score import score

    _, _, f1 = score(preds, refs, model_type=cfg.bertscore.model_type, lang="en")
    return {"bertscore_f1": [float(x) for x in f1]}


def score_pairs(preds: List[str], refs: List[str], metric_names: List[str], cfg: DictConfig) -> Dict[str, List[float]]:
    """Per-pair scores of every metric; a pair with an empty prediction scores 0 without calling the metric."""
    keep = [i for i, p in enumerate(preds) if p.strip()]
    out: Dict[str, List[float]] = {}
    for name in metric_names:
        if not keep:
            continue
        result = METRICS.get(name)([preds[i] for i in keep], [refs[i] for i in keep], cfg)
        for metric, values in result.items():
            full = [0.0] * len(preds)
            for i, v in zip(keep, values):
                full[i] = float(v)
            out[metric] = full
    return out


def score_rows(rows: List[dict], metric_names: List[str], cfg: DictConfig) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Per-sample scores {group: {id: {metric: score}}} for prediction rows with keys id, view, aspects,
    reference (full answer text), references ({aspect: summary}) and prediction. Groups: "<view>/full",
    "<view>/aspect" and "aspect/<name>"."""
    pairs = []  # (group, id, row index, kind) with kind "full" or an aspect name
    preds, refs = [], []
    parsed_rows = []
    for k, row in enumerate(rows):
        parsed = parse_answer(row["prediction"], row["aspects"])
        parsed_rows.append(parsed)
        pairs.append((f"{row['view']}/full", str(row["id"]), k, "full"))
        preds.append(row["prediction"])
        refs.append(row["reference"])
        for aspect in row["aspects"]:
            pairs.append((f"aspect/{aspect}", str(row["id"]), k, aspect))
            preds.append(parsed.get(aspect, ""))
            refs.append(row["references"][aspect])
    scores = score_pairs(preds, refs, metric_names, cfg)

    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    per_row_aspect: Dict[int, Dict[str, List[float]]] = {}
    for j, (group, rid, k, kind) in enumerate(pairs):
        values = {m: s[j] for m, s in scores.items()}
        out.setdefault(group, {})[rid] = values
        if kind != "full":
            acc = per_row_aspect.setdefault(k, {})
            for m, v in values.items():
                acc.setdefault(m, []).append(v)
    for k, acc in per_row_aspect.items():
        row = rows[k]
        out.setdefault(f"{row['view']}/aspect", {})[str(row["id"])] = {m: sum(v) / len(v) for m, v in acc.items()}
    return out


def summarize(scores: Dict[str, Dict[str, Dict[str, float]]]) -> Dict[str, Dict[str, float]]:
    """{group: {"n": samples, metric: mean score x100}} from per-sample scores."""
    out = {}
    for group, by_id in scores.items():
        metrics = sorted({m for v in by_id.values() for m in v})
        out[group] = {"n": len(by_id),
                      **{m: 100 * sum(v[m] for v in by_id.values()) / len(by_id) for m in metrics}}
    return out


def unlearning_summary(current: dict, baseline: dict) -> Dict[str, float]:
    """How much of the baseline (Original) score survives on each side and level, for every metric both share:
    {"<side>_<level>_<metric>_retention": current / baseline}. Forget should drop towards 0, retain stay near 1."""
    out = {}
    for side in ("forget", "retain"):
        for level in LEVELS:
            group = f"{side}/{level}"
            if group not in current or group not in baseline:
                continue
            for metric, base in baseline[group].items():
                if metric != "n" and base and metric in current[group]:
                    out[f"{side}_{level}_{metric}_retention"] = current[group][metric] / base
    return out
