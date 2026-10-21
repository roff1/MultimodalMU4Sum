"""Training methods: every model variant the project compares is produced by the same `train` stage.

Standard baselines   original (all aspects), retrain (retain aspects, from scratch),
                     finetune (Original continued on the retain aspects),
                     default (Original fine-tuned on the forget aspects, target = refusal text)
Unlearning baselines ga, ga_gd, ga_kd, npo (sample-wise), tw_ga_gd, tw_ga_kd (token-wise); all start from Original

A method is a factory `fn(training_cfg) -> TrainingMethod`; register new ones with
@TRAINING_METHODS.register("name") and select them with `training.method` in a configs/training preset.
`step(model, batch, reference)` returns (loss, logs). Standard methods get plain model batches; sample-wise
unlearning methods get {"forget": model_inputs | None, "retain": model_inputs | None} (two views of the same
records, each asking only for its own aspects); token-wise methods get plain batches carrying `token_kind`
(+1 retain token, -1 forget token, 0 not part of the answer). `reference` is the frozen Original model (only
built when `needs_reference`). Methods with `forget_target` train on the forget aspects with that fixed text as
target instead of the summaries.

Every loss is computed on the answer tokens only (prompt, image and padding positions are masked), as a
per-sample mean over the sample's answer tokens, then averaged over the samples (MU4Sum reference code).
`labels` and `token_kind` are removed before the forward pass, so the loss never comes from the model head.
"""
from dataclasses import dataclass
from typing import Callable, NamedTuple, Optional

from omegaconf import DictConfig

from mu4sum.utils.registry import Registry

TRAINING_METHODS = Registry("training method")

# What a method trains on (train split of the dataset, forget/retain aspects come from the dataset preset).
ALL, RETAIN, FORGET, FORGET_RETAIN, TOKENWISE = "all", "retain", "forget", "forget+retain", "tokenwise"
# Where its trainable weights start from.
FROM_BASE, FROM_ORIGINAL = "base", "original"

_AUX_KEYS = ("labels", "token_kind")


@dataclass
class TrainingMethod:
    """How one method trains."""

    step: Callable
    data: str
    init: str
    needs_reference: bool = False
    forget_target: Optional[str] = None  # replaces the target summaries of the forget aspects

    @property
    def paired(self) -> bool:
        """True when batches are {"forget": ..., "retain": ...} instead of plain SFT batches."""
        return self.data in (FORGET, FORGET_RETAIN)

    @property
    def needs_retain(self) -> bool:
        """True when the epoch must also contain retain samples."""
        return self.data == FORGET_RETAIN

    @property
    def token_wise(self) -> bool:
        """True when batches carry `token_kind` (forget and retain aspects share every answer)."""
        return self.data == TOKENWISE


class _Answer(NamedTuple):
    """Next-token distributions at the answer positions of a batch (all tensors flattened over answer tokens)."""

    logp: "torch.Tensor"       # [n_tokens, vocab] log-probabilities
    targets: "torch.Tensor"    # [n_tokens] gold token ids
    sample: "torch.Tensor"     # [n_tokens] index of the sample each token belongs to
    kind: Optional["torch.Tensor"]  # [n_tokens] token_kind, when the batch has it
    batch_size: int

    def token_nll(self):
        """Negative log-likelihood of every answer token."""
        return -self.logp.gather(-1, self.targets[:, None]).squeeze(-1)


def _answer(model, inputs) -> _Answer:
    import torch

    logits = model(**{k: v for k, v in inputs.items() if k not in _AUX_KEYS}).logits[:, :-1]
    targets = inputs["labels"][:, 1:]
    mask = targets != -100
    kind = inputs["token_kind"][:, 1:][mask] if "token_kind" in inputs else None
    return _Answer(
        torch.log_softmax(logits[mask].float(), dim=-1), 
        targets[mask], 
        mask.nonzero()[:, 0],
        kind, 
        targets.shape[0]
    )


def _per_sample_mean(values, sample, batch_size, weight=None):
    """Sum of `values` per sample divided by the sample's token count (`weight` counts tokens, default all)."""
    import torch

    count = torch.ones_like(values) if weight is None else weight
    total = values.new_zeros(batch_size).index_add(0, sample, values)
    n = values.new_zeros(batch_size).index_add(0, sample, count)
    return total / n.clamp_min(1)


def _nll(model, inputs):
    """Per-sample mean token NLL of the gold answer: [batch]."""
    a = _answer(model, inputs)
    return _per_sample_mean(a.token_nll(), a.sample, a.batch_size)


def _token_kl(reference, a: _Answer):
    """KL(reference || current) at every answer token: [n_tokens]."""
    return (reference.logp.exp() * (reference.logp - a.logp)).sum(dim=-1)


def _kl(model, reference, inputs):
    """Per-sample mean token KL(reference || model): [batch]."""
    import torch

    with torch.no_grad():
        ref = _answer(reference, inputs)
    a = _answer(model, inputs)
    return _per_sample_mean(_token_kl(ref, a), a.sample, a.batch_size)


def _mean(x):
    return x.mean() if x is not None else 0.0


def _logs(**values):
    return {k: float(v.detach().mean()) for k, v in values.items() if v is not None}


def _sft_step(model, batch, reference):
    nll = _nll(model, batch)
    return nll.mean(), {}


@TRAINING_METHODS.register("original")
def original(cfg: DictConfig) -> TrainingMethod:
    """New LoRA adapter trained on every aspect, forget and retain alike."""
    return TrainingMethod(_sft_step, 
        data=ALL, 
        init=FROM_BASE
    )


@TRAINING_METHODS.register("retrain")
def retrain(cfg: DictConfig) -> TrainingMethod:
    """New LoRA adapter trained from the pretrained model on the retain aspects only."""
    return TrainingMethod(_sft_step, 
        data=RETAIN, 
        init=FROM_BASE
    )


@TRAINING_METHODS.register("finetune")
def finetune(cfg: DictConfig) -> TrainingMethod:
    """Original adapter trained further on the retain aspects only."""
    return TrainingMethod(_sft_step, 
        data=RETAIN, 
        init=FROM_ORIGINAL
    )


@TRAINING_METHODS.register("default")
def default(cfg: DictConfig) -> TrainingMethod:
    """Original fine-tuned on the forget aspects with every target replaced by `training.refusal_text`."""
    return TrainingMethod(
        lambda model, batch, reference: (_nll(model, batch["forget"]).mean(), {}),
        data=FORGET,
        init=FROM_ORIGINAL,
        forget_target=str(cfg.refusal_text)
    )


@TRAINING_METHODS.register("ga")
def ga(cfg: DictConfig) -> TrainingMethod:
    """Gradient ascent: maximise the NLL on the forget set, loss = -NLL_forget."""

    def step(model, batch, reference):
        forget = _nll(model, batch["forget"])
        return -forget.mean(), _logs(forget_nll=forget)

    return TrainingMethod(
        step, 
        data=FORGET, 
        init=FROM_ORIGINAL
    )


@TRAINING_METHODS.register("ga_gd")
def ga_gd(cfg: DictConfig) -> TrainingMethod:
    """GA + gradient descent on the retain set over one mixed batch: every forget sample weighs -1 and every
    retain sample +alpha, loss = (-sum NLL_forget + alpha * sum NLL_retain) / batch size."""
    alpha = float(cfg.alpha)

    def step(model, batch, reference):
        forget = _nll(model, batch["forget"]) if batch["forget"] is not None else None
        retain = _nll(model, batch["retain"]) if batch["retain"] is not None else None
        n = sum(len(x) for x in (forget, retain) if x is not None)
        loss = (-(forget.sum() if forget is not None else 0.0) + alpha * (retain.sum() if retain is not None else 0.0)) / n
        return loss, _logs(forget_nll=forget, retain_nll=retain)

    return TrainingMethod(
        step, 
        data=FORGET_RETAIN, 
        init=FROM_ORIGINAL
    )


@TRAINING_METHODS.register("ga_kd")
def ga_kd(cfg: DictConfig) -> TrainingMethod:
    """GA + knowledge distillation: loss = -mean NLL_forget + alpha * mean KL(original || current) on the retain
    samples (per-token KL between next-token distributions, averaged over each answer)."""
    alpha = float(cfg.alpha)

    def step(model, batch, reference):
        forget = _nll(model, batch["forget"]) if batch["forget"] is not None else None
        kl = _kl(model, reference, batch["retain"]) if batch["retain"] is not None else None
        return -_mean(forget) + alpha * _mean(kl), _logs(forget_nll=forget, retain_kl=kl)

    return TrainingMethod(
        step, 
        data=FORGET_RETAIN, 
        init=FROM_ORIGINAL, 
        needs_reference=True
    )


@TRAINING_METHODS.register("npo")
def npo(cfg: DictConfig) -> TrainingMethod:
    """Negative preference optimisation with Original as reference model, plus NLL on the retain samples:
    loss = alpha * mean NLL_retain + gamma * mean_i [-(2/beta) log sigmoid(-beta * r_i)], where r_i is the
    difference between the current and Original mean token log-probability of forget answer i."""
    alpha, beta, gamma = float(cfg.alpha), float(cfg.beta), float(cfg.gamma)

    def step(model, batch, reference):
        import torch
        import torch.nn.functional as F

        retain = _nll(model, batch["retain"]) if batch["retain"] is not None else None
        penalty = log_ratio = None
        if batch["forget"] is not None:
            with torch.no_grad():
                ref = -_nll(reference, batch["forget"])
            log_ratio = -_nll(model, batch["forget"]) - ref
            penalty = -(2.0 / beta) * F.logsigmoid(-beta * log_ratio)
        return alpha * _mean(retain) + gamma * _mean(penalty), _logs(retain_nll=retain, forget_log_ratio=log_ratio)

    return TrainingMethod(
        step, 
        data=FORGET_RETAIN, 
        init=FROM_ORIGINAL, 
        needs_reference=True
    )


@TRAINING_METHODS.register("tw_ga_gd")
def tw_ga_gd(cfg: DictConfig) -> TrainingMethod:
    """Token-wise GA&GD: one pass over the full answer, forget-aspect tokens are pushed up in NLL (weight -beta)
    and every other token is trained down (weight alpha); per-sample sum / answer tokens, mean over samples."""
    alpha, beta = float(cfg.alpha), float(cfg.beta)

    def step(model, batch, reference):
        a = _answer(model, batch)
        nll = a.token_nll()
        weight = alpha * (a.kind > 0).to(nll.dtype) - beta * (a.kind < 0).to(nll.dtype)
        per_sample = _per_sample_mean(weight * nll, a.sample, a.batch_size, (a.kind != 0).to(nll.dtype))
        forget = a.kind < 0
        return per_sample.mean(), {"forget_token_nll": float(nll.detach()[forget].mean()) if forget.any() else 0.0}

    return TrainingMethod(
        step, 
        data=TOKENWISE, 
        init=FROM_ORIGINAL
    )


@TRAINING_METHODS.register("tw_ga_kd")
def tw_ga_kd(cfg: DictConfig) -> TrainingMethod:
    """Token-wise GA&KD: forget-aspect tokens get -beta * NLL, every other token alpha * KL(original || current);
    per-sample sum / answer tokens, mean over samples."""
    alpha, beta = float(cfg.alpha), float(cfg.beta)

    def step(model, batch, reference):
        import torch

        with torch.no_grad():
            ref = _answer(reference, batch)
        a = _answer(model, batch)
        nll, kl = a.token_nll(), _token_kl(ref, a)
        values = torch.where(a.kind < 0, -beta * nll, torch.where(a.kind > 0, alpha * kl, torch.zeros_like(nll)))
        per_sample = _per_sample_mean(values, a.sample, a.batch_size, (a.kind != 0).to(nll.dtype))
        retain = a.kind > 0
        return per_sample.mean(), {"retain_token_kl": float(kl.detach()[retain].mean()) if retain.any() else 0.0}

    return TrainingMethod(
        step, 
        data=TOKENWISE, 
        init=FROM_ORIGINAL, 
        needs_reference=True
    )
