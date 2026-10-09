"""Unlearning objectives. Each method is a factory returning an UnlearningMethod; register new ones with
@UNLEARNING_METHODS.register("name") and select them with `unlearning.method` in a YAML preset.

Batches are {"forget": model_inputs, "retain": model_inputs | None}; `model(**inputs).loss` is the mean
token NLL of the answer.
"""
from dataclasses import dataclass
from typing import Callable

from omegaconf import DictConfig

from mu4sum.utils.registry import Registry

UNLEARNING_METHODS = Registry("unlearning method")


@dataclass
class UnlearningMethod:
    """A step_fn returning (loss, logs) for a batch, and whether batches must include a retain set."""

    step_fn: Callable
    needs_retain: bool


@UNLEARNING_METHODS.register("grad_ascent")
def grad_ascent(cfg: DictConfig) -> UnlearningMethod:
    """Maximise the NLL on the forget set: loss = -NLL_forget."""

    def step(model, batch):
        """Return -NLL on the forget inputs as the loss, logging the NLL."""
        forget = model(**batch["forget"]).loss
        return -forget, {"forget_nll": float(forget.detach())}

    return UnlearningMethod(step, needs_retain=False)


@UNLEARNING_METHODS.register("grad_diff")
def grad_diff(cfg: DictConfig) -> UnlearningMethod:
    """Gradient difference: loss = -NLL_forget + alpha * NLL_retain."""
    alpha = float(cfg.alpha)

    def step(model, batch):
        """Return -NLL_forget + alpha * NLL_retain as the loss, logging both NLLs."""
        forget = model(**batch["forget"]).loss
        retain = model(**batch["retain"]).loss
        return -forget + alpha * retain, {"forget_nll": float(forget.detach()),
                                          "retain_nll": float(retain.detach())}

    return UnlearningMethod(step, needs_retain=True)
