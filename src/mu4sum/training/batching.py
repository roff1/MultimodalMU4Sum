"""Deterministic, resumable batch ordering.

The batch order of an epoch is a pure function of (seed, epoch), so a run resumed from a checkpoint
sees exactly the batches an uninterrupted run would have seen (it just skips the ones already consumed).
"""
from typing import Callable, List

import numpy as np

PlanFn = Callable[[int], List[list]]


def shuffled_plan(n: int, batch_size: int, seed: int) -> PlanFn:
    """Plan that splits a permutation of range(n), seeded by seed + epoch, into index batches."""

    def plan(epoch: int) -> List[list]:
        """Return the index batches for the given epoch."""
        perm = np.random.RandomState(seed + epoch).permutation(n).tolist()
        return [perm[i:i + batch_size] for i in range(0, n, batch_size)]

    return plan


def paired_plan(n_forget: int, n_retain: int, batch_size: int, seed: int) -> PlanFn:
    """Each batch = `batch_size` forget samples + `batch_size` retain samples (retain cycles if shorter).
    Indices are ("forget"|"retain", i) tuples; `n_retain == 0` yields forget-only batches."""

    def plan(epoch: int) -> List[list]:
        """Return the paired forget/retain batches for the given epoch."""
        rs =np.random.RandomState(seed + epoch)
        forget_perm = rs.permutation(n_forget).tolist()
        retain_perm = rs.permutation(n_retain).tolist() if n_retain else []
        batches = []
        for b, start in enumerate(range(0, n_forget, batch_size)):
            batch = [("forget", i) for i in forget_perm[start:start + batch_size]]
            if retain_perm:
                batch += [("retain", retain_perm[(b * batch_size + j) % n_retain]) for j in range(batch_size)]
            batches.append(batch)
        return batches

    return plan


class EpochBatchSampler:
    """Batch sampler for torch DataLoader(batch_sampler=...) that can start mid-epoch."""

    def __init__(self, plan_fn: PlanFn):
        self.plan_fn = plan_fn
        self.batches_per_epoch = len(plan_fn(0))
        self.epoch, self.start = 0, 0

    def set_epoch(self, epoch: int, start: int = 0) -> None:
        """Choose the epoch to iterate and the batch index to start from (mid-epoch resume)."""
        self.epoch, self.start = epoch, start

    def __iter__(self):
        yield from self.plan_fn(self.epoch)[self.start:]

    def __len__(self) -> int:
        return self.batches_per_epoch - self.start


class PairedDataset:
    """Serves ("forget"|"retain", i) keys produced by `paired_plan`."""

    def __init__(self, forget, retain=None):
        self.forget, self.retain = forget, retain

    def __len__(self) -> int:
        return len(self.forget)

    def __getitem__(self, key):
        """Return the forget or retain sample for the key, tagged with its side under "_side"."""
        side, i = key
        return {**(self.forget if side == "forget" else self.retain)[i], "_side": side}
