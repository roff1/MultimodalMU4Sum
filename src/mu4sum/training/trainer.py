"""Generic resumable training loop shared by every training method (see mu4sum.training.methods).

What differs between stages is only (dataset, batch plan, step_fn): step_fn(model, batch) returns
(loss, logs). The loop handles accumulation, mixed precision, clipping, logging, exact resumption and, when
validation data is given, model selection: the validation loss (the method's own loss on held-out samples) is
computed every `eval_steps` steps (every epoch end when 0) and at the end, and the final model is the best one.
"""
import math
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

from omegaconf import DictConfig

from mu4sum.training.batching import EpochBatchSampler, PlanFn
from mu4sum.training.checkpoint import (PeftCheckpointIO, best_checkpoint, final_checkpoint, latest_checkpoint,
                                        load_best_info, load_checkpoint, save_best, save_checkpoint, save_final)
from mu4sum.utils.io import append_jsonl
from mu4sum.utils.reproducibility import get_rng_state, seed_worker, set_rng_state, set_seed

StepFn = Callable[[object, object], Tuple[object, Dict[str, float]]]


@dataclass
class TrainSpec:
    """Hyperparameters and logging/checkpoint cadence of a training stage."""

    epochs: int  # ignored when max_steps is set
    grad_accum_steps: int
    lr: float
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.0
    num_workers: int = 0
    bf16: bool = True
    save_steps: int = 100
    log_steps: int = 10
    keep_last_checkpoints: int = 2
    max_steps: int = 0   # optimizer steps; 0 = epochs * steps per epoch
    eval_steps: int = 0  # validate every N optimizer steps; 0 = at every epoch end

    @classmethod
    def from_cfg(cls, section: DictConfig) -> "TrainSpec":
        """Build a spec from a config section, keeping defaults for keys the section lacks."""
        names = cls.__dataclass_fields__
        return cls(**{k: section[k] for k in names if k in section})


def to_device(obj, device):
    """Move a tensor-like (or dict of them) to `device`; None and other objects pass through."""
    if obj is None:
        return None
    if hasattr(obj, "to"):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: to_device(v, device) for k, v in obj.items()}
    return obj


@dataclass
class EvalData:
    """Validation set: same structure as the training inputs (dataset, collate function, batch plan)."""

    dataset: object
    collate_fn: Callable
    plan_fn: PlanFn


class Trainer:
    """Resumable training loop driven by a dataset, a batch plan and a per-batch step_fn."""

    def __init__(self, model, dataset, collate_fn, plan_fn: PlanFn, step_fn: StepFn, spec: TrainSpec,
                 run_dir: Path, seed: int, deterministic: bool = True, io=None,
                 evaluation: Optional[EvalData] = None):
        self.model, self.dataset, self.collate_fn = model, dataset, collate_fn
        self.plan_fn, self.step_fn, self.spec, self.evaluation = plan_fn, step_fn, spec, evaluation
        self.run_dir, self.seed, self.deterministic = Path(run_dir), seed, deterministic
        self.io = io or PeftCheckpointIO()
        self.ckpt_root = self.run_dir / "checkpoints"

    def _make_optimizer(self, total_steps: int):
        """AdamW over trainable params, with linear warmup then linear decay to zero."""
        import torch

        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=self.spec.lr, weight_decay=self.spec.weight_decay)
        warmup = int(total_steps * self.spec.warmup_ratio)

        def lr_lambda(step: int) -> float:
            if warmup and step < warmup:
                return (step + 1) / warmup
            return max(0.0, (total_steps - step) / max(1, total_steps - warmup))

        return optimizer, torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    def _validate(self, device, autocast) -> float:
        """Mean validation loss over the validation batches; leaves model mode and global RNG untouched."""
        import torch
        from torch.utils.data import DataLoader

        ev, rng = self.evaluation, get_rng_state()
        sampler = EpochBatchSampler(ev.plan_fn)
        loader = DataLoader(ev.dataset, batch_sampler=sampler, collate_fn=ev.collate_fn,
                            num_workers=self.spec.num_workers, worker_init_fn=seed_worker,
                            generator=torch.Generator().manual_seed(self.seed))
        self.model.eval()
        losses = []
        with torch.no_grad():
            for batch in loader:
                with autocast():
                    loss, _ = self.step_fn(self.model, to_device(batch, device))
                losses.append(float(loss))
        self.model.train()
        set_rng_state(rng)
        return sum(losses) / len(losses)

    def _record_validation(self, step: int, epoch: int, device, autocast, log_path: Path) -> None:
        """Validate, log, and keep the adapter if it has the lowest validation loss so far."""
        loss = self._validate(device, autocast)
        append_jsonl(log_path, {"step": step, "epoch": epoch, "eval_loss": loss})
        print(f"[Train] step {step} eval_loss={loss:.4f}")
        best = load_best_info(self.ckpt_root)
        if best is None or loss < best["eval_loss"]:
            save_best(self.ckpt_root, step, loss, self.model, self.io)

    def fit(self) -> Path:
        """Train to completion, resuming from the latest checkpoint, and return the final checkpoint.

        If a finished final checkpoint already exists it is returned without training."""
        import torch
        from torch.utils.data import DataLoader

        done = final_checkpoint(self.ckpt_root)
        if done is not None:
            print(f"[Train] Already finished ({done}), nothing to do.")
            return done

        set_seed(self.seed, self.deterministic)
        spec = self.spec
        sampler = EpochBatchSampler(self.plan_fn)
        n_batches = sampler.batches_per_epoch
        steps_per_epoch = math.ceil(n_batches / spec.grad_accum_steps)
        total_steps = spec.max_steps or steps_per_epoch * spec.epochs
        n_epochs = math.ceil(total_steps / steps_per_epoch)
        optimizer, scheduler = self._make_optimizer(total_steps)

        progress = {"step": 0, "epoch": 0, "batch_in_epoch": 0}
        resume_from = latest_checkpoint(self.ckpt_root)
        if resume_from is not None:
            progress = load_checkpoint(resume_from, self.model, optimizer, scheduler, self.io)
            print(f"[Train] Resumed from {resume_from} (step {progress['step']}/{total_steps})")

        device = next(self.model.parameters()).device
        use_amp = spec.bf16 and device.type == "cuda"
        autocast = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if use_amp else nullcontext
        log_path = self.run_dir / "train_log.jsonl"

        self.model.train()
        for epoch in range(progress["epoch"], n_epochs):
            sampler.set_epoch(epoch, progress["batch_in_epoch"])
            # A dedicated generator keeps the DataLoader from consuming the global RNG, which is
            # what makes a resumed run bit-identical to an uninterrupted one.
            loader = DataLoader(self.dataset, batch_sampler=sampler, collate_fn=self.collate_fn,
                                num_workers=spec.num_workers, worker_init_fn=seed_worker,
                                generator=torch.Generator().manual_seed(self.seed + epoch))
            in_group, group_logs = 0, {}
            for batch in loader:
                group_size = min(spec.grad_accum_steps, n_batches - (progress["batch_in_epoch"] - in_group))
                with autocast():
                    loss, logs = self.step_fn(self.model, to_device(batch, device))
                (loss / group_size).backward()
                for k, v in {"loss": float(loss.detach()), **logs}.items():
                    group_logs[k] = group_logs.get(k, 0.0) + v / group_size
                progress["batch_in_epoch"] += 1
                in_group += 1

                if in_group == group_size:
                    if spec.max_grad_norm:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), spec.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    progress["step"] += 1
                    in_group = 0
                    if progress["step"] % spec.log_steps == 0:
                        record = {"step": progress["step"], "epoch": epoch,
                                  "lr": scheduler.get_last_lr()[0], **group_logs}
                        append_jsonl(log_path, record)
                        print(f"[Train] step {progress['step']}/{total_steps} "
                              + " ".join(f"{k}={v:.4f}" for k, v in group_logs.items()))
                    group_logs = {}
                    last = progress["step"] >= total_steps
                    if self.evaluation is not None:
                        end_of_epoch = progress["batch_in_epoch"] == n_batches
                        due = progress["step"] % spec.eval_steps == 0 if spec.eval_steps else end_of_epoch
                        if due or last:
                            self._record_validation(progress["step"], epoch, device, autocast, log_path)
                    if progress["step"] % spec.save_steps == 0 and not last:
                        save_checkpoint(self.ckpt_root, progress["step"], self.model, optimizer, scheduler,
                                        {**progress, "epoch": epoch}, self.io, spec.keep_last_checkpoints)
                    if last:
                        break
            progress["epoch"], progress["batch_in_epoch"] = epoch + 1, 0

        best = best_checkpoint(self.ckpt_root)
        if best is not None:
            self.io.load(self.model, best)
            print(f"[Train] Selected the best checkpoint: {load_best_info(self.ckpt_root)}")
        return save_final(self.ckpt_root, self.model, self.io)
