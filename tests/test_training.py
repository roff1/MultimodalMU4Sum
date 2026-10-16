import re
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402
from torch import nn  # noqa: E402

from mu4sum.preprocessing.answers import format_answer  # noqa: E402
from mu4sum.training.batching import EpochBatchSampler, PairedDataset, mixed_plan, shuffled_plan  # noqa: E402
from mu4sum.training.checkpoint import final_checkpoint, latest_checkpoint, load_best_info  # noqa: E402
from mu4sum.training.collate import PairedCollator, TokenWiseCollator  # noqa: E402
from mu4sum.training.methods import TRAINING_METHODS  # noqa: E402
from mu4sum.training.trainer import EvalData, Trainer, TrainSpec  # noqa: E402
from mu4sum.utils.reproducibility import get_rng_state, set_rng_state, set_seed  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402


class Toy(nn.Module):
    """Tiny regressor with dropout (so RNG restoration matters). Mimics `model(**batch).loss`."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(4, 8), nn.Dropout(0.3), nn.Linear(8, 1))

    def forward(self, x, labels):
        return SimpleNamespace(loss=((self.net(x) - labels) ** 2).mean())


class StateDictIO:
    def save(self, model, path):
        torch.save(model.state_dict(), path / "w.pt")

    def load(self, model, path):
        model.load_state_dict(torch.load(path / "w.pt"))


class Data:
    def __init__(self, n=10):
        g = torch.Generator().manual_seed(0)
        self.x, self.y = torch.randn(n, 4, generator=g), torch.randn(n, 1, generator=g)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return {"x": self.x[i], "labels": self.y[i]}


def collate(samples):
    return {"x": torch.stack([s["x"] for s in samples]), "labels": torch.stack([s["labels"] for s in samples])}


def make_trainer(tmp_path, step_fn=None, evaluation=None, **spec_kw):
    set_seed(0)
    model = Toy()
    data = Data()
    spec = TrainSpec(**{"epochs": 3, "grad_accum_steps": 2, "lr": 1e-2, "bf16": False, "save_steps": 2,
                        "log_steps": 1, "keep_last_checkpoints": 2, **spec_kw})
    step = step_fn or (lambda m, b: (m(**b).loss, {}))
    return model, Trainer(model, data, collate, shuffled_plan(len(data), 3, 0), step, spec, tmp_path,
                          seed=0, io=StateDictIO(), evaluation=evaluation)


def test_batch_plan_is_pure_function_of_seed_and_epoch():
    plan = shuffled_plan(10, 3, seed=5)
    assert plan(0) == plan(0) and plan(0) != plan(1)
    assert sorted(i for b in plan(2) for i in b) == list(range(10))
    sampler = EpochBatchSampler(plan)
    sampler.set_epoch(1, start=2)
    assert list(sampler) == plan(1)[2:] and len(sampler) == 2


def test_resume_is_bit_identical_to_uninterrupted_run(tmp_path):
    model_a, trainer_a = make_trainer(tmp_path / "a")
    trainer_a.fit()

    # Interrupted run: crash on the 7th micro-batch (a checkpoint exists at optimizer step 2 and 4).
    calls = {"n": 0}

    def crashing(m, b):
        calls["n"] += 1
        if calls["n"] == 7:
            raise KeyboardInterrupt
        return m(**b).loss, {}

    _, trainer_b = make_trainer(tmp_path / "b", step_fn=crashing)
    with pytest.raises(KeyboardInterrupt):
        trainer_b.fit()
    assert latest_checkpoint(tmp_path / "b" / "checkpoints") is not None
    assert final_checkpoint(tmp_path / "b" / "checkpoints") is None

    model_b, trainer_b2 = make_trainer(tmp_path / "b")  # fresh process: new model, same dir
    trainer_b2.fit()
    for pa, pb in zip(model_a.parameters(), model_b.parameters()):
        assert torch.equal(pa, pb)
    assert final_checkpoint(tmp_path / "b" / "checkpoints") is not None


def test_finished_run_is_not_retrained(tmp_path):
    _, trainer = make_trainer(tmp_path)
    trainer.fit()
    calls = {"n": 0}

    def counting(m, b):
        calls["n"] += 1
        return m(**b).loss, {}

    _, again = make_trainer(tmp_path, step_fn=counting)
    again.fit()
    assert calls["n"] == 0


def test_old_checkpoints_are_pruned(tmp_path):
    _, trainer = make_trainer(tmp_path, save_steps=1, keep_last_checkpoints=2)
    trainer.fit()
    assert len(list((tmp_path / "checkpoints").glob("step-*"))) <= 2


def test_rng_state_roundtrip():
    set_seed(1)
    state = get_rng_state()
    a = (torch.rand(3), __import__("random").random(), __import__("numpy").random.rand())
    set_rng_state(state)
    b = (torch.rand(3), __import__("random").random(), __import__("numpy").random.rand())
    assert torch.equal(a[0], b[0]) and a[1:] == b[1:]


def test_mixed_plan_shuffles_forget_and_retain_together():
    plan = mixed_plan(n_forget=5, n_retain=3, batch_size=2, seed=0)
    batches = plan(0)
    assert len(batches) == 4 and plan(0) == batches and plan(1) != batches
    assert sorted(k for b in batches for k in b) == sorted([("forget", i) for i in range(5)]
                                                           + [("retain", i) for i in range(3)])
    assert any({s for s, _ in b} == {"forget", "retain"} for b in batches)
    assert all(s == "forget" for b in mixed_plan(5, 0, 2, 0)(0) for s, _ in b)

    ds = PairedDataset([{"v": 1}], [{"v": 2}])
    assert len(ds) == 2 and ds[("retain", 0)] == {"v": 2, "_side": "retain"}
    assert len(PairedDataset([{"v": 1}])) == 1


def test_max_steps_overrides_epochs_and_stops_mid_epoch(tmp_path):
    calls = {"n": 0}

    def counting(m, b):
        calls["n"] += 1
        return m(**b).loss, {}

    _, trainer = make_trainer(tmp_path, step_fn=counting, max_steps=3)  # 4 batches/epoch, 2 per step
    trainer.fit()
    assert calls["n"] == 6 and final_checkpoint(tmp_path / "checkpoints") is not None

    _, longer = make_trainer(tmp_path / "long", step_fn=counting, max_steps=9)  # more than the 3 epochs of steps
    calls["n"] = 0
    longer.fit()
    assert calls["n"] == 18


def test_validation_selects_the_best_checkpoint(tmp_path):
    val_losses, snapshots, seen = [3.0, 1.0, 2.0], {}, {"val_batches": 0}

    def step(m, b):
        if m.training:
            return m(**b).loss, {}
        k = seen["val_batches"] // 4  # the validation set has 4 batches
        if seen["val_batches"] % 4 == 0:
            snapshots[k] = {n: p.detach().clone() for n, p in m.state_dict().items()}
        seen["val_batches"] += 1
        return torch.tensor(val_losses[k]), {}

    model, trainer = make_trainer(tmp_path, step_fn=step, evaluation=EvalData(Data(), collate, shuffled_plan(10, 3, 0)))
    trainer.fit()
    assert seen["val_batches"] == 12  # validated at the end of each of the 3 epochs
    assert load_best_info(tmp_path / "checkpoints") == {"step": 4, "eval_loss": 1.0}
    for name, value in model.state_dict().items():  # the final model is the epoch-2 one, not the last one
        assert torch.equal(value, snapshots[1][name])
    assert final_checkpoint(tmp_path / "checkpoints") is not None


def test_validation_does_not_change_the_training_trajectory(tmp_path):
    model_a, trainer_a = make_trainer(tmp_path / "a")
    trainer_a.fit()

    seen = {"val_batches": 0}

    def step(m, b):
        if m.training:
            return m(**b).loss, {}
        torch.rand(3)  # a validation that draws from the global RNG must not leak into training
        seen["val_batches"] += 1
        return torch.tensor(100.0 - seen["val_batches"]), {}  # keeps improving: the last model is the best one

    model_b, trainer_b = make_trainer(tmp_path / "b", step_fn=step, eval_steps=1,
                                      evaluation=EvalData(Data(), collate, shuffled_plan(10, 3, 0)))
    trainer_b.fit()
    assert seen["val_batches"] == 6 * 4 and load_best_info(tmp_path / "b" / "checkpoints")["step"] == 6
    for pa, pb in zip(model_a.parameters(), model_b.parameters()):
        assert torch.equal(pa, pb)


VOCAB, SEQ = 7, 6


class ToyLM(nn.Module):
    """Tiny LM returning `.logits` like the VLM does; `labels` / `token_kind` must never reach it."""

    def __init__(self):
        super().__init__()
        self.emb, self.out = nn.Embedding(VOCAB, 8), nn.Linear(8, VOCAB)

    def forward(self, input_ids):
        return SimpleNamespace(logits=self.out(self.emb(input_ids)))


def lm_batch(n=4, seed=0, kinds=False):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, VOCAB, (n, SEQ), generator=g)
    labels = ids.clone()
    labels[:, :2] = -100  # prompt
    labels[0, -1] = -100  # padding in one sample: answers may have different lengths
    batch = {"input_ids": ids, "labels": labels}
    if kinds:
        kind = torch.where(torch.rand(n, SEQ, generator=g) < 0.5, 1, -1)
        batch["token_kind"] = torch.where(labels == -100, 0, kind)
    return batch


def toy_method(name, **cfg):
    return TRAINING_METHODS.get(name)(OmegaConf.create(cfg))


def token_nll(model, batch):
    """Independent computation: NLL of every position ([n, SEQ-1]), 0 outside the answer."""
    logits = model(input_ids=batch["input_ids"]).logits[:, :-1]
    labels = batch["labels"][:, 1:]
    nll = F.cross_entropy(logits.reshape(-1, VOCAB), labels.reshape(-1), ignore_index=-100, reduction="none")
    return nll.view(len(labels), -1)


def sample_nll(model, batch):
    """Per-sample mean NLL over the answer tokens."""
    mask = batch["labels"][:, 1:] != -100
    return token_nll(model, batch).sum(1) / mask.sum(1)


def test_methods_declare_what_they_train_on_and_start_from():
    cfg = dict(alpha=1.0, beta=0.1, gamma=1.0, refusal_text="no")
    table = {n: (m.data, m.init, m.needs_reference)
             for n, m in ((n, toy_method(n, **cfg)) for n in TRAINING_METHODS.names())}
    assert table == {
        "original": ("all", "base", False),
        "retrain": ("retain", "base", False),
        "finetune": ("retain", "original", False),
        "default": ("forget", "original", False),
        "ga": ("forget", "original", False),
        "ga_gd": ("forget+retain", "original", False),
        "ga_kd": ("forget+retain", "original", True),
        "npo": ("forget+retain", "original", True),
        "tw_ga_gd": ("tokenwise", "original", False),
        "tw_ga_kd": ("tokenwise", "original", True),
    }
    default = toy_method("default", refusal_text="I can't assist you on this")
    assert default.forget_target == "I can't assist you on this" and default.paired and not default.needs_retain
    assert all(toy_method(n, **cfg).forget_target is None for n in TRAINING_METHODS.names() if n != "default")
    assert toy_method("ga").paired and not toy_method("ga").needs_retain
    assert toy_method("ga_gd", alpha=1.0).needs_retain and toy_method("npo", **cfg).needs_retain
    assert not toy_method("original").paired and not toy_method("finetune").paired
    assert toy_method("tw_ga_gd", **cfg).token_wise and not toy_method("ga_gd", **cfg).token_wise
    with pytest.raises(KeyError, match="ga_gd"):
        TRAINING_METHODS.get("missing")


def test_losses_are_per_sample_token_means_over_the_answer_only():
    model, batch = ToyLM(), lm_batch()
    for name in ("original", "retrain", "finetune"):
        loss, logs = toy_method(name).step(model, batch, None)
        assert loss.item() == pytest.approx(sample_nll(model, batch).mean().item()) and logs == {}
    # not the token mean of the whole batch: the padded sample has fewer answer tokens
    token_mean = (token_nll(model, batch).sum() / (batch["labels"][:, 1:] != -100).sum()).item()
    assert loss.item() != pytest.approx(token_mean)
    # `labels` and `token_kind` are never forwarded to the model (ToyLM.forward accepts input_ids only)
    other = {**batch, "token_kind": torch.ones_like(batch["labels"])}
    assert toy_method("original").step(model, other, None)[0].item() == pytest.approx(loss.item())


def test_ga_and_ga_gd_losses_push_forget_up_and_retain_down():
    model = ToyLM().eval()
    forget, retain = lm_batch(seed=1), lm_batch(n=2, seed=2)
    f, r = sample_nll(model, forget), sample_nll(model, retain)

    loss, logs = toy_method("ga").step(model, {"forget": forget, "retain": None}, None)
    assert loss.item() == pytest.approx(-f.mean().item()) and logs["forget_nll"] == pytest.approx(f.mean().item())

    # one mixed batch: forget samples weigh -1, retain samples +alpha, normalised by the whole batch
    loss, _ = toy_method("ga_gd", alpha=0.5).step(model, {"forget": forget, "retain": retain}, None)
    assert loss.item() == pytest.approx(((-f.sum() + 0.5 * r.sum()) / 6).item())
    loss, _ = toy_method("ga_gd", alpha=0.5).step(model, {"forget": None, "retain": retain}, None)
    assert loss.item() == pytest.approx((0.5 * r.mean()).item())


def test_default_trains_on_the_forget_view_only():
    model, forget = ToyLM(), lm_batch(seed=1)
    loss, _ = toy_method("default", refusal_text="x").step(model, {"forget": forget, "retain": None}, None)
    assert loss.item() == pytest.approx(sample_nll(model, forget).mean().item())


def test_ga_kd_is_a_kl_to_the_reference_on_retain_answers():
    torch.manual_seed(0)
    model, reference = ToyLM().eval(), ToyLM().eval()
    forget, retain = lm_batch(seed=1), lm_batch(seed=2)
    method = toy_method("ga_kd", alpha=2.0)
    batch = {"forget": forget, "retain": retain}
    f = sample_nll(model, forget).mean().item()

    loss, logs = method.step(model, batch, model)  # reference identical to the model: no distillation cost
    assert logs["retain_kl"] == pytest.approx(0.0, abs=1e-6) and loss.item() == pytest.approx(-f, abs=1e-5)

    reference.requires_grad_(False)
    loss, logs = method.step(model, batch, reference)
    mask = retain["labels"][:, 1:] != -100
    kl_tokens = F.kl_div(F.log_softmax(model(input_ids=retain["input_ids"]).logits[:, :-1], -1),
                         F.log_softmax(reference(input_ids=retain["input_ids"]).logits[:, :-1], -1),
                         log_target=True, reduction="none").sum(-1)
    kl = ((kl_tokens * mask).sum(1) / mask.sum(1)).mean()  # per-sample mean, then mean over samples
    assert logs["retain_kl"] == pytest.approx(kl.item(), rel=1e-4) and kl.item() > 0
    assert loss.item() == pytest.approx(-f + 2.0 * kl.item(), rel=1e-4)
    loss.backward()
    assert all(p.grad is None for p in reference.parameters())  # the reference is never trained


def test_npo_loss_uses_mean_token_log_ratio_and_a_retain_term():
    torch.manual_seed(0)
    model, reference = ToyLM(), ToyLM().eval().requires_grad_(False)
    forget, retain = lm_batch(seed=1), lm_batch(n=2, seed=2)
    method = toy_method("npo", alpha=0.7, beta=0.5, gamma=1.5)

    loss, logs = method.step(model, {"forget": forget, "retain": None}, model)  # log-ratio 0: loss = gamma*(2/beta)*log 2
    assert logs["forget_log_ratio"] == pytest.approx(0.0, abs=1e-5)
    assert loss.item() == pytest.approx(1.5 * 4.0 * torch.log(torch.tensor(2.0)).item(), rel=1e-5)

    batch = {"forget": forget, "retain": retain}
    loss, _ = method.step(model, batch, reference)
    ratio = sample_nll(reference, forget) - sample_nll(model, forget)  # mean token log-prob difference
    npo = -(2 / 0.5) * F.logsigmoid(-0.5 * ratio)
    expected = 0.7 * sample_nll(model, retain).mean() + 1.5 * npo.mean()
    assert loss.item() == pytest.approx(expected.item(), rel=1e-4)
    loss.backward()
    assert all(p.grad is None for p in reference.parameters())

    model.zero_grad()  # a descent step on the forget term makes the forget answers less likely
    method.step(model, {"forget": forget, "retain": None}, reference)[0].backward()
    before = sample_nll(model, forget).mean().item()
    with torch.no_grad():
        for p in model.parameters():
            p -= 0.01 * p.grad
    assert sample_nll(model, forget).mean().item() > before


def test_token_wise_losses_weigh_forget_and_retain_tokens_separately():
    torch.manual_seed(0)
    model, reference = ToyLM().eval(), ToyLM().eval().requires_grad_(False)
    batch = lm_batch(seed=3, kinds=True)
    kind = batch["token_kind"][:, 1:]
    nll = token_nll(model, batch)
    n_tokens = (kind != 0).sum(1)

    loss, logs = toy_method("tw_ga_gd", alpha=0.5, beta=2.0).step(model, batch, None)
    weights = 0.5 * (kind > 0) - 2.0 * (kind < 0)
    assert loss.item() == pytest.approx(((weights * nll).sum(1) / n_tokens).mean().item(), rel=1e-5)
    assert logs["forget_token_nll"] == pytest.approx(nll[kind < 0].mean().item(), rel=1e-5)

    loss, logs = toy_method("tw_ga_kd", alpha=0.5, beta=2.0).step(model, batch, model)  # no drift: KL term is 0
    assert logs["retain_token_kl"] == pytest.approx(0.0, abs=1e-6)
    assert loss.item() == pytest.approx(((-2.0 * nll * (kind < 0)).sum(1) / n_tokens).mean().item(), rel=1e-5)

    loss, logs = toy_method("tw_ga_kd", alpha=0.5, beta=2.0).step(model, batch, reference)
    kl = F.kl_div(F.log_softmax(model(input_ids=batch["input_ids"]).logits[:, :-1], -1),
                  F.log_softmax(reference(input_ids=batch["input_ids"]).logits[:, :-1], -1),
                  log_target=True, reduction="none").sum(-1)
    values = -2.0 * nll * (kind < 0) + 0.5 * kl * (kind > 0)
    assert loss.item() == pytest.approx((values.sum(1) / n_tokens).mean().item(), rel=1e-4)
    assert logs["retain_token_kl"] == pytest.approx(kl[kind > 0].mean().item(), rel=1e-4)
    loss.backward()
    assert all(p.grad is None for p in reference.parameters())


class FakeTokenizer:
    """Whitespace-ish tokenizer with character offsets (words, newline runs and the <eos> marker)."""

    is_fast = True

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        return {"offset_mapping": [m.span() for m in re.finditer(r"<eos>|[^\s<]+|\n+", text)]}


class FakeBase:
    eos = "<eos>"
    processor = SimpleNamespace(tokenizer=FakeTokenizer())

    def __init__(self, labels):
        self.labels = labels

    def __call__(self, samples):
        return {"labels": self.labels.clone()}


def token_wise_sample():
    target, spans = format_answer({"a": "x y", "b": "z"}, ["a", "b"])
    return {"id": "s1", "target": target, "spans": spans}


def test_token_wise_collator_marks_forget_blocks_and_answer_tokens():
    sample = token_wise_sample()
    # tokens: {"aspect_summaries": | {"a": | "x | y", | "b": | "z"}} | <eos>   (a = tokens 2-4, b = 5-6)
    assert TokenWiseCollator(FakeBase(None), ["b"]).kinds(sample) == [1, 1, 1, 1, -1, -1, 1]
    assert TokenWiseCollator(FakeBase(None), ["a"]).kinds(sample) == [1, -1, -1, -1, 1, 1, 1]

    labels = torch.tensor([[-100, -100, *range(7), -100]])  # prompt, answer (7 tokens), padding
    out = TokenWiseCollator(FakeBase(labels), ["b"])([sample])
    assert out["token_kind"].tolist() == [[0, 0, 1, 1, 1, 1, -1, -1, 1, 0]]
    bad = torch.tensor([[-100, -100, *range(6), -100]])
    with pytest.raises(ValueError, match="misaligned"):
        TokenWiseCollator(FakeBase(bad), ["b"])([sample])


class LMData:
    def __init__(self, seed):
        self.batch = lm_batch(8, seed)

    def __len__(self):
        return len(self.batch["input_ids"])

    def __getitem__(self, i):
        return {k: v[i] for k, v in self.batch.items()}


def lm_collate(samples):
    return {k: torch.stack([s[k] for s in samples]) for k in ("input_ids", "labels")}


def test_unlearning_method_runs_in_the_trainer_with_a_frozen_reference(tmp_path):
    set_seed(0)
    model, reference = ToyLM(), ToyLM()
    reference.load_state_dict(model.state_dict())
    reference.requires_grad_(False)
    forget, retain = LMData(1), LMData(2)
    method = toy_method("ga_kd", alpha=1.0)
    before = sample_nll(model, forget.batch).mean().item()

    spec = TrainSpec(epochs=3, grad_accum_steps=1, lr=1e-2, bf16=False, save_steps=100, log_steps=1)
    Trainer(model, PairedDataset(forget, retain), PairedCollator(lm_collate),
            mixed_plan(len(forget), len(retain), 4, 0), lambda m, b: method.step(m, b, reference),
            spec, tmp_path, seed=0, io=StateDictIO()).fit()

    assert sample_nll(model, forget.batch).mean().item() > before  # the forget answers became less likely
    assert final_checkpoint(tmp_path / "checkpoints") is not None
