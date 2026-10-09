import copy
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from mu4sum.training.batching import EpochBatchSampler, PairedDataset, paired_plan, shuffled_plan  # noqa: E402
from mu4sum.training.checkpoint import final_checkpoint, latest_checkpoint  # noqa: E402
from mu4sum.training.trainer import Trainer, TrainSpec  # noqa: E402
from mu4sum.unlearning.methods import UNLEARNING_METHODS  # noqa: E402
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


def make_trainer(tmp_path, step_fn=None, **spec_kw):
    set_seed(0)
    model = Toy()
    data = Data()
    spec = TrainSpec(**{"epochs": 3, "grad_accum_steps": 2, "lr": 1e-2, "bf16": False, "save_steps": 2,
                        "log_steps": 1, "keep_last_checkpoints": 2, **spec_kw})
    step = step_fn or (lambda m, b: (m(**b).loss, {}))
    return model, Trainer(model, data, collate, shuffled_plan(len(data), 3, 0), step, spec, tmp_path,
                          seed=0, io=StateDictIO())


def test_batch_plan_is_pure_function_of_seed_and_epoch():
    plan = shuffled_plan(10, 3, seed=5)
    assert plan(0) == plan(0) and plan(0) != plan(1)
    assert sorted(i for b in plan(2) for i in b) == list(range(10))
    sampler = EpochBatchSampler(plan)
    sampler.set_epoch(1, start=2)
    assert list(sampler) == plan(1)[2:] and len(sampler) == 2


def test_paired_plan_pairs_forget_with_cycling_retain():
    plan = paired_plan(n_forget=5, n_retain=3, batch_size=2, seed=0)
    batches = plan(0)
    assert len(batches) == 3
    for b in batches:
        assert sum(s == "forget" for s, _ in b) <= 2 and sum(s == "retain" for s, _ in b) == 2
    assert sorted(i for b in batches for s, i in b if s == "forget") == list(range(5))
    assert all(s == "forget" for b in paired_plan(5, 0, 2, 0)(0) for s, _ in b)

    ds = PairedDataset([{"v": 1}], [{"v": 2}])
    assert ds[("retain", 0)] == {"v": 2, "_side": "retain"}


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


def test_unlearning_losses_push_forget_up_and_retain_down():
    model = Toy().eval()
    forget = {"x": torch.randn(4, 4), "labels": torch.randn(4, 1)}
    retain = {"x": torch.randn(4, 4), "labels": torch.randn(4, 1)}
    f, r = model(**forget).loss.item(), model(**retain).loss.item()

    ascent = UNLEARNING_METHODS.get("grad_ascent")(OmegaConf.create({"alpha": 0.0}))
    assert not ascent.needs_retain
    loss, logs = ascent.step_fn(model, {"forget": forget, "retain": None})
    assert loss.item() == pytest.approx(-f) and logs["forget_nll"] == pytest.approx(f)

    diff = UNLEARNING_METHODS.get("grad_diff")(OmegaConf.create({"alpha": 0.5}))
    assert diff.needs_retain
    loss, _ = diff.step_fn(model, {"forget": forget, "retain": retain})
    assert loss.item() == pytest.approx(-f + 0.5 * r)

    with pytest.raises(KeyError, match="grad_ascent"):
        UNLEARNING_METHODS.get("missing")
