import math
from pathlib import Path

import numpy as np
import pytest
import torch

from nebulalm.model.config import ModelShape
from nebulalm.model.transformer import NebulaLM
from nebulalm.packing.packer import pack_documents, write_shard
from nebulalm.training.checkpoint import latest_valid, load_checkpoint, save_checkpoint, step_path
from nebulalm.training.data import EpochSampler, ShardWindows
from nebulalm.training.optimizer import build_optimizer
from nebulalm.training.schedule import branch_schedule, cosine_multiplier, multiplier, wsd_multiplier
from nebulalm.training.trainer import Trainer, run

SHAPE = ModelShape(name="tiny", n_layer=2, n_head=2, d_model=32, d_mlp=64, context_length=256, vocab_size=40, dropout=0.1)
OPT = {"peak_lr": 3e-3, "betas": [0.9, 0.95], "weight_decay": 0.1}
IDENTITY = {"data": "fixture", "tokenizer": "fixture"}


def _shard(tmp_path: Path, documents: int = 24, name: str = "train.bin") -> Path:
    rng = np.random.default_rng(5)
    docs = []
    for index in range(documents):
        length = int(rng.integers(40, 300))
        ids = [int(value) for value in rng.integers(2, 40, size=length)]
        docs.append((f"doc{index}", ids, length * 3))
    path = tmp_path / name
    write_shard(path, pack_documents(docs, eos_id=1, pad_id=0))
    return path


def _trainer(path: Path, schedule: dict, *, microbatch=2, accumulation=2, seed=3, dropout=None) -> Trainer:
    shape = SHAPE if dropout is None else ModelShape(**{**SHAPE.__dict__, "dropout": dropout})
    torch.manual_seed(seed)
    model = NebulaLM(shape, seed=17014)
    windows = ShardWindows([path])
    return Trainer(
        model, build_optimizer(model, OPT), schedule, windows, EpochSampler(len(windows), seed=11),
        peak_lr=OPT["peak_lr"], microbatch=microbatch, accumulation=accumulation,
    )


CONSTANT = {"kind": "constant"}


def test_shard_windows_match_the_packer(tmp_path):
    windows = ShardWindows([_shard(tmp_path)])
    batch = windows.gather(np.array([0, 1]))
    assert batch["input_ids"].shape == (2, 256)
    assert int(batch["loss_mask"][:, 0].sum()) == 0
    assert windows.utf8_bytes() > 0


def test_truncated_shard_is_rejected(tmp_path):
    path = _shard(tmp_path)
    path.write_bytes(path.read_bytes()[:-10])
    with pytest.raises(ValueError, match="does not match"):
        ShardWindows([path])


def test_sampler_covers_every_window_once_per_epoch_and_resumes():
    sampler = EpochSampler(10, seed=1)
    first = sampler.next_indices(10)
    assert sorted(first.tolist()) == list(range(10))
    sampler.next_indices(3)
    state = sampler.state_dict()
    expected = sampler.next_indices(9)
    restored = EpochSampler(10, seed=1)
    restored.load_state_dict(state)
    assert restored.next_indices(9).tolist() == expected.tolist()


def test_m1_sampler_keeps_the_m0_window_order():
    m0 = EpochSampler(50, seed=4)
    m1 = EpochSampler(56, seed=4, base_size=50)
    for _ in range(2):
        base_order = m0.next_indices(50).tolist()
        extended = m1.next_indices(56).tolist()
        assert [index for index in extended if index < 50] == base_order
        assert sorted(index for index in extended if index >= 50) == list(range(50, 56))


def test_tiny_model_overfits_one_batch(tmp_path):
    path = tmp_path / "one.bin"
    ids = [int(v) for v in np.random.default_rng(0).integers(2, 40, size=200)]
    write_shard(path, pack_documents([("d", ids, 600)], eos_id=1, pad_id=0))
    trainer = _trainer(path, CONSTANT, microbatch=1, accumulation=1, dropout=0.0)
    for _ in range(300):
        row = trainer.train_step()
    assert row["train_nll"] < 0.10


def test_accumulation_matches_one_large_batch(tmp_path):
    path = _shard(tmp_path)
    small = _trainer(path, CONSTANT, microbatch=2, accumulation=2, dropout=0.0)
    large = _trainer(path, CONSTANT, microbatch=4, accumulation=1, dropout=0.0)
    small.train_step()
    large.train_step()
    # Adam's first update is close to lr * sign(grad), so compare the gradients themselves.
    for (name, a), (_, b) in zip(small.model.named_parameters(), large.model.named_parameters()):
        torch.testing.assert_close(a.grad, b.grad, rtol=1e-4, atol=1e-7, msg=name)


def test_schedule_advances_once_per_update_not_per_microbatch(tmp_path):
    schedule = {"kind": "wsd", "warmup": 4, "decay_start": 8, "total": 10}
    trainer = _trainer(_shard(tmp_path), schedule, microbatch=1, accumulation=3)
    lrs = [trainer.train_step()["lr"] for _ in range(4)]
    assert lrs == pytest.approx([OPT["peak_lr"] * (k + 1) / 4 for k in range(4)])
    assert trainer.state.step == 4


def test_wsd_and_cosine_shapes():
    assert wsd_multiplier(0, warmup=10, decay_start=80, total=100) == pytest.approx(0.1)
    assert wsd_multiplier(50, warmup=10, decay_start=80, total=100) == 1.0
    assert wsd_multiplier(100, warmup=10, decay_start=80, total=100) == pytest.approx(0.1)
    assert cosine_multiplier(100, warmup=10, total=100) == pytest.approx(0.1)
    assert cosine_multiplier(10, warmup=10, total=100) == pytest.approx(1.0)


def test_a_branch_is_identical_to_the_trunk_until_its_decay_starts():
    trunk = branch_schedule(100, 8, warmup=20)
    two = branch_schedule(100, 2, warmup=20)
    assert two["decay_start"] == 160 and two["total"] == 200
    for step in range(two["decay_start"]):
        assert multiplier(step, two) == multiplier(step, trunk)
    assert multiplier(199, two) < multiplier(199, trunk)


def test_resume_reproduces_the_uninterrupted_run_exactly(tmp_path):
    path = _shard(tmp_path)
    schedule = {"kind": "wsd", "warmup": 2, "decay_start": 6, "total": 8}
    straight = _trainer(path, schedule)
    run(straight, tmp_path / "straight", max_steps=8, identity=IDENTITY, config={}, checkpoint_every=4, log_every=0)

    first = _trainer(path, schedule)
    run(first, tmp_path / "split", max_steps=4, identity=IDENTITY, config={}, checkpoint_every=4, log_every=0)
    torch.manual_seed(999)  # a fresh process would not share RNG state
    second = _trainer(path, schedule, seed=999)
    run(second, tmp_path / "split", max_steps=8, identity=IDENTITY, config={}, checkpoint_every=4, log_every=0)

    assert second.state.step == straight.state.step == 8
    assert second.state.tokens_seen == straight.state.tokens_seen
    for (name, a), (_, b) in zip(straight.model.named_parameters(), second.model.named_parameters()):
        torch.testing.assert_close(a, b, rtol=0, atol=0, msg=name)
    a_state = straight.optimizer.state_dict()["state"]
    b_state = second.optimizer.state_dict()["state"]
    for key in a_state:
        torch.testing.assert_close(a_state[key]["exp_avg"], b_state[key]["exp_avg"], rtol=0, atol=0)


def test_corrupt_checkpoint_is_rejected_and_the_previous_one_survives(tmp_path):
    good = step_path(tmp_path, 4)
    save_checkpoint(good, {"identity": IDENTITY, "value": 1})
    bad = step_path(tmp_path, 8)
    save_checkpoint(bad, {"identity": IDENTITY, "value": 2})
    bad.write_bytes(bad.read_bytes()[:-20])
    with pytest.raises(ValueError, match="hash"):
        load_checkpoint(bad)
    path, payload = latest_valid(tmp_path, IDENTITY)
    assert path == good and payload["value"] == 1


def test_identity_mismatch_fails_closed(tmp_path):
    path = step_path(tmp_path, 1)
    save_checkpoint(path, {"identity": IDENTITY})
    with pytest.raises(ValueError, match="identity mismatch"):
        load_checkpoint(path, {"data": "other", "tokenizer": "fixture"})


def test_evaluation_is_token_weighted_and_restores_train_mode(tmp_path):
    path = _shard(tmp_path)
    trainer = _trainer(path, CONSTANT)
    trainer.model.train()
    windows = ShardWindows([path])
    result = trainer.evaluate(windows, batch_size=3)
    assert trainer.model.training
    trainer.model.eval()
    total, tokens = 0.0, 0
    with torch.no_grad():
        for index in range(len(windows)):
            batch = windows.gather(np.array([index]))
            nll, count = trainer.model.sequence_nll(batch["input_ids"], batch["loss_mask"], batch["segment_ids"])
            total += float(nll)
            tokens += count
    assert result["valid_tokens"] == tokens == windows.target_tokens()
    assert result["mean_nll"] == pytest.approx(total / tokens, rel=1e-5)
    assert result["bits_per_byte"] == pytest.approx(total / math.log(2) / windows.utf8_bytes(), rel=1e-5)


def test_non_finite_loss_stops_the_run(tmp_path):
    trainer = _trainer(_shard(tmp_path), CONSTANT)
    with torch.no_grad():
        trainer.model.final_ln.weight.fill_(float("nan"))
    with pytest.raises(FloatingPointError):
        trainer.train_step()
