"""Training step, evaluation, and the run loop.

One optimizer update = `accumulation` microbatches. Each microbatch's summed
NLL is divided by the valid-token count of the whole update, so the
accumulated gradient equals the gradient of one large token-weighted batch.
Under FP16 the scaler unscales before clipping; an update the scaler skips
does not advance the step counter or the schedule.
"""

from __future__ import annotations

import json
import math
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import nn

from nebulalm.evaluation.metrics import metric_block
from nebulalm.training.checkpoint import (
    latest_valid,
    restore_rng,
    retain,
    rng_state,
    save_checkpoint,
    step_path,
    write_json_atomic,
)
from nebulalm.training.data import EpochSampler, ShardWindows
from nebulalm.training.schedule import multiplier


@dataclass
class TrainState:
    step: int = 0
    tokens_seen: int = 0
    skipped_steps: int = 0
    best_val_bpb: float = math.inf
    best_step: int = -1
    history: list[dict] = field(default_factory=list)


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        schedule: dict,
        train: ShardWindows,
        sampler: EpochSampler,
        *,
        peak_lr: float,
        microbatch: int,
        accumulation: int,
        clip_norm: float = 1.0,
        precision: str = "fp32",
        device: str = "cpu",
    ) -> None:
        if precision not in {"fp32", "fp16", "bf16"}:
            raise ValueError(f"unknown precision {precision}")
        if precision != "fp32" and not device.startswith("cuda"):
            raise ValueError("mixed precision needs a CUDA device")
        self.model = model.to(device)
        self.optimizer = optimizer
        self.schedule = schedule
        self.train = train
        self.sampler = sampler
        self.peak_lr = peak_lr
        self.microbatch = microbatch
        self.accumulation = accumulation
        self.clip_norm = clip_norm
        self.precision = precision
        self.device = device
        self.scaler = torch.amp.GradScaler("cuda", enabled=precision == "fp16")
        self.state = TrainState()

    def _autocast(self):
        if self.precision == "fp32":
            return nullcontext()
        dtype = torch.float16 if self.precision == "fp16" else torch.bfloat16
        return torch.autocast(device_type="cuda", dtype=dtype)

    def _set_lr(self) -> float:
        lr = self.peak_lr * multiplier(self.state.step, self.schedule)
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        return lr

    def train_step(self) -> dict:
        self.model.train()
        lr = self._set_lr()
        batches = [self.train.gather(self.sampler.next_indices(self.microbatch)) for _ in range(self.accumulation)]
        total_tokens = sum(int(batch["loss_mask"][:, 1:].sum()) for batch in batches)
        if total_tokens == 0:
            raise ValueError("an optimizer update with no target tokens")
        self.optimizer.zero_grad(set_to_none=True)
        summed = 0.0
        for batch in batches:
            batch = {key: value.to(self.device, non_blocking=True) for key, value in batch.items()}
            with self._autocast():
                nll, _count = self.model.sequence_nll(batch["input_ids"], batch["loss_mask"], batch["segment_ids"])
            loss = nll / total_tokens
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at step {self.state.step}")
            self.scaler.scale(loss).backward()
            summed += float(nll.detach())
        self.scaler.unscale_(self.optimizer)
        grad_norm = float(torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_norm))
        scale_before = self.scaler.get_scale() if self.scaler.is_enabled() else 1.0
        self.scaler.step(self.optimizer)
        self.scaler.update()
        skipped = self.scaler.is_enabled() and self.scaler.get_scale() < scale_before
        if skipped:
            self.state.skipped_steps += 1
        else:
            if not math.isfinite(grad_norm):
                raise FloatingPointError(f"non-finite gradient norm at step {self.state.step}")
            self.state.step += 1
            self.state.tokens_seen += total_tokens
        return {
            "step": self.state.step,
            "lr": lr,
            "train_nll": summed / total_tokens,
            "grad_norm": grad_norm,
            "tokens": total_tokens,
            "skipped": skipped,
            "passes": self.sampler.passes(),
            "scale": self.scaler.get_scale() if self.scaler.is_enabled() else None,
        }

    @torch.no_grad()
    def evaluate(self, windows: ShardWindows, batch_size: int = 32) -> dict:
        was_training = self.model.training
        self.model.eval()
        total_nll, total_tokens, total_bytes = 0.0, 0, 0
        for start in range(0, len(windows), batch_size):
            indices = torch.arange(start, min(start + batch_size, len(windows))).numpy()
            batch = {key: value.to(self.device) for key, value in windows.gather(indices).items()}
            with self._autocast():
                nll, count = self.model.sequence_nll(batch["input_ids"], batch["loss_mask"], batch["segment_ids"])
            total_nll += float(nll)
            total_tokens += count
            total_bytes += int(batch["byte_count"].sum())
        self.model.train(was_training)
        return metric_block(total_nll, total_tokens, total_bytes)

    def payload(self, identity: dict, config: dict) -> dict:
        return {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "sampler": self.sampler.state_dict(),
            "schedule": dict(self.schedule),
            "state": vars(self.state),
            "rng": rng_state(),
            "identity": identity,
            "config": config,
        }

    def load_payload(self, payload: dict) -> None:
        if payload["schedule"] != self.schedule:
            raise ValueError("checkpoint schedule differs from this run; use a branch instead of a resume")
        self.model.load_state_dict(payload["model"])
        self.optimizer.load_state_dict(payload["optimizer"])
        self.scaler.load_state_dict(payload["scaler"])
        self.sampler.load_state_dict(payload["sampler"])
        self.state = TrainState(**payload["state"])
        restore_rng(payload["rng"])

    def branch_from(self, payload: dict) -> None:
        """Start a cooldown branch: same weights, optimizer, data position and RNG; new schedule."""
        schedule = self.schedule
        payload = dict(payload, schedule=schedule)
        self.load_payload(payload)


def run(
    trainer: Trainer,
    run_dir: Path,
    *,
    max_steps: int,
    identity: dict,
    config: dict,
    val: ShardWindows | None = None,
    eval_every: int = 0,
    checkpoint_every: int = 0,
    log_every: int = 10,
    extra_checkpoints: set[int] | None = None,
    resume: bool = True,
) -> TrainState:
    """Train to max_steps, resuming from the newest valid checkpoint in run_dir."""
    run_dir.mkdir(parents=True, exist_ok=True)
    if resume:
        found = latest_valid(run_dir, identity)
        if found:
            trainer.load_payload(found[1])
    best_path = run_dir / "checkpoints" / "best.pt"
    log_path = run_dir / "train_log.jsonl"
    extra = extra_checkpoints or set()
    started = time.perf_counter()
    window_tokens = 0
    while trainer.state.step < max_steps:
        row = trainer.train_step()
        window_tokens += row["tokens"]
        step = trainer.state.step
        if row["skipped"]:
            continue
        if log_every and step % log_every == 0:
            elapsed = time.perf_counter() - started
            row["tokens_per_second"] = window_tokens / elapsed if elapsed else None
            started, window_tokens = time.perf_counter(), 0
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row) + "\n")
        evaluated = None
        if val is not None and eval_every and (step % eval_every == 0 or step == max_steps):
            evaluated = trainer.evaluate(val)
            trainer.state.history.append({"step": step, "tokens_seen": trainer.state.tokens_seen, **evaluated})
            if evaluated["bits_per_byte"] < trainer.state.best_val_bpb:
                trainer.state.best_val_bpb = evaluated["bits_per_byte"]
                trainer.state.best_step = step
                save_checkpoint(best_path, trainer.payload(identity, config))
        if (checkpoint_every and step % checkpoint_every == 0) or step == max_steps or step in extra:
            path = step_path(run_dir, step)
            save_checkpoint(path, trainer.payload(identity, config))
            keep = {best_path} | {step_path(run_dir, kept) for kept in extra}
            retain(run_dir, keep)
            write_json_atomic(run_dir / "state.json", {"step": step, **{k: v for k, v in vars(trainer.state).items() if k != "history"}})
    return trainer.state
