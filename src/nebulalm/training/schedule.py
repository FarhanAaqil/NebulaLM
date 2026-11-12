"""Learning-rate multipliers. The optimizer LR is peak_lr times the multiplier.

wsd     warmup-stable-decay (blueprint amendment A2). A run of total steps S
        warms up for warmup steps, holds the peak, and decays from
        decay_start to S as 0.1 + 0.9 * (1 - sqrt(progress)).
cosine  linear warmup, then cosine decay to 10% of peak at S (TRAINING.md).

A branch for E passes is a wsd schedule whose total is S_E and whose decay
starts at 0.8 * S_E. Up to that step it is identical to the trunk, so a
branch can resume from the trunk checkpoint at that step.
"""

from __future__ import annotations

import math

FLOOR = 0.1


def warmup_steps(total_steps: int, fraction: float = 0.01) -> int:
    return max(1, math.ceil(fraction * total_steps))


def wsd_multiplier(step: int, *, warmup: int, decay_start: int, total: int) -> float:
    if step < warmup:
        return (step + 1) / warmup
    if step < decay_start:
        return 1.0
    span = max(1, total - decay_start)
    progress = min(1.0, (step - decay_start) / span)
    return FLOOR + (1.0 - FLOOR) * (1.0 - math.sqrt(progress))


def cosine_multiplier(step: int, *, warmup: int, total: int) -> float:
    if step < warmup:
        return (step + 1) / warmup
    span = max(1, total - warmup)
    progress = min(1.0, (step - warmup) / span)
    return FLOOR + (1.0 - FLOOR) * 0.5 * (1.0 + math.cos(math.pi * progress))


def multiplier(step: int, schedule: dict) -> float:
    kind = schedule["kind"]
    if kind == "wsd":
        return wsd_multiplier(
            step, warmup=int(schedule["warmup"]), decay_start=int(schedule["decay_start"]), total=int(schedule["total"])
        )
    if kind == "cosine":
        return cosine_multiplier(step, warmup=int(schedule["warmup"]), total=int(schedule["total"]))
    if kind == "constant":
        return 1.0
    raise ValueError(f"unknown schedule {kind}")


def branch_schedule(steps_per_pass: int, passes: float, warmup: int, decay_fraction: float = 0.2) -> dict:
    """WSD schedule for a run of `passes` corpus passes, decaying over its last 20%."""
    total = max(1, round(steps_per_pass * passes))
    return {"kind": "wsd", "warmup": warmup, "decay_start": round(total * (1.0 - decay_fraction)), "total": total}
