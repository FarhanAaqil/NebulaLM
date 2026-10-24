"""Cluster-level split assignment.

Whole clusters move together. The hash is not adjusted after the assignment
is observed.
"""

from __future__ import annotations

from nebulalm.utils.hashing import keyed_unit_interval

ALGORITHM = "hmac-sha256-v1"
TRAIN_FRACTION = 0.94
VALIDATION_FRACTION = 0.97


def assign_cluster(cluster_id: str, seed: int) -> str:
    unit = keyed_unit_interval(str(seed), f"{ALGORITHM}|{cluster_id}")
    if unit < TRAIN_FRACTION:
        return "train"
    if unit < VALIDATION_FRACTION:
        return "validation"
    return "test"
