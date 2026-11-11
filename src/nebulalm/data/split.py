"""Cluster-level split assignment.

Whole clusters move together. The hash is not adjusted after the assignment
is observed.
"""

from __future__ import annotations

from nebulalm.utils.hashing import keyed_unit_interval

ALGORITHM = "hmac-sha256-v1"
TRAIN_FRACTION = 0.94
VALIDATION_FRACTION = 0.97

# Decision 004: train 91, validation 3, test 3, probe 2, cloze reserve 1.
# Cumulative upper bounds, checked in order.
POOLS_ALGORITHM = "hmac-sha256-pools-v1"
POOL_BOUNDS = (
    ("train", 0.91),
    ("validation", 0.94),
    ("test", 0.97),
    ("probe", 0.99),
    ("cloze_reserve", 1.0),
)


def assign_cluster(cluster_id: str, seed: int) -> str:
    """Three-way split used by the authored sample slice."""
    unit = keyed_unit_interval(str(seed), f"{ALGORITHM}|{cluster_id}")
    if unit < TRAIN_FRACTION:
        return "train"
    if unit < VALIDATION_FRACTION:
        return "validation"
    return "test"


def assign_pool(cluster_id: str, seed: int) -> str:
    """Five-way split used by the production corpus."""
    unit = keyed_unit_interval(str(seed), f"{POOLS_ALGORITHM}|{cluster_id}")
    for name, bound in POOL_BOUNDS:
        if unit < bound:
            return name
    return POOL_BOUNDS[-1][0]
