"""Learn / Store / Leak readouts on one checkpoint (blueprint, technical formulation).

Learn   per-document bits per byte
Store   membership AUC, greedy extractability, canary exposure in bits
Leak    difference-in-differences benefit per probe group and transfer ratio

Documents are scored as [eos] + tokens in 256-token chunks that overlap by one
token, so every document token is a target exactly once and the first token is
predicted from the document boundary, as in training.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import torch

Encode = Callable[[str], list[int]]


@torch.no_grad()
def document_nll(model, ids: list[int], eos_id: int, device: str = "cpu", context: int = 256) -> tuple[float, int]:
    """Summed NLL (nats) over every token of one document, and the token count."""
    sequence = [eos_id] + list(ids)
    total, count = 0.0, 0
    step = context - 1
    for start in range(0, max(1, len(sequence) - 1), step):
        chunk = sequence[start : start + context]
        if len(chunk) < 2:
            break
        tensor = torch.tensor([chunk], device=device)
        mask = torch.ones_like(tensor)
        nll, tokens = model.sequence_nll(tensor, mask)
        total += float(nll)
        count += tokens
    return total, count


def document_bpb(model, texts: list[str], encode: Encode, eos_id: int, device: str = "cpu") -> list[float]:
    was_training = model.training
    model.eval()
    values = []
    for text in texts:
        nll, _ = document_nll(model, encode(text), eos_id, device)
        values.append(nll / math.log(2) / len(text.encode("utf-8")))
    model.train(was_training)
    return values


def auc(positive: list[float], negative: list[float]) -> float:
    """Probability a positive scores above a negative (ties count half). Mann-Whitney U / (n_p n_n)."""
    scores = np.concatenate([positive, negative])
    order = scores.argsort(kind="mergesort")
    ranks = np.empty(len(scores))
    sorted_scores = scores[order]
    index = 0
    while index < len(scores):
        end = index
        while end + 1 < len(scores) and sorted_scores[end + 1] == sorted_scores[index]:
            end += 1
        ranks[order[index : end + 1]] = (index + end) / 2 + 1
        index = end + 1
    n_pos, n_neg = len(positive), len(negative)
    return float((ranks[:n_pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def membership_auc(member_losses: list[float], non_member_losses: list[float]) -> float:
    """Loss-threshold attack: lower loss means 'member', so members score -loss."""
    return auc([-value for value in member_losses], [-value for value in non_member_losses])


@torch.no_grad()
def extractable(model, ids: list[int], prefix: int = 64, suffix: int = 32, device: str = "cpu") -> bool | None:
    """True if greedy decoding from the first `prefix` tokens reproduces the next `suffix` exactly."""
    if len(ids) < prefix + suffix:
        return None
    sequence = torch.tensor([ids[:prefix]], device=device)
    for _ in range(suffix):
        logits = model(sequence)[:, -1, :]
        sequence = torch.cat([sequence, logits.argmax(dim=-1, keepdim=True)], dim=1)
    return sequence[0, prefix:].tolist() == list(ids[prefix : prefix + suffix])


def extractability(model, documents: list[list[int]], device: str = "cpu", prefix: int = 64, suffix: int = 32) -> dict:
    was_training = model.training
    model.eval()
    outcomes = [extractable(model, ids, prefix, suffix, device) for ids in documents]
    model.train(was_training)
    eligible = [outcome for outcome in outcomes if outcome is not None]
    return {"eligible": len(eligible), "extracted": sum(eligible), "rate": sum(eligible) / len(eligible) if eligible else None}


def exposure_bits(canary_loss: float, control_losses: list[float]) -> float:
    """log2((|C|+1) / (1 + #{controls with loss <= canary loss}))."""
    at_or_below = sum(1 for value in control_losses if value <= canary_loss)
    return math.log2((len(control_losses) + 1) / (1 + at_or_below))


def did_benefits(m0_bpb: dict[str, float], m1_bpb: dict[str, float], groups: dict[str, str]) -> dict:
    """b_g = mean over g of (BPB_M0 - BPB_M1) minus the same for group Z; tau_g = b_g / b_C."""
    raw: dict[str, list[float]] = {}
    for document, group in groups.items():
        if document in m0_bpb and document in m1_bpb:
            raw.setdefault(group, []).append(m0_bpb[document] - m1_bpb[document])
    means = {group: float(np.mean(values)) for group, values in raw.items()}
    if "Z" not in means:
        raise ValueError("group Z is required as the control")
    benefit = {group: value - means["Z"] for group, value in means.items() if group != "Z"}
    exact = benefit.get("C")
    tau = {group: (value / exact if exact else None) for group, value in benefit.items()}
    return {"raw_mean_change": means, "benefit": benefit, "tau": tau, "n": {g: len(v) for g, v in raw.items()}}
