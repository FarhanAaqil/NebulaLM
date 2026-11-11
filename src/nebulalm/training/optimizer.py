"""AdamW with the architecture's parameter groups.

Decay: matrix-valued weights, including the tied token embedding once.
No decay: biases and LayerNorm parameters.
"""

from __future__ import annotations

import torch
from torch import nn


def parameter_groups(model: nn.Module) -> dict[str, list[tuple[str, nn.Parameter]]]:
    decay: list[tuple[str, nn.Parameter]] = []
    no_decay: list[tuple[str, nn.Parameter]] = []
    seen: set[int] = set()
    for name, param in model.named_parameters():
        if not param.requires_grad or id(param) in seen:
            continue
        seen.add(id(param))
        if param.ndim < 2:
            no_decay.append((name, param))
        else:
            decay.append((name, param))
    return {"decay": decay, "no_decay": no_decay}


def build_optimizer(model: nn.Module, settings: dict) -> torch.optim.AdamW:
    groups = parameter_groups(model)
    return torch.optim.AdamW(
        [
            {"params": [param for _, param in groups["decay"]], "weight_decay": float(settings["weight_decay"])},
            {"params": [param for _, param in groups["no_decay"]], "weight_decay": 0.0},
        ],
        lr=float(settings["peak_lr"]),
        betas=tuple(float(beta) for beta in settings["betas"]),
        eps=float(settings.get("eps", 1e-8)),
    )


def group_report(model: nn.Module) -> dict[str, list[str]]:
    """Names per group, saved in run metadata."""
    return {key: [name for name, _ in rows] for key, rows in parameter_groups(model).items()}
