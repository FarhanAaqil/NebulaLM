"""Model contracts. The Phase 1 probe and the later full model both satisfy this."""

from __future__ import annotations

from typing import Protocol

import torch


class CausalLossModel(Protocol):
    def sequence_nll(self, input_ids: torch.Tensor, loss_mask: torch.Tensor, segment_ids: torch.Tensor | None = None) -> tuple[torch.Tensor, int]:
        """Return summed NLL in nats and the valid-token count."""
