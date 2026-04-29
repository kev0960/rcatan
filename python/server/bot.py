"""Inference helpers for the trained PPO actor.

Loads weights once and exposes:
  - `pick_argmax(state, mask)` — single best action.
  - `top_k(state, mask, k)` — top-k actions ranked by masked logits.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from train.ppo import load_actor


class Bot:
    def __init__(self, actor: torch.nn.Module, device: str | torch.device):
        self.actor = actor
        self.device = device

    @classmethod
    def from_checkpoint(cls, path: str | Path) -> "Bot":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        actor = load_actor(path, device=device)
        return cls(actor=actor, device=device)

    @torch.no_grad()
    def _masked_logits(
        self, state: list[float], mask: list[bool]
    ) -> torch.Tensor:
        s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        m = torch.tensor(mask, dtype=torch.bool, device=self.device).unsqueeze(0)
        logits = self.actor(s)
        logits[~m] = float("-inf")
        return logits.squeeze(0)

    def pick_argmax(self, state: list[float], mask: list[bool]) -> int:
        logits = self._masked_logits(state, mask)
        return int(torch.argmax(logits).item())

    def top_k(
        self, state: list[float], mask: list[bool], k: int = 3
    ) -> list[tuple[int, float]]:
        """Return up to `k` (action_index, logit) pairs in descending order.
        Skips masked-out actions; if fewer than k are legal, returns fewer."""
        logits = self._masked_logits(state, mask)
        finite = torch.isfinite(logits)
        n_legal = int(finite.sum().item())
        if n_legal == 0:
            return []
        k = min(k, n_legal)
        values, indices = torch.topk(logits, k=k)
        return [
            (int(idx.item()), float(val.item()))
            for val, idx in zip(values, indices)
        ]


class RandomBot:
    """Fallback used when no checkpoint is available — picks any legal action."""

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def pick_argmax(self, state: list[float], mask: list[bool]) -> int:
        legal = np.flatnonzero(np.asarray(mask, dtype=bool))
        return int(self.rng.choice(legal))

    def top_k(
        self, state: list[float], mask: list[bool], k: int = 3
    ) -> list[tuple[int, float]]:
        legal = np.flatnonzero(np.asarray(mask, dtype=bool))
        if legal.size == 0:
            return []
        chosen = self.rng.choice(legal, size=min(k, legal.size), replace=False)
        return [(int(i), 0.0) for i in chosen]
