"""
BattleFormat: everything the shared model / player / training code needs to know
about a specific Showdown format.

The policy architecture (turn encoder + causal trajectory transformer) and the
PPO loop are format-agnostic. A format supplies:
  - a vocabulary (tokenizer) and the observation encoder that produces
    {"numbers": float32[numbers_dim], "text": str} from a live poke-env Battle
  - a fixed-size discrete action space: mask, slot → BattleOrder, slot → label
  - model sizing (vocab, numbers_dim, n_actions, max_seq_len)
  - teams, for formats where the player brings one (none for random battles)
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from functools import cached_property
from typing import Optional

import numpy as np

from protean.tokenizer import Tokenizer


class BattleFormat(ABC):
    name:        str    # Showdown format id, e.g. "gen1ou"
    numbers_dim: int    # length of obs["numbers"]
    n_actions:   int    # size of the discrete action space
    max_seq_len: int    # turn-encoder positions (text tokens + 1 CLS)
    needs_team:  bool = True   # False for random-battle formats (server generates teams)

    # ── Vocabulary ───────────────────────────────────────────────────────

    @abstractmethod
    def _load_tokenizer(self) -> Tokenizer: ...

    @cached_property
    def tokenizer(self) -> Tokenizer:
        return self._load_tokenizer()

    # ── Live-battle bridge (poke-env Battle) ─────────────────────────────

    @abstractmethod
    def battle_to_obs(
        self, battle, prev_my_move: str = "", prev_opp_move: str = "",
    ) -> dict[str, np.ndarray]:
        """Encode the current decision point as {"numbers", "text"}."""

    @abstractmethod
    def action_mask(self, battle) -> np.ndarray:
        """bool[n_actions] — True where the slot is a legal choice."""

    @abstractmethod
    def action_to_order(self, idx: int, battle):
        """Translate an action slot into a poke-env BattleOrder."""

    @abstractmethod
    def describe_action(self, idx: int, battle) -> Optional[tuple[str, str]]:
        """
        (kind, value) for an action slot — ("move", move_id) or ("switch", species) —
        or None if the slot is empty. Used for prev-move context and decision logs.
        """

    # ── Teams ────────────────────────────────────────────────────────────

    def training_teams(self) -> list[str]:
        """Teams to rotate through in self-play. Empty for random-battle formats."""
        return []

    def get_team(self, name: str) -> Optional[str]:
        """Look up a named team. Random-battle formats return None (no team needed)."""
        if self.needs_team:
            raise NotImplementedError(f"{self.name} does not define named teams")
        return None

    # ── Model sizing ─────────────────────────────────────────────────────

    def model_kwargs(self) -> dict:
        """Constructor kwargs for ProteanPolicy that depend on this format."""
        return {
            "vocab_size":  self.tokenizer.vocab_size,
            "numbers_dim": self.numbers_dim,
            "n_actions":   self.n_actions,
            "max_seq_len": self.max_seq_len,
        }
