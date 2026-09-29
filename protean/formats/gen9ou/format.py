"""Gen 9 OU BattleFormat."""
from __future__ import annotations

import json
from functools import cached_property
from typing import Optional

import numpy as np

from protean.formats.base import BattleFormat
from protean.formats.gen9ou import battle as _battle
from protean.formats.gen9ou.obs import NUMBERS_DIM
from protean.formats.gen9ou.vocab import DATA_DIR, get_tokenizer
from protean.tokenizer import Tokenizer

TEAMS_PATH = DATA_DIR / "gen9ou_teams.json"   # built by scripts/build_gen9ou_teams.py


class Gen9OUFormat(BattleFormat):
    name        = "gen9ou"
    numbers_dim = NUMBERS_DIM          # 55
    n_actions   = _battle.N_ACTIONS    # 4 moves + 5 switches + 4 tera moves (metamon order)
    max_seq_len = 128                  # 112 text tokens + 1 CLS
    needs_team  = True

    def _load_tokenizer(self) -> Tokenizer:
        return get_tokenizer()

    def battle_to_obs(self, battle, prev_my_move: str = "", prev_opp_move: str = ""):
        return _battle.battle_to_obs(battle, prev_my_move, prev_opp_move)

    def action_mask(self, battle) -> np.ndarray:
        return _battle.battle_to_action_mask(battle)

    def action_to_order(self, idx: int, battle):
        return _battle.action_idx_to_order(idx, battle)

    def describe_action(self, idx: int, battle) -> Optional[tuple[str, str]]:
        return _battle.describe_action(idx, battle)

    # ── Teams ────────────────────────────────────────────────────────────

    @cached_property
    def _teams(self) -> dict[str, list[str]]:
        return json.loads(TEAMS_PATH.read_text())

    def training_teams(self) -> list[str]:
        """~2k legal teams predicted from May '26 ladder replays."""
        return self._teams["ladder"]

    def get_team(self, name: str) -> str:
        """'competitive:<i>' (expert sample teams) or 'ladder:<i>'."""
        pool, _, idx = name.partition(":")
        if pool not in self._teams or not idx.isdigit():
            raise ValueError(f"Team name must be 'competitive:<i>' or 'ladder:<i>', got {name!r}")
        return self._teams[pool][int(idx) % len(self._teams[pool])]
