"""Gen 9 Random Battle BattleFormat."""
from __future__ import annotations

from typing import Optional

import numpy as np

from protean.formats.base import BattleFormat
from protean.formats.gen9randombattle import battle as _battle
from protean.formats.gen9randombattle.obs import NUMBERS_DIM, TEXT_LEN
from protean.formats.gen9randombattle.vocab import get_tokenizer
from protean.tokenizer import Tokenizer


class Gen9RandomBattleFormat(BattleFormat):
    name        = "gen9randombattle"
    numbers_dim = NUMBERS_DIM          # 88
    n_actions   = _battle.N_ACTIONS    # 4 moves + 4 tera moves + 5 switches
    max_seq_len = 128                  # 112 text tokens + 1 CLS
    needs_team  = False                # server generates teams

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
