"""Gen1OU BattleFormat — wires the gen1ou vocab, obs encoder and teams together."""
from __future__ import annotations

from typing import Optional

import numpy as np

from protean.formats.base import BattleFormat
from protean.formats.gen1ou import battle as _battle
from protean.formats.gen1ou import teams as _teams
from protean.formats.gen1ou.obs_space import NUMBERS_DIM, Gen1ActionSpace
from protean.formats.gen1ou.vocab import get_tokenizer
from protean.tokenizer import Tokenizer


class Gen1OUFormat(BattleFormat):
    name        = "gen1ou"
    numbers_dim = NUMBERS_DIM            # 48
    n_actions   = Gen1ActionSpace.N_ACTIONS   # 4 moves + 5 switches
    max_seq_len = 128                    # 77 text tokens + 1 CLS, with headroom
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

    def training_teams(self) -> list[str]:
        return list(_teams.ALL_TEAMS)

    def get_team(self, name: str) -> str:
        return _teams.get_team(name)
