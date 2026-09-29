"""
Gen9 OU live bridge: poke-env Battle → obs / action mask / BattleOrder.

Action space (N_ACTIONS = 13) — identical to metamon's, so dataset labels are
used as-is:
    0-3    use move 1-4          (alphabetical by cleaned move name)
    4-8    switch to bench 1-5   (alphabetical by cleaned species name)
    9-12   use move 1-4 + terastallize

Ported from metamon.interface.UniversalAction.action_idx_to_BattleOrder.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from poke_env.environment import Battle
from poke_env.player import Player
from poke_env.player.battle_order import BattleOrder, DefaultBattleOrder

from protean.formats.gen9ou.obs import N_MOVE_SLOTS, N_SWITCH_SLOTS, state_to_obs
from protean.formats.gen9ou.universal import (
    move_name, pokemon_name, state_from_battle, switch_candidates,
)

SWITCH_START = N_MOVE_SLOTS                     # 4
TERA_START   = N_MOVE_SLOTS + N_SWITCH_SLOTS    # 9
N_ACTIONS    = TERA_START + N_MOVE_SLOTS        # 13


def battle_to_obs(battle: Battle, prev_my_move: str = "", prev_opp_move: str = ""):
    # Previous moves come from poke-env's previous_move inside the universal state
    # (as in metamon), so the player's own prev-move tracking is not used here.
    return state_to_obs(state_from_battle(battle))


def _move_options(battle: Battle) -> list:
    valid = {m.id for m in battle.available_moves}
    if valid in ({"struggle"}, {"fight"}):
        # Forced single move: every move slot maps to it (dataset labels it slot 0).
        return [battle.available_moves[0]] * N_MOVE_SLOTS
    return sorted(battle.active_pokemon.moves.values(), key=lambda m: move_name(m.id))


def _slot(battle: Battle, idx: int) -> tuple[Optional[object], bool]:
    """(Move or Pokemon for this slot if currently selectable, wants_tera)."""
    valid_moves = {m.id for m in battle.available_moves}
    wants_tera = idx >= TERA_START
    if wants_tera:
        idx -= TERA_START
    if idx < SWITCH_START:
        if battle.force_switch or battle.active_pokemon is None:
            return None, False
        options = _move_options(battle)
        if idx < len(options) and options[idx].id in valid_moves:
            return options[idx], wants_tera
        return None, False
    switch_idx = idx - SWITCH_START
    options = sorted(switch_candidates(battle), key=lambda p: pokemon_name(p.species))
    valid_switches = {p.name for p in battle.available_switches}
    if switch_idx < len(options) and options[switch_idx].name in valid_switches:
        return options[switch_idx], False
    return None, False


def battle_to_action_mask(battle: Battle) -> np.ndarray:
    mask = np.zeros(N_ACTIONS, dtype=bool)
    if {m.id for m in battle.available_moves} == {"recharge"}:
        mask[0] = True    # only option; action_idx_to_order sends it for slot 0
        return mask
    can_tera = battle.can_tera is not None
    for idx in range(N_ACTIONS):
        target, wants_tera = _slot(battle, idx)
        mask[idx] = target is not None and (can_tera or not wants_tera)
    if not mask.any():
        mask[0] = True    # nothing expressible — action_idx_to_order falls back to /choose default
    return mask


def action_idx_to_order(idx: int, battle: Battle) -> BattleOrder:
    if {m.id for m in battle.available_moves} == {"recharge"}:
        return Player.create_order(battle.available_moves[0])
    target, wants_tera = _slot(battle, idx)
    if target is None:
        return DefaultBattleOrder()
    if idx < SWITCH_START or idx >= TERA_START:
        return Player.create_order(target, terastallize=wants_tera and battle.can_tera is not None)
    return Player.create_order(target)


def describe_action(idx: int, battle: Battle) -> Optional[tuple[str, str]]:
    target, wants_tera = _slot(battle, idx)
    if target is None:
        return None
    if SWITCH_START <= idx < TERA_START:
        return ("switch", pokemon_name(target.species))
    return ("tera" if wants_tera else "move", move_name(target.id))
