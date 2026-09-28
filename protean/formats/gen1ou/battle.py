"""
Gen1OU live-battle bridge: poke-env Battle → obs / action mask / BattleOrder.

Produces the same obs as Gen1OUObservationSpace.row_to_obs does for dataset
rows, so a BC-trained model plays with the inputs it was trained on.
"""
from __future__ import annotations

import json

import numpy as np
from poke_env.environment import Battle, Pokemon, Status
from poke_env.player import Player

# ---------------------------------------------------------------------------
# poke-env 0.8.x compatibility patch
# to_id_str() crashes on None ability (gen1 has no abilities). Patch it to
# return "" for None before any poke-env Pokemon objects are created.
# ---------------------------------------------------------------------------
import poke_env.data.normalize as _poke_norm
import poke_env.environment.pokemon as _poke_pk

_orig_to_id_str = _poke_norm.to_id_str

def _safe_to_id_str(name):  # type: ignore[override]
    if name is None:
        return ""
    return _orig_to_id_str(name)

_poke_norm.to_id_str = _safe_to_id_str
_poke_pk.to_id_str   = _safe_to_id_str   # patch the already-imported ref

from protean.formats.gen1ou.obs_space import (
    _sorted_moves, _build_obs, _norm_conditions,
    N_MOVE_SLOTS, N_SWITCH_SLOTS,
)
from protean.tokenizer import _clean

# ---------------------------------------------------------------------------
# Observation bridge
# ---------------------------------------------------------------------------

_STATUS_MAP = {
    "brn": "brn",
    "par": "par",
    "slp": "slp",
    "frz": "frz",
    "psn": "psn",
    "tox": "tox",
    None:  "",
}


def _poke_status(pokemon: Pokemon) -> str:
    if pokemon.status is None:
        return ""
    return _STATUS_MAP.get(pokemon.status.name.lower(), pokemon.status.name.lower())


def _poke_boosts(pokemon: Pokemon) -> dict[str, int]:
    # poke-env stores boosts as a dict keyed by stat name strings
    return dict(pokemon.boosts) if hasattr(pokemon, "boosts") else {}


def _pk_to_bench_dict(pokemon: Pokemon) -> dict:
    return {
        "species":        _clean(pokemon.species),
        "hp":             float(pokemon.current_hp_fraction),
        "status":         _poke_status(pokemon),
        "fainted":        pokemon.fainted,
        "revealed_moves": [_clean(m) for m in pokemon.moves],
    }


def known_moves(battle: Battle) -> list[str]:
    """
    Return all currently known move IDs for the active pokemon.

    poke-env populates battle.active_pokemon.moves only as moves are *used*,
    so on turn 1 it may be empty. battle.available_moves always contains the
    moves available *this turn* from the server's |request| message.
    Union the two so we always have the full move set once all 4 are revealed.
    """
    active = battle.active_pokemon
    known: set[str] = set(active.moves.keys()) if active else set()
    if not battle.force_switch:
        known |= {_clean(m.id) for m in battle.available_moves}
    return _sorted_moves(list(known))


def battle_to_obs(
    battle: Battle,
    prev_my_move:  str = "",
    prev_opp_move: str = "",
) -> dict[str, np.ndarray]:
    """
    Convert a live poke-env Battle to our observation format.

    Own team: all moves are known (it's our team) — no inference needed.
    Opponent: only revealed information is used (no inference during play;
    the obs format handles unknown moves gracefully via <blank> tokens).
    """
    my_active  = battle.active_pokemon
    opp_active = battle.opponent_active_pokemon

    # --- my active ---
    my_species  = _clean(my_active.species) if my_active else "missingno"
    my_hp       = float(my_active.current_hp_fraction) if my_active else 1.0
    my_status   = _poke_status(my_active) if my_active else ""
    my_boosts   = _poke_boosts(my_active) if my_active else {}
    # Union observed moves with available_moves so turn-1 moves are always visible
    my_moves    = known_moves(battle)

    # --- my bench ---
    my_bench = [
        _pk_to_bench_dict(p)
        for p in battle.team.values()
        if not p.active and not p.fainted
    ]

    # --- opponent active ---
    opp_species = _clean(opp_active.species) if opp_active else "missingno"
    opp_hp      = float(opp_active.current_hp_fraction) if opp_active else 1.0
    opp_status  = _poke_status(opp_active) if opp_active else ""
    opp_boosts  = _poke_boosts(opp_active) if opp_active else {}

    # --- opponent bench (revealed only) ---
    opp_bench = [
        _pk_to_bench_dict(p)
        for p in battle.opponent_team.values()
        if not p.active and not p.fainted
    ]

    opp_remaining = sum(
        1 for p in battle.opponent_team.values() if not p.fainted
    )

    # --- field ---
    # battle.weather is a Weather enum or None
    weather   = battle.weather.name.lower() if battle.weather else ""
    # side_conditions is a dict {SideCondition: int}
    my_conds  = json.dumps([c.name.lower() for c in battle.side_conditions])
    opp_conds = json.dumps([c.name.lower() for c in battle.opponent_side_conditions])

    # --- forced switch ---
    forced = battle.force_switch

    return _build_obs(
        my_active_species=my_species,
        my_active_hp=my_hp,
        my_active_status=my_status,
        my_active_boosts=my_boosts,
        my_active_moves=my_moves,
        my_bench=my_bench,
        opp_active_species=opp_species,
        opp_active_hp=opp_hp,
        opp_active_status=opp_status,
        opp_active_boosts=opp_boosts,
        opp_bench=opp_bench,
        opp_remaining=opp_remaining,
        weather=weather,
        my_conditions=_norm_conditions(my_conds),
        opp_conditions=_norm_conditions(opp_conds),
        prev_my_move=prev_my_move,
        prev_opp_move=prev_opp_move,
        forced_switch=forced,
    )


# ---------------------------------------------------------------------------
# Action bridge
# ---------------------------------------------------------------------------

def battle_to_action_mask(battle: Battle) -> np.ndarray:
    """
    Build the 9-slot boolean action mask from the live battle state.
    Slots 0-3: available moves (alphabetically ordered against full moveset).
    Slots 4-8: available switches.
    """
    mask = np.zeros(N_MOVE_SLOTS + N_SWITCH_SLOTS, dtype=bool)

    switches = battle.available_switches[:N_SWITCH_SLOTS]
    # Never voluntarily switch into a sleeping Pokémon in Gen 1 — sleep persists
    # through switches and the switched-in mon wastes the turn. Fallback: allow
    # sleeping switches if every available switch is asleep (no choice).
    all_asleep = bool(switches) and all(p.status == Status.SLP for p in switches)

    if battle.force_switch:
        for i, p in enumerate(switches):
            mask[N_MOVE_SLOTS + i] = (p.status != Status.SLP) or all_asleep
        return mask

    # Move slots — map available_moves back to their alphabetical slot index.
    # Use known_moves() which unions active_pokemon.moves with available_moves
    # so slot assignment is correct even on turn 1 when no moves have been used.
    all_moves_sorted = known_moves(battle)
    available_ids    = {_clean(m.id) for m in battle.available_moves}
    for i, m in enumerate(all_moves_sorted[:N_MOVE_SLOTS]):
        if _clean(m) in available_ids:
            mask[i] = True

    # Switch slots — exclude sleeping bench Pokémon
    for i, p in enumerate(switches):
        mask[N_MOVE_SLOTS + i] = (p.status != Status.SLP) or all_asleep

    # Emergency fallback — should never be needed
    if not mask.any():
        mask[0] = True

    return mask


def action_idx_to_order(idx: int, battle: Battle):
    """
    Convert a 9-slot action index to a poke-env BattleOrder.
    Move slots 0-3 map to the active pokemon's alphabetically-sorted moves.
    Switch slots 4-8 map to available_switches in the order poke-env provides.
    """

    if idx < N_MOVE_SLOTS:
        sorted_moves = known_moves(battle)
        if idx < len(sorted_moves):
            move_id = sorted_moves[idx]
            for m in battle.available_moves:
                if _clean(m.id) == _clean(move_id):
                    return Player.create_order(m)
        # Fallback — shouldn't happen if mask is correct
        if battle.available_moves:
            return Player.create_order(battle.available_moves[0])
        if battle.available_switches:
            return Player.create_order(battle.available_switches[0])

    else:
        switch_idx = idx - N_MOVE_SLOTS
        switches = battle.available_switches
        if switch_idx < len(switches):
            return Player.create_order(switches[switch_idx])
        if switches:
            return Player.create_order(switches[0])

    return Player.choose_default_move(battle)


def describe_action(idx: int, battle: Battle) -> tuple[str, str] | None:
    """(kind, value) for a 9-slot action index, or None if the slot is empty."""
    if idx < N_MOVE_SLOTS:
        sorted_moves = known_moves(battle)
        if idx < len(sorted_moves):
            return ("move", sorted_moves[idx])
    else:
        switch_idx = idx - N_MOVE_SLOTS
        switches = battle.available_switches
        if switch_idx < len(switches):
            return ("switch", _clean(switches[switch_idx].species))
    return None
