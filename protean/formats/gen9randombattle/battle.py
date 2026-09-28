"""
Gen9 random-battle live bridge: poke-env Battle → obs / action mask / BattleOrder.

Action space (N_ACTIONS = 13):
    0-3    use move 1-4            (alphabetical order of the active's moveset)
    4-7    use move 1-4 + terastallize
    8-12   switch to bench slot 1-5 (alive non-active team members, team order —
           the same order as the <switch> entries in the obs)
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from poke_env.environment import Battle, Field, Move, Pokemon, SideCondition
from poke_env.player import Player
from poke_env.player.battle_order import BattleOrder, DefaultBattleOrder

from protean.formats.gen9randombattle.dex import canonical_species, load_dex
from protean.formats.gen9randombattle.obs import N_MOVE_SLOTS, N_SWITCH_SLOTS, build_obs
from protean.tokenizer import _clean

N_ACTIONS  = 2 * N_MOVE_SLOTS + N_SWITCH_SLOTS
TERA_START = N_MOVE_SLOTS
SWITCH_START = 2 * N_MOVE_SLOTS

# Pseudo-moves the server can force (not part of any moveset)
_PSEUDO_MOVES = {"struggle", "recharge"}
_STACKABLE    = {SideCondition.SPIKES, SideCondition.TOXIC_SPIKES}


# ---------------------------------------------------------------------------
# Pokémon → obs dict
# ---------------------------------------------------------------------------

def _enum_id(e) -> str:
    return e.name.lower().replace("_", "")


def _types(p: Pokemon) -> list[str]:
    valid = set(load_dex()["types"])
    return [t for t in (_enum_id(t) for t in p.types if t is not None) if t in valid]


def _status(p: Pokemon) -> str:
    return p.status.name.lower() if p.status is not None else ""


def _item(p: Pokemon) -> str:
    if p.item == "unknown_item":
        return "unknownitem"
    return _clean(p.item) if p.item else "noitem"


def _moveset(p: Pokemon) -> list[str]:
    return sorted(m for m in p.moves if m not in _PSEUDO_MOVES)


def _own_tera_types(battle: Battle) -> dict[str, str]:
    """ident → tera type for our team, from the server's last |request| (poke-env doesn't store it)."""
    req = battle.last_request or {}
    return {
        mon["ident"]: _clean(mon.get("teraType") or "")
        for mon in req.get("side", {}).get("pokemon", [])
    }


def _mon_dict(p: Pokemon, tera_type: str = "") -> dict:
    if p.is_terastallized and p.tera_type is not None:
        tera_type = _enum_id(p.tera_type)
    return {
        "species":       canonical_species(p.species),
        "hp":            float(p.current_hp_fraction),
        "level":         p.level,
        "status":        _status(p),
        "boosts":        dict(p.boosts),
        "types":         _types(p),
        "tera_type":     tera_type,
        "terastallized": bool(p.is_terastallized),
        "item":          _item(p),
        "ability":       _clean(p.ability) if p.ability else "unknownability",
        "moves":         _moveset(p),
        "move_pp":       {mid: m.current_pp / max(m.max_pp, 1) for mid, m in p.moves.items()},
    }


def _side_dict(conditions: dict) -> dict[str, int]:
    return {_enum_id(c): (v if c in _STACKABLE else 1) for c, v in conditions.items()}


def _bench(battle: Battle) -> list[Pokemon]:
    """Alive non-active team members in team order — switch slots 1-5."""
    return [p for p in battle.team.values() if not p.active and not p.fainted][:N_SWITCH_SLOTS]


def battle_to_obs(
    battle: Battle,
    prev_my_move:  str = "",
    prev_opp_move: str = "",
) -> dict[str, np.ndarray]:
    tera_types = _own_tera_types(battle)
    idents     = {id(p): ident for ident, p in battle.team.items()}

    def own(p: Pokemon) -> dict:
        return _mon_dict(p, tera_types.get(idents.get(id(p), ""), ""))

    my_active  = battle.active_pokemon
    opp_active = battle.opponent_active_pokemon
    terrain = next((_enum_id(f) for f in battle.fields if f.name.endswith("_TERRAIN")), "")

    return build_obs(
        my_active=own(my_active),
        my_bench=[own(p) for p in _bench(battle)],
        my_alive=sum(not p.fainted for p in battle.team.values()),
        my_tera_available=not any(p.is_terastallized for p in battle.team.values()),
        opp_active=_mon_dict(opp_active),
        opp_bench=[_mon_dict(p) for p in battle.opponent_team.values()
                   if not p.active and not p.fainted],
        opp_alive=6 - sum(p.fainted for p in battle.opponent_team.values()),
        opp_tera_available=not any(p.is_terastallized for p in battle.opponent_team.values()),
        weather=next((_enum_id(w) for w in battle.weather), ""),
        terrain=terrain,
        trick_room=Field.TRICK_ROOM in battle.fields,
        gravity=Field.GRAVITY in battle.fields,
        my_side=_side_dict(battle.side_conditions),
        opp_side=_side_dict(battle.opponent_side_conditions),
        prev_my_move=prev_my_move,
        prev_opp_move=prev_opp_move,
        forced_switch=bool(battle.force_switch),
    )


# ---------------------------------------------------------------------------
# Action bridge
# ---------------------------------------------------------------------------

def _slot_moves(battle: Battle) -> tuple[list[Optional[Move]], bool]:
    """
    Selectable Move per move slot (None if not selectable this turn), plus whether
    the slots are the real moveset (False when the server forces Struggle/recharge
    or another move outside the known set — then slot 0 carries that move).
    """
    if battle.force_switch or battle.active_pokemon is None:
        return [None] * N_MOVE_SLOTS, True
    avail = {m.id: m for m in battle.available_moves}
    ids   = _moveset(battle.active_pokemon)[:N_MOVE_SLOTS]
    slots = [avail.get(mid) for mid in ids] + [None] * (N_MOVE_SLOTS - len(ids))
    if avail and not any(slots):
        return [battle.available_moves[0]] + [None] * (N_MOVE_SLOTS - 1), False
    return slots, True


def _switch_slots(battle: Battle) -> list[Optional[Pokemon]]:
    available = {p.species for p in battle.available_switches}
    bench = _bench(battle)
    slots = [p if p.species in available else None for p in bench]
    return slots + [None] * (N_SWITCH_SLOTS - len(slots))


def battle_to_action_mask(battle: Battle) -> np.ndarray:
    mask = np.zeros(N_ACTIONS, dtype=bool)
    moves, real = _slot_moves(battle)
    can_tera = battle.can_tera is not None and real
    for i, m in enumerate(moves):
        if m is not None:
            mask[i] = True
            mask[TERA_START + i] = can_tera
    for i, p in enumerate(_switch_slots(battle)):
        mask[SWITCH_START + i] = p is not None
    if not mask.any():
        mask[0] = True   # nothing legal we can express — action_to_order falls back to /choose default
    return mask


def action_idx_to_order(idx: int, battle: Battle) -> BattleOrder:
    if idx < SWITCH_START:
        moves, real = _slot_moves(battle)
        m = moves[idx % N_MOVE_SLOTS]
        if m is not None:
            tera = idx >= TERA_START and real and battle.can_tera is not None
            return Player.create_order(m, terastallize=tera)
    else:
        p = _switch_slots(battle)[idx - SWITCH_START]
        if p is not None:
            return Player.create_order(p)
    return DefaultBattleOrder()


def describe_action(idx: int, battle: Battle) -> Optional[tuple[str, str]]:
    if idx < SWITCH_START:
        m = _slot_moves(battle)[0][idx % N_MOVE_SLOTS]
        if m is None:
            return None
        return ("tera" if idx >= TERA_START else "move", m.id)
    p = _switch_slots(battle)[idx - SWITCH_START]
    return ("switch", p.species) if p is not None else None
