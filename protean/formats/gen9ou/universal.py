"""
Metamon "UniversalState" for gen9ou, as plain dicts.

The metamon parsed-replay dataset (jakegrigsby/metamon-parsed-replays, gen9ou)
stores each turn as a UniversalState dict. To make live play see exactly what
BC training saw, live poke-env battles are converted to the *same* dict schema
here, ported from metamon.interface (UniversalState.from_Battle and helpers,
github.com/UT-Austin-RPL/metamon), and a single encoder (obs.py) reads both.

State dict keys:
    format, player_active_pokemon, opponent_active_pokemon, available_switches,
    player_prev_move, opponent_prev_move, opponents_remaining, player_conditions,
    opponent_conditions, weather, battle_field, forced_switch, battle_won,
    battle_lost, can_tera, opponent_teampreview

Pokemon dict keys:
    name, base_species, hp_pct, types ("t1 t2", sorted), tera_type, item, ability,
    lvl, status, effect, moves [move dicts], {atk,spa,def,spd,spe,accuracy,evasion}_boost,
    base_{atk,spa,def,spd,spe,hp}

Move dict keys:
    name, move_type, category, base_power, accuracy, priority, current_pp, max_pp
"""
from __future__ import annotations

from typing import Optional

from poke_env.environment import Battle, Move, Pokemon, PokemonType, Status


# ---------------------------------------------------------------------------
# Name cleaning (metamon.backend.replay_parser.str_parsing)
# ---------------------------------------------------------------------------

def clean_no_numbers(name) -> str:
    return "".join(c for c in str(name) if c.isalpha()).lower()


def clean_name(name: str) -> str:
    return "".join(c for c in name if c.isalnum()).lower()


def pokemon_name(name: str) -> str:
    return clean_name(name).strip()


def move_name(name: str) -> str:
    move_id = clean_no_numbers(name)
    if move_id.startswith("hiddenpower"):
        move_id = "hiddenpower"
    elif move_id == "vicegrip":
        move_id = "visegrip"
    elif move_id.startswith("return"):
        move_id = "return"
    elif move_id.startswith("frustration"):
        move_id = "frustration"
    return move_id.strip()


# ---------------------------------------------------------------------------
# Field normalisers (UniversalPokemon / UniversalState static methods)
# ---------------------------------------------------------------------------

def universal_item(item: Optional[str]) -> str:
    if item is None or item == "unknown_item":
        return "unknownitem"
    if item.strip() in {"", "No Item", "noitem"}:
        return "noitem"
    return clean_no_numbers(item)


def universal_ability(ability: Optional[str]) -> str:
    if ability is None or ability == "unknown_ability":
        return "unknownability"
    if ability.strip() in {"", "No Ability", "noability"}:
        return "noability"
    return clean_no_numbers(ability)


def universal_status(status: Optional[Status]) -> str:
    return "nostatus" if status is None else clean_no_numbers(status.name)


def universal_types(types: list, force_two: bool = True) -> str:
    types = list(types)
    if force_two:
        types += [None] * (2 - len(types))
    out = []
    for t in types:
        if t is None:
            out.append("notype")
        elif isinstance(t, PokemonType):
            out.append(clean_name(t.name))
        else:
            out.append(clean_name(t))
    return " ".join(sorted(out))


def _most_recent(rep: dict, empty: str) -> str:
    if not rep:
        return empty
    return clean_no_numbers(max(rep.keys(), key=rep.get).name)


def universal_conditions(side_conditions: dict) -> str:
    return _most_recent(side_conditions, "noconditions")


def universal_field(fields: dict) -> str:
    return _most_recent(fields, "nofield")


def universal_weather(weather: dict) -> str:
    if not weather:
        return "noweather"
    return clean_no_numbers(next(iter(weather)).name)


# ---------------------------------------------------------------------------
# poke-env → universal dicts
# ---------------------------------------------------------------------------

BLANK_MOVE = {
    "name": "nomove", "move_type": "nomove", "category": "nomove", "base_power": 0,
    "accuracy": 1.0, "priority": 0, "current_pp": 0, "max_pp": 0,
}


def move_to_universal(move: Optional[Move]) -> dict:
    if move is None:
        return dict(BLANK_MOVE)
    return {
        "name":       move_name(move.id),
        "move_type":  clean_name(move.type.name),
        "category":   clean_name(move.category.name),
        "base_power": move.base_power,
        "accuracy":   move.accuracy,
        "priority":   move.priority,
        "current_pp": move.current_pp,
        "max_pp":     move.max_pp,
    }


def pokemon_to_universal(p: Pokemon) -> dict:
    effect = min(p.effects, key=p.effects.get) if p.effects else None
    return {
        "name":         pokemon_name(p.species),
        "base_species": pokemon_name(p.base_species),
        "hp_pct":       float(p.current_hp_fraction),
        "types":        universal_types(p.types),
        "tera_type":    universal_types([p.tera_type], force_two=False),
        "item":         universal_item(p.item),
        "ability":      universal_ability(p.ability),
        "lvl":          p.level,
        "status":       universal_status(p.status),
        "effect":       clean_no_numbers(effect.name) if effect else "noeffect",
        "moves":        [move_to_universal(m) for m in p.moves.values()][:4],
        **{f"{stat}_boost": boost for stat, boost in p.boosts.items()},
        **{f"base_{stat}": val for stat, val in p.base_stats.items()},
    }


def switch_candidates(battle: Battle) -> list[Pokemon]:
    """Bench mons the switch slots refer to (fainted ones while Revival Blessing is pending)."""
    if battle.reviving:
        return [p for p in battle.team.values() if p.fainted and not p.active]
    return [p for p in battle.team.values() if not p.fainted and not p.active]


def state_from_battle(battle: Battle) -> dict:
    """UniversalState.from_Battle, as a dict with the dataset's schema."""
    force_switch = battle.force_switch
    if isinstance(force_switch, list):
        force_switch = force_switch[0]
    return {
        "format":                  "gen9ou",
        "player_active_pokemon":   pokemon_to_universal(battle.active_pokemon),
        "opponent_active_pokemon": pokemon_to_universal(battle.opponent_active_pokemon),
        "available_switches":      [pokemon_to_universal(p) for p in switch_candidates(battle)],
        "player_prev_move":        move_to_universal(battle.active_pokemon.previous_move),
        "opponent_prev_move":      move_to_universal(battle.opponent_active_pokemon.previous_move),
        "opponents_remaining":     6 - sum(p.status == Status.FNT for p in battle.opponent_team.values()),
        "player_conditions":       universal_conditions(battle.side_conditions),
        "opponent_conditions":     universal_conditions(battle.opponent_side_conditions),
        "weather":                 universal_weather(battle.weather),
        "battle_field":            universal_field(battle.fields),
        "forced_switch":           bool(force_switch),
        "battle_won":              bool(battle.won),
        "battle_lost":             bool(battle.lost),
        "can_tera":                battle.can_tera is not None,
        "opponent_teampreview":    [pokemon_name(p.base_species)
                                    for p in battle.teampreview_opponent_team if p is not None],
    }
