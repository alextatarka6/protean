"""
Static gen1 Pokédex lookup using poke-env's bundled data.

In Gen 1, Attack/Defense/Speed are standard, but there is only one Special stat
(SpA = SpD). poke-env reports separate spa/spd values which are always equal for
gen1 Pokémon — we use spa as the canonical 'spc' value.

Usage:
    from protean.pokedex import get_base_stats, get_types
    stats = get_base_stats("starmie")   # {"hp": 60, "atk": 75, "def": 85, "spc": 100, "spe": 115}
    types = get_types("starmie")        # ["water", "psychic"]
"""
from __future__ import annotations

import re
from functools import lru_cache
from typing import Optional

_PAD_STATS = {"hp": 0, "atk": 0, "def": 0, "spc": 0, "spe": 0}
_PAD_TYPES = ["notype", "notype"]


def _clean(name: str) -> str:
    """Normalize a species/move name: lowercase, alphanumeric only."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


@lru_cache(maxsize=1)
def _gen1_data():
    from poke_env.data import GenData
    return GenData.from_gen(1)


def get_base_stats(species: str) -> dict[str, int]:
    """
    Return gen1 base stats for a species as {"hp", "atk", "def", "spc", "spe"}.

    "spc" is the Gen 1 Special stat (spa == spd in all gen1 Pokémon).
    Returns zeroed stats if species is not found.
    """
    gd = _gen1_data()
    key = _clean(species)
    entry = gd.pokedex.get(key)
    if entry is None:
        return dict(_PAD_STATS)
    bs = entry["baseStats"]
    return {
        "hp":  bs.get("hp",  0),
        "atk": bs.get("atk", 0),
        "def": bs.get("def", 0),
        "spc": bs.get("spa", 0),  # spa == spd in gen1
        "spe": bs.get("spe", 0),
    }


def get_types(species: str) -> list[str]:
    """
    Return the type(s) of a species as a list of lowercase strings.

    Always returns exactly two entries; single-type Pokémon get ["type", "notype"].
    Returns ["notype", "notype"] if species is not found.
    """
    gd = _gen1_data()
    key = _clean(species)
    entry = gd.pokedex.get(key)
    if entry is None:
        return list(_PAD_TYPES)
    raw = [t.lower() for t in entry.get("types", [])]
    if len(raw) == 0:
        return list(_PAD_TYPES)
    if len(raw) == 1:
        return [raw[0], "notype"]
    return raw[:2]


def get_move_data(move: str) -> dict:
    """
    Return gen1 move data as:
        {"type": str, "category": str, "base_power": int, "accuracy": float, "priority": int}

    accuracy is in [0.0, 1.0]; moves that never miss have accuracy=1.0.
    Returns a blank-move dict if the move is not found.
    """
    gd = _gen1_data()
    key = _clean(move)
    entry = gd.moves.get(key)
    if entry is None:
        return {
            "type": "normal",
            "category": "status",
            "base_power": 0,
            "accuracy": 1.0,
            "priority": 0,
        }
    raw_acc = entry.get("accuracy", 100)
    accuracy = 1.0 if raw_acc is True else float(raw_acc) / 100.0
    return {
        "type":       entry.get("type", "normal").lower(),
        "category":   entry.get("category", "status").lower(),
        "base_power": int(entry.get("basePower", 0)),
        "accuracy":   accuracy,
        "priority":   int(entry.get("priority", 0)),
    }
