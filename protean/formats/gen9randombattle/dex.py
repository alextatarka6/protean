"""
Gen9 dex + random-battle set lookups, backed by protean/data/gen9randombattle_dex.json
(exported from the local Showdown server by scripts/build_gen9rb_data.py).

Random battles draw every Pokémon from a small, public list of sets per species
(level, role, movepool, abilities, tera types). Given what the opponent has
revealed so far we can narrow those sets and fill in the rest — much sharper
than usage-stat inference in OU.
"""
from __future__ import annotations

import json
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Optional

DEX_PATH = Path(__file__).resolve().parents[2] / "data" / "gen9randombattle_dex.json"

_PAD_STATS = {"hp": 0, "atk": 0, "def": 0, "spa": 0, "spd": 0, "spe": 0}
_PAD_MOVE  = {"type": "notype", "category": "status", "base_power": 0,
              "accuracy": 1.0, "priority": 0, "pp": 1}


@lru_cache(maxsize=1)
def load_dex() -> dict:
    return json.loads(DEX_PATH.read_text())


def canonical_species(species: str) -> str:
    """Map cosmetic formes (e.g. florgesyellow) to the species id used in the dex / vocab."""
    return load_dex()["aliases"].get(species, species)


def base_stats(species: str) -> dict[str, int]:
    entry = load_dex()["species"].get(species)
    return entry["base_stats"] if entry else _PAD_STATS


def move_data(move: str) -> dict:
    return load_dex()["moves"].get(move, _PAD_MOVE)


def randbats_entry(species: str) -> Optional[dict]:
    """Randbats {level, sets} for a species, falling back to its base species for
    formes without their own entry (mimikyubusted, pikachusinnoh, ...)."""
    dex = load_dex()
    entry = dex["randbats"].get(species)
    if entry is None and species in dex["base_species"]:
        entry = dex["randbats"].get(dex["base_species"][species])
    return entry


def randbats_level(species: str) -> Optional[int]:
    entry = randbats_entry(species)
    return entry["level"] if entry else None


def consistent_sets(
    species: str,
    revealed_moves: list[str],
    ability: Optional[str] = None,
    tera_type: Optional[str] = None,
) -> list[dict]:
    """Randbats sets for `species` that could have produced what's been revealed."""
    entry = randbats_entry(species)
    if not entry:
        return []
    revealed = set(revealed_moves)
    out = []
    for s in entry["sets"]:
        if not revealed <= set(s["moves"]):
            continue
        if ability and s["abilities"] and ability not in s["abilities"]:
            continue
        if tera_type and s["tera_types"] and tera_type not in s["tera_types"]:
            continue
        out.append(s)
    # Revealed info can fall outside every listed set (e.g. set data changed
    # since export, or forme mismatch) — fall back to all sets rather than none.
    return out or list(entry["sets"])


def infer_opponent(
    species: str,
    revealed_moves: list[str],
    ability: Optional[str] = None,
    tera_type: Optional[str] = None,
) -> tuple[list[str], Optional[str], Optional[str]]:
    """
    Fill in an opponent Pokémon's unrevealed info from its randbats sets.

    Returns (moves, ability, tera_type):
      moves     — revealed moves first, then the most common remaining moves across
                  consistent sets, up to 4
      ability   — revealed, or the unique ability across consistent sets, else None
      tera_type — revealed, or the unique tera type across consistent sets, else None
    """
    moves = list(revealed_moves)[:4]
    sets = consistent_sets(species, moves, ability, tera_type)
    if not sets:
        return moves, ability, tera_type

    counts: Counter[str] = Counter()
    for s in sets:
        counts.update(m for m in s["moves"] if m not in moves)
    for m, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        if len(moves) >= 4:
            break
        moves.append(m)

    if not ability:
        abilities = {a for s in sets for a in s["abilities"]}
        ability = next(iter(abilities)) if len(abilities) == 1 else None
    if not tera_type:
        teras = {t for s in sets for t in s["tera_types"]}
        tera_type = next(iter(teras)) if len(teras) == 1 else None
    return moves, ability, tera_type
