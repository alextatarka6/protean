"""
Team inference: fill in missing pokemon information using usage stats.

Mirrors the NaiveUsagePredictor approach from metamon:
  - Missing move slots are sampled from the usage-stats move distribution,
    conditioned on which moves are already revealed (those are excluded).
  - Missing item / ability are sampled from their respective distributions.
  - Unrevealed team members are inferred by sampling from the combined
    teammates co-occurrence scores across already-known pokemon.

All sampling uses numpy for weighted choice without replacement. The rng
argument accepts any numpy Generator (np.random.default_rng(seed=...)),
making results reproducible when needed.
"""
from __future__ import annotations

import copy
from typing import Optional

import numpy as np

from protean.backend.replay_parser.types import BattlePokemon, SideState
from protean.backend.usage_stats import MovesetStats


def _sample_weighted(
    weights: dict[str, float],
    n: int,
    rng: np.random.Generator,
    exclude: Optional[set[str]] = None,
) -> list[str]:
    """
    Sample up to n keys from a weighted dict without replacement.
    Keys in `exclude` and the sentinel strings "Other"/"Nothing" are skipped.
    Returns fewer than n items if not enough valid candidates exist.
    """
    skip = {"Other", "Nothing"}
    if exclude:
        skip |= exclude
    valid = {k: float(v) for k, v in weights.items() if k not in skip and float(v) > 0}
    if not valid:
        return []
    keys = list(valid.keys())
    probs = np.array([valid[k] for k in keys], dtype=np.float64)
    probs /= probs.sum()
    n = min(n, len(keys))
    return rng.choice(keys, size=n, replace=False, p=probs).tolist()


def _infer_pokemon(
    pk: BattlePokemon,
    stats: MovesetStats,
    gen: int,
    rng: np.random.Generator,
) -> BattlePokemon:
    """
    Return a copy of `pk` with missing move/item/ability slots filled via sampling.
    Already-revealed information is preserved exactly.
    """
    pk = copy.deepcopy(pk)

    # Fill move slots up to 4, excluding already-revealed moves
    needed = 4 - len(pk.revealed_moves)
    if needed > 0 and stats.moves:
        sampled = _sample_weighted(
            stats.moves, needed, rng, exclude=set(pk.revealed_moves)
        )
        pk.revealed_moves.extend(sampled)

    # Item (gen 2+)
    if gen >= 2 and pk.item is None and stats.items:
        chosen = _sample_weighted(stats.items, 1, rng)
        if chosen:
            pk.item = chosen[0]

    # Ability (gen 3+)
    if gen >= 3 and pk.ability is None and stats.abilities:
        chosen = _sample_weighted(stats.abilities, 1, rng)
        if chosen:
            pk.ability = chosen[0]

    return pk


def _infer_team_members(
    known_species: set[str],
    format_stats: dict[str, MovesetStats],
    needed: int,
    rng: np.random.Generator,
) -> list[str]:
    """
    Infer `needed` additional species using teammates co-occurrence.

    At each step, scores every candidate species by summing the teammate
    co-occurrence weights from all currently-known species, then samples
    one candidate from the resulting distribution. The selected species
    is added to the known set before the next iteration.
    """
    result: list[str] = []
    all_known = set(known_species)

    for _ in range(needed):
        scores: dict[str, float] = {}
        for species in all_known:
            stats = format_stats.get(species.lower())
            if stats is None:
                continue
            for candidate, pct in stats.teammates.items():
                if (candidate not in all_known
                        and candidate in format_stats  # only infer species valid in this format
                        and float(pct) > 0):
                    scores[candidate] = scores.get(candidate, 0.0) + float(pct)

        if not scores:
            break

        keys = list(scores.keys())
        probs = np.array([scores[k] for k in keys], dtype=np.float64)
        probs /= probs.sum()
        chosen = rng.choice(keys, p=probs)

        result.append(chosen)
        all_known.add(chosen)

    return result


def infer_side(
    side: SideState,
    format_stats: dict[str, MovesetStats],
    gen: int,
    team_size: int = 6,
    rng: Optional[np.random.Generator] = None,
) -> SideState:
    """
    Return a deep-copied SideState with usage-stats inference applied.

    For each revealed pokemon:
      - Fills remaining move slots (up to 4) by sampling from the usage-stats
        move distribution, conditioned on already-revealed moves.
      - Fills item (gen 2+) and ability (gen 3+) if not observed.

    For unrevealed team slots (len(team) < team_size):
      - Infers additional pokemon by sampling from teammates co-occurrence
        data aggregated across all currently-known species.
      - Each inferred pokemon is then given a sampled moveset/item/ability.

    The original SideState is not modified.

    Args:
        side:         the SideState to enrich (not modified in place)
        format_stats: pre-loaded usage stats for the format
        gen:          game generation (affects which fields are inferred)
        team_size:    target team size (default 6)
        rng:          numpy Generator for reproducible sampling; if None,
                      a fresh default_rng() is used
    """
    if rng is None:
        rng = np.random.default_rng()

    side = copy.deepcopy(side)
    known_species = {pk.species for pk in side.team}

    # Enrich each already-revealed pokemon
    for i, pk in enumerate(side.team):
        stats = format_stats.get(pk.species.lower())
        if stats is not None:
            side.team[i] = _infer_pokemon(pk, stats, gen, rng)

    # Infer and append unrevealed team members
    slots_needed = team_size - len(side.team)
    if slots_needed > 0:
        new_species = _infer_team_members(known_species, format_stats, slots_needed, rng)
        for species in new_species:
            pk = BattlePokemon(species=species, nickname=species, level=100, hp=1.0)
            stats = format_stats.get(species.lower())
            if stats is not None:
                pk = _infer_pokemon(pk, stats, gen, rng)
            side.team.append(pk)

    return side
