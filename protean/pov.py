"""
POV reconstruction: convert a ParsedBattle into per-player viewpoint sequences.

Follows the two-phase approach from the metamon replay reconstruction diagram:

  Phase 1 (parser): simulate the battle forward, accumulating revealed state turn by turn.

  Phase 2 (here — backward reconstruction):
    a. Collect the most complete revealed state of the POV player's team across ALL turns.
    b. Run team inference ONCE on that complete set → a fixed 6-pokemon initial team.
    c. For each turn, overlay the inferred initial team with the turn's observed
       HP / status / boosts. Moves, items, and abilities come from the inferred initial
       state and are stable across all snapshots.

The opponent's side is never inferred — only pokemon that have been switched into
battle by that turn are included, and only their observed moves/HP/status.

Turns where the POV player's action is None are excluded from the output.
"""
from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Optional

import numpy as np

from protean.backend.replay_parser.types import (
    BattlePokemon,
    ParsedBattle,
    POVReplay,
    POVSnapshot,
    SideState,
    Winner,
)

if TYPE_CHECKING:
    from protean.backend.usage_stats import MovesetStats


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mask_opp_side(side: SideState) -> SideState:
    """Return a new SideState with only battle-revealed opponent pokemon."""
    return SideState(
        player=side.player,
        active=side.active,
        team=[pk for pk in side.team if pk.seen_in_battle],
        conditions=side.conditions,
    )


def _most_revealed_my_side(battle: ParsedBattle, pov_player: str) -> SideState:
    """
    Collect the most complete revealed state of the POV player's team across all turns.

    Iterates every TurnSnapshot and unions the revealed moves / items / abilities for
    each pokemon. The resulting SideState represents the INITIAL state (hp=1.0,
    no conditions, no active) but with the maximum information observed during battle.
    """
    best: dict[str, BattlePokemon] = {}
    last_player: str = pov_player

    for turn in battle.turns:
        side = turn.p1 if pov_player == "p1" else turn.p2
        last_player = side.player
        for pk in side.team:
            if pk.species not in best:
                best[pk.species] = copy.deepcopy(pk)
                # Reset to initial state — HP/status/boosts are applied per-turn later
                best[pk.species].hp = 1.0
                best[pk.species].status = None
                best[pk.species].fainted = False
            else:
                existing = best[pk.species]
                for move in pk.revealed_moves:
                    if move not in existing.revealed_moves:
                        existing.revealed_moves.append(move)
                if existing.item is None and pk.item is not None:
                    existing.item = pk.item
                if existing.ability is None and pk.ability is not None:
                    existing.ability = pk.ability

    return SideState(
        player=last_player,
        active=None,
        team=list(best.values()),
        conditions={},
    )


def _overlay_inferred_on_turn(
    inferred: SideState,
    turn_side: SideState,
) -> SideState:
    """
    Build a SideState using the inferred team composition but turn-specific HP/status/boosts.

    For each pokemon in the inferred team:
      - If revealed by this turn: apply observed HP, status, boosts, fainted.
      - If not yet revealed: assume full HP, no status (unknown from POV player's perspective).

    Moves, item, and ability always come from `inferred` (stable across all turns).
    """
    turn_by_species = {pk.species: pk for pk in turn_side.team}

    overlaid: list[BattlePokemon] = []
    for pk in inferred.team:
        pk = copy.deepcopy(pk)
        obs = turn_by_species.get(pk.species)
        if obs is not None:
            pk.hp = obs.hp
            pk.status = obs.status
            pk.boosts = copy.deepcopy(obs.boosts)
            pk.fainted = obs.fainted
        else:
            pk.hp = 1.0
            pk.status = None
            pk.fainted = False
        overlaid.append(pk)

    active_species = turn_side.active.species if turn_side.active else None
    active = next((pk for pk in overlaid if pk.species == active_species), None)

    return SideState(
        player=turn_side.player,
        active=active,
        team=overlaid,
        conditions=copy.deepcopy(turn_side.conditions),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def reconstruct_pov(
    battle: ParsedBattle,
    pov_player: str = "p1",
    format_stats: Optional[dict[str, MovesetStats]] = None,
    rng: Optional[np.random.Generator] = None,
) -> Optional[POVReplay]:
    """
    Build a POVReplay from one player's perspective using backward reconstruction.

    Args:
        battle:       a fully parsed and validated ParsedBattle
        pov_player:   "p1" or "p2"
        format_stats: pre-loaded usage stats (from load_format_stats); if None,
                      only observed information is used (no inference)
        rng:          numpy Generator for reproducible sampling

    Returns:
        POVReplay, or None if no turns have a known action for the POV player.
    """
    from protean.backend.team_inference import infer_side

    if rng is None and format_stats is not None:
        rng = np.random.default_rng()

    if pov_player == "p1":
        player_name, opponent_name = battle.p1_name, battle.p2_name
    else:
        player_name, opponent_name = battle.p2_name, battle.p1_name

    # Phase 2a — collect best-known state across the whole battle
    most_revealed = _most_revealed_my_side(battle, pov_player)

    # Phase 2b — infer once: fill move slots, items, abilities, unrevealed team members
    if format_stats is not None:
        inferred_initial = infer_side(most_revealed, format_stats, battle.gen, rng=rng)
    else:
        inferred_initial = most_revealed

    # Phase 2c — for each turn, overlay inferred team with observed HP/status/boosts
    snapshots: list[POVSnapshot] = []
    for turn in battle.turns:
        if pov_player == "p1":
            my_action = turn.p1_action
            opp_action = turn.p2_action
            my_side_at_turn = turn.p1
            opp_side = turn.p2
        else:
            my_action = turn.p2_action
            opp_action = turn.p1_action
            my_side_at_turn = turn.p2
            opp_side = turn.p1

        if my_action is None:
            continue

        my_side = _overlay_inferred_on_turn(inferred_initial, my_side_at_turn)
        opp_side = _mask_opp_side(opp_side)

        snapshots.append(POVSnapshot(
            turn_number=turn.turn_number,
            my_side=my_side,
            opp_side=opp_side,
            field=turn.field,
            my_action=my_action,
            opp_action=opp_action,
        ))

    if not snapshots:
        return None

    return POVReplay(
        battle_id=battle.battle_id,
        format=battle.format,
        gen=battle.gen,
        pov_player=pov_player,
        player_name=player_name,
        opponent_name=opponent_name,
        winner=battle.winner,
        snapshots=snapshots,
    )


def reconstruct_both_povs(
    battle: ParsedBattle,
    format_stats: Optional[dict[str, MovesetStats]] = None,
    rng: Optional[np.random.Generator] = None,
) -> tuple[Optional[POVReplay], Optional[POVReplay]]:
    """Convenience wrapper: returns (p1_pov, p2_pov)."""
    return (
        reconstruct_pov(battle, "p1", format_stats, rng),
        reconstruct_pov(battle, "p2", format_stats, rng),
    )
