"""Phase 1 stub: round-robin tournament evaluator."""
import asyncio
from typing import List

from poke_env.player.player import Player


async def round_robin(players: List[Player], n_battles: int = 10) -> dict[str, dict]:
    """Run every pair of players against each other and return win/loss records.

    Returns a dict keyed by username with {"wins": int, "losses": int}.
    """
    results: dict[str, dict] = {p.username: {"wins": 0, "losses": 0} for p in players}

    for i, p1 in enumerate(players):
        for p2 in players[i + 1:]:
            wins_before_p1 = p1.n_won_battles
            finished_before_p1 = p1.n_finished_battles

            await p1.battle_against(p2, n_battles=n_battles)

            p1_wins = p1.n_won_battles - wins_before_p1
            p1_played = p1.n_finished_battles - finished_before_p1
            p2_wins = p1_played - p1_wins

            results[p1.username]["wins"] += p1_wins
            results[p1.username]["losses"] += p2_wins
            results[p2.username]["wins"] += p2_wins
            results[p2.username]["losses"] += p1_wins

    return results
