"""Run N battles between two chosen players on the local Showdown server.

Usage:
  python scripts/run_battle.py random heuristic
  python scripts/run_battle.py maxdamage heuristic --battles 20

Players: random | maxdamage | heuristic
"""
import argparse
import asyncio
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from poke_env.player.baselines import RandomPlayer
from poke_env.ps_client.account_configuration import AccountConfiguration
from poke_env.ps_client.server_configuration import LocalhostServerConfiguration

from protean.agents.heuristic import HeuristicPlayer, MaxDamagePlayer

PLAYERS = {
    "random": RandomPlayer,
    "maxdamage": MaxDamagePlayer,
    "heuristic": HeuristicPlayer,
}


def _make(name: str, slot: int):
    cls = PLAYERS[name]
    username = f"{name.capitalize()}{slot}"
    return cls(
        battle_format="gen9randombattle",
        account_configuration=AccountConfiguration(username, None),
        server_configuration=LocalhostServerConfiguration,
        max_concurrent_battles=1,
    )


async def main(p1_name: str, p2_name: str, n: int) -> None:
    p1 = _make(p1_name, 1)
    p2 = _make(p2_name, 2)

    print(f"{p1.username} vs {p2.username}  —  {n} battles\n")

    await p1.battle_against(p2, n_battles=n)

    p1_wins = p1.n_won_battles
    p2_wins = p2.n_won_battles
    played = p1.n_finished_battles

    print(f"  {p1.username:<20} {p1_wins}W / {played - p1_wins}L  ({p1_wins/played:.0%})")
    print(f"  {p2.username:<20} {p2_wins}W / {played - p2_wins}L  ({p2_wins/played:.0%})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run battles between two players.")
    parser.add_argument("p1", choices=PLAYERS, help="Player 1")
    parser.add_argument("p2", choices=PLAYERS, help="Player 2")
    parser.add_argument("--battles", type=int, default=10)
    args = parser.parse_args()

    asyncio.run(main(args.p1, args.p2, args.battles))
