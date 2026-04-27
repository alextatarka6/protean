"""Phase 0 smoke test: two RandomPlayers battling on the local Showdown server."""
import argparse
import asyncio

from poke_env.ps_client.server_configuration import LocalhostServerConfiguration
from poke_env.player.baselines import RandomPlayer
from poke_env.ps_client.account_configuration import AccountConfiguration
from poke_env.battle.abstract_battle import AbstractBattle


_battle_count = 0
_printed: set[str] = set()


def _print_battle_summary(battle: AbstractBattle) -> None:
    global _battle_count
    if battle.battle_tag in _printed:
        return
    _printed.add(battle.battle_tag)
    _battle_count += 1

    if battle.won:
        winner = battle.player_username
        loser = battle.opponent_username
        winner_team = battle.team
        loser_team = battle.opponent_team
    else:
        winner = battle.opponent_username
        loser = battle.player_username
        winner_team = battle.opponent_team
        loser_team = battle.team

    def survivors(team: dict) -> list[str]:
        return [mon.name for mon in team.values() if not mon.fainted]

    def full_team(team: dict) -> list[str]:
        return [mon.name for mon in team.values()]

    print(f"\n--- Battle {_battle_count} ({battle.turn} turns) ---")
    print(f"  Winner : {winner}")
    print(f"    Team     : {', '.join(full_team(winner_team))}")
    print(f"    Survived : {', '.join(survivors(winner_team)) or 'none'}")
    print(f"  Loser  : {loser}")
    print(f"    Team     : {', '.join(full_team(loser_team))}")
    print(f"    Survived : {', '.join(survivors(loser_team)) or 'none'}")


class VerboseRandomPlayer(RandomPlayer):
    def _battle_finished_callback(self, battle: AbstractBattle) -> None:
        _print_battle_summary(battle)


async def main(n_battles: int) -> None:
    p1 = VerboseRandomPlayer(
        battle_format="gen9randombattle",
        account_configuration=AccountConfiguration("RandomBot1", None),
        server_configuration=LocalhostServerConfiguration,
        max_concurrent_battles=1,
    )
    p2 = VerboseRandomPlayer(
        battle_format="gen9randombattle",
        account_configuration=AccountConfiguration("RandomBot2", None),
        server_configuration=LocalhostServerConfiguration,
        max_concurrent_battles=1,
    )

    await p1.battle_against(p2, n_battles=n_battles)

    print(f"\n=== Final record ===")
    print(f"  RandomBot1  {p1.n_won_battles}W / {p1.n_finished_battles - p1.n_won_battles}L")
    print(f"  RandomBot2  {p2.n_won_battles}W / {p2.n_finished_battles - p2.n_won_battles}L")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run two random bots against each other.")
    parser.add_argument("--battles", type=int, default=10, help="Number of battles to play.")
    args = parser.parse_args()
    asyncio.run(main(args.battles))
