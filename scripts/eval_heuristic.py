"""Phase 1 evaluation: measure HeuristicPlayer win rates.

Targets:
  - vs RandomPlayer    ≥ 95 % win rate
  - vs MaxDamagePlayer > 50 % win rate

Usage:
  python scripts/eval_heuristic.py            # 100 battles each
  python scripts/eval_heuristic.py --battles 50
"""
import argparse
import asyncio

from poke_env.player.baselines import RandomPlayer
from poke_env.ps_client.account_configuration import AccountConfiguration
from poke_env.ps_client.server_configuration import LocalhostServerConfiguration

from protean.agents.heuristic import HeuristicPlayer, MaxDamagePlayer


def _player(cls, name):
    return cls(
        battle_format="gen9randombattle",
        account_configuration=AccountConfiguration(name, None),
        server_configuration=LocalhostServerConfiguration,
        max_concurrent_battles=1,
    )


async def run_matchup(p1, p2, n: int, label: str) -> float:
    wins_before = p1.n_won_battles
    finished_before = p1.n_finished_battles

    await p1.battle_against(p2, n_battles=n)

    played = p1.n_finished_battles - finished_before
    wins = p1.n_won_battles - wins_before
    rate = wins / played if played else 0.0

    print(f"{label}")
    print(f"  {p1.username}:  {wins}W / {played - wins}L  ({rate:.1%})")
    return rate


async def main(n: int) -> None:
    h1 = _player(HeuristicPlayer, "Heuristic1")
    r1 = _player(RandomPlayer, "Random1")
    rate_vs_random = await run_matchup(
        h1, r1, n, f"HeuristicPlayer vs RandomPlayer ({n} battles)"
    )

    print()

    h2 = _player(HeuristicPlayer, "Heuristic2")
    m1 = _player(MaxDamagePlayer, "MaxDamage1")
    rate_vs_max = await run_matchup(
        h2, m1, n, f"HeuristicPlayer vs MaxDamagePlayer ({n} battles)"
    )

    print()
    passed = True
    if rate_vs_random >= 0.95:
        print(f"  PASS  vs Random    ({rate_vs_random:.1%} ≥ 95 %)")
    else:
        print(f"  FAIL  vs Random    ({rate_vs_random:.1%} < 95 %)")
        passed = False

    if rate_vs_max > 0.50:
        print(f"  PASS  vs MaxDamage ({rate_vs_max:.1%} > 50 %)")
    else:
        print(f"  FAIL  vs MaxDamage ({rate_vs_max:.1%} ≤ 50 %)")
        passed = False

    if passed:
        print("\nPhase 1 heuristic targets met.")
    else:
        print("\nSome targets not met — check the heuristic logic.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--battles", type=int, default=100)
    args = parser.parse_args()
    asyncio.run(main(args.battles))
