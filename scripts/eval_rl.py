"""
Evaluate a PPO (or BC) checkpoint via live battles on the local Showdown server.

Runs three match-ups, each for --n-battles battles:
  1. Eval agent  vs  Random player   — baseline floor
  2. Eval agent  vs  BC policy       — measures RL improvement over imitation
  3. Eval agent  vs  itself (PPO)    — sanity check; should be ~50%

The eval agent always plays greedy (argmax, sample=False).

Usage:
    # Start server first:
    ./scripts/start_server.sh &

    python scripts/eval_rl.py \\
        --checkpoint checkpoints/ppo_ep0006000.pt \\
        --bc-checkpoint checkpoints/bc_final.pt

    # Fewer battles for a quick check:
    python scripts/eval_rl.py \\
        --checkpoint checkpoints/ppo_ep0006000.pt \\
        --bc-checkpoint checkpoints/bc_final.pt \\
        --n-battles 40
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# rl_env must be imported before any other poke-env code (applies the
# to_id_str(None) monkey-patch for gen1 ability handling).
from protean.rl_env import Gen1OUPlayer, LOCAL_SERVER
from protean.model import Gen1OUPolicy
from protean.tokenizer import get_tokenizer
from protean.teams import ALL_TEAMS

from poke_env.player import RandomPlayer
from poke_env.ps_client.account_configuration import AccountConfiguration


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model(checkpoint_path: str, device: torch.device) -> Gen1OUPolicy:
    tokenizer = get_tokenizer()
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = Gen1OUPolicy(vocab_size=tokenizer.vocab_size).to(device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    ep = ckpt.get("episode", ckpt.get("step", "?"))
    print(f"  Loaded {checkpoint_path}  (episode/step {ep})")
    return model


def make_random_player(username: str, team: str) -> RandomPlayer:
    return RandomPlayer(
        account_configuration=AccountConfiguration(username, None),
        server_configuration=LOCAL_SERVER,
        battle_format="gen1ou",
        team=team,
        max_concurrent_battles=1,
    )


def print_result(label: str, player: Gen1OUPlayer, n: int) -> None:
    wins   = player.n_won_battles
    played = player.n_finished_battles
    rate   = wins / played if played else 0.0
    print(f"  {label:<35} {wins:>3} / {played:<3}  win rate = {rate:.1%}")


# ---------------------------------------------------------------------------
# Match runner
# ---------------------------------------------------------------------------

async def run_match(player: Gen1OUPlayer, opponent, n_battles: int) -> None:
    """Run n_battles between player and opponent, with a brief pause between
    rounds to let the server clear previous challenge state."""
    batch = 4   # run 4 at a time (matches train_ppo parallelism)
    done  = 0
    while done < n_battles:
        this_batch = min(batch, n_battles - done)
        await player.battle_against(opponent, n_battles=this_batch)
        done += this_batch
        time.sleep(0.5)   # allow POKE_LOOP to finish teardown


# ---------------------------------------------------------------------------
# Main eval
# ---------------------------------------------------------------------------

def evaluate(args: argparse.Namespace) -> None:
    device = get_device()
    print(f"Device: {device}\n")

    print("Loading checkpoints…")
    eval_model = load_model(args.checkpoint, device)
    bc_model   = load_model(args.bc_checkpoint, device)
    print()

    n = args.n_battles
    teams = ALL_TEAMS

    print(f"Running {n} battles per match-up  (greedy eval agent)\n")
    print(f"{'Match-up':<35} {'W/P':>7}  Win rate")
    print("-" * 55)

    # ------------------------------------------------------------------
    # 1. Eval vs Random
    # ------------------------------------------------------------------
    eval_v_rand = Gen1OUPlayer(
        model=eval_model, device=device, sample=False,
        username="Eval_PPO_1", team=teams[0],
    )
    rand_opp = make_random_player("Eval_Rand_1", teams[1])
    asyncio.run(run_match(eval_v_rand, rand_opp, n))
    print_result("PPO vs Random", eval_v_rand, n)

    time.sleep(1.0)

    # ------------------------------------------------------------------
    # 2. Eval vs BC
    # ------------------------------------------------------------------
    eval_v_bc = Gen1OUPlayer(
        model=eval_model, device=device, sample=False,
        username="Eval_PPO_2", team=teams[0],
    )
    bc_opp = Gen1OUPlayer(
        model=bc_model, device=device, sample=False,
        username="Eval_BC_1", team=teams[2],
    )
    asyncio.run(run_match(eval_v_bc, bc_opp, n))
    print_result("PPO vs BC policy", eval_v_bc, n)

    time.sleep(1.0)

    # ------------------------------------------------------------------
    # 3. Eval vs itself  (sanity check)
    # ------------------------------------------------------------------
    eval_a = Gen1OUPlayer(
        model=eval_model, device=device, sample=False,
        username="Eval_PPO_A", team=teams[1],
    )
    eval_b = Gen1OUPlayer(
        model=eval_model, device=device, sample=False,
        username="Eval_PPO_B", team=teams[3],
    )
    asyncio.run(run_match(eval_a, eval_b, n))
    print_result("PPO vs PPO (self)", eval_a, n)

    print()
    print("Done.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a PPO checkpoint via live battles")
    p.add_argument("--checkpoint",    type=str, required=True,
                   help="PPO checkpoint to evaluate (.pt)")
    p.add_argument("--bc-checkpoint", type=str, required=True,
                   help="BC checkpoint for comparison (.pt)")
    p.add_argument("--n-battles",     type=int, default=100,
                   help="Battles per match-up (default: 100)")
    return p.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
