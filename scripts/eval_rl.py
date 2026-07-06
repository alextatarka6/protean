"""
Evaluate PPO or BC checkpoint(s) via live battles on the local Showdown server.

BC-only mode  (--eval-bc):
  Runs 2 match-ups for --n-battles battles each:
    1. BC vs Random       — baseline floor
    2. BC vs itself       — sanity check; should be ~50%

Single-checkpoint mode  (--checkpoint):
  Runs 3 match-ups for --n-battles battles each:
    1. PPO vs Random      — baseline floor
    2. PPO vs BC policy   — measures RL improvement over imitation
    3. PPO vs itself      — sanity check; should be ~50%

Sweep mode  (--sweep):
  Iterates every ppo_ep*.pt in --checkpoint-dir (sorted), runs match-ups
  1 & 2 for --n-battles battles each, and prints a summary table.
  Players are created once and model weights are hot-swapped between
  checkpoints — no username collisions, no reconnects.

Usage:
    # BC baseline
    python scripts/eval_rl.py --eval-bc \\
        --bc-checkpoint checkpoints/bc_final.pt

    # Single PPO checkpoint
    python scripts/eval_rl.py \\
        --checkpoint checkpoints/ppo_ep0006000.pt \\
        --bc-checkpoint checkpoints/bc_final.pt

    # Sweep all checkpoints (40 battles each for speed)
    python scripts/eval_rl.py --sweep \\
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


def load_model(checkpoint_path: str, device: torch.device,
               verbose: bool = True) -> Gen1OUPolicy:
    tokenizer = get_tokenizer()
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = Gen1OUPolicy(vocab_size=tokenizer.vocab_size).to(device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    if verbose:
        ep = ckpt.get("episode", ckpt.get("step", "?"))
        print(f"  Loaded {checkpoint_path}  (episode/step {ep})")
    return model


def swap_weights(model: Gen1OUPolicy, checkpoint_path: str,
                 device: torch.device) -> int:
    """Load weights from checkpoint into an existing model in-place.
    Returns the episode number stored in the checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    return int(ckpt.get("episode", ckpt.get("step", 0)))


def make_random_player(username: str, team: str) -> RandomPlayer:
    return RandomPlayer(
        account_configuration=AccountConfiguration(username, None),
        server_configuration=LOCAL_SERVER,
        battle_format="gen1ou",
        team=team,
        max_concurrent_battles=1,
    )


# ---------------------------------------------------------------------------
# Match runner
# ---------------------------------------------------------------------------

async def run_match(player, opponent, n_battles: int) -> None:
    """Run n_battles, 4 at a time, with brief pauses for server teardown."""
    batch = 4
    done  = 0
    while done < n_battles:
        this_batch = min(batch, n_battles - done)
        await player.battle_against(opponent, n_battles=this_batch)
        done += this_batch
        time.sleep(0.5)


def run_and_measure(player, opponent, n: int) -> tuple[int, int]:
    """Run n battles, return (wins_this_run, battles_this_run)."""
    w0 = player.n_won_battles
    p0 = player.n_finished_battles
    asyncio.run(run_match(player, opponent, n))
    time.sleep(1.0)
    return player.n_won_battles - w0, player.n_finished_battles - p0


# ---------------------------------------------------------------------------
# BC-only eval
# ---------------------------------------------------------------------------

def evaluate_bc(args: argparse.Namespace) -> None:
    device = get_device()
    print(f"Device: {device}\n")

    print("Loading BC checkpoint…")
    bc_model = load_model(args.bc_checkpoint, device)
    print()

    n     = args.n_battles
    teams = ALL_TEAMS

    print(f"Running {n} battles per match-up  (greedy eval agent)\n")
    print(f"{'Match-up':<35} {'W/P':>7}  Win rate")
    print("-" * 55)

    # 1. BC vs Random
    bc_v_rand = Gen1OUPlayer(model=bc_model, device=device, sample=False,
                             username="EvalBC_1", team=teams[0])
    rand_opp  = make_random_player("EvalBC_Rand", teams[1])
    w, p = run_and_measure(bc_v_rand, rand_opp, n)
    print(f"  {'BC vs Random':<33} {w:>3} / {p:<3}  {w/max(p,1):.1%}")

    # 2. BC vs itself
    bc_a = Gen1OUPlayer(model=bc_model, device=device, sample=False,
                        username="EvalBC_A", team=teams[1])
    bc_b = Gen1OUPlayer(model=bc_model, device=device, sample=False,
                        username="EvalBC_B", team=teams[2])
    w, p = run_and_measure(bc_a, bc_b, n)
    print(f"  {'BC vs BC (self)':<33} {w:>3} / {p:<3}  {w/max(p,1):.1%}")

    print("\nDone.")


# ---------------------------------------------------------------------------
# Single-checkpoint eval
# ---------------------------------------------------------------------------

def evaluate(args: argparse.Namespace) -> None:
    device = get_device()
    print(f"Device: {device}\n")

    print("Loading checkpoints…")
    eval_model = load_model(args.checkpoint, device)
    bc_model   = load_model(args.bc_checkpoint, device)
    print()

    n     = args.n_battles
    teams = ALL_TEAMS

    print(f"Running {n} battles per match-up  (greedy eval agent)\n")
    print(f"{'Match-up':<35} {'W/P':>7}  Win rate")
    print("-" * 55)

    # 1. PPO vs Random
    eval_v_rand = Gen1OUPlayer(model=eval_model, device=device, sample=False,
                               username="Eval_PPO_1", team=teams[0])
    rand_opp    = make_random_player("Eval_Rand_1", teams[1])
    w, p = run_and_measure(eval_v_rand, rand_opp, n)
    print(f"  {'PPO vs Random':<33} {w:>3} / {p:<3}  {w/max(p,1):.1%}")

    # 2. PPO vs BC
    eval_v_bc = Gen1OUPlayer(model=eval_model, device=device, sample=False,
                             username="Eval_PPO_2", team=teams[0])
    bc_opp    = Gen1OUPlayer(model=bc_model, device=device, sample=False,
                             username="Eval_BC_1", team=teams[2])
    w, p = run_and_measure(eval_v_bc, bc_opp, n)
    print(f"  {'PPO vs BC policy':<33} {w:>3} / {p:<3}  {w/max(p,1):.1%}")

    # 3. PPO vs itself
    eval_a = Gen1OUPlayer(model=eval_model, device=device, sample=False,
                          username="Eval_PPO_A", team=teams[1])
    eval_b = Gen1OUPlayer(model=eval_model, device=device, sample=False,
                          username="Eval_PPO_B", team=teams[3])
    w, p = run_and_measure(eval_a, eval_b, n)
    print(f"  {'PPO vs PPO (self)':<33} {w:>3} / {p:<3}  {w/max(p,1):.1%}")

    print("\nDone.")


# ---------------------------------------------------------------------------
# Sweep mode
# ---------------------------------------------------------------------------

def sweep(args: argparse.Namespace) -> None:
    device = get_device()
    print(f"Device: {device}\n")

    ckpt_dir = Path(args.checkpoint_dir)
    checkpoints = sorted(ckpt_dir.glob("ppo_ep*.pt"))
    if not checkpoints:
        print(f"No ppo_ep*.pt files found in {ckpt_dir}")
        return

    # Optional episode range filter
    if args.min_episode or args.max_episode:
        def _ep(p: Path) -> int:
            try: return int(p.stem.replace("ppo_ep", ""))
            except ValueError: return -1
        checkpoints = [
            p for p in checkpoints
            if (args.min_episode is None or _ep(p) >= args.min_episode)
            and (args.max_episode is None or _ep(p) <= args.max_episode)
        ]
    if not checkpoints:
        print("No checkpoints match the episode range filter.")
        return

    print(f"Found {len(checkpoints)} checkpoints in {ckpt_dir}")
    print(f"Battles per match-up: {args.n_battles}\n")

    # Create a shared eval model — weights will be hot-swapped per checkpoint.
    # Players hold a reference to this model object, so swapping in-place
    # is immediately visible to all players without reconnecting.
    eval_model = Gen1OUPolicy(vocab_size=get_tokenizer().vocab_size).to(device)
    eval_model.eval()

    print("Loading BC checkpoint…")
    bc_model = load_model(args.bc_checkpoint, device, verbose=True)
    print()

    # Create players once — reused across all checkpoints.
    n     = args.n_battles
    teams = ALL_TEAMS

    eval_v_rand = Gen1OUPlayer(model=eval_model, device=device, sample=False,
                               username="Sweep_PPO_R", team=teams[0])
    rand_opp    = make_random_player("Sweep_Rand", teams[1])

    eval_v_bc = Gen1OUPlayer(model=eval_model, device=device, sample=False,
                             username="Sweep_PPO_B", team=teams[0])
    bc_opp    = Gen1OUPlayer(model=bc_model, device=device, sample=False,
                             username="Sweep_BC", team=teams[2])

    # Header
    print(f"{'Episode':>8}  {'vs Random':>10}  {'vs BC':>8}")
    print("-" * 35)

    results = []
    for ckpt_path in checkpoints:
        ep = swap_weights(eval_model, str(ckpt_path), device)

        w_r, p_r = run_and_measure(eval_v_rand, rand_opp, n)
        w_b, p_b = run_and_measure(eval_v_bc,   bc_opp,   n)

        wr_r = w_r / max(p_r, 1)
        wr_b = w_b / max(p_b, 1)
        results.append((ep, wr_r, wr_b))

        print(f"{ep:>8}  {wr_r:>9.1%}  {wr_b:>7.1%}")

    # Summary
    print("\n" + "=" * 35)
    best_rand = max(results, key=lambda x: x[1])
    best_bc   = max(results, key=lambda x: x[2])
    print(f"Best vs Random:  ep {best_rand[0]:>6}  {best_rand[1]:.1%}")
    print(f"Best vs BC:      ep {best_bc[0]:>6}  {best_bc[2]:.1%}")
    print("\nDone.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate PPO checkpoint(s) via live battles")

    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--eval-bc", action="store_true",
                      help="Evaluate BC checkpoint only (vs Random + vs itself)")
    mode.add_argument("--checkpoint", type=str,
                      help="Single PPO checkpoint to evaluate (.pt)")
    mode.add_argument("--sweep", action="store_true",
                      help="Sweep all ppo_ep*.pt in --checkpoint-dir")

    p.add_argument("--bc-checkpoint",  type=str, required=True,
                   help="BC checkpoint (.pt); used as opponent for PPO modes, subject for --eval-bc")
    p.add_argument("--checkpoint-dir", type=str, default="checkpoints",
                   help="Directory to scan for ppo_ep*.pt  (default: checkpoints/)")
    p.add_argument("--min-episode",    type=int, default=None,
                   help="Only evaluate checkpoints at or after this episode")
    p.add_argument("--max-episode",    type=int, default=None,
                   help="Only evaluate checkpoints at or before this episode")
    p.add_argument("--n-battles",      type=int, default=100,
                   help="Battles per match-up per checkpoint (default: 100; use 40 for sweeps)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.eval_bc:
        evaluate_bc(args)
    elif args.sweep:
        sweep(args)
    else:
        evaluate(args)
