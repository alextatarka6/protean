"""
Play against a trained Gen1OU checkpoint in the terminal.

The bot prints its policy distribution after each of its decisions so you can
see what it was considering. Both you and the bot connect to the local Showdown
server as players — the server handles all battle logic.

Future: for a visual UI, route the human side through the Showdown browser
at http://localhost:8001 instead of stdin/stdout.

Usage:
    ./scripts/start_server.sh &
    python scripts/play_vs_agent.py --checkpoint checkpoints/ppo_final.pt
    python scripts/play_vs_agent.py --checkpoint checkpoints/bc_final.pt \\
        --bot-team offensive --human-team standard --n-battles 3
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from protean.rl_env import Gen1OUPlayer, HumanPlayer
from protean.model import Gen1OUPolicy
from protean.tokenizer import get_tokenizer
from protean.teams import get_team


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model(path: str, device: torch.device) -> Gen1OUPolicy:
    ckpt = torch.load(path, map_location=device)
    model = Gen1OUPolicy(vocab_size=get_tokenizer().vocab_size).to(device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    ep = ckpt.get("episode", ckpt.get("step", "?"))
    print(f"Loaded {path}  (episode/step {ep})")
    return model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Play against a Gen1OU checkpoint")
    p.add_argument("--checkpoint",     required=True,
                   help="Path to checkpoint (.pt) — BC or PPO")
    p.add_argument("--bot-team",       default="standard",
                   help="Bot team: standard/offensive/balanced/stall  (default: standard)")
    p.add_argument("--human-team",     default="balanced",
                   help="Your team: standard/offensive/balanced/stall  (default: balanced)")
    p.add_argument("--n-battles",      type=int, default=1,
                   help="Number of battles  (default: 1)")
    p.add_argument("--sample",         action="store_true",
                   help="Sample from policy distribution (default: greedy)")
    p.add_argument("--bot-username",   default="ProteanBot")
    p.add_argument("--human-username", default="Human")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device()
    print(f"Device: {device}")

    model = load_model(args.checkpoint, device)

    bot = Gen1OUPlayer(
        model=model, device=device,
        sample=args.sample, verbose=True,
        username=args.bot_username,
        team=get_team(args.bot_team),
    )
    human = HumanPlayer(
        username=args.human_username,
        team=get_team(args.human_team),
    )

    mode = "sample" if args.sample else "greedy"
    print(f"\nBot:    {args.bot_username}  ({args.bot_team} team, {mode})")
    print(f"You:    {args.human_username}  ({args.human_team} team)")
    print(f"Format: {args.n_battles} battle(s)\n")

    wins = 0
    for i in range(args.n_battles):
        if args.n_battles > 1:
            print(f"\n{'='*44}")
            print(f"Battle {i + 1} / {args.n_battles}")

        w0 = human.n_won_battles
        asyncio.run(human.battle_against(bot, n_battles=1))
        time.sleep(1.0)

        won = human.n_won_battles > w0
        wins += int(won)
        print(f"\n{'You win!' if won else 'Bot wins.'}")

    if args.n_battles > 1:
        p = args.n_battles
        print(f"\nFinal: {wins}W / {p - wins}L  ({wins / p:.0%} win rate)")


if __name__ == "__main__":
    main()
