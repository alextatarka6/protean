"""
Play rated Gen1OU ladder games on the real Pokémon Showdown server.

Credentials are read from environment variables PS_USERNAME and PS_PASSWORD.
Register a bot account at https://play.pokemonshowdown.com first.

To spectate the bot live, go to https://play.pokemonshowdown.com,
click "Watch a battle", and search for the bot's username.

Usage:
    export PS_USERNAME="YourBotName"
    export PS_PASSWORD="yourpassword"
    python scripts/ladder.py --checkpoint checkpoints/ppo_final.pt

    # Override credentials via flags (avoid — exposed in process list):
    python scripts/ladder.py --checkpoint checkpoints/ppo_final.pt \\
        --username YourBotName --password yourpassword

    # Play 20 games on the offensive team:
    python scripts/ladder.py --checkpoint checkpoints/ppo_final.pt \\
        --n-games 20 --team offensive
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from protean.rl_env import Gen1OUPlayer, SHOWDOWN_SERVER
from protean.model import Gen1OUPolicy
from protean.tokenizer import get_tokenizer
from protean.teams import get_team

from poke_env.environment import Battle


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


class LadderPlayer(Gen1OUPlayer):
    """Gen1OUPlayer that prints a result line and logs stats after each ladder battle."""

    def __init__(self, *args, history_file: str = "ladder_history.jsonl",
                 search_timeout: int = 120, **kwargs):
        self._history_file   = history_file
        self._search_timeout = search_timeout
        self._timer_set: set[str] = set()
        super().__init__(*args, **kwargs)

    async def _handle_battle_request(self, battle, maybe_default_order=False):
        if battle.battle_tag not in self._timer_set:
            self._timer_set.add(battle.battle_tag)
            await self.ps_client.send_message("/timer on", battle.battle_tag)
        await super()._handle_battle_request(battle, maybe_default_order)

    async def _ladder(self, n_games: int, sleep_between=None) -> None:
        await self.ps_client.logged_in.wait()
        start_time = perf_counter()

        for game_num in range(n_games):
            print(f"  Searching for game {game_num + 1}/{n_games}...", flush=True)
            async with self._battle_start_condition:
                await self.ps_client.search_ladder_game(self._format, self.next_team)
                print("  Queued. Waiting for opponent...", flush=True)
                try:
                    await asyncio.wait_for(
                        self._battle_start_condition.wait(),
                        timeout=self._search_timeout,
                    )
                except asyncio.TimeoutError:
                    print(
                        f"  No opponent found after {self._search_timeout}s — cancelling search.",
                        flush=True,
                    )
                    await self.ps_client.send_message("/cancelsearch")
                    break
                print("  Opponent found! Battle starting...", flush=True)
                while self._battle_count_queue.full():
                    async with self._battle_end_condition:
                        await self._battle_end_condition.wait()
                await self._battle_semaphore.acquire()
                if game_num < n_games - 1 and sleep_between is not None:
                    await asyncio.sleep(random.randint(0, sleep_between))

        await self._battle_count_queue.join()
        self.logger.info(
            "Laddering (%d battles) finished in %fs",
            n_games,
            perf_counter() - start_time,
        )

    def _battle_finished_callback(self, battle: Battle) -> None:
        super()._battle_finished_callback(battle)
        w = self.n_won_battles
        p = self.n_finished_battles
        result = "WIN " if battle.won else "LOSS"
        opp = battle.opponent_username or "?"
        rating     = battle.rating
        opp_rating = battle.opponent_rating
        rating_str = f"  (rating: {rating})" if rating is not None else ""
        print(f"  [{result}]  vs {opp:<20}  {w}W / {p - w}L  ({w / max(p, 1):.0%}){rating_str}")

        record = {
            "timestamp":  datetime.now(timezone.utc).isoformat(),
            "game":       p,
            "won":        bool(battle.won),
            "opponent":   opp,
            "our_rating": rating,
            "opp_rating": opp_rating,
        }
        with open(self._history_file, "a") as f:
            f.write(json.dumps(record) + "\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Play rated Gen1OU ladder on Pokémon Showdown")
    p.add_argument("--checkpoint", required=True,
                   help="Path to checkpoint (.pt)")
    p.add_argument("--team",       default="standard",
                   help="Team: standard/offensive/balanced/stall  (default: standard)")
    p.add_argument("--n-games",    type=int, default=10,
                   help="Number of ladder games to play  (default: 10)")
    p.add_argument("--username",   default=None,
                   help="PS username (overrides PS_USERNAME env var)")
    p.add_argument("--password",   default=None,
                   help="PS password (overrides PS_PASSWORD env var)")
    p.add_argument("--sample",          action="store_true",
                   help="Sample from policy (default: greedy)")
    p.add_argument("--search-timeout",  type=int, default=120,
                   help="Seconds to wait for a game to start before giving up (default: 120)")
    p.add_argument("--history-file",    default="ladder_history.jsonl",
                   help="JSONL file to append game results to (default: ladder_history.jsonl)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    username = args.username or os.environ.get("PS_USERNAME")
    password = args.password or os.environ.get("PS_PASSWORD")

    if not username or not password:
        print("Error: PS username and password are required.")
        print("Set PS_USERNAME and PS_PASSWORD environment variables, or pass --username/--password.")
        sys.exit(1)

    device = get_device()
    print(f"Device: {device}")
    model = load_model(args.checkpoint, device)

    player = LadderPlayer(
        model=model,
        device=device,
        sample=args.sample,
        username=username,
        password=password,
        team=get_team(args.team),
        server_configuration=SHOWDOWN_SERVER,
        history_file=args.history_file,
        search_timeout=args.search_timeout,
    )

    mode = "sample" if args.sample else "greedy"
    print(f"\nAccount:  {username}")
    print(f"Team:     {args.team}")
    print(f"Mode:     {mode}")
    print(f"Games:    {args.n_games}")
    print(f"\nSpectate: https://play.pokemonshowdown.com — search for '{username}'")
    print(f"History:  {args.history_file}")
    print()

    print("Connecting...", flush=True)
    deadline = time.time() + 15
    while not player.ps_client.logged_in.is_set():
        if time.time() > deadline:
            print("Error: login timed out after 15 seconds. Check username/password.")
            sys.exit(1)
        time.sleep(0.1)
    if player.next_team is None:
        print("Error: no team set. Pass --team <name> to specify a team.")
        sys.exit(1)
    print(f"Logged in as {username}. Searching for games...\n")

    try:
        asyncio.run(player.ladder(args.n_games))
    except KeyboardInterrupt:
        pass

    w = player.n_won_battles
    p = player.n_finished_battles
    print(f"\nFinal: {w}W / {p - w}L  ({w / max(p, 1):.0%} over {p} games)")


if __name__ == "__main__":
    main()
