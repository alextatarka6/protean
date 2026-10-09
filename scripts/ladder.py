"""
Play rated ladder games (default format: gen1ou) on the real Pokémon Showdown server.

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
import logging
import os
import random
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import torch
from dotenv import load_dotenv
from websockets.exceptions import ConnectionClosed

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from protean import cli_errors
from protean.live import BRANCH as LIVE_BRANCH, LivePublisher, format_event, snapshot_battle
from protean.formats import FORMAT_NAMES, BattleFormat, get_format
from protean.rl_env import ProteanPlayer, SHOWDOWN_SERVER
from protean.model import ProteanPolicy

from poke_env.environment import Battle

cli_errors.install()  # one-line errors for this unattended script (PROTEAN_TRACEBACK=1 to disable)


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model(path: str, fmt: BattleFormat, device: torch.device) -> ProteanPolicy:
    ckpt = torch.load(path, map_location=device)
    model = ProteanPolicy(**fmt.model_kwargs()).to(device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    ep = ckpt.get("episode", ckpt.get("step", "?"))
    print(f"Loaded {path}  (episode/step {ep})")
    return model


class LadderPlayer(ProteanPlayer):
    """ProteanPlayer that prints a result line and logs stats after each ladder battle."""

    def __init__(self, *args, history_file: str = "ladder_history.jsonl",
                 search_timeout: int = 120, log_dir: str = "ladder_logs",
                 checkpoint: str = "", live=None, **kwargs):
        self._live           = live                  # LivePublisher or None
        self._events: dict[str, list[dict]] = {}     # battle_tag -> recent public events
        self._history_file   = history_file
        self._search_timeout = search_timeout
        self._log_dir        = Path(log_dir)
        self._checkpoint     = checkpoint
        self._timer_set: set[str] = set()
        self._proto_log: dict[str, list[str]] = {}   # battle_tag -> raw protocol lines
        self._decisions: dict[str, list[dict]] = {}  # battle_tag -> our choices
        super().__init__(*args, **kwargs)

    async def _handle_battle_message(self, split_messages):
        tag = split_messages[0][0].lstrip(">").strip()
        lines = ["|".join(m) for m in split_messages[1:]]
        self._proto_log.setdefault(tag, []).extend(lines)
        await super()._handle_battle_message(split_messages)
        if self._live is not None:
            try:
                battle = self.battles.get(tag)
                if battle is not None:
                    ev = self._events.setdefault(tag, [])
                    ev.extend(e for e in (format_event(l, battle.player_role) for l in lines) if e)
                    del ev[:-12]
                    self._live.update(tag, snapshot_battle(battle, ev))
            except Exception as e:  # the website must never break play
                print(f"  [live] snapshot failed: {type(e).__name__}: {e}", flush=True)

    def choose_move(self, battle):
        order = super().choose_move(battle)
        self._decisions.setdefault(battle.battle_tag, []).append({
            "turn":   battle.turn,
            "order":  str(order),
            "active": battle.active_pokemon.species if battle.active_pokemon else None,
            "hp":     round(battle.active_pokemon.current_hp_fraction, 2) if battle.active_pokemon else None,
            "opp":    battle.opponent_active_pokemon.species if battle.opponent_active_pokemon else None,
            "opp_hp": round(battle.opponent_active_pokemon.current_hp_fraction, 2) if battle.opponent_active_pokemon else None,
        })
        return order

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
            if self._live is not None:
                self._live.set_searching(True)
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
                    if self._live is not None:
                        self._live.set_searching(False)
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
        print(f"  [{result}]  vs {opp:<20}  {w}W / {p - w}L  ({w / max(p, 1):.0%})", flush=True)
        if self._live is not None:
            self._live.finish(battle.battle_tag, bool(battle.won), opp, battle.rating)
            self._events.pop(battle.battle_tag, None)

        # |raw| rating messages arrive after |win|, so delay the write to let
        # poke-env parse them into battle.rating / battle.opponent_rating first.
        def _write_record():
            rating     = battle.rating
            opp_rating = battle.opponent_rating
            if rating is not None:
                print(f"    rating: {rating}  opp: {opp_rating}", flush=True)
            record = {
                "timestamp":  datetime.now(timezone.utc).isoformat(),
                "format":     self.fmt.name,
                "game":       p,
                "won":        bool(battle.won),
                "opponent":   opp,
                "our_rating": rating,
                "opp_rating": opp_rating,
                "checkpoint": self._checkpoint,
            }
            with open(self._history_file, "a") as f:
                f.write(json.dumps(record) + "\n")

            # Full per-battle log: raw protocol + our decisions, for post-hoc review.
            try:
                self._log_dir.mkdir(parents=True, exist_ok=True)
                stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
                name = f"{stamp}_{'W' if battle.won else 'L'}_{opp}.json".replace("/", "_")
                tag = battle.battle_tag
                with open(self._log_dir / name, "w") as f:
                    json.dump({**record, "battle_tag": tag,
                               "decisions": self._decisions.pop(tag, []),
                               "protocol": self._proto_log.pop(tag, [])}, f)
            except OSError as e:
                print(f"    could not save battle log: {e}", flush=True)

        threading.Timer(3.0, _write_record).start()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Play rated ladder games on Pokémon Showdown")
    p.add_argument("--format",     default="gen1ou", choices=FORMAT_NAMES)
    p.add_argument("--checkpoint", required=True,
                   help="Path to checkpoint (.pt)")
    p.add_argument("--team",       default="zam_egg_zap",
                   help="Team name (ignored for random-battle formats)  (default: zam_egg_zap)")
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
    p.add_argument("--history-len",     type=int, default=None,
                   help="Turns of obs history fed to the model (default: 1 for bc_* checkpoints, else 10)")
    p.add_argument("--publish-live", action="store_true",
                   help="Push live battle state to the repo's live-data branch for the website")
    p.add_argument("--live-interval", type=float, default=15.0,
                   help="Seconds between live-state pushes (default: 15)")
    p.add_argument("--log-dir",         default="ladder_logs",
                   help="Directory for per-battle logs (default: ladder_logs)")
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
    fmt   = get_format(args.format)
    model = load_model(args.checkpoint, fmt, device)

    # BC checkpoints were trained on single turns (K=1); feeding them K=10 windows
    # degrades them badly (offline: accuracy 0.62 -> 0.50, switch rate 22% -> 41%).
    stem = Path(args.checkpoint).stem
    history_len = args.history_len or (1 if stem.startswith("bc_") else 10)
    tag = stem if history_len == 10 else f"{stem}_k{history_len}"

    live = None
    if args.publish_live:
        live = LivePublisher(bot=username, checkpoint=tag, interval=args.live_interval)
        live.start()
        print(f"Live:     publishing to branch '{LIVE_BRANCH}' every {args.live_interval:.0f}s")

    player = LadderPlayer(
        fmt=fmt,
        model=model,
        device=device,
        sample=args.sample,
        username=username,
        password=password,
        team=fmt.get_team(args.team),
        server_configuration=SHOWDOWN_SERVER,
        history_file=args.history_file,
        search_timeout=args.search_timeout,
        log_dir=args.log_dir,
        checkpoint=tag,
        history_len=history_len,
        live=live,
        ping_timeout=60.0,  # default 20s drops the connection on brief stalls (MPS, laptop idle)
    )

    mode = "sample" if args.sample else "greedy"
    print(f"\nAccount:  {username}")
    print(f"Format:   {fmt.name}")
    if fmt.needs_team:
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
    if fmt.needs_team and player.next_team is None:
        print("Error: no team set. Pass --team <name> to specify a team.")
        sys.exit(1)
    print(f"Logged in as {username}. Searching for games...\n")

    # poke-env's listen() swallows websocket errors (e.g. connection reset) and just
    # returns, leaving ladder() waiting forever. Hard-exit so a supervisor can restart.
    class _QuietDisconnect(logging.Filter):
        """Replace poke-env's multi-page websocket traceback with one line, and drop
        the harmless "nothing to choose" rejection (a duplicate/late move request
        for a battle that has already ended or moved on)."""
        def filter(self, record: logging.LogRecord) -> bool:
            if "nothing to choose" in record.getMessage():
                return False
            exc = record.exc_info[1] if record.exc_info else None
            if isinstance(exc, ConnectionClosed):
                print(f"\nConnection lost: {exc}", flush=True)
                return False
            return True

    player.ps_client.logger.addFilter(_QuietDisconnect())

    class _QuietShielded(logging.Filter):
        """asyncio logs the same dropped connection again as 'exception in shielded future'."""
        def filter(self, record: logging.LogRecord) -> bool:
            exc = record.exc_info[1] if record.exc_info else None
            return not (isinstance(exc, ConnectionClosed) or "ConnectionClosed" in record.getMessage())

    logging.getLogger("asyncio").addFilter(_QuietShielded())

    def _on_listen_done(_fut) -> None:
        print("Reconnecting needed — exiting.", flush=True)
        os._exit(2)

    player.ps_client._listening_coroutine.add_done_callback(_on_listen_done)

    try:
        asyncio.run(player.ladder(args.n_games))
    except KeyboardInterrupt:
        pass

    w = player.n_won_battles
    p = player.n_finished_battles
    print(f"\nFinal: {w}W / {p - w}L  ({w / max(p, 1):.0%} over {p} games)")


if __name__ == "__main__":
    main()
