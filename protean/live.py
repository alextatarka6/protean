"""
Publish live battle state for the website.

The site (docs/index.html) is static, so the bot pushes a small JSON snapshot to a
dedicated `live-data` branch of the GitHub repo and the page polls it from
raw.githubusercontent.com. That keeps `main` free of churn and needs no server.

A standalone git repo in .live_repo/ (gitignored) holds a single commit that is
amended and force-pushed each time, so the branch never grows history.

    pub = LivePublisher(bot="proteanbot", checkpoint="bc_final")
    pub.start()
    pub.update(tag, snapshot_dict)      # called as the battle progresses
    pub.finish(tag, won, opponent)      # battle over
    pub.set_searching(True)             # queued for a game
"""
from __future__ import annotations

import json
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BRANCH = "live-data"
KEEP_RECENT = 5


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class LivePublisher:
    def __init__(self, bot: str = "", checkpoint: str = "", interval: float = 15.0,
                 heartbeat: float = 120.0, workdir: Path | None = None) -> None:
        self.bot, self.checkpoint = bot, checkpoint
        self.interval, self.heartbeat = interval, heartbeat
        self.dir = Path(workdir) if workdir else ROOT / ".live_repo"
        self._lock = threading.Lock()
        self._battles: dict[str, dict] = {}
        self._recent: list[dict] = []
        self._searching = False
        self._dirty = True
        self._last_push = 0.0
        self._last_error = ""
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="live-publisher", daemon=True)

    # ---- called from the battle loop (any thread) -------------------------
    def start(self) -> None:
        self._thread.start()

    def update(self, tag: str, snapshot: dict) -> None:
        with self._lock:
            self._battles[tag] = {**snapshot, "tag": tag, "status": "live"}
            self._searching = False
            self._dirty = True

    def finish(self, tag: str, won: bool, opponent: str, rating: int | None = None) -> None:
        with self._lock:
            self._battles.pop(tag, None)
            self._recent.insert(0, {"tag": tag, "won": bool(won), "opponent": opponent,
                                    "rating": rating, "ended_at": _now()})
            del self._recent[KEEP_RECENT:]
            self._dirty = True

    def set_searching(self, searching: bool) -> None:
        with self._lock:
            self._searching = searching
            self._dirty = True

    # ---- publisher thread ---------------------------------------------------
    def _document(self) -> dict:
        with self._lock:
            return {"updated_at": _now(), "bot": self.bot, "checkpoint": self.checkpoint,
                    "searching": self._searching and not self._battles,
                    "battles": list(self._battles.values()), "recent": list(self._recent)}

    def _git(self, *args: str) -> str:
        r = subprocess.run(["git", *args], cwd=self.dir, check=True, capture_output=True,
                           text=True, timeout=60)
        return r.stdout.strip()

    def _ensure_repo(self) -> None:
        if (self.dir / ".git").exists():
            return
        self.dir.mkdir(parents=True, exist_ok=True)
        url = subprocess.run(["git", "-C", str(ROOT), "remote", "get-url", "origin"],
                             check=True, capture_output=True, text=True).stdout.strip()
        self._git("init", "-q")
        self._git("remote", "add", "origin", url)

    def _identity(self) -> list[str]:
        def cfg(key: str, default: str) -> str:
            r = subprocess.run(["git", "-C", str(ROOT), "config", key],
                               capture_output=True, text=True)
            return r.stdout.strip() or default
        return ["-c", f"user.name={cfg('user.name', 'protean-live')}",
                "-c", f"user.email={cfg('user.email', 'protean-live@users.noreply.github.com')}"]

    def _publish(self) -> None:
        self._ensure_repo()
        tmp = self.dir / "live.json.tmp"
        tmp.write_text(json.dumps(self._document(), separators=(",", ":")))
        tmp.replace(self.dir / "live.json")
        self._git("add", "live.json")
        has_commit = subprocess.run(["git", "rev-parse", "-q", "--verify", "HEAD"], cwd=self.dir,
                                    capture_output=True).returncode == 0
        commit = ["commit", "-q", "-m", "live state", "--no-gpg-sign"]
        self._git(*self._identity(), *(commit + (["--amend", "--allow-empty"] if has_commit else [])))
        self._git("push", "-q", "-f", "origin", f"HEAD:refs/heads/{BRANCH}")

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            with self._lock:
                due = self._dirty or (time.time() - self._last_push) > self.heartbeat
                self._dirty = False
            if not due:
                continue
            try:
                self._publish()
                self._last_push, self._last_error = time.time(), ""
            except (subprocess.SubprocessError, OSError) as e:
                msg = (getattr(e, "stderr", "") or str(e)).strip().splitlines()
                err = f"{type(e).__name__}: {msg[-1] if msg else ''}"
                if err != self._last_error:  # print each distinct failure once
                    print(f"  [live] publish failed — {err}", flush=True)
                    self._last_error = err
                with self._lock:
                    self._dirty = True


# ---------------------------------------------------------------------------
# Snapshot helpers (poke-env Battle -> JSON-able dict)
# ---------------------------------------------------------------------------

_STATUS_WORDS = {"par": "paralyzed", "brn": "burned", "slp": "put to sleep",
                 "frz": "frozen", "psn": "poisoned", "tox": "badly poisoned"}


def _mon_name(ident: str) -> str:
    """'p1a: Alakazam' -> 'Alakazam'"""
    return ident.split(": ", 1)[-1]


def format_event(line: str, role: str | None) -> dict | None:
    """One protocol line -> {"side": "us"|"opp", "text": ...}, or None if not interesting."""
    p = line.split("|")
    if len(p) < 3:
        return None
    kind, ident = p[1], p[2]
    if not ident[:2] in ("p1", "p2") or role is None:
        return None
    side = "us" if ident[:2] == role else "opp"
    name = _mon_name(ident)
    if kind == "move" and len(p) > 3:
        return {"side": side, "text": f"{name} used {p[3]}"}
    if kind in ("switch", "drag"):
        return {"side": side, "text": f"Sent out {name}"}
    if kind == "faint":
        return {"side": side, "text": f"{name} fainted"}
    if kind == "-status" and len(p) > 3:
        return {"side": side, "text": f"{name} was {_STATUS_WORDS.get(p[3], p[3])}"}
    if kind == "cant":
        return {"side": side, "text": f"{name} can't move"}
    return None


def _mon(p) -> dict:
    return {"s": p.species, "hp": round(float(p.current_hp_fraction), 2),
            "st": p.status.name.lower() if p.status else None,
            "fainted": bool(p.fainted), "active": bool(p.active)}


def snapshot_battle(battle, events: list[dict]) -> dict:
    """Current public state of a battle. Opponent team is padded to 6 with unknowns."""
    ours = [_mon(p) for p in battle.team.values()]
    theirs = [_mon(p) for p in battle.opponent_team.values()]
    theirs += [None] * max(0, 6 - len(theirs))
    return {"turn": battle.turn, "opponent": battle.opponent_username or "?",
            "rating": battle.rating, "opp_rating": battle.opponent_rating,
            "us": ours, "opp": theirs, "events": events[-6:]}
