"""
Run scripts/ladder.py forever, restarting it whenever it exits (search timeout,
crash, disconnect, finished n-games). Only stops on Ctrl+C / SIGTERM.

Each run is a fresh subprocess, so a wedged poke-env connection can't poison the
next one. Any extra flags are passed straight through to ladder.py.

Usage:
    python scripts/ladder_forever.py --checkpoint checkpoints/ppo_ep0000500.pt
    python scripts/ladder_forever.py --checkpoint checkpoints/ppo_ep0000500.pt \\
        --n-games 20 --search-timeout 600

Defaults added if you don't pass them: --n-games 20, --search-timeout 600.
"""
from __future__ import annotations

import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from protean import cli_errors  # noqa: E402

cli_errors.install()

LADDER = Path(__file__).resolve().parent / "ladder.py"
BUILDER = LADDER.with_name("build_ladder_dataset.py")
ROOT = LADDER.parents[1]

MIN_HEALTHY_RUN = 120   # a run shorter than this (s) counts as a quick failure
BASE_DELAY = 5          # seconds between normal restarts
MAX_DELAY = 300         # cap for exponential backoff on repeated quick failures


def log(msg: str) -> None:
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] [supervisor] {msg}", flush=True)


def build_cmd(extra: list[str]) -> list[str]:
    cmd = [sys.executable, str(LADDER), *extra]
    if "--n-games" not in extra:
        cmd += ["--n-games", "20"]
    if "--search-timeout" not in extra:
        cmd += ["--search-timeout", "600"]
    return cmd


def update_dataset() -> None:
    """Fold newly saved battle logs into data/ladder_dataset.jsonl (idempotent, non-fatal)."""
    try:
        subprocess.run([sys.executable, str(BUILDER)], cwd=ROOT, check=True, timeout=600)
    except (subprocess.SubprocessError, OSError) as e:
        log(f"dataset update failed ({type(e).__name__}); will retry after next run")


def main() -> None:
    cmd = build_cmd(sys.argv[1:])
    stopping = False
    proc: subprocess.Popen | None = None

    def stop(signum, frame):
        nonlocal stopping
        stopping = True
        log("Stop requested — shutting down current run...")
        if proc is not None and proc.poll() is None:
            proc.send_signal(signal.SIGINT)  # lets ladder.py print its final tally

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    run = 0
    quick_failures = 0
    while not stopping:
        run += 1
        log(f"Run #{run}: {' '.join(cmd)}")
        started = time.time()
        proc = subprocess.Popen(cmd, cwd=ROOT)
        while True:
            try:
                code = proc.wait()
                break
            except KeyboardInterrupt:
                pass  # handled by stop(); keep waiting for child to exit
        proc = None
        elapsed = time.time() - started
        update_dataset()
        if stopping:
            break

        if elapsed < MIN_HEALTHY_RUN:
            quick_failures += 1
            delay = min(BASE_DELAY * 2 ** quick_failures, MAX_DELAY)
        else:
            quick_failures = 0
            delay = BASE_DELAY
        log(f"ladder.py exited (code {code}) after {elapsed:.0f}s. Restarting in {delay}s...")

        end = time.time() + delay
        while not stopping and time.time() < end:
            time.sleep(0.5)

    log("Stopped.")


if __name__ == "__main__":
    main()
