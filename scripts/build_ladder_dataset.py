"""
Turn saved ladder battle logs (ladder_logs/*.json, written by ladder.py) into
parsed-replay rows in the same schema as the HF gen1ou dataset, for fine-tuning.

Each battle yields up to two rows (both POVs), run through the same parser and POV
reconstruction as scripts/build_gen1ou_dataset.py. Extra columns:
    source      "ladder"
    is_bot      True for the bot's POV, False for the opponent's (a human's) POV
    own_rating  ladder rating of the POV player at game end (may be None)
    opp_rating  rating of the other player
    checkpoint  which checkpoint the bot was playing with

Output is JSONL, one row per line, so it can be appended to and deduplicated:
    data/ladder_dataset.jsonl
Load with  datasets.load_dataset("json", data_files="data/ladder_dataset.jsonl").

Idempotent: battles already in the output file are skipped, so run it as often as you like.

Usage:
    python scripts/build_ladder_dataset.py
    python scripts/build_ladder_dataset.py --logs ladder_logs --out data/ladder_dataset.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_gen1ou_dataset import _battle_is_gen1_clean, _build_row  # noqa: E402
from protean.backend.replay_parser.parser import parse_battle  # noqa: E402
from protean.backend.usage_stats import load_format_stats  # noqa: E402
from protean.pov import reconstruct_both_povs  # noqa: E402

FORMAT = "gen1ou"


def _norm(name: str) -> str:
    return "".join(c for c in name.lower() if c.isalnum())


def _opponent_side(d: dict) -> str | None:
    """'p1'/'p2' for the human, from the first |player| lines (later ones blank out a
    side when that player leaves, which would leave the parser with an empty name)."""
    for line in d["protocol"]:
        parts = line.split("|")
        if len(parts) > 3 and parts[1] == "player" and _norm(parts[3]) == _norm(d["opponent"]):
            return parts[2]
    return None


def _existing_keys(out: Path) -> set[tuple[str, str]]:
    keys: set[tuple[str, str]] = set()
    if out.exists():
        with open(out) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    keys.add((r["battle_id"], r["pov_player"]))
                except (json.JSONDecodeError, KeyError):
                    continue
    return keys


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--logs", default=str(ROOT / "ladder_logs"))
    ap.add_argument("--out", default=str(ROOT / "data" / "ladder_dataset.jsonl"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    logs = sorted(Path(args.logs).glob("*.json"))
    seen = _existing_keys(out)
    seen_battles = {b for b, _ in seen}

    todo = []
    for p in logs:
        try:
            d = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError) as e:
            print(f"skip {p.name}: unreadable ({e})")
            continue
        if d.get("battle_tag") not in seen_battles:
            todo.append((p, d))
    print(f"{len(logs)} logs, {len(seen_battles)} already built, {len(todo)} new")
    if not todo:
        return

    stats = load_format_stats(FORMAT, rank="1630")
    added = skipped = 0
    with open(out, "a") as f:
        for p, d in todo:
            tag = d["battle_tag"]
            battle = parse_battle(tag, "\n".join(d["protocol"]), FORMAT)
            if battle is None or not _battle_is_gen1_clean(battle):
                print(f"skip {p.name}: unparseable or not clean gen1")
                skipped += 1
                continue
            opp_side = _opponent_side(d)
            if opp_side is None:
                print(f"skip {p.name}: can't tell which side is the opponent")
                skipped += 1
                continue
            rng = np.random.default_rng(args.seed + hash(tag) % 100_000)
            for pov in reconstruct_both_povs(battle, format_stats=stats, rng=rng):
                if pov is None:
                    continue
                row = _build_row(pov)
                is_bot = pov.pov_player != opp_side
                row.update(
                    source="ladder",
                    is_bot=is_bot,
                    own_rating=d.get("our_rating") if is_bot else d.get("opp_rating"),
                    opp_rating=d.get("opp_rating") if is_bot else d.get("our_rating"),
                    checkpoint=d.get("checkpoint", ""),
                )
                f.write(json.dumps(row) + "\n")
                added += 1
    print(f"added {added} rows from {len(todo) - skipped} battles ({skipped} skipped) -> {out}")


if __name__ == "__main__":
    main()
