"""
Convert metamon's gen9ou parsed replays into pre-tokenised BC shards.

Source: jakegrigsby/metamon-parsed-replays, gen9ou.tar.gz (~20 GB). The tarball is
streamed straight from HuggingFace (or read from --tar) — no full download needed.
Each member is one battle from one player's POV: lz4-compressed JSON of
{"states": [UniversalState dict, ...], "actions": [int, ...]} (action -1 = not revealed).

Output (--out-dir, default data/gen9ou_bc/): shard_00000.npz, ... each holding one row per turn:
    tokens   uint16 [N, 112]   obs text → gen9ou vocab ids (<unk> = 0)
    numbers  float16 [N, 55]
    action   int8   [N]        metamon action index 0-12, -1 if not revealed
    legal    uint16 [N]        bitmask of metamon's maybe_valid_actions
    rating   int16  [N]        POV player's ladder rating (0 = unrated / tournament)
    won      bool   [N]        POV player won the battle
    holdout  bool   [N]        10% split by CRC32 of the Showdown battle id (both POVs agree)
    battle   int32  [N]        battle index within the shard; turn int16 [N]

Usage:
    python scripts/build_gen9ou_bc_data.py --max-battles 20000          # quick subset
    python scripts/build_gen9ou_bc_data.py --workers 14                 # everything
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import tarfile
import time
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from protean.formats.gen9ou.obs import N_MOVE_SLOTS, NUMBERS_DIM, TEXT_LEN, state_to_obs
from protean.formats.gen9ou.vocab import get_tokenizer

REPO = "jakegrigsby/metamon-parsed-replays"
TERA_START = 9

# {battle_id}_{rating|Unrated}_{pov}_vs_{opp}_{DD-MM-YYYY}_{WIN|LOSS}.json.lz4
_NAME_RE = re.compile(r"^(?P<id>.+?)_(?P<rating>\d+|Unrated)_.+_(?P<result>WIN|LOSS)\.json\.lz4$")

_tokenizer = None


def _legal_bitmask(state: dict) -> int:
    """metamon UniversalAction.maybe_valid_actions as a 13-bit mask."""
    bits = 0
    if not state["forced_switch"]:
        n_moves = min(len(state["player_active_pokemon"]["moves"]), N_MOVE_SLOTS)
        for i in range(n_moves):
            bits |= 1 << i
            if state["can_tera"]:
                bits |= 1 << (TERA_START + i)
    for i in range(min(len(state["available_switches"]), 5)):
        bits |= 1 << (4 + i)
    return bits


def convert_battle(item: tuple[str, bytes]) -> dict | None:
    """Worker: one replay file → per-turn arrays (None if unparseable)."""
    import lz4.frame
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = get_tokenizer()

    name, raw = item
    m = _NAME_RE.match(Path(name).name)
    if m is None:
        return None
    try:
        data = json.loads(lz4.frame.decompress(raw))
        states, actions = data["states"], data["actions"]
        n = min(len(states), len(actions))
        if n == 0:
            return None
        tokens  = np.empty((n, TEXT_LEN), dtype=np.uint16)
        numbers = np.empty((n, NUMBERS_DIM), dtype=np.float16)
        legal   = np.empty(n, dtype=np.uint16)
        for t in range(n):
            obs = state_to_obs(states[t])
            tokens[t]  = np.maximum(_tokenizer.tokenize(str(obs["text"])), 0)
            numbers[t] = obs["numbers"]
            legal[t]   = _legal_bitmask(states[t])
    except Exception as e:  # malformed replay — skip, but say why
        return {"error": f"{name}: {type(e).__name__}: {e}"}

    battle_id = m["id"]
    return {
        "tokens":  tokens,
        "numbers": numbers,
        "action":  np.asarray(actions[:n], dtype=np.int8),
        "legal":   legal,
        "rating":  0 if m["rating"] == "Unrated" else min(int(m["rating"]), 32767),
        "won":     m["result"] == "WIN",
        "holdout": zlib.crc32(battle_id.encode()) % 100 < 10,
    }


def iter_members(tar_path: str | None):
    """Yield (name, bytes) for every replay file, streaming from HF unless --tar is given."""
    if tar_path:
        fileobj, resp = open(tar_path, "rb"), None
    else:
        import requests
        from huggingface_hub import hf_hub_url
        from huggingface_hub.utils import build_hf_headers
        url = hf_hub_url(REPO, "gen9ou.tar.gz", repo_type="dataset")
        resp = requests.get(url, stream=True, timeout=120, headers=build_hf_headers())
        resp.raise_for_status()
        resp.raw.decode_content = True
        fileobj = resp.raw
    try:
        with tarfile.open(fileobj=fileobj, mode="r|gz") as tf:
            for member in tf:
                if member.isfile() and member.name.endswith(".json.lz4"):
                    yield member.name, tf.extractfile(member).read()
    finally:
        fileobj.close()
        if resp is not None:
            resp.close()


class ShardWriter:
    def __init__(self, out_dir: Path, turns_per_shard: int, start_index: int):
        self.out_dir, self.turns_per_shard = out_dir, turns_per_shard
        self.index = start_index
        self.buf: list[dict] = []
        self.n_turns = 0

    def add(self, battle: dict) -> None:
        self.buf.append(battle)
        self.n_turns += len(battle["action"])
        if self.n_turns >= self.turns_per_shard:
            self.flush()

    def flush(self) -> None:
        if not self.buf:
            return
        lens = [len(b["action"]) for b in self.buf]
        per_turn = lambda key, dtype: np.repeat(np.array([b[key] for b in self.buf], dtype=dtype), lens)
        np.savez(
            self.out_dir / f"shard_{self.index:05d}.npz",
            tokens=np.concatenate([b["tokens"] for b in self.buf]),
            numbers=np.concatenate([b["numbers"] for b in self.buf]),
            action=np.concatenate([b["action"] for b in self.buf]),
            legal=np.concatenate([b["legal"] for b in self.buf]),
            rating=per_turn("rating", np.int16),
            won=per_turn("won", bool),
            holdout=per_turn("holdout", bool),
            battle=np.repeat(np.arange(len(self.buf), dtype=np.int32), lens),
            turn=np.concatenate([np.arange(n, dtype=np.int16) for n in lens]),
        )
        print(f"  wrote shard_{self.index:05d}.npz  ({len(self.buf):,} battles, {self.n_turns:,} turns)", flush=True)
        self.index += 1
        self.buf, self.n_turns = [], 0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir",         default="data/gen9ou_bc")
    p.add_argument("--tar",             default=None, help="local gen9ou.tar.gz (default: stream from HF)")
    p.add_argument("--max-battles",     type=int, default=None)
    p.add_argument("--workers",         type=int, default=8)
    p.add_argument("--turns-per-shard", type=int, default=500_000)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if any(out_dir.glob("shard_*.npz")):
        raise SystemExit(f"{out_dir} already has shards — pick an empty --out-dir")
    writer = ShardWriter(out_dir, args.turns_per_shard, start_index=0)

    n_battles = n_errors = 0
    t0 = time.time()
    chunk: list[tuple[str, bytes]] = []
    with ProcessPoolExecutor(args.workers) as pool:
        def drain(items):
            nonlocal n_battles, n_errors
            for res in pool.map(convert_battle, items, chunksize=64):
                if res is None or "error" in res:
                    n_errors += 1
                    if res is not None and n_errors <= 5:
                        print(f"  skip {res['error']}", flush=True)
                    continue
                writer.add(res)
                n_battles += 1
            rate = n_battles / (time.time() - t0)
            print(f"{n_battles:,} battles ({n_errors} skipped) | {rate:.0f} battles/s", flush=True)

        for item in iter_members(args.tar):
            chunk.append(item)
            if len(chunk) >= 4096:
                drain(chunk)
                chunk = []
            if args.max_battles and n_battles + len(chunk) >= args.max_battles:
                break
        if chunk:
            drain(chunk)
    writer.flush()
    (out_dir / "meta.json").write_text(json.dumps({
        "format": "gen9ou", "source": REPO, "battles": n_battles, "skipped": n_errors,
        "text_len": TEXT_LEN, "numbers_dim": NUMBERS_DIM, "vocab_size": get_tokenizer().vocab_size,
    }, indent=2))
    print(f"Done: {n_battles:,} battles in {time.time() - t0:.0f}s → {out_dir}")


if __name__ == "__main__":
    main()
