"""
Build a HuggingFace dataset of gen1ou POV trajectories from metamon raw replays.

Each row is one player's perspective on one battle (two rows per source battle).

Schema (per row)
----------------
Flat metadata:
  battle_id, format, gen, pov_player, player_name, opponent_name, won, num_turns

Per-turn parallel sequences (one entry per usable turn):
  turn_numbers        – int
  my_active_species   – str
  my_active_hp        – float  (0.0–1.0)
  my_active_status    – str    ('' if none)
  my_active_boosts    – str    (JSON: {"atk": 0, ...})
  my_team             – str    (JSON: list of pokemon dicts, inference-completed)
  opp_active_species  – str
  opp_active_hp       – float
  opp_active_status   – str
  opp_seen_team       – str    (JSON: revealed opponent pokemon only)
  weather             – str    ('' if none)
  field_conditions    – str    (JSON: list of active conditions)
  my_side_conditions  – str    (JSON: list of active side conditions)
  opp_side_conditions – str
  my_action_kind      – str    ('move' | 'switch')
  my_action_value     – str    (move name or species)
  my_action_forced    – bool
  opp_action_kind     – str    ('' if unknown)
  opp_action_value    – str    ('' if unknown)

Usage
-----
    # save locally (default)
    python scripts/build_gen1ou_dataset.py

    # push to HuggingFace Hub
    python scripts/build_gen1ou_dataset.py --repo your-username/protean-gen1ou

    # push as public, use explicit token
    python scripts/build_gen1ou_dataset.py --repo your-username/protean-gen1ou --public --token hf_...

    # control parallelism
    python scripts/build_gen1ou_dataset.py --workers 8
"""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import Dataset, concatenate_datasets, load_from_disk
from huggingface_hub import hf_hub_download
from tqdm import tqdm

from protean.backend.replay_parser.parser import parse_battle
from protean.backend.replay_parser.types import BattlePokemon, POVReplay
from protean.backend.usage_stats import load_format_stats
from protean.pokedex import _clean as _clean_name
from protean.pov import reconstruct_both_povs

FORMAT = "gen1ou"
SOURCE_REPO = "jakegrigsby/metamon-raw-replays"
GEN1OU_SHARDS = [35, 36]


# ---------------------------------------------------------------------------
# Gen1 species validator
# ---------------------------------------------------------------------------

def _build_gen1_species_set() -> frozenset[str]:
    """Return cleaned names of all gen1 species (dex 1-151) from poke-env."""
    from poke_env.data import GenData
    gd = GenData.from_gen(1)
    return frozenset(
        _clean_name(name)
        for name, data in gd.pokedex.items()
        if 1 <= data.get("num", 0) <= 151
    )


# Built once at import time — cheap since poke-env caches the data.
_GEN1_SPECIES: frozenset[str] = _build_gen1_species_set()


def _battle_is_gen1_clean(battle) -> bool:
    """
    Return True only if every species that appeared in the battle is a gen1
    Pokémon (dex 1-151).  Rejects mislabeled replays that contain gen2+ mons.
    """
    for turn in battle.turns:
        for side in (turn.p1, turn.p2):
            for pk in side.team:
                if _clean_name(pk.species) not in _GEN1_SPECIES:
                    return False
    return True


# ---------------------------------------------------------------------------
# Worker-process state (loaded once per process via initializer)
# ---------------------------------------------------------------------------

_worker_stats: dict | None = None


def _init_worker(stats: dict) -> None:
    global _worker_stats
    _worker_stats = stats


def _process_row(args: tuple) -> list[dict]:
    """Parse one replay and return 0–2 POV row dicts. Runs in a worker process."""
    row_id, log, formatid, seed = args
    rng = np.random.default_rng(seed)
    battle = parse_battle(row_id, log, formatid)
    if battle is None:
        return []
    if not _battle_is_gen1_clean(battle):
        return []
    p1_pov, p2_pov = reconstruct_both_povs(battle, format_stats=_worker_stats, rng=rng)
    result = []
    for pov in (p1_pov, p2_pov):
        if pov is not None:
            result.append(_build_row(pov))
    return result


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

def _pk_to_dict(pk: BattlePokemon) -> dict:
    return {
        "species": pk.species,
        "hp": float(pk.hp),
        "status": pk.status or "",
        "fainted": pk.fainted,
        "boosts": pk.boosts,
        "revealed_moves": pk.revealed_moves,
        "item": pk.item or "",
        "ability": pk.ability or "",
        "seen_in_battle": pk.seen_in_battle,
    }


def _build_row(pov: POVReplay) -> dict:
    turn_numbers: list[int] = []
    my_active_species: list[str] = []
    my_active_hp: list[float] = []
    my_active_status: list[str] = []
    my_active_boosts: list[str] = []
    my_team: list[str] = []
    opp_active_species: list[str] = []
    opp_active_hp: list[float] = []
    opp_active_status: list[str] = []
    opp_seen_team: list[str] = []
    weather: list[str] = []
    field_conditions: list[str] = []
    my_side_conditions: list[str] = []
    opp_side_conditions: list[str] = []
    my_action_kind: list[str] = []
    my_action_value: list[str] = []
    my_action_forced: list[bool] = []
    opp_action_kind: list[str] = []
    opp_action_value: list[str] = []

    for snap in pov.snapshots:
        turn_numbers.append(snap.turn_number)

        active = snap.my_side.active
        my_active_species.append(active.species if active else "")
        my_active_hp.append(float(active.hp) if active else 1.0)
        my_active_status.append(active.status or "" if active else "")
        my_active_boosts.append(json.dumps(active.boosts) if active else "{}")
        my_team.append(json.dumps([_pk_to_dict(pk) for pk in snap.my_side.team]))

        opp_active = snap.opp_side.active
        opp_active_species.append(opp_active.species if opp_active else "")
        opp_active_hp.append(float(opp_active.hp) if opp_active else 1.0)
        opp_active_status.append(opp_active.status or "" if opp_active else "")
        opp_seen_team.append(json.dumps([_pk_to_dict(pk) for pk in snap.opp_side.team]))

        weather.append(snap.field.weather or "")
        field_conditions.append(json.dumps(list(snap.field.conditions.keys())))
        my_side_conditions.append(json.dumps(list(snap.my_side.conditions.keys())))
        opp_side_conditions.append(json.dumps(list(snap.opp_side.conditions.keys())))

        my_action_kind.append(snap.my_action.kind)
        my_action_value.append(snap.my_action.value)
        my_action_forced.append(snap.my_action.forced)

        opp_act = snap.opp_action
        opp_action_kind.append(opp_act.kind if opp_act else "")
        opp_action_value.append(opp_act.value if opp_act else "")

    return {
        "battle_id": pov.battle_id,
        "format": pov.format,
        "gen": pov.gen,
        "pov_player": pov.pov_player,
        "player_name": pov.player_name,
        "opponent_name": pov.opponent_name,
        "won": pov.won,
        "num_turns": len(pov.snapshots),
        "turn_numbers": turn_numbers,
        "my_active_species": my_active_species,
        "my_active_hp": my_active_hp,
        "my_active_status": my_active_status,
        "my_active_boosts": my_active_boosts,
        "my_team": my_team,
        "opp_active_species": opp_active_species,
        "opp_active_hp": opp_active_hp,
        "opp_active_status": opp_active_status,
        "opp_seen_team": opp_seen_team,
        "weather": weather,
        "field_conditions": field_conditions,
        "my_side_conditions": my_side_conditions,
        "opp_side_conditions": opp_side_conditions,
        "my_action_kind": my_action_kind,
        "my_action_value": my_action_value,
        "my_action_forced": my_action_forced,
        "opp_action_kind": opp_action_kind,
        "opp_action_value": opp_action_value,
    }


# ---------------------------------------------------------------------------
# Per-shard processing
# ---------------------------------------------------------------------------

def _process_shard(
    shard_df: pd.DataFrame,
    stats: dict,
    base_seed: int,
    n_workers: int,
    shard_label: str,
) -> tuple[list[dict], int]:
    """Process one shard in parallel. Returns (rows, parse_failed_count)."""
    args = [
        (str(row["id"]), row["log"], row["formatid"], base_seed + i)
        for i, (_, row) in enumerate(shard_df.iterrows())
    ]

    rows: list[dict] = []
    parse_failed = 0

    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_init_worker,
        initargs=(stats,),
    ) as executor:
        for result in tqdm(
            executor.map(_process_row, args, chunksize=50),
            total=len(args),
            desc=f"  {shard_label}",
        ):
            if not result:
                parse_failed += 1
            else:
                rows.extend(result)

    return rows, parse_failed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo", default="",
        help="HuggingFace Hub repo id to push to, e.g. username/protean-gen1ou. "
             "If omitted the dataset is saved locally to ./gen1ou_dataset/.",
    )
    parser.add_argument(
        "--public", action="store_true", default=False,
        help="Push as a public repo (default: private).",
    )
    parser.add_argument(
        "--token", default=None,
        help="HuggingFace write token. Falls back to HF_TOKEN env var, then cached login.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Base RNG seed.")
    parser.add_argument(
        "--workers", type=int, default=os.cpu_count(),
        help="Number of parallel worker processes (default: all CPUs).",
    )
    args = parser.parse_args()

    print(f"Workers: {args.workers}  Seed: {args.seed}\n")

    print("Loading gen1ou usage stats...")
    stats = load_format_stats(FORMAT, rank="1630")
    print(f"  {len(stats)} species loaded\n")

    shard_datasets: list[Dataset] = []
    total_raw = 0
    total_parse_failed = 0
    global_row_offset = 0

    for shard_idx in GEN1OU_SHARDS:
        cache_path = Path(f"gen1ou_dataset_shard{shard_idx}")

        # Resume: load from disk if this shard was already processed
        if cache_path.exists():
            print(f"Shard {shard_idx}: loading cached result from {cache_path}/")
            ds = load_from_disk(str(cache_path))
            shard_datasets.append(ds)
            global_row_offset += len(ds)  # approximate; actual row count varies
            print(f"  {len(ds):,} POV rows loaded\n")
            continue

        filename = f"data/train-{shard_idx:05d}-of-00046.parquet"
        print(f"Shard {shard_idx}: downloading {filename}...")
        path = hf_hub_download(
            repo_id=SOURCE_REPO,
            filename=filename,
            repo_type="dataset",
        )
        df = pd.read_parquet(path)
        shard_df = df[df["formatid"] == FORMAT].reset_index(drop=True)
        n_source = len(shard_df)
        total_raw += n_source
        print(f"  {n_source:,} gen1ou rows")

        t0 = time.time()
        rows, parse_failed = _process_shard(
            shard_df, stats,
            base_seed=args.seed + global_row_offset,
            n_workers=args.workers,
            shard_label=f"shard {shard_idx}",
        )
        elapsed = time.time() - t0
        total_parse_failed += parse_failed
        global_row_offset += n_source

        print(f"  {n_source / elapsed:.0f} battles/sec  "
              f"{parse_failed} failed  {len(rows):,} POV rows")

        ds = Dataset.from_list(rows)
        ds.save_to_disk(str(cache_path))
        print(f"  Saved to {cache_path}/\n")
        shard_datasets.append(ds)

    # Combine shards
    print("Combining shards...")
    final_ds = concatenate_datasets(shard_datasets)

    total_steps = sum(r for r in final_ds["num_turns"])
    avg_turns = total_steps / len(final_ds)
    print(f"\n{'='*50}")
    print(f"Raw gen1ou replays  : {total_raw:,}")
    print(f"Parse failures      : {total_parse_failed:,}")
    print(f"POV rows            : {len(final_ds):,}")
    print(f"Avg turns / POV     : {avg_turns:.1f}")
    print(f"Total timesteps     : {total_steps:,}")
    print(f"{'='*50}\n")

    out = Path("gen1ou_dataset")
    print(f"Saving final dataset to {out}/")
    final_ds.save_to_disk(str(out))
    print("Saved.")

    if args.repo:
        from huggingface_hub import HfApi
        token = args.token or os.environ.get("HF_TOKEN")
        print(f"\nPushing to hub: {args.repo}  (private={not args.public})")
        final_ds.push_to_hub(args.repo, private=not args.public, token=token)
        print("Upload complete.")
        print("Squashing repo history to drop orphaned LFS objects...")
        HfApi().super_squash_history(repo_id=args.repo, repo_type="dataset", token=token)
        print("Squash complete.")
    else:
        print(f"\nTo push later:\n"
              f"  python scripts/build_gen1ou_dataset.py "
              f"--repo alextatarka6/protean-gen1ou --token hf_...")


if __name__ == "__main__":
    main()
