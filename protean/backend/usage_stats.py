"""
Access layer for the jakegrigsby/metamon-usage-stats HuggingFace dataset.

The dataset stores one tar.gz archive per generation:
    movesets_data/gen{N}.tar.gz

Inside each archive the path structure is:
    gen{N}/{tier}/{rank}/{YYYY-MM}.json

Each JSON maps species name → {count, moves, items, abilities, teammates, ...}
where nested dicts map names to float usage percentages.

Archives are downloaded once via huggingface_hub and extracted to a sibling
directory (e.g. .../gen4_extracted/). Subsequent calls read from disk.
"""
from __future__ import annotations

import json
import os
import re
import tarfile
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

HF_REPO = "jakegrigsby/metamon-usage-stats"
DEFAULT_RANK = "1630"


@dataclass
class MovesetStats:
    count: float
    moves: dict[str, float]
    items: dict[str, float]
    abilities: dict[str, float]
    teammates: dict[str, float]

    def top_moves(self, n: int = 4) -> list[tuple[str, float]]:
        return sorted(self.moves.items(), key=lambda x: x[1], reverse=True)[:n]

    def top_items(self, n: int = 3) -> list[tuple[str, float]]:
        return sorted(self.items.items(), key=lambda x: x[1], reverse=True)[:n]

    def top_abilities(self, n: int = 2) -> list[tuple[str, float]]:
        return sorted(self.abilities.items(), key=lambda x: x[1], reverse=True)[:n]


def _parse_format(format_id: str) -> tuple[int, str]:
    """'gen4nu' → (4, 'nu')"""
    m = re.match(r"gen(\d+)(.+)", format_id.lower())
    if not m:
        raise ValueError(f"Cannot parse format: {format_id!r}")
    return int(m.group(1)), m.group(2)


@lru_cache(maxsize=10)
def _get_gen_dir(gen: int) -> str:
    """
    Download movesets_data/gen{N}.tar.gz and extract it.
    Returns the path to the extracted gen{N}/ directory.
    Extraction is skipped if the directory already exists.
    """
    from huggingface_hub import hf_hub_download

    archive = hf_hub_download(
        repo_id=HF_REPO,
        filename=f"movesets_data/gen{gen}.tar.gz",
        repo_type="dataset",
    )
    extract_root = archive.replace(".tar.gz", "_extracted")
    gen_dir = os.path.join(extract_root, f"gen{gen}")

    if not os.path.isdir(gen_dir):
        os.makedirs(extract_root, exist_ok=True)
        with tarfile.open(archive, "r:gz") as tf:
            tf.extractall(extract_root)

    return gen_dir


@lru_cache(maxsize=512)
def _read_json(path: str) -> dict:
    """Read and cache a single JSON file. Returns {} if missing or unreadable."""
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


def list_dates(gen: int, tier: str, rank: str = DEFAULT_RANK) -> list[str]:
    """
    Return sorted list of available YYYY-MM date strings for this gen/tier/rank.
    """
    gen_dir = _get_gen_dir(gen)
    rank_dir = os.path.join(gen_dir, tier, rank)
    if not os.path.isdir(rank_dir):
        return []
    dates = [
        f[:-5]  # strip .json
        for f in os.listdir(rank_dir)
        if f.endswith(".json") and re.match(r"^\d{4}-\d{2}$", f[:-5])
    ]
    return sorted(dates)


def load_moveset_stats(
    species: str,
    format_id: str,
    dates: Optional[list[str]] = None,
    rank: str = DEFAULT_RANK,
) -> Optional[MovesetStats]:
    """
    Load and merge moveset usage stats for a single species across monthly snapshots.

    Args:
        species:   Pokémon species name as it appears in the dataset, e.g. "Garchomp"
        format_id: format string, e.g. "gen4nu"
        dates:     list of "YYYY-MM" strings; if None, all available dates are used
        rank:      Smogon ELO rank cutoff folder (default "1630")
    """
    gen, tier = _parse_format(format_id)
    gen_dir = _get_gen_dir(gen)
    rank_dir = os.path.join(gen_dir, tier, rank)

    if dates is None:
        dates = list_dates(gen, tier, rank)

    buckets: list[dict] = []
    for date in dates:
        data = _read_json(os.path.join(rank_dir, f"{date}.json"))
        key = species.lower()
        if key in data:
            buckets.append(data[key])

    if not buckets:
        return None

    return _merge_buckets(buckets)


def load_format_stats(
    format_id: str,
    dates: Optional[list[str]] = None,
    rank: str = DEFAULT_RANK,
) -> dict[str, MovesetStats]:
    """
    Load stats for all species in a format, merged across the given dates.
    Efficient for bulk lookups — reads each monthly file once.

    Args:
        format_id: format string, e.g. "gen4nu"
        dates:     list of "YYYY-MM" strings; if None, all available dates are used
        rank:      Smogon ELO rank cutoff folder (default "1630")
    """
    gen, tier = _parse_format(format_id)
    gen_dir = _get_gen_dir(gen)
    rank_dir = os.path.join(gen_dir, tier, rank)

    if dates is None:
        dates = list_dates(gen, tier, rank)

    per_species: dict[str, list[dict]] = {}
    for date in dates:
        data = _read_json(os.path.join(rank_dir, f"{date}.json"))
        for species, stats in data.items():
            per_species.setdefault(species, []).append(stats)

    return {species: _merge_buckets(buckets) for species, buckets in per_species.items()}


def _merge_buckets(buckets: list[dict]) -> MovesetStats:
    """Merge multiple monthly stat dicts into one MovesetStats via weighted average."""
    total_count = sum(float(b.get("count", 1.0)) for b in buckets)
    if total_count == 0.0:
        total_count = 1.0

    def _merge(key: str) -> dict[str, float]:
        out: dict[str, float] = {}
        for b in buckets:
            w = float(b.get("count", 1.0)) / total_count
            for name, pct in b.get(key, {}).items():
                out[name] = out.get(name, 0.0) + float(pct) * w
        return out

    return MovesetStats(
        count=total_count,
        moves=_merge("moves"),
        items=_merge("items"),
        abilities=_merge("abilities"),
        teammates=_merge("teammates"),
    )
