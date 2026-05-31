"""
Stream raw replays from the HuggingFace metamon dataset and yield parsed battles.

Usage:
    from protean.backend.replay_parser import iter_parsed_battles

    for battle in iter_parsed_battles(formats=["gen1ou", "gen2ou"]):
        print(battle.battle_id, len(battle.turns), battle.winner)
"""
from __future__ import annotations

from typing import Iterator, Optional

from .parser import parse_battle
from .types import ParsedBattle

HF_DATASET = "jakegrigsby/metamon-raw-replays"

# Gen 1–4 format prefixes — anything else is skipped by default
_GEN14_PREFIXES = ("gen1", "gen2", "gen3", "gen4")


def iter_raw_replays(
    split: str = "train",
    formats: Optional[list[str]] = None,
) -> Iterator[dict]:
    """
    Stream raw replay dicts from the HuggingFace dataset.
    Each dict has keys: id, formatid, format, players, uploadtime, log.

    Args:
        split: dataset split (typically just "train" for this dataset)
        formats: explicit list of formatid strings to keep, e.g. ["gen1ou"].
                 If None, keeps all Gen 1–4 formats.
    """
    from datasets import load_dataset  # deferred so import errors surface clearly

    ds = load_dataset(HF_DATASET, split=split, streaming=True)
    for entry in ds:
        fmt = entry.get("formatid", "")
        if formats is not None:
            if fmt not in formats:
                continue
        else:
            if not any(fmt.startswith(p) for p in _GEN14_PREFIXES):
                continue
        yield entry


def iter_parsed_battles(
    split: str = "train",
    formats: Optional[list[str]] = None,
) -> Iterator[ParsedBattle]:
    """
    Stream parsed ParsedBattle objects from the HuggingFace dataset.
    Incomplete battles and unsupported formats are silently dropped.

    Args:
        split: dataset split
        formats: explicit list of formatid strings to keep.
                 If None, keeps all Gen 1–4 formats.
    """
    for entry in iter_raw_replays(split=split, formats=formats):
        result = parse_battle(
            battle_id=entry.get("id", ""),
            log=entry.get("log", ""),
            format_id=entry.get("formatid", ""),
        )
        if result is not None:
            yield result
