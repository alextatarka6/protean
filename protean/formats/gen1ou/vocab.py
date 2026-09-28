"""
Gen1OU vocabulary: gen1 Pokémon species, moves, types, status conditions,
weather, side conditions, and special structural tokens. Much smaller than
metamon's full tokenizer (no items, abilities, tera, gen5-9 content).

Usage:
    from protean.formats.gen1ou.vocab import build_gen1ou_tokenizer, get_tokenizer

    tok = build_gen1ou_tokenizer()       # build from poke-env data
    tok.save(DEFAULT_VOCAB_PATH)

    tok = get_tokenizer()                # load the pre-built vocab
"""
from __future__ import annotations

from pathlib import Path

from protean.tokenizer import Tokenizer, UNKNOWN_TOKEN, _clean


# ---------------------------------------------------------------------------
# Vocab builder
# ---------------------------------------------------------------------------

# Special structural tokens (same angle-bracket convention as metamon)
_SPECIAL_TOKENS = [
    "<gen1ou>",
    "<player>", "<opponent>",
    "<move>", "<switch>", "<moveset>",
    "<conditions>",
    "<player_prev>", "<opp_prev>",
    "<anychoice>", "<forcedswitch>",
    "<blank>",                  # padding for missing moves / bench slots
]

# Gen1 status conditions (stored as short strings by our parser, empty = healthy)
_STATUS_TOKENS = ["nostatus", "par", "brn", "slp", "frz", "psn", "tox", "fnt"]

# Gen1 has no weather, but we include a token so the field is always filled
_WEATHER_TOKENS = ["noweather"]

# Volatile / in-battle effects visible on the active Pokémon
_EFFECT_TOKENS = [
    "noeffect", "confusion", "leechseed", "substitute",
    "partiallytrapped", "lightscreen", "reflect",
]

# Special action values that aren't move or species names
_ACTION_TOKENS = ["recharge"]   # forced recharge turn after Hyper Beam

# Side conditions (Reflect/Light Screen are permanent in gen1)
_CONDITION_TOKENS = ["noconditions", "reflect", "lightscreen"]

# Gen1 types (no Dark, Steel, Fairy)
_TYPE_TOKENS = [
    "normal", "fire", "water", "electric", "grass", "ice",
    "fighting", "poison", "ground", "flying", "psychic",
    "bug", "rock", "ghost", "dragon", "notype", "???",
]

# Move categories
_CATEGORY_TOKENS = ["physical", "special", "status", "nomove"]



def build_gen1ou_tokenizer(scan_dataset: bool = True) -> Tokenizer:
    """
    Build the gen1ou vocabulary.

    Token ordering:
      1. Special structural tokens
      2. Status / weather / effect / condition / type / category tokens
      3. All gen1 species names (dex 1-151, including formes)
      4. All gen1 move names
      5. (optional) Any additional tokens found by scanning the HF dataset —
         catches gen2+ species that appear from team inference.

    Args:
        scan_dataset: If True (default), stream the atatark2/protean-gen1ou
                      dataset and add any unseen tokens. Set False to skip
                      the network call (e.g. in offline environments).
    """
    from poke_env.data import GenData
    gd = GenData.from_gen(1)

    tok = Tokenizer()

    # 1. Special tokens
    for t in _SPECIAL_TOKENS:
        tok._add(t)

    # 2. Fixed categorical tokens
    for group in (_STATUS_TOKENS, _WEATHER_TOKENS, _EFFECT_TOKENS,
                  _CONDITION_TOKENS, _TYPE_TOKENS, _CATEGORY_TOKENS, _ACTION_TOKENS):
        for t in group:
            tok._add(t)

    # 3. Gen1 species (dex 1-151)
    for name, data in sorted(gd.pokedex.items()):
        if 1 <= data.get("num", 0) <= 151:
            tok._add(_clean(name))

    # 4. Gen1 moves (all 167 entries in poke-env's gen1 data)
    for name in sorted(gd.moves.keys()):
        tok._add(_clean(name))

    # 5. Scan actual dataset to catch any tokens not in the gen1 dex
    #    (e.g. gen2+ species introduced by team inference fallback)
    if scan_dataset:
        _extend_from_dataset(tok)

    return tok


def _extend_from_dataset(tok: Tokenizer) -> None:
    """Scan the atatark2/protean-gen1ou HF dataset and add any unseen tokens."""
    import json as _json
    from datasets import load_dataset

    print("Scanning dataset to extend tokenizer vocab...")
    ds = load_dataset("atatark2/protean-gen1ou", split="train")

    new_count = 0
    for row in ds:
        for t in range(row["num_turns"]):
            # Species tokens
            for species_field in ("my_active_species", "opp_active_species"):
                w = _clean(row[species_field][t])
                if w and tok[w] == UNKNOWN_TOKEN:
                    tok._add(w); new_count += 1

            # Move tokens from team JSON blobs
            for team_field in ("my_team", "opp_seen_team"):
                team = _json.loads(row[team_field][t])
                for pk in team:
                    w = _clean(pk["species"])
                    if w and tok[w] == UNKNOWN_TOKEN:
                        tok._add(w); new_count += 1
                    for move in pk.get("revealed_moves", []):
                        w = _clean(move)
                        if w and tok[w] == UNKNOWN_TOKEN:
                            tok._add(w); new_count += 1

            # Action value tokens (move names / species)
            for val_field in ("my_action_value", "opp_action_value"):
                w = _clean(row[val_field][t])
                if w and tok[w] == UNKNOWN_TOKEN:
                    tok._add(w); new_count += 1

    print(f"  Added {new_count} new tokens from dataset scan (vocab now {len(tok)})")


# ---------------------------------------------------------------------------
# Default pre-built vocab path
# ---------------------------------------------------------------------------

DEFAULT_VOCAB_PATH = Path(__file__).resolve().parents[2] / "data" / "gen1ou_vocab.json"


def get_tokenizer() -> Tokenizer:
    """
    Load the pre-built gen1ou tokenizer from disk.
    If the vocab file doesn't exist yet, build it and save it first.
    """
    if not DEFAULT_VOCAB_PATH.exists():
        tok = build_gen1ou_tokenizer()
        tok.save(DEFAULT_VOCAB_PATH)
        return tok
    return Tokenizer.load(DEFAULT_VOCAB_PATH)
