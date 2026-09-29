"""
Gen9 OU vocabulary.

Built from every species / move / item / ability in the local Showdown gen9 dex
(protean/data/gen9_dex.json) plus poke-env's status / effect / side-condition /
field / weather enums, all passed through metamon's name cleaning so tokens match
both the parsed-replay dataset and universal.state_from_battle().

Token 0 is <unk>: Tokenizer maps unseen words to -1 and the model clamps ids to
>= 0, so anything missing lands on <unk>.

Rebuild (after scripts/build_gen9_dex.py):
    python -m protean.formats.gen9ou.vocab
"""
from __future__ import annotations

import json
from pathlib import Path

from poke_env.environment import Effect, Field, SideCondition, Status, Weather

from protean.formats.gen9ou.universal import clean_name, clean_no_numbers, move_name, pokemon_name
from protean.tokenizer import Tokenizer

DATA_DIR   = Path(__file__).resolve().parents[2] / "data"
DEX_PATH   = DATA_DIR / "gen9_dex.json"
VOCAB_PATH = DATA_DIR / "gen9ou_vocab.json"

SPECIAL_TOKENS = [
    "<unk>",   # must stay at index 0
    "<gen9ou>", "<anychoice>", "<forcedswitch>",
    "<player>", "<opponent>", "<move>", "<switch>", "<moveset>",
    "<opp_moves>", "<opp_team>", "<conditions>", "<player_prev>", "<opp_prev>",
    "<blank>",
]

PLACEHOLDER_TOKENS = [
    "nostatus", "noeffect", "noconditions", "nofield", "noweather",
    "notype", "nomove", "unknownitem", "noitem", "unknownability", "noability",
    "physical", "special", "status",
]


def build_tokenizer() -> Tokenizer:
    dex = json.loads(DEX_PATH.read_text())
    tok = Tokenizer()
    for t in SPECIAL_TOKENS + PLACEHOLDER_TOKENS:
        tok._add(t)
    for enum in (Status, Effect, SideCondition, Field, Weather):
        for e in enum:
            tok._add(clean_no_numbers(e.name))
    for t in dex["types"]:
        tok._add(clean_name(t))
    for s in dex["species"]:
        tok._add(pokemon_name(s))
    for m in dex["moves"]:
        tok._add(move_name(m))
    for name in dex["items"] + dex["abilities"]:
        tok._add(clean_no_numbers(name))
    return tok


def get_tokenizer() -> Tokenizer:
    if not VOCAB_PATH.exists():
        tok = build_tokenizer()
        tok.save(VOCAB_PATH)
        return tok
    return Tokenizer.load(VOCAB_PATH)


if __name__ == "__main__":
    tok = build_tokenizer()
    tok.save(VOCAB_PATH)
    print(f"Wrote {VOCAB_PATH} ({tok.vocab_size} tokens)")
