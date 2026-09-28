"""
Gen9 random-battle vocabulary: structural tokens, statuses, types, weather,
terrain, and every species / move / item / ability in the exported gen9 dex.

Token 0 is <unk>: Tokenizer maps unseen words to -1 and the model clamps ids
to >= 0, so anything missing from the vocab lands on <unk>.

    from protean.formats.gen9randombattle.vocab import build_tokenizer, VOCAB_PATH
    build_tokenizer().save(VOCAB_PATH)
"""
from __future__ import annotations

from pathlib import Path

from protean.formats.gen9randombattle.dex import load_dex
from protean.tokenizer import Tokenizer

VOCAB_PATH = Path(__file__).resolve().parents[2] / "data" / "gen9randombattle_vocab.json"

SPECIAL_TOKENS = [
    "<unk>",   # must stay at index 0
    "<gen9randombattle>",
    "<anychoice>", "<forcedswitch>",
    "<player>", "<opponent>",
    "<move>", "<switch>", "<moveset>",
    "<opp_moves>", "<opp_bench>",
    "<conditions>",
    "<player_prev>", "<opp_prev>",
    "<blank>",
]

STATUS_TOKENS   = ["nostatus", "brn", "par", "slp", "frz", "psn", "tox", "fnt"]
CATEGORY_TOKENS = ["physical", "special", "status"]
UNKNOWN_TOKENS  = ["notype", "unknowntera", "unknownitem", "noitem", "unknownability"]
WEATHER_TOKENS  = ["noweather", "raindance", "sunnyday", "sandstorm", "hail", "snow",
                   "desolateland", "primordialsea", "deltastream"]
TERRAIN_TOKENS  = ["noterrain", "electricterrain", "grassyterrain", "mistyterrain", "psychicterrain"]
ACTION_TOKENS   = ["recharge"]


def build_tokenizer() -> Tokenizer:
    dex = load_dex()
    tok = Tokenizer()
    for group in (SPECIAL_TOKENS, STATUS_TOKENS, CATEGORY_TOKENS, UNKNOWN_TOKENS,
                  WEATHER_TOKENS, TERRAIN_TOKENS, ACTION_TOKENS, dex["types"]):
        for t in group:
            tok._add(t)
    for group in (dex["species"], dex["moves"], dex["items"], dex["abilities"]):
        for t in sorted(group):
            tok._add(t)
    return tok


def get_tokenizer() -> Tokenizer:
    """Load the pre-built vocab, building and saving it on first use."""
    if not VOCAB_PATH.exists():
        tok = build_tokenizer()
        tok.save(VOCAB_PATH)
        return tok
    return Tokenizer.load(VOCAB_PATH)
