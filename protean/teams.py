"""
Gen1OU teams for self-play training.

Teams are written in Pokémon Showdown's readable format.
Gen1 has no items, no abilities, no natures — only species + 4 moves.
All Pokémon are level 100 with max stats by default.

Multiple teams are provided so learner/opponent pairs can be randomly
assigned different teams across episodes, improving self-play diversity.
"""
from __future__ import annotations
import random

# ---------------------------------------------------------------------------
# Team definitions
# ---------------------------------------------------------------------------

# Standard tournament team: Tauros + Chansey backbone with strong coverage
TEAM_STANDARD = """
Tauros
- Body Slam
- Hyper Beam
- Blizzard
- Earthquake

Chansey
- Soft-Boiled
- Thunder Wave
- Ice Beam
- Seismic Toss

Alakazam
- Psychic
- Recover
- Thunder Wave
- Seismic Toss

Starmie
- Blizzard
- Thunder Wave
- Recover
- Thunderbolt

Exeggutor
- Sleep Powder
- Psychic
- Explosion
- Double-Edge

Snorlax
- Body Slam
- Earthquake
- Self-Destruct
- Reflect
""".strip()

# Offensive team: fast breakers + Zapdos for Electric coverage
TEAM_OFFENSIVE = """
Tauros
- Body Slam
- Hyper Beam
- Blizzard
- Fire Blast

Zapdos
- Thunderbolt
- Thunder Wave
- Drill Peck
- Agility

Starmie
- Blizzard
- Thunderbolt
- Recover
- Thunder Wave

Gengar
- Hypnosis
- Night Shade
- Explosion
- Psychic

Exeggutor
- Sleep Powder
- Psychic
- Explosion
- Stun Spore

Snorlax
- Body Slam
- Earthquake
- Amnesia
- Self-Destruct
""".strip()

# Balanced team: Chansey wall + Rhydon + mixed coverage
TEAM_BALANCED = """
Tauros
- Body Slam
- Hyper Beam
- Blizzard
- Earthquake

Chansey
- Soft-Boiled
- Thunder Wave
- Seismic Toss
- Ice Beam

Jolteon
- Thunderbolt
- Thunder Wave
- Pin Missile
- Double Kick

Starmie
- Surf
- Blizzard
- Recover
- Thunder Wave

Rhydon
- Earthquake
- Rock Slide
- Body Slam
- Substitute

Exeggutor
- Sleep Powder
- Psychic
- Explosion
- Double-Edge
""".strip()

# Slowbro stall: double walls + Lapras
TEAM_STALL = """
Tauros
- Body Slam
- Hyper Beam
- Blizzard
- Earthquake

Chansey
- Soft-Boiled
- Thunder Wave
- Ice Beam
- Seismic Toss

Slowbro
- Amnesia
- Surf
- Ice Beam
- Thunder Wave

Alakazam
- Psychic
- Recover
- Thunder Wave
- Seismic Toss

Lapras
- Blizzard
- Thunderbolt
- Body Slam
- Sing

Snorlax
- Body Slam
- Reflect
- Self-Destruct
- Earthquake
""".strip()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

# TEAM_STALL excluded from training rotation: triple-recovery moveset makes
# hp_gained reward (now removed) and stalling incentives too strong.  Keep it
# accessible via get_team("stall") for eval purposes.
ALL_TEAMS: list[str] = [
    TEAM_STANDARD,
    TEAM_OFFENSIVE,
    TEAM_BALANCED,
]


def random_team(rng: random.Random | None = None) -> str:
    """Return a randomly chosen gen1ou team string."""
    if rng is not None:
        return rng.choice(ALL_TEAMS)
    return random.choice(ALL_TEAMS)


def get_team(name: str) -> str:
    """
    Return a team by name.
    Valid names: 'standard', 'offensive', 'balanced', 'stall'.
    """
    mapping = {
        "standard":  TEAM_STANDARD,
        "offensive": TEAM_OFFENSIVE,
        "balanced":  TEAM_BALANCED,
        "stall":     TEAM_STALL,
    }
    if name not in mapping:
        raise ValueError(f"Unknown team {name!r}. Valid: {list(mapping)}")
    return mapping[name]
