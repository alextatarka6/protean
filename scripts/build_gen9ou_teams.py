"""
Build the gen9ou team pool from metamon's team sets (jakegrigsby/metamon-teams),
keeping only teams the local Showdown server accepts today.

    competitive  — expert sample teams scraped from forums (eval / ladder default)
    ladder       — teams predicted from May 2026 ladder replays ("gl_05_26"),
                   randomly sampled; used for self-play diversity

Writes protean/data/gen9ou_teams.json: {"competitive": [text, ...], "ladder": [text, ...]}
in Showdown export format (poke-env 0.8.3.3's ConstantTeambuilder crashes on packed teams).

Usage:
    python scripts/build_gen9ou_teams.py --n-ladder 2000
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import tarfile
from pathlib import Path

from huggingface_hub import hf_hub_download

ROOT       = Path(__file__).resolve().parents[1]
SERVER_DIR = ROOT / "server" / "pokemon-showdown"
OUT_PATH   = ROOT / "protean" / "data" / "gen9ou_teams.json"
REPO       = "jakegrigsby/metamon-teams"

# Reads a JSON list of export-format teams on stdin; writes a JSON list of
# normalised export-format teams, or null where the validator rejects the team.
_VALIDATE_JS = r"""
const {Teams} = require('./dist/sim/teams');
const {TeamValidator} = require('./dist/sim/team-validator');
const validator = TeamValidator.get('gen9ou');
let input = '';
process.stdin.on('data', c => input += c);
process.stdin.on('end', () => {
  const out = JSON.parse(input).map(text => {
    const team = Teams.import(text);
    if (!team || team.length !== 6) return null;
    return validator.validateTeam(team) ? null : Teams.export(team);
  });
  process.stdout.write(JSON.stringify(out));
});
"""


def load_teams(set_name: str) -> list[str]:
    path = hf_hub_download(REPO, f"{set_name}/gen9ou.tar.gz", repo_type="dataset")
    with tarfile.open(path) as tf:
        return [tf.extractfile(m).read().decode() for m in tf.getmembers() if m.isfile()]


def validate(teams: list[str]) -> list[str]:
    out = subprocess.run(
        ["node", "-e", _VALIDATE_JS], cwd=SERVER_DIR, input=json.dumps(teams),
        check=True, capture_output=True, text=True,
    ).stdout
    return [t for t in json.loads(out) if t]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n-ladder", type=int, default=2000, help="ladder teams to keep")
    p.add_argument("--seed",     type=int, default=0)
    args = p.parse_args()

    competitive = validate(load_teams("competitive"))
    print(f"competitive: {len(competitive)} legal")

    ladder_all = load_teams("gl_05_26")
    random.Random(args.seed).shuffle(ladder_all)
    ladder: list[str] = []
    for i in range(0, len(ladder_all), 1000):     # validate in chunks until we have enough
        ladder += validate(ladder_all[i:i + 1000])
        print(f"ladder: {len(ladder)} legal after checking {min(i + 1000, len(ladder_all))}")
        if len(ladder) >= args.n_ladder:
            break
    ladder = ladder[:args.n_ladder]

    OUT_PATH.write_text(json.dumps({"competitive": competitive, "ladder": ladder}))
    print(f"Wrote {OUT_PATH.relative_to(ROOT)} ({OUT_PATH.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
