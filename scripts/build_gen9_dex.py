"""
Export gen9 species / move / item / ability / type names from the local Showdown
checkout, for building the gen9ou vocabulary.

Uses the server's own Dex (server/pokemon-showdown/dist) so every name the
server can send is covered. Writes protean/data/gen9_dex.json:
    {"species": [...], "moves": [...], "items": [...], "abilities": [...], "types": [...]}
(Showdown ids; the vocab builder applies metamon's name cleaning.)

Re-run after updating the server submodule, then rebuild the vocab:
    python scripts/build_gen9_dex.py
    python -m protean.formats.gen9ou.vocab
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT       = Path(__file__).resolve().parents[1]
SERVER_DIR = ROOT / "server" / "pokemon-showdown"
OUT_PATH   = ROOT / "protean" / "data" / "gen9_dex.json"

_EXPORT_JS = r"""
const {Dex} = require('./dist/sim/dex');
const d = Dex.forGen(9);
const ids = (xs) => xs.filter(x => x.exists).map(x => x.id).sort();
process.stdout.write(JSON.stringify({
  species:   ids(d.species.all().filter(s => s.num > 0)),
  moves:     ids(d.moves.all()),
  items:     ids(d.items.all()),
  abilities: ids(d.abilities.all().filter(a => a.num > 0)),
  types:     d.types.names().map(t => d.toID(t)),
}));
"""


def main() -> None:
    if not (SERVER_DIR / "dist").exists():
        raise SystemExit("server/pokemon-showdown/dist missing — run ./scripts/start_server.sh once to build it")
    out = subprocess.run(
        ["node", "-e", _EXPORT_JS], cwd=SERVER_DIR, check=True, capture_output=True, text=True,
    ).stdout
    data = json.loads(out)
    OUT_PATH.write_text(json.dumps(data, separators=(",", ":")))
    print(f"Wrote {OUT_PATH.relative_to(ROOT)}: " + ", ".join(f"{len(v)} {k}" for k, v in data.items()))


if __name__ == "__main__":
    main()
