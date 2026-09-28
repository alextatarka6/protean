"""
Export gen9 dex + random-battle set data from the local Showdown checkout.

Uses the server's own Dex (server/pokemon-showdown/dist) so species/move/item/
ability ids exactly match what the server generates in gen9randombattle.

Writes protean/data/gen9randombattle_dex.json:
    species:   {id: {"types": [..], "base_stats": {hp, atk, def, spa, spd, spe}}}
    aliases:   {cosmetic_forme_id: species_id}
    base_species: {forme_id: base_species_id}   (fallback for randbats set lookup)
    moves:     {id: {"type", "category", "base_power", "accuracy", "priority", "pp"}}
    items:     [id, ...]
    abilities: [id, ...]
    types:     [id, ...]
    randbats:  {species_id: {"level": int, "sets": [{"role", "moves", "abilities", "tera_types"}]}}

Re-run after updating the server submodule, then rebuild the vocab:
    python scripts/build_gen9rb_data.py
    python -c "from protean.formats.gen9randombattle.vocab import build_tokenizer, VOCAB_PATH; build_tokenizer().save(VOCAB_PATH)"
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT       = Path(__file__).resolve().parents[1]
SERVER_DIR = ROOT / "server" / "pokemon-showdown"
OUT_PATH   = ROOT / "protean" / "data" / "gen9randombattle_dex.json"

_EXPORT_JS = r"""
const {Dex} = require('./dist/sim/dex');
const d = Dex.forGen(9);
const sets = require('./data/random-battles/gen9/sets.json');
const setIds = new Set(Object.keys(sets));

const species = {};
const aliases = {};   // cosmetic forme id → species id (e.g. florgesyellow → florges)
const baseSpecies = {};   // forme id → species whose randbats sets it uses (mimikyubusted → mimikyu)
for (const s of d.species.all()) {
  if (!s.exists || s.num <= 0) continue;
  if (s.isNonstandard && !setIds.has(s.id)) continue;
  species[s.id] = {types: s.types.map(t => d.toID(t)), base_stats: s.baseStats};
  for (const c of s.cosmeticFormes || []) aliases[d.toID(c)] = s.id;
  const base = d.toID(s.battleOnly || s.baseSpecies);
  if (base !== s.id) baseSpecies[s.id] = base;
}

const moves = {};
for (const m of d.moves.all()) {
  if (!m.exists || (m.isNonstandard && m.isNonstandard !== 'Unobtainable')) continue;
  moves[m.id] = {
    type: d.toID(m.type), category: d.toID(m.category), base_power: m.basePower,
    accuracy: m.accuracy === true ? 1.0 : m.accuracy / 100, priority: m.priority, pp: m.pp,
  };
}

const items = d.items.all().filter(i => i.exists && !i.isNonstandard).map(i => i.id);
const abilities = d.abilities.all().filter(a => a.exists && a.num > 0 && !a.isNonstandard).map(a => a.id);
const types = d.types.names().map(t => d.toID(t));

const randbats = {};
for (const [id, entry] of Object.entries(sets)) {
  randbats[id] = {
    level: entry.level,
    sets: entry.sets.map(s => ({
      role: s.role,
      moves: s.movepool.map(m => d.toID(m)),
      abilities: (s.abilities || []).map(a => d.toID(a)),
      tera_types: (s.teraTypes || []).map(t => d.toID(t)),
    })),
  };
}

process.stdout.write(JSON.stringify({species, aliases, base_species: baseSpecies, moves, items, abilities, types, randbats}));
"""


def main() -> None:
    if not (SERVER_DIR / "dist").exists():
        raise SystemExit("server/pokemon-showdown/dist missing — run ./scripts/start_server.sh once to build it")
    out = subprocess.run(
        ["node", "-e", _EXPORT_JS], cwd=SERVER_DIR, check=True, capture_output=True, text=True,
    ).stdout
    data = json.loads(out)
    OUT_PATH.write_text(json.dumps(data, sort_keys=True, separators=(",", ":")))
    print(
        f"Wrote {OUT_PATH.relative_to(ROOT)}: {len(data['species'])} species, "
        f"{len(data['moves'])} moves, {len(data['items'])} items, "
        f"{len(data['abilities'])} abilities, {len(data['randbats'])} randbats species"
    )


if __name__ == "__main__":
    main()
