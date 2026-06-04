# Protean — Gen1OU Pokémon AI Roadmap

A from-scratch implementation of a Pokémon Showdown AI for Gen 1 OU, inspired by the
[metamon paper](https://arxiv.org/abs/2406.08070). The pipeline goes from raw battle replays
all the way to a trained agent that can play live matches.

---

## Phase 1 — Data: Replay Parsing & Dataset ✅

Turn raw Pokémon Showdown battle logs into a structured, ML-ready HuggingFace dataset.

### 1.1 Replay Parser ✅
**`protean/backend/replay_parser/`**

Parses raw `.log` text from Pokémon Showdown replays into structured Python objects.

- Extracts turn-by-turn game state: active Pokémon, HP, status, boosts, weather, side conditions
- Reconstructs actions taken by each player each turn
- Handles spectator-POV logs (replays don't reveal hidden info like the opponent's full team)
- Fixed gen1-4 parsing bugs: removed incorrect team-preview requirement, fixed species-clause logic

Key types (`types.py`):
- `BattlePokemon` — species, HP, status, boosts, revealed moves, item, ability
- `SideState` — active Pokémon + bench + side conditions (Reflect, Light Screen, etc.)
- `TurnSnapshot` — full game state at the start of each turn
- `ParsedBattle` — complete battle with both sides' perspectives

### 1.2 POV Reconstruction ✅
**`protean/pov.py`**, **`protean/backend/team_inference.py`**

Converts a spectator-view `ParsedBattle` into two first-person `POVReplay` objects (one per player).

- Fills unrevealed moves/items/abilities using Smogon usage statistics (`MovesetStats`)
- Infers full opposing team when not all 6 Pokémon were revealed, using teammate co-occurrence sampling
- Produces `POVSnapshot` per turn: what the player actually knew at decision time

### 1.3 Usage Stats ✅
**`protean/backend/usage_stats.py`**

Downloads and caches Smogon monthly usage stats from `jakegrigsby/metamon-usage-stats`.

- `MovesetStats` — per-species: move/item/ability/teammate frequencies
- `load_format_stats("gen1ou")` — loads all 139 gen1ou species, merged across months

### 1.4 Dataset Pipeline ✅
**`scripts/build_gen1ou_dataset.py`**

Builds the final HuggingFace dataset from raw replays.

- Source: `jakegrigsby/metamon-raw-replays` (shards 35 & 36 contain gen1ou)
- Parallelized with `ProcessPoolExecutor` (~1,400 battles/sec)
- Incremental shard saves — safe to interrupt and resume
- Both POVs per battle (following metamon paper design)
- No rating filter (all skill levels included)
- Per-sequence rows: each row is one full battle from one player's POV

**Dataset schema** (each row = one POV trajectory):
| Column | Type | Description |
|--------|------|-------------|
| `battle_id` | str | Unique battle identifier |
| `format` | str | Always `"gen1ou"` |
| `won` | bool | Did this player win? |
| `num_turns` | int | Length of this trajectory |
| `my_active_species` | list[str] | Active Pokémon species per turn |
| `my_active_hp` | list[float] | Active Pokémon HP% per turn |
| `my_active_status` | list[str] | Status condition per turn |
| `my_active_boosts` | list[str] | Stat boosts (JSON) per turn |
| `my_team` | list[str] | Full inferred team (JSON) per turn |
| `opp_active_species` | list[str] | Opponent's active Pokémon per turn |
| `opp_active_hp` | list[float] | Opponent's HP% per turn |
| `opp_active_status` | list[str] | Opponent's status per turn |
| `opp_seen_team` | list[str] | Opponent's revealed team (JSON) per turn |
| `weather` | list[str] | Field weather per turn |
| `my_side_conditions` | list[str] | E.g. Reflect, Light Screen (JSON) |
| `opp_side_conditions` | list[str] | Opponent's side conditions (JSON) |
| `my_action_kind` | list[str] | `"move"` or `"switch"` |
| `my_action_value` | list[str] | Move name or species switched to |
| `my_action_forced` | list[bool] | True for post-faint forced switches |

**Published dataset**: `atatark2/protean-gen1ou` on HuggingFace Hub

---

## Phase 2 — Encoding: State → Tensors 🔄

Convert raw game state into model-ready tensor observations.

### 2.1 Gen1 Pokédex ✅
**`protean/pokedex.py`**

Static lookup of gen1 base stats for all Pokémon species.

- Used to fill in the numerical observation features (base Atk/Def/Spc/Spe/HP)
- Gen1 has a single "Special" stat (SpA = SpD), unlike later gens

### 2.2 Tokenizer ✅
**`protean/tokenizer.py`**, **`protean/data/gen1ou_vocab.json`**

Simple vocabulary mapping: `word → integer index`. Built from gen1ou data only.

- 481-token vocab: gen1 species, move names, types, status conditions, special structural tokens
- Much smaller than metamon's (no items, abilities, tera types, gen5-9 content)
- Same interface as metamon's `PokemonTokenizer`
- Includes a one-time patch for 22 non-gen1 tokens present in the existing published dataset
  due to mislabeled source replays; these will be removed when the dataset is rebuilt

### 2.3 Observation & Action Space ✅
**`protean/obs_space.py`**

Converts game state at a single timestep into model inputs.

**`Gen1OUObservationSpace`** — produces `{"numbers": np.ndarray[48], "text": str}`:
- `text`: space-separated token string encoding all visible game state
  - Active Pokémon: species, types, status, moves (name/type/category)
  - Bench: species + revealed moves (x5 switches)
  - Opponent's active: species, types, status
  - Field: weather, side conditions
  - Previous moves (player + opponent)
- `numbers`: 48-dim float32 vector
  - HP%, base stats, stat boosts for active Pokémon
  - Move base power, accuracy, priority (x4)
  - HP% for each benched Pokémon (x5)
  - HP%, base stats, boosts for opponent's active Pokémon

**`Gen1ActionSpace`** — fixed 9-slot output:
- Slots 0–3: use move 1–4
- Slots 4–8: switch to benched Pokémon 1–5
- Invalid actions masked at inference time

**Single interface** — accepts either:
- `POVSnapshot` (live play via poke-env)
- Dataset row + turn index (offline training from HF dataset)

---

## Dataset Rebuild ✅

Before Phase 3, rebuild `atatark2/protean-gen1ou` with two bug fixes applied:

1. **Team inference fix** (`protean/backend/team_inference.py`) — candidates are now
   filtered to only species present in `format_stats`, preventing non-gen1 Pokémon
   from being inferred as teammates.
2. **Source replay filter** (to add in `scripts/build_gen1ou_dataset.py`) — reject
   any parsed battle where an active Pokémon species is not in the gen1ou usage stats
   pool, removing mislabeled replays from the source data.

Completed. Patch tokens removed from `tokenizer.py`; vocab is now a clean 458-token
pure gen1ou set.

---

## Phase 3 — Model: Architecture 📋

The neural network that maps observations to action probabilities.

### 3.1 Text Encoder
Transformer that embeds the token sequence into a context vector.
Uses the gen1 tokenizer vocab as its embedding table.

### 3.2 Numerical Encoder
Small MLP that processes the 48-dim numbers vector.

### 3.3 Policy Head
Fuses text + numerical embeddings → 9-dim action logits.
Masked softmax over valid actions.

### 3.4 Value Head (for RL)
Additional output head → scalar state value estimate. Used during PPO fine-tuning.

---

## Phase 4 — Training 📋

### 4.1 Behavioural Cloning (BC) Pretraining
Supervised training on the `atatark2/protean-gen1ou` dataset.
- Loss: cross-entropy on `my_action_kind` + `my_action_value`
- Reward shaping not needed — pure imitation of human play
- Teaches the model what good Pokémon play looks like

### 4.2 Reward Function
Per-turn shaped reward, matching metamon's `DefaultShapedReward`:
```
reward = 1.0 * (damage_done + hp_gain)
       + 0.5 * (gave_status - took_status)
       + 1.0 * (removed_pokemon - lost_pokemon)
       + 100.0 * victory  # +100 win, -100 loss
```
Fully computable from stored dataset columns at training time.

### 4.3 RL Fine-tuning (PPO)
Self-play or vs. random/heuristic opponents via poke-env.
- Initialise from BC-pretrained weights
- Fine-tune with Proximal Policy Optimisation

---

## Phase 5 — Evaluation & Deployment 📋

### 5.1 Evaluation
- Win rate vs. random agent
- Win rate vs. metamon's gen1ou baseline (if available)
- Ladder performance on Pokémon Showdown

### 5.2 Live Play Server
Connect trained agent to Pokémon Showdown via websocket and play live matches.
