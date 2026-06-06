# Protean — Project Context for Claude

Gen1OU Pokémon Showdown AI. Pipeline: raw replays → parsed dataset → state encoder → BC-pretrained policy → RL fine-tuning.

---

## Current Status

| Phase | Status |
|-------|--------|
| 1 — Replay parsing & dataset | ✅ Done |
| 2 — State encoder | ✅ Done |
| 3 — Model architecture + BC training | ✅ Done |
| 4 — RL fine-tuning (PPO) | 🔄 Next |
| 5 — Evaluation & live play | 📋 Planned |

Full roadmap: `ROADMAP.md`

---

## Project Structure

```
protean/
  backend/
    replay_parser/
      parser.py          # Parses raw Showdown .log text → ParsedBattle
      types.py           # BattlePokemon, SideState, TurnSnapshot, ParsedBattle, POVSnapshot, POVReplay
      loader.py          # Streams raw replays from HF
    team_inference.py    # Fills unrevealed moves/team slots via usage stats sampling
    usage_stats.py       # MovesetStats, load_format_stats("gen1ou")
  pov.py                 # reconstruct_both_povs(battle, format_stats, rng) → (p1_pov, p2_pov)
  pokedex.py             # get_base_stats(species), get_types(species), get_move_data(move)
  tokenizer.py           # Gen1Tokenizer (459 tokens), get_tokenizer(), build_gen1ou_tokenizer()
  obs_space.py           # Gen1OUObservationSpace, Gen1ActionSpace
  model.py               # Gen1OUPolicy (3.52M params) — policy + value heads
  data/
    gen1ou_vocab.json    # Pre-built 459-token vocabulary

scripts/
  build_gen1ou_dataset.py   # Builds HF dataset from raw replays
  train_bc.py               # BC training loop (Phase 3 — complete)
  eval_bc.py                # BC evaluation: overall/move/switch accuracy + confusion matrix
  start_server.sh           # Start local Showdown server on port 8001
  train_ppo.py              # (Phase 4 — to be created) PPO self-play training loop

server/
  pokemon-showdown/         # Git submodule: smogon/pokemon-showdown
    config/config.js        # Local config: port 8001, no auth, no rated battles
```

---

## Dataset

**HuggingFace repo**: `atatark2/protean-gen1ou`
**Rows**: ~194,715 (both POVs per battle, no rating filter)
**Source**: `jakegrigsby/metamon-raw-replays` shards 35 & 36

Each row = one full battle from one player's POV. Key columns:
- `won` (bool), `num_turns` (int)
- Per-turn lists: `my_active_species`, `my_active_hp`, `my_active_status`, `my_active_boosts` (JSON), `my_team` (JSON), `opp_active_species`, `opp_active_hp`, `opp_active_status`, `opp_seen_team` (JSON), `weather`, `my_side_conditions` (JSON), `opp_side_conditions` (JSON)
- Per-turn actions: `my_action_kind` ("move"|"switch"), `my_action_value`, `my_action_forced`, `opp_action_kind`, `opp_action_value`

---

## State Encoder (Phase 2)

### Observation space
`Gen1OUObservationSpace.row_to_obs(row, turn_idx)` → `{"numbers": np.float32[48], "text": np.str_}`

**Text** (~71 space-separated tokens):
```
<gen1ou> <anychoice|forcedswitch>
<player> {species} {type1} {type2} {status}
  <move> {name} {type} {category}   (×4, alphabetical, padded with <blank>)
  <switch> {species} <moveset> {m1} {m2} {m3} {m4}   (×5 alive bench, padded)
<opponent> {species} {type1} {type2} {status}
<conditions> {weather} {my_conditions} {opp_conditions}
<player_prev> {move|<blank>}
<opp_prev> {move|<blank>}
```

**Numbers** (48-dim float32):
```
[0]      opponents_remaining / 6.0
[1-15]   player active: hp, lvl/100, atk/255, spc/255, def/255, spc/255, spe/255, hp_stat/255, 7 boosts/6
[16-27]  player moves ×4: base_power/200, accuracy, priority/5
[28-32]  player bench ×5: hp
[33-47]  opponent active: same 15 features as player active
```

Note: Gen1 has no items or abilities. SpA = SpD = "Special" stat (spc). All Pokémon are level 100.

### Action space
`Gen1ActionSpace` — 9 discrete slots:
- 0–3: use move 1–4 (alphabetical order)
- 4–8: switch to bench Pokémon 1–5

`row_to_action_idx(row, turn_idx)` → int (0–8, or -1 if unmappable)
`action_mask(row, turn_idx)` → bool[9]

### Tokenizer
- 459 tokens: special structural (`<player>`, `<move>`, etc.), gen1 species, gen1 moves, types, statuses
- `get_tokenizer()` loads from `protean/data/gen1ou_vocab.json`
- `tokenize(text)` → np.int32 array

---

## Model (Phase 3 — complete)

**Architecture**: `Gen1OUPolicy` in `protean/model.py`

```
text (71 tokens) → TokenEmbedding(459, 256) + PosEmbedding(128, 256)
               → Transformer(layers=4, d_model=256, nhead=8, ffn_dim=1024)
               → CLS token [256]
                      ↓
numbers (48)   → Linear(48→256) → ReLU [256]
                      ↓
               Concat [512] → MLP(512→256→256) → state repr [256]
                      ↓
         ┌─────────────────────────────┐
         │ policy_head: Linear(256→9) │   → masked log-softmax → action log-probs
         │ value_head:  Linear(256→1) │   → scalar V(s) estimate (for PPO)
         └─────────────────────────────┘
```

**3.52M parameters.** Runs on MPS (Apple Silicon GPU).

Key implementation notes:
- `forward(tokens, numbers, action_mask=None)` → `(log_probs, value)`
- Action mask applied only at inference time (not during BC loss computation — avoids inf loss when parser state lags action)
- Value head computed on `state.detach()` — trunk trains from policy gradient only

**BC Training results** (`checkpoints/bc_final.pt`, 50k steps, switch_weight=2.0):
- Overall holdout accuracy: **57.1%**
- Move accuracy: **60.3%** (slots 0–3)
- Switch accuracy: **47.1%** (slots 4–8)
- Optimizer: AdamW lr=3e-4, cosine decay, 2k warmup steps
- Class weights: moves=1.0, switches=2.0 (offsets ~3:1 imbalance)
- Holdout split: deterministic 10% via CRC32 hash of `battle_id`

---

## RL Fine-tuning (Phase 4 — next)

### Design decisions
- **Environment**: Local Showdown server (`server/pokemon-showdown/`, port 8001) via poke-env
- **Self-play**: 4 parallel battles; opponent weights synced to learner every 50 episodes
- **Critic**: Shared trunk, separate `value_head: Linear(256→1)`, gradient-stopped from trunk
- **Reward** (dense, per turn + terminal):
  ```
  1.0*(damage_dealt + hp_gained) + 0.5*(gave_status − took_status)
  + 1.0*(KOs_dealt − KOs_taken) + 100.0*victory
  ```
- **KL penalty**: `β=0.01 * KL(π_RL ‖ π_BC)` — frozen BC checkpoint as anchor to prevent catastrophic forgetting
- **PPO hyperparameters**: clip ε=0.2, GAE γ=0.99 λ=0.95, 4 epochs/rollout, minibatch 256, lr=1e-4

### Server setup
```bash
# Start local Showdown server (port 8001):
./scripts/start_server.sh

# Or in background:
./scripts/start_server.sh &
```

poke-env `ServerConfiguration`:
```python
from poke_env import ServerConfiguration
LOCAL = ServerConfiguration(
    websocket_url="ws://localhost:8001/showdown/websocket",
    authentication_url="https://play.pokemonshowdown.com/action.php?"
)
```

---

## Key Design Decisions

- **Both POVs per battle** — doubles dataset size, follows metamon paper
- **No rating filter** — all skill levels included
- **Per-sequence rows** — one row per trajectory (not per timestep)
- **Observations computed at training time** — dataset stores raw state; `row_to_obs` called in the training loop
- **Move ordering** — alphabetical within active Pokémon's moveset (consistent across turns)
- **Gen1 specifics** — no items, no abilities, no weather, SpA=SpD=Special stat
- **BC loss** — no action mask during training (mask only at inference); prevents inf loss when parser-revealed move state lags the action taken
- **Holdout split** — deterministic 10% via `zlib.crc32(battle_id.encode()) % 100 < 10`

---

## Important Files Outside `protean/`

- `ROADMAP.md` — full phase-by-phase roadmap with completion status
- `scripts/build_gen1ou_dataset.py` — dataset pipeline (run to rebuild)
- `scripts/train_bc.py` — BC training (Phase 3, complete)
- `scripts/eval_bc.py` — BC evaluation script
- `scripts/start_server.sh` — start local Showdown server on port 8001
- `.gitignore` — excludes `gen1ou_dataset*/`, `.venv/`, `__pycache__/`, `checkpoints/`

---

## Dev Environment

```bash
cd /Users/alextatarka/projects/protean
source .venv/bin/activate   # or prefix commands with .venv/bin/python
```

HuggingFace account: `atatark2` (write token cached via `huggingface-cli login`)
