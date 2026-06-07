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
- Bug fixed: case-sensitive move dedup in `_most_revealed_my_side` caused >4 moves to accumulate; now normalised to lowercase

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
- `super_squash_history` called after each HF push to prevent LFS history bloat

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

**Published dataset**: `atatark2/protean-gen1ou` on HuggingFace Hub (~194,715 rows)

---

## Phase 2 — Encoding: State → Tensors ✅

Convert raw game state into model-ready tensor observations.

### 2.1 Gen1 Pokédex ✅
**`protean/pokedex.py`**

Static lookup of gen1 base stats for all Pokémon species.

- Used to fill in the numerical observation features (base Atk/Def/Spc/Spe/HP)
- Gen1 has a single "Special" stat (SpA = SpD), unlike later gens

### 2.2 Tokenizer ✅
**`protean/tokenizer.py`**, **`protean/data/gen1ou_vocab.json`**

Simple vocabulary mapping: `word → integer index`. Built from gen1ou data only.

- 459-token vocab: gen1 species, move names, types, status conditions, special structural tokens
- Much smaller than metamon's (no items, abilities, tera types, gen5-9 content)
- Same interface as metamon's `PokemonTokenizer`

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
- Slots 0–3: use move 1–4 (alphabetical order)
- Slots 4–8: switch to benched Pokémon 1–5
- Invalid actions masked at inference time; `row_to_action_idx` returns -1 for unmappable actions

---

## Phase 3 — Model Architecture + BC Training ✅

### 3.1 Model Architecture ✅
**`protean/model.py`** — `Gen1OUPolicy`, 3.52M parameters

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

- Pre-norm transformer (LayerNorm before attention/FFN)
- CLS token prepended to sequence; its output used as the text representation
- Value head computed on `state.detach()` — trunk trains only from policy gradient

### 3.2 BC Training ✅
**`scripts/train_bc.py`**

Behavioural cloning on `atatark2/protean-gen1ou`, streamed from HuggingFace.

- Loss: `F.nll_loss` on 9-slot action, **no action mask during training** (masking the target slot to −∞ causes inf loss when parser-revealed move state lags the action taken; mask used only at inference)
- Class weights: `moves=1.0, switches=2.0` — offsets the ~3:1 move/switch imbalance in the dataset
- Holdout split: deterministic 10% via `zlib.crc32(battle_id.encode()) % 100 < 10`
- Optimizer: AdamW lr=3e-4, betas=(0.9, 0.95), weight_decay=1e-2
- Schedule: CosineAnnealingLR + 2k-step linear warmup
- Grad clip: 1.0; batch size: 256; shuffle buffer: 10k samples

### 3.3 BC Evaluation ✅
**`scripts/eval_bc.py`**

Evaluates a checkpoint on holdout/train/all splits.

```
python scripts/eval_bc.py --checkpoint checkpoints/bc_final.pt --confusion
```

**Final results** (`bc_final.pt`, 50k steps, switch_weight=2.0, holdout set):

| Metric | Accuracy |
|--------|----------|
| Overall | **57.1%** |
| Move (slots 0–3) | **60.3%** |
| Switch (slots 4–8) | **47.1%** |

Cleared the >50% threshold for proceeding to RL fine-tuning.

---

## Phase 4 — RL Fine-tuning (PPO) ✅

### 4.0 Server Setup ✅
**`server/pokemon-showdown/`** (git submodule), **`scripts/start_server.sh`**

Local Pokémon Showdown server for self-play battles.

- Port 8001 (avoids collision with any other local Showdown instance)
- No login server auth — bots connect freely without passwords
- No rated battles
- poke-env 0.8.3.3 verified working

```bash
./scripts/start_server.sh        # foreground
./scripts/start_server.sh &      # background
```

### 4.1 RL Environment ✅
**`protean/rl_env.py`**, **`protean/teams.py`**

poke-env bridge connecting live `Battle` objects to our observation/action space.

- `Gen1OUPlayer(Player)` — subclasses poke-env's `Player`, overrides `choose_move`; collects `Transition` objects into a thread-safe buffer drained by the PPO loop
- `battle_to_obs(battle, prev_my_move, prev_opp_move)` — bridges poke-env `Battle` → our obs format; unions `active_pokemon.moves` with `available_moves` so action mask is correct on turn 1
- `battle_to_action_mask(battle)` — 9-slot bool mask; alphabetical move slots, sequential switch slots
- `action_idx_to_order(idx, battle)` — decodes policy slot → poke-env `BattleOrder`
- `compute_reward(prev, curr, won)` — dense shaped reward scaled to O(0.01)/turn so GAE returns stay in [-1.5, +1.5]
- `teams.py` — 4 gen1ou teams (standard, offensive, balanced, stall); `random_team()` helper

**Reward function:**
```
reward = -0.002                           # per-step cost (bumped from -0.001)
       + 0.01 * damage_dealt             # offense only; hp_gained removed (see below)
       + 0.005 * (gave_status − took_status)
       + 0.01  * (KOs_dealt − KOs_taken)
       + 1.0   * victory   # terminal only; ±1 (not ±100, to keep vf_loss O(1))
```
`hp_gained` (healing reward) was removed: with recovery Pokémon (Soft-Boiled,
Recover), each heal netted ~+0.005/turn vs the old -0.001 step penalty, making
recovery-spam the locally optimal policy and producing 1000-turn stall games.
Step penalty doubled so a 1000-turn draw (-2.0) beats losing (-1.0) decisively.
`TEAM_STALL` removed from the training rotation (still accessible for eval via
`get_team("stall")`) for the same reason — triple recovery moves amplify the issue.

**Key poke-env gotchas encountered:**
- All instance attrs must be set before `super().__init__()` — POKE_LOOP background thread starts mid-`Player.__init__`
- `threading.Lock` (not `asyncio.Lock`) to guard shared buffer across main thread and POKE_LOOP
- `to_id_str(None)` monkey-patched at module import — gen1 has no abilities; poke-env passes `None` to `to_id_str`
- 0.5s sleep between `asyncio.run()` calls — lets server-side teardown complete before next challenge

### 4.2 PPO Training ✅
**`scripts/train_ppo.py`**

Self-play PPO fine-tuning initialised from the BC checkpoint.

```bash
python scripts/train_ppo.py --bc-checkpoint checkpoints/bc_final.pt
python scripts/train_ppo.py --bc-checkpoint checkpoints/bc_final.pt \
    --resume checkpoints/ppo_ep0000500.pt
```

**Self-play setup:**
- 4 learner + 4 opponent players, each pair on a different team rotation
- Opponent weights synced to learner every 50 episodes
- Rollout buffer: 1024 steps before each PPO update

**PPO hyperparameters:**
- GAE: γ=0.99, λ=0.95; clip ε=0.2; 4 epochs/rollout; minibatch 256
- AdamW lr=1e-4, weight_decay=1e-2; grad clip 1.0; vf_coef=0.1
- `ratio.clamp(max=10)` — prevents `inf * 0 = nan` in policy gradient
- Gradient norm guard — skips `optimizer.step()` if norm is non-finite

**KL regularization:**
- `β=0.01 * KL(π_RL ‖ π_BC)` added to PPO loss
- Frozen BC checkpoint as reference; prevents catastrophic forgetting

**MPS-specific fix:**
- Action mask uses `-1e9` fill (not `-inf`) — `log_softmax` backward through `-inf` produces NaN gradients on Apple Silicon GPU; `-1e9` underflows to 0 in float32 (identical forward, stable backward)

**Checkpoints:** saved every 500 episodes to `checkpoints/ppo_ep*.pt`. Final: `checkpoints/ppo_final.pt`

### 4.3 Success Criteria
- Win rate vs. random agent: >90%
- Win rate vs. frozen BC policy: >60%
- No catastrophic forgetting: move accuracy in self-play stays >50%

---

## Phase 5 — Evaluation & Deployment 🔄

### 5.1 Evaluation
- Win rate vs. random agent
- Win rate vs. metamon's gen1ou baseline (if available)
- Ladder performance on Pokémon Showdown

### 5.2 Live Play Server
Connect trained agent to Pokémon Showdown via websocket and play live matches.
