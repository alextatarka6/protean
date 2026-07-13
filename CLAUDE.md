# Protean — Project Context for Claude

Gen1OU Pokémon Showdown AI. Pipeline: raw replays → parsed dataset → state encoder → BC-pretrained policy → RL fine-tuning.

---

## Current Status

| Phase | Status |
|-------|--------|
| 1 — Replay parsing & dataset | ✅ Done |
| 2 — State encoder | ✅ Done |
| 3 — Model architecture + BC training | ✅ Done |
| 4 — RL fine-tuning (PPO) | ✅ Done |
| 5 — Evaluation & live play | 🔄 In progress |

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
  tokenizer.py           # Gen1Tokenizer (460 tokens), get_tokenizer(), build_gen1ou_tokenizer()
  obs_space.py           # Gen1OUObservationSpace, Gen1ActionSpace
  model.py               # Gen1OUPolicy (5.10M params) — turn encoder + causal trajectory transformer
  rl_env.py              # poke-env bridge: Gen1OUPlayer, battle_to_obs, compute_reward
  teams.py               # 4 training teams + TEAM_STALL (eval only) + random_team() helper
  data/
    gen1ou_vocab.json    # Pre-built 460-token vocabulary

scripts/
  build_gen1ou_dataset.py   # Builds HF dataset from raw replays
  train_bc.py               # BC training loop (Phase 3 — complete)
  eval_bc.py                # BC evaluation: overall/move/switch accuracy + confusion matrix
  start_server.sh           # Start local Showdown server on port 8001
  train_ppo.py              # PPO self-play training loop (Phase 4 — complete)
  eval_rl.py                # Live battle evaluation — BC baseline, PPO vs BC/Random, sweep mode
  ladder.py                 # Rated ladder play on real PS server; reads .env for credentials
  play_vs_agent.py          # Human vs bot on local server

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

**Text** (77 space-separated tokens):
```
<gen1ou> <anychoice|forcedswitch>
<player> {species} {type1} {type2} {status}
  <move> {name} {type} {category}   (×4, alphabetical, padded with <blank>)
  <switch> {species} <moveset> {m1} {m2} {m3} {m4}   (×5 alive bench, padded)
<opponent> {species} {type1} {type2} {status}
<conditions> {weather} {my_conditions} {opp_conditions}
<player_prev> {move|<blank>}
<opp_prev> {move|<blank>}
<bench_status> {s1} {s2} {s3} {s4} {s5}
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

At inference time, switching into sleeping bench Pokémon is hard-masked (sleep persists through switches in Gen 1).

### Tokenizer
- 460 tokens: special structural (`<player>`, `<move>`, `<bench_status>`, etc.), gen1 species, gen1 moves, types, statuses
- `get_tokenizer()` loads from `protean/data/gen1ou_vocab.json`
- `tokenize(text)` → np.int32 array

---

## Model (Phase 3 — complete)

**Architecture**: `Gen1OUPolicy` in `protean/model.py` — two-stage, matching metamon (arXiv 2504.04395).

```
Stage 1 — Turn Encoder (shared weights, runs once per turn in history):
  text (77 tokens) → TokenEmbedding(460, 256) + PosEmbedding(128, 256)
                 → Transformer(layers=4, d_model=256, nhead=8, ffn_dim=1024)
                 → CLS token [256]
                        ↓
  numbers (48)   → Linear(48→256) → ReLU [256]
                        ↓
                 Concat [512] → MLP(512→256) → turn_emb [256]

Stage 2 — Causal Trajectory Encoder (over K=10 turns):
  [turn_emb_{t-9}, ..., turn_emb_t]
    → TrajPosEmbed(10, 256) + CausalTransformer(layers=2, d_model=256, nhead=8)
    → state [256]  (last position = current turn with full history context)
           ↓
   ┌─────────────────────────────┐
   │ policy_head: Linear(256→9) │   → masked log-softmax → action log-probs
   │ value_head:  Linear(256→1) │   → scalar V(s) estimate (for PPO)
   └─────────────────────────────┘
```

**5.10M parameters.** Runs on MPS (Apple Silicon GPU).

Key implementation notes:
- `forward(tokens, numbers, action_mask=None)` accepts `(B, K, T)` or `(B, T)` (K=1 compat for BC)
- `encode_turn_only()` bypasses trajectory encoder — used for KL vs BC (BC was trained K=1)
- Trajectory transformer zero-initialized so it starts as near-identity; trains via PPO
- Value head zero-initialized to avoid huge MSE gradients on first PPO update (returns now ±100)
- BC checkpoint loads via `strict=False` — turn encoder weights restored, traj layers init to zero

**BC Training results** (`checkpoints/bc_final.pt`, 140k steps, switch_weight=2.0):
- switch_weight=2.0 (partial class-imbalance correction — biases toward moves in uncertain situations)
- Optimizer: AdamW lr=3e-4, cosine decay, 2k warmup steps
- Holdout split: deterministic 10% via CRC32 hash of `battle_id`

---

## RL Fine-tuning (Phase 4 — complete)

### Design
- **Environment**: Local Showdown server (`server/pokemon-showdown/`, port 8001) via poke-env 0.8.3.3
- **Self-play**: 4 parallel battle pairs; 3 frozen BC + 1 self-play opponent, sync every 20 episodes
- **History**: K=10 turns of obs history per transition; trajectory transformer learns from context
- **Critic**: Shared trunk, separate `value_head: Linear(256→1)`, zero-init, gradient-stopped from trunk
- **Reward** (matching metamon arXiv 2504.04395, Appendix E.1):
  ```
  1.0*hp_dealt + 1.0*hp_gained + 0.5*(gave_status − took_status)
  + 1.0*(KOs_dealt − KOs_taken) ± 100.0 terminal
  ```
  No per-step penalty — metamon omits it; step penalties destabilised all prior runs.
  ±100 terminal dominates shaping so win/loss is the primary learning signal.
- **KL penalty**: `β=0.1 * KL(π_RL ‖ π_BC)` via `encode_turn_only()` — anchors single-turn reasoning to BC
- **PPO hyperparameters**: clip ε=0.2, GAE γ=0.999 λ=0.95, 4 epochs/rollout, minibatch 256, lr=1e-4, vf_coef=0.1
- **γ=0.999**: Long-horizon discounting critical for Gen1OU — metamon ablation finding; 100+ turn battles need near-undiscounted returns

### PPO results
- ep500: **94% win rate vs frozen BC** (best eval checkpoint)
- Win rate declines monotonically after ep500 vs frozen BC (self-play opponent adapts faster than BC does)
- Ladder results (real PS server): ~50% win rate — vs-BC metric not a reliable proxy for human play

### Key implementation gotchas (poke-env + MPS)
- **All `Gen1OUPlayer` attrs must be set before `super().__init__()`** — poke-env starts the POKE_LOOP background thread partway through `Player.__init__`; any attribute not yet set when a battle message arrives raises `AttributeError`
- **`threading.Lock` not `asyncio.Lock`** — `drain_buffer()` runs on the main thread; `choose_move` and `_battle_finished_callback` run on POKE_LOOP; they are on different event loops
- **`-1e9` not `-inf` for action masking** — `log_softmax` backward on MPS produces NaN gradients through `-inf` inputs; `-1e9` underflows to 0 in float32 (identical forward behaviour) but has well-defined backward
- **`ratio.clamp(max=10)` in PPO** — prevents `inf * 0 = nan` when a valid action collapses to near-zero probability
- **Gradient norm guard** — `clip_grad_norm_` returns the pre-clip norm; if non-finite, skip `optimizer.step()` to avoid corrupting all weights
- **`battle.available_moves` for turn-1 moves** — `active_pokemon.moves` is empty until a move is used; union with `available_moves` (always populated from server `|request|`) for correct slot mapping
- **`to_id_str(None)` monkey-patch** — gen1 has no abilities; poke-env passes `None` to `to_id_str` which crashes iterating it; patched in `rl_env.py` before any poke-env Pokemon objects are created
- **0.5s inter-iteration sleep** — gives POKE_LOOP time to finish server-side teardown before the next challenge is issued; prevents `|popup|You are already challenging someone` dropped challenges
- **KL via `encode_turn_only()`** — BC was trained on single-turn (K=1) obs; passing K=10 to bc_model would use randomly-init traj weights; use current turn only for KL

### Server setup
```bash
./scripts/start_server.sh &
```

### Training
```bash
python scripts/train_ppo.py --bc-checkpoint checkpoints/bc_final.pt
# Resume:
python scripts/train_ppo.py --bc-checkpoint checkpoints/bc_final.pt --resume checkpoints/ppo_ep0000500.pt
```

---

## Live Play (Phase 5 — in progress)

### Ladder
```bash
python scripts/ladder.py --checkpoint checkpoints/ppo_ep0000500.pt --n-games 20 --search-timeout 600
```

Credentials loaded from `.env` (PS_USERNAME, PS_PASSWORD). Results logged to `ladder_history.jsonl`.
Gen1OU queues can be slow — use `--search-timeout 600` (10 min).

### Current observations
- Bot incorrectly switches into sleeping bench Pokémon → **fixed** via hard action mask in `battle_to_action_mask()`
- Rhydon teams are a weakness for zam_egg_zap — Exeggutor (the Rhydon answer) needs protecting; model hasn't yet learned to conserve HP for this matchup
- vs-BC win rate (94% at ep500) does not translate directly to ladder win rate (~50%); need more diverse training signal

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
- **Two-stage model** — turn encoder (BC-pretrained) + causal trajectory transformer (PPO-trained); trajectory layers zero-init so BC weights aren't disrupted on first PPO update

---

## Important Files Outside `protean/`

- `ROADMAP.md` — full phase-by-phase roadmap with completion status
- `scripts/build_gen1ou_dataset.py` — dataset pipeline (run to rebuild)
- `scripts/train_bc.py` — BC training (Phase 3, complete)
- `scripts/eval_bc.py` — BC evaluation script (offline: accuracy on HF dataset)
- `scripts/train_ppo.py` — PPO self-play training (Phase 4, complete)
- `scripts/eval_rl.py` — live battle evaluation; modes: `--eval-bc`, `--checkpoint`, `--sweep`
- `scripts/ladder.py` — play rated Gen1OU ladder on real PS server; requires `PS_USERNAME`/`PS_PASSWORD` env vars
- `scripts/play_vs_agent.py` — human vs bot on local server
- `scripts/start_server.sh` — start local Showdown server on port 8001
- `.gitignore` — excludes `gen1ou_dataset*/`, `.venv/`, `__pycache__/`, `checkpoints/`
- `.env` — PS credentials (gitignored): `PS_USERNAME`, `PS_PASSWORD`

---

## Dev Environment

```bash
cd /Users/alextatarka/projects/protean
source .venv/bin/activate   # or prefix commands with .venv/bin/python
```

HuggingFace account: `atatark2` (write token cached via `huggingface-cli login`)
