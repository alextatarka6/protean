# Protean — Gen1OU Pokémon Showdown AI

A from-scratch RL agent for Pokémon Showdown Gen 1 OU, inspired by the
[metamon paper](https://arxiv.org/abs/2406.08070). Pipeline: raw replays →
parsed dataset → BC pretraining → PPO self-play fine-tuning.

**Current status:** Phase 4 complete (PPO). Phase 5 (evaluation & live play) in progress.

---

## Quick start

### 1. Start the local Showdown server

```bash
./scripts/start_server.sh        # foreground
./scripts/start_server.sh &      # background
```

Server runs on port 8001 with auth disabled. Required for all RL training and evaluation.

### 2. Activate the Python environment

```bash
source .venv/bin/activate
```

### 3. Play on the ladder

```bash
export PS_USERNAME="YourBotName"
export PS_PASSWORD="yourpassword"
python scripts/ladder.py --checkpoint checkpoints/ppo_final.pt
```

Register a bot account at https://play.pokemonshowdown.com first.
Spectate live by searching the bot's username on the PS website.

### 4. Evaluate a checkpoint

```bash
# BC baseline (offline — accuracy on HF holdout set)
python scripts/eval_bc.py --checkpoint checkpoints/bc_final.pt --confusion

# BC baseline (live battles vs Random + self)
python scripts/eval_rl.py --eval-bc \
    --bc-checkpoint checkpoints/bc_final.pt

# Single PPO checkpoint vs Random + BC (live battles)
python scripts/eval_rl.py \
    --checkpoint checkpoints/ppo_final.pt \
    --bc-checkpoint checkpoints/bc_final.pt

# Sweep all PPO checkpoints (40 live battles each)
python scripts/eval_rl.py --sweep \
    --bc-checkpoint checkpoints/bc_final.pt \
    --n-battles 40 \
    --min-episode 4000
```

`eval_bc.py` runs offline against the HuggingFace dataset (no server needed).
`eval_rl.py` runs live battles on the local Showdown server (server must be running).

---

## Training

### BC pretraining (Phase 3 — complete)

```bash
python scripts/train_bc.py
```

Streams from `atatark2/protean-gen1ou` on HuggingFace. Final checkpoint:
`checkpoints/bc_final.pt` (57.1% holdout accuracy, 50k steps).

### PPO self-play (Phase 4 — complete)

```bash
# Fresh run from BC
python scripts/train_ppo.py --bc-checkpoint checkpoints/bc_final.pt

# Resume from checkpoint
python scripts/train_ppo.py \
    --bc-checkpoint checkpoints/bc_final.pt \
    --resume checkpoints/ppo_ep0005000.pt
```

Checkpoints saved every 500 episodes to `checkpoints/ppo_ep*.pt`.

---

## Project structure

```
protean/
  backend/
    replay_parser/    # Parses raw Showdown .log text → ParsedBattle
    team_inference.py # Fills unrevealed moves via usage stats sampling
    usage_stats.py    # MovesetStats, load_format_stats("gen1ou")
  pov.py             # reconstruct_both_povs → (p1_pov, p2_pov)
  pokedex.py         # get_base_stats, get_types, get_move_data
  tokenizer.py       # Gen1Tokenizer (459 tokens)
  obs_space.py       # Gen1OUObservationSpace, Gen1ActionSpace
  model.py           # Gen1OUPolicy (3.52M params) — policy + value heads
  rl_env.py          # poke-env bridge: Gen1OUPlayer, compute_reward
  teams.py           # 3 gen1ou training teams + stall team for eval

scripts/
  build_gen1ou_dataset.py  # Builds HF dataset from raw replays
  train_bc.py              # BC training loop
  eval_bc.py               # BC accuracy eval (overall/move/switch)
  train_ppo.py             # PPO self-play training loop
  eval_rl.py               # Live battle evaluation (BC, PPO, sweep)
  ladder.py                # Rated ladder play on the real Showdown server
  start_server.sh          # Start local Showdown server on port 8001

server/
  pokemon-showdown/        # Git submodule: smogon/pokemon-showdown

checkpoints/
  bc_final.pt              # BC checkpoint (57.1% accuracy)
  ppo_ep*.pt               # PPO snapshots every 500 episodes
```

---

## Model

`Gen1OUPolicy` — 3.52M parameters, runs on MPS (Apple Silicon).

```
text (71 tokens)  → Transformer(4L, d=256, h=8) → CLS [256]
numbers (48-dim)  → Linear → ReLU              → [256]
                    Concat [512] → MLP          → state [256]
                    ├── policy_head → log-probs [9]
                    └── value_head  → V(s) [1]
```

9 action slots: move 1–4 (alphabetical) + switch to bench 1–5.

---

## Reward (metamon Appendix E.1)

```
r = -0.002                              # per-step cost
  + 0.01 * (hp_dealt + hp_gained)      # net HP differential
  + 0.005 * (gave_status - took_status)
  + 0.01  * (kos_dealt - kos_taken)
  ± 1.0                                # terminal win/loss
```

---

## Dataset

HuggingFace: `atatark2/protean-gen1ou` (~194,715 rows, both POVs per battle).
Source: `jakegrigsby/metamon-raw-replays` shards 35 & 36.
