# Protean — Gen1OU Pokémon Showdown AI

A from-scratch RL agent for Pokémon Showdown Gen 1 OU, inspired by the
[metamon paper](https://arxiv.org/abs/2504.04395). Pipeline: raw replays →
parsed dataset → BC pretraining → PPO self-play fine-tuning.

**Current status:** Phase 5 — live ladder play and evaluation.

---

## Quick start

### 1. Set up credentials

Copy your Pokémon Showdown bot account credentials into `.env`:

```
PS_USERNAME=YourBotName
PS_PASSWORD=yourpassword
```

Register a bot account at https://play.pokemonshowdown.com first.

### 2. Activate the Python environment

```bash
source .venv/bin/activate
```

### 3. Play on the ladder

```bash
python scripts/ladder.py --checkpoint checkpoints/ppo_ep0000500.pt
```

Spectate live by searching the bot's username on the PS website.
Results are logged to `ladder_history.jsonl`.

### 4. Start the local server (for training/eval)

```bash
./scripts/start_server.sh &
```

Server runs on port 8001 with auth disabled.

### 5. Evaluate a checkpoint

```bash
# BC baseline (offline — accuracy on HF holdout set)
python scripts/eval_bc.py --checkpoint checkpoints/bc_final.pt --confusion

# Sweep PPO checkpoints vs BC (live battles, server required)
python scripts/eval_rl.py --sweep \
    --bc-checkpoint checkpoints/bc_final.pt \
    --min-episode 500 --max-episode 3500 \
    --n-battles 50
```

---

## Training

### BC pretraining

```bash
python scripts/train_bc.py
```

Streams from `atatark2/protean-gen1ou` on HuggingFace. Default: 200k steps,
switch_weight=2.0. Current checkpoint: `bc_final.pt` (140k steps).

### PPO self-play

```bash
# Start server, then train
./scripts/start_server.sh &
sleep 3
python scripts/train_ppo.py --bc-checkpoint checkpoints/bc_final.pt

# Resume from checkpoint
python scripts/train_ppo.py \
    --bc-checkpoint checkpoints/bc_final.pt \
    --resume checkpoints/ppo_ep0002500.pt
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
  tokenizer.py       # Gen1Tokenizer (460 tokens)
  obs_space.py       # Gen1OUObservationSpace, Gen1ActionSpace
  model.py           # Gen1OUPolicy (5.10M params) — turn encoder + trajectory transformer
  rl_env.py          # poke-env bridge: Gen1OUPlayer, compute_reward
  teams.py           # 4 training teams (standard, offensive, balanced, zam_egg_zap) + stall

scripts/
  build_gen1ou_dataset.py  # Builds HF dataset from raw replays
  train_bc.py              # BC training loop
  eval_bc.py               # BC accuracy eval (overall/move/switch)
  train_ppo.py             # PPO self-play training loop
  eval_rl.py               # Live battle evaluation (BC, PPO, sweep)
  ladder.py                # Rated ladder play on the real Showdown server
  play_vs_agent.py         # Human vs bot on local server
  start_server.sh          # Start local Showdown server on port 8001

server/
  pokemon-showdown/        # Git submodule: smogon/pokemon-showdown

checkpoints/
  bc_final.pt              # BC checkpoint (140k steps, switch_weight=2.0)
  ppo_ep*.pt               # PPO snapshots every 500 episodes
```

---

## Model

`Gen1OUPolicy` — 5.10M parameters, runs on MPS (Apple Silicon).
Two-stage architecture matching metamon (arXiv 2504.04395):

```
Stage 1 — Turn Encoder (shared weights across history):
  text (77 tokens)  → Transformer(4L, d=256, h=8) → CLS [256]
  numbers (48-dim)  → Linear → ReLU               → [256]
                      Concat [512] → MLP           → turn_emb [256]

Stage 2 — Causal Trajectory Encoder (over K=10 turns):
  [turn_emb_{t-9}, ..., turn_emb_t]
    → CausalTransformer(2L, d=256, h=8)
    → state [256]  (last position = current turn)
    ├── policy_head → log-probs [9]
    └── value_head  → V(s) [1]
```

9 action slots: move 1–4 (alphabetical) + switch to bench 1–5.
Action mask blocks switching into sleeping Pokémon (sleep persists in Gen 1).

---

## Observation space

**Text** (77 tokens): `<gen1ou>` context + player active + moves×4 + bench×5 +
opponent active + conditions + prev moves + `<bench_status>` (5 status tokens).

**Numbers** (48-dim float32): HP fractions, base stats, boosts for active
Pokémon on both sides + bench HP + move power/accuracy.

---

## Reward (metamon Appendix E.1)

```
r = 1.0 * hp_dealt
  + 1.0 * hp_gained
  + 0.5 * (gave_status - took_status)
  + 1.0 * (kos_dealt - kos_taken)
  ± 100.0                              # terminal win/loss dominates
```

No per-step penalty. γ=0.999 (long-horizon discounting for Gen1's 100+ turn games).

---

## Dataset

HuggingFace: `atatark2/protean-gen1ou` (~194,715 rows, both POVs per battle).
Source: `jakegrigsby/metamon-raw-replays` shards 35 & 36.
