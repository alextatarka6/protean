# Pokémon Showdown RL Agent
 
A self-play reinforcement learning agent for Pokémon Showdown Random Battles (Gen 9).
 
**Current status:** Phase 0 — environment setup and orientation.
 
See `pokemon-rl-plan.md` (one level up) for the full project plan.
 
## Quick start
 
### 1. Set up the local Showdown server
 
You need a local Showdown server so you don't hit the public ladder during development. From this repo's parent directory:
 
```bash
bash scripts/setup_showdown.sh
```
 
This clones `smogon/pokemon-showdown`, installs dependencies, and starts the server on port 8000 with auth disabled. Leave this running in its own terminal.
 
### 2. Set up the Python environment
 
Use Python 3.10 or newer. From the repo root:
 
```bash
python -m venv .venv
source .venv/bin/activate     # on Windows: .venv\Scripts\activate
pip install -e .[dev]
```
 
### 3. Run the smoke test
 
Two random agents battling each other against the local server:
 
```bash
python scripts/random_vs_random.py --battles 10
```
 
Expected output: a win/loss tally roughly 50/50. If you get a connection error, your Showdown server isn't running on `localhost:8000`.
 
### 4. Watch a battle in the browser
 
While agents are battling, open `http://localhost:8000` in your browser, log in as a guest, and search for the bot's username (`RandomBot1` or `RandomBot2`). You can spectate live games — extremely useful for debugging.
 
## Project structure
 
```
pokemon-rl/
├── README.md
├── pyproject.toml          # dependencies + package config
├── .gitignore
├── src/pokerl/
│   ├── __init__.py
│   ├── agents/             # bot implementations (heuristic, supervised, RL)
│   │   ├── __init__.py
│   │   └── heuristic.py    # Phase 1 stub — max-damage bot
│   ├── encoding/           # battle state -> tensor
│   │   ├── __init__.py
│   │   └── battle.py       # Phase 1 stub — feature encoder
│   └── eval/               # evaluation harness
│       ├── __init__.py
│       └── arena.py        # Phase 1 stub — round-robin tournaments
├── scripts/
│   ├── random_vs_random.py # Phase 0 smoke test
│   └── setup_showdown.sh   # one-shot Showdown server setup
└── tests/
    └── test_imports.py     # sanity test that everything imports
```
 
## Phase roadmap
 
- **Phase 0** (this week): get the env running, understand the protocol. ← *you are here*
- **Phase 1**: heuristic baseline + state encoder
- **Phase 2**: RL fundamentals on toy environments (CartPole, LunarLander)
- **Phase 3**: supervised pretraining on Smogon replays
- **Phase 4**: PPO self-play fine-tuning
- **Phase 5**: opponent modeling / belief states
- **Phase 6**: ladder evaluation
## Notes
 
The `poke-env` API has shifted across versions. If imports fail, check the [poke-env docs](https://poke-env.readthedocs.io/) for the version pinned in `pyproject.toml` and adjust. The `random_vs_random.py` script targets poke-env >= 0.8.
 
Don't run any of this against the public Showdown ladder yet. Use the local server.