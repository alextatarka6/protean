# Self-Play RL Agent for Pokémon Showdown — Project Plan

**Format:** Random Battles (Gen 9 recommended — most active, simplest team-building model)
**Your starting point:** Solid supervised ML, new to RL, laptop CPU + potential AWS EC2

This plan is designed so each phase produces a working artifact you can play against, not just code that compiles. It also front-loads RL fundamentals using your existing supervised-ML intuitions, so you're not learning RL in the abstract while also fighting Pokémon-specific complexity.

---

## Guiding principles

A few things worth internalizing before you write any code.

**Build the simplest thing that works at every stage.** A heuristic bot that picks the highest-expected-damage move beats a half-broken neural net every time. You want a baseline you can beat, then beat the baseline, then beat that.

**Self-play RL from scratch is a trap.** Random initialization in a 9-action stochastic partially-observable game means your agent spends weeks learning that Tackle does damage. Bootstrap from supervised pretraining on human replays, then fine-tune with self-play. This is what Metamon (UT Austin, 2024) did and it's the right call.

**Compute reality check.** The Showdown simulator is fast and CPU-friendly — you can run thousands of self-play games per hour on a laptop. The bottleneck is neural net training, which benefits from a GPU. EC2 spot instances (g4dn.xlarge or g5.xlarge) are the right answer when you get there. For now, your laptop is fine.

**Pokémon is poker, not chess.** You don't know the opponent's full team. Strong play requires reasoning about hidden information — what set the opponent is *probably* running given what they've revealed. Plan for this from day one in your state representation.

---

## Phase 0 — Setup and orientation (week 1)

The goal of this phase is to feel the environment in your hands. No ML yet.

Read the [poke-env docs](https://poke-env.readthedocs.io/) end to end — this is the Python library that wraps Showdown's protocol and gives you a Gym-style env. Skim [pmariglia's showdown bot](https://github.com/pmariglia/showdown) to see what a heuristic agent looks like. Read the [Metamon paper/repo](https://github.com/UT-Austin-RPL/metamon) — it's the most relevant prior art. Skim Smogon's [Random Battles set generator](https://github.com/smogon/pokemon-showdown/blob/master/data/random-battles/) so you understand what space your opponent's set is drawn from.

Set up a local Showdown server (instructions in poke-env docs) so you don't hit the public ladder while training. Run two random agents against each other. Watch a battle in the browser. Log a few full battle traces and read them.

**Deliverable:** two `RandomPlayer` agents from poke-env battling each other locally, with logs you can inspect. You should be able to answer: "what does an action look like, what does an observation look like, what does a reward look like."

---

## Phase 1 — Heuristic baseline and state encoder (weeks 2–3)

Now build the dumbest reasonable bot. This forces you to design your state encoding, which is the hardest design decision in the whole project.

The bot itself is straightforward: on each turn, for each legal move, compute expected damage using a damage calculator (poke-env has helpers, or use [@smogon/calc](https://github.com/smogon/damage-calc) bindings). Pick the move with the highest expected damage. Switch when your active mon is at a type disadvantage worse than some threshold.

While you build this, you'll need to extract features from the battle state. **Do this carefully — your RL agent will use the same encoder.** A good starting representation includes, for each side: the active Pokémon (species one-hot or embedding, current HP fraction, status, stat boosts, item if revealed, ability if revealed), the bench (same fields, possibly less detail), field state (weather, terrain, hazards on each side, screens), and recent move history (last few moves for both players).

For Random Battles specifically, you also want to encode *what the opponent could be running*. Maintain a belief: a probability distribution over the opponent's possible moves/items/abilities given the Random Battles set generator and what they've revealed so far. Even a rough "set of possible movesets" is much better than nothing.

**Deliverable:** a heuristic bot that beats `RandomPlayer` ~95% of the time and beats `MaxDamagePlayer` (poke-env's built-in greedy bot) >50% of the time. A clean `encode_battle(battle) -> tensor` function with a documented schema.

---

## Phase 2 — RL fundamentals on toy environments (weeks 4–5)

Don't skip this. You said you've done supervised but no RL — go learn RL on a toy problem before applying it to Pokémon, or you'll spend weeks unable to tell whether bugs are in your RL code or your Pokémon code.

Work through Spinning Up in Deep RL by OpenAI (the [intro](https://spinningup.openai.com/) and the PPO section specifically). Implement DQN on CartPole, then PPO on LunarLander, using [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) or [CleanRL](https://github.com/vwxyzjn/cleanrl). CleanRL is more educational — single-file implementations you can actually read.

Key concepts to internalize: policy vs value networks, advantage estimation (GAE), the actor-critic loop, entropy bonus and why exploration matters, action masking (you'll need this — most Pokémon actions are illegal at any given turn), and the difference between on-policy (PPO) and off-policy (DQN, SAC) algorithms. For Pokémon, **PPO with action masking is the standard choice** — it handles the variable-legal-action problem cleanly.

**Deliverable:** a working PPO implementation (yours or SB3-wrapped) that solves LunarLander, and a written one-pager in your own words explaining what the policy gradient is and why PPO clips it.

---

## Phase 3 — Supervised pretraining on replays (weeks 6–8)

Now you bridge supervised ML (which you know) into the Pokémon domain. The task: given an encoded battle state, predict the action a human took. Treat it as classification over the (masked) action space.

Pull replay data from Smogon's [replay database](https://replay.pokemonshowdown.com/) — there are scrapers floating around, or you can download bulk dumps. Filter to high-ladder Random Battles. Parse each replay turn-by-turn into `(state, action_mask, action_taken)` tuples. A few hundred thousand turns is plenty to start; millions if you want a strong prior.

Train a policy network: state encoder → MLP or small transformer → softmax over 9 actions, masked. This is a vanilla supervised classification problem — your existing skills apply directly. Measure top-1 accuracy on held-out replays. Expect ~40–55% top-1 in Random Battles (humans don't agree on the right move that often, so the ceiling isn't 100%).

The same architecture, with the perspective flipped and conditioned on revealed information, gives you the **opponent move predictor** you asked about earlier. Train it the same way.

**Deliverable:** a policy network that imitates human play, evaluated by both top-k accuracy on held-out replays and win rate against your Phase 1 heuristic bot. The pretrained weights are your starting point for Phase 4.

---

## Phase 4 — RL fine-tuning via self-play (weeks 9–12)

This is the headline phase. Take your supervised-pretrained policy, drop it into poke-env, and fine-tune with PPO via self-play.

Set up the loop: spawn N parallel battles between the current policy and either (a) past versions of itself or (b) a fixed pool of opponents (your heuristic bot, your supervised model, frozen snapshots). Collect rollouts. Compute advantages. PPO update. Repeat. Reward signal: +1 win, -1 loss, with optional shaping (small rewards for damage dealt, fainted opponents) to speed up early learning — but plan to remove shaping once it's training stably, because it biases the policy.

A few things that will trip you up. Action masking must be applied before the softmax, not after, or your gradients will be wrong. The value network needs the same masked input. Self-play against only-yourself causes mode collapse — keep a pool of past opponents (the AlphaStar-style "league" idea, even a tiny version helps). Random Battles' team randomness adds variance; train on many seeds.

This is where EC2 starts to matter. A `g4dn.xlarge` spot instance ($0.15–0.25/hr) running 32+ parallel self-play workers will give you orders of magnitude more throughput than your laptop. You don't need it for Phase 0–3.

**Deliverable:** a fine-tuned agent that beats your Phase 3 supervised model >55% of the time and beats your Phase 1 heuristic bot >70% of the time, evaluated over hundreds of games.

---

## Phase 5 — Opponent modeling and belief states (weeks 13–16, optional but high-leverage)

This is where you go from "decent bot" to "interesting bot." The agent so far treats the hidden opponent set as a black box. Now make it explicit.

Build a separate **opponent model**: given the visible state and history, output a distribution over the opponent's possible (moves, item, ability, EVs). For Random Battles you can enumerate possible sets from the generator and condition on revealed info — this is closer to Bayesian filtering than deep learning, though a neural net can learn the posterior too. Feed the marginalized belief into your policy network as additional features. Optionally, do a shallow lookahead (1–2 turns) using the belief as a simulator.

This is also where you build the "predict the opponent's next move" feature you originally asked about — the opponent model gives you exactly that, as a byproduct.

**Deliverable:** an agent that meaningfully outperforms the Phase 4 agent specifically in matchups where hidden information matters (early game, ambiguous sets).

---

## Phase 6 — Evaluation and ladder (ongoing)

Throughout, you need rigorous evaluation. Don't trust win rates against a single opponent — use a fixed eval suite (your heuristic, your supervised model, past self-play snapshots, optionally the public Showdown ladder under a bot account, with rate limits).

Set up tracking from Phase 1 onward — Weights & Biases or even just CSV logging. Plot win rate vs each opponent over time, plot policy entropy (collapse warning sign), plot value loss, plot game length distribution.

Once your Phase 4 agent is stable, you can put it on the public Random Battles ladder and watch it climb. Be respectful — rate-limit, follow Showdown's bot rules, and don't ladder under a name that suggests it's a human.

---

## Realistic timeline

Working ~10 hours/week, 16 weeks (4 months) to a Phase 4 agent that plays competently. Phase 5 is another 4–8 weeks. Each phase will probably take longer than you estimate; plan for that. The first project is mostly about getting the infrastructure right — the second iteration is where you actually compete.

## Common pitfalls

The biggest one is jumping straight to self-play RL without supervised pretraining. You'll watch loss curves go nowhere for weeks. Don't.

The second is under-investing in the state encoder. Garbage features, garbage policy. Spend time on this in Phase 1, and write tests — given a known battle state, your encoder should produce a known tensor.

The third is no eval discipline. "It feels stronger" is not data. Have a fixed eval suite from week 2.

The fourth is debugging RL like supervised learning. RL bugs are sneakier — a buggy reward function or a misapplied mask can produce policies that look reasonable but are subtly wrong. Sanity-check by training on a tiny problem you understand (a 1v1 fixed-team matchup) before scaling up.

## Resources, in order of usefulness

[poke-env](https://github.com/hsahovic/poke-env) — your environment library, non-negotiable.
[Metamon](https://github.com/UT-Austin-RPL/metamon) — most directly relevant prior work, read the paper.
[pmariglia/showdown](https://github.com/pmariglia/showdown) — strong heuristic bot, great reference for the action/state space.
[Spinning Up in Deep RL](https://spinningup.openai.com/) — best RL intro for someone with ML background.
[CleanRL](https://github.com/vwxyzjn/cleanrl) — single-file PPO implementations you can actually read.
[Smogon Random Battles set generator](https://github.com/smogon/pokemon-showdown/tree/master/data/random-battles) — the prior over opponent builds.

## What to do this week

Set up Python 3.10+, install poke-env, get a local Showdown server running, and have two `RandomPlayer`s battle each other. Read one Metamon paper section per day. Don't write any ML yet — feel the environment first.
