# Reproducing Metamon: Offline RL for Competitive Pokémon — Project Plan

**Paper:** [Human-Level Competitive Pokémon via Scalable Offline Reinforcement Learning with Transformers](https://arxiv.org/abs/2504.04395) (Grigsby et al., RLC 2025)  
**Codebase + data:** [metamon.tech](https://metamon.tech)  
**Starting point:** Solid supervised ML, new to RL, MacBook Pro (local) + school HPC (for scale)

This plan follows the paper's actual training pipeline — offline RL on human replay data, with self-play only as a late fine-tuning step — rather than the more common self-play-from-scratch approach. Each phase produces something you can point to, and the plan is explicitly scoped so earlier phases run on your laptop while later phases use school compute to test the paper's scaling claims.

---

## Guiding principles

**The paper's core insight is about data, not environment interaction.** A decade of human Showdown replays is a richer learning signal than anything you can generate via self-play from scratch. Your job is to unlock that data and train on it well — not to build a simulator-based training loop from the start.

**Offline RL is supervised learning with a twist.** If you know supervised ML, you already understand ~80% of what's happening here. A Decision Transformer is a sequence model conditioned on return-to-go; imitation learning is classification. The RL machinery is mostly in how you label and weight the data.

**Reproduction ≠ replication.** You won't retrain the full paper model — that took significant compute. A scoped reproduction means: build the same pipeline, train a smaller version, verify that your metrics trend in the same direction as the paper's, and ideally run one scaling experiment with school compute. That's a legitimate and impressive result.

**Use the released code as a reference, not a crutch.** The authors released everything at metamon.tech. Read their code to understand design decisions, but implement the core components yourself so you actually learn them. Use their checkpoints as baselines to evaluate against.

**Format:** Gen 1–4 competitive singles (the paper's focus) rather than Gen 9 Random Battles. This lets you directly compare against the paper's numbers. If you want Gen 9, the pipeline is identical but you lose the apples-to-apples comparison.

---

## Phase 0 — Orientation (week 1)

The goal is to understand the paper's pipeline end-to-end before writing any original code.

Clone [metamon.tech](https://metamon.tech). Get their environment set up. Load a released checkpoint and watch it play a real battle — either locally via poke-env or on the public Showdown ladder. Read the paper once through, then re-read the data pipeline section carefully. Read the [poke-env docs](https://poke-env.readthedocs.io/) end to end. Understand the Showdown protocol: what a battle log looks like from a spectator's perspective, and how that differs from what a player sees.

Look at one raw replay log from [replay.pokemonshowdown.com](https://replay.pokemonshowdown.com). Then look at the paper's reconstructed trajectory format. Map the transformation in your head before you code it.

**Deliverable:** You can run the paper's released agent against a `RandomPlayer` locally and inspect the full battle trace. You can answer: what does a raw replay look like, what does a reconstructed trajectory look like, and what is the transformation between them?

---

## Phase 1 — Data pipeline (weeks 2–4)

This is the paper's most underrated contribution and the right place to start. The authors built a pipeline to convert third-person spectator logs (what Showdown saves) into first-person agent trajectories (what you need to train on). You're going to build your own version of this.

**Step 1: Acquire replay data.** The paper's full dataset spans over a decade of Showdown replays. For your reproduction, pull a manageable slice — a few months of high-ladder Gen 1 OU replays is fine to start. The paper's repo likely includes scraping utilities; use them or write your own against the Showdown replay API.

**Step 2: Parse spectator logs into turn sequences.** A spectator log contains both players' moves and all revealed information. Parse each battle into a sequence of turns: `(turn_number, all_revealed_state, p1_action, p2_action, outcome)`. Handle edge cases: forfeits, disconnects, incomplete battles. Filter to completed battles only.

**Step 3: Reconstruct the first-person perspective.** This is the hard part. For each player's trajectory, you only have access to what that player *could have observed*: their own team in full, the opponent's revealed Pokémon, moves used, items revealed, and field state. Hidden information (opponent's unrevealed team members, sets not yet shown) must be handled carefully — you track what's been revealed, not what's true. The paper calls this the "first-person reconstruction" problem. Your output for each turn is: `(first_person_observation, legal_action_mask, action_taken, return_estimate)`.

**Step 4: Encode observations as tensors.** Design your state encoder. For each side: active Pokémon (species embedding, HP fraction, status, stat boosts, revealed item/ability), bench (same fields, lower fidelity), field state (weather, terrain, hazards, screens, turn count), and move history (last N moves for both players). Hidden fields use a sentinel value — don't impute, just mark as unknown. This schema is fixed for the rest of the project; design it carefully and write tests.

**Step 5: Compute return estimates.** For offline RL you need a return signal per trajectory. The simplest: `+1` for win, `-1` for loss, discounted back through turns. The paper uses a more sophisticated approach — read their reward section and decide how closely you want to follow it.

**Deliverable:** A dataset of parsed, encoded trajectories stored as tensors with documented schema. A data loader that streams batches. Basic statistics: number of battles, turns per battle distribution, action distribution, win rate distribution by ladder rank. Your encoder passes unit tests on known battle states.

---

## Phase 2 — Imitation learning baseline (weeks 5–7)

Now train the first model: given a trajectory so far, predict the action a human took. This is the paper's first training stage and your first ML result.

**Architecture:** Start with a small causal transformer — the paper uses a transformer over the turn sequence, so do the same rather than an MLP. Input: a sequence of encoded observations up to the current turn. Output: a softmax over the action space, masked by the legal action mask. The masking must happen *before* softmax, not after — illegal actions get `-inf` logit. This is a standard supervised classification problem; your existing skills apply directly.

**Training:** Cross-entropy loss on `action_taken`. Standard train/val split by battle (not by turn — you don't want turns from the same battle in both sets). Use AdamW, cosine LR schedule, gradient clipping. This runs fine on your MacBook for small models (2–4 transformer layers, 128–256 hidden dim).

**Evaluation:** Top-1 accuracy on held-out replays, and win rate when playing against `RandomPlayer` and `MaxDamagePlayer` via poke-env. The paper reports ~40–55% top-1 in competitive formats (humans don't agree on the optimal move, so the ceiling isn't 100%). Win rate against heuristic bots is the more meaningful metric.

**Scale experiment (first school compute use):** Train 3–4 model sizes (ranging from ~1M to ~50M parameters) and plot accuracy vs model size. This directly replicates one of the paper's core scaling claims and is a genuine research result. Run this on school HPC — even a few hours on an A100 gives you data points the paper reports.

**Deliverable:** A trained IL model that outperforms `MaxDamagePlayer`, with a scaling curve across model sizes. Weights saved. This is your first concrete comparison point against the paper.

---

## Phase 3 — Offline RL with Decision Transformer (weeks 8–11)

This is the paper's central training stage. You're going from "predict what humans did" to "learn to act better than the average human by conditioning on high-return trajectories."

**The Decision Transformer setup:** A Decision Transformer (DT) is a causal transformer that takes as input not just the observation sequence, but also the sequence of past returns and a *desired future return* (return-to-go). At inference, you condition on a high return-to-go to get the model to act like a high-performing player. Training is still supervised — you predict actions given the trajectory and return-to-go conditioning — but the return conditioning teaches the model to differentiate between good and bad play rather than just averaging over all humans.

**Implementation steps:**
1. Augment your data loader to include return-to-go at each turn (total future reward from that turn onward in the trajectory).
2. Modify your transformer to accept return-to-go as an additional input token or conditioning signal (see the original DT paper for the architecture).
3. Train with the same cross-entropy objective, but now the model learns to condition its action predictions on the desired return.
4. At inference, set return-to-go to a high value (e.g., 1.0 for "I want to win") and let the model generate moves autoregressively.

**Evaluation:** Win rate against your Phase 2 IL model (target: >55%), your heuristic baseline, and the paper's released checkpoints. The DT should outperform IL because it learns to discriminate between high- and low-return plays rather than averaging over all human behavior.

**School compute:** This is where you want GPU time. Train multiple model sizes, replicate the paper's offline RL scaling curve, and compare your results to their reported numbers. A meaningful deviation is worth investigating and potentially writing up. Request compute time with the framing: "reproducing offline RL scaling experiments from a published RLC 2025 paper."

**Deliverable:** A Decision Transformer agent that beats your IL baseline, with a scaling curve. Direct comparison of your accuracy/win-rate numbers to the paper's Table 1 or equivalent.

---

## Phase 4 — Self-play fine-tuning (weeks 12–14)

Only now does online self-play enter the picture — as a fine-tuning step on top of the offline-trained policy, not as the primary training loop. This is much cheaper and more stable than self-play from scratch because the policy already knows how to play Pokémon.

Set up the loop via poke-env: your Phase 3 model plays against past versions of itself and against your heuristic/IL baselines. Collect rollouts. Fine-tune with PPO (use [CleanRL](https://github.com/vwxyzjn/cleanrl) — single-file implementations you can actually read). Keep a frozen pool of past checkpoints as opponents to avoid mode collapse.

Key implementation notes: action masking must be applied before softmax (same as before). The value network shares the transformer backbone with a separate head. Keep the KL penalty between the fine-tuned policy and the offline policy — this is the "offline fine-tuning" stage the paper describes and it prevents catastrophic forgetting of the human priors you trained in.

This phase can run on your MacBook since you're doing limited fine-tuning, not training from scratch.

**Deliverable:** A fine-tuned agent that beats your Phase 3 DT model and approaches the paper's reported ladder ranking. Evaluate over hundreds of games against your eval suite.

---

## Phase 5 — Ladder evaluation and write-up (weeks 15–16)

Put your best agent on the Showdown ladder (follow their bot rules, rate-limit, don't pretend to be human). Track your ladder ranking over time and compare to the paper's reported top-10% result.

Write a brief technical report (2–4 pages) documenting: your data pipeline, model architecture, training stages, and results vs the paper. Include your scaling curves. This is the artifact that goes on your resume and GitHub.

**Deliverable:** Public GitHub repo with clean code, a results README, and a short write-up. Your agent's ladder ranking documented. This is the resume artifact.

---

## School compute — how to get it and what to ask for

Most universities with CS departments have GPU clusters accessible to undergrads — check for a SLURM portal, research computing office, or cloud credits program. Common options:

- **University HPC cluster:** Apply through your research computing office. The ask: "I'm reproducing a published paper (RLC 2025) and need GPU access to run scaling experiments on a fixed offline dataset." This is textbook legitimate use. Request 1–4 A100/V100 nodes for 1–2 weeks.
- **Professor's lab cluster:** If you have any connection to an ML or systems professor, ask to be added to their allocation. Offer to share results. Professors generally appreciate students doing real paper reproduction work.
- **Google TPU Research Cloud / AWS research credits:** Both offer free credits for academic projects. Apply early — approval takes weeks.

**What you actually need:** For Phase 2 scaling experiments, a few hours on a single A100 is enough to train 4–5 model sizes. For Phase 3, a day or two of A100 time gets you a serious DT model. You don't need a massive allocation — targeted experiments are more valuable than long training runs anyway.

---

## Realistic timeline

Working ~10 hours/week:

- **Weeks 1–4:** Phases 0–1. Mostly engineering; no ML training yet.
- **Weeks 5–7:** Phase 2. First ML results; first school compute request.
- **Weeks 8–11:** Phase 3. Core offline RL work; main school compute use.
- **Weeks 12–14:** Phase 4. Self-play fine-tuning.
- **Weeks 15–16:** Evaluation, write-up, GitHub cleanup.

Total: ~4 months to a Phase 4 agent with documented results. Each phase will probably take longer than estimated — plan for that, especially Phase 1.

---

## Resume framing

At each stage you have something to say:

- **After Phase 1:** "Built a data pipeline to reconstruct first-person agent trajectories from third-person Pokémon Showdown replay logs; processed N battles across M turns."
- **After Phase 2:** "Trained imitation learning baselines across model scales (1M–50M parameters) on a dataset of competitive Pokémon replays; reproduced scaling behavior from Grigsby et al. (RLC 2025)."
- **After Phase 3:** "Implemented and trained a Decision Transformer for offline RL on competitive game data; evaluated against paper-released checkpoints on Pokémon Showdown."
- **After Phase 4:** "Fine-tuned offline RL agent via PPO self-play; agent reached top X% of active Pokémon Showdown players."

---

## Common pitfalls

**Skipping the data pipeline.** It's tempting to use the paper's released dataset directly. Don't — building the pipeline yourself is where you learn the most and is the most defensible project contribution.

**Using an MLP instead of a transformer.** The paper's scaling results are about sequence models. If you use an MLP, you can't replicate the key results and lose the most interesting part of the project.

**Not comparing to the paper's numbers.** Every evaluation should ask "how does this compare to Table X in the paper?" That comparison is the point of a reproduction.

**Requesting too much school compute too early.** Get your pipeline working on small data locally first. Only go to the cluster once you have a working training loop — debugging on a cluster is painful and wastes allocation.

**No eval discipline.** Track win rates against a fixed eval suite (RandomPlayer, MaxDamagePlayer, your IL model, paper checkpoints) from Phase 2 onward. "It feels stronger" is not data.

---

## Resources, in order of usefulness

[metamon.tech](https://metamon.tech) — paper code, data, and checkpoints. Your primary reference.  
[Grigsby et al. 2025](https://arxiv.org/abs/2504.04395) — read it at least twice.  
[Decision Transformer paper](https://arxiv.org/abs/2106.01345) (Chen et al., 2021) — the offline RL architecture you're implementing.  
[poke-env](https://github.com/hsahovic/poke-env) — Python environment library for Showdown.  
[CleanRL](https://github.com/vwxyzjn/cleanrl) — single-file PPO for Phase 4 fine-tuning.  
[Spinning Up in Deep RL](https://spinningup.openai.com/) — background RL reading; focus on the policy gradient section.  
[pmariglia/showdown](https://github.com/pmariglia/showdown) — strong heuristic bot, useful as an eval opponent.

---

## What to do this week

Set up Python 3.10+, install poke-env, clone metamon.tech, and run a released checkpoint in a local battle. Read the paper's data pipeline section (Section 3) carefully and sketch the transformation from a raw replay log to a first-person trajectory on paper. Don't write any ML code yet — understand the data first.

---

## Extension — Gen 9 Random Battles

This section is intentionally scoped outside the main plan. Complete Phase 5 first. Gen 9 Random Battles is a natural next step because it's the highest-traffic format on Showdown (replay data is abundant and continuously generated) and removes team-building from the problem entirely — every battle uses a randomly assigned team, so the agent focuses on in-battle decision-making.

The pipeline changes are modest but real:

**State encoder.** Gen 9 introduces new mechanics — Terastallization, new abilities, new items, and a larger move pool. Your Phase 1 encoder schema was designed for Gen 1–4 and will need a new schema. Design it fresh rather than patching the old one; the hidden-information treatment is the same but the feature space is wider. The legal action mask needs to include the Terastallize action.

**Data.** Random Battles replays are freely available via the Showdown replay API at scale — filter for `gen9randombattle` format. Because teams are random and revealed at battle start, the first-person reconstruction problem is simpler: both teams are known from turn 1. The flip side is that the action distribution is noisier (players are piloting unfamiliar sets), so your IL ceiling may be lower than in the Gen 1–4 work.

**Comparison baseline.** You lose the apples-to-apples comparison with the paper, but you gain a large and active player pool to evaluate against on the live ladder. The paper's ladder ranking methodology still applies — you're just measuring against a different format.

**Why this is interesting.** The paper's core claims — that offline RL on human replays scales predictably and outperforms self-play-from-scratch — should generalize to any format. Testing whether those scaling curves hold in Gen 9 Random Battles is a legitimate extension result that goes beyond reproduction into original contribution. If your Gen 9 scaling curves mirror the Gen 1–4 results, that's evidence the method is format-agnostic.

**Suggested framing for the write-up:** "We extended the pipeline to Gen 9 Random Battles to test whether the paper's scaling behavior generalizes across formats. Results show [X], suggesting [conclusion about format-agnostic vs. format-specific scaling]."
