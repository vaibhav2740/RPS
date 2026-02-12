# ✊✋✌️ RPS Playground — Algorithm Battle Arena

A modular Rock-Paper-Scissors algorithm testing playground with **103 built-in bots**, **Elo ratings**, **three competition modes**, and a beautiful **dark-themed Web UI**.

---

## 📦 Setup

```bash
# Install the only dependency
pip3 install flask

# Clone / navigate to the project
cd /path/to/fun
```

---

## 🚀 Quick Start

### Web UI (recommended)

```bash
python3 -m rps_playground.web
```

Open **http://localhost:5000** in your browser.

### CLI

```bash
# List all algorithms
python3 -m rps_playground.main --list

# Head-to-Head
python3 -m rps_playground.main head-to-head \
  --algo-a "Markov Predictor" --algo-b "Frequency Analyzer" \
  --rounds 1000 --seed 42

# One vs All
python3 -m rps_playground.main one-vs-all \
  --algo "Pattern Detector" --rounds 1000 --seed 42

# Full Tournament
python3 -m rps_playground.main tournament --rounds 1000 --seed 42

# Export results
python3 -m rps_playground.main tournament \
  --rounds 1000 --seed 42 --export json --output results.json
```

---

## 🌐 Web UI Guide

The web interface has **three tabs**:

### ⚔️ Head-to-Head

Run a single match between any two algorithms.

1. **Select Algorithm A** and **Algorithm B** from the dropdowns
2. **Set rounds** (default 1000) — more rounds = more statistically significant
3. **Set seed** (optional) — for reproducible results; leave empty for random
4. Click **⚔️ FIGHT!**
5. View the result card showing wins, losses, draws, win %, and **Elo change**

### 🏆 Tournament

Full round-robin — every algorithm plays every other algorithm (5253 matches total with 103 bots).

1. **Set rounds per match** and optional **seed**
2. Click **🏆 RUN**
3. View the **leaderboard** sorted by Elo rating
4. Scroll down to see the **Head-to-Head matrix** (W/L/D for every pairing)

### 🤖 One vs All

Test a single algorithm against the entire pool of 103 bots.

1. **Select your algorithm** from the dropdown
2. **Set rounds** and optional **seed**
3. Click **🤖 RUN**
4. View **individual matchup results** (win/loss/draw for each opponent)
5. See the **overall ranking** table

---

## 🤖 Algorithm Reference

All 103 algorithms explained in detail.

> **Notation used throughout:**
> - `counter(X)` = the move that beats X. So `counter(Rock) = Paper`, `counter(Paper) = Scissors`, `counter(Scissors) = Rock`.
> - `loses_to(X)` = the move that loses to X. So `loses_to(Rock) = Scissors`.
> - `opp[t]` = opponent's move at round t. `my[t]` = our move at round t.
> - `opp[-1]` = opponent's most recent move. `opp[-3]` = opponent's move from 3 rounds ago.

---

### 1. Always Rock
**Type:** Static · **Complexity:** Trivial

Plays `Rock` every single round, unconditionally. Zero intelligence — exists purely as a baseline. Any algorithm that can detect a bias should beat it easily by always playing `Paper`.

**Expected win rate vs Pure Random:** 33.3% (can only beat Scissors, which Random plays ⅓ of the time)

---

### 2. Always Paper
**Type:** Static · **Complexity:** Trivial

Plays `Paper` every single round. Beats Always Rock 100% of the time, but loses 100% to Always Scissors.

---

### 3. Always Scissors
**Type:** Static · **Complexity:** Trivial

Plays `Scissors` every single round. Completes the trio of constant-move baselines.

---

### 4. Pure Random
**Type:** Randomized · **Complexity:** Trivial

Each round, picks Rock, Paper, or Scissors with equal probability (⅓ each).

**Why it matters:** This is the **Nash equilibrium** strategy for RPS. Against Pure Random, no strategy can achieve a long-run win rate above 33.3%. However, Pure Random also can't *exploit* predictable opponents — it will always converge to ~33% wins / 33% losses / 33% draws regardless of opponent.

**Math:**
```
P(win) = P(loss) = P(draw) = 1/3 against any opponent, in expectation
```

---

### 5. Cycle
**Type:** Deterministic · **Complexity:** Trivial

Follows a fixed repeating sequence: `Rock → Paper → Scissors → Rock → Paper → Scissors → ...`

```
Round:  0     1      2         3     4      5
Move:   Rock  Paper  Scissors  Rock  Paper  Scissors  ...
Formula: move = MOVES[round_num % 3]
```

Guarantees a perfectly uniform move distribution, but the fixed pattern is trivially exploitable by any pattern detector.

---

### 6. Persistent Random 🎭
**Type:** Randomized · **Complexity:** Simple

Picks a random move and plays it for a **random duration** (5-15 rounds). Then picks a new move and repeats.

```
Round t:
  if remaining_duration > 0:
    play current_move
  else:
    current_move = Random()
    remaining_duration = Random(5, 15)
    play current_move
```

**Why it works:** It mimics a "stubborn" player but switches unpredictably. This confuses pattern detectors that expect either constant play (Always Rock) or frequent switching (Random).


---

### 7. Tit-for-Tat
**Type:** Reactive · **Complexity:** Simple

Identical to Mirror Opponent, except it always opens with `Rock` instead of a random move. Famous from Axelrod's iterated Prisoner's Dilemma tournaments — the simplest strategy that's "nice" (never initiates conflict) and "retaliatory" (immediately copies any opponent behavior).

```
Round 0: Rock
Round t: play opp[t-1]
```

---

### 8. Anti-Tit-for-Tat
**Type:** Reactive · **Complexity:** Simple

Plays the move that **beats** the opponent's last move. If they played Rock, plays Paper.

```
Round 0: Random
Round t: play counter(opp[t-1])
```

**Why it works:** If an opponent tends to repeat moves, this directly exploits that repetition. It's the simplest "smart" strategy — but it fails against opponents who anticipate it.

---

### 9. Frequency Analyzer
**Type:** Analytical · **Complexity:** Medium

Counts every move the opponent has ever played, finds their most common move, and plays the counter.

```
Round t:
  counts = {Rock: n_R, Paper: n_P, Scissors: n_S}  (from all opp history)
  most_common = argmax(counts)
  play counter(most_common)
```

**Example after 100 rounds:** If opponent played Rock 50 times, Paper 30 times, Scissors 20 times → most common is Rock → play Paper.

**Weakness:** Treats all history equally. If an opponent switched strategy at round 50, the first 50 rounds of irrelevant data still pollute the counts. This is exactly what Decay Analyzer (#21) fixes.

---

### 10. Markov Predictor
**Type:** Analytical · **Complexity:** Medium

Builds a **first-order Markov transition table** — tracking what the opponent plays *after* each of their moves.

```
Transition table T:
  T[Rock]     = {Rock: 5, Paper: 15, Scissors: 3}
  T[Paper]    = {Rock: 8, Paper: 2,  Scissors: 12}
  T[Scissors] = {Rock: 10, Paper: 7, Scissors: 1}

Round t:
  last_opp = opp[t-1]
  predicted = argmax(T[last_opp])   ← "after they played X, they most often play Y"
  play counter(predicted)
```

**Example:** If the table shows that after playing Rock, the opponent plays Paper 65% of the time → predict Paper → play Scissors.

**Math:** This models the opponent as a Markov chain with transition probabilities:
```
P(opp[t] = j | opp[t-1] = i) = T[i][j] / Σ_k T[i][k]
```

---

### 11. Spiral 🌀
**Type:** Deterministic · **Complexity:** Medium

Plays a **double-cycle** pattern: `R, R, P, P, S, S, ...`

```
Round:  0  1  2  3  4  5
Move:   R  R  P  P  S  S
```

**Why it works:** It's a structured pattern that breaks simple "Cycle" detectors (which expect period 3). It also confuses "Anti-Repeat" bots because it repeats every move once.


---

### 12. Win-Stay, Lose-Shift (WSLS)
**Type:** Adaptive · **Complexity:** Simple

A classic reinforcement learning principle:
- **Won or drew** last round? → Repeat the same move (it's working, don't change)
- **Lost** last round? → Switch to what would have beaten the opponent

```
Round 0: Random
Round t:
  if my[t-1] beat opp[t-1]:   STAY  → play my[t-1]
  if my[t-1] == opp[t-1]:     STAY  → play my[t-1]
  if my[t-1] lost to opp[t-1]: SHIFT → play counter(opp[t-1])
```

**Why it's clever:** Against constant-move opponents, WSLS finds the winning move in at most 2 rounds and then locks onto it permanently.

---

### 13. Meta-Predictor
**Type:** Ensemble · **Complexity:** High

Runs **3 sub-predictors** simultaneously, tracks which has been most accurate, and follows the best one:

| Sub-predictor | Prediction logic |
|---------------|-----------------|
| **Frequency** | Opponent plays their overall most common move |
| **Repeat** | Opponent repeats their last move |
| **Anti-repeat** | Opponent plays `counter(their last move)` |

```
Round t:
  pred_1 = most_common(opp_history)    ← frequency
  pred_2 = opp[t-1]                    ← repeat
  pred_3 = counter(opp[t-1])           ← anti-repeat

  # Score each predictor: +1 every time it correctly predicted opp's last move
  scores[i] += 1 if pred_i matched actual opp[t-1]

  best = argmax(scores)
  play counter(predictions[best])
```

---

### 14. Noise Strategy
**Type:** Adaptive · **Complexity:** Simple

80% of the time: plays `counter(most_common_opponent_move)` — identical to Frequency Analyzer.
20% of the time: plays a **random** move.

```
Round t:
  if random() < 0.2:
    play Random
  else:
    play counter(most_common(opp_history))
```

**Why the noise?** Without it, a smart opponent can detect you're doing frequency analysis, predict your move, and counter it. The 20% randomness makes you partially unpredictable — a simple form of mixed strategy.

---

### 15. Adaptive Hybrid
**Type:** Adaptive · **Complexity:** Medium

Maintains 3 sub-strategies and switches between them based on **tracked win rates**:

| Strategy 0 | Frequency Counter (same as #9) |
|---|---|
| **Strategy 1** | Mirror (same as #6) |
| **Strategy 2** | Pure Random (same as #4) |

Every 50 rounds, computes win rate for each strategy and switches to the best:
```
Every 50 rounds:
  rate[i] = wins[i] / plays[i]  for each strategy
  active_strategy = argmax(rate)
```

---

### 16. Last-Move Counter
**Type:** Reactive · **Complexity:** Simple

Plays `counter(opp[-1])` — identical to Anti-Tit-for-Tat (#8), except starts with a **random** move instead of a fixed one.

---

### 17. Weighted Random
**Type:** Randomized · **Complexity:** Medium

Picks moves randomly, but the probability of each counter-move is proportional to how often the opponent plays the corresponding move.

```
Round t:
  freq_R = count(opp played Rock) / total_rounds
  freq_P = count(opp played Paper) / total_rounds
  freq_S = count(opp played Scissors) / total_rounds

  P(play Paper)    = freq_R    ← counter Rock
  P(play Scissors) = freq_P    ← counter Paper
  P(play Rock)     = freq_S    ← counter Scissors
```

**Example:** If opponent played 60% Rock, 30% Paper, 10% Scissors → play Paper with 60% probability, Scissors with 30%, Rock with 10%.

**Difference from Frequency Analyzer:** FA always plays the single best counter deterministically. Weighted Random distributes proportionally, adding randomness that makes it harder to predict.

---

### 18. Punisher
**Type:** Punishment · **Complexity:** Medium

Plays randomly by default. But when it detects the opponent **repeating the same move 3 times in a row**, it enters "punish mode" for **10 rounds** — aggressively countering their most common recent move.

```
if opp[-1] == opp[-2] == opp[-3]:
  ENTER PUNISH MODE for 10 rounds
  during punish: play counter(most_common(opp[-10:]))

otherwise: play Random
```

---

### 19. Forgiver
**Type:** Punishment · **Complexity:** Medium

Same as Punisher, but **forgives after 3 rounds** instead of 10. Returns to random play faster, giving the opponent a chance to change behavior.

---

### 20. Chaos Strategy
**Type:** Randomized · **Complexity:** Simple

Each round, randomly picks one of 5 sub-strategies with equal probability:

| Roll | Strategy |
|------|----------|
| 0 | Play random |
| 1 | Counter opponent's last move |
| 2 | Mirror opponent's last move |
| 3 | Play cycle position (`R→P→S`) |
| 4 | Counter opponent's most common |

Maximally unpredictable but extremely inconsistent — no strategic coherence between rounds.

---

### 21. Decay Analyzer 🧠
**Type:** Analytical · **Complexity:** Medium

Like Frequency Analyzer (#9), but applies **exponential decay** so recent moves matter far more than old ones.

```
decay_factor = 0.9

weighted_count[move] = Σ (0.9^i) for each round i where opp played that move
                       (i=0 is most recent, i=1 is second-most-recent, etc.)

predicted = argmax(weighted_count)
play counter(predicted)
```

**Math — how fast does old data fade?**
```
Weight at time (now - k): 0.9^k

k=0:   weight = 1.000  (current round)
k=5:   weight = 0.590  (5 rounds ago)
k=10:  weight = 0.349  (10 rounds ago)
k=20:  weight = 0.122  (20 rounds ago)
k=50:  weight = 0.005  (50 rounds ago — effectively ignored)
```

**Half-life:** `0.9^k = 0.5 → k ≈ 6.6 rounds`. So the bot's "memory" has a half-life of ~7 rounds — it adapts to strategy switches within 10-15 rounds.

---

### 22. Historian 🧠
**Type:** Analytical · **Complexity:** High

Inspired by **Lempel-Ziv (LZ77) compression**. Looks at the last N moves as a "context", searches for that exact context earlier in history, and predicts what the opponent played *immediately after* that previous occurrence.

```
Round t:
  for ctx_len in [10, 9, 8, ..., 1]:
    context = opp[-ctx_len:]
    search for context in opp[0 : t-ctx_len]
    if found at position i AND (i + ctx_len) < t:
      predicted = opp[i + ctx_len]
      play counter(predicted)
      STOP
  fallback: counter(opp[-1])
```

**Example:** Opponent history = `[R,P,S,R,R,P,S,R,P]`. Current context (last 3) = `[S,R,P]`. This appeared before at positions 2-4 → after it came `R` (position 5). Predict Rock → play Paper.

**Why longest match?** Longer context = more specific = higher confidence prediction. A 1-move context is nearly useless. A 5-move match is very strong evidence.

---

### 23. Reverse Psychologist 🎭
**Type:** Psychology · **Complexity:** Simple

Applies **2nd-order reasoning** — assumes the opponent is trying to predict and counter YOUR last move:

```
Level 0: "I played Rock"
Level 1: "Opponent thinks I'll play Rock again"  
Level 2: "Opponent will play Paper (counter of Rock)"
My move: "I play Scissors (counter of Paper)"

Formula: play counter(counter(my[-1]))
```

**The math simplifies to:** `counter(counter(X))` = the move that **loses** to X.
- `counter(counter(Rock))` = `counter(Paper)` = `Scissors`
- `counter(counter(Paper))` = `counter(Scissors)` = `Rock`
- `counter(counter(Scissors))` = `counter(Rock)` = `Paper`

So effectively: the Reverse Psychologist always plays the move that **loses** to its own previous move. This creates a backward cycle: `R → S → P → R → S → P → ...`

**Beats:** Anti-Tit-for-Tat, Last-Move Counter (they counter our last → we counter their counter)
**Loses to:** Mirror opponent (they copy us → counter(counter(X)) loses to copy of X)

---

### 24. Echo 🎭
**Type:** Reactive · **Complexity:** Simple

Copies the opponent's move from **3 rounds ago** instead of last round.

```
delay = 3
Round t: play opp[t - 3]    (if t >= 3, else random)
```

**Why delay=3?** Against a Cycle bot (`R→P→S→R→...` with period 3), echoing 3 rounds back perfectly mirrors the current round. Against strategies that only look at `opp[-1]`, the 3-round lag creates confusion — they can't easily model what Echo will play next because it depends on distant history.

---

### 25. Trojan Horse 🎪
**Type:** Deception · **Complexity:** Medium

A classic **bait-and-switch** in two phases:

**Phase 1 — Bait (rounds 0-29):** Plays `Rock` every round, deliberately. This makes it look like "Always Rock" to any adaptive opponent.

**Phase 2 — Exploit (rounds 30+):** Any smart opponent has by now adapted to counter "Always Rock" by playing Paper. The Trojan suddenly switches to countering the opponent's recent moves (last 10 rounds).

```
if round < 30:
  play Rock                              ← deliberate bait
else:
  recent = most_common(opp[-10:])
  play counter(recent)                   ← exploit their adaptation
```

**Expected dynamics:**
- Rounds 0-29: Opponent adapts → starts playing Paper heavily
- Round 30: Trojan predicts Paper → plays Scissors → wins
- The opponent is caught off-guard by the sudden strategy shift

**Trade-off:** Trojan sacrifices ~30 rounds of potential losses to set up long-term exploitation. In a 1000-round match, the 970 exploitation rounds easily outweigh the 30-round investment.

---

### 26. Reluctant Gambler 🧮
**Type:** Statistical · **Complexity:** High

Plays **pure random** until it has statistical evidence (p < 0.05) that the opponent is biased. Uses a chi-squared test.

```
Round t:
  window = opp[-100:]    (last 100 moves, or all if < 100)
  n = len(window)
  expected = n / 3       (expected count per move if uniform)

  χ² = Σ (observed_i - expected)² / expected    for i in {R, P, S}

  if χ² > 5.991:    ← critical value for df=2, α=0.05
    # Statistically significant bias detected
    play counter(most_common(window))
  else:
    play Random     ← not enough evidence, stay safe
```

**The math:** Under the null hypothesis ("opponent plays uniformly"), χ² follows a chi-squared distribution with 2 degrees of freedom. The critical value at p=0.05 is **5.991**.

**Example:** In 100 rounds, opponent plays Rock 45, Paper 30, Scissors 25.
```
expected = 100/3 = 33.33
χ² = (45-33.33)²/33.33 + (30-33.33)²/33.33 + (25-33.33)²/33.33
   = 4.08 + 0.33 + 2.08 = 6.49

6.49 > 5.991 → SIGNIFICANT → exploit by playing Paper (counters Rock)
```

**Why it's clever:** Most bots blindly exploit from round 1. Reluctant Gambler waits for mathematical certainty, avoiding premature exploitation of what might just be random noise.

---

### 27. Entropy Guardian 🧮
**Type:** Statistical · **Complexity:** High

Monitors its **own move distribution entropy**. If it's becoming too predictable, forces random play to stay unreadable.

```
Round t:
  window = my[-30:]    (last 30 of own moves)
  counts = count each move in window
  n = len(window)

  H = -Σ (c_i/n) × log₂(c_i/n)     ← Shannon entropy

  H_max = log₂(3) ≈ 1.585 bits      ← maximum entropy (uniform distribution)

  if H < 0.7 × H_max:               ← below 70% of max entropy
    play Random                       ← "I'm too predictable, randomize!"
  else:
    play counter(most_common(opp[-20:]))   ← safe to exploit
```

**Entropy values:**
| Own distribution | Entropy H | Action |
|---|---|---|
| (33%, 33%, 33%) | 1.585 bits (max) | Exploit |
| (50%, 30%, 20%) | 1.485 bits (94%) | Exploit |
| (70%, 20%, 10%) | 1.157 bits (73%) | Exploit |
| (80%, 10%, 10%) | 0.922 bits (58%) | ⚠️ RANDOM (too predictable) |
| (100%, 0%, 0%) | 0.000 bits (0%) | ⚠️ RANDOM (fully predictable) |

---

### 28. Second Guess 🎭
**Type:** Psychology · **Complexity:** Medium

Like Reverse Psychologist, but **verifies whether the opponent is actually countering** before committing to 2nd-order logic.

```
Track: how often does opp[t] == counter(my[t-1])?

opponent_counter_rate = counter_count / total_checked

if opponent_counter_rate > 40% AND total_checked > 10:
  # Opponent IS countering us → use 2nd-order thinking
  predicted = counter(my[-1])     ← what they'll play
  play counter(predicted)          ← beat their counter
else:
  # Opponent is NOT countering → fall back to frequency analysis
  play counter(most_common(opp[-20:]))
```

**Advantage over Reverse Psychologist:** RP blindly assumes 2nd-order thinking every round. Second Guess checks the evidence first — if the opponent isn't actually countering, it uses a different (more appropriate) strategy.

---

### 29. Majority Rule 🧮
**Type:** Ensemble · **Complexity:** Medium

Runs **5 independent strategies** simultaneously each round and takes a **majority vote**:

| Voter | Strategy |
|-------|----------|
| 1 | `counter(opp[-1])` — counter last move |
| 2 | `opp[-1]` — mirror last move |
| 3 | `counter(most_common(opp_history))` — frequency counter |
| 4 | Anti-cycle (detects stepping patterns in opponent moves) |
| 5 | Random wildcard |

```
Round t:
  votes = [vote_1, vote_2, vote_3, vote_4, vote_5]
  play mode(votes)    ← most common vote wins
```

**Example:** Votes = [Paper, Rock, Paper, Scissors, Paper] → Paper wins with 3/5 → play Paper.

**Why it works (Condorcet jury theorem):** If each voter is correct more than 50% of the time, the majority vote is correct more often than any individual voter. The ensemble smooths out individual strategy weaknesses.

---

### 30. Phase Shifter 🎪
**Type:** Deception · **Complexity:** Medium

Alternates between two **distinct phases** in a repeating 60-round cycle:

**Aggressive phase (rounds 0-39 of each cycle):**
Looks at the opponent's last 15 moves, finds their most common move, and plays the counter. This is "aggressive" because it's **actively analyzing and exploiting** the opponent — committing fully to a predicted counter-move rather than playing safe.

```
Aggressive: play counter(most_common(opp[-15:]))
```

**Defensive phase (rounds 40-59 of each cycle):**
Plays **purely random** — no analysis, no exploitation. This is "defensive" because it **stops being predictable**, effectively resetting any model the opponent has built of Phase Shifter's behavior.

```
Defensive: play Random
```

```
Cycle: |---- Aggressive (40 rounds) ----|-- Defensive (20 rounds) --|
       0                                39 40                     59
       |---- Aggressive (40 rounds) ----|-- Defensive (20 rounds) --|
       60                               99 100                   119
```

**Why the asymmetry (40/20)?** Exploitation is only valuable if you do enough of it to accumulate wins. 40 rounds of exploitation offsets the "wasted" 20 random rounds. But those 20 random rounds prevent the opponent from locking onto a counter-strategy.

---

### 31. De Bruijn Walker 🎪
**Type:** Anti-pattern · **Complexity:** Medium

Follows a **De Bruijn sequence** — a cyclic sequence where every possible n-gram of length k appears exactly once.

For our alphabet {R, P, S} (size 3) and n=2, there are 3² = 9 possible 2-grams: `RR, RP, RS, PR, PP, PS, SR, SP, SS`. The De Bruijn sequence packs all 9 into a cycle of length 9:

```
Sequence: R R P R S P P S S
          └─┘ ← RR
            └─┘ ← RP
              └─┘ ← PR (wraps)
                └─┘ ← RS
                  └─┘ ← SP
                    └─┘ ← PP
                      └─┘ ← PS
                        └─┘ ← SS
                          └─┘ ← SR (wraps to start)
```

**Why it defeats pattern detectors:** Pattern detectors search for repeated subsequences. In a De Bruijn sequence, no 2-gram repeats within one cycle, so there are no repeating patterns to find.

**Occasional exploitation (10%):** With 10% probability, deviates from the sequence to exploit a strong opponent bias (> 50% one move).

---

### 32. Iocaine Powder 🧠
**Type:** Meta-reasoning · **Complexity:** Very High

Named after the battle of wits in *The Princess Bride* ("I know that you know that I know..."). Runs **6 meta-strategies** that each make a prediction about the opponent's next move, then follows whichever predictor has been most accurate recently.

**The 6 predictors:**

| # | Name | Prediction |
|---|------|-----------|
| 1 | Naive repeat | `opp[-1]` (they repeat last move) |
| 2 | Counter-me | `counter(my[-1])` (they counter my last) |
| 3 | Double bluff | `counter(counter(opp[-1]))` (they go one level deeper) |
| 4 | Mirror counter | `counter(counter(opp[-1]))` from THEIR perspective |
| 5 | Deep counter | `counter(counter(counter(my[-1])))` |
| 6 | Frequency | Most common of `opp[-10:]` |

**Scoring with exponential decay:**
```
After each round:
  for each predictor i:
    score[i] *= 0.95              ← decay old scores
    if prediction[i] == actual_opp_move:
      score[i] += 1.0            ← reward correct prediction

  best = argmax(scores)
  play counter(predictions[best])
```

**Why it's powerful:** It automatically adapts to whatever reasoning level the opponent is on. If the opponent is naive (repeats moves), predictor #1 scores highest. If the opponent is clever (counters your last), predictor #2 dominates. The decay factor (0.95) lets it shift between predictors as the opponent changes strategy.

---

### 33. Intentional Loser 💀
**Type:** Special · **Complexity:** Simple

Deliberately tries to **lose every round**. Predicts the opponent's next move using frequency analysis, then plays the move that **loses** to it.

```
Round 0: play Scissors     ← most openings are Rock, so Scissors loses
Round t:
  predicted = most_common(opp_history)
  play loses_to(predicted)       ← the move that LOSES to the prediction
```

**Move mapping (opposite of counter):**

| Predicted opponent move | Losing move (what we play) |
|---|---|
| Rock | Scissors (Rock beats Scissors) |
| Paper | Rock (Paper beats Rock) |
| Scissors | Paper (Scissors beats Paper) |

**Why include it?** Two reasons:
1. **Sanity check:** Any algorithm that manages to lose to Intentional Loser has a serious bug.
2. **Floor baseline:** It defines the absolute worst possible performance. Your custom algorithm should at least beat the bot that's *trying to lose*.

---

### 34. Q-Learner 🧠🎰
**Type:** Reinforcement Learning · **Complexity:** High

Tabular **Q-Learning** — the classic RL algorithm. Maintains a table of Q-values: `Q(state, action) → expected reward`. Learns which move to play in each situation through trial and error.

**State space (v4):** `(my[-1], opp[-1], opp[-2], last_outcome)` → 81 states + warm-up = **243+ Q-values**. **Experience replay:** stores 200 transitions, replays 10 random ones per round for multi-pass learning (DQN-inspired). **Pre-trained** via 120-round self-play against 5 archetypes.

**Learning rule (after each round):**
```
reward = +1.0 (win), 0.0 (draw), -1.0 (loss)

Q(s, a) ← Q(s, a) + α × (reward - Q(s, a))

α = 0.3 (learning rate)
ε = 0.3 × 0.99^t (exploration, decays from 30% → 5%)
```

**Action selection:** ε-greedy — with probability ε play random (explore), otherwise play `argmax_a Q(s, a)` (exploit).

**Convergence:** After ~50-100 rounds, the Q-table converges. Against Always Rock, it learns `Q((*, Rock), Paper) ≈ 1.0` and achieves **94.6% win rate**.

**Epsilon decay schedule:**
```
Round 0:   ε = 0.300  (30% random exploration)
Round 50:  ε = 0.182  (18%)
Round 100: ε = 0.110  (11%)
Round 200: ε = 0.050  (5% floor — never fully stops exploring)
```

---

### 35. Thompson Sampler 🧠🎰
**Type:** Bayesian Bandit · **Complexity:** High

Bayesian multi-armed bandit using **Beta-Bernoulli model**. For each (state, action) pair, maintains a Beta distribution encoding belief about the win probability.

**Per (state, action):**
```
Beta(α, β)  where:
  α = 1 + wins       (successes + prior)
  β = 1 + losses     (failures + prior)
  Prior: Beta(1, 1) = Uniform(0, 1)
```

**Decision rule (each round):**
```
For current state s:
  For each action a ∈ {R, P, S}:
    sample θ_a ~ Beta(α_sa, β_sa)
  Play action with highest θ_a sample
```

**Why it's powerful (vs ε-greedy):** Thompson Sampling doesn't need a tuned ε parameter. The Beta posterior **automatically** narrows as evidence accumulates — wide posteriors (uncertain) produce diverse samples (exploration), narrow posteriors (confident) produce consistent samples (exploitation).

**Regret bound:** `O(√(KT log T))` where K=3 actions, T=rounds.

**Example evolution:**
```
Round 10:  Beta(3, 2) for Paper vs Rock → wide, explores
Round 50:  Beta(20, 3) for Paper vs Rock → narrow, exploits Paper
Round 200: Beta(80, 5) for Paper vs Rock → locked on Paper (~94%)
```

---

### 36. UCB Explorer 🧠🎰
**Type:** Bandit (UCB1) · **Complexity:** High

Upper Confidence Bound — picks the action with the highest **optimistic estimate**, combining the average reward with an exploration bonus.

**UCB1 formula:**
```
UCB(s, a) = Q̄(s, a) + c × √(ln(N_s) / n_sa)

Where:
  Q̄(s, a) = average reward of action a in state s
  N_s     = total visits to state s
  n_sa    = times action a was played in state s
  c       = √2 ≈ 1.414 (exploration constant)
```

**How the exploration bonus works:**
```
          Average     Times    Bonus        UCB
Rock:     0.50        100      1.414×√(ln(300)/100) = 0.217    0.717
Paper:    0.80          5      1.414×√(ln(300)/5)   = 1.482    2.282 ← selected!
Scissors: 0.20        195      1.414×√(ln(300)/195) = 0.155    0.355
```

Paper has only been tried 5 times but has a high average → the exploration bonus inflates its UCB → it gets explored. Once it accumulates data, the bonus shrinks and the average dominates.

**Theoretical guarantee:** O(log T) regret — mathematically optimal for bandits.

---

### 37. Gradient Learner 🧠🎰
**Type:** Policy Gradient · **Complexity:** High

Softmax **policy gradient** without neural networks. Unlike Q-learning (which learns deterministic greedy policies), Gradient Learner can learn **stochastic (mixed) strategies** — crucial in RPS where the Nash equilibrium IS a mixed strategy.

**Policy:**
```
π(a|s) = exp(h(s,a)) / Σ_b exp(h(s,b))     ← softmax over preferences
```

**Gradient update (after each round):**
```
advantage = reward - baseline
baseline  = running average of all rewards

For the chosen action a:
  h(s, a) += α × advantage × (1 - π(a|s))

For all other actions b ≠ a:
  h(s, b) -= α × advantage × π(b|s)

α = 0.1
```

**Why it learns mixed strategies:** If Rock wins sometimes and Paper wins sometimes (because opponent is random), the preferences for both stay elevated, and the softmax outputs ~33%/33%/33% — the Nash equilibrium. Q-learning would oscillate between deterministic choices.

---

### 38. Bayesian Predictor 🧮
**Type:** Bayesian Statistics · **Complexity:** Medium

Maintains a **Dirichlet prior** over the opponent's move distribution. The Dirichlet is the conjugate prior for categorical distributions — updates are trivial.

```
Prior: Dir(1, 1, 1)     ← uniform (no bias assumed)

After observing opponent's last 50 moves:
  n_R, n_P, n_S = counts in window

Posterior: Dir(1 + n_R, 1 + n_P, 1 + n_S)

Sample from posterior (via Gamma trick):
  g_R ~ Gamma(1 + n_R, 1),  g_P ~ Gamma(1 + n_P, 1),  g_S ~ Gamma(1 + n_S, 1)
  prob_R = g_R / (g_R + g_P + g_S)
  ...

Play counter(argmax(sampled probs))
```

**Why sample instead of using the posterior mean?** Sampling adds **exploration** — when the posterior is uncertain (few observations), samples vary widely, preventing premature commitment.

---

### 39. N-Gram Predictor 🧮
**Type:** Sequence Modeling · **Complexity:** High

Builds n-gram models over the **joint history** of both players' moves — `(my_move, opp_move)` pairs — rather than just opponent moves alone.

```
Joint history: [(R,P), (S,R), (P,P), (R,P), (S,R), ...]

For n = 3, 2, 1 (longest first):
  context = last n pairs
  Count what opponent played after this context in the past
  If matches found: predict most common continuation → counter it
```

**Key advantage over Pattern Detector (#11):** Captures **interactive** patterns. Pattern Detector only sees the opponent's moves: `[P, R, P, P, R, ...]`. N-Gram Predictor sees the conversation: `[(R,P), (S,R), ...]` and notices things like “when I play Rock and they play Paper, they usually follow with Scissors.”

---

### 40. Anti-Strategy Detector 🧮
**Type:** Meta-analysis · **Complexity:** High

Maintains **5 archetype detectors**, each hypothesizing a different opponent strategy. Scores each detector by recent predictive accuracy (with exponential decay) and follows the best one.

| Detector | Hypothesis | Counter |
|----------|-----------|--------|
| Constant | Opponent repeats last move | Play counter of their last |
| Cycle | Opponent follows R→P→S | Play 2 steps ahead in cycle |
| Mirror | Opponent copies MY last move | Play counter of my last |
| Counter | Opponent counters MY last move | Play counter of counter of my last |
| Frequency | Opponent counters my most common | Vary own play |

**Scoring (each round):**
```
For each detector i:
  score[i] *= 0.9                  ← exponential decay
  if detector i correctly predicted opp[-1]:
    score[i] += 1.0

best_detector = argmax(scores)
play counter(best_detector's current prediction)
```

**Strength:** Instead of generic frequency analysis, this bot applies **targeted counters**. If it detects the opponent is mirroring you, it doesn't just counter-last — it plays the specific move that beats the mirror of its own last move.

---

### 41. Mixture Model 🧮
**Type:** Ensemble (Hedge algorithm) · **Complexity:** High

Runs **5 expert strategies** and dynamically adjusts their weights using the **multiplicative weights update** (a.k.a. Hedge algorithm).

**Experts:**

| # | Expert | Strategy |
|---|--------|----------|
| 0 | Counter-last | `counter(opp[-1])` |
| 1 | Frequency | `counter(most_common(opp))` |
| 2 | Markov | `counter(transition_predict(opp[-1]))` |
| 3 | WSLS | Win-Stay, Lose-Shift |
| 4 | Random | Uniform random |

**Weight update (after each round):**
```
For each expert i:
  loss_i = 0.0 (if expert would have won)
           0.5 (if draw)
           1.0 (if expert would have lost)

  w_i ← w_i × (1 - η × loss_i)

Normalize: w_i ← w_i / Σ_j w_j
η = 0.15
```

**Action selection:** Each expert votes for a move, votes are weighted by expert weights. Move with highest total weight wins.

**Theoretical guarantee:** Hedge achieves **regret ≤ O(√(T log K))** where K = 5 experts. After T rounds, the mixture performs nearly as well as the **best single expert in hindsight** — without needing to know which expert is best in advance.

---

### 42. Sleeper Agent 🕵️
**Type:** Deception / Delayed Exploitation · **Complexity:** High

Plays **pure random for 80 rounds**, silently collecting clean opponent data. Then activates a multi-predictor ensemble.

```
Rounds 0-79: DORMANT — play uniformly random
  └─ Opponent faces what looks like Pure Random
  └─ Their behavior is "natural" (unaffected by adversarial pressure)

Rounds 80+: ACTIVE — exploit with ensemble:
  1. Frequency prediction (window of 50)
  2. Markov prediction (transition from last move)
  3. Pattern prediction (last 3 moves)
  └─ Majority vote → counter the predicted move
```

**Why it works:** Most algorithms adapt to their opponent. During the dormant phase, the opponent is adapting to "Pure Random" (which has nothing to exploit). The dormant data reveals the opponent's **true biases** — biases they can't hide because they think they're facing noise.

---

### 43. Shapeshifter 🦎
**Type:** Anti-modeling · **Complexity:** Medium

Cycles through **5 completely different strategies every 40 rounds**, making it nearly impossible for pattern-based opponents to build a model.

```
Rounds  0-39:  Pure Random (unreadable)
Rounds 40-79:  Counter-last (reactive)
Rounds 80-119: Frequency counter (statistical)
Rounds 120-159: Markov predictor (sequential)
Rounds 160-199: Win-Stay Lose-Shift (adaptive)
Rounds 200+:   cycle repeats...
```

**Design rationale:** By the time an opponent accumulates ~40 rounds of data on one strategy, Shapeshifter switches to something completely different. The opponent's model becomes stale immediately.

---

### 44. Hot Streak 🔥
**Type:** Momentum-based · **Complexity:** Medium

Rides winning streaks and retreats during losing streaks.

```
If win streak ≥ 2: repeat last move    ("it's working, don't change")
If lose streak ≥ 3: play random         ("reset the pattern")
Otherwise: frequency counter (window=20) ("play smart")
```

**Psychology:** In RPS, repeating a winning move is often effective because humans (and frequency-based algorithms) expect you to switch. Hot Streak exploits this expectation.

---

### 45. Markov Generator 🎲
**Type:** Randomized · **Complexity:** High

Generates moves based on a **random internal Markov chain**. Does **not** look at the opponent at all.

1.  Current state: `last_idx` (0, 1, or 2).
2.  Matrix `M[3][3]`: Randomly generated probabilities of moving from state `i` to `j`.
3.  Next move: Sampled from `M[last_idx]`.

**Resets:** Every 50 rounds, it generates a **new random matrix**.

**Why it works:** It produces "structured noise". It's not pure random (it has patterns), but the patterns change every 50 rounds and are unrelated to the game state. This can confuse advanced pattern predictors that try to find a logic where none exists.

---

### 46. Monte Carlo Predictor 🎲
**Type:** Simulation-based · **Complexity:** Medium

Builds a transition model from observed opponent moves, then runs **50 Monte Carlo simulations** to estimate the next move probability.

```
Transition model: P(next | current=Rock) = {Rock: 5, Paper: 12, Scissors: 3}

50 simulations:
  Each sim: randomly sample from transitions[opp[-1]]

Result: Paper appeared 30/50 times → predict Paper → play Scissors
```

**Advantage over deterministic Markov:** The Monte Carlo sampling adds noise that prevents opponents from reverse-engineering our prediction and countering it.

---

### 47. Grudge Holder 😤
**Type:** Emotional / memory-based · **Complexity:** Medium

Maintains a **grudge score** for each of our own moves. Moves that have historically led to losses accumulate grudge points. The algorithm avoids high-grudge moves.

```
Grudges: {Rock: 5, Paper: 1, Scissors: 8}

Best moves (min grudge): [Paper]  ← grudge score 1
If frequency counter agrees with Paper → play Paper
Otherwise → random among best moves
```

**Wins slowly forgive:** When a move wins, its grudge score decreases by 1. This means historically bad moves can rehabilitate over time if the opponent changes strategy.

---

### 48. Chameleon 🦎
**Type:** Distribution mirroring · **Complexity:** Simple

Copies the **opponent's exact move distribution** (not their specific moves).

```
Opponent's last 50: Rock=25, Paper=15, Scissors=10
Opponent plays:     50% R, 30% P, 20% S
Chameleon plays:    50% R, 30% P, 20% S  ← matching distribution
```

**Against biased opponents:** Forces many draws (same distribution → both pick the same move more often). The draw rate goes up while the loss rate drops.

**Against random:** Becomes random too — can't be exploited.

---

### 49. Fibonacci Player 🌀
**Type:** Mathematical sequence · **Complexity:** Medium

Uses the **Fibonacci sequence mod 3** as a base pattern — a period-8 cycle that's harder for pattern detectors to identify than simple R→P→S.

```
Fibonacci mod 3: [0, 1, 1, 2, 0, 2, 2, 1] → period 8
Mapped to moves: [R, P, P, S, R, S, S, P]

Mix: 70% Fibonacci pattern, 30% frequency counter
```

**Why period 8?** Simple cycle detectors look for period-3 patterns. The Fibonacci mod 3 has period 8. And the 30% frequency exploitation means even if the pattern is detected, the opponent can't fully exploit it.

---

### 50. Lempel-Ziv Predictor 🧮📐
**Type:** Information Theory \u00b7 **Complexity:** High

Based on the deep connection from information theory: **a good compressor IS a good predictor**. Uses the LZ78 compression algorithm to build a dictionary of observed opponent subsequences.

```
Dictionary (incrementally built):
  () → {R:12, P:8, S:5}       ← root: overall frequency
  (R,) → {P:6, R:2, S:1}     ← after seeing R
  (R,P) → {S:4, R:1}         ← after seeing R,P
  (R,P,S) → {R:3}            ← after R,P,S → always R

Current phrase: (R, P)
Prediction: S (most common continuation) → play Rock
```

**Why compression = prediction (Shannon's insight):** If a sequence has low Kolmogorov complexity (it's compressible), there's structure to exploit. LZ78 naturally discovers this structure. If the sequence is incompressible (random), the dictionary stays shallow and we fall back to frequency.

---

### 51. Context Tree 🧮📐
**Type:** Bayesian Universal Prediction \u00b7 **Complexity:** Very High

**Provably optimal** universal sequence predictor. An upgraded N-Gram that computes a proper **Bayesian mixture over ALL context depths** (0 to 6) simultaneously, using the **Krichevsky-Trofimov (KT) estimator** and **CTW prior**.

```
Depth 0: P(next) from all observations        weight = 0.5^0 \u00d7 n
Depth 1: P(next | opp[-1])                    weight = 0.5^1 \u00d7 n
Depth 2: P(next | opp[-2], opp[-1])            weight = 0.5^2 \u00d7 n
...
Depth 6: P(next | opp[-6]...opp[-1])           weight = 0.5^6 \u00d7 n

KT estimator: P(m) = (count(m) + 0.5) / (total + 1.5)

Final prediction = weighted average across all depths
```

**Dual context system:** Uses BOTH opponent-only contexts AND joint (my, opp) pair contexts with 1.5× bonus weight for joint matches — capturing interactive patterns the original N-Gram finds, but with optimal depth selection.

---

### 52. Max Entropy Predictor 🧮📐
**Type:** Statistical Mechanics / MaxEnt \u00b7 **Complexity:** High

Based on **Jaynes' Maximum Entropy principle** — the foundation of statistical mechanics. Finds the probability distribution with **maximum entropy** (least bias) subject to 3 observed constraints.

```
Feature 1: Marginal frequency (window=40)
  f₁(m) = (count(m) + 1) / (total + 3)

Feature 2: 1st-order transition P(next | opp[-1])
  f₂(m) = (transition_count(m) + 1) / (trans_total + 3)

Feature 3: 2nd-order transition P(next | opp[-2], opp[-1])
  f₃(m) = (bigram_count(m) + 1) / (bigram_total + 3)

MaxEnt log-linear model:
  P(m) \u221d f₁(m)^0.3 \u00d7 f₂(m)^0.4 \u00d7 f₃(m)^0.3
```

**Why MaxEnt?** It's the **only** distribution that uses ALL the information from constraints without adding ANY assumptions beyond them. It's the "maximally honest" prediction.

---

### 53. Poison Pill 💊
**Type:** Trojan / Deception \u00b7 **Complexity:** Medium

Deliberately **plants a bias** to manipulate the opponent's model, then **exploits their adaptation**.

```
90-round cycles (3 phases × 30 rounds):

Phase 1: Plant 85% Rock bias
  → Opponent detects Rock bias...
  → Opponent adapts to play Paper...

Phase 2: Exploit with 85% Scissors
  → Their Paper loses to our Scissors!
  → 30 rounds of exploitation before they adjust

Phase 3: Plant next poison (Paper bias)
  → Rotation ensures each 90-round cycle uses different bait

Poison rotation: R→P→S shifts each super-cycle to prevent meta-detection
```

**Key timing:** Most algorithms need 15-25 rounds to detect bias. Poison Pill changes every 30 — always one step ahead.

---

### 54. Mirror Breaker 🪞💥
**Type:** Counter-reactive / Trojan \u00b7 **Complexity:** High

**Specifically designed** to create and exploit feedback loops against reactive strategies. Achieved **100% win rate** vs Mirror Opponent in testing.

```
Phase 1 (rounds 0-19): Diagnostic sequence (R,P,S,R,P,S...)
  → Observe how opponent reacts

Phase 2: Detection
  If opponent copies my previous move (mirror):
    → Play counter(my_last) forever — they copy my last,
      I play what beats my last = what beats their next move

  If opponent counters my previous move:
    → Play counter(counter(my_last)) — one step deeper

  Otherwise: fall back to Markov prediction
```

**Why it works:** Reactive opponents create a **deterministic feedback loop**. Once Mirror Breaker identifies the loop, it's solved — the opponent's next move is completely determined by our last move, which we control.

---

### 55. The Usurper 👑
**Type:** Meta-strategy / Identification \u00b7 **Complexity:** Very High

Identifies the opponent's **strategy archetype** using 6 parallel detectors with exponential decay scoring, then becomes a **strictly better version** of that strategy. Achieved **99% win rate** vs Cycle.

| Archetype Detected | What They Do | Usurper's Counter |
|---|---|---|
| Constant | Repeat last | `counter(opp[-1])` |
| Cycle | R→P→S | Play 2 ahead in cycle |
| Mirror | Copy my[-1] | `counter(my[-1])` |
| Counter | `counter(my[-1])` | `counter(counter(my[-1]))` |
| Frequency | Counter my most common | Play my LEAST common |
| Random | Uniform | Markov + frequency combo |

**Difference from Anti-Strategy Detector (#40):** ASD counter-predicts individual moves. The Usurper counter-STRATEGIZES — it exploits the **mathematical weakness** of the entire strategy class.

---

### 56. Double Bluff 🃏
**Type:** Multi-level reasoning \u00b7 **Complexity:** High

Implements **recursive reasoning** with 3 levels (which is complete since R→P→S has period 3):

```
Level 0: counter(their_most_common)              — "I predict them"
Level 1: counter(counter(their_most_common))      — "They predict level 0"
Level 2: counter(counter(counter(their_common)))   — "They predict level 1"

Note: Level 3 = Level 0 (mod 3 cycle)
```

**Adaptive depth:** Tracks which level would have won each round (with 0.9 decay). Follows the best-performing level. Against naive opponents, Level 0 dominates. Against sophisticated opponents who model our frequency counter, Level 1 kicks in.

---

### 57. Frequency Disruptor 📡
**Type:** Signal Jamming / Deception \u00b7 **Complexity:** Medium

Creates **deliberate false signals** to corrupt opponents' frequency models, then exploits the resulting confusion.

```
30-round cycles:

Phase 1 (rounds 0-19): Broadcast fake pattern
  70% pattern (e.g., heavy Rock), 30% random noise
  → Opponent's frequency model locks onto Rock bias

Phase 2 (rounds 20-24): Exploit
  85% counter(counter(dominant_fake))
  → We planted heavy Rock → they adapted to Paper
  → counter(counter(Rock)) = counter(Paper) = Scissors
  → Our Scissors destroys their Paper adaptation

Phase 3 (rounds 25-29): Real analysis
  Frequency counter on their ACTUAL play (revealed during Phase 2)

Fake patterns rotate: Heavy-R → Heavy-P → Heavy-S
  → Each 30-round cycle uses a different poison
```

---

### 58. Deep Historian 📖🔬
**Type:** Upgraded Historian \u00b7 **Complexity:** High

An upgraded version of Historian (#22) with 4 major improvements:

```
Original Historian: fixed length-4, opponent-only patterns, no weighting
Deep Historian:     variable length 2-5, joint (my,opp) patterns, recency decay

Pattern matching:
  Current context: [(my, opp)] pairs from last 5 rounds
  Search all history for matching joint patterns
  Weight matches: weight = 0.95^age (recent = higher)
  Predict: argmax of weighted continuation counts

Try lengths: 5 → 4 → 3 → 2 (longest match wins)
  Only predict if total_weight > 0.5 (enough confidence)
```

**Key upgrade:** Joint patterns capture **interactive** dynamics — e.g., "whenever I played Rock and they played Scissors, then I played Paper and they played Rock, they next play Scissors." Original Historian only sees opponent moves.

---

### 59. Adaptive N-Gram 📊🔬
**Type:** Upgraded N-Gram Predictor \u00b7 **Complexity:** Very High

An upgraded version of N-Gram Predictor (#39) with meta-learning:

```
Improvements over original N-Gram:
  1. Dynamic context: tries n=5,4,3,2,1 (original: 3,2,1)
  2. Decay-weighted: weight = 0.9^age (recent transitions 3× heavier)
  3. Accuracy tracking: learns which n works best per opponent
  4. Joint (my,opp) contexts overlaid on opponent-only contexts

Meta-learner:
  accuracy[n] *= 0.95                    (decay)
  if prediction from n was correct:
    accuracy[n] += 1.0                   (reward)

  best_n = argmax(accuracy[n] × confidence[n])
```

**vs Context Tree (#51):** Context Tree uses Bayesian CTW theory with a fixed weighting prior. Adaptive N-Gram uses **empirical accuracy tracking** — it learns which context length works best for THIS specific opponent.

---

### 60. Regret Minimizer ♠\ufe0f
**Type:** Game Theory / Online Learning \u00b7 **Complexity:** High

**Regret Matching** — the core algorithm behind **Libratus and Pluribus**, the AIs that beat world champions at poker.

```
After each round:
  For each possible move m:
    regret(m) += payoff(m, opp_move) - payoff(my_actual_move, opp_move)

Strategy:
  If any regret > 0:
    strategy(m) = max(0, regret(m)) / Σ max(0, regret(m'))
  Else:
    strategy = uniform random (1/3, 1/3, 1/3)

Play: sample from strategy distribution
```

**Theoretical guarantee:** Converges to **Nash equilibrium** — in RPS, that's (1/3, 1/3, 1/3). Against a Nash-playing opponent, it draws. Against a non-Nash opponent, it **exploits their deviations**. Average regret goes to 0 as T → ∞.

---

### 61. Fourier Predictor 📐🎵
**Type:** Signal Processing / Frequency Analysis \u00b7 **Complexity:** Very High

Applies the **Discrete Fourier Transform (DFT)** to detect hidden periodic patterns in the opponent's move sequence.

```
Encode moves: R=0, P=1, S=2 → signal x[n]

DFT: X[k] = Σ x[n] × e^(-2πi·k·n/N)  for k = 1..N/2

Steps:
  1. Window last 64 moves
  2. Compute DFT manually (no numpy)
  3. Find top 3 dominant frequency components
  4. Extrapolate signal to predict x[N]
  5. Map back to move and counter
```

**Why DFT works:** If the opponent has ANY periodic pattern (Cycle=period 3, Fibonacci=period 8, De Bruijn=period 27), the DFT will find the dominant frequency. Even noisy periodicity is detectable.

---

### 62. Eigenvalue Predictor 📐🔢
**Type:** Linear Algebra / Markov Analysis \u00b7 **Complexity:** High

Builds the opponent's 3\u00d73 **transition matrix** and uses **power iteration** to compute the dominant eigenvector (stationary distribution).

```
Transition matrix M:
  M[i][j] = P(opp plays j | opp played i)

Power iteration (10 steps):
  π₀ = [1/3, 1/3, 1/3]
  π(t+1) = M^T × π(t)
  → converges to stationary distribution π*

Prediction: 60% current-row + 40% stationary
  P(next) = 0.6 × M[opp[-1]] + 0.4 × π*
```

**vs Markov Predictor (#10):** Markov only uses the current transition row. Eigenvalue Predictor blends the transition with the **long-term stationary behavior**, capturing both what the opponent does AFTER their last move AND their overall bias.

---

### 63. Q-Learner v5 🧠📈
**Type:** RL / Linear Function Approximation · **Complexity:** Very High

Replaces the lookup table `Q(s,a)` with a **linear model** `Q(s,a) = w^T · φ(s,a)`, using a 16-dimensional feature vector. Features include action one-hot, opponent's last move, frequency bias, transition probabilities, last outcome, and bias. Updates via SGD with experience replay.

**Key advantage over v4:** Generalizes across similar states — "playing Paper when opponent is biased toward Rock" transfers to ALL states with Rock-biased opponents.

---

### 64. Thompson Sampler v5 🎲📈
**Type:** Bayesian RL / Linear Regression · **Complexity:** Very High

Bayesian linear regression over the 16-dimensional feature vector. Maintains per-action covariance matrices and posterior distributions. Samples weight vectors from the posterior via Cholesky decomposition for Thompson Sampling.

Uses pure-Python matrix operations (Gauss-Jordan inversion, Cholesky decomposition) — zero external dependencies.

---

### 65. UCB Explorer v5 🔭📈
**Type:** Contextual Bandits / LinUCB · **Complexity:** Very High

LinUCB (contextual bandits) with the 16-dimensional feature vector. UCB bonus: `w^T·φ + α·√(φ^T · A^{-1} · φ)`. Balances exploitation (predicted reward) with exploration (uncertainty in feature space).

---

### 66. Gradient Learner v5 📉📈
**Type:** Policy Gradient / REINFORCE · **Complexity:** Very High

Linear softmax policy over features. Per-action preference `h(a) = w_a^T · φ(s)`. Policy gradient via REINFORCE: `Δw_a = α · (r - baseline) · ∇ log π(a|s)`.

---

### 67. Hidden Markov Oracle 🔮🧬
**Type:** Hidden Markov Model (NLP/Speech) · **Complexity:** Very High

Assumes the opponent has **3 hidden internal states** (e.g., aggressive, defensive, random). Uses the **Forward algorithm** for state inference and **online Baum-Welch** to learn transition and emission matrices. Predicts by marginalizing over hidden states.

**Why it's powerful:** Detects implicit mode switches (Shapeshifter changes every 40 rounds, Sleeper Agent has dormant/active phases).

---

### 68. Genetic Strategist 🧬🦎
**Type:** Evolutionary Computation · **Complexity:** High

Maintains a **population of 20 strategy genomes** (response tables mapping `(opp[-1], opp[-2]) → Move`). Every 25 rounds: selection (keep top 10), crossover (swap random entries), mutation (5% per gene). Uses the fittest genome's response table.

**Key difference:** Adapts its **entire strategy structure**, not just parameters.

---

### 69. PID Controller 🎛️🔧
**Type:** Control Theory / Robotics · **Complexity:** Medium

Treats the game as a **feedback control problem**. P (proportional) adjusts based on current error, I (integral) corrects persistent bias, D (derivative) anticipates strategy changes. Control signal mapped to moves via softmax.

---

### 70. Chaos Engine 🌀🔥
**Type:** Nonlinear Dynamics / Chaos Theory · **Complexity:** Medium

Uses the **logistic map** `x_{n+1} = 3.99 × x_n × (1 - x_n)` in the fully chaotic regime. 70% chaos (deterministic but unpredictable), 30% frequency exploitation. Reseeds every 50 rounds using hash of recent outcomes.

---

### 71. Level-k Reasoner 🧠♟️
**Type:** Behavioral Economics / Cognitive Hierarchy · **Complexity:** High

From Nagel (1995) and Camerer (2003). Detects the opponent's **cognitive reasoning level** (0-4) by simulating what each level would play and comparing to actual moves. Then plays at level k+1 — one step above the opponent.

---

### 72. UCB-NGram Fusion ⚡🔀
**Type:** Hybrid / Meta-Strategy · **Complexity:** High

Fuses UCB bandit exploration with N-Gram prediction. Three layers: (1) Strategy layer (UCB, N-Gram, frequency), (2) Softmax selection weighted by rolling win rates + phase modifiers, (3) **Meta-prediction layer** that detects when our own moves become predictable and counter-rotates.

---

### 73. Iocaine Powder Plus 🧪⚗️
**Type:** Meta-Strategy / Ensemble · **Complexity:** High

Upgraded Iocaine Powder with **12 meta-strategies** (adds Markov, bigram, trigram counters + mirrors). Uses sliding-window scoring with faster exponential decay (0.92).

---

### 74. Dynamic Mixture 🔄🎯
**Type:** Ensemble / Hedge + Evolution · **Complexity:** High

Upgraded Mixture Model with **8 experts** (adds Markov-2, recent frequency, win-pattern). Features **expert pruning** (drop <25% accuracy after 100 rounds) and **expert spawning** (clone best expert every 200 rounds).

---

### 75. Hierarchical Bayesian 📊🔬
**Type:** Bayesian Statistics / Hierarchical Model · **Complexity:** High

Upgraded Bayesian Predictor that **learns its own prior** via evidence maximization. Features **change-point detection** (resets when KL divergence exceeds threshold) and **multi-window ensemble** (combines windows of 20, 50, 100 rounds).

---

### 76. Self-Model Detector 🔍🤖
**Type:** Strategy Identification / Self-Play · **Complexity:** High

Upgraded Anti-Strategy Detector with **10 candidate strategy simulations** (constant, cycle, mirror, counter, frequency, Markov, WSLS, anti-TFT, pattern cycle, decay frequency). Identifies which strategy the opponent most resembles and counters it.

---

## 📊 Elo Rating System



- **Starting Elo**: 1500 for all algorithms
- **K-factor**: 32 (standard competitive)
- **Match outcomes**: Win = 1.0, Draw = 0.5, Loss = 0.0
- **Formula**: Standard Elo expected score calculation

```
E(A) = 1 / (1 + 10^((R_B - R_A) / 400))
R'(A) = R(A) + K × (S(A) - E(A))
```

Elo ratings are computed **chronologically** across all matches in a tournament, so the order of play matters. Leaderboards are sorted by Elo.

---

## 🔧 Adding Custom Algorithms

1. Open `rps_playground/algorithms.py`
2. Create a new class:

```python
class MyAlgorithm(Algorithm):
    name = "My Algorithm"

    def choose(self, round_num, my_history, opp_history):
        # round_num: current round (0-indexed)
        # my_history: list of your past Move choices
        # opp_history: list of opponent's past Move choices
        # self.rng: seeded Random instance for reproducibility
        return Move.ROCK  # your logic here
```

3. Add it to the registry:

```python
ALL_ALGORITHM_CLASSES = [
    ...,
    MyAlgorithm,  # add here
]
```

4. It instantly appears in both the CLI and Web UI.

---

## 📁 Project Structure

```
rps_playground/
├── __init__.py          # Package init
├── engine.py            # Core: Move enum, winner logic, MatchResult, run_match
├── algorithms.py        # 76 algorithms + base class + registry
├── tournament.py        # 3 modes: head-to-head, one-vs-all, round-robin
├── stats.py             # Elo system, leaderboard, H2H matrix, pretty-print
├── export.py            # JSON and CSV export
├── main.py              # CLI entry point (argparse)
├── web.py               # Flask web server
└── templates/
    └── index.html       # Web UI frontend
```

---

## ⚙️ CLI Reference

```
python3 -m rps_playground.main <command> [options]

Commands:
  head-to-head   Mode 1: Algo A vs Algo B
  one-vs-all     Mode 2: Custom algo vs all 62 bots
  tournament     Mode 3: Full round-robin tournament

Global:
  --list         List all available algorithms

Options (all commands):
  --rounds N     Rounds per match (default: 1000)
  --seed S       RNG seed for reproducibility
  --export FMT   Export format: json | csv
  --output PATH  Export file path

head-to-head only:
  --algo-a NAME  Name of algorithm A
  --algo-b NAME  Name of algorithm B

one-vs-all only:
  --algo NAME    Name of your algorithm
```

---

## 🎲 Reproducibility

Use the `--seed` flag (CLI) or seed field (Web UI) to get **identical results** across runs:

```bash
# These two runs produce identical output
python3 -m rps_playground.main tournament --rounds 1000 --seed 42
python3 -m rps_playground.main tournament --rounds 1000 --seed 42
```

Each algorithm receives its own deterministic RNG derived from the master seed.

---

## 📤 Export

```bash
# JSON — full match data + leaderboard + H2H matrix
python3 -m rps_playground.main tournament --export json --output results.json

# CSV — leaderboard table only
python3 -m rps_playground.main tournament --export csv --output results.csv
```

---

*Built with Python 🐍 + Flask*
