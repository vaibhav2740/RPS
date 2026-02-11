# 🎮 RPS Playground — Algorithm Battle Arena

A modular Rock-Paper-Scissors algorithm testing playground with **20 built-in bots**, **Elo ratings**, **three competition modes**, and a beautiful **dark-themed Web UI**.

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

Full round-robin — every algorithm plays every other algorithm (190 matches total).

1. **Set rounds per match** and optional **seed**
2. Click **🏆 RUN**
3. View the **leaderboard** sorted by Elo rating
4. Scroll down to see the **Head-to-Head matrix** (W/L/D for every pairing)

### 🤖 One vs All

Test a single algorithm against the entire pool of 20 bots.

1. **Select your algorithm** from the dropdown
2. **Set rounds** and optional **seed**
3. Click **🤖 RUN**
4. View **individual matchup results** (win/loss/draw for each opponent)
5. See the **overall ranking** table

---

## 🤖 Algorithm Reference

All 20 algorithms explained:

### Static Strategies

| # | Algorithm | Strategy |
|---|-----------|----------|
| 1 | **Always Rock** | Plays Rock every round. Simple baseline — easily exploited by Paper-playing bots. |
| 2 | **Always Paper** | Plays Paper every round. Beats constant Rock players but loses to Scissors. |
| 3 | **Always Scissors** | Plays Scissors every round. Completes the constant-move trio. |

### Randomized Strategies

| # | Algorithm | Strategy |
|---|-----------|----------|
| 4 | **Pure Random** | Picks Rock, Paper, or Scissors uniformly at random each round. Theoretically unbeatable in expectation (Nash equilibrium), but can't exploit predictable opponents. |
| 17 | **Weighted Random** | Picks moves randomly but **weighted by opponent's frequency** — if opponent plays Rock often, this bot plays Paper more often. Adapts gradually. |
| 20 | **Chaos Strategy** | Randomly selects a **different sub-strategy each round** (random, counter-last, mirror, cycle, or frequency counter). Unpredictable but inconsistent. |

### Reactive Strategies

| # | Algorithm | Strategy |
|---|-----------|----------|
| 5 | **Cycle** | Plays Rock → Paper → Scissors → Rock → ... in a fixed repeating cycle. Predictable but guarantees equal distribution. |
| 6 | **Mirror Opponent** | Copies the opponent's **last move**. If they played Rock last round, plays Rock this round. Simple mimicry. |
| 7 | **Tit-for-Tat** | Starts with Rock, then **mirrors the opponent's previous move**. Classic cooperative strategy from game theory. |
| 8 | **Anti-Tit-for-Tat** | Plays whatever **beats the opponent's last move**. If they played Rock, plays Paper. Direct counter-strategy. |
| 16 | **Last-Move Counter** | Identical to Anti-Tit-for-Tat but starts with a random move. Plays the move that **would have beaten** the opponent's previous move. |

### Analytical Strategies

| # | Algorithm | Strategy |
|---|-----------|----------|
| 9 | **Frequency Analyzer** | Tracks the **overall frequency** of opponent's moves. Plays the counter to their most common move. Exploits opponents with strong biases. |
| 10 | **Markov Predictor** | Builds a **first-order Markov chain** of opponent transitions (e.g., "after Rock, they usually play Paper"). Predicts next move from the transition table and counters it. |
| 11 | **Pattern Detector** | Looks for **repeating n-gram patterns** (length 2-5) in opponent history. If a pattern is found, predicts and counters the next move. Falls back to counter-last. |
| 13 | **Meta-Predictor** | Runs **three sub-predictors** (frequency, last-move-repeat, anti-last) simultaneously. Tracks which predictor has been most accurate and follows the best one. Ensemble approach. |

### Adaptive Strategies

| # | Algorithm | Strategy |
|---|-----------|----------|
| 12 | **Win-Stay, Lose-Shift** | If it **won or drew** the last round, repeats the same move. If it **lost**, switches to what would have beaten the opponent. Classic reinforcement principle. |
| 14 | **Noise Strategy** | 80% of the time plays the **counter to opponent's most common move**. 20% of the time plays **randomly**. The noise prevents easy counter-exploitation. |
| 15 | **Adaptive Hybrid** | Runs three sub-strategies (frequency counter, mirror, random) and tracks their **win rates over time**. Every 50 rounds, switches to the strategy performing best. |

### Punishment Strategies

| # | Algorithm | Strategy |
|---|-----------|----------|
| 18 | **Punisher** | Plays randomly by default. When it **detects the opponent repeating a move 3+ times**, enters "punish mode" for 10 rounds — aggressively countering their most common recent move. |
| 19 | **Forgiver** | Like Punisher, but **forgives after only 3 rounds** instead of 10. More lenient — returns to random play quickly, giving the opponent a chance to change. |

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
├── algorithms.py        # 20 algorithms + base class + registry
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
  one-vs-all     Mode 2: Custom algo vs all 20 bots
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
