
import sys
import os
sys.path.append(os.getcwd())

from rps_playground.algorithms import get_algorithm_by_name
from rps_playground.engine import run_match
from collections import defaultdict
import time

# Elite Pool
ELITE_BOTS = [
    # Enhanced Supreme (Phase 7)
    "The Singularity", "The Black Hole", "Phoenix",
    # Supreme (Phase 6)
    "The Time Traveler", "The Collective", "The Mirror World",
    # Novel (Phase 5)
    "Super Omniscient", "The Void", "The Doppelganger", "The Architect",
    # Classic Champions
    "Phantom Ensemble", "The Hydra", "Geometry Bot",
    # Baselines
    "Markov Predictor", "Iocaine Powder"
]

ROUNDS = 100
REPEATS = 10

print(f"--- CHAMPIONS LEAGUE BENCHMARK ---")
print(f"Bots: {len(ELITE_BOTS)}")
print(f"Match Rounds: {ROUNDS}")
print(f"Tournament Repeats: {REPEATS}")
print("-" * 40)

total_wins = defaultdict(int)
tournament_wins = defaultdict(int) # Who won the tournament (highest score)

start_time = time.time()

for t in range(REPEATS):
    print(f"Running Tournament {t+1}/{REPEATS}...", end="", flush=True)

    # Score tracking for this tournament
    scores = defaultdict(float) # 1.0 for win, 0.5 for draw

    # Round Robin
    for i, name_a in enumerate(ELITE_BOTS):
        for j, name_b in enumerate(ELITE_BOTS):
            if i >= j: continue # Avoid self-play and duplicates

            bot_a = get_algorithm_by_name(name_a)
            bot_b = get_algorithm_by_name(name_b)

            # Reset bots
            bot_a.reset()
            bot_b.reset()

            # Run match
            # Note: seed needs to change per tournament to check robustness
            res = run_match(bot_a, bot_b, rounds=ROUNDS, seed=t*1000 + i*j, record_moves=False)

            if res.a_wins > res.b_wins:
                scores[name_a] += 1.0
                total_wins[name_a] += 1
            elif res.b_wins > res.a_wins:
                scores[name_b] += 1.0
                total_wins[name_b] += 1
            else:
                scores[name_a] += 0.5
                scores[name_b] += 0.5

    # Find winner of this tournament
    best_score = -1
    winners = []
    for bot, score in scores.items():
        if score > best_score:
            best_score = score
            winners = [bot]
        elif score == best_score:
            winners.append(bot)

    for w in winners:
        tournament_wins[w] += 1

    print(f" Winner: {', '.join(winners)} ({best_score} pts)")

print("-" * 40)
print(f"Completed in {time.time() - start_time:.2f}s")
print("\n--- FINAL RESULTS ---")
print("Algorithm | Tournament Wins | Total Match Wins")
print("|---|---|---|")

# Sort by Total Match Wins
sorted_bots = sorted(ELITE_BOTS, key=lambda x: total_wins[x], reverse=True)

for name in sorted_bots:
    print(f"{name:20} | {tournament_wins[name]:2} | {total_wins[name]}")
