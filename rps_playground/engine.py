"""Core game engine for Rock-Paper-Scissors matches."""

from enum import Enum
from dataclasses import dataclass, field
from collections import Counter
import random
from typing import Optional


class Move(Enum):
    ROCK = "rock"
    PAPER = "paper"
    SCISSORS = "scissors"


# What each move beats
BEATS = {
    Move.ROCK: Move.SCISSORS,
    Move.SCISSORS: Move.PAPER,
    Move.PAPER: Move.ROCK,
}

# What beats each move
BEATEN_BY = {v: k for k, v in BEATS.items()}


def determine_winner(move_a: Move, move_b: Move) -> int:
    """Return 1 if A wins, -1 if B wins, 0 for draw."""
    if move_a == move_b:
        return 0
    if BEATS[move_a] == move_b:
        return 1
    return -1


@dataclass
class MatchResult:
    """Result of a match between two algorithms."""
    algo_a_name: str
    algo_b_name: str
    rounds: int
    a_wins: int = 0
    b_wins: int = 0
    draws: int = 0
    a_moves: list = field(default_factory=list)
    b_moves: list = field(default_factory=list)

    @property
    def a_win_pct(self) -> float:
        return (self.a_wins / self.rounds * 100) if self.rounds else 0.0

    @property
    def b_win_pct(self) -> float:
        return (self.b_wins / self.rounds * 100) if self.rounds else 0.0

    @property
    def draw_pct(self) -> float:
        return (self.draws / self.rounds * 100) if self.rounds else 0.0

    @property
    def a_move_distribution(self) -> dict[str, int]:
        return dict(Counter(m.value for m in self.a_moves))

    @property
    def b_move_distribution(self) -> dict[str, int]:
        return dict(Counter(m.value for m in self.b_moves))

    @property
    def a_most_common_move(self) -> str:
        if not self.a_moves:
            return "N/A"
        return Counter(m.value for m in self.a_moves).most_common(1)[0][0]

    @property
    def b_most_common_move(self) -> str:
        if not self.b_moves:
            return "N/A"
        return Counter(m.value for m in self.b_moves).most_common(1)[0][0]

    def to_dict(self) -> dict:
        return {
            "algo_a": self.algo_a_name,
            "algo_b": self.algo_b_name,
            "rounds": self.rounds,
            "a_wins": self.a_wins,
            "b_wins": self.b_wins,
            "draws": self.draws,
            "a_win_pct": round(self.a_win_pct, 2),
            "b_win_pct": round(self.b_win_pct, 2),
            "draw_pct": round(self.draw_pct, 2),
            "a_move_distribution": self.a_move_distribution,
            "b_move_distribution": self.b_move_distribution,
        }


def run_match(
    algo_a,
    algo_b,
    rounds: int = 1000,
    seed: Optional[int] = None,
) -> MatchResult:
    """Run a match of N rounds between two algorithm instances.

    Each algorithm gets its own seeded RNG derived from the master seed.
    """
    master_rng = random.Random(seed)
    algo_a_seed = master_rng.randint(0, 2**31)
    algo_b_seed = master_rng.randint(0, 2**31)

    algo_a.rng = random.Random(algo_a_seed)
    algo_b.rng = random.Random(algo_b_seed)
    algo_a.reset()
    algo_b.reset()

    result = MatchResult(
        algo_a_name=algo_a.name,
        algo_b_name=algo_b.name,
        rounds=rounds,
    )

    a_history: list[Move] = []
    b_history: list[Move] = []

    for round_num in range(rounds):
        move_a = algo_a.choose(round_num, a_history.copy(), b_history.copy())
        move_b = algo_b.choose(round_num, b_history.copy(), a_history.copy())

        outcome = determine_winner(move_a, move_b)
        if outcome == 1:
            result.a_wins += 1
        elif outcome == -1:
            result.b_wins += 1
        else:
            result.draws += 1

        a_history.append(move_a)
        b_history.append(move_b)
        result.a_moves.append(move_a)
        result.b_moves.append(move_b)

    return result
