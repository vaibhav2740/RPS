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

# Pre-computed winner table: (move_a, move_b) → outcome
# Eliminates function call + branching from the hot loop
_WINNER_TABLE = {
    (Move.ROCK, Move.ROCK): 0,
    (Move.ROCK, Move.PAPER): -1,
    (Move.ROCK, Move.SCISSORS): 1,
    (Move.PAPER, Move.ROCK): 1,
    (Move.PAPER, Move.PAPER): 0,
    (Move.PAPER, Move.SCISSORS): -1,
    (Move.SCISSORS, Move.ROCK): -1,
    (Move.SCISSORS, Move.PAPER): 1,
    (Move.SCISSORS, Move.SCISSORS): 0,
}


def determine_winner(move_a: Move, move_b: Move) -> int:
    """Return 1 if A wins, -1 if B wins, 0 for draw."""
    return _WINNER_TABLE[move_a, move_b]


class _FrozenHistory:
    """O(1) immutable view of a move history list.

    Wraps a reference to the engine's internal history list without
    copying.  Supports all read operations that algorithms use
    (indexing, slicing, iteration, len, Counter, ``in``, bool) but
    prevents accidental mutation (no append / pop / insert methods).

    Created ONCE before the match loop and reused every round — the
    view automatically sees new moves as the underlying list grows.
    """
    __slots__ = ('_data',)

    def __init__(self, data: list):
        self._data = data

    def __getitem__(self, key):
        return self._data[key]

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        return iter(self._data)

    def __contains__(self, item):
        return item in self._data

    def __bool__(self):
        return bool(self._data)

    def __repr__(self):
        return f"FrozenHistory({self._data!r})"


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
    record_moves: bool = True,
) -> MatchResult:
    """Run a match of N rounds between two algorithm instances.

    Each algorithm gets its own seeded RNG derived from the master seed.

    Args:
        record_moves: If False, skip storing per-round moves in the result.
                      Set to False for bulk tournament runs to save memory.
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

    # Internal mutable lists — only the engine appends to these
    a_history: list[Move] = []
    b_history: list[Move] = []

    # Immutable views shared with algorithms — created ONCE, O(1)
    # They automatically see new moves as the underlying lists grow
    a_frozen = _FrozenHistory(a_history)
    b_frozen = _FrozenHistory(b_history)

    # Local references for the hot loop (avoid repeated dict/attr lookups)
    winner_table = _WINNER_TABLE
    a_choose = algo_a.choose
    b_choose = algo_b.choose
    a_wins = 0
    b_wins = 0
    draws = 0

    if record_moves:
        a_moves_list = result.a_moves
        b_moves_list = result.b_moves
        for round_num in range(rounds):
            move_a = a_choose(round_num, a_frozen, b_frozen)
            move_b = b_choose(round_num, b_frozen, a_frozen)

            outcome = winner_table[move_a, move_b]
            if outcome == 1:
                a_wins += 1
            elif outcome == -1:
                b_wins += 1
            else:
                draws += 1

            a_history.append(move_a)
            b_history.append(move_b)
            a_moves_list.append(move_a)
            b_moves_list.append(move_b)
    else:
        # No move recording — tighter loop for bulk tournament runs
        for round_num in range(rounds):
            move_a = a_choose(round_num, a_frozen, b_frozen)
            move_b = b_choose(round_num, b_frozen, a_frozen)

            outcome = winner_table[move_a, move_b]
            if outcome == 1:
                a_wins += 1
            elif outcome == -1:
                b_wins += 1
            else:
                draws += 1

            a_history.append(move_a)
            b_history.append(move_b)

    result.a_wins = a_wins
    result.b_wins = b_wins
    result.draws = draws
    return result

