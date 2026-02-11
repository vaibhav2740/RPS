"""20 baseline Rock-Paper-Scissors algorithms."""

from abc import ABC, abstractmethod
from collections import Counter
import random
from .engine import Move, BEATS, BEATEN_BY


class Algorithm(ABC):
    """Base class for RPS algorithms."""

    def __init__(self):
        self.rng: random.Random = random.Random()

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def choose(self, round_num: int, my_history: list[Move], opp_history: list[Move]) -> Move:
        ...

    def reset(self):
        """Reset any internal state between matches."""
        pass

    def __repr__(self):
        return f"<{self.name}>"


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

MOVES = [Move.ROCK, Move.PAPER, Move.SCISSORS]


def _counter_move(move: Move) -> Move:
    """Return the move that beats `move`."""
    return BEATEN_BY[move]


# ---------------------------------------------------------------------------
# 1-3: Constant strategies
# ---------------------------------------------------------------------------

class AlwaysRock(Algorithm):
    name = "Always Rock"
    def choose(self, round_num, my_history, opp_history):
        return Move.ROCK


class AlwaysPaper(Algorithm):
    name = "Always Paper"
    def choose(self, round_num, my_history, opp_history):
        return Move.PAPER


class AlwaysScissors(Algorithm):
    name = "Always Scissors"
    def choose(self, round_num, my_history, opp_history):
        return Move.SCISSORS


# ---------------------------------------------------------------------------
# 4: Pure Random
# ---------------------------------------------------------------------------

class PureRandom(Algorithm):
    name = "Pure Random"
    def choose(self, round_num, my_history, opp_history):
        return self.rng.choice(MOVES)


# ---------------------------------------------------------------------------
# 5: Cycle R → P → S
# ---------------------------------------------------------------------------

class Cycle(Algorithm):
    name = "Cycle"
    def choose(self, round_num, my_history, opp_history):
        return MOVES[round_num % 3]


# ---------------------------------------------------------------------------
# 6: Mirror Opponent (copy last move)
# ---------------------------------------------------------------------------

class MirrorOpponent(Algorithm):
    name = "Mirror Opponent"
    def choose(self, round_num, my_history, opp_history):
        if not opp_history:
            return self.rng.choice(MOVES)
        return opp_history[-1]


# ---------------------------------------------------------------------------
# 7: Tit-for-Tat (start with Rock, then mirror)
# ---------------------------------------------------------------------------

class TitForTat(Algorithm):
    name = "Tit-for-Tat"
    def choose(self, round_num, my_history, opp_history):
        if not opp_history:
            return Move.ROCK
        return opp_history[-1]


# ---------------------------------------------------------------------------
# 8: Anti-Tit-for-Tat (play what beats opponent's last)
# ---------------------------------------------------------------------------

class AntiTitForTat(Algorithm):
    name = "Anti-Tit-for-Tat"
    def choose(self, round_num, my_history, opp_history):
        if not opp_history:
            return self.rng.choice(MOVES)
        return _counter_move(opp_history[-1])


# ---------------------------------------------------------------------------
# 9: Frequency Analyzer
# ---------------------------------------------------------------------------

class FrequencyAnalyzer(Algorithm):
    name = "Frequency Analyzer"
    def choose(self, round_num, my_history, opp_history):
        if not opp_history:
            return self.rng.choice(MOVES)
        counts = Counter(opp_history)
        most_common = counts.most_common(1)[0][0]
        return _counter_move(most_common)


# ---------------------------------------------------------------------------
# 10: Markov Predictor (1st-order transition model)
# ---------------------------------------------------------------------------

class MarkovPredictor(Algorithm):
    name = "Markov Predictor"

    def reset(self):
        self._transitions: dict[Move, Counter] = {m: Counter() for m in MOVES}

    def choose(self, round_num, my_history, opp_history):
        if len(opp_history) < 2:
            return self.rng.choice(MOVES)
        # Update transition table with the latest transition
        prev = opp_history[-2]
        curr = opp_history[-1]
        self._transitions[prev][curr] += 1
        # Predict opponent's next move based on their last move
        last = opp_history[-1]
        if self._transitions[last]:
            predicted = self._transitions[last].most_common(1)[0][0]
        else:
            predicted = self.rng.choice(MOVES)
        return _counter_move(predicted)


# ---------------------------------------------------------------------------
# 11: Pattern Detector (looks for repeating N-grams)
# ---------------------------------------------------------------------------

class PatternDetector(Algorithm):
    name = "Pattern Detector"

    def choose(self, round_num, my_history, opp_history):
        if len(opp_history) < 3:
            return self.rng.choice(MOVES)
        # Try pattern lengths 5 down to 2
        for length in range(min(5, len(opp_history) - 1), 1, -1):
            pattern = tuple(opp_history[-length:])
            # Search for this pattern earlier in history
            for i in range(len(opp_history) - length):
                if tuple(opp_history[i:i + length]) == pattern:
                    if i + length < len(opp_history):
                        predicted = opp_history[i + length]
                        return _counter_move(predicted)
        # Fallback
        return _counter_move(opp_history[-1])


# ---------------------------------------------------------------------------
# 12: Win-Stay, Lose-Shift
# ---------------------------------------------------------------------------

class WinStayLoseShift(Algorithm):
    name = "Win-Stay-Lose-Shift"
    def choose(self, round_num, my_history, opp_history):
        if not my_history:
            return self.rng.choice(MOVES)
        my_last = my_history[-1]
        opp_last = opp_history[-1]
        if BEATS[my_last] == opp_last:
            # Won last round — stay
            return my_last
        elif my_last == opp_last:
            # Drew — stay
            return my_last
        else:
            # Lost — shift to what would have beaten opponent
            return _counter_move(opp_last)


# ---------------------------------------------------------------------------
# 13: Meta-Predictor (ensemble of sub-predictors)
# ---------------------------------------------------------------------------

class MetaPredictor(Algorithm):
    name = "Meta-Predictor"

    def reset(self):
        # Track scores of sub-strategies
        self._scores = [0, 0, 0]  # freq, markov, pattern

    def choose(self, round_num, my_history, opp_history):
        if len(opp_history) < 3:
            return self.rng.choice(MOVES)

        # Sub-predictor 1: Frequency
        counts = Counter(opp_history)
        freq_pred = counts.most_common(1)[0][0]

        # Sub-predictor 2: Last move repeat
        last_pred = opp_history[-1]

        # Sub-predictor 3: Anti-last (opponent's counter of own last)
        anti_pred = _counter_move(opp_history[-1])

        predictions = [freq_pred, last_pred, anti_pred]

        # Score each predictor based on last round accuracy
        if len(opp_history) >= 2:
            prev_opp = opp_history[-1]
            prev_counts = Counter(opp_history[:-1])
            prev_freq = prev_counts.most_common(1)[0][0] if prev_counts else Move.ROCK
            prev_last = opp_history[-2]
            prev_anti = _counter_move(opp_history[-2])
            prev_preds = [prev_freq, prev_last, prev_anti]
            for i, p in enumerate(prev_preds):
                if p == prev_opp:
                    self._scores[i] += 1

        # Pick the best scoring predictor
        best_idx = self._scores.index(max(self._scores))
        predicted = predictions[best_idx]
        return _counter_move(predicted)


# ---------------------------------------------------------------------------
# 14: Noise Strategy (best counter + random noise 20%)
# ---------------------------------------------------------------------------

class NoiseStrategy(Algorithm):
    name = "Noise Strategy"
    def choose(self, round_num, my_history, opp_history):
        if not opp_history or self.rng.random() < 0.2:
            return self.rng.choice(MOVES)
        counts = Counter(opp_history)
        most_common = counts.most_common(1)[0][0]
        return _counter_move(most_common)


# ---------------------------------------------------------------------------
# 15: Adaptive Hybrid (switches between strategies)
# ---------------------------------------------------------------------------

class AdaptiveHybrid(Algorithm):
    name = "Adaptive Hybrid"

    def reset(self):
        self._strategy = 0  # 0=freq, 1=mirror, 2=random
        self._strat_wins = [0, 0, 0]
        self._strat_plays = [0, 0, 0]
        self._switch_interval = 50

    def choose(self, round_num, my_history, opp_history):
        if not opp_history:
            return self.rng.choice(MOVES)

        # Periodically switch to best performing strategy
        if round_num > 0 and round_num % self._switch_interval == 0:
            rates = []
            for i in range(3):
                if self._strat_plays[i] > 0:
                    rates.append(self._strat_wins[i] / self._strat_plays[i])
                else:
                    rates.append(0)
            self._strategy = rates.index(max(rates))

        # Execute current strategy
        if self._strategy == 0:
            counts = Counter(opp_history)
            move = _counter_move(counts.most_common(1)[0][0])
        elif self._strategy == 1:
            move = opp_history[-1]
        else:
            move = self.rng.choice(MOVES)

        self._strat_plays[self._strategy] += 1

        # Track wins from last round
        if len(my_history) >= 1:
            my_last = my_history[-1]
            opp_last = opp_history[-1]
            if BEATS[my_last] == opp_last:
                self._strat_wins[self._strategy] += 1

        return move


# ---------------------------------------------------------------------------
# 16: Last-Move Counter
# ---------------------------------------------------------------------------

class LastMoveCounter(Algorithm):
    name = "Last-Move Counter"
    def choose(self, round_num, my_history, opp_history):
        if not opp_history:
            return self.rng.choice(MOVES)
        return _counter_move(opp_history[-1])


# ---------------------------------------------------------------------------
# 17: Weighted Random (weighted by opponent frequency)
# ---------------------------------------------------------------------------

class WeightedRandom(Algorithm):
    name = "Weighted Random"
    def choose(self, round_num, my_history, opp_history):
        if not opp_history:
            return self.rng.choice(MOVES)
        counts = Counter(opp_history)
        total = sum(counts.values())
        # Weight counter-moves by opponent's frequency of each move
        weights = []
        counter_moves = []
        for move in MOVES:
            freq = counts.get(move, 0)
            if freq > 0:
                counter_moves.append(_counter_move(move))
                weights.append(freq / total)
        if not counter_moves:
            return self.rng.choice(MOVES)
        return self.rng.choices(counter_moves, weights=weights, k=1)[0]


# ---------------------------------------------------------------------------
# 18: Punisher (cooperates then punishes detected patterns)
# ---------------------------------------------------------------------------

class Punisher(Algorithm):
    name = "Punisher"

    def reset(self):
        self._punish_until = -1

    def choose(self, round_num, my_history, opp_history):
        if len(opp_history) < 3:
            return self.rng.choice(MOVES)

        # Detect if opponent is repeating a move
        if round_num < self._punish_until:
            # In punish mode: counter their most common
            counts = Counter(opp_history[-10:])
            return _counter_move(counts.most_common(1)[0][0])

        # Check if opponent repeated last move 3+ times
        if opp_history[-1] == opp_history[-2] == opp_history[-3]:
            self._punish_until = round_num + 10
            return _counter_move(opp_history[-1])

        return self.rng.choice(MOVES)


# ---------------------------------------------------------------------------
# 19: Forgiver (like Punisher but forgives after fewer rounds)
# ---------------------------------------------------------------------------

class Forgiver(Algorithm):
    name = "Forgiver"

    def reset(self):
        self._punish_until = -1

    def choose(self, round_num, my_history, opp_history):
        if len(opp_history) < 3:
            return self.rng.choice(MOVES)

        if round_num < self._punish_until:
            counts = Counter(opp_history[-5:])
            return _counter_move(counts.most_common(1)[0][0])

        if opp_history[-1] == opp_history[-2] == opp_history[-3]:
            self._punish_until = round_num + 3  # forgives sooner
            return _counter_move(opp_history[-1])

        return self.rng.choice(MOVES)


# ---------------------------------------------------------------------------
# 20: Chaos Strategy (randomly picks a sub-strategy each round)
# ---------------------------------------------------------------------------

class ChaosStrategy(Algorithm):
    name = "Chaos Strategy"
    def choose(self, round_num, my_history, opp_history):
        strategy = self.rng.randint(0, 4)
        if strategy == 0:
            return self.rng.choice(MOVES)
        elif strategy == 1:
            if opp_history:
                return _counter_move(opp_history[-1])
            return self.rng.choice(MOVES)
        elif strategy == 2:
            if opp_history:
                return opp_history[-1]
            return self.rng.choice(MOVES)
        elif strategy == 3:
            return MOVES[round_num % 3]
        else:
            if opp_history:
                counts = Counter(opp_history)
                return _counter_move(counts.most_common(1)[0][0])
            return self.rng.choice(MOVES)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

ALL_ALGORITHM_CLASSES = [
    AlwaysRock, AlwaysPaper, AlwaysScissors,
    PureRandom, Cycle, MirrorOpponent,
    TitForTat, AntiTitForTat, FrequencyAnalyzer,
    MarkovPredictor, PatternDetector, WinStayLoseShift,
    MetaPredictor, NoiseStrategy, AdaptiveHybrid,
    LastMoveCounter, WeightedRandom, Punisher,
    Forgiver, ChaosStrategy,
]


def get_all_algorithms() -> list[Algorithm]:
    """Return fresh instances of all 20 algorithms."""
    return [cls() for cls in ALL_ALGORITHM_CLASSES]


def get_algorithm_by_name(name: str) -> Algorithm:
    """Get a single algorithm instance by name (case-insensitive)."""
    name_lower = name.lower()
    for cls in ALL_ALGORITHM_CLASSES:
        if cls.name.lower() == name_lower:
            return cls()
    available = ", ".join(cls.name for cls in ALL_ALGORITHM_CLASSES)
    raise ValueError(f"Unknown algorithm: '{name}'. Available: {available}")
