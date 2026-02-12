"""103 Rock-Paper-Scissors algorithms (20 baseline + 83 upgraded/creative)."""

from abc import ABC, abstractmethod
from collections import Counter
import random
from .engine import Move, BEATS, BEATEN_BY


class Algorithm(ABC):
    """Base class for RPS algorithms."""

    def __init__(self):
        self.rng: random.Random = random.Random()
        self._m2i = {Move.ROCK: 0, Move.PAPER: 1, Move.SCISSORS: 2}
        self._i2m = {0: Move.ROCK, 1: Move.PAPER, 2: Move.SCISSORS}
        self._base = self
        self.reset()

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

    def set_match_context(self, opponent_name: str, opponent_history: list[dict]):
        """Receive metadata about the opponent before a match begins.

        Called by the competition tournament mode. Override in subclasses
        to exploit opponent name and tournament record.

        Args:
            opponent_name: Name of the opponent algorithm.
            opponent_history: List of dicts with opponent's prior match results:
                [{"opponent": str, "result": "win"|"loss"|"draw",
                  "score": "W-L", "rounds": int}]
        """
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
    """Always chooses Rock.
    
    The most predictable strategy possible. It serves as a baseline to test if other
    algorithms can exploit a static opponent.
    
    **Type**: Baseline
    **Complexity**: O(1)
    """
    name = "Always Rock"
    def choose(self, round_num, my_history, opp_history):
        return Move.ROCK


class AlwaysPaper(Algorithm):
    """Always chooses Paper.
    
    A static baseline strategy. Useful for sanity checking dynamic algorithms
    to ensure they can learn to play Scissors.
    
    **Type**: Baseline
    **Complexity**: O(1)
    """
    name = "Always Paper"
    def choose(self, round_num, my_history, opp_history):
        return Move.PAPER


class AlwaysScissors(Algorithm):
    """Always chooses Scissors.
    
    A static baseline strategy. Useful for sanity checking dynamic algorithms
    to ensure they can learn to play Rock.
    
    **Type**: Baseline
    **Complexity**: O(1)
    """
    name = "Always Scissors"
    def choose(self, round_num, my_history, opp_history):
        return Move.SCISSORS


# ---------------------------------------------------------------------------
# 4: Pure Random
# ---------------------------------------------------------------------------

class PureRandom(Algorithm):
    """Chooses a move completely at random.
    
    Each move (Rock, Paper, Scissors) has an equal probability (33.3%).
    This strategy is theoretically unexploitable in the long run (Nash Equilibrium),
    but it also cannot exploit an opponent's weaknesses.
    
    **Type**: Baseline
    **Complexity**: O(1)
    """
    name = "Pure Random"
    def choose(self, round_num, my_history, opp_history):
        return self.rng.choice(MOVES)


# ---------------------------------------------------------------------------
# 5: Cycle R → P → S
# ---------------------------------------------------------------------------

class Cycle(Algorithm):
    """Cycles through Rock -> Paper -> Scissors repeatedly.
    
    A simple deterministic pattern. Easy to exploit once detected.
    
    **Sequence**: R, P, S, R, P, S...
    **Type**: Pattern
    """
    name = "Cycle"
    def choose(self, round_num, my_history, opp_history):
        return MOVES[round_num % 3]


# ---------------------------------------------------------------------------
# 6: Mirror Opponent (copy last move)
# ---------------------------------------------------------------------------

class PersistentRandom(Algorithm):
    """Plays a random move and sticks with it for a random duration.
    
    Instead of changing every turn, it picks a move (e.g., Rock) and plays it
    for 5-15 rounds, then switches to a new random move. This confuses algorithms
    that expect frequent changes or static play.
    
    **Type**: Baseline / Pattern
    """
    name = "Persistent Random"
    
    def reset(self):
        self._current_move = self.rng.choice(MOVES)
        self._remaining = 0
        
    def choose(self, round_num, my_history, opp_history):
        if self._remaining <= 0:
             self._current_move = self.rng.choice(MOVES)
             self._remaining = self.rng.randint(5, 15)
        
        self._remaining -= 1
        return self._current_move


# ---------------------------------------------------------------------------
# 7: Tit-for-Tat (start with Rock, then mirror)
# ---------------------------------------------------------------------------

class TitForTat(Algorithm):
    """Repeats the opponent's last move.
    
    If you play Rock, it plays Rock next. Famous in Game Theory (Prisoner's Dilemma),
    but in RPS, it creates a delay where it effectively loses to a player who knows
    the pattern (playing the counter to what they *just* played).
    
    **Type**: Reactive
    """
    name = "Tit-for-Tat"
    def choose(self, round_num, my_history, opp_history):
        if not opp_history:
            return Move.ROCK
        return opp_history[-1]


# ---------------------------------------------------------------------------
# 8: Anti-Tit-for-Tat (play what beats opponent's last)
# ---------------------------------------------------------------------------

class AntiTitForTat(Algorithm):
    """Plays the move that beats the opponent's last move.
    
    If you played Rock, it assumes you might play Rock again (or it just wants to beat Rock),
    so it plays Paper. Effective against "repeaters".
    
    **Type**: Reactive
    """
    name = "Anti-Tit-for-Tat"
    def choose(self, round_num, my_history, opp_history):
        if not opp_history:
            return self.rng.choice(MOVES)
        return _counter_move(opp_history[-1])


# ---------------------------------------------------------------------------
# 9: Frequency Analyzer
# ---------------------------------------------------------------------------

class FrequencyAnalyzer(Algorithm):
    """Counters the opponent's most frequent move.
    
    If you play Rock 50% of the time, this bot will play Paper (the counter to Rock).
    It exploits players who have a bias towards one move.
    
    **Type**: Frequency
    **Logic**: `Counter(opp_history).most_common(1)`
    """
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
    """Predicts next move based on the previous move (1st-order Markov Chain).
    
    Builds a transition matrix tracking `Last Move -> Next Move` probabilities.
    if you often play Paper after Rock, it will predict Paper when it sees Rock,
    and play Scissors to counter.
    
    **Type**: Pattern / Probability
    **Order**: 1 (looks back 1 move)
    """
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

class Spiral(Algorithm):
    """Plays R, R, P, P, S, S... in a continuous spiral pattern.
    
    A variation of Cycle that repeats each move twice before switching.
    Slightly harder to detect than a simple cycle for very naive bots,
    but still easily exploitable.
    
    **Sequence**: R, R, P, P, S, S...
    **Type**: Pattern
    """
    name = "Spiral"

    def choose(self, round_num, my_history, opp_history):
        # Period 6: 0,1 -> R; 2,3 -> P; 4,5 -> S
        idx = (round_num // 2) % 3
        return MOVES[idx]


# ---------------------------------------------------------------------------
# 12: Win-Stay, Lose-Shift
# ---------------------------------------------------------------------------

class WinStayLoseShift(Algorithm):
    """Standard Pavlov strategy: Win = Stay, Lose = Switch.
    
    If it wins, it plays the same move again.
    If it loses, it switches to the move that would have beaten the opponent's last move.
    
    **Type**: Reactive / Heuristic
    **Also known as**: Win-Stay, Lose-Switch
    """
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
    """Ensemble strategy that uses multiple sub-predictors.
    
    It runs three internal strategies (Frequency, Last-Move, Anti-Last) and
    tracks their virtual scores. It then chooses the move suggested by the
    currently winning sub-strategy.
    
    **Sub-bots**: Frequency, Last Move, Anti-Last
    **Type**: Ensemble / Meta
    """
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
    """Plays optimally (Frequency Counter), but adds 20% random noise.
    
    Most of the time (80%), it plays the counter to your most frequent move.
    Occasionally (20%), it plays completely randomly to throw off predictors
    trying to exploit *it*.
    
    **Type**: Frequency / Stochastic
    **Noise Level**: 20%
    """
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
    """Switches between Strategy A, B, and C based on performance.
    
    Every 50 rounds, it evaluates which internal strategy has the highest win rate
    and switches to it.
    
    **Strategies**: Frequency, Mirror, Random
    **Interval**: 50 rounds
    **Type**: Adaptive / Ensemble
    """
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
    """Plays the counter to your OWN last move.
    
    If it played Rock last time, it assumes you will try to beat Rock (with Paper),
    so it plays Scissors (to beat Paper).
    
    **Type**: Reactive / Level-K
    **Logic**: `Counter(my_last_move)`
    """
    name = "Last-Move Counter"
    def choose(self, round_num, my_history, opp_history):
        if not opp_history:
            return self.rng.choice(MOVES)
        return _counter_move(opp_history[-1])


# ---------------------------------------------------------------------------
# 17: Weighted Random (weighted by opponent frequency)
# ---------------------------------------------------------------------------

class WeightedRandom(Algorithm):
    """Plays moves randomly, but weighted by your own move distribution.
    
    If you play Rock 70% of the time, this bot will play Paper (Rock's counter)
    roughly 70% of the time. It matches your biases probablistically rather than
    deterministically.
    
    **Type**: Frequency / Stochastic
    """
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
    """Cooperates until it detects a repeated pattern, then punishes it.
    
    Normally plays random. If it sees you play the same move 3 times in a row,
    it enters "Punish Mode" for 10 rounds, aggressively countering that move.
    
    **Type**: Psychological / Reactive
    **Trigger**: 3 repeats
    """
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
    """Like Punisher, but forgives faster (shorter grudge).
    
    If it detects you repeating moves, it punishes you, but returns to neutral/random
    play after only 3 rounds instead of 10.
    
    **Type**: Psychological / Reactive
    **Trigger**: 3 repeats
    """
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
    """Randomly picks a different STRATEGY each round.
    
    Instead of picking a move, it picks a strategy (Random, Counter, Mirror, Cycle, or Frequency)
    and executes one move from it. Extremely unpredictable.
    
    **Type**: Ensemble / Chaos
    **Strategies**: 5 unique logics
    """
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


# ===========================================================================
#  CREATIVE ALGORITHMS (21-32)
# ===========================================================================


# ---------------------------------------------------------------------------
# 21: Decay Analyzer (exponential recency-weighted frequency)
# ---------------------------------------------------------------------------

class DecayAnalyzer(Algorithm):
    """Frequency analysis where recent moves matter exponentially more.
    
    Uses a decay factor (0.9) so the last 10-20 moves dominate the prediction.
    Adapts much faster than vanilla Frequency Analyzer to changing opponents.
    
    **Type**: Frequency / Adaptive
    **Decay Factor**: 0.9
    """
    name = "Decay Analyzer"

    def choose(self, round_num, my_history, opp_history):
        if not opp_history:
            return self.rng.choice(MOVES)
        weights = {m: 0.0 for m in MOVES}
        decay = 0.9
        w = 1.0
        for move in reversed(opp_history):
            weights[move] += w
            w *= decay
        predicted = max(weights, key=weights.get)
        return _counter_move(predicted)


# ---------------------------------------------------------------------------
# 22: Historian (longest context matching — like LZ compression)
# ---------------------------------------------------------------------------

class Historian(Algorithm):
    """Finds the longest matching suffix in past history.
    
    Inspired by Lempel-Ziv compression, it looks for the longest sequence in the past
    that matches the most recent moves, and predicts what the opponent played *next*
    in that historical context.
    
    **Type**: Pattern / Compression
    **Logic**: Longest Common Suffix
    """
    name = "Historian"

    def choose(self, round_num, my_history, opp_history):
        if len(opp_history) < 2:
            return self.rng.choice(MOVES)
        # Try decreasing context lengths
        max_ctx = min(10, len(opp_history) - 1)
        for ctx_len in range(max_ctx, 0, -1):
            context = tuple(opp_history[-ctx_len:])
            # Search for this context in earlier history
            for i in range(len(opp_history) - ctx_len):
                candidate = tuple(opp_history[i:i + ctx_len])
                if candidate == context and (i + ctx_len) < len(opp_history):
                    predicted = opp_history[i + ctx_len]
                    return _counter_move(predicted)
        return _counter_move(opp_history[-1])


# ---------------------------------------------------------------------------
# 23: Reverse Psychologist (2nd order thinking)
# ---------------------------------------------------------------------------

class ReversePsychologist(Algorithm):
    """Assumes the opponent is trying to counter YOUR last move.
    
    Thinks: "I played X -> opponent expects me to play X again -> opponent will play counter(X) -> I should play counter(counter(X))."
    This creates a 2-level reasoning chain that beats naive counter-last strategies.
    
    **Type**: Psychological / Level-K
    **Level**: 2
    """
    name = "Reverse Psychologist"

    def choose(self, round_num, my_history, opp_history):
        if not my_history:
            return self.rng.choice(MOVES)
        # Opponent will try to beat my last move
        what_beats_me = _counter_move(my_history[-1])
        # So I counter what they'll play
        return _counter_move(what_beats_me)


# ---------------------------------------------------------------------------
# 24: Echo (plays opponent's move from N rounds ago)
# ---------------------------------------------------------------------------

class Echo(Algorithm):
    """Copies the opponent's move from 3 rounds ago.
    
    Creates a temporal echo. Useful when opponents have cyclic patterns with a period
    greater than 1, and confusing for strategies that only look at the immediate last move.
    
    **Type**: Pattern / Delay
    **Delay**: 3 rounds
    """
    name = "Echo"

    def choose(self, round_num, my_history, opp_history):
        delay = 3
        if len(opp_history) < delay:
            return self.rng.choice(MOVES)
        return opp_history[-delay]


# ---------------------------------------------------------------------------
# 25: Trojan Horse (bait-and-switch)
# ---------------------------------------------------------------------------

class TrojanHorse(Algorithm):
    """Feeds a predictable pattern to bait the opponent, then switches.
    
    Plays Rock for the first 30 rounds to bait the opponent into playing Paper.
    Then abruptly switches to exploiting their adaptation.
    
    **Type**: Psychological / Trap
    **Bait Duration**: 30 rounds
    """
    name = "Trojan Horse"

    def reset(self):
        self._bait_rounds = 30

    def choose(self, round_num, my_history, opp_history):
        if round_num < self._bait_rounds:
            # Bait phase: play Rock predictably
            return Move.ROCK
        # Exploitation phase: counter opponent's recent adaptation
        if len(opp_history) >= 5:
            recent = Counter(opp_history[-10:])
            predicted = recent.most_common(1)[0][0]
            return _counter_move(predicted)
        return self.rng.choice(MOVES)


# ---------------------------------------------------------------------------
# 26: Reluctant Gambler (waits for statistical confidence)
# ---------------------------------------------------------------------------

class ReluctantGambler(Algorithm):
    """Plays random until statistically confident (Chi-Squared test).
    
    Uses a Chi-Squared test to detect if the opponent's move distribution is
    significantly biased. Only when confident (p<0.05) does it switch to
    exploitation mode.
    
    **Type**: Statistical / Hybrid
    **Threshold**: Chi-Sq > 5.99
    """
    name = "Reluctant Gambler"

    def choose(self, round_num, my_history, opp_history):
        if len(opp_history) < 15:
            return self.rng.choice(MOVES)
        # Use last 100 rounds for recency
        window = opp_history[-100:]
        counts = Counter(window)
        n = len(window)
        expected = n / 3.0
        # Chi-squared statistic
        chi_sq = sum((counts.get(m, 0) - expected) ** 2 / expected for m in MOVES)
        # Threshold ~5.99 for df=2, p=0.05
        if chi_sq > 5.99:
            # Statistically significant bias detected — exploit it
            predicted = counts.most_common(1)[0][0]
            return _counter_move(predicted)
        # Not confident — play random
        return self.rng.choice(MOVES)


# ---------------------------------------------------------------------------
# 27: Entropy Guardian (balances unpredictability vs exploitation)
# ---------------------------------------------------------------------------

class EntropyGuardian(Algorithm):
    """Monitors its OWN entropy to avoid becoming predictable.
    
    If its own move distribution becomes too low-entropy (predictable), it
    forces valid random moves to restore balance. Otherwise, it adapts to the opponent.
    
    **Type**: Defensive / Adaptive
    **Metric**: Shannon Entropy
    """
    name = "Entropy Guardian"

    def choose(self, round_num, my_history, opp_history):
        if len(my_history) < 5:
            return self.rng.choice(MOVES)
        # Calculate own entropy over last 30 moves
        import math
        window = my_history[-30:]
        counts = Counter(window)
        n = len(window)
        entropy = -sum(
            (c / n) * math.log2(c / n)
            for c in counts.values() if c > 0
        )
        max_entropy = math.log2(3)  # ~1.585
        # If own play is too predictable, force randomness
        if entropy < max_entropy * 0.7:
            return self.rng.choice(MOVES)
        # Otherwise exploit
        if opp_history:
            opp_counts = Counter(opp_history[-20:])
            predicted = opp_counts.most_common(1)[0][0]
            return _counter_move(predicted)
        return self.rng.choice(MOVES)


# ---------------------------------------------------------------------------
# 28: Second Guess (counter-counter reasoning)
# ---------------------------------------------------------------------------

class SecondGuess(Algorithm):
    """Counters the opponent's counter-strategy.
    
    Detects if the opponent is effectively countering our last moves. If so,
    it switches to a "counter-counter" logic. If not, it falls back to standard
    frequency analysis.
    
    **Type**: Meta / Adaptive
    **Logic**: Conditional Counter
    """
    name = "Second Guess"

    def reset(self):
        self._opponent_countered = 0
        self._total_checked = 0

    def choose(self, round_num, my_history, opp_history):
        if len(my_history) < 2:
            return self.rng.choice(MOVES)
        # Check if opponent has been countering our moves
        if len(opp_history) >= 2:
            expected_counter = _counter_move(my_history[-2])
            if opp_history[-1] == expected_counter:
                self._opponent_countered += 1
            self._total_checked += 1

        if self._total_checked > 10 and self._opponent_countered / self._total_checked > 0.4:
            # They're countering us — counter their counter
            their_next = _counter_move(my_history[-1])
            return _counter_move(their_next)
        else:
            # They're not countering — use frequency analysis
            if opp_history:
                counts = Counter(opp_history[-20:])
                predicted = counts.most_common(1)[0][0]
                return _counter_move(predicted)
            return self.rng.choice(MOVES)


# ---------------------------------------------------------------------------
# 29: Majority Rule (ensemble vote of 5 strategies)
# ---------------------------------------------------------------------------

class MajorityRule(Algorithm):
    """Ensemble of 5 strategies voting on the next move.
    
    Strategies: Counter-Last, Mirror, Frequency, Anti-Cycle, and Random.
    The move with the most votes is chosen. Robust against various opponents.
    
    **Type**: Ensemble / Voting
    **Size**: 5 Sub-strategies
    """
    name = "Majority Rule"

    def choose(self, round_num, my_history, opp_history):
        votes = []

        # Strategy 1: Counter opponent's last move
        if opp_history:
            votes.append(_counter_move(opp_history[-1]))
        else:
            votes.append(self.rng.choice(MOVES))

        # Strategy 2: Mirror opponent
        if opp_history:
            votes.append(opp_history[-1])
        else:
            votes.append(self.rng.choice(MOVES))

        # Strategy 3: Frequency counter
        if opp_history:
            counts = Counter(opp_history)
            votes.append(_counter_move(counts.most_common(1)[0][0]))
        else:
            votes.append(self.rng.choice(MOVES))

        # Strategy 4: Anti-cycle (predict cycle position)
        if len(opp_history) >= 3:
            # If opponent follows a cycle, predict next in sequence
            diffs = []
            for i in range(-1, -4, -1):
                idx = MOVES.index(opp_history[i])
                prev_idx = MOVES.index(opp_history[i - 1]) if abs(i - 1) <= len(opp_history) else 0
                diffs.append((idx - prev_idx) % 3)
            if len(set(diffs)) == 1:
                # Consistent stepping pattern detected
                next_idx = (MOVES.index(opp_history[-1]) + diffs[0]) % 3
                votes.append(_counter_move(MOVES[next_idx]))
            else:
                votes.append(self.rng.choice(MOVES))
        else:
            votes.append(self.rng.choice(MOVES))

        # Strategy 5: Random wildcard
        votes.append(self.rng.choice(MOVES))

        # Majority vote
        vote_counts = Counter(votes)
        return vote_counts.most_common(1)[0][0]


# ---------------------------------------------------------------------------
# 30: Phase Shifter (alternates aggressive / defensive phases)
# ---------------------------------------------------------------------------

class PhaseShifter(Algorithm):
    """Alternates between Aggressive (Exploit) and Defensive (Random) phases.
    
    40 rounds of aggressive exploitation followed by 20 rounds of pure randomness.
    The random phase 'washs' the history, confusing opponents that try to model it.
    
    **Type**: Temporal / Hybrid
    **Phases**: 40/20 rounds
    """
    name = "Phase Shifter"

    def choose(self, round_num, my_history, opp_history):
        cycle_pos = round_num % 60
        if cycle_pos < 40:
            # Aggressive phase
            if opp_history:
                recent = Counter(opp_history[-15:])
                predicted = recent.most_common(1)[0][0]
                return _counter_move(predicted)
            return self.rng.choice(MOVES)
        else:
            # Defensive phase — pure random
            return self.rng.choice(MOVES)


# ---------------------------------------------------------------------------
# 31: De Bruijn Walker (systematic sequence coverage)
# ---------------------------------------------------------------------------

class DeBruijnWalker(Algorithm):
    """Plays a De Bruijn sequence covering all 2-move combinations.
    
    Ensures that every pair of moves (RR, RP, RS, etc.) appears exactly once
    in the cycle. Hard for pattern detectors to find simple repetitions.
    Occasionally deviates to exploit strong biases.
    
    **Type**: Pattern / Mathematical
    **Sequence**: De Bruijn (n=2, k=3)
    """
    name = "De Bruijn Walker"

    def reset(self):
        # De Bruijn sequence for alphabet {0,1,2} with n=2
        # Covers all 9 possible 2-grams in a length-9 cycle
        self._sequence = [0, 0, 1, 0, 2, 1, 1, 2, 2]  # RRPRSPPSQ
        self._pos = 0

    def choose(self, round_num, my_history, opp_history):
        # 10% chance to exploit if strong bias detected
        if opp_history and len(opp_history) > 20 and self.rng.random() < 0.1:
            counts = Counter(opp_history[-20:])
            top_freq = counts.most_common(1)[0][1] / len(opp_history[-20:])
            if top_freq > 0.5:  # Strong bias
                return _counter_move(counts.most_common(1)[0][0])

        move = MOVES[self._sequence[self._pos % len(self._sequence)]]
        self._pos += 1
        return move


# ---------------------------------------------------------------------------
# 32: Iocaine Powder (multi-layered meta-reasoning)
# ---------------------------------------------------------------------------

class IocainePowder(Algorithm):
    """A famous meta-strategy from the programming contest era.
    
    Runs 6 meta-strategies (Naive, Double-Bluff, Triple-Bluff, etc.) and
    dynamically weights them based on recent success. Plays the prediction
    of the currently most successful meta-strategy.
    
    **Type**: Meta / Historic
    **Origin**: Dan Egnor (1999)
    """
    name = "Iocaine Powder"

    def reset(self):
        self._meta_scores = [0.0] * 6

    def choose(self, round_num, my_history, opp_history):
        if len(opp_history) < 3:
            return self.rng.choice(MOVES)

        # Generate 6 predictions of what opponent will play
        predictions = []

        # 1. Opponent repeats their last move
        predictions.append(opp_history[-1])
        # 2. Opponent counters our last move
        predictions.append(_counter_move(my_history[-1]))
        # 3. Opponent counters what beats their last (double bluff)
        predictions.append(_counter_move(_counter_move(opp_history[-1])))
        # 4. Opponent plays what THEY think beats what we'll play
        #    (assuming we counter their last)
        our_expected = _counter_move(opp_history[-1])
        predictions.append(_counter_move(our_expected))
        # 5. Opponent plays what beats our second-order guess
        our_second = _counter_move(_counter_move(my_history[-1]))
        predictions.append(_counter_move(our_second))
        # 6. Frequency-weighted prediction
        counts = Counter(opp_history[-10:])
        predictions.append(counts.most_common(1)[0][0])

        # Score each predictor based on how well it predicted last round
        if len(opp_history) >= 2:
            prev_opp = opp_history[-1]
            prev_preds = []
            prev_preds.append(opp_history[-2])  # 1
            prev_preds.append(_counter_move(my_history[-2]) if len(my_history) >= 2 else Move.ROCK)  # 2
            prev_preds.append(_counter_move(_counter_move(opp_history[-2])))  # 3
            prev_our_exp = _counter_move(opp_history[-2])
            prev_preds.append(_counter_move(prev_our_exp))  # 4
            prev_our_sec = _counter_move(_counter_move(my_history[-2]) if len(my_history) >= 2 else Move.ROCK)
            prev_preds.append(_counter_move(prev_our_sec))  # 5
            prev_counts = Counter(opp_history[-11:-1])
            prev_preds.append(prev_counts.most_common(1)[0][0] if prev_counts else Move.ROCK)  # 6

            decay = 0.95
            for i in range(6):
                self._meta_scores[i] *= decay
                if prev_preds[i] == prev_opp:
                    self._meta_scores[i] += 1.0

        # Pick best meta-strategy's prediction and counter it
        best_idx = self._meta_scores.index(max(self._meta_scores))
        predicted = predictions[best_idx]
        return _counter_move(predicted)


# ---------------------------------------------------------------------------
# 33: Intentional Loser (deliberately tries to lose)
# ---------------------------------------------------------------------------

def _losing_move(move: Move) -> Move:
    """Return the move that LOSES to `move`."""
    return BEATS[move]


class IntentionalLoser(Algorithm):
    """Deliberately tries to LOSE.
    
    Predicts what you will play and plays the move that loses to it.
    Useful for testing if other algorithms are working correctly (they should win 100%).
    
    **Type**: Testing / Debug
    **Goal**: 0% Win Rate
    """
    name = "Intentional Loser"

    def choose(self, round_num, my_history, opp_history):
        if not opp_history:
            # Round 0: lose to the most common opening (Rock)
            return Move.SCISSORS
        # Predict opponent's most likely move
        counts = Counter(opp_history)
        predicted = counts.most_common(1)[0][0]
        # Play the move that LOSES to it
        return _losing_move(predicted)


# ===========================================================================
#  RL / ML ALGORITHMS (34-37) — v4: Experience replay + outcome-aware state
# ===========================================================================


def _pretrain_against_archetypes(algo, rounds_per: int = 120):
    """Pre-train an RL algorithm via self-play against 5 archetypal opponents.

    The algorithm's own choose() method is called repeatedly, so it
    updates its internal state (Q-values, posteriors, etc.) naturally.
    """
    archetypes = [
        lambda opp, my: Move.ROCK,                                   # Always Rock
        lambda opp, my: Move.PAPER,                                  # Always Paper
        lambda opp, my: Move.SCISSORS,                               # Always Scissors
        lambda opp, my: MOVES[len(opp) % 3],                         # Cycle R→P→S
        lambda opp, my: my[-1] if my else Move.ROCK,                  # Mirror
    ]
    for opp_fn in archetypes:
        my_hist: list[Move] = []
        opp_hist: list[Move] = []
        for r in range(rounds_per):
            action = algo.choose(r, my_hist, opp_hist)
            opp_action = opp_fn(opp_hist, my_hist)
            my_hist.append(action)
            opp_hist.append(opp_action)
    # Reset round counter / epsilon so the real match starts fresh
    if hasattr(algo, '_rounds_played'):
        algo._rounds_played = 0
    if hasattr(algo, '_epsilon'):
        algo._epsilon = 0.15  # start lower since we're pre-trained


# ---------------------------------------------------------------------------
# 34: Q-Learner (tabular Q-Learning)
# ---------------------------------------------------------------------------

class QLearner(Algorithm):
    """Tabular Q-Learning with experience replay.
    
    State = (my_last, opp_last, opp_2nd_last, outcome).
    Uses experience replay (buffer 200, batch 10) to learn from past mistakes.
    Pre-trained against archetypes to avoid cold-start stupidity.
    
    **Type**: RL / Q-Learning
    **State Space**: ~81 states
    """
    name = "Q-Learner"

    def reset(self):
        self._q_table: dict[tuple, dict[Move, float]] = {}
        self._alpha = 0.2
        self._epsilon = 0.3
        self._last_state = None
        self._last_action = None
        self._rounds_played = 0
        self._replay_buffer: list[tuple] = []  # (state, action, reward)
        self._max_buffer = 200
        _pretrain_against_archetypes(self)

    def _get_state(self, my_history, opp_history):
        if not my_history:
            return ("START",)
        # Compute last-round outcome
        outcome = "D"  # draw
        if len(my_history) >= 1 and len(opp_history) >= 1:
            if BEATS[my_history[-1]] == opp_history[-1]:
                outcome = "W"
            elif my_history[-1] != opp_history[-1]:
                outcome = "L"
        if len(opp_history) < 2:
            return ("EARLY", my_history[-1], opp_history[-1], outcome)
        return (my_history[-1], opp_history[-1], opp_history[-2], outcome)

    def _get_q(self, state, action):
        if state not in self._q_table:
            self._q_table[state] = {m: 0.0 for m in MOVES}
        return self._q_table[state][action]

    def _set_q(self, state, action, value):
        if state not in self._q_table:
            self._q_table[state] = {m: 0.0 for m in MOVES}
        self._q_table[state][action] = value

    def choose(self, round_num, my_history, opp_history):
        # Update Q-values from last round's outcome
        if self._last_state is not None and len(my_history) >= 1:
            my_last = my_history[-1]
            opp_last = opp_history[-1]
            if BEATS[my_last] == opp_last:
                reward = 1.0
            elif my_last == opp_last:
                reward = 0.0
            else:
                reward = -1.0

            old_q = self._get_q(self._last_state, self._last_action)
            self._set_q(self._last_state, self._last_action,
                        old_q + self._alpha * (reward - old_q))

            # Store transition for experience replay
            self._replay_buffer.append(
                (self._last_state, self._last_action, reward))
            if len(self._replay_buffer) > self._max_buffer:
                self._replay_buffer.pop(0)

            # Experience replay: re-learn from 10 random past transitions
            if len(self._replay_buffer) >= 20:
                for _ in range(10):
                    s, a, r = self.rng.choice(self._replay_buffer)
                    oq = self._get_q(s, a)
                    self._set_q(s, a, oq + self._alpha * (r - oq))

        self._rounds_played += 1

        # Decay exploration: ε decays from 0.3 → 0.05 over 200 rounds
        self._epsilon = max(0.05, 0.3 * (0.99 ** self._rounds_played))

        state = self._get_state(my_history, opp_history)
        self._last_state = state

        # ε-greedy action selection
        if self.rng.random() < self._epsilon:
            action = self.rng.choice(MOVES)
        else:
            q_values = {m: self._get_q(state, m) for m in MOVES}
            max_q = max(q_values.values())
            best_actions = [m for m, q in q_values.items() if q == max_q]
            action = self.rng.choice(best_actions)

        self._last_action = action
        return action


# ---------------------------------------------------------------------------
# 35: Thompson Sampler (Bayesian bandit with Beta distributions)
# ---------------------------------------------------------------------------

class ThompsonSampler(Algorithm):
    """Bayesian multi-armed bandit using Beta distributions.
    
    Maintains a Beta(alpha, beta) distribution for each action in each state.
    Samples from these distributions to decide the next move, naturally balancing
    exploration and exploitation.
    
    **Type**: RL / Bayesian Bandit
    **Model**: Beta-Bernoulli
    """
    name = "Thompson Sampler"

    def reset(self):
        # Beta(α, β) params per (state, action)
        self._alpha: dict[tuple, dict[Move, float]] = {}
        self._beta: dict[tuple, dict[Move, float]] = {}
        self._last_state = None
        self._last_action = None
        _pretrain_against_archetypes(self)

    def _get_state(self, my_history, opp_history):
        if not my_history:
            return ("START",)
        outcome = "D"
        if len(my_history) >= 1 and len(opp_history) >= 1:
            if BEATS[my_history[-1]] == opp_history[-1]:
                outcome = "W"
            elif my_history[-1] != opp_history[-1]:
                outcome = "L"
        if len(opp_history) < 2:
            return ("EARLY", opp_history[-1], outcome)
        return (my_history[-1], opp_history[-1], opp_history[-2], outcome)

    def _ensure_state(self, state):
        if state not in self._alpha:
            self._alpha[state] = {m: 1.0 for m in MOVES}
            self._beta[state] = {m: 1.0 for m in MOVES}

    def choose(self, round_num, my_history, opp_history):
        import math

        # Update from last round
        if self._last_state is not None and len(my_history) >= 1:
            my_last = my_history[-1]
            opp_last = opp_history[-1]
            self._ensure_state(self._last_state)
            if BEATS[my_last] == opp_last:
                self._alpha[self._last_state][self._last_action] += 1.0
            elif my_last != opp_last:
                self._beta[self._last_state][self._last_action] += 1.0
            # Draws: slight win credit
            else:
                self._alpha[self._last_state][self._last_action] += 0.5
                self._beta[self._last_state][self._last_action] += 0.5

        state = self._get_state(my_history, opp_history)
        self._ensure_state(state)
        self._last_state = state

        # Sample from Beta posteriors
        samples = {}
        for m in MOVES:
            a = self._alpha[state][m]
            b = self._beta[state][m]
            samples[m] = self.rng.betavariate(a, b)

        action = max(samples, key=samples.get)
        self._last_action = action
        return action


# ---------------------------------------------------------------------------
# 36: UCB Explorer (Upper Confidence Bound)
# ---------------------------------------------------------------------------

class UCBExplorer(Algorithm):
    """Upper Confidence Bound (UCB1) bandit algorithm.
    
    Selects actions based on their average reward plus an uncertainty bonus.
    Optimizes the trade-off between exploiting known winning moves and exploring
    statistically uncertain ones.
    
    **Type**: RL / Bandit
    **Formula**: UCB1
    """
    name = "UCB Explorer"

    def reset(self):
        self._counts: dict[tuple, dict[Move, int]] = {}
        self._rewards: dict[tuple, dict[Move, float]] = {}
        self._total: dict[tuple, int] = {}
        self._last_state = None
        self._last_action = None
        _pretrain_against_archetypes(self)

    def _get_state(self, my_history, opp_history):
        if not my_history:
            return ("START",)
        outcome = "D"
        if len(my_history) >= 1 and len(opp_history) >= 1:
            if BEATS[my_history[-1]] == opp_history[-1]:
                outcome = "W"
            elif my_history[-1] != opp_history[-1]:
                outcome = "L"
        if len(opp_history) < 2:
            return ("EARLY", opp_history[-1], outcome)
        return (my_history[-1], opp_history[-1], opp_history[-2], outcome)

    def _ensure_state(self, state):
        if state not in self._counts:
            self._counts[state] = {m: 0 for m in MOVES}
            self._rewards[state] = {m: 0.0 for m in MOVES}
            self._total[state] = 0

    def choose(self, round_num, my_history, opp_history):
        import math

        # Update from last round
        if self._last_state is not None and len(my_history) >= 1:
            my_last = my_history[-1]
            opp_last = opp_history[-1]
            if BEATS[my_last] == opp_last:
                reward = 1.0
            elif my_last == opp_last:
                reward = 0.5
            else:
                reward = 0.0
            self._ensure_state(self._last_state)
            self._rewards[self._last_state][self._last_action] += reward
            self._counts[self._last_state][self._last_action] += 1
            self._total[self._last_state] += 1

        state = self._get_state(my_history, opp_history)
        self._ensure_state(state)
        self._last_state = state

        c = math.sqrt(2)
        n_total = self._total[state]

        # If any action hasn't been tried, try it
        for m in MOVES:
            if self._counts[state][m] == 0:
                self._last_action = m
                return m

        # UCB1 formula
        ucb_values = {}
        for m in MOVES:
            n_a = self._counts[state][m]
            avg_reward = self._rewards[state][m] / n_a
            exploration = c * math.sqrt(math.log(n_total) / n_a)
            ucb_values[m] = avg_reward + exploration

        action = max(ucb_values, key=ucb_values.get)
        self._last_action = action
        return action


# ---------------------------------------------------------------------------
# 37: Gradient Learner (softmax policy gradient)
# ---------------------------------------------------------------------------

class GradientLearner(Algorithm):
    """Policy Gradient method with Softmax action selection.
    
    Maintains numerical preferences for each action. Updates these preferences
    via gradient ascent on the expected reward. Includes entropy regularization
    to prevent premature convergence to suboptimal strategies.
    
    **Type**: RL / Policy Gradient
    **Policy**: Softmax
    """
    name = "Gradient Learner"

    def reset(self):
        self._preferences: dict[tuple, dict[Move, float]] = {}
        self._avg_reward = 0.0
        self._reward_count = 0
        self._last_state = None
        self._last_action = None
        self._last_probs: dict[Move, float] = {}
        _pretrain_against_archetypes(self)

    def _get_state(self, my_history, opp_history):
        if not my_history:
            return ("START",)
        outcome = "D"
        if len(my_history) >= 1 and len(opp_history) >= 1:
            if BEATS[my_history[-1]] == opp_history[-1]:
                outcome = "W"
            elif my_history[-1] != opp_history[-1]:
                outcome = "L"
        if len(opp_history) < 2:
            return ("EARLY", my_history[-1], opp_history[-1], outcome)
        return (my_history[-1], opp_history[-1], opp_history[-2], outcome)

    def _ensure_state(self, state):
        if state not in self._preferences:
            self._preferences[state] = {m: 0.0 for m in MOVES}

    def _softmax(self, state):
        import math
        self._ensure_state(state)
        h = self._preferences[state]
        max_h = max(h.values())
        exp_h = {m: math.exp(h[m] - max_h) for m in MOVES}
        total = sum(exp_h.values())
        return {m: exp_h[m] / total for m in MOVES}

    def choose(self, round_num, my_history, opp_history):
        alpha = 0.1  # learning rate

        # Update from last round
        if self._last_state is not None and len(my_history) >= 1:
            my_last = my_history[-1]
            opp_last = opp_history[-1]
            if BEATS[my_last] == opp_last:
                reward = 1.0
            elif my_last == opp_last:
                reward = 0.0
            else:
                reward = -1.0

            # Update baseline
            self._reward_count += 1
            self._avg_reward += (reward - self._avg_reward) / self._reward_count

            # Gradient update
            advantage = reward - self._avg_reward
            self._ensure_state(self._last_state)
            for m in MOVES:
                if m == self._last_action:
                    self._preferences[self._last_state][m] += (
                        alpha * advantage * (1 - self._last_probs[m])
                    )
                else:
                    self._preferences[self._last_state][m] -= (
                        alpha * advantage * self._last_probs[m]
                    )

        state = self._get_state(my_history, opp_history)
        self._last_state = state

        probs = self._softmax(state)
        self._last_probs = probs

        # Sample from policy
        r = self.rng.random()
        cumulative = 0.0
        for m in MOVES:
            cumulative += probs[m]
            if r <= cumulative:
                self._last_action = m
                return m

        self._last_action = MOVES[-1]
        return MOVES[-1]


# ===========================================================================
#  RL v5: LINEAR FUNCTION APPROXIMATION (63-66 — new classes, v4 kept intact)
# ===========================================================================


def _build_feature_vector(my_history, opp_history, action, window: int = 30):
    """Build a 16-dimensional feature vector for linear RL algorithms.

    Features:
      [0-2]   Action one-hot (R, P, S)
      [3-5]   Opponent's last move one-hot (R, P, S)
      [6-8]   Opponent frequency bias over last `window` rounds
      [9-11]  Opponent transition probability from last move
      [12-14] Last outcome one-hot (W, L, D)
      [15]    Bias term (always 1.0)
    """
    phi = [0.0] * 16

    # Action one-hot
    phi[MOVES.index(action)] = 1.0

    # Opponent's last move one-hot
    if opp_history:
        phi[3 + MOVES.index(opp_history[-1])] = 1.0

    # Opponent frequency bias (last `window` rounds)
    if opp_history:
        recent = opp_history[-window:] if len(opp_history) > window else opp_history
        n = len(recent)
        for m in recent:
            phi[6 + MOVES.index(m)] += 1.0 / n

    # Opponent transition probability from last move
    if len(opp_history) >= 2:
        last_opp = opp_history[-1]
        transitions = [0, 0, 0]
        total_trans = 0
        for k in range(len(opp_history) - 1):
            if opp_history[k] == last_opp:
                transitions[MOVES.index(opp_history[k + 1])] += 1
                total_trans += 1
        if total_trans > 0:
            for j in range(3):
                phi[9 + j] = transitions[j] / total_trans

    # Last outcome one-hot
    if my_history and opp_history:
        if BEATS[my_history[-1]] == opp_history[-1]:
            phi[12] = 1.0  # WIN
        elif my_history[-1] != opp_history[-1]:
            phi[13] = 1.0  # LOSS
        else:
            phi[14] = 1.0  # DRAW

    # Bias
    phi[15] = 1.0
    return phi


def _dot(w, phi):
    """Dot product of two lists."""
    return sum(a * b for a, b in zip(w, phi))


# ---------------------------------------------------------------------------
# 63: Q-Learner v5 (Linear Function Approximation)
# ---------------------------------------------------------------------------

class QLearnerV5(Algorithm):
    """Q-Learning with Linear Function Approximation.
    
    Replaces the tabular Q-table with a linear model: Q(s,a) = w * phi(s,a).
    Uses a 16-dimensional feature vector (one-hot actions, outcome, frequency bias, etc.)
    to generalize across states, allowing it to learn faster in complex environments.
    
    **Type**: RL / Linear Approx
    **Features**: 16-dim vector
    """
    name = "Q-Learner v5"

    def reset(self):
        self._w = [0.0] * 48  # 16 features × 3 actions
        self._alpha = 0.05
        self._epsilon = 0.3
        self._rounds_played = 0
        self._last_action = None
        self._last_phi_idx = None  # (start_idx, phi)
        self._replay_buffer: list[tuple] = []
        self._max_buffer = 200
        _pretrain_against_archetypes(self)

    def _q_value(self, my_history, opp_history, action):
        phi = _build_feature_vector(my_history, opp_history, action)
        idx = MOVES.index(action) * 16
        return _dot(self._w[idx:idx+16], phi), phi, idx

    def _sgd_update(self, start_idx, phi, delta):
        for i in range(16):
            self._w[start_idx + i] += self._alpha * delta * phi[i]

    def choose(self, round_num, my_history, opp_history):
        # Update from last round
        if self._last_action is not None and len(my_history) >= 1:
            my_last = my_history[-1]
            opp_last = opp_history[-1]
            if BEATS[my_last] == opp_last:
                reward = 1.0
            elif my_last == opp_last:
                reward = 0.0
            else:
                reward = -1.0

            old_q = _dot(
                self._w[self._last_phi_idx[0]:self._last_phi_idx[0]+16],
                self._last_phi_idx[1]
            )
            delta = reward - old_q
            self._sgd_update(self._last_phi_idx[0], self._last_phi_idx[1], delta)

            # Experience replay
            self._replay_buffer.append(
                (self._last_phi_idx[0], self._last_phi_idx[1], reward))
            if len(self._replay_buffer) > self._max_buffer:
                self._replay_buffer.pop(0)
            if len(self._replay_buffer) >= 20:
                for _ in range(10):
                    s_idx, s_phi, r = self.rng.choice(self._replay_buffer)
                    oq = _dot(self._w[s_idx:s_idx+16], s_phi)
                    self._sgd_update(s_idx, s_phi, r - oq)

        self._rounds_played += 1
        self._epsilon = max(0.05, 0.3 * (0.99 ** self._rounds_played))

        # ε-greedy
        if self.rng.random() < self._epsilon:
            action = self.rng.choice(MOVES)
            _, phi, idx = self._q_value(my_history, opp_history, action)
        else:
            best_q, best_action, best_phi, best_idx = -1e9, MOVES[0], None, 0
            for m in MOVES:
                q, phi, idx = self._q_value(my_history, opp_history, m)
                if q > best_q:
                    best_q, best_action, best_phi, best_idx = q, m, phi, idx
            action, phi, idx = best_action, best_phi, best_idx

        self._last_action = action
        self._last_phi_idx = (idx, phi)
        return action


# ---------------------------------------------------------------------------
# 64: Thompson Sampler v5 (Bayesian Linear Regression — pure Python)
# ---------------------------------------------------------------------------

def _mat_zeros(n):
    """Create n×n zero matrix."""
    return [[0.0] * n for _ in range(n)]

def _mat_identity(n):
    """Create n×n identity matrix."""
    m = _mat_zeros(n)
    for i in range(n):
        m[i][i] = 1.0
    return m

def _mat_add_outer(A, v):
    """A += v·vᵀ (rank-1 update, in-place)."""
    n = len(v)
    for i in range(n):
        for j in range(n):
            A[i][j] += v[i] * v[j]

def _mat_vec_mul(A, v):
    """Return A·v."""
    n = len(v)
    result = [0.0] * n
    for i in range(n):
        for j in range(n):
            result[i] += A[i][j] * v[j]
    return result

def _mat_inverse(A):
    """Gauss-Jordan inversion of n×n matrix. Returns A⁻¹."""
    n = len(A)
    # Augmented matrix [A | I]
    aug = [row[:] + [1.0 if j == i else 0.0 for j in range(n)] for i, row in enumerate(A)]

    for col in range(n):
        # Find pivot
        max_row = col
        for row in range(col + 1, n):
            if abs(aug[row][col]) > abs(aug[max_row][col]):
                max_row = row
        aug[col], aug[max_row] = aug[max_row], aug[col]

        pivot = aug[col][col]
        if abs(pivot) < 1e-12:
            aug[col][col] = 1e-6  # regularise
            pivot = 1e-6

        for j in range(2 * n):
            aug[col][j] /= pivot

        for row in range(n):
            if row == col:
                continue
            factor = aug[row][col]
            for j in range(2 * n):
                aug[row][j] -= factor * aug[col][j]

    return [row[n:] for row in aug]

def _cholesky_lower(A):
    """Cholesky decomposition: returns L such that A = L·Lᵀ.

    Falls back to diagonal √A[i][i] if matrix is not positive-definite.
    """
    import math
    n = len(A)
    L = _mat_zeros(n)
    for i in range(n):
        for j in range(i + 1):
            s = sum(L[i][k] * L[j][k] for k in range(j))
            if i == j:
                val = A[i][i] - s
                L[i][j] = math.sqrt(max(val, 1e-10))
            else:
                if abs(L[j][j]) < 1e-12:
                    L[i][j] = 0.0
                else:
                    L[i][j] = (A[i][j] - s) / L[j][j]
    return L


class ThompsonSamplerV5(Algorithm):
    """Thompson Sampling with Bayesian Linear Regression.
    
    Models the reward function as a linear combination of features.
    Maintains a posterior distribution over the weight vector w for each action.
    Samples weights to estimate Q-values, balancing exploration/exploitation via uncertainty.
    
    **Type**: RL / Bayesian Linear
    **Math**: Cholesky Decomposition
    """
    name = "Thompson Sampler v5"

    def reset(self):
        d = 16  # feature dimension
        lam = 1.0  # prior precision
        # Per action: A = Σ⁻¹ (precision), b = Σ φr (moment)
        self._A = [[[lam if i == j else 0.0 for j in range(d)] for i in range(d)] for _ in range(3)]
        self._b = [[0.0] * d for _ in range(3)]
        self._d = d
        self._last_action = None
        self._last_phi = None
        _pretrain_against_archetypes(self)

    def choose(self, round_num, my_history, opp_history):
        d = self._d

        # Update from last round
        if self._last_action is not None and len(my_history) >= 1:
            my_last = my_history[-1]
            opp_last = opp_history[-1]
            if BEATS[my_last] == opp_last:
                reward = 1.0
            elif my_last == opp_last:
                reward = 0.0
            else:
                reward = -1.0

            a_idx = MOVES.index(self._last_action)
            phi = self._last_phi
            # A += φφᵀ
            _mat_add_outer(self._A[a_idx], phi)
            # b += r·φ
            for i in range(d):
                self._b[a_idx][i] += reward * phi[i]

        # Thompson sampling: sample weights from posterior, pick best
        best_q, best_action = -1e9, MOVES[0]
        best_phi = None
        for m in MOVES:
            phi = _build_feature_vector(my_history, opp_history, m)
            a_idx = MOVES.index(m)
            A_inv = _mat_inverse(self._A[a_idx])
            mu = _mat_vec_mul(A_inv, self._b[a_idx])

            # Sample: w ~ N(mu, A_inv) via Cholesky
            L = _cholesky_lower(A_inv)
            z = [self.rng.gauss(0, 1) for _ in range(d)]
            w_sample = [mu[i] + sum(L[i][j] * z[j] for j in range(i + 1)) for i in range(d)]

            q = _dot(w_sample, phi)
            if q > best_q:
                best_q, best_action, best_phi = q, m, phi

        self._last_action = best_action
        self._last_phi = best_phi
        return best_action


# ---------------------------------------------------------------------------
# 65: UCB Explorer v5 (LinUCB — contextual bandits, pure Python)
# ---------------------------------------------------------------------------

class UCBExplorerV5(Algorithm):
    """LinUCB (Contextual Bandit with Linear Payoff).
    
    Uses Ridge Regression to estimate the expected reward of each action given the
    current context (features). Adds an exploration bonus based on the confidence
    interval of the estimate.
    
    **Type**: RL / Contextual Bandit
    **Algorithm**: LinUCB
    """
    name = "UCB Explorer v5"

    def reset(self):
        d = 16
        self._d = d
        self._alpha_ucb = 1.5  # exploration coefficient
        self._A = [[[1.0 if i == j else 0.0 for j in range(d)] for i in range(d)] for _ in range(3)]
        self._b = [[0.0] * d for _ in range(3)]
        self._last_action = None
        self._last_phi = None
        _pretrain_against_archetypes(self)

    def choose(self, round_num, my_history, opp_history):
        import math
        d = self._d

        # Update from last round
        if self._last_action is not None and len(my_history) >= 1:
            my_last = my_history[-1]
            opp_last = opp_history[-1]
            if BEATS[my_last] == opp_last:
                reward = 1.0
            elif my_last == opp_last:
                reward = 0.5
            else:
                reward = 0.0

            a_idx = MOVES.index(self._last_action)
            phi = self._last_phi
            _mat_add_outer(self._A[a_idx], phi)
            for i in range(d):
                self._b[a_idx][i] += reward * phi[i]

        # LinUCB action selection
        best_ucb, best_action = -1e9, MOVES[0]
        best_phi = None
        for m in MOVES:
            phi = _build_feature_vector(my_history, opp_history, m)
            a_idx = MOVES.index(m)
            A_inv = _mat_inverse(self._A[a_idx])
            theta = _mat_vec_mul(A_inv, self._b[a_idx])
            exploit = _dot(theta, phi)
            # Exploration bonus: α·√(φᵀ·A⁻¹·φ)
            A_inv_phi = _mat_vec_mul(A_inv, phi)
            explore = self._alpha_ucb * math.sqrt(max(0, _dot(phi, A_inv_phi)))
            ucb = exploit + explore
            if ucb > best_ucb:
                best_ucb, best_action, best_phi = ucb, m, phi

        self._last_action = best_action
        self._last_phi = best_phi
        return best_action


# ---------------------------------------------------------------------------
# 66: Gradient Learner v5 (Linear Softmax REINFORCE)
# ---------------------------------------------------------------------------

class GradientLearnerV5(Algorithm):
    """Policy Gradient with Linear Function Approximation.
    
    Parametrizes the policy as a softmax over linear preferences: pi(a|s) = softmax(w * phi(s,a)).
    Updates weights via REINFORCE gradient ascent. capable of handling continuous state features.
    
    **Type**: RL / Policy Gradient
    **Features**: 16-dim
    """
    name = "Gradient Learner v5"

    def reset(self):
        self._w = [[0.0] * 16 for _ in range(3)]  # weights per action
        self._alpha = 0.02
        self._avg_reward = 0.0
        self._reward_count = 0
        self._last_action = None
        self._last_phi = None
        self._last_probs = None
        _pretrain_against_archetypes(self)

    def _softmax_probs(self, my_history, opp_history):
        import math
        hs = []
        phis = []
        for m in MOVES:
            phi = _build_feature_vector(my_history, opp_history, m)
            phis.append(phi)
            hs.append(_dot(self._w[MOVES.index(m)], phi))
        max_h = max(hs)
        exp_h = [math.exp(h - max_h) for h in hs]
        total = sum(exp_h)
        probs = [e / total for e in exp_h]
        return probs, phis

    def choose(self, round_num, my_history, opp_history):
        # Update from last round
        if self._last_action is not None and len(my_history) >= 1:
            my_last = my_history[-1]
            opp_last = opp_history[-1]
            if BEATS[my_last] == opp_last:
                reward = 1.0
            elif my_last == opp_last:
                reward = 0.0
            else:
                reward = -1.0

            self._reward_count += 1
            self._avg_reward += (reward - self._avg_reward) / self._reward_count
            advantage = reward - self._avg_reward

            a_idx = MOVES.index(self._last_action)
            for i, m in enumerate(MOVES):
                for j in range(16):
                    if i == a_idx:
                        self._w[i][j] += (
                            self._alpha * advantage
                            * (1 - self._last_probs[i]) * self._last_phi[j]
                        )
                    else:
                        self._w[i][j] -= (
                            self._alpha * advantage
                            * self._last_probs[i] * self._last_phi[j]
                        )

        probs, phis = self._softmax_probs(my_history, opp_history)
        self._last_probs = probs

        # Sample from policy
        r = self.rng.random()
        cumulative = 0.0
        chosen_idx = len(MOVES) - 1
        for i, p in enumerate(probs):
            cumulative += p
            if r <= cumulative:
                chosen_idx = i
                break

        action = MOVES[chosen_idx]
        self._last_action = action
        self._last_phi = phis[chosen_idx]
        return action


# ===========================================================================
#  ADVANCED COMPETITIVE ALGORITHMS (38-41)
# ===========================================================================


# ---------------------------------------------------------------------------
# 38: Bayesian Predictor (Dirichlet prior with sliding window)
# ---------------------------------------------------------------------------

class BayesianPredictor(Algorithm):
    """Maintains a Dirichlet prior over the opponent's move distribution.
    
    Uses a sliding window of the last 50 moves. Samples from the posterior Dirichlet
    distribution to estimate probabilities, then counters the most likely move.
    Robust against noise.
    
    **Type**: Probabilistic / Bayesian
    **Prior**: Dirichlet(1,1,1)
    """
    name = "Bayesian Predictor"

    def choose(self, round_num, my_history, opp_history):
        if not opp_history:
            return self.rng.choice(MOVES)

        window = opp_history[-50:]
        counts = Counter(window)

        # Dirichlet posterior parameters (uninformative prior α=1)
        alphas = {m: counts.get(m, 0) + 1.0 for m in MOVES}

        # Sample from Dirichlet by sampling independent Gammas
        samples = {}
        for m in MOVES:
            samples[m] = self.rng.gammavariate(alphas[m], 1.0)
        total = sum(samples.values())
        probs = {m: samples[m] / total for m in MOVES}

        # Counter the most probable move from the sample
        predicted = max(probs, key=probs.get)
        return _counter_move(predicted)


# ---------------------------------------------------------------------------
# 39: N-Gram Predictor (joint history n-grams)
# ---------------------------------------------------------------------------

class NGramPredictor(Algorithm):
    """Builds an N-Gram model of the JOINT history.
    
    Looks at sequences of (MyMove, OppMove) pairs.
    Example: "When I play Rock and they play Paper..." -> what happens next?
    Captures interactive patterns that single-history predictors miss.
    
    **Type**: Pattern / N-Gram
    **Context**: Joint History
    """
    name = "N-Gram Predictor"

    def choose(self, round_num, my_history, opp_history):
        if len(opp_history) < 2:
            return self.rng.choice(MOVES)

        # Build joint history: list of (my_move, opp_move) tuples
        joint = list(zip(my_history, opp_history))

        # Try decreasing n-gram lengths
        for n in [3, 2, 1]:
            if len(joint) < n + 1:
                continue

            context = tuple(joint[-n:])
            # Count what opponent played after this context
            next_counts = Counter()
            for i in range(len(joint) - n):
                candidate = tuple(joint[i:i + n])
                if candidate == context and (i + n) < len(opp_history):
                    next_counts[opp_history[i + n]] += 1

            if next_counts:
                predicted = next_counts.most_common(1)[0][0]
                return _counter_move(predicted)

        # Fallback
        return _counter_move(opp_history[-1])


# ---------------------------------------------------------------------------
# 40: Anti-Strategy Detector (archetype identification)
# ---------------------------------------------------------------------------

class AntiStrategyDetector(Algorithm):
    """Identifies and counters specific opponent archetypes.
    
    Runs multiple "detectors" (Cycle, Mirror, Counter, Frequency) in parallel.
    If one detector acts as a good predictor of the opponent's moves, this bot
    switches to the specific hard counter for that archetype.
    
    **Type**: Meta / Adaptive
    **Detects**: 5 archetypes
    """
    name = "Anti-Strategy Detector"

    def reset(self):
        self._detector_scores = [0.0] * 5

    def choose(self, round_num, my_history, opp_history):
        if len(opp_history) < 5:
            return self.rng.choice(MOVES)

        window = min(20, len(opp_history) - 1)

        # Generate predictions from each detector for past moves
        predictions = [None] * 5

        # --- Predict NEXT move with each detector ---

        # 0: Constant detector — opponent plays same as last
        predictions[0] = opp_history[-1]

        # 1: Cycle detector — opponent follows R→P→S cycle
        cycle_pos = MOVES.index(opp_history[-1])
        predictions[1] = MOVES[(cycle_pos + 1) % 3]

        # 2: Mirror detector — opponent copies MY last move
        predictions[2] = my_history[-1] if my_history else Move.ROCK

        # 3: Counter detector — opponent counters MY last move
        predictions[3] = _counter_move(my_history[-1]) if my_history else Move.ROCK

        # 4: Frequency detector — opponent counters my most common
        my_counts = Counter(my_history)
        if my_counts:
            predictions[4] = _counter_move(my_counts.most_common(1)[0][0])
        else:
            predictions[4] = Move.ROCK

        # Score each detector on recent accuracy
        decay = 0.9
        for i in range(5):
            self._detector_scores[i] *= decay

        if len(opp_history) >= 2 and len(my_history) >= 2:
            actual = opp_history[-1]

            # What would each detector have predicted for this round?
            retro_preds = []
            retro_preds.append(opp_history[-2])  # constant
            retro_preds.append(MOVES[(MOVES.index(opp_history[-2]) + 1) % 3])  # cycle
            retro_preds.append(my_history[-2] if len(my_history) >= 2 else Move.ROCK)  # mirror
            retro_preds.append(_counter_move(my_history[-2]) if len(my_history) >= 2 else Move.ROCK)  # counter
            my_past_counts = Counter(my_history[:-1])
            retro_preds.append(
                _counter_move(my_past_counts.most_common(1)[0][0]) if my_past_counts else Move.ROCK
            )  # frequency

            for i, pred in enumerate(retro_preds):
                if pred == actual:
                    self._detector_scores[i] += 1.0

        # Use best detector's prediction
        best = self._detector_scores.index(max(self._detector_scores))
        return _counter_move(predictions[best])


# ---------------------------------------------------------------------------
# 41: Mixture Model (Hedge / multiplicative weights)
# ---------------------------------------------------------------------------

class MixtureModel(Algorithm):
    """Multiplicative Weights (Hedge) algorithm.
    
    Maintains weights for 5 expert strategies (Counter, Frequency, Markov, WSLS, Random).
    Updates weights based on their performance every round. Theoretically achieves low regret.
    
    **Type**: Meta / Learning
    **Algorithm**: Hedge (Exp3-like)
    """
    name = "Mixture Model"

    def reset(self):
        self._weights = [1.0] * 5
        self._eta = 0.15  # learning rate
        self._last_votes: list[Move] = []

    def choose(self, round_num, my_history, opp_history):
        # Update weights from last round outcomes
        if self._last_votes and len(opp_history) >= 1:
            opp_last = opp_history[-1]
            for i, vote in enumerate(self._last_votes):
                if BEATS[vote] == opp_last:
                    loss = 0.0  # expert won
                elif vote == opp_last:
                    loss = 0.5  # draw
                else:
                    loss = 1.0  # expert lost
                self._weights[i] *= (1.0 - self._eta * loss)
            # Normalize
            w_sum = sum(self._weights)
            if w_sum > 0:
                self._weights = [w / w_sum for w in self._weights]
            else:
                self._weights = [0.2] * 5

        # Generate expert votes
        votes = []

        # Expert 0: Counter last move
        if opp_history:
            votes.append(_counter_move(opp_history[-1]))
        else:
            votes.append(self.rng.choice(MOVES))

        # Expert 1: Frequency counter
        if opp_history:
            counts = Counter(opp_history)
            votes.append(_counter_move(counts.most_common(1)[0][0]))
        else:
            votes.append(self.rng.choice(MOVES))

        # Expert 2: Markov-like (counter what they played after their last move)
        if len(opp_history) >= 2:
            last = opp_history[-1]
            transitions = Counter()
            for j in range(len(opp_history) - 1):
                if opp_history[j] == last:
                    transitions[opp_history[j + 1]] += 1
            if transitions:
                votes.append(_counter_move(transitions.most_common(1)[0][0]))
            else:
                votes.append(_counter_move(last))
        else:
            votes.append(self.rng.choice(MOVES))

        # Expert 3: Win-Stay Lose-Shift
        if my_history:
            my_last = my_history[-1]
            opp_last = opp_history[-1] if opp_history else Move.ROCK
            if BEATS[my_last] == opp_last or my_last == opp_last:
                votes.append(my_last)
            else:
                votes.append(_counter_move(opp_last))
        else:
            votes.append(self.rng.choice(MOVES))

        # Expert 4: Random
        votes.append(self.rng.choice(MOVES))

        self._last_votes = votes

        # Weighted vote: accumulate weight for each move
        move_weights = {m: 0.0 for m in MOVES}
        for i, vote in enumerate(votes):
            move_weights[vote] += self._weights[i]

        # Pick move with highest accumulated weight
        return max(move_weights, key=move_weights.get)


# ===========================================================================
#  CREATIVE / DECEPTION / DIVERSE ALGORITHMS (42-49)
# ===========================================================================


# ---------------------------------------------------------------------------
# 42: Sleeper Agent (dormant → activated)
# ---------------------------------------------------------------------------

class SleeperAgent(Algorithm):
    """Dormant for 80 rounds, then activates with high-quality data.
    
    Plays purely randomly for the first 80 rounds to collect "clean" data on how
    the opponent plays against a random bot (unbiased distribution). Then activates
    a strong ensemble predictor on this clean dataset.
    
    **Type**: Deceptive / Trap
    **Wake-up**: Round 80
    """
    name = "Sleeper Agent"
    _DORMANT_ROUNDS = 80

    def choose(self, round_num, my_history, opp_history):
        # Phase 1: Dormant — collect clean data
        if round_num < self._DORMANT_ROUNDS:
            return self.rng.choice(MOVES)

        # Phase 2: Active — exploit with best predictor
        predictions = []

        # Frequency prediction
        counts = Counter(opp_history[-50:])
        predictions.append(counts.most_common(1)[0][0])

        # Markov prediction
        if len(opp_history) >= 2:
            last = opp_history[-1]
            transitions = Counter()
            for j in range(len(opp_history) - 1):
                if opp_history[j] == last:
                    transitions[opp_history[j + 1]] += 1
            if transitions:
                predictions.append(transitions.most_common(1)[0][0])

        # Pattern prediction (last 3)
        if len(opp_history) >= 4:
            pattern = tuple(opp_history[-3:])
            for i in range(len(opp_history) - 3):
                if tuple(opp_history[i:i + 3]) == pattern:
                    if i + 3 < len(opp_history):
                        predictions.append(opp_history[i + 3])

        # Voting
        if predictions:
            vote_counts = Counter(predictions)
            predicted = vote_counts.most_common(1)[0][0]
            return _counter_move(predicted)

        return self.rng.choice(MOVES)


# ---------------------------------------------------------------------------
# 43: Shapeshifter (rotates entire strategy every N rounds)
# ---------------------------------------------------------------------------

class Shapeshifter(Algorithm):
    """Rotates its entire strategy every 40 rounds.
    
    Cycles through 5 distinct strategies (Random, Counter, Frequency, Markov, WSLS)
    in fixed 40-round blocks. Prevents the opponent from settling on a counter-strategy.
    
    **Type**: Temporal / Hybrid
    **Phase Length**: 40 rounds
    """
    name = "Shapeshifter"
    _PHASE_LEN = 40

    def choose(self, round_num, my_history, opp_history):
        phase = (round_num // self._PHASE_LEN) % 5

        if phase == 0:
            # Pure Random
            return self.rng.choice(MOVES)

        if phase == 1:
            # Counter-last
            if opp_history:
                return _counter_move(opp_history[-1])
            return self.rng.choice(MOVES)

        if phase == 2:
            # Frequency counter (window)
            if opp_history:
                counts = Counter(opp_history[-30:])
                return _counter_move(counts.most_common(1)[0][0])
            return self.rng.choice(MOVES)

        if phase == 3:
            # Markov
            if len(opp_history) >= 2:
                last = opp_history[-1]
                transitions = Counter()
                for j in range(len(opp_history) - 1):
                    if opp_history[j] == last:
                        transitions[opp_history[j + 1]] += 1
                if transitions:
                    return _counter_move(transitions.most_common(1)[0][0])
            return self.rng.choice(MOVES)

        # phase == 4: Win-Stay Lose-Shift
        if my_history and opp_history:
            if BEATS[my_history[-1]] == opp_history[-1] or my_history[-1] == opp_history[-1]:
                return my_history[-1]
            return _counter_move(opp_history[-1])
        return self.rng.choice(MOVES)


# ---------------------------------------------------------------------------
# 44: Hot Streak (momentum-based play)
# ---------------------------------------------------------------------------

class HotStreak(Algorithm):
    """Rides winning streaks and retreats during losing streaks.
    
    When winning, it repeats the winning move (it's working).
    When losing 3+ in a row, it switches to pure random to reset the opponent's model.
    During neutral phases, it falls back to a frequency counter.
    
    **Type**: Momentum / Hybrid
    **Logic**: Win-Stay, Lose-Random
    """
    name = "Hot Streak"

    def reset(self):
        self._streak = 0  # positive = wins, negative = losses

    def choose(self, round_num, my_history, opp_history):
        if not my_history:
            return self.rng.choice(MOVES)

        # Track streak
        if BEATS[my_history[-1]] == opp_history[-1]:
            self._streak = max(1, self._streak + 1)
        elif my_history[-1] == opp_history[-1]:
            self._streak = 0
        else:
            self._streak = min(-1, self._streak - 1)

        # Hot streak: repeat winning move
        if self._streak >= 2:
            return my_history[-1]

        # Cold streak: go random to reset
        if self._streak <= -3:
            return self.rng.choice(MOVES)

        # Neutral: frequency counter
        counts = Counter(opp_history[-20:])
        return _counter_move(counts.most_common(1)[0][0])


# ---------------------------------------------------------------------------
# 45: Contrarian (plays the least expected move)
# ---------------------------------------------------------------------------

class MarkovGenerator(Algorithm):
    """Generates moves based on a random internal Markov chain.
    
    Creates a random 3x3 transition matrix on reset and follows it to generate moves.
    It does not adapt to the opponent; instead, it acts as a 'structured noise' generator
    that can confuse predictors looking for human-like patterns.
    
    **Type**: Generator / Stochastic
    **Resets**: Every 50 rounds
    """
    name = "Markov Generator"
    
    def reset(self):
        # 3x3 transition matrix
        self._matrix = [
            [self.rng.random() for _ in range(3)],
            [self.rng.random() for _ in range(3)],
            [self.rng.random() for _ in range(3)]
        ]
        # Normalize
        for row in self._matrix:
            total = sum(row)
            for i in range(3): row[i] /= total
            
        self._last_idx = self.rng.randint(0, 2)
        self._rounds_until_reset = 50

    def choose(self, round_num, my_history, opp_history):
        if self._rounds_until_reset <= 0:
            self.reset()
            
        self._rounds_until_reset -= 1
        
        # Select next move based on current state and matrix probabilities
        probs = self._matrix[self._last_idx]
        next_idx = self.rng.choices([0,1,2], weights=probs, k=1)[0]
        self._last_idx = next_idx
        
        return MOVES[next_idx]


# ---------------------------------------------------------------------------
# 46: Monte Carlo Predictor
# ---------------------------------------------------------------------------

class MonteCarloPredictor(Algorithm):
    """Simulates random playouts to find the best counter.
    
    Estimates the probability of each opponent move by running Monte Carlo
    simulations over the observed transition patterns. Plays the move with the
    highest expected win rate.
    
    **Type**: Probabilistic / Monte Carlo
    **Simulations**: 50 per round
    """
    name = "Monte Carlo Predictor"
    _N_SIMS = 50

    def choose(self, round_num, my_history, opp_history):
        if len(opp_history) < 3:
            return self.rng.choice(MOVES)

        # Build transition probabilities
        transitions: dict[Move, list[Move]] = {m: [] for m in MOVES}
        for i in range(len(opp_history) - 1):
            transitions[opp_history[i]].append(opp_history[i + 1])

        last_opp = opp_history[-1]
        if not transitions[last_opp]:
            return _counter_move(last_opp)

        # Simulate N possible next moves
        sim_counts = Counter()
        for _ in range(self._N_SIMS):
            simulated = self.rng.choice(transitions[last_opp])
            sim_counts[simulated] += 1

        # Counter the most likely simulated move
        predicted = sim_counts.most_common(1)[0][0]
        return _counter_move(predicted)


# ---------------------------------------------------------------------------
# 47: Grudge Holder
# ---------------------------------------------------------------------------

class GrudgeHolder(Algorithm):
    """Remembers moves that caused losses and avoids them.
    
    Tracks which of our moves got beaten and by what. Refuses to repeat moves
    that have historically led to losses against specific opponent replies.
    Exploits the opponent's winning patterns.
    
    **Type**: Reactive / History
    **Memory**: Infinite
    """
    name = "Grudge Holder"

    def reset(self):
        self._grudges: dict[Move, int] = {m: 0 for m in MOVES}

    def choose(self, round_num, my_history, opp_history):
        if not my_history:
            return self.rng.choice(MOVES)

        # Update grudges from last round
        my_last = my_history[-1]
        opp_last = opp_history[-1]
        if BEATS[opp_last] == my_last:  # we lost
            self._grudges[my_last] += 1
        elif BEATS[my_last] == opp_last:  # we won
            self._grudges[my_last] = max(0, self._grudges[my_last] - 1)

        # Find move with least grudge score
        best_moves = []
        min_grudge = min(self._grudges.values())
        for m in MOVES:
            if self._grudges[m] == min_grudge:
                best_moves.append(m)

        # Among safe moves, also consider frequency counter
        if opp_history:
            counts = Counter(opp_history[-20:])
            counter = _counter_move(counts.most_common(1)[0][0])
            if counter in best_moves:
                return counter

        return self.rng.choice(best_moves)


# ---------------------------------------------------------------------------
# 48: Chameleon (mirrors opponent's distribution)
# ---------------------------------------------------------------------------

class Chameleon(Algorithm):
    """Mirrors the opponent's move distribution.
    
    If the opponent plays 50% Rock, Chameleon plays 50% Rock.
    This creates "frequency proximity," forcing many draws against biased opponents
    while maintaining enough randomness to prevent easy countering.
    
    **Type**: Mimicry / Stochastic
    **Target**: Opponent Distribution
    """
    name = "Chameleon"

    def choose(self, round_num, my_history, opp_history):
        if not opp_history:
            return self.rng.choice(MOVES)

        # Calculate opponent's distribution
        window = opp_history[-50:]
        counts = Counter(window)
        total = len(window)

        # Build probability distribution matching opponent's
        probs = []
        for m in MOVES:
            probs.append(counts.get(m, 0) / total)

        # Sample from opponent's distribution
        r = self.rng.random()
        cumulative = 0.0
        for i, m in enumerate(MOVES):
            cumulative += probs[i]
            if r <= cumulative:
                return m
        return MOVES[-1]


# ---------------------------------------------------------------------------
# 49: Fibonacci Player (mathematical sequence cycling)
# ---------------------------------------------------------------------------

class FibonacciPlayer(Algorithm):
    """Uses the Fibonacci sequence to determine moves.
    
    Follows the Fibonacci sequence mod 3 (period 8 pattern: 0,1,1,2,0,2,2,1).
    70% of the time it plays this sequence; 30% of the time it exploits the opponent.
    Hard to detect due to the complex base pattern.
    
    **Type**: Pattern / Mathematical
    **Sequence**: Fibonacci mod 3
    """
    name = "Fibonacci Player"
    # Fibonacci mod 3 has period 8: [0,1,1,2,0,2,2,1]
    _FIB_SEQ = [0, 1, 1, 2, 0, 2, 2, 1]

    def choose(self, round_num, my_history, opp_history):
        # 30% of the time, exploit opponent
        if opp_history and self.rng.random() < 0.3:
            counts = Counter(opp_history[-20:])
            return _counter_move(counts.most_common(1)[0][0])

        # 70% follow Fibonacci pattern
        idx = round_num % len(self._FIB_SEQ)
        return MOVES[self._FIB_SEQ[idx]]


# ===========================================================================
#  MATH-HEAVY / PROVEN CONCEPT ALGORITHMS (50-52)
# ===========================================================================


# ---------------------------------------------------------------------------
# 50: Lempel-Ziv Predictor (compression = prediction)
# ---------------------------------------------------------------------------

class LempelZivPredictor(Algorithm):
    """LZ78 compression-based sequence predictor.
    
    Builds a dictionary of observed subsequences incrementally.
    Uses the current context to find the longest matching phrase in the dictionary
    and predicts the likely continuation.
    
    **Type**: Pattern / Compression
    **Algorithm**: LZ78
    """
    name = "Lempel-Ziv Predictor"

    def reset(self):
        # phrase -> Counter of what move followed this phrase
        self._phrases: dict[tuple, Counter] = {(): Counter()}
        self._current: tuple = ()

    def choose(self, round_num, my_history, opp_history):
        if not opp_history:
            return self.rng.choice(MOVES)

        last = opp_history[-1]

        # Record this move as a continuation of the previous phrase
        if self._current in self._phrases:
            self._phrases[self._current][last] += 1

        # Try to extend current phrase
        extended = self._current + (last,)
        if extended in self._phrases:
            self._current = extended  # known phrase: keep building
        else:
            self._phrases[extended] = Counter()  # new phrase: register it
            self._current = ()  # reset to root

        # Predict: what usually follows the current phrase context?
        if self._current in self._phrases and self._phrases[self._current]:
            predicted = self._phrases[self._current].most_common(1)[0][0]
            return _counter_move(predicted)

        # Fallback: try all observed phrases, longest match first
        for depth in range(min(6, len(opp_history)), 0, -1):
            ctx = tuple(opp_history[-depth:])
            if ctx in self._phrases and self._phrases[ctx]:
                predicted = self._phrases[ctx].most_common(1)[0][0]
                return _counter_move(predicted)

        # Last resort: frequency
        counts = Counter(opp_history[-30:])
        return _counter_move(counts.most_common(1)[0][0])


# ---------------------------------------------------------------------------
# 51: Context Tree (upgraded N-Gram — Bayesian universal predictor)
# ---------------------------------------------------------------------------

class ContextTree(Algorithm):
    """Context Tree Weighting (CTW) universal predictor.
    
    A Bayesian mixture over all possible context depths (0 to 6).
    Computes a weighted average of predictions from different historical depths,
    giving more weight to depths with better predictive track records.
    
    **Type**: Bayesian / Universal
    **Algorithm**: CTW
    """
    name = "Context Tree"
    _MAX_DEPTH = 6

    def reset(self):
        # Two-level context tracking:
        # 1. Opponent-only contexts (for basic patterns)
        self._opp_ctx: dict[tuple, Counter] = {}
        # 2. Joint (my, opp) contexts (for interactive patterns)
        self._joint_ctx: dict[tuple, Counter] = {}

    def _get_ctx(self, store, ctx_key):
        if ctx_key not in store:
            store[ctx_key] = Counter()
        return store[ctx_key]

    def choose(self, round_num, my_history, opp_history):
        if not opp_history:
            return self.rng.choice(MOVES)

        last = opp_history[-1]

        # Update contexts with what we just observed
        for d in range(min(self._MAX_DEPTH, len(opp_history))):
            # Opponent-only context
            opp_ctx = tuple(opp_history[-(d + 1):-1]) if d > 0 else ()
            self._get_ctx(self._opp_ctx, opp_ctx)[last] += 1

            # Joint context (if available)
            if d <= len(my_history) and d > 0:
                joint_pairs = tuple(
                    (my_history[-(d + 1) + i], opp_history[-(d + 1) + i])
                    for i in range(d)
                )
                self._get_ctx(self._joint_ctx, joint_pairs)[last] += 1

        # Predict: weighted mixture over all depths
        total_probs = {m: 0.0 for m in MOVES}
        total_weight = 0.0

        for d in range(min(self._MAX_DEPTH, len(opp_history)) + 1):
            # CTW prior: 0.5^d (deeper = smaller prior)
            prior = 0.5 ** d

            # Opponent-only prediction at this depth
            opp_ctx = tuple(opp_history[-d:]) if d > 0 else ()
            counts = self._get_ctx(self._opp_ctx, opp_ctx)
            n = sum(counts.values())
            if n > 0:
                weight = prior * n
                for m in MOVES:
                    # KT estimator: (count + 0.5) / (total + 1.5)
                    total_probs[m] += weight * (counts.get(m, 0) + 0.5) / (n + 1.5)
                total_weight += weight

            # Joint prediction at this depth (bonus weight)
            if d > 0 and d <= len(my_history):
                joint_pairs = tuple(
                    (my_history[-d + i], opp_history[-d + i])
                    for i in range(d)
                )
                jcounts = self._get_ctx(self._joint_ctx, joint_pairs)
                jn = sum(jcounts.values())
                if jn > 0:
                    jweight = prior * jn * 1.5  # bonus for joint context
                    for m in MOVES:
                        total_probs[m] += jweight * (jcounts.get(m, 0) + 0.5) / (jn + 1.5)
                    total_weight += jweight

        if total_weight > 0:
            predicted = max(total_probs, key=total_probs.get)
            return _counter_move(predicted)

        return self.rng.choice(MOVES)


# ---------------------------------------------------------------------------
# 52: Maximum Entropy Predictor (Jaynes' principle)
# ---------------------------------------------------------------------------

class MaxEntropyPredictor(Algorithm):
    """Prediction via Jaynes' Maximum Entropy principle.
    
    Finds the probability distribution with maximum entropy subject to
    constraints from marginal frequencies, 1st-order, and 2nd-order transitions.
    Provides the "least biased" estimate consistent with observed data.
    
    **Type**: Statistical / MaxEnt
    **Inference**: Log-Linear Model
    """
    name = "Max Entropy Predictor"

    def reset(self):
        # Adaptive feature weights (start equal)
        self._weights = [0.3, 0.4, 0.3]
        self._last_prediction = None

    def choose(self, round_num, my_history, opp_history):
        import math

        if len(opp_history) < 3:
            return self.rng.choice(MOVES)

        # Adapt weights based on last prediction accuracy
        if self._last_prediction is not None and len(opp_history) >= 2:
            actual = opp_history[-1]
            if self._last_prediction == actual:
                pass  # correct — keep weights
            else:
                # Reduce weight of features that were wrong
                # (computed retroactively below)
                pass

        # Feature 1: Marginal frequency (window=40)
        window = opp_history[-40:]
        freq_counts = Counter(window)
        total = len(window)
        f1 = {m: (freq_counts.get(m, 0) + 1) / (total + 3) for m in MOVES}

        # Feature 2: 1st-order transition P(next | opp[-1])
        last = opp_history[-1]
        trans = Counter()
        for i in range(len(opp_history) - 1):
            if opp_history[i] == last:
                trans[opp_history[i + 1]] += 1
        t_total = sum(trans.values())
        if t_total > 0:
            f2 = {m: (trans.get(m, 0) + 1) / (t_total + 3) for m in MOVES}
        else:
            f2 = {m: 1 / 3 for m in MOVES}

        # Feature 3: 2nd-order transition P(next | opp[-2], opp[-1])
        if len(opp_history) >= 3:
            pair = (opp_history[-2], opp_history[-1])
            pair_trans = Counter()
            for i in range(len(opp_history) - 2):
                if (opp_history[i], opp_history[i + 1]) == pair:
                    if i + 2 < len(opp_history):
                        pair_trans[opp_history[i + 2]] += 1
            p_total = sum(pair_trans.values())
            if p_total > 0:
                f3 = {m: (pair_trans.get(m, 0) + 1) / (p_total + 3) for m in MOVES}
            else:
                f3 = {m: 1 / 3 for m in MOVES}
        else:
            f3 = {m: 1 / 3 for m in MOVES}

        # MaxEnt log-linear combination (geometric weighted mean)
        w1, w2, w3 = self._weights
        combined = {}
        for m in MOVES:
            combined[m] = (f1[m] ** w1) * (f2[m] ** w2) * (f3[m] ** w3)

        # Normalize
        total_c = sum(combined.values())
        for m in MOVES:
            combined[m] /= total_c

        predicted = max(combined, key=combined.get)
        self._last_prediction = predicted
        return _counter_move(predicted)


# ===========================================================================
#  TROJAN / DECEPTION / WEIRD ALGORITHMS (53-57)
# ===========================================================================


# ---------------------------------------------------------------------------
# 53: Poison Pill (bait-and-switch deception)
# ---------------------------------------------------------------------------

class PoisonPill(Algorithm):
    """A highly deceptive bait-and-switch strategy.
    
    Cycles through three phases: 1. Planting a bias (Bait), 2. Exploitating
    the opponent's adaptation (Snare), and 3. Shifting the bait.
    Keeps the opponent perpetually one step behind by changing logic every 30 rounds.
    
    **Type**: Deceptive / Strategic
    **Cycle**: 90 Rounds
    """
    name = "Poison Pill"

    def choose(self, round_num, my_history, opp_history):
        cycle = round_num % 90
        # Rotate the poison order each super-cycle
        rotation = (round_num // 90) % 3
        poisons = [
            (Move.ROCK, Move.SCISSORS),   # bait Rock → exploit with Scissors
            (Move.PAPER, Move.ROCK),      # bait Paper → exploit with Rock
            (Move.SCISSORS, Move.PAPER),  # bait Scissors → exploit with Paper
        ]
        # Shift base by rotation so each 90-round cycle is different
        base = rotation

        if cycle < 30:
            bait = poisons[(base + 0) % 3][0]
            return bait if self.rng.random() < 0.85 else self.rng.choice(MOVES)
        elif cycle < 60:
            exploit = poisons[(base + 0) % 3][1]
            return exploit if self.rng.random() < 0.85 else self.rng.choice(MOVES)
        else:
            bait = poisons[(base + 1) % 3][0]
            return bait if self.rng.random() < 0.85 else self.rng.choice(MOVES)


# ---------------------------------------------------------------------------
# 54: Mirror Breaker (feedback loop exploiter)
# ---------------------------------------------------------------------------

class MirrorBreaker(Algorithm):
    """Exploits mirror and reactive strategies.
    
    Uses a diagnostic phase to detect if the opponent is mirroring or countering.
    Once detected, it sets a feedback trap that turns the opponent's
    reactive logic against them.
    
    **Type**: Anti-Meta / Specialist
    **Detection**: Feedback Loop
    """
    name = "Mirror Breaker"

    def reset(self):
        self._detected = None  # 'mirror', 'counter', None

    def choose(self, round_num, my_history, opp_history):
        # Diagnostic phase: first 20 rounds
        if round_num < 20:
            return MOVES[round_num % 3]

        # Detection: compare first 15 reactions
        if self._detected is None and len(opp_history) >= 15 and len(my_history) >= 15:
            mirror_hits = sum(
                1 for i in range(1, 15)
                if opp_history[i] == my_history[i - 1]
            )
            counter_hits = sum(
                1 for i in range(1, 15)
                if opp_history[i] == _counter_move(my_history[i - 1])
            )
            if mirror_hits >= 9:
                self._detected = 'mirror'
            elif counter_hits >= 9:
                self._detected = 'counter'

        if self._detected == 'mirror' and my_history:
            # They'll copy our last move → play counter of our last move
            return _counter_move(my_history[-1])

        if self._detected == 'counter' and my_history:
            # They'll counter our last move → play same as our last move
            # (they play counter(our_last), we play our_last = loses_to their move)
            # Actually: they play counter(my[-1]), we want to beat that
            # counter(counter(my[-1])) = loses_to(my[-1])... no.
            # They play X = counter(my[-1]). We want counter(X) = counter(counter(my[-1]))
            expected_opp = _counter_move(my_history[-1])
            return _counter_move(expected_opp)

        # Default: Markov prediction
        if len(opp_history) >= 2:
            last = opp_history[-1]
            transitions = Counter()
            for j in range(len(opp_history) - 1):
                if opp_history[j] == last:
                    transitions[opp_history[j + 1]] += 1
            if transitions:
                return _counter_move(transitions.most_common(1)[0][0])

        return self.rng.choice(MOVES)


# ---------------------------------------------------------------------------
# 55: The Usurper (strategy identification + surpass)
# ---------------------------------------------------------------------------

class TheUsurper(Algorithm):
    """Identifies and counters opponent archetypes by becoming "better."
    
    Detects if the opponent follows a known template (Constant, Cycle, Mirror, etc.).
    Once identified, it switches to a strategy that is mathematically superior to
    the entire archetype, not just the next move.
    
    **Type**: Anti-Meta / Adaptive
    **Confidence**: > 60%
    """
    name = "The Usurper"

    def reset(self):
        # Scores for each archetype
        self._scores = {'constant': 0.0, 'cycle': 0.0, 'mirror': 0.0,
                        'counter': 0.0, 'frequency': 0.0, 'random': 0.0}

    def choose(self, round_num, my_history, opp_history):
        if len(opp_history) < 8:
            return self.rng.choice(MOVES)

        # Score archetypes with exponential decay
        decay = 0.92
        for k in self._scores:
            self._scores[k] *= decay

        if len(opp_history) >= 2 and len(my_history) >= 2:
            actual = opp_history[-1]
            prev_opp = opp_history[-2]
            prev_my = my_history[-2]

            # Constant: opp repeats
            if actual == prev_opp:
                self._scores['constant'] += 1.0

            # Cycle: opp follows R→P→S
            expected_cycle = MOVES[(MOVES.index(prev_opp) + 1) % 3]
            if actual == expected_cycle:
                self._scores['cycle'] += 1.0

            # Mirror: opp copies my previous
            if actual == prev_my:
                self._scores['mirror'] += 1.0

            # Counter: opp counters my previous
            if actual == _counter_move(prev_my):
                self._scores['counter'] += 1.0

            # Frequency: opp counters my most common
            past_counts = Counter(my_history[:-1])
            if past_counts:
                freq_expected = _counter_move(past_counts.most_common(1)[0][0])
                if actual == freq_expected:
                    self._scores['frequency'] += 1.0

        best = max(self._scores, key=self._scores.get)
        confidence = self._scores[best] / max(1, sum(self._scores.values()))

        if confidence < 0.3:
            best = 'random'

        # SURPASS each archetype
        if best == 'constant':
            # Opponent repeats → counter their last
            return _counter_move(opp_history[-1])

        elif best == 'cycle':
            # Opponent cycles R→P→S → play 2 ahead
            next_opp = MOVES[(MOVES.index(opp_history[-1]) + 1) % 3]
            return _counter_move(next_opp)

        elif best == 'mirror':
            # They copy my last → play counter(my_last)
            # They'll play my[-1], we play counter(my[-1])
            return _counter_move(my_history[-1])

        elif best == 'counter':
            # They counter my last → play counter(counter(my[-1]))
            return _counter_move(_counter_move(my_history[-1]))

        elif best == 'frequency':
            # They counter my most common → deliberately play least common
            my_counts = Counter(my_history[-30:])
            for m in MOVES:
                if m not in my_counts:
                    my_counts[m] = 0
            return min(MOVES, key=lambda m: my_counts[m])

        else:  # random
            # Best general-purpose: combined Markov + frequency
            if len(opp_history) >= 2:
                last = opp_history[-1]
                transitions = Counter()
                for j in range(len(opp_history) - 1):
                    if opp_history[j] == last:
                        transitions[opp_history[j + 1]] += 1
                if transitions:
                    return _counter_move(transitions.most_common(1)[0][0])
            counts = Counter(opp_history[-20:])
            return _counter_move(counts.most_common(1)[0][0])


# ---------------------------------------------------------------------------
# 56: Double Bluff (multi-level deception)
# ---------------------------------------------------------------------------

class DoubleBluff(Algorithm):
    """Adaptive multi-level reasoning strategy.
    
    Tracks three levels of recursive reasoning: 0 (Direct Counter), 1 (Counter-Counter),
    and 2 (Counter-Counter-Counter). Dynamically selects the level that has been
    most successful against the current opponent.
    
    **Type**: Meta / Recursive
    **Levels**: 3 (Modular)
    """
    name = "Double Bluff"

    def reset(self):
        self._level_scores = [1.0, 1.0, 1.0]
        self._decay = 0.9

    def choose(self, round_num, my_history, opp_history):
        if len(opp_history) < 5:
            return self.rng.choice(MOVES)

        # Predict opponent's most likely move
        counts = Counter(opp_history[-30:])
        opp_predicted = counts.most_common(1)[0][0]

        # 3 reasoning levels
        level_moves = [
            _counter_move(opp_predicted),                              # L0
            _counter_move(_counter_move(opp_predicted)),               # L1
            _counter_move(_counter_move(_counter_move(opp_predicted))), # L2
        ]

        # Retroactive scoring: which level would have won last round?
        if len(opp_history) >= 2:
            for i in range(3):
                self._level_scores[i] *= self._decay

            opp_last = opp_history[-1]
            past_counts = Counter(opp_history[:-1][-30:])
            if past_counts:
                past_pred = past_counts.most_common(1)[0][0]
                retro_moves = [
                    _counter_move(past_pred),
                    _counter_move(_counter_move(past_pred)),
                    _counter_move(_counter_move(_counter_move(past_pred))),
                ]
                for i in range(3):
                    if BEATS[retro_moves[i]] == opp_last:
                        self._level_scores[i] += 1.5
                    elif retro_moves[i] == opp_last:
                        self._level_scores[i] += 0.3

        # Follow the best-performing level
        best = self._level_scores.index(max(self._level_scores))
        return level_moves[best]


# ---------------------------------------------------------------------------
# 57: Frequency Disruptor (creates false patterns + disrupts)
# ---------------------------------------------------------------------------

class FrequencyDisruptor(Algorithm):
    """Misleads frequency detectors with false patterns.
    
    Establishes a deceptive pattern to bait the opponent into a specific counter.
    Once the bait is taken, it abruptly switches to exploit the opponent's adaptation.
    A sophisticated version of the Poison Pill.
    
    **Type**: Deceptive / Strategic
    **Phase**: Multi-Phase Cycle
    """
    name = "Frequency Disruptor"

    def reset(self):
        self._fake_patterns = [
            [Move.ROCK, Move.ROCK, Move.PAPER, Move.ROCK, Move.ROCK],     # Heavy R
            [Move.PAPER, Move.SCISSORS, Move.PAPER, Move.PAPER, Move.SCISSORS], # Heavy P
            [Move.SCISSORS, Move.SCISSORS, Move.ROCK, Move.SCISSORS, Move.SCISSORS],  # Heavy S
        ]
        self._pattern_idx = 0

    def choose(self, round_num, my_history, opp_history):
        cycle_len = 30
        phase = round_num % cycle_len

        # Rotate which fake pattern we use each cycle
        if round_num % cycle_len == 0 and round_num > 0:
            self._pattern_idx = (self._pattern_idx + 1) % 3

        pattern = self._fake_patterns[self._pattern_idx]

        if phase < 20:
            # Phase 1: Broadcast fake pattern (70% pattern, 30% random)
            if self.rng.random() < 0.7:
                return pattern[phase % len(pattern)]
            return self.rng.choice(MOVES)

        elif phase < 25:
            # Phase 2: Exploit — opponents who adapted to heavy R play Paper
            # So we play Scissors to beat their Paper adaptation
            # General: counter(counter(dominant_fake_move))
            dominant = Counter(pattern).most_common(1)[0][0]
            exploit = _counter_move(_counter_move(dominant))
            return exploit if self.rng.random() < 0.85 else self.rng.choice(MOVES)

        else:
            # Phase 3: Real frequency analysis on their actual play
            if opp_history:
                counts = Counter(opp_history[-15:])
                return _counter_move(counts.most_common(1)[0][0])
            return self.rng.choice(MOVES)


# ===========================================================================
#  UPGRADED VARIANTS (58-59) — keep originals, new names
# ===========================================================================


# ---------------------------------------------------------------------------
# 58: Deep Historian (upgraded Historian)
# ---------------------------------------------------------------------------

class DeepHistorian(Algorithm):
    """Upgraded pattern recognition with joint context.
    
    Searches history for matching sequences of variable length (2-5).
    Weights matches by recency and considers joint (player + opponent) history
    to capture interactive patterns.
    
    **Type**: Pattern / History
    **Weighting**: Exponential Decay
    """
    name = "Deep Historian"

    def choose(self, round_num, my_history, opp_history):
        if len(opp_history) < 6:
            return self.rng.choice(MOVES)

        # Try variable-length joint patterns (longest match first)
        for pattern_len in range(5, 1, -1):
            if pattern_len >= len(opp_history):
                continue

            # Build current context: joint (my, opp) pairs
            current = tuple(
                (my_history[-pattern_len + i], opp_history[-pattern_len + i])
                for i in range(pattern_len)
            )

            # Search history for matches with recency decay
            weighted_counts: dict[Move, float] = {m: 0.0 for m in MOVES}
            total_weight = 0.0
            n = len(opp_history)

            for i in range(n - pattern_len):
                candidate = tuple(
                    (my_history[i + j], opp_history[i + j])
                    for j in range(pattern_len)
                )
                if candidate == current and (i + pattern_len) < n:
                    # Exponential recency weight: recent matches count more
                    age = n - (i + pattern_len)
                    weight = 0.95 ** age  # recent = higher weight
                    next_move = opp_history[i + pattern_len]
                    weighted_counts[next_move] += weight
                    total_weight += weight

            if total_weight > 0.5:  # enough confidence
                predicted = max(weighted_counts, key=weighted_counts.get)
                return _counter_move(predicted)

        # Fallback: simple frequency
        counts = Counter(opp_history[-25:])
        return _counter_move(counts.most_common(1)[0][0])


# ---------------------------------------------------------------------------
# 59: Adaptive N-Gram (upgraded N-Gram Predictor)
# ---------------------------------------------------------------------------

class AdaptiveNGram(Algorithm):
    """Upgraded N-Gram predictor with dynamic context and decay.
    
    Uses a meta-learner to select the best-performing context length (n=1 to 5).
    Weights transitions by recency and incorporates joint (player + opponent)
    history for deeper pattern analysis.
    
    **Type**: Pattern / N-Gram
    **Selection**: Meta-Learner
    """
    name = "Adaptive N-Gram"

    def reset(self):
        # Track accuracy for each n-gram length
        self._accuracy = {n: 1.0 for n in range(1, 6)}  # n=1..5
        self._decay = 0.95
        self._last_prediction = None
        self._last_n_used = None

    def choose(self, round_num, my_history, opp_history):
        if len(opp_history) < 3:
            return self.rng.choice(MOVES)

        # Update accuracy tracking
        if self._last_prediction is not None and self._last_n_used is not None:
            actual = opp_history[-1]
            for n in self._accuracy:
                self._accuracy[n] *= self._decay
            if self._last_prediction == actual:
                self._accuracy[self._last_n_used] += 1.0

        # For each n-gram length, compute prediction
        predictions: dict[int, tuple[Move, float]] = {}

        for n in range(5, 0, -1):
            if n >= len(opp_history):
                continue

            # Opponent-only n-gram
            context = tuple(opp_history[-n:])
            weighted_counts: dict[Move, float] = {m: 0.0 for m in MOVES}

            for i in range(len(opp_history) - n):
                candidate = tuple(opp_history[i:i + n])
                if candidate == context and (i + n) < len(opp_history):
                    age = len(opp_history) - (i + n)
                    weight = 0.9 ** age
                    weighted_counts[opp_history[i + n]] += weight

            total = sum(weighted_counts.values())
            if total > 0:
                best_move = max(weighted_counts, key=weighted_counts.get)
                confidence = weighted_counts[best_move] / total
                predictions[n] = (best_move, confidence)

            # Also try joint n-gram (bonus if it matches)
            if n <= len(my_history) and n >= 2:
                joint_ctx = tuple(
                    (my_history[-n + i], opp_history[-n + i])
                    for i in range(n)
                )
                joint_counts: dict[Move, float] = {m: 0.0 for m in MOVES}
                for i in range(len(opp_history) - n):
                    j_candidate = tuple(
                        (my_history[i + j], opp_history[i + j])
                        for j in range(n)
                    )
                    if j_candidate == joint_ctx and (i + n) < len(opp_history):
                        age = len(opp_history) - (i + n)
                        weight = 0.9 ** age
                        joint_counts[opp_history[i + n]] += weight

                j_total = sum(joint_counts.values())
                if j_total > 0:
                    j_best = max(joint_counts, key=joint_counts.get)
                    j_conf = joint_counts[j_best] / j_total
                    # Use joint prediction if higher confidence
                    if n in predictions:
                        _, opp_conf = predictions[n]
                        if j_conf > opp_conf:
                            predictions[n] = (j_best, j_conf * 1.2)
                    else:
                        predictions[n] = (j_best, j_conf * 1.2)

        # Select best n based on accuracy tracking × confidence
        best_n = None
        best_score = -1.0

        for n, (pred, conf) in predictions.items():
            score = self._accuracy.get(n, 0.5) * conf
            if score > best_score:
                best_score = score
                best_n = n

        if best_n is not None and best_n in predictions:
            predicted, _ = predictions[best_n]
            self._last_prediction = predicted
            self._last_n_used = best_n
            return _counter_move(predicted)

        self._last_prediction = None
        self._last_n_used = None
        counts = Counter(opp_history[-20:])
        return _counter_move(counts.most_common(1)[0][0])


# ===========================================================================
#  MATH-HEAVY / DEEP CONCEPTUAL ALGORITHMS (60-62)
# ===========================================================================


# ---------------------------------------------------------------------------
# 60: Regret Minimizer (Regret Matching — poker AI foundation)
# ---------------------------------------------------------------------------

class RegretMinimizer(Algorithm):
    """Regret Matching strategy (based on Game Theory).
    
    Tracks cumulative "regret" for each move—how much better that move would
    have performed vs the actual choice. Plays moves proportionally to their
    positive regret, converging toward a Nash Equilibrium.
    
    **Type**: Game Theory / Nash
    **Algorithm**: Regret Matching
    """
    name = "Regret Minimizer"

    def reset(self):
        self._regrets = {m: 0.0 for m in MOVES}
        self._strategy_sum = {m: 0.0 for m in MOVES}

    def _payoff(self, my_move, opp_move):
        if BEATS[my_move] == opp_move:
            return 1.0
        elif my_move == opp_move:
            return 0.0
        else:
            return -1.0

    def _get_strategy(self):
        """Compute current strategy from positive regrets."""
        positive = {m: max(0.0, self._regrets[m]) for m in MOVES}
        total = sum(positive.values())
        if total > 0:
            return {m: positive[m] / total for m in MOVES}
        else:
            return {m: 1.0 / 3.0 for m in MOVES}

    def choose(self, round_num, my_history, opp_history):
        # Update regrets from last round
        if my_history and opp_history:
            my_last = my_history[-1]
            opp_last = opp_history[-1]
            actual_payoff = self._payoff(my_last, opp_last)

            for m in MOVES:
                # How much better would move m have been?
                alt_payoff = self._payoff(m, opp_last)
                self._regrets[m] += alt_payoff - actual_payoff

        # Get strategy from regrets
        strategy = self._get_strategy()

        # Accumulate average strategy (for convergence)
        for m in MOVES:
            self._strategy_sum[m] += strategy[m]

        # Sample from strategy
        r = self.rng.random()
        cumulative = 0.0
        for m in MOVES:
            cumulative += strategy[m]
            if r <= cumulative:
                return m
        return MOVES[-1]


# ---------------------------------------------------------------------------
# 61: Fourier Predictor (DFT-based periodicity detection)
# ---------------------------------------------------------------------------

class FourierPredictor(Algorithm):
    """Periodicity detection via Discrete Fourier Transform (DFT).
    
    Encodes move history as a signal and applies DFT to find dominant frequency
    components. Extrapolates these frequencies to predict the next move.
    Highly effective against any periodic or cyclic patterns.
    
    **Type**: Signal Processing / Pattern
    **Algorithm**: DFT
    """
    name = "Fourier Predictor"

    def choose(self, round_num, my_history, opp_history):
        import math

        if len(opp_history) < 10:
            return self.rng.choice(MOVES)

        # Encode moves as complex-valued signal
        window = opp_history[-64:]  # use last 64 for efficiency
        N = len(window)
        signal = [MOVES.index(m) for m in window]  # 0, 1, 2

        # Compute DFT manually (no numpy)
        magnitudes = []
        phases = []
        for k in range(1, N // 2):  # skip DC component (k=0)
            re = 0.0
            im = 0.0
            for n in range(N):
                angle = -2.0 * math.pi * k * n / N
                re += signal[n] * math.cos(angle)
                im += signal[n] * math.sin(angle)
            mag = math.sqrt(re * re + im * im)
            phase = math.atan2(im, re)
            magnitudes.append((mag, k, phase))

        if not magnitudes:
            return self.rng.choice(MOVES)

        # Find top 3 dominant frequencies
        magnitudes.sort(reverse=True)
        top_freqs = magnitudes[:3]

        # Extrapolate next value using dominant frequencies
        prediction = 0.0
        total_mag = sum(m for m, _, _ in top_freqs)

        if total_mag > 0:
            for mag, k, phase in top_freqs:
                weight = mag / total_mag
                # Predict x[N] by evaluating the frequency component at t=N
                angle = 2.0 * math.pi * k * N / N + phase
                prediction += weight * math.cos(angle)

        # Map prediction back to a move (0, 1, 2)
        predicted_idx = round(prediction) % 3
        if predicted_idx < 0:
            predicted_idx += 3
        predicted = MOVES[predicted_idx]
        return _counter_move(predicted)


# ---------------------------------------------------------------------------
# 62: Eigenvalue Predictor (transition matrix eigenvector)
# ---------------------------------------------------------------------------

class EigenvaluePredictor(Algorithm):
    """Stationary distribution predictor using Power Iteration.
    
    Analyzes the opponent's transition matrix and computes its dominant
    eigenvector (stationary distribution). Predicts based on long-term
    behavioral trends and the current transition state.
    
    **Type**: Linear Algebra / Markov
    **Math**: Power Iteration
    """
    name = "Eigenvalue Predictor"

    def reset(self):
        # Transition counts: [from][to]
        self._transitions = {m: {n: 0 for n in MOVES} for m in MOVES}

    def choose(self, round_num, my_history, opp_history):
        if len(opp_history) < 5:
            return self.rng.choice(MOVES)

        # Update transition matrix
        if len(opp_history) >= 2:
            prev = opp_history[-2]
            curr = opp_history[-1]
            self._transitions[prev][curr] += 1

        # Build row-stochastic transition matrix
        matrix = {}
        for i in MOVES:
            row_total = sum(self._transitions[i].values())
            if row_total > 0:
                matrix[i] = {j: self._transitions[i][j] / row_total
                             for j in MOVES}
            else:
                matrix[i] = {j: 1.0 / 3.0 for j in MOVES}

        # Power iteration to find stationary distribution (10 iterations)
        pi = {m: 1.0 / 3.0 for m in MOVES}
        for _ in range(10):
            new_pi = {m: 0.0 for m in MOVES}
            for j in MOVES:
                for i in MOVES:
                    new_pi[j] += pi[i] * matrix[i][j]
            # Normalize
            total = sum(new_pi.values())
            if total > 0:
                pi = {m: new_pi[m] / total for m in MOVES}

        # Prediction: combine current-row + stationary
        last_opp = opp_history[-1]
        current_row = matrix[last_opp]

        combined = {}
        for m in MOVES:
            # 60% current transition + 40% stationary
            combined[m] = 0.6 * current_row[m] + 0.4 * pi[m]

        predicted = max(combined, key=combined.get)
        return _counter_move(predicted)



# ===========================================================================
#  NEW-FIELD ALGORITHMS (67-71)
# ===========================================================================


# ---------------------------------------------------------------------------
# 67: Hidden Markov Oracle (Hidden Markov Models / NLP)
# ---------------------------------------------------------------------------

class HiddenMarkovOracle(Algorithm):
    """Mood-discovery predictor using Hidden Markov Models.
    
    Discovers the opponent's hidden 'moods' (internal states) and their move
    distributions. Uses the Baum-Welch algorithm for online parameter refinement.
    Predicts moves by marginalizing probabilities over all possible hidden states.
    
    **Type**: ML / HMM
    **States**: 3 Hidden states
    """
    name = "Hidden Markov Oracle"

    def reset(self):
        n_states = 3
        n_obs = 3  # R, P, S
        self._n_states = n_states
        # Transition matrix A[i][j] = P(s_j | s_i)
        self._A = [[1.0 / n_states] * n_states for _ in range(n_states)]
        # Emission matrix B[i][o] = P(obs_o | state_i)
        # Initialize with slight bias to make states distinguishable
        self._B = [
            [0.5, 0.25, 0.25],  # state 0: biased Rock
            [0.25, 0.5, 0.25],  # state 1: biased Paper
            [0.25, 0.25, 0.5],  # state 2: biased Scissors
        ]
        # Initial state distribution
        self._pi = [1.0 / n_states] * n_states
        # Forward variable: α[i] = P(o_1..o_t, s_t=i)
        self._alpha = list(self._pi)
        self._obs_window: list[int] = []  # recent observations as indices
        self._update_interval = 20

    def _forward_step(self, obs_idx):
        """Update forward variable with new observation."""
        n = self._n_states
        new_alpha = [0.0] * n
        for j in range(n):
            s = sum(self._alpha[i] * self._A[i][j] for i in range(n))
            new_alpha[j] = s * self._B[j][obs_idx]
        # Normalize to prevent underflow
        total = sum(new_alpha)
        if total > 0:
            new_alpha = [a / total for a in new_alpha]
        else:
            new_alpha = [1.0 / n] * n
        self._alpha = new_alpha

    def _update_parameters(self):
        """Simplified online Baum-Welch on recent observation window."""
        if len(self._obs_window) < 10:
            return
        obs = self._obs_window[-self._update_interval:]
        n = self._n_states
        T = len(obs)

        # Forward pass
        alphas = [[0.0] * n for _ in range(T)]
        alphas[0] = [self._pi[i] * self._B[i][obs[0]] for i in range(n)]
        total = sum(alphas[0])
        if total > 0:
            alphas[0] = [a / total for a in alphas[0]]

        for t in range(1, T):
            for j in range(n):
                s = sum(alphas[t-1][i] * self._A[i][j] for i in range(n))
                alphas[t][j] = s * self._B[j][obs[t]]
            total = sum(alphas[t])
            if total > 0:
                alphas[t] = [a / total for a in alphas[t]]

        # Backward pass
        betas = [[0.0] * n for _ in range(T)]
        betas[T-1] = [1.0] * n

        for t in range(T-2, -1, -1):
            for i in range(n):
                betas[t][i] = sum(
                    self._A[i][j] * self._B[j][obs[t+1]] * betas[t+1][j]
                    for j in range(n)
                )
            total = sum(betas[t])
            if total > 0:
                betas[t] = [b / total for b in betas[t]]

        # Update emission probs B
        for i in range(n):
            for o in range(3):
                num = sum(alphas[t][i] * betas[t][i]
                          for t in range(T) if obs[t] == o)
                den = sum(alphas[t][i] * betas[t][i] for t in range(T))
                if den > 1e-10:
                    self._B[i][o] = max(0.01, 0.7 * self._B[i][o] + 0.3 * (num / den))
            # Normalize
            row_sum = sum(self._B[i])
            self._B[i] = [b / row_sum for b in self._B[i]]

        # Update transition probs A
        for i in range(n):
            for j in range(n):
                num = sum(
                    alphas[t][i] * self._A[i][j] * self._B[j][obs[t+1]] * betas[t+1][j]
                    for t in range(T-1)
                )
                den = sum(alphas[t][i] * betas[t][i] for t in range(T-1))
                if den > 1e-10:
                    self._A[i][j] = max(0.01, 0.7 * self._A[i][j] + 0.3 * (num / den))
            row_sum = sum(self._A[i])
            self._A[i] = [a / row_sum for a in self._A[i]]

    def choose(self, round_num, my_history, opp_history):
        if not opp_history:
            return self.rng.choice(MOVES)

        obs_idx = MOVES.index(opp_history[-1])
        self._obs_window.append(obs_idx)
        self._forward_step(obs_idx)

        # Periodic parameter update
        if len(self._obs_window) % self._update_interval == 0:
            self._update_parameters()

        # Predict next observation by marginalizing over hidden states
        n = self._n_states
        pred = [0.0, 0.0, 0.0]
        for o in range(3):
            for j in range(n):
                s = sum(self._alpha[i] * self._A[i][j] for i in range(n))
                pred[o] += s * self._B[j][o]

        # Counter the most likely predicted move
        predicted_move = MOVES[pred.index(max(pred))]
        return _counter_move(predicted_move)


# ---------------------------------------------------------------------------
# 68: Genetic Strategist (Evolutionary Computation)
# ---------------------------------------------------------------------------

class GeneticStrategist(Algorithm):
    """Evolutionary strategist using a population of genomes.
    
    Each genome encodes a response table for move context. Evolves the population
    via selection, crossover, and mutation based on fitness against the opponent’s
    recent history.
    
    **Type**: Evolutionary Computing / GA
    **Population**: 20 Individual genomes
    """
    name = "Genetic Strategist"

    def reset(self):
        self._pop_size = 20
        self._population = []
        for _ in range(self._pop_size):
            genome = {}
            for m1 in MOVES:
                for m2 in MOVES:
                    genome[(m1, m2)] = self.rng.choice(MOVES)
            self._population.append(genome)
        self._fitness = [0.0] * self._pop_size
        self._best_genome = self._population[0]
        self._evolve_interval = 25

    def _evaluate_fitness(self, genome, opp_history):
        """Score a genome against recent opponent history."""
        if len(opp_history) < 3:
            return 0.0
        window = opp_history[-50:] if len(opp_history) > 50 else opp_history
        score = 0
        for k in range(2, len(window)):
            key = (window[k-1], window[k-2])
            our_move = genome.get(key, self.rng.choice(MOVES))
            opp_move = window[k]
            if BEATS[our_move] == opp_move:
                score += 1
            elif our_move != opp_move:
                score -= 1
        return score

    def _evolve(self, opp_history):
        """Run one generation: evaluate, select, crossover, mutate."""
        # Evaluate fitness
        for i, genome in enumerate(self._population):
            self._fitness[i] = self._evaluate_fitness(genome, opp_history)

        # Sort by fitness (descending)
        ranked = sorted(range(self._pop_size), key=lambda i: self._fitness[i], reverse=True)

        # Selection: keep top half
        survivors = [self._population[i] for i in ranked[:self._pop_size // 2]]

        # Crossover: pair survivors to produce children
        children = []
        for c in range(self._pop_size // 2):
            p1 = survivors[c % len(survivors)]
            p2 = survivors[(c + 1) % len(survivors)]
            child = {}
            for key in p1:
                child[key] = p1[key] if self.rng.random() < 0.5 else p2[key]
            children.append(child)

        # Mutation
        for child in children:
            for key in child:
                if self.rng.random() < 0.05:
                    child[key] = self.rng.choice(MOVES)

        self._population = survivors + children
        self._best_genome = self._population[0]  # fittest after sort

    def choose(self, round_num, my_history, opp_history):
        if len(opp_history) < 2:
            return self.rng.choice(MOVES)

        # Periodic evolution
        if round_num > 0 and round_num % self._evolve_interval == 0:
            self._evolve(opp_history)

        # Use the best genome's response
        key = (opp_history[-1], opp_history[-2])
        return self._best_genome.get(key, self.rng.choice(MOVES))


# ---------------------------------------------------------------------------
# 69: PID Controller (Control Theory / Robotics)
# ---------------------------------------------------------------------------

class PIDController(Algorithm):
    """Feedback control strategy using PID principles.
    
    Regulates move selection by treating the win/loss rate as a control error.
    Applies Proportional, Integral, and Derivative corrections to the strategy
    mixture via softmax.
    
    **Type**: Control Theory / PID
    **Params**: Kp=0.5, Ki=0.05, Kd=0.2
    """
    name = "PID Controller"

    def reset(self):
        self._kp = 0.5
        self._ki = 0.05
        self._kd = 0.2
        # Per-move tracking
        self._wins = {m: 0 for m in MOVES}
        self._plays = {m: 0 for m in MOVES}
        self._error_prev = {m: 0.0 for m in MOVES}
        self._integral = {m: 0.0 for m in MOVES}
        self._total_rounds = 0

    def choose(self, round_num, my_history, opp_history):
        import math

        # Update from last round
        if my_history and opp_history:
            my_last = my_history[-1]
            opp_last = opp_history[-1]
            self._plays[my_last] += 1
            if BEATS[my_last] == opp_last:
                self._wins[my_last] += 1
            self._total_rounds += 1

        if self._total_rounds < 3:
            return self.rng.choice(MOVES)

        # Compute PID control signal for each move
        control = {}
        for m in MOVES:
            # Expected win rate vs actual
            expected = 1.0 / 3.0  # baseline expectation
            actual = self._wins[m] / max(self._plays[m], 1)
            error = expected - actual  # positive = underperforming

            # The move that BEATS m: we want to play what beats the
            # opponent's likely next move, so we track error differently
            # Error = how good is playing counter to opponent's bias
            opp_freq = sum(1 for o in opp_history[-30:] if o == m) / max(len(opp_history[-30:]), 1)
            counter = _counter_move(m)
            error = opp_freq - 1.0/3.0  # positive if opponent plays m more than expected

            # PID terms
            p_term = self._kp * error
            self._integral[m] += error
            self._integral[m] = max(-10, min(10, self._integral[m]))  # anti-windup
            i_term = self._ki * self._integral[m]
            d_term = self._kd * (error - self._error_prev[m])
            self._error_prev[m] = error

            # Control signal for the COUNTER move
            control[counter] = control.get(counter, 0) + (p_term + i_term + d_term)

        # Softmax over control signals
        max_u = max(control.values())
        exp_u = {m: math.exp(min(control.get(m, 0) - max_u, 10)) for m in MOVES}
        total = sum(exp_u.values())
        probs = {m: exp_u[m] / total for m in MOVES}

        # Sample
        r = self.rng.random()
        cumulative = 0.0
        for m in MOVES:
            cumulative += probs[m]
            if r <= cumulative:
                return m
        return MOVES[-1]


# ---------------------------------------------------------------------------
# 70: Chaos Engine (Nonlinear Dynamics / Physics)
# ---------------------------------------------------------------------------

class ChaosEngine(Algorithm):
    """Chaotic move generator using the Logistic Map.
    
    Generates deterministic but highly unpredictable sequences using non-linear
    dynamics (logistic map). Blends chaos with frequency exploitation to remain
    hard to model while still being competitive.
    
    **Type**: Mathematical / Chaos
    **Dynamics**: Logistic Map (r=3.99)
    """
    name = "Chaos Engine"

    def reset(self):
        self._r = 3.99  # fully chaotic regime
        self._x = 0.4 + self.rng.random() * 0.2  # x₀ ∈ (0.4, 0.6)
        self._chaos_ratio = 0.7
        self._reseed_interval = 50

    def _step(self):
        """Advance the logistic map by one step."""
        self._x = self._r * self._x * (1.0 - self._x)
        return self._x

    def _reseed(self, opp_history, my_history):
        """Reseed x₀ from recent outcome hash."""
        if len(opp_history) < 10:
            return
        # Hash from last 10 moves
        h = 0
        for i in range(max(0, len(opp_history)-10), len(opp_history)):
            h = (h * 31 + MOVES.index(opp_history[i])) & 0xFFFFFFFF
        if len(my_history) >= 10:
            for i in range(max(0, len(my_history)-10), len(my_history)):
                h = (h * 37 + MOVES.index(my_history[i])) & 0xFFFFFFFF
        # Map to (0.1, 0.9) to avoid fixed points
        self._x = 0.1 + (h % 10000) / 10000.0 * 0.8

    def choose(self, round_num, my_history, opp_history):
        # Periodic reseed
        if round_num > 0 and round_num % self._reseed_interval == 0:
            self._reseed(opp_history, my_history)

        x = self._step()

        if self.rng.random() < self._chaos_ratio:
            # Chaotic move
            return MOVES[int(x * 3) % 3]
        else:
            # Exploit opponent frequency bias
            if not opp_history:
                return self.rng.choice(MOVES)
            counts = Counter(opp_history[-30:])
            predicted = counts.most_common(1)[0][0]
            return _counter_move(predicted)


# ---------------------------------------------------------------------------
# 71: Level-k Reasoner (Behavioral Economics / Cognitive Science)
# ---------------------------------------------------------------------------

class LevelKReasoner(Algorithm):
    """Cognitive hierarchy model for reasoning depth.
    
    Simulates various levels of recursive thinking (Counter, Counter-Counter, etc.)
    and detects which level the opponent is using. Plays one level higher to exploit
    the opponent's reasoning depth.
    
    **Type**: Behavioral / Game Theory
    **Max Level**: k=4
    """
    name = "Level-k Reasoner"

    def reset(self):
        self._regret = {m: 0.0 for m in MOVES}
        self._detection_window = 50
        self._detected_level = 1  # assume level 1 by default

    def _simulate_level0(self, opp_history, round_idx):
        """Level 0: uniform random — each move equally likely."""
        return None  # can't simulate a specific move, just uniform

    def _simulate_level1_move(self, opp_history, idx):
        """What a Level-1 player (frequency counter) would play at round idx."""
        if idx == 0:
            return MOVES[0]
        history_before = opp_history[:idx]
        counts = Counter(history_before)
        if not counts:
            return MOVES[0]
        predicted = counts.most_common(1)[0][0]
        return _counter_move(predicted)

    def _simulate_level2_move(self, opp_history, my_simulated_l1, idx):
        """Level 2: counter what Level-1 would play."""
        l1_move = self._simulate_level1_move(opp_history, idx)
        return _counter_move(l1_move)

    def _detect_opponent_level(self, opp_history, my_history):
        """Score each level against opponent's actual moves."""
        if len(opp_history) < 20:
            return 1

        window = min(self._detection_window, len(opp_history))
        start = len(opp_history) - window

        scores = [0] * 5  # level 0 through 4

        for t in range(start, len(opp_history)):
            actual = opp_history[t]

            # Level 0: uniform random — score 1 for any match (baseline ~1/3)
            scores[0] += 1  # constant baseline

            # Level 1: frequency counter
            if t > 0:
                # "opp_history" from the opponent's perspective is actually
                # my_history (what the opponent sees as their opponent)
                l1_counts = Counter(my_history[:t])
                if l1_counts:
                    l1_pred = l1_counts.most_common(1)[0][0]
                    l1_move = _counter_move(l1_pred)
                    if actual == l1_move:
                        scores[1] += 3

            # Level 2: counter level 1
            if t > 1:
                l1_counts = Counter(my_history[:t])
                if l1_counts:
                    l1_pred = l1_counts.most_common(1)[0][0]
                    l1_move = _counter_move(l1_pred)
                    l2_move = _counter_move(l1_move)
                    if actual == l2_move:
                        scores[2] += 3

            # Level 3: counter level 2
            if t > 2:
                l1_counts = Counter(my_history[:t])
                if l1_counts:
                    l1_pred = l1_counts.most_common(1)[0][0]
                    l1_move = _counter_move(l1_pred)
                    l2_move = _counter_move(l1_move)
                    l3_move = _counter_move(l2_move)
                    if actual == l3_move:
                        scores[3] += 3

            # Level 4: counter level 3
            if t > 3:
                l1_counts = Counter(my_history[:t])
                if l1_counts:
                    l1_pred = l1_counts.most_common(1)[0][0]
                    l1_move = _counter_move(l1_pred)
                    l2_move = _counter_move(l1_move)
                    l3_move = _counter_move(l2_move)
                    l4_move = _counter_move(l3_move)
                    if actual == l4_move:
                        scores[4] += 3

        return scores.index(max(scores))

    def _regret_matching_move(self):
        """Regret Matching fallback when level is unclear."""
        positive_regret = {m: max(0, r) for m, r in self._regret.items()}
        total = sum(positive_regret.values())
        if total == 0:
            return self.rng.choice(MOVES)
        r = self.rng.random() * total
        cumulative = 0.0
        for m in MOVES:
            cumulative += positive_regret[m]
            if r <= cumulative:
                return m
        return MOVES[-1]

    def choose(self, round_num, my_history, opp_history):
        # Update regret from last round
        if my_history and opp_history:
            opp_last = opp_history[-1]
            my_last = my_history[-1]
            for m in MOVES:
                if BEATS[m] == opp_last:
                    self._regret[m] += 1.0
                elif m == opp_last:
                    self._regret[m] += 0.0
                else:
                    self._regret[m] -= 0.5
                # Reduce regret for what we actually played
                if m == my_last:
                    if BEATS[m] == opp_last:
                        pass  # we won, no regret
                    elif m != opp_last:
                        self._regret[m] -= 1.0

        if len(opp_history) < 20:
            return self.rng.choice(MOVES)

        # Detect opponent's level every 25 rounds
        if round_num % 25 == 0:
            self._detected_level = self._detect_opponent_level(opp_history, my_history)

        level = self._detected_level

        # Play one level above the detected level
        if level == 0:
            # Opponent is random → frequency counter (level 1)
            counts = Counter(opp_history[-30:])
            predicted = counts.most_common(1)[0][0]
            return _counter_move(predicted)
        elif level == 1:
            # Opponent is level 1 → counter their frequency-counter move
            # They predict our most common, counter it
            my_counts = Counter(my_history[-30:])
            my_common = my_counts.most_common(1)[0][0]
            their_move = _counter_move(my_common)
            return _counter_move(their_move)
        elif level == 2:
            # Level 3: counter level 2
            my_counts = Counter(my_history[-30:])
            my_common = my_counts.most_common(1)[0][0]
            l1 = _counter_move(my_common)
            l2 = _counter_move(l1)
            return _counter_move(l2)
        elif level == 3:
            my_counts = Counter(my_history[-30:])
            my_common = my_counts.most_common(1)[0][0]
            l1 = _counter_move(my_common)
            l2 = _counter_move(l1)
            l3 = _counter_move(l2)
            return _counter_move(l3)
        else:
            # High level or unclear → regret matching
            return self._regret_matching_move()

# ===========================================================================
#  COMPETITION META-ALGORITHM (72)
# ===========================================================================


# ---------------------------------------------------------------------------
# 72: UCB-NGram Fusion (Hybrid — exploits complementary phase advantages)
# ---------------------------------------------------------------------------

class UCBNGramFusion(Algorithm):
    """Hybrid strategy fusing bandit exploration with pattern matching.
    
    Combines UCB bandit logic for early-game exploration with N-Gram prediction
    for late-game exploitation. Uses a meta-prediction layer to resist being
    pattern-matched by opponents.
    
    **Type**: Hybrid / Ensemble
    **Components**: UCB + N-Gram
    """
    name = "UCB-NGram Fusion"

    def reset(self):
        import math
        self._math = math

        # Strategy accuracy (cumulative wins per strategy)
        self._strat_wins = {"ucb": 1, "ngram": 1, "freq": 1}
        self._strat_plays = {"ucb": 1, "ngram": 1, "freq": 1}

        # UCB bandit component
        self._ucb_counts = {m: 0 for m in MOVES}
        self._ucb_rewards = {m: 0.0 for m in MOVES}
        self._ucb_total = 0

        # Anti-exploitation: recent loss tracking
        self._recent_results = []  # list of 1(win), 0(draw), -1(loss)

        # Meta-prediction: what strategy we used last round
        self._last_strategy = None
        self._last_move = None

    def _ucb_choose(self, opp_history):
        """UCB1 action selection for exploration."""
        math = self._math
        if self._ucb_total < 3:
            return MOVES[self._ucb_total]
        c = math.sqrt(2)
        best_ucb, best_move = -1e9, MOVES[0]
        for m in MOVES:
            if self._ucb_counts[m] == 0:
                return m
            avg = self._ucb_rewards[m] / self._ucb_counts[m]
            explore = c * math.sqrt(math.log(self._ucb_total) / self._ucb_counts[m])
            ucb = avg + explore
            if ucb > best_ucb:
                best_ucb, best_move = ucb, m
        return best_move

    def _ngram_predict(self, opp_history):
        """N-Gram pattern prediction (n=4,3,2,1)."""
        for n in [4, 3, 2, 1]:
            if n >= len(opp_history):
                continue
            context = tuple(opp_history[-n:])
            counts = {m: 0 for m in MOVES}
            total = 0
            for i in range(len(opp_history) - n):
                if tuple(opp_history[i:i + n]) == context and (i + n) < len(opp_history):
                    counts[opp_history[i + n]] += 1
                    total += 1
            if total >= 2:
                return _counter_move(max(counts, key=counts.get))
        return None

    def _freq_choice(self, opp_history):
        """Frequency counter baseline."""
        if not opp_history:
            return self.rng.choice(MOVES)
        counts = Counter(opp_history[-30:])
        return _counter_move(counts.most_common(1)[0][0])

    def _meta_predict_self(self, my_history):
        """Simulate what a pattern-matcher would predict WE will play.

        If our own moves are predictable, an N-Gram opponent can exploit us.
        Returns the move they'd expect us to play (or None if unpredictable).
        """
        if len(my_history) < 8:
            return None

        # Check if our recent moves have a frequency bias
        recent = my_history[-20:]
        counts = Counter(recent)
        most_common, most_count = counts.most_common(1)[0]
        if most_count / len(recent) > 0.45:
            # We're biased — an opponent would predict this and counter it
            return most_common

        # Check if our last-2 moves form a detectable pattern
        if len(my_history) >= 4:
            bigram = (my_history[-2], my_history[-1])
            pattern_count = 0
            for i in range(len(my_history) - 2):
                if (my_history[i], my_history[i+1]) == bigram:
                    pattern_count += 1
            if pattern_count >= 3:
                return most_common

        return None

    def choose(self, round_num, my_history, opp_history):
        math = self._math

        # Update from last round
        if my_history and opp_history:
            my_last = my_history[-1]
            opp_last = opp_history[-1]

            # Track wins/losses
            if BEATS[my_last] == opp_last:
                self._recent_results.append(1)
            elif my_last == opp_last:
                self._recent_results.append(0)
            else:
                self._recent_results.append(-1)
            if len(self._recent_results) > 30:
                self._recent_results.pop(0)

            # Update UCB component
            self._ucb_counts[my_last] += 1
            self._ucb_total += 1
            if BEATS[my_last] == opp_last:
                self._ucb_rewards[my_last] += 1.0
            elif my_last == opp_last:
                self._ucb_rewards[my_last] += 0.5

            # Update strategy accuracy
            if self._last_strategy is not None:
                won = BEATS[my_last] == opp_last
                self._strat_plays[self._last_strategy] += 1
                if won:
                    self._strat_wins[self._last_strategy] += 1

        if len(opp_history) < 5:
            return self.rng.choice(MOVES)

        # Anti-exploitation: if losing badly, inject randomness
        if len(self._recent_results) >= 10:
            recent_score = sum(self._recent_results[-10:])
            if recent_score < -3:  # losing >6.5 out of 10
                self._last_strategy = "ucb"
                return self.rng.choice(MOVES)

        # Get predictions from all strategies
        ucb_move = self._ucb_choose(opp_history)
        ngram_move = self._ngram_predict(opp_history)
        freq_move = self._freq_choice(opp_history)

        # Strategy selection via softmax on win rates
        candidates = {"ucb": ucb_move, "freq": freq_move}
        if ngram_move is not None:
            candidates["ngram"] = ngram_move

        # Compute softmax weights from win rates
        temps = {}
        for s in candidates:
            win_rate = self._strat_wins[s] / max(self._strat_plays[s], 1)
            temps[s] = math.exp(5.0 * win_rate)  # temperature=5

        total_temp = sum(temps.values())
        weights = {s: temps[s] / total_temp for s in temps}

        # Phase modifiers
        if len(opp_history) < 80:
            weights["ucb"] = weights.get("ucb", 0) * 1.4
        if len(opp_history) > 150 and "ngram" in weights:
            weights["ngram"] *= 1.5

        # Select strategy by weighted random
        total_w = sum(weights.values())
        r = self.rng.random() * total_w
        cumulative = 0.0
        chosen_strat = "freq"
        for s, w in weights.items():
            cumulative += w
            if r <= cumulative:
                chosen_strat = s
                break

        chosen_move = candidates[chosen_strat]

        # META-PREDICTION LAYER: detect if WE are predictable
        self_prediction = self._meta_predict_self(my_history)
        if self_prediction is not None and chosen_move == self_prediction:
            # Opponent likely expects us to play this — rotate!
            # Play what beats their counter to our predicted move
            their_counter = _counter_move(self_prediction)
            chosen_move = _counter_move(their_counter)

        self._last_strategy = chosen_strat
        self._last_move = chosen_move
        return chosen_move


# ===========================================================================
#  UPGRADED COMPETITIVE ALGORITHMS (73-76)
# ===========================================================================


# ---------------------------------------------------------------------------
# 73: Iocaine Powder Plus (upgraded Iocaine Powder)
# ---------------------------------------------------------------------------

class IocainePowderPlus(Algorithm):
    """Upgraded Iocaine Powder with 12 meta-strategies.
    
    A sophisticated ensemble that manages 12 distinct meta-strategies, including
    Markov counters and variable-length pattern matchers. Uses sliding-window
    scoring with exponential decay to follow the best performing predictor.
    
    **Type**: Meta-Strategy / Specialist
    **Experts**: 12 Meta-strategies
    """
    name = "Iocaine Powder Plus"

    def reset(self):
        self._meta_scores = [0.0] * 12

    def _generate_predictions(self, my_history, opp_history):
        """Generate 12 meta-strategy predictions."""
        preds = []

        # 1. Opponent repeats last
        preds.append(opp_history[-1])
        # 2. Opponent counters our last
        preds.append(_counter_move(my_history[-1]))
        # 3. Double bluff: counter what beats their last
        preds.append(_counter_move(_counter_move(opp_history[-1])))
        # 4. Opponent predicts we counter their last, counters that
        preds.append(_counter_move(_counter_move(opp_history[-1])))
        # 5. Opponent plays what beats our second-order guess
        preds.append(_counter_move(_counter_move(my_history[-1])))
        # 6. Frequency-weighted (recent 15)
        counts = Counter(opp_history[-15:])
        preds.append(counts.most_common(1)[0][0])

        # 7. Markov: transition from their last move
        last = opp_history[-1]
        trans = Counter()
        for j in range(len(opp_history) - 1):
            if opp_history[j] == last:
                trans[opp_history[j + 1]] += 1
        if trans:
            preds.append(trans.most_common(1)[0][0])
        else:
            preds.append(last)

        # 8. Bigram: (opp[-2], opp[-1]) → next
        if len(opp_history) >= 3:
            bigram = (opp_history[-2], opp_history[-1])
            bg_counts = Counter()
            for j in range(len(opp_history) - 2):
                if (opp_history[j], opp_history[j+1]) == bigram:
                    bg_counts[opp_history[j+2]] += 1
            if bg_counts:
                preds.append(bg_counts.most_common(1)[0][0])
            else:
                preds.append(opp_history[-1])
        else:
            preds.append(opp_history[-1])

        # 9. Trigram: (opp[-3], opp[-2], opp[-1]) → next
        if len(opp_history) >= 4:
            trigram = (opp_history[-3], opp_history[-2], opp_history[-1])
            tg_counts = Counter()
            for j in range(len(opp_history) - 3):
                if (opp_history[j], opp_history[j+1], opp_history[j+2]) == trigram:
                    tg_counts[opp_history[j+3]] += 1
            if tg_counts:
                preds.append(tg_counts.most_common(1)[0][0])
            else:
                preds.append(opp_history[-1])
        else:
            preds.append(opp_history[-1])

        # 10-12: Mirror versions (from opponent's POV — swap my/opp)
        # Opponent mirrors our Markov prediction of them
        preds.append(_counter_move(preds[6]))  # counter their Markov prediction
        preds.append(_counter_move(preds[7]))  # counter their bigram
        preds.append(_counter_move(preds[8]))  # counter their trigram

        return preds

    def choose(self, round_num, my_history, opp_history):
        if len(opp_history) < 4:
            return self.rng.choice(MOVES)

        preds = self._generate_predictions(my_history, opp_history)

        # Score each meta-strategy — how well did it predict LAST round?
        if len(opp_history) >= 5 and len(my_history) >= 5:
            actual = opp_history[-1]
            # Reconstruct predictions from PREVIOUS round state
            prev_my = my_history[:-1]
            prev_opp = opp_history[:-1]
            prev_preds = self._generate_predictions(prev_my, prev_opp)

            decay = 0.92
            for i in range(12):
                self._meta_scores[i] *= decay
                if prev_preds[i] == actual:
                    self._meta_scores[i] += 1.0

        best_idx = self._meta_scores.index(max(self._meta_scores))
        return _counter_move(preds[best_idx])


# ---------------------------------------------------------------------------
# 74: Dynamic Mixture (upgraded Mixture Model)
# ---------------------------------------------------------------------------

class DynamicMixture(Algorithm):
    """Advanced Mixture Model with dynamic expert management.
    
    Maintains a pool of 8 diverse experts (Markov, WSLS, Frequency, etc.).
    Features expert pruning (removing underperformers) and spawning (cloning and
    mutating top performers) to adapt to shifting opponent strategies.
    
    **Type**: Ensemble / Adaptive
    **Experts**: 8 Dynamic experts
    """
    name = "Dynamic Mixture"

    def reset(self):
        self._n_experts = 8
        self._weights = [1.0 / self._n_experts] * self._n_experts
        self._eta = 0.12
        self._last_votes: list[Move] = []
        self._expert_correct = [0] * self._n_experts
        self._expert_total = [0] * self._n_experts
        self._active = [True] * self._n_experts

    def _generate_expert_votes(self, my_history, opp_history):
        """Generate votes from all 8 experts."""
        votes = []

        # Expert 0: Counter last move
        if opp_history:
            votes.append(_counter_move(opp_history[-1]))
        else:
            votes.append(self.rng.choice(MOVES))

        # Expert 1: Frequency counter
        if opp_history:
            counts = Counter(opp_history)
            votes.append(_counter_move(counts.most_common(1)[0][0]))
        else:
            votes.append(self.rng.choice(MOVES))

        # Expert 2: Markov-1 transition
        if len(opp_history) >= 2:
            last = opp_history[-1]
            trans = Counter()
            for j in range(len(opp_history) - 1):
                if opp_history[j] == last:
                    trans[opp_history[j + 1]] += 1
            if trans:
                votes.append(_counter_move(trans.most_common(1)[0][0]))
            else:
                votes.append(_counter_move(last))
        else:
            votes.append(self.rng.choice(MOVES))

        # Expert 3: Win-Stay Lose-Shift
        if my_history and opp_history:
            if BEATS[my_history[-1]] == opp_history[-1] or my_history[-1] == opp_history[-1]:
                votes.append(my_history[-1])
            else:
                votes.append(_counter_move(opp_history[-1]))
        else:
            votes.append(self.rng.choice(MOVES))

        # Expert 4: Random
        votes.append(self.rng.choice(MOVES))

        # Expert 5: Markov-2 (bigram transition)
        if len(opp_history) >= 3:
            bigram = (opp_history[-2], opp_history[-1])
            bg = Counter()
            for j in range(len(opp_history) - 2):
                if (opp_history[j], opp_history[j+1]) == bigram:
                    bg[opp_history[j+2]] += 1
            if bg:
                votes.append(_counter_move(bg.most_common(1)[0][0]))
            else:
                votes.append(self.rng.choice(MOVES))
        else:
            votes.append(self.rng.choice(MOVES))

        # Expert 6: Recent frequency (last 20 only)
        if len(opp_history) >= 10:
            recent = Counter(opp_history[-20:])
            votes.append(_counter_move(recent.most_common(1)[0][0]))
        else:
            votes.append(self.rng.choice(MOVES))

        # Expert 7: Win-pattern (what beat us before)
        if len(opp_history) >= 5 and len(my_history) >= 5:
            winning_moves = [opp_history[i] for i in range(len(opp_history))
                             if i < len(my_history) and BEATS[opp_history[i]] == my_history[i]]
            if winning_moves:
                common_win = Counter(winning_moves).most_common(1)[0][0]
                votes.append(_counter_move(common_win))
            else:
                votes.append(self.rng.choice(MOVES))
        else:
            votes.append(self.rng.choice(MOVES))

        return votes

    def choose(self, round_num, my_history, opp_history):
        # Update weights from last round
        if self._last_votes and len(opp_history) >= 1:
            opp_last = opp_history[-1]
            for i, vote in enumerate(self._last_votes):
                if not self._active[i]:
                    continue
                if BEATS[vote] == opp_last:
                    loss = 0.0
                    self._expert_correct[i] += 1
                elif vote == opp_last:
                    loss = 0.5
                else:
                    loss = 1.0
                self._expert_total[i] += 1
                self._weights[i] *= (1.0 - self._eta * loss)

            # Normalize active weights
            w_sum = sum(self._weights[i] for i in range(self._n_experts) if self._active[i])
            if w_sum > 0:
                for i in range(self._n_experts):
                    if self._active[i]:
                        self._weights[i] /= w_sum

        # Expert pruning: after 100 rounds, drop <25% accuracy
        if len(opp_history) == 100:
            for i in range(self._n_experts):
                if self._expert_total[i] > 50:
                    acc = self._expert_correct[i] / self._expert_total[i]
                    if acc < 0.25 and i != 4:  # don't prune random
                        self._active[i] = False
                        self._weights[i] = 0.0

        # Expert spawning: every 200 rounds, clone best
        if round_num > 0 and round_num % 200 == 0:
            best_i = max(range(self._n_experts),
                         key=lambda i: self._weights[i] if self._active[i] else -1)
            worst_i = min(range(self._n_experts),
                          key=lambda i: self._weights[i] if self._active[i] else float('inf'))
            if best_i != worst_i and self._active[best_i]:
                self._weights[worst_i] = self._weights[best_i] * 0.5
                self._active[worst_i] = True
                self._expert_correct[worst_i] = self._expert_correct[best_i]
                self._expert_total[worst_i] = self._expert_total[best_i]

        # Generate votes
        votes = self._generate_expert_votes(my_history, opp_history)
        self._last_votes = votes

        # Weighted vote
        move_weights = {m: 0.0 for m in MOVES}
        for i, vote in enumerate(votes):
            if self._active[i]:
                move_weights[vote] += self._weights[i]

        return max(move_weights, key=move_weights.get)


# ---------------------------------------------------------------------------
# 75: Hierarchical Bayesian (upgraded Bayesian Predictor)
# ---------------------------------------------------------------------------

class HierarchicalBayesian(Algorithm):
    """Bayesian predictor with change-point detection.
    
    Uses an evolving Hierarchical Bayesian model to predict moves. Features
    Evidence Maximization to learn priors and KL-divergence-based change-point
    detection to respond to abrupt shifts in the opponent's strategy.
    
    **Type**: Bayesian / Probabilistic
    **Logic**: Change-point detection
    """
    name = "Hierarchical Bayesian"

    def reset(self):
        import math
        self._math = math
        # Learned prior (starts uninformative)
        self._alpha = {m: 1.0 for m in MOVES}
        # Change-point detection
        self._prev_distribution = {m: 1.0 / 3.0 for m in MOVES}
        self._kl_threshold = 0.5
        self._since_changepoint = 0

    def _dirichlet_sample(self, alphas):
        """Sample from Dirichlet by sampling independent Gammas."""
        samples = {}
        for m in MOVES:
            samples[m] = self.rng.gammavariate(max(alphas[m], 0.01), 1.0)
        total = sum(samples.values())
        if total > 0:
            return {m: samples[m] / total for m in MOVES}
        return {m: 1.0 / 3.0 for m in MOVES}

    def _kl_divergence(self, p, q):
        """Compute KL(p || q) for move distributions."""
        math = self._math
        kl = 0.0
        for m in MOVES:
            pi = max(p[m], 1e-10)
            qi = max(q[m], 1e-10)
            kl += pi * math.log(pi / qi)
        return kl

    def _compute_prediction(self, opp_window):
        """Compute posterior prediction from a window."""
        counts = Counter(opp_window)
        alphas = {m: self._alpha[m] + counts.get(m, 0) for m in MOVES}
        return self._dirichlet_sample(alphas)

    def choose(self, round_num, my_history, opp_history):
        if len(opp_history) < 3:
            return self.rng.choice(MOVES)

        # Change-point detection: compare recent vs previous distribution
        if len(opp_history) >= 20:
            recent_counts = Counter(opp_history[-20:])
            n = len(opp_history[-20:])
            recent_dist = {m: (recent_counts.get(m, 0) + 0.5) / (n + 1.5)
                           for m in MOVES}

            kl = self._kl_divergence(recent_dist, self._prev_distribution)
            if kl > self._kl_threshold:
                # Change point detected — reset prior to recent empirical
                self._alpha = {m: max(recent_dist[m] * 3, 0.5) for m in MOVES}
                self._since_changepoint = 0

            self._prev_distribution = recent_dist
        self._since_changepoint += 1

        # Learn prior: update α toward observed frequencies
        if len(opp_history) >= 50 and round_num % 25 == 0:
            long_counts = Counter(opp_history)
            total_n = len(opp_history)
            for m in MOVES:
                observed_frac = long_counts.get(m, 0) / total_n
                # Slowly adjust prior toward observed distribution
                self._alpha[m] = 0.9 * self._alpha[m] + 0.1 * (observed_frac * 3 + 0.5)

        # Multi-window ensemble: windows of 20, 50, 100
        predictions = []
        ensemble_weights = [0.5, 0.3, 0.2]  # recent windows weighted more

        for window_size, w in zip([20, 50, 100], ensemble_weights):
            if len(opp_history) >= window_size:
                pred = self._compute_prediction(opp_history[-window_size:])
                predictions.append((pred, w))
            elif opp_history:
                pred = self._compute_prediction(opp_history)
                predictions.append((pred, w))

        if not predictions:
            return self.rng.choice(MOVES)

        # Weighted combination
        combined = {m: 0.0 for m in MOVES}
        total_w = sum(w for _, w in predictions)
        for pred, w in predictions:
            for m in MOVES:
                combined[m] += pred[m] * w / total_w

        predicted = max(combined, key=combined.get)
        return _counter_move(predicted)


# ---------------------------------------------------------------------------
# 76: Self-Model Detector (upgraded Anti-Strategy Detector)
# ---------------------------------------------------------------------------

class SelfModelDetector(Algorithm):
    """Anti-strategy detector with self-simulation.
    
    Identifies the opponent's current strategy by simulating what 10 different
    archetype strategies would play in the current context. Switches to the
    hard counter for the most likely identified strategy.
    
    **Type**: Anti-Meta / Classifier
    **Archetypes**: 10 Strategy candidates
    """
    name = "Self-Model Detector"

    def reset(self):
        self._detector_scores = [0.0] * 10
        self._detector_names = [
            "constant", "cycle", "mirror", "counter",
            "frequency", "markov", "wsls",
            "anti_tft", "pattern_cycle", "decay_freq"
        ]

    def _simulate_prediction(self, detector_idx, my_history, opp_history, t):
        """Simulate what detector i would predict for round t."""
        if t < 2:
            return Move.ROCK

        # All predictions are for what the OPPONENT will play
        if detector_idx == 0:  # Constant: repeats last
            return opp_history[t-1]
        elif detector_idx == 1:  # Cycle: R→P→S
            return MOVES[(MOVES.index(opp_history[t-1]) + 1) % 3]
        elif detector_idx == 2:  # Mirror: copies our last
            return my_history[t-1] if t-1 < len(my_history) else Move.ROCK
        elif detector_idx == 3:  # Counter: counters our last
            return _counter_move(my_history[t-1]) if t-1 < len(my_history) else Move.ROCK
        elif detector_idx == 4:  # Frequency: plays their most common
            counts = Counter(opp_history[:t])
            return counts.most_common(1)[0][0] if counts else Move.ROCK
        elif detector_idx == 5:  # Markov: transition from last move
            if t < 2:
                return Move.ROCK
            last = opp_history[t-1]
            trans = Counter()
            for j in range(t - 1):
                if opp_history[j] == last and j + 1 < t:
                    trans[opp_history[j + 1]] += 1
            return trans.most_common(1)[0][0] if trans else last
        elif detector_idx == 6:  # WSLS: win-stay lose-shift
            if t < 2:
                return Move.ROCK
            opp_prev = opp_history[t-1]
            my_prev = my_history[t-2] if t-2 < len(my_history) else Move.ROCK
            if BEATS[opp_prev] == my_prev or opp_prev == my_prev:
                return opp_prev  # opponent stays
            else:
                return _counter_move(my_prev)  # opponent shifts
        elif detector_idx == 7:  # Anti-TFT: plays what beats our last
            return _counter_move(my_history[t-1]) if t-1 < len(my_history) else Move.ROCK
        elif detector_idx == 8:  # Pattern cycle: longer cycle detection
            if t >= 6:
                for period in [2, 3, 4]:
                    match = True
                    for k in range(period):
                        if opp_history[t - 1 - k] != opp_history[t - 1 - k - period]:
                            match = False
                            break
                    if match:
                        return opp_history[t - period]
            return opp_history[t-1]
        elif detector_idx == 9:  # Decay frequency: recent-weighted
            if t < 5:
                return Move.ROCK
            weight_sum = {m: 0.0 for m in MOVES}
            for j in range(max(0, t - 30), t):
                age = t - j
                weight_sum[opp_history[j]] += 0.95 ** age
            return max(weight_sum, key=weight_sum.get)

        return Move.ROCK

    def choose(self, round_num, my_history, opp_history):
        if len(opp_history) < 6:
            return self.rng.choice(MOVES)

        # Score each detector on recent accuracy
        t = len(opp_history)
        actual = opp_history[-1]

        decay = 0.9
        for i in range(10):
            self._detector_scores[i] *= decay
            # What would this detector have predicted for the last round?
            pred = self._simulate_prediction(i, my_history, opp_history, t - 1)
            if pred == actual:
                self._detector_scores[i] += 1.0

        # Predict NEXT move with each detector
        predictions = []
        for i in range(10):
            predictions.append(self._simulate_prediction(i, my_history, opp_history, t))

        # Use best detector's prediction
        best = self._detector_scores.index(max(self._detector_scores))
        return _counter_move(predictions[best])



# ===========================================================================
#  CREATIVE BOOST ALGORITHMS (77-100)
# ===========================================================================


# ---------------------------------------------------------------------------
# 77: PiBot (Digits of Pi)
# ---------------------------------------------------------------------------

class PiBot(Algorithm):
    """Deterministic sequence generator based on the digits of Pi.
    
    Extracts digits from the decimal expansion of Pi to determine moves.
    Maps digits 0-9 into Rock, Paper, or Scissors, creating a high-entropy
    sequence that is difficult to predict without knowing the source.
    
    **Type**: Mathematical / Deterministic
    **Source**: Digits of Pi
    """
    name = "PiBot"
    _PI_DIGITS = "31415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679821480865132823066470938446095505822317253594081284811174502841027019385211055596446229489549303819644288109756659334461284756482337867831652712019091456485669234603486104543266482133936072602491412737245870066063155881748815209209628292540917153643678925903600113305305488204665213841469519415116094330572703657595919530921861173819326117931051185480744623799627495673518857527248912279381830119491298336733624406566430860213949463952247371907021798609437027705392171762931767523846748184676694051320005681271452635608277857713427577896091736371787214684409012249534301465495853710507922796892589235420199561121290219608640344181598136297747713099605187072113499999983729780499510597317328160963185950244594553469083026425223082533446850352619311881710100031378387528865875332083814206171776691473035982534904287554687311595628638823537875937519577818577805321712268066130019278766111959092164201989"

    def choose(self, round_num, my_history, opp_history):
        digit = int(self._PI_DIGITS[round_num % len(self._PI_DIGITS)])
        if digit <= 2:
            return Move.ROCK
        elif digit <= 5:
            return Move.PAPER
        else:
            return Move.SCISSORS


# ---------------------------------------------------------------------------
# 78: GoldenRatio (Chaos via Phi)
# ---------------------------------------------------------------------------

class GoldenRatio(Algorithm):
    """Quasi-periodic sequence generator based on the Golden Ratio (φ).
    
    Uses the fractional parts of multiples of the Golden Ratio to generate
    moves. This produces a low-discrepancy, deterministic sequence that appears
    stochastic to simple pattern matchers.
    
    **Type**: Mathematical / Chaos
    **Constant**: Phi (1.6180...)
    """
    name = "Golden Ratio"
    _PHI = 1.61803398875

    def choose(self, round_num, my_history, opp_history):
        val = (round_num * self._PHI) % 1.0
        return MOVES[int(val * 3)]


# ---------------------------------------------------------------------------
# 79: StockBroker (Market Simulation)
# ---------------------------------------------------------------------------

class StockBroker(Algorithm):
    """Market-based strategy treating moves as volatile stocks.
    
    Assigns a value to each move that increases when it would have won and
    decreases otherwise. Incorporates "market noise" and executes the move with
    the highest current valuation.
    
    **Type**: Economic / Simulation
    **Logic**: Market Valuation
    """
    name = "Stock Broker"

    def reset(self):
        self._prices = {m: 100.0 for m in MOVES}
        self._volatility = 2.0

    def choose(self, round_num, my_history, opp_history):
        if opp_history:
            last_opp = opp_history[-1]
            # Update market
            for m in MOVES:
                # Inflation/Decay
                self._prices[m] *= 0.99
                
                # Market reaction
                if BEATS[m] == last_opp:       # Winning move
                    self._prices[m] *= 1.05    # +5%
                elif m == last_opp:            # Draw move
                    self._prices[m] *= 1.0     # 0%
                else:                          # Losing move
                    self._prices[m] *= 0.90    # -10%
                
                # Add noise
                noise = (self.rng.random() - 0.5) * self._volatility
                self._prices[m] += noise

        # Pick highest value
        return max(self._prices, key=self._prices.get)


# ---------------------------------------------------------------------------
# 80: QuantumCollapse (Superposition)
# ---------------------------------------------------------------------------

class QuantumCollapse(Algorithm):
    """Probabilistic strategy based on wave function collapse.
    
    Maintains a "superposition" (probability distribution) of moves. Wins cause
    constructive interference (reinforcing the move), while losses cause
    destructive interference (collapsing the probability).
    
    **Type**: Probabilistic / Stochastic
    **Logic**: Interference & Collapse
    """
    name = "Quantum Collapse"

    def reset(self):
        self._probs = {m: 1.0/3.0 for m in MOVES}

    def choose(self, round_num, my_history, opp_history):
        # Collapse/Observe based on last round result
        if my_history and opp_history:
            my_last = my_history[-1]
            opp_last = opp_history[-1]
            
            if BEATS[my_last] == opp_last:
                # Win: Constructive interference
                self._probs[my_last] *= 1.5
            elif BEATS[opp_last] == my_last:
                # Loss: Destructive interference (collapse)
                self._probs[my_last] *= 0.1
            
            # Renormalize
            total = sum(self._probs.values())
            for m in MOVES:
                self._probs[m] /= total
            
            # Entropy injection (uncertainty principle)
            for m in MOVES:
                self._probs[m] = self._probs[m] * 0.9 + 0.1 * (1.0/3.0)

        # Sample from distribution
        r = self.rng.random()
        cumulative = 0.0
        for m in MOVES:
            cumulative += self._probs[m]
            if r <= cumulative:
                return m
        return MOVES[-1]


# ---------------------------------------------------------------------------
# 81: SoundWave (Oscillation)
# ---------------------------------------------------------------------------

class SoundWave(Algorithm):
    """Oscillatory move generator based on interfering sine waves.
    
    Combines two sine waves of different frequencies to generate moves. This
    creates a complex periodic signal representing the "sound" of the strategy,
    difficult to decode for simple frequency counters.
    
    **Type**: Mathematical / Signal
    **Source**: Sine Interference
    """
    name = "Sound Wave"

    def choose(self, round_num, my_history, opp_history):
        import math
        # Combine two waves: fast freq and slow freq
        t = round_num
        wave = math.sin(t * 0.2) + math.sin(t * 0.05)
        # wave range is approx [-2, 2]
        
        # Map to 0, 1, 2
        value = (wave + 2) / 4.0  # Normalized 0-1
        idx = int(value * 3) % 3
        return MOVES[idx]



# ---------------------------------------------------------------------------
# 82: Ackermann (Recursive Depth)
# ---------------------------------------------------------------------------

class Ackermann(Algorithm):
    """Recursive strategy using the Ackermann function.
    
    Computes a non-linear lookback depth based on the extremely fast-growing
    Ackermann function. Uses this dynamic depth to select a historical move to
    counter, making the "attention" of the bot highly non-linear.
    
    **Type**: Mathematical / Recursive
    **Function**: Ackermann(m, n)
    """
    name = "Ackermann"
    _MEMO = {}

    def _ackermann(self, m, n):
        if (m, n) in self._MEMO:
            return self._MEMO[(m, n)]
        if m == 0:
            return n + 1
        elif n == 0:
            res = self._ackermann(m - 1, 1)
        else:
            res = self._ackermann(m - 1, self._ackermann(m, n - 1))
        
        self._MEMO[(m, n)] = res
        return res

    def choose(self, round_num, my_history, opp_history):
        if not opp_history:
            return self.rng.choice(MOVES)

        # Inputs must be small to avoid recursion depth errors
        # A(3, n) grows very deep. We stick to m <= 3, n <= 6
        m = (round_num // 10) % 4
        n = round_num % 5
        
        try:
            val = self._ackermann(m, n)
        except RecursionError:
            val = 10  # Fallback
            
        # Use value as lookback index
        idx = val % len(opp_history)
        predicted = opp_history[-idx-1]
        return _counter_move(predicted)


# ---------------------------------------------------------------------------
# 83: PrimeHunter (Prime Number Strategy)
# ---------------------------------------------------------------------------

class PrimeHunter(Algorithm):
    """Reactive strategy that only strikes on prime-numbered rounds.
    
    Strikes with a hard counter on rounds that are prime numbers, while playing
    purely random on composite rounds. This selective aggression creates a temporal
    pattern that is difficult for non-mathematical bots to track.
    
    **Type**: Mathematical / Reactive
    **Aggression**: Prime Rounds Only
    """
    name = "Prime Hunter"

    def _is_prime(self, n):
        if n < 2: return False
        if n == 2: return True
        if n % 2 == 0: return False
        for i in range(3, int(n**0.5) + 1, 2):
            if n % i == 0:
                return False
        return True

    def choose(self, round_num, my_history, opp_history):
        # round_num is 0-indexed, so we check using mathematical 1-based index
        n = round_num + 1
        if self._is_prime(n):
            if opp_history:
                return _counter_move(opp_history[-1])
            return self.rng.choice(MOVES)
        else:
            return self.rng.choice(MOVES)


# ---------------------------------------------------------------------------
# 84: CompressionBot (Information Theory)
# ---------------------------------------------------------------------------

class CompressionBot(Algorithm):
    """Predictor based on Information Theory and NCD.
    
    Uses zlib compression to estimate the Normalized Compression Distance (NCD).
    Identifies the next move that would result in the highest compression ratio
    for the opponent's sequence, implying it's the most likely continuation.
    
    **Type**: Information Theory / ML
    **Algorithm**: Zlib-based NCD
    """
    name = "Compression Bot"

    def choose(self, round_num, my_history, opp_history):
        if len(opp_history) < 10:
            return self.rng.choice(MOVES)
            
        import zlib
        
        # Convert history to bytes string
        # e.g. R, P, S -> "RPS"
        # We define rough mapping for compression text
        char_map = {Move.ROCK: b'R', Move.PAPER: b'P', Move.SCISSORS: b'S'}
        
        history_bytes = b"".join(char_map[m] for m in opp_history)
        
        best_pred = None
        min_size = float('inf')
        
        # Hypothesize opponent's next move
        for m in MOVES:
            candidate = history_bytes + char_map[m]
            # Compress
            compressed = zlib.compress(candidate)
            size = len(compressed)
            
            if size < min_size:
                min_size = size
                best_pred = m
            elif size == min_size and self.rng.random() < 0.5:
                best_pred = m
                
        return _counter_move(best_pred)


# ---------------------------------------------------------------------------
# 85: EquilibriumBreaker (Nash Deviation)
# ---------------------------------------------------------------------------

class EquilibriumBreaker(Algorithm):
    """Frequency-based exploiter that punishes Nash deviations.
    
    Monitors move distributions and identifies any deviation from the optimal
    33.3% Nash equilibrium. Once a bias is detected, it scales its aggressive
    countering to match the severity of the deviation.
    
    **Type**: Game Theory / Reactive
    **Target**: Nash Deviation
    """
    name = "Equilibrium Breaker"

    def choose(self, round_num, my_history, opp_history):
        if not opp_history:
            return self.rng.choice(MOVES)
            
        counts = Counter(opp_history)
        total = len(opp_history)
        
        # Find if any move is over-represented
        most_common = counts.most_common(1)[0]
        move, count = most_common
        freq = count / total
        
        if freq > 0.35:
            # Exploit: play the counter to their favorite move
            # But mix in some randomness
            if self.rng.random() < (freq - 0.33) * 3: # Scale aggression with deviation
                return _counter_move(move)
        
        return self.rng.choice(MOVES)


# ---------------------------------------------------------------------------
# 86: DelayedMirror (Lagged Copy)
# ---------------------------------------------------------------------------

class DelayedMirror(Algorithm):
    """Copycat strategy with a temporal lag.
    
    Mirrors the opponent's move from two rounds ago. Designed to defeat bots
    that expect immediate reactions or rely on predicting immediate mirroring
    patterns.
    
    **Type**: Mimicry / Deceptive
    **Lag**: 2 Rounds
    """
    name = "Delayed Mirror"

    def choose(self, round_num, my_history, opp_history):
        if len(opp_history) < 2:
            return self.rng.choice(MOVES)
        return opp_history[-2]



# ---------------------------------------------------------------------------
# 87: GeneSequencer (Biological Pattern)
# ---------------------------------------------------------------------------

class GeneSequencer(Algorithm):
    """Sequence matcher inspired by biological gene alignment.
    
    Treats move history as a DNA sequence and searches for the current context in
    the past. Unlike exact matchers, it allows for a "mutation" (single mismatch)
    during alignment to capture fuzzy patterns.
    
    **Type**: Pattern / Biological
    **Tolerance**: 1 Mutation (Hamming=1)
    """
    name = "Gene Sequencer"

    def choose(self, round_num, my_history, opp_history):
        if len(opp_history) < 20:
            return self.rng.choice(MOVES)
            
        # Target gene: last 6 moves
        gene_len = 6
        target = opp_history[-gene_len:]
        
        # Search history for approximate matches (hamming distance <= 1)
        best_match_idx = -1
        
        # Look back from end
        for i in range(len(opp_history) - gene_len - 1, -1, -1):
            candidate = opp_history[i : i+gene_len]
            if len(candidate) != gene_len: continue
            
            # Hamming distance
            dist = sum(1 for a, b in zip(candidate, target) if a != b)
            
            if dist <= 1:
                # Found a homologous sequence!
                best_match_idx = i
                break
                
        if best_match_idx != -1:
            # Predict the move that followed the gene
            return _counter_move(opp_history[best_match_idx + gene_len])
            
        return self.rng.choice(MOVES)


# ---------------------------------------------------------------------------
# 88: Zodiac (Cyclic Personalities)
# ---------------------------------------------------------------------------

class Zodiac(Algorithm):
    """Ensemble of 12 distinct personalities cycling by round.
    
    Assigns a different strategy "sign" to each round in a 12-round cycle.
    Strategies range from aggressive and stubborn to analytic and chaotic,
    providing a wide variety of behaviors in a predictable but complex pattern.
    
    **Type**: Meta / Personality
    **Cycle**: 12 Sign Rotation
    """
    name = "Zodiac"

    def choose(self, round_num, my_history, opp_history):
        sign = round_num % 12
        
        if sign == 0:   # Aries: Rock (Aggressive)
            return Move.ROCK
        elif sign == 1: # Taurus: Repeat last (Stubborn)
            return my_history[-1] if my_history else Move.ROCK
        elif sign == 2: # Gemini: R or P (Dual)
            return self.rng.choice([Move.ROCK, Move.PAPER])
        elif sign == 3: # Cancer: Paper (Defensive/Shell)
            return Move.PAPER
        elif sign == 4: # Leo: Play whatever has won most (Pride)
            if not my_history: return Move.ROCK
            wins = Counter([m for i, m in enumerate(my_history) if i < len(opp_history) and BEATS[m] == opp_history[i]])
            return wins.most_common(1)[0][0] if wins else Move.ROCK
        elif sign == 5: # Virgo: Frequency Counter (Analytic)
            if not opp_history: return Move.PAPER
            return _counter_move(Counter(opp_history).most_common(1)[0][0])
        elif sign == 6: # Libra: Play least frequent (Balance)
            if not my_history: return Move.PAPER
            return Counter(my_history).most_common()[-1][0]
        elif sign == 7: # Scorpio: Counter last (Vengeful)
            if not opp_history: return Move.SCISSORS
            return _counter_move(opp_history[-1])
        elif sign == 8: # Sagittarius: Random (Free spirit)
            return self.rng.choice(MOVES)
        elif sign == 9: # Capricorn: WSLS (Practical)
            if len(my_history) < 2: return Move.ROCK
            last_res = 1 if BEATS[my_history[-1]] == opp_history[-1] else -1 if BEATS[opp_history[-1]] == my_history[-1] else 0
            if last_res >= 0: return my_history[-1]
            else: return _counter_move(opp_history[-1])
        elif sign == 10: # Aquarius: Chaos
            return MOVES[int((round_num * 1.618) % 1 * 3)]
        else: # Pisces: Mirror (Empathetic)
            return opp_history[-1] if opp_history else Move.PAPER


# ---------------------------------------------------------------------------
# 89: NeuroEvo (Single Neuron Mutation)
# ---------------------------------------------------------------------------

class NeuroEvo(Algorithm):
    """Minimal neural network with weight mutation.
    
    A single-layer perceptron that maps recent context to move scores. Whenever
    the bot experiences a losing streak, it "evolves" by injecting random noise
    into its weights to find a better configuration.
    
    **Type**: ML / Evolutionary
    **Model**: Perceptron
    """
    name = "Neuro Evo"

    def reset(self):
        # 3 moves * 2 players = 6 inputs. 3 outputs (scores for R,P,S).
        # Weights: 6x3 matrix flatten to 18
        self._weights = [self.rng.uniform(-1, 1) for _ in range(18)]
        self._streak = 0

    def choose(self, round_num, my_history, opp_history):
        if not opp_history:
            return self.rng.choice(MOVES)

        # Update streak
        last_res = 0 # draw
        if BEATS[my_history[-1]] == opp_history[-1]:
            self._streak = 0
        elif BEATS[opp_history[-1]] == my_history[-1]:
            self._streak += 1
        else:
            self._streak = 0
            
        # Mutate if stuck losing
        if self._streak >= 3:
            self._weights = [w + self.rng.uniform(-0.5, 0.5) for w in self._weights]
            self._streak = 0

        # Encode input (One-hot for last moves: My[R,P,S], Opp[R,P,S])
        inputs = [0] * 6
        inputs[MOVES.index(my_history[-1])] = 1
        inputs[3 + MOVES.index(opp_history[-1])] = 1
        
        # Forward pass
        scores = [0.0] * 3
        for out_i in range(3):
            for in_i in range(6):
                scores[out_i] += inputs[in_i] * self._weights[in_i * 3 + out_i]

        # Argmax
        best_idx = scores.index(max(scores))
        return MOVES[best_idx]


# ---------------------------------------------------------------------------
# 90: GeometryBot (Centroid Strategy)
# ---------------------------------------------------------------------------

class GeometryBot(Algorithm):
    """Vector-based strategy using centroid centering.
    
    Maps moves to vectors on a 2D unit circle and calculates the centroid of the
    opponent's distribution. Plays the move corresponding to the vector opposite
    the centroid to counter the opponent's spatial bias.
    
    **Type**: Mathematical / Geometric
    **Domain**: 2D Unit Circle
    """
    name = "Geometry Bot"

    def choose(self, round_num, my_history, opp_history):
        if not opp_history:
            return self.rng.choice(MOVES)

        # Unit vectors
        vecs = {
            Move.ROCK: (1.0, 0.0),
            Move.PAPER: (-0.5, 0.8660254),
            Move.SCISSORS: (-0.5, -0.8660254)
        }
        
        # Compute centroid of last 20 moves
        x_sum, y_sum = 0.0, 0.0
        window = opp_history[-20:]
        for m in window:
            vx, vy = vecs[m]
            x_sum += vx
            y_sum += vy
            
        avg_x = x_sum / len(window)
        avg_y = y_sum / len(window)
        
        # We want to counter the bias.
        # If bias is towards Rock ((1,0)), we want to play Paper.
        # So we look for the move vector that is closest to (-avg_y, avg_x)? 
        # No, simpler: Expected move is roughly the centroid direction.
        # We want to beat the expected move.
        # If expected is Rock, we want Paper.
        # Paper is +120 degrees from Rock.
        # So specific target angle = centroid_angle + 120 degrees.
        
        import math
        centroid_angle = math.atan2(avg_y, avg_x)
        target_angle = centroid_angle + (2 * math.pi / 3) # +120 deg
        
        # Find move closest to target angle
        best_move = None
        max_dot = -2.0
        
        tx = math.cos(target_angle)
        ty = math.sin(target_angle)
        
        for m, (vx, vy) in vecs.items():
            dot = tx*vx + ty*vy
            if dot > max_dot:
                max_dot = dot
                best_move = m
                
        return best_move



# ---------------------------------------------------------------------------
# 91: GamblersFallacy (Monte Carlo Fallacy)
# ---------------------------------------------------------------------------

class GamblersFallacy(Algorithm):
    """Predictor based on the Monte Carlo fallacy.
    
    Assumes that if a move has not occurred recently, it is "due" to happen.
    Tracks the counts of opponent moves in a recent window and predicts that the
    least frequent move is the most likely next choice.
    
    **Type**: Psychological / Heuristic
    **Window**: 20 Rounds
    """
    name = "Gambler's Fallacy"

    def choose(self, round_num, my_history, opp_history):
        if not opp_history:
            return self.rng.choice(MOVES)
            
        window = opp_history[-20:]
        counts = Counter(window)
        
        # Find least common move
        # Note: most_common[:-1] works, but we need to handle 0 counts
        best_move = None
        min_count = float('inf')
        
        for m in MOVES:
            c = counts[m]
            if c < min_count:
                min_count = c
                best_move = m
            elif c == min_count and self.rng.random() < 0.5:
                best_move = m
                
        # If we think opponent will play 'best_move', we counter it
        return _counter_move(best_move)


# ---------------------------------------------------------------------------
# 92: NashStabilizer (Forced Balance)
# ---------------------------------------------------------------------------

class NashStabilizer(Algorithm):
    """Self-balancing strategy targeting a perfect Nash distribution.
    
    Monitors its own move history and intentionally plays the move it has used
    least. This enforces an asymptotically uniform distribution, making the bot
    highly resistant to frequency-based exploitation.
    
    **Type**: Defensive / Game Theory
    **Target**: Uniform Distribution
    """
    name = "Nash Stabilizer"

    def choose(self, round_num, my_history, opp_history):
        if not my_history:
            return self.rng.choice(MOVES)
            
        counts = Counter(my_history)
        
        # Play the move I have played least
        best_move = None
        min_count = float('inf')
        
        for m in MOVES:
            c = counts[m]
            if c < min_count:
                min_count = c
                best_move = m
            elif c == min_count and self.rng.random() < 0.5:
                best_move = m
                
        return best_move


# ---------------------------------------------------------------------------
# 93: StubbornLoser (Contrarian WSLS)
# ---------------------------------------------------------------------------

class StubbornLoser(Algorithm):
    """Contrarian strategy that doubles down on losses.
    
    The inverse of the Win-Stay Lose-Shift (WSLS) logic. It shifts its move after
    a win and stubbornly repeats its move after a loss, frustrating opponents
    who expect standard reactive patterns.
    
    **Type**: Reactive / Contrarian
    **Logic**: Lose-Stay, Win-Shift
    """
    name = "Stubborn Loser"

    def choose(self, round_num, my_history, opp_history):
        if not my_history:
            return self.rng.choice(MOVES)
            
        my_last = my_history[-1]
        opp_last = opp_history[-1]
        
        if BEATS[my_last] == opp_last:
            # Won: Shift
            # Shift to what? Random non-last
            available = [m for m in MOVES if m != my_last]
            return self.rng.choice(available)
        elif my_last == opp_last:
            # Draw: Random
            return self.rng.choice(MOVES)
        else:
            # Lost: Stay (Stubborn)
            return my_last


# ---------------------------------------------------------------------------
# 94: TraitorMirror (Mirror with Betrayal)
# ---------------------------------------------------------------------------

class TraitorMirror(Algorithm):
    """Mirror strategy with periodic betrayal.
    
    Builds the opponent's "trust" by mirroring their moves 80% of the time.
    In the remaining 20%, it counter-plays the expected mirror move to exploit
    players who have adapted to the copycat behavior.
    
    **Type**: Mimicry / Deceptive
    **Base**: 80% Mirror
    """
    name = "Traitor Mirror"

    def choose(self, round_num, my_history, opp_history):
        if not opp_history:
            return self.rng.choice(MOVES)
            
        opp_last = opp_history[-1]
        
        if self.rng.random() < 0.8:
            # Mirror (Trust)
            return opp_last
        else:
            # Betray (Counter)
            return _counter_move(opp_last)


# ---------------------------------------------------------------------------
# 95: OpponentPersona (Pattern Classification)
# ---------------------------------------------------------------------------

class OpponentPersona(Algorithm):
    """Classifier that counters the opponent's general persona.
    
    Categorizes the opponent as Aggressive, Defensive, or Evasive based on their
    most common move choice. Switches to a dedicated counter-persona to exploit
    the detected behavioral bias.
    
    **Type**: Classifier / Strategy
    **Personas**: 3 Archetypes
    """
    name = "Opponent Persona"

    def choose(self, round_num, my_history, opp_history):
        if len(opp_history) < 10:
            return self.rng.choice(MOVES)
            
        counts = Counter(opp_history[-30:])
        total = sum(counts.values())
        
        if counts[Move.ROCK] / total > 0.4:
            return Move.PAPER # Counter Aggressive
        elif counts[Move.PAPER] / total > 0.4:
            return Move.SCISSORS # Counter Defensive
        elif counts[Move.SCISSORS] / total > 0.4:
            return Move.ROCK # Counter Evasive
            
        return self.rng.choice(MOVES)


# ---------------------------------------------------------------------------
# 96: ExponentialBackoff (Recovery Mode)
# ---------------------------------------------------------------------------

class ExponentialBackoff(Algorithm):
    """Reactive strategy with a randomization "reset" mechanism.
    
    Monitors for losing streaks and enters a "backoff" mode when detected. During
    backoff, it plays purely random for a duration that grows exponentially with
    the streak length, breaking the opponent's predictive lock.
    
    **Type**: Defensive / Stochastic
    **Recovery**: Exponential Randomization
    """
    name = "Exponential Backoff"
    
    def reset(self):
        self._backoff_rounds = 0
        
    def choose(self, round_num, my_history, opp_history):
        if self._backoff_rounds > 0:
            self._backoff_rounds -= 1
            return self.rng.choice(MOVES)
            
        if not my_history:
            return self.rng.choice(MOVES)
            
        # Check losing streak
        losses = 0
        for i in range(len(my_history)-1, -1, -1):
            if BEATS[opp_history[i]] == my_history[i]:
                losses += 1
            else:
                break
                
        if losses > 1:
            self._backoff_rounds = 2 ** min(5, losses) # Cap at 32 rounds
            return self.rng.choice(MOVES)
            
        # Default strategy: Counter last (heuristic)
        return _counter_move(opp_history[-1])


# ---------------------------------------------------------------------------
# 97: PatternBreaker (Self-Analysis)
# ---------------------------------------------------------------------------

class PatternBreaker(Algorithm):
    """Self-monitoring bot that prevents predictable behavior.
    
    Analyzes its own recent move history for repetitions or common cycles. If a
    pattern is detected, it intentionally deviates from the expected next move,
    ensuring it remains difficult for the opponent to model.
    
    **Type**: Anti-Meta / Self-Analysis
    **Trigger**: Self-Pattern Detection
    """
    name = "Pattern Breaker"

    def choose(self, round_num, my_history, opp_history):
        if len(my_history) < 3:
            return self.rng.choice(MOVES)
            
        # Check for repetition R-R-R
        if my_history[-1] == my_history[-2] == my_history[-3]:
            # Don't play it again!
            bad_move = my_history[-1]
            return self.rng.choice([m for m in MOVES if m != bad_move])
            
        # Check for cycle R-P-S
        # (Assuming index order is R,P,S)
        last_idx = MOVES.index(my_history[-1])
        prev_idx = MOVES.index(my_history[-2])
        if (prev_idx + 1) % 3 == last_idx:
            # We are cycling up. Don't complete the cycle.
            # Next in cycle would be (last_idx + 1) % 3
            bad_move = MOVES[(last_idx + 1) % 3]
            return self.rng.choice([m for m in MOVES if m != bad_move])
            
        # Default: Counter opponent
        return _counter_move(opp_history[-1])


# ---------------------------------------------------------------------------
# 98: SlidingWindowVote (Ensemble)
# ---------------------------------------------------------------------------

class SlidingWindowVote(Algorithm):
    """Majority-vote ensemble using multi-scale historical windows.
    
    Generates predictions from three distinct time horizons (Short, Medium, and
    Long term). Plays the move that beats the majority prediction, providing a
    stable response that balances recent shifts with long-term trends.
    
    **Type**: Ensemble / Statistical
    **Windows**: 5, 20, 100 Rounds
    """
    name = "Sliding Window Vote"

    def choose(self, round_num, my_history, opp_history):
        if len(opp_history) < 5:
            return self.rng.choice(MOVES)
            
        votes = []
        for w in [5, 20, 100]:
            if len(opp_history) >= w:
                most_common = Counter(opp_history[-w:]).most_common(1)[0][0]
                votes.append(most_common)
                
        if not votes:
            return self.rng.choice(MOVES)
            
        # Majority vote on what OP will play
        final_prediction = Counter(votes).most_common(1)[0][0]
        return _counter_move(final_prediction)


# ---------------------------------------------------------------------------
# 99: DoubleAgent (Mode Switcher)
# ---------------------------------------------------------------------------

class DoubleAgent(Algorithm):
    """Deceptive strategy that alternates between cooperative and aggressive modes.
    
    Switches its underlying personality every 10 rounds, alternating between
    mirroring the opponent and countering them. This temporal oscillation prevents
    opponents from locking onto a single behavioral model.
    
    **Type**: Deceptive / Specialist
    **Cycle**: 20 Round Period
    """
    name = "Double Agent"

    def choose(self, round_num, my_history, opp_history):
        if not opp_history: return self.rng.choice(MOVES)
        
        mode = (round_num // 10) % 2
        
        if mode == 0:
            # Cooperative/Mirror
            return opp_history[-1]
        else:
            # Aggressive/Counter
            return _counter_move(opp_history[-1])


# ---------------------------------------------------------------------------
# 100: CounterStrike (Anti-WSLS)
# ---------------------------------------------------------------------------

class CounterStrike(Algorithm):
    """Specialized exploiter targeting WSLS behavioral patterns.
    
    Operates on the assumption that the opponent follows Win-Stay Lose-Shift
    logic. Predicts whether the opponent will "Stay" or "Shift" based on the
    previous round's result and executes the optimal counter-strategy.
    
    **Type**: Specialists / Anti-Meta
    **Target**: WSLS Players
    """
    name = "Counter Strike"

    def choose(self, round_num, my_history, opp_history):
        if not my_history:
            return self.rng.choice(MOVES)
            
        my_last = my_history[-1]
        opp_last = opp_history[-1]
        
        if BEATS[my_last] == opp_last:
            # I Won. Opponent Lost.
            # WSLS Opponent will Shift -> to Counter(my_last).
            expected_opp_move = _counter_move(my_last)
            return _counter_move(expected_opp_move)
            
        elif BEATS[opp_last] == my_last:
            # I Lost. Opponent Won.
            # WSLS Opponent will Stay -> OppLast.
            expected_opp_move = opp_last
            return _counter_move(expected_opp_move)
            
        else:
            # Draw. WSLS behavior undefined (usually Random or Stay).
            # We assume Stay.
            return _counter_move(opp_last)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class TheHydra(Algorithm):
    """Multi-headed ensemble using a Hedge-weighted tournament engine.
    
    A high-performance meta-algorithm that manages 8 specialized competition
    experts. Uses multiplicative weight updates to dynamically adapt to the
    opponent's strategy within a 100-round match.
    
    **Type**: Meta-Strategy / Competition
    **Experts**: 8 Hedge-weighted experts
    """
    name = "The Hydra"

    def reset(self):
        from rps_playground.competition import CompetitionBot
        self._bot = CompetitionBot()
        self._move_to_int = {Move.ROCK: 0, Move.PAPER: 1, Move.SCISSORS: 2}
        self._int_to_move = {0: Move.ROCK, 1: Move.PAPER, 2: Move.SCISSORS}
        self._opponent_name = ""
        self._opponent_history = []

    def set_match_context(self, opponent_name, opponent_history):
        """Exploit opponent metadata in competition tournament mode."""
        self._opponent_name = opponent_name
        self._opponent_history = opponent_history

    def choose(self, round_num, my_history, opp_history):
        # Build history in competition format: list of [my_int, opp_int]
        history = []
        for my_m, opp_m in zip(my_history, opp_history):
            history.append([self._move_to_int[my_m], self._move_to_int[opp_m]])

        result = self._bot.make_move(
            history, self._opponent_name, self._opponent_history
        )
        return self._int_to_move[result]


# ===========================================================================
#  NEXT-GEN COMPETITION ALGORITHMS (73-76)
# ===========================================================================

import numpy as _np


# ---------------------------------------------------------------------------
# 73: Tournament Scout — deep opponent history analysis
# ---------------------------------------------------------------------------

class TournamentScout(Algorithm):
    """Intelligent archetype classifier for tournament play.
    
    Scans the opponent's full tournament history to identify their primary
    archetype (Static, Pattern, Random, etc.). Selects a dedicated counter-strategy
    and refines it with live in-match transition analysis.
    
    **Type**: Classifier / Strategy
    **Logic**: Archetype Detection
    """
    name = "Tournament Scout"

    def reset(self):
        self._opponent_name = ""
        self._opponent_history = []
        self._archetype = "UNKNOWN"
        # In-match tracking
        self._opp_freq = [0, 0, 0]
        self._opp_transitions = {m: [0.0, 0.0, 0.0] for m in range(3)}
        self._my_last = None
        self._opp_last = None
        self._my_wins = 0
        self._opp_wins = 0
        self._round_num = 0
        self._move_to_int = {Move.ROCK: 0, Move.PAPER: 1, Move.SCISSORS: 2}
        self._int_to_move = {0: Move.ROCK, 1: Move.PAPER, 2: Move.SCISSORS}

    def set_match_context(self, opponent_name, opponent_history):
        self._opponent_name = opponent_name
        self._opponent_history = opponent_history
        self._archetype = self._classify(opponent_name, opponent_history)

    def _classify(self, name, history):
        name_l = name.lower()
        if 'always' in name_l or 'constant' in name_l:
            return 'STATIC'
        if any(k in name_l for k in ['random', 'chaos', 'noise', 'uniform']):
            return 'RANDOM'
        if any(k in name_l for k in ['copy', 'mirror', 'tit']):
            return 'REACTIVE'
        if any(k in name_l for k in ['frequency', 'freq', 'counter']):
            return 'FREQUENCY'
        if any(k in name_l for k in ['meta', 'hydra', 'ensemble', 'mixture']):
            return 'META'

        if not history:
            return 'UNKNOWN'

        wins = sum(1 for m in history if m.get('result') == 'win')
        total = len(history)
        wr = wins / max(total, 1)

        # Analyze margin of victories
        margins = []
        for m in history:
            score = m.get('score', '0-0')
            parts = score.split('-')
            if len(parts) == 2:
                try:
                    w, l = int(parts[0]), int(parts[1])
                    margins.append(w - l)
                except ValueError:
                    pass

        # Big domination margins suggest a pattern exploiter
        big_wins = sum(1 for mg in margins if mg > 30)
        big_losses = sum(1 for mg in margins if mg < -30)

        if wr > 0.7 and big_wins > total * 0.3:
            return 'PATTERN_EXPLOITER'
        elif wr > 0.6:
            return 'ADAPTIVE'
        elif wr < 0.25:
            return 'WEAK'
        elif big_losses > total * 0.4:
            return 'PREDICTABLE'
        else:
            return 'UNKNOWN'

    def choose(self, round_num, my_history, opp_history):
        self._round_num = round_num

        # Update in-match tracking
        if opp_history:
            opp_int = self._move_to_int[opp_history[-1]]
            self._opp_freq[opp_int] += 1
            if self._opp_last is not None:
                self._opp_transitions[self._opp_last][opp_int] += 1
            self._opp_last = opp_int

        if my_history:
            my_int = self._move_to_int[my_history[-1]]
            self._my_last = my_int
            # Track wins
            if opp_history:
                opp_int = self._move_to_int[opp_history[-1]]
                if (my_int - opp_int) % 3 == 1:  # win
                    self._my_wins += 1
                elif (opp_int - my_int) % 3 == 1:  # loss
                    self._opp_wins += 1

        # Phase 1: Opening (first 5 rounds) — based on archetype
        if round_num < 5:
            return self._opening_move(round_num)

        # Phase 2: Main strategy — archetype-driven with live adaptation
        return self._main_strategy(round_num, opp_history)

    def _opening_move(self, round_num):
        if self._archetype == 'STATIC':
            # Try to figure out what they always play
            return [Move.PAPER, Move.SCISSORS, Move.ROCK, Move.PAPER, Move.SCISSORS][round_num]
        elif self._archetype == 'REACTIVE':
            return [Move.ROCK, Move.PAPER, Move.SCISSORS, Move.ROCK, Move.PAPER][round_num]
        elif self._archetype in ('RANDOM', 'META'):
            return self.rng.choice(MOVES)
        else:
            return [Move.PAPER, Move.ROCK, Move.SCISSORS, Move.PAPER, Move.ROCK][round_num]

    def _main_strategy(self, round_num, opp_history):
        # If we're losing badly, switch to pure Nash
        if self._opp_wins - self._my_wins > 12:
            return self.rng.choice(MOVES)

        # If opponent looks random (balanced frequencies), play Nash
        total_opp = sum(self._opp_freq)
        if total_opp > 15:
            expected = total_opp / 3
            chi2 = sum((f - expected) ** 2 / expected for f in self._opp_freq)
            if chi2 < 2.5:  # looks random
                return self.rng.choice(MOVES)

        # Use archetype-specific strategy
        if self._archetype in ('STATIC', 'WEAK', 'PREDICTABLE'):
            return self._exploit_frequency()
        elif self._archetype == 'FREQUENCY':
            return self._counter_frequency()
        elif self._archetype == 'REACTIVE':
            return self._counter_reactive()
        elif self._archetype in ('META', 'ADAPTIVE', 'PATTERN_EXPLOITER'):
            # Mix Nash with light exploitation
            if self.rng.random() < 0.4:
                return self.rng.choice(MOVES)
            return self._exploit_transitions()
        else:
            # Unknown: use transition prediction with some noise
            if self.rng.random() < 0.2:
                return self.rng.choice(MOVES)
            return self._exploit_transitions()

    def _exploit_frequency(self):
        most_common = self._opp_freq.index(max(self._opp_freq))
        return self._int_to_move[(most_common + 1) % 3]

    def _counter_frequency(self):
        # They'll counter our most played → play what beats their counter
        my_total = self._my_wins + self._opp_wins + (self._round_num - self._my_wins - self._opp_wins)
        if my_total > 0 and self._my_last is not None:
            their_prediction = max(range(3), key=lambda m: self._opp_freq[m])
            their_counter = (their_prediction + 1) % 3
            return self._int_to_move[(their_counter + 1) % 3]
        return self.rng.choice(MOVES)

    def _counter_reactive(self):
        # They copy our last move → play what beats our last move's counter
        if self._my_last is not None:
            return self._int_to_move[(self._my_last + 2) % 3]
        return self.rng.choice(MOVES)

    def _exploit_transitions(self):
        if self._opp_last is None:
            return self.rng.choice(MOVES)
        trans = self._opp_transitions[self._opp_last]
        total = sum(trans)
        if total < 2:
            return self.rng.choice(MOVES)
        predicted = trans.index(max(trans))
        return self._int_to_move[(predicted + 1) % 3]


# ---------------------------------------------------------------------------
# 74: Neural Prophet — numpy MLP with online learning
# ---------------------------------------------------------------------------

class NeuralProphet(Algorithm):
    """Multilayer Perceptron (MLP) for online move prediction.
    
    A 3-layer neural network built in pure numpy that learns to predict the
    opponent's next move in real-time. Features online training via SGD with
    momentum and adaptive learning rates based on opponent strength.
    
    **Type**: ML / Neural Network
    **Model**: MLP (27-32-16-3)
    """
    name = "Neural Prophet"

    def reset(self):
        # Xavier initialization for weights
        rng = _np.random.RandomState(42)
        self._W1 = rng.randn(27, 32) * _np.sqrt(2.0 / 27)
        self._b1 = _np.zeros(32)
        self._W2 = rng.randn(32, 16) * _np.sqrt(2.0 / 32)
        self._b2 = _np.zeros(16)
        self._W3 = rng.randn(16, 3) * _np.sqrt(2.0 / 16)
        self._b3 = _np.zeros(3)
        # Pre-train biases to predict uniform
        self._b3[:] = 0.0

        self._move_to_int = {Move.ROCK: 0, Move.PAPER: 1, Move.SCISSORS: 2}
        self._int_to_move = {0: Move.ROCK, 1: Move.PAPER, 2: Move.SCISSORS}
        self._lr = 0.05
        self._last_features = None
        self._opp_freq = _np.array([0.0, 0.0, 0.0])
        self._my_freq = _np.array([0.0, 0.0, 0.0])
        self._transitions = _np.ones((3, 3))  # Laplace smoothing
        self._wld = _np.array([0.0, 0.0, 0.0])  # win, loss, draw counts
        self._my_wins = 0
        self._opp_wins = 0
        self._opponent_name = ""
        self._opponent_history = []

    def set_match_context(self, opponent_name, opponent_history):
        self._opponent_name = opponent_name
        self._opponent_history = opponent_history
        # Pre-bias based on opponent strength
        if opponent_history:
            wins = sum(1 for m in opponent_history if m.get('result') == 'win')
            wr = wins / max(len(opponent_history), 1)
            if wr > 0.7:
                # Strong opponent — add noise to our predictions
                self._lr = 0.03
            elif wr < 0.3:
                # Weak — learn faster to exploit
                self._lr = 0.08

    def _build_features(self, my_history, opp_history, round_num):
        x = _np.zeros(27)
        if opp_history:
            opp_last = self._move_to_int[opp_history[-1]]
            x[opp_last] = 1.0
        if len(opp_history) >= 2:
            opp_2 = self._move_to_int[opp_history[-2]]
            x[3 + opp_2] = 1.0
        if my_history:
            my_last = self._move_to_int[my_history[-1]]
            x[6 + my_last] = 1.0
        if len(my_history) >= 2:
            my_2 = self._move_to_int[my_history[-2]]
            x[9 + my_2] = 1.0
        # Frequencies
        total_opp = max(self._opp_freq.sum(), 1)
        x[12:15] = self._opp_freq / total_opp
        total_my = max(self._my_freq.sum(), 1)
        x[15:18] = self._my_freq / total_my
        # Transition probabilities
        if opp_history:
            opp_last = self._move_to_int[opp_history[-1]]
            trans = self._transitions[opp_last]
            x[18:21] = trans / max(trans.sum(), 1)
        # WLD rate (last 20)
        total_wld = max(self._wld.sum(), 1)
        x[21:24] = self._wld / total_wld
        # Progress and margin
        x[24] = round_num / 100.0
        x[25] = (self._my_wins - self._opp_wins) / max(round_num, 1)
        x[26] = 1.0  # bias
        return x

    def _forward(self, x):
        """Forward pass. Returns (probs, cache)."""
        z1 = x @ self._W1 + self._b1
        a1 = _np.maximum(z1, 0)  # ReLU
        z2 = a1 @ self._W2 + self._b2
        a2 = _np.maximum(z2, 0)  # ReLU
        z3 = a2 @ self._W3 + self._b3
        # Softmax
        z3 -= z3.max()
        exp_z = _np.exp(z3)
        probs = exp_z / exp_z.sum()
        return probs, (x, z1, a1, z2, a2, z3)

    def _backprop(self, probs, target, cache):
        """Backpropagation with cross-entropy loss."""
        x, z1, a1, z2, a2, z3 = cache
        # dL/dz3 = probs - one_hot(target)
        dz3 = probs.copy()
        dz3[target] -= 1.0

        # Layer 3
        dW3 = _np.outer(a2, dz3)
        db3 = dz3
        da2 = dz3 @ self._W3.T

        # Layer 2 (ReLU)
        dz2 = da2 * (z2 > 0).astype(float)
        dW2 = _np.outer(a1, dz2)
        db2 = dz2
        da1 = dz2 @ self._W2.T

        # Layer 1 (ReLU)
        dz1 = da1 * (z1 > 0).astype(float)
        dW1 = _np.outer(x, dz1)
        db1 = dz1

        # SGD update
        lr = self._lr
        self._W3 -= lr * _np.clip(dW3, -1, 1)
        self._b3 -= lr * _np.clip(db3, -1, 1)
        self._W2 -= lr * _np.clip(dW2, -1, 1)
        self._b2 -= lr * _np.clip(db2, -1, 1)
        self._W1 -= lr * _np.clip(dW1, -1, 1)
        self._b1 -= lr * _np.clip(db1, -1, 1)

    def choose(self, round_num, my_history, opp_history):
        # Update tracking from last round
        if opp_history:
            opp_int = self._move_to_int[opp_history[-1]]
            self._opp_freq[opp_int] += 1
            if len(opp_history) >= 2:
                prev = self._move_to_int[opp_history[-2]]
                self._transitions[prev][opp_int] += 1
        if my_history:
            my_int = self._move_to_int[my_history[-1]]
            self._my_freq[my_int] += 1
            if opp_history:
                opp_int = self._move_to_int[opp_history[-1]]
                if (my_int - opp_int) % 3 == 1:
                    self._wld[0] += 1
                    self._my_wins += 1
                elif (opp_int - my_int) % 3 == 1:
                    self._wld[1] += 1
                    self._opp_wins += 1
                else:
                    self._wld[2] += 1

        # Online learning: train on last round's prediction
        if self._last_features is not None and opp_history:
            target = self._move_to_int[opp_history[-1]]
            probs, cache = self._forward(self._last_features)
            self._backprop(probs, target, cache)

        # First 3 rounds: play heuristically
        if round_num < 3:
            return self.rng.choice(MOVES)

        # If losing badly → Nash
        if round_num > 25 and self._opp_wins - self._my_wins > 10:
            return self.rng.choice(MOVES)

        # Forward pass to predict opponent's next move
        features = self._build_features(my_history, opp_history, round_num)
        self._last_features = features.copy()
        probs, _ = self._forward(features)

        # Counter the most likely opponent move
        predicted_opp = int(_np.argmax(probs))
        counter = (predicted_opp + 1) % 3

        # Add 10% noise for unpredictability
        if self.rng.random() < 0.10:
            return self.rng.choice(MOVES)
        return self._int_to_move[counter]


# ---------------------------------------------------------------------------
# 75: LSTM Predictor — recurrent sequence model in numpy
# ---------------------------------------------------------------------------

class LSTMPredictor(Algorithm):
    """Recurrent neural network using Long Short-Term Memory.
    
    Processes the sequence of game moves using a custom LSTM cell implemented in
    numpy. Capable of capturing long-range temporal dependencies and rhythmic
    patterns that traditional fixed-window matchers might overlook.
    
    **Type**: ML / RNN
    **Logic**: LSTM Cell
    """
    name = "LSTM Predictor"

    def reset(self):
        self._hidden_size = 16
        self._input_size = 6  # one-hot(my_move) + one-hot(opp_move)
        hs = self._hidden_size
        inp = self._input_size

        # LSTM gate weights (input, forget, cell, output)
        rng = _np.random.RandomState(123)
        scale = 0.1
        self._Wf = rng.randn(inp + hs, hs) * scale  # forget gate
        self._bf = _np.ones(hs)  # bias forget gate to 1 (remember by default)
        self._Wi = rng.randn(inp + hs, hs) * scale  # input gate
        self._bi = _np.zeros(hs)
        self._Wc = rng.randn(inp + hs, hs) * scale  # cell candidate
        self._bc = _np.zeros(hs)
        self._Wo = rng.randn(inp + hs, hs) * scale  # output gate
        self._bo = _np.zeros(hs)

        # Output layer: hidden → 3 (move prediction)
        self._Wy = rng.randn(hs, 3) * scale
        self._by = _np.zeros(3)

        # State
        self._h = _np.zeros(hs)
        self._c = _np.zeros(hs)

        self._move_to_int = {Move.ROCK: 0, Move.PAPER: 1, Move.SCISSORS: 2}
        self._int_to_move = {0: Move.ROCK, 1: Move.PAPER, 2: Move.SCISSORS}

        self._lr = 0.02
        self._my_wins = 0
        self._opp_wins = 0

        # Store last prediction for learning
        self._last_probs = None
        self._last_h = None

        self._opponent_name = ""
        self._opponent_history = []

    def set_match_context(self, opponent_name, opponent_history):
        self._opponent_name = opponent_name
        self._opponent_history = opponent_history

    def _sigmoid(self, x):
        x = _np.clip(x, -10, 10)
        return 1.0 / (1.0 + _np.exp(-x))

    def _tanh(self, x):
        return _np.tanh(_np.clip(x, -10, 10))

    def _lstm_step(self, x_t):
        """One LSTM step. Updates h, c. Returns new h."""
        concat = _np.concatenate([x_t, self._h])

        f = self._sigmoid(concat @ self._Wf + self._bf)
        i = self._sigmoid(concat @ self._Wi + self._bi)
        c_hat = self._tanh(concat @ self._Wc + self._bc)
        self._c = f * self._c + i * c_hat
        o = self._sigmoid(concat @ self._Wo + self._bo)
        self._h = o * self._tanh(self._c)

        return self._h

    def _predict(self, h):
        """Predict opponent's next move from hidden state."""
        logits = h @ self._Wy + self._by
        logits -= logits.max()
        exp_l = _np.exp(logits)
        return exp_l / exp_l.sum()

    def _online_update(self, target, probs, h_before):
        """Simple online gradient update (output layer only for speed)."""
        grad = probs.copy()
        grad[target] -= 1.0
        dWy = _np.outer(h_before, grad)
        dby = grad
        self._Wy -= self._lr * _np.clip(dWy, -0.5, 0.5)
        self._by -= self._lr * _np.clip(dby, -0.5, 0.5)

    def choose(self, round_num, my_history, opp_history):
        # Update tracking
        if my_history and opp_history:
            my_int = self._move_to_int[my_history[-1]]
            opp_int = self._move_to_int[opp_history[-1]]
            if (my_int - opp_int) % 3 == 1:
                self._my_wins += 1
            elif (opp_int - my_int) % 3 == 1:
                self._opp_wins += 1

        # Online learning from last round
        if self._last_probs is not None and opp_history:
            target = self._move_to_int[opp_history[-1]]
            self._online_update(target, self._last_probs, self._last_h)

        # Build input and run LSTM step
        if my_history and opp_history:
            x_t = _np.zeros(self._input_size)
            x_t[self._move_to_int[my_history[-1]]] = 1.0
            x_t[3 + self._move_to_int[opp_history[-1]]] = 1.0
            self._last_h = self._h.copy()
            self._lstm_step(x_t)

        # First 3 rounds: random
        if round_num < 3:
            self._last_probs = None
            return self.rng.choice(MOVES)

        # If losing badly → Nash
        if round_num > 25 and self._opp_wins - self._my_wins > 10:
            self._last_probs = None
            return self.rng.choice(MOVES)

        # Predict
        probs = self._predict(self._h)
        self._last_probs = probs.copy()

        predicted_opp = int(_np.argmax(probs))
        counter = (predicted_opp + 1) % 3

        # 10% exploration
        if self.rng.random() < 0.10:
            return self.rng.choice(MOVES)
        return self._int_to_move[counter]


# ---------------------------------------------------------------------------
# 76: Meta-Learner — adaptive ensemble with tournament-aware pre-selection
# ---------------------------------------------------------------------------

class MetaLearner(Algorithm):
    """Adaptive meta-ensemble with tournament-aware pre-selection.
    
    Mantains an ensemble of 6 specialized strategies (Frequency, Markov, Nash,
    etc.). Uses the Exp3 bandit algorithm to dynamically adjust weights based on
    real-time performance, with initial biases derived from tournament history.
    
    **Type**: Meta-Strategy / Bandit
    **Expert Count**: 6 Strategies
    """
    name = "Meta-Learner"

    def reset(self):
        self._n_strategies = 6
        self._weights = _np.ones(self._n_strategies) / self._n_strategies
        self._gamma = 0.15  # exploration rate
        self._eta = 0.3     # learning rate
        self._move_to_int = {Move.ROCK: 0, Move.PAPER: 1, Move.SCISSORS: 2}
        self._int_to_move = {0: Move.ROCK, 1: Move.PAPER, 2: Move.SCISSORS}
        self._opp_freq = _np.zeros(3)
        self._my_freq = _np.zeros(3)
        self._transitions = _np.ones((3, 3))  # Laplace
        self._last_strategy = 0
        self._last_move = None
        self._opp_last = None
        self._my_wins = 0
        self._opp_wins = 0
        self._opp_momentum = _np.zeros(3)
        self._opponent_name = ""
        self._opponent_history = []

    def set_match_context(self, opponent_name, opponent_history):
        self._opponent_name = opponent_name
        self._opponent_history = opponent_history
        self._pre_select_weights(opponent_name, opponent_history)

    def _pre_select_weights(self, name, history):
        """Bias strategy weights based on opponent profile."""
        name_l = name.lower()
        w = self._weights

        # Name-based biases
        if any(k in name_l for k in ['always', 'constant', 'static']):
            w[0] = 5.0  # frequency exploitation
            w[1] = 3.0  # transition
        elif any(k in name_l for k in ['random', 'chaos', 'noise']):
            w[3] = 5.0  # Nash
            w[0] = 0.3
        elif any(k in name_l for k in ['copy', 'mirror', 'tit']):
            w[2] = 4.0  # anti-frequency
            w[5] = 3.0  # WSLS
        elif any(k in name_l for k in ['frequency', 'freq', 'counter']):
            w[2] = 4.0  # anti-frequency
        elif any(k in name_l for k in ['meta', 'hydra', 'ensemble']):
            w[3] = 3.0  # Nash-heavy vs meta
            w[4] = 2.0  # gradient

        # Tournament record analysis
        if history:
            wins = sum(1 for m in history if m.get('result') == 'win')
            wr = wins / max(len(history), 1)
            if wr > 0.7:
                w[3] *= 2.0  # Nash vs strong
            elif wr < 0.3:
                w[0] *= 2.0  # exploit weak
                w[1] *= 2.0

        # Normalize
        total = w.sum()
        if total > 0:
            self._weights = w / total

    def _strategy_move(self, strategy_idx, my_history, opp_history, round_num):
        """Execute a specific strategy."""
        if strategy_idx == 0:
            # Frequency exploitation
            if sum(self._opp_freq) < 3:
                return self.rng.randint(0, 2)
            return (int(_np.argmax(self._opp_freq)) + 1) % 3

        elif strategy_idx == 1:
            # Transition prediction
            if self._opp_last is None:
                return self.rng.randint(0, 2)
            trans = self._transitions[self._opp_last]
            predicted = int(_np.argmax(trans))
            return (predicted + 1) % 3

        elif strategy_idx == 2:
            # Anti-frequency (counter what counters our most-played)
            if sum(self._my_freq) < 3:
                return self.rng.randint(0, 2)
            our_most = int(_np.argmax(self._my_freq))
            their_counter = (our_most + 1) % 3
            return (their_counter + 1) % 3

        elif strategy_idx == 3:
            # Nash
            return self.rng.randint(0, 2)

        elif strategy_idx == 4:
            # Gradient tracking (counter accelerating moves)
            if sum(self._opp_freq) < 5:
                return self.rng.randint(0, 2)
            return (int(_np.argmax(self._opp_momentum)) + 1) % 3

        elif strategy_idx == 5:
            # Win-stay-lose-shift
            if self._last_move is None:
                return self.rng.randint(0, 2)
            if my_history and opp_history:
                my_int = self._move_to_int[my_history[-1]]
                opp_int = self._move_to_int[opp_history[-1]]
                if (my_int - opp_int) % 3 == 1:  # we won
                    return my_int
                else:
                    return self.rng.randint(0, 2)
            return self.rng.randint(0, 2)

        return self.rng.randint(0, 2)

    def choose(self, round_num, my_history, opp_history):
        # Update tracking
        if opp_history:
            opp_int = self._move_to_int[opp_history[-1]]
            old_freq = self._opp_freq.copy()
            self._opp_freq[opp_int] += 1
            if self._opp_last is not None:
                self._transitions[self._opp_last][opp_int] += 1
            self._opp_last = opp_int
            # Momentum: how frequency is changing
            total = max(self._opp_freq.sum(), 1)
            new_pct = self._opp_freq / total
            old_total = max(old_freq.sum(), 1)
            old_pct = old_freq / old_total if old_total > 0 else _np.zeros(3)
            self._opp_momentum = 0.7 * self._opp_momentum + 0.3 * (new_pct - old_pct)

        if my_history:
            my_int = self._move_to_int[my_history[-1]]
            self._my_freq[my_int] += 1
            self._last_move = my_int
            if opp_history:
                opp_int = self._move_to_int[opp_history[-1]]
                if (my_int - opp_int) % 3 == 1:
                    self._my_wins += 1
                elif (opp_int - my_int) % 3 == 1:
                    self._opp_wins += 1

        # Reward last strategy
        if round_num > 0 and my_history and opp_history:
            my_int = self._move_to_int[my_history[-1]]
            opp_int = self._move_to_int[opp_history[-1]]
            if (my_int - opp_int) % 3 == 1:
                reward = 1.0
            elif (opp_int - my_int) % 3 == 1:
                reward = -0.5
            else:
                reward = 0.0

            # Exp3 update
            prob_used = (1 - self._gamma) * self._weights[self._last_strategy] + \
                        self._gamma / self._n_strategies
            estimated_reward = reward / max(prob_used, 0.01)
            self._weights[self._last_strategy] *= _np.exp(
                self._eta * estimated_reward / self._n_strategies
            )
            # Normalize
            total = self._weights.sum()
            if total > 0:
                self._weights /= total
            else:
                self._weights = _np.ones(self._n_strategies) / self._n_strategies

        # If losing badly → boost Nash
        if round_num > 25 and self._opp_wins - self._my_wins > 10:
            return self.rng.choice(MOVES)

        # Select strategy via Exp3 distribution
        probs = (1 - self._gamma) * self._weights + self._gamma / self._n_strategies
        strategy = self.rng.choices(range(self._n_strategies), weights=probs.tolist())[0]
        self._last_strategy = strategy

        move_int = self._strategy_move(strategy, my_history, opp_history, round_num)
        return self._int_to_move[move_int]



# ===========================================================================
#  ELITE COMPETITION ALGORITHMS (77-81)
#  Based on: Iocaine Powder, Greenberg, RPSContest winners, Kaggle top
#  solutions, dllu.net strategies, and PPO-inspired RL
# ===========================================================================


# ---------------------------------------------------------------------------
# 77: History Matcher — Iocaine Powder-style multi-length history matching
# ---------------------------------------------------------------------------

class HistoryMatcher(Algorithm):
    """Multi-scale history matching with 6 meta-strategies.
    
    An implementation inspired by the legendary Iocaine Powder bot. It searches
    opponent and player history for repeating patterns at multiple scales and
    evaluates 6 distinct meta-strategies to decide the best counter-move.
    
    **Type**: Pattern Matcher / Meta
    **Logic**: Multi-scale Correlation
    """
    name = "History Matcher"

    def reset(self):
        self._m2i = {Move.ROCK: 0, Move.PAPER: 1, Move.SCISSORS: 2}
        self._i2m = {0: Move.ROCK, 1: Move.PAPER, 2: Move.SCISSORS}
        self._opp_hist = []
        self._my_hist = []
        self._max_len = 20
        # 6 meta-strategies × 20 lengths = 120 predictors
        self._n_meta = 6
        self._scores = _np.zeros((self._n_meta, self._max_len))
        self._predictions = _np.zeros((self._n_meta, self._max_len), dtype=int)
        self._my_wins = 0
        self._opp_wins = 0

    def set_match_context(self, opponent_name, opponent_history):
        pass  # Pure in-match learning

    def _find_match(self, history, length):
        """Find the last occurrence of the most recent `length` elements."""
        if len(history) <= length:
            return -1
        pattern = history[-length:]
        # Search backwards
        for i in range(len(history) - length - 1, -1, -1):
            if history[i:i+length] == pattern:
                return i + length  # position after match
        return -1

    def _predict_opp(self, meta_idx, length):
        """Generate prediction for a specific metastrategy + length combo."""
        oh = self._opp_hist
        mh = self._my_hist

        if meta_idx == 0:
            # P0: Match opponent history
            pos = self._find_match(oh, length)
            if pos >= 0 and pos < len(oh):
                return oh[pos]
        elif meta_idx == 1:
            # P'0: What would beat opponent's P0 counter?
            pos = self._find_match(oh, length)
            if pos >= 0 and pos < len(oh):
                opp_pred = oh[pos]
                opp_counter = (opp_pred + 1) % 3
                return (opp_counter + 1) % 3  # beat their counter
        elif meta_idx == 2:
            # P1: Match MY history pattern, predict their response
            pos = self._find_match(mh, length)
            if pos >= 0 and pos < len(oh):
                return oh[pos]
        elif meta_idx == 3:
            # P'1: Counter P1
            pos = self._find_match(mh, length)
            if pos >= 0 and pos < len(oh):
                opp_pred = oh[pos]
                return (opp_pred + 2) % 3
        elif meta_idx == 4:
            # P2: Match interleaved history
            combined = []
            for m, o in zip(mh, oh):
                combined.append(m * 3 + o)
            pos = self._find_match(combined, length)
            if pos >= 0 and pos < len(oh):
                return oh[pos]
        elif meta_idx == 5:
            # P'2: Counter P2
            combined = []
            for m, o in zip(mh, oh):
                combined.append(m * 3 + o)
            pos = self._find_match(combined, length)
            if pos >= 0 and pos < len(oh):
                opp_pred = oh[pos]
                return (opp_pred + 2) % 3

        return -1  # no prediction

    def choose(self, round_num, my_history, opp_history):
        # Update internal history
        if my_history and opp_history:
            m = self._m2i[my_history[-1]]
            o = self._m2i[opp_history[-1]]
            self._my_hist.append(m)
            self._opp_hist.append(o)

            if (m - o) % 3 == 1:
                self._my_wins += 1
            elif (o - m) % 3 == 1:
                self._opp_wins += 1

            # Score all predictors based on what actually happened
            for mi in range(self._n_meta):
                for li in range(self._max_len):
                    pred = self._predictions[mi, li]
                    if pred >= 0:
                        if pred == o:
                            self._scores[mi, li] += 1.0
                        else:
                            self._scores[mi, li] -= 0.5
                        # Decay
                        self._scores[mi, li] *= 0.95

        # First 2 rounds: random
        if round_num < 2:
            return self.rng.choice(MOVES)

        # If losing badly → Nash
        if round_num > 20 and self._opp_wins - self._my_wins > 10:
            return self.rng.choice(MOVES)

        # Generate all predictions
        best_score = -999
        best_pred = -1
        for mi in range(self._n_meta):
            for li in range(min(len(self._opp_hist), self._max_len)):
                pred = self._predict_opp(mi, li + 1)
                self._predictions[mi, li] = pred
                if pred >= 0 and self._scores[mi, li] > best_score:
                    best_score = self._scores[mi, li]
                    best_pred = pred

        if best_pred >= 0 and best_score > 0:
            return self._i2m[(best_pred + 1) % 3]

        # Fallback: decayed frequency counter
        if len(self._opp_hist) > 3:
            freq = _np.zeros(3)
            decay = 0.9
            w = 1.0
            for o in reversed(self._opp_hist):
                freq[o] += w
                w *= decay
            return self._i2m[(int(_np.argmax(freq)) + 1) % 3]

        return self.rng.choice(MOVES)


# ---------------------------------------------------------------------------
# 78: Bayes Ensemble — Bayesian model averaging over 12 predictors
# ---------------------------------------------------------------------------

class BayesEnsemble(Algorithm):
    """Bayesian model averaging over 12 diverse predictors.
    
    Maintains a posterior probability distribution over a suite of 12 distinct
    predictors (Frequency, Markov, De Bruijn, etc.). Uses Bayes' rule to update
    weights based on predictive accuracy, selecting the move favored by the
    most credible models.
    
    **Type**: Bayesian / Ensemble
    **Experts**: 12 Probabilistic models
    """
    name = "Bayes Ensemble"

    def reset(self):
        self._n_pred = 12
        self._log_weights = _np.zeros(self._n_pred)  # log posteriors
        self._m2i = {Move.ROCK: 0, Move.PAPER: 1, Move.SCISSORS: 2}
        self._i2m = {0: Move.ROCK, 1: Move.PAPER, 2: Move.SCISSORS}
        self._opp_hist = []
        self._my_hist = []
        # Predictor state
        self._freq = [_np.ones(3) / 3 for _ in range(3)]  # 3 decay rates
        self._trans_opp = _np.ones((3, 3))  # P(next|opp_last)
        self._trans_my = _np.ones((3, 3))   # P(next|my_last)
        self._trans_pair = _np.ones((9, 3))  # P(next|pair)
        self._trans_opp2 = _np.ones((9, 3))  # P(next|opp_last_2)
        self._wld_trans = _np.ones((3, 3))   # P(next|outcome)
        self._streak_len = 0
        self._streak_move = -1
        self._my_wins = 0
        self._opp_wins = 0

    def set_match_context(self, opponent_name, opponent_history):
        # Use tournament data to bias priors
        if opponent_history:
            wins = sum(1 for m in opponent_history if m.get('result') == 'win')
            wr = wins / max(len(opponent_history), 1)
            if wr > 0.65:
                # Strong opponent — boost uniform predictor
                self._log_weights[0] += 2.0
            elif wr < 0.3:
                # Weak — boost exploitation predictors
                self._log_weights[1] += 1.0
                self._log_weights[4] += 1.0

    def _get_predictions(self, round_num):
        """Get probability distributions from all 12 predictors."""
        preds = _np.ones((self._n_pred, 3)) / 3  # default uniform

        # 0: Uniform
        # (already set)

        # 1-3: Decayed frequency
        for i, alpha in enumerate([0.5, 0.8, 0.95]):
            if self._opp_hist:
                f = _np.ones(3) * 0.01
                w = 1.0
                for o in reversed(self._opp_hist):
                    f[o] += w
                    w *= alpha
                preds[1 + i] = f / f.sum()

        # 4: Markov on opp last move
        if self._opp_hist:
            last = self._opp_hist[-1]
            row = self._trans_opp[last]
            preds[4] = row / row.sum()

        # 5: Markov on my last move
        if self._my_hist:
            last = self._my_hist[-1]
            row = self._trans_my[last]
            preds[5] = row / row.sum()

        # 6: Markov on pair
        if self._opp_hist and self._my_hist:
            pair = self._my_hist[-1] * 3 + self._opp_hist[-1]
            row = self._trans_pair[pair]
            preds[6] = row / row.sum()

        # 7: Order-2 Markov on opp
        if len(self._opp_hist) >= 2:
            state = self._opp_hist[-2] * 3 + self._opp_hist[-1]
            row = self._trans_opp2[state]
            preds[7] = row / row.sum()

        # 8: De Bruijn / cycle detection
        if len(self._opp_hist) >= 6:
            for period in [2, 3, 4]:
                if len(self._opp_hist) >= period * 2:
                    recent = self._opp_hist[-period:]
                    prev = self._opp_hist[-period*2:-period]
                    if recent == prev:
                        next_move = self._opp_hist[-period]
                        preds[8] = _np.array([0.05, 0.05, 0.05])
                        preds[8][next_move] = 0.9
                        break

        # 9: Streak continuation
        if self._streak_len >= 2 and self._streak_move >= 0:
            preds[9] = _np.array([0.15, 0.15, 0.15])
            preds[9][self._streak_move] = 0.7

        # 10: Anti-pattern (opponent models our frequency)
        if self._my_hist and len(self._my_hist) > 5:
            my_freq = _np.zeros(3)
            for m in self._my_hist:
                my_freq[m] += 1
            our_most = int(_np.argmax(my_freq))
            # Opponent counters our most → predict their counter
            their_counter = (our_most + 1) % 3
            preds[10] = _np.array([0.2, 0.2, 0.2])
            preds[10][their_counter] = 0.6

        # 11: WLD conditional
        if self._my_hist and self._opp_hist:
            m = self._my_hist[-1]
            o = self._opp_hist[-1]
            if (m - o) % 3 == 1:
                outcome = 0  # we won
            elif (o - m) % 3 == 1:
                outcome = 1  # we lost
            else:
                outcome = 2  # draw
            row = self._wld_trans[outcome]
            preds[11] = row / row.sum()

        return preds

    def choose(self, round_num, my_history, opp_history):
        # Update state
        if my_history and opp_history:
            m = self._m2i[my_history[-1]]
            o = self._m2i[opp_history[-1]]

            # Bayesian update: log_weight += log P(o | predictor)
            if round_num > 0:
                preds = self._get_predictions(round_num - 1)
                for pi in range(self._n_pred):
                    self._log_weights[pi] += _np.log(max(preds[pi][o], 1e-10))
                # Normalize to prevent overflow
                self._log_weights -= self._log_weights.max()

            # Update predictor state
            self._opp_hist.append(o)
            self._my_hist.append(m)

            if o >= 0:
                if len(self._opp_hist) >= 2:
                    prev_o = self._opp_hist[-2]
                    self._trans_opp[prev_o][o] += 1
                if len(self._my_hist) >= 2:
                    prev_m = self._my_hist[-2]
                    self._trans_my[prev_m][o] += 1
                if len(self._opp_hist) >= 2 and len(self._my_hist) >= 2:
                    pair = self._my_hist[-2] * 3 + self._opp_hist[-2]
                    self._trans_pair[pair][o] += 1
                if len(self._opp_hist) >= 3:
                    state = self._opp_hist[-3] * 3 + self._opp_hist[-2]
                    self._trans_opp2[state][o] += 1

                # Outcome
                if (m - o) % 3 == 1:
                    self._wld_trans[0][o] += 1
                    self._my_wins += 1
                elif (o - m) % 3 == 1:
                    self._wld_trans[1][o] += 1
                    self._opp_wins += 1
                else:
                    self._wld_trans[2][o] += 1

            # Streak
            if self._opp_hist and len(self._opp_hist) >= 2:
                if self._opp_hist[-1] == self._opp_hist[-2]:
                    self._streak_len += 1
                    self._streak_move = self._opp_hist[-1]
                else:
                    self._streak_len = 0
                    self._streak_move = self._opp_hist[-1]

        # First 2 rounds: random
        if round_num < 2:
            return self.rng.choice(MOVES)

        # If losing badly → Nash
        if round_num > 20 and self._opp_wins - self._my_wins > 10:
            return self.rng.choice(MOVES)

        # Get predictions and weight them
        preds = self._get_predictions(round_num)
        weights = _np.exp(self._log_weights - self._log_weights.max())
        weights /= weights.sum()

        # Bayesian model average
        combined = _np.zeros(3)
        for pi in range(self._n_pred):
            combined += weights[pi] * preds[pi]
        combined /= combined.sum()

        # Counter predicted move
        predicted = int(_np.argmax(combined))
        confidence = combined[predicted]

        # If low confidence, add noise
        if confidence < 0.4:
            if self.rng.random() < 0.3:
                return self.rng.choice(MOVES)

        return self._i2m[(predicted + 1) % 3]


# ---------------------------------------------------------------------------
# 79: Geometry Bot — anti-rotation metastrategy (dllu.net inspired)
# ---------------------------------------------------------------------------

class GeometryEnsemble(Algorithm):
    """Anti-rotation ensemble with Boltzmann weighted experts.
    
    A sophisticated meta-strategy that treats moves as rotations on a circle.
    Maintains 12 distinct experts tracking both own and opponent rotations, using
    Boltzmann (softmax) scoring with exponential decay for selection.
    
    **Type**: Meta-Strategy / Specialist
    **Logic**: Anti-Rotation Layer
    """
    name = "Geometry Ensemble"

    def reset(self):
        self._m2i = {Move.ROCK: 0, Move.PAPER: 1, Move.SCISSORS: 2}
        self._i2m = {0: Move.ROCK, 1: Move.PAPER, 2: Move.SCISSORS}
        self._opp_last = -1
        self._my_last = -1
        # 6 rotation predictors + 6 anti-rotation
        self._n_pred = 12
        self._scores = _np.zeros(self._n_pred)
        self._decay = 0.9
        self._temperature = 0.5
        self._my_wins = 0
        self._opp_wins = 0

    def set_match_context(self, opponent_name, opponent_history):
        pass

    def _get_predictions(self):
        """Get predicted opponent move from each strategy."""
        preds = _np.full(self._n_pred, -1)
        if self._opp_last >= 0:
            # R+0, R+1, R+2: they play rotation of their last
            for r in range(3):
                preds[r] = (self._opp_last + r) % 3
        if self._my_last >= 0:
            # A+0, A+1, A+2: they play rotation of our last
            for r in range(3):
                preds[3 + r] = (self._my_last + r) % 3
        # Anti-rotation: they counter our rotation predictors
        if self._my_last >= 0 and self._opp_last >= 0:
            for r in range(3):
                # They know we'd predict R+r, so they play counter
                pred_r = (self._opp_last + r) % 3
                our_counter = (pred_r + 1) % 3
                preds[6 + r] = (our_counter + 1) % 3  # beat our counter
            for r in range(3):
                pred_a = (self._my_last + r) % 3
                our_counter = (pred_a + 1) % 3
                preds[9 + r] = (our_counter + 1) % 3
        return preds

    def choose(self, round_num, my_history, opp_history):
        if my_history and opp_history:
            m = self._m2i[my_history[-1]]
            o = self._m2i[opp_history[-1]]

            if (m - o) % 3 == 1:
                self._my_wins += 1
            elif (o - m) % 3 == 1:
                self._opp_wins += 1

            # Score predictors
            preds = self._get_predictions()
            for i in range(self._n_pred):
                if preds[i] >= 0:
                    if preds[i] == o:
                        self._scores[i] += 1.0
                    elif (preds[i] + 1) % 3 == o:
                        self._scores[i] -= 0.7
                    self._scores[i] *= self._decay

            self._opp_last = o
            self._my_last = m

        if round_num < 2:
            if my_history:
                self._my_last = self._m2i[my_history[-1]]
            return self.rng.choice(MOVES)

        # If losing badly → Nash
        if round_num > 20 and self._opp_wins - self._my_wins > 10:
            return self.rng.choice(MOVES)

        # Boltzmann selection
        preds = self._get_predictions()
        valid = preds >= 0
        if not _np.any(valid):
            return self.rng.choice(MOVES)

        valid_scores = self._scores[valid]
        valid_preds = preds[valid]

        # Softmax
        temp = self._temperature
        exp_s = _np.exp((valid_scores - valid_scores.max()) / max(temp, 0.01))
        probs = exp_s / exp_s.sum()

        # Weighted prediction
        opp_probs = _np.zeros(3)
        for pr, prob in zip(valid_preds, probs):
            opp_probs[int(pr)] += prob

        predicted = int(_np.argmax(opp_probs))
        confidence = opp_probs[predicted]

        if confidence < 0.35:
            return self.rng.choice(MOVES)

        return self._i2m[(predicted + 1) % 3]


# ---------------------------------------------------------------------------
# 80: PPO Agent — pre-trained policy gradient (numpy)
# ---------------------------------------------------------------------------

class PhantomEnsemble(Algorithm):
    """Universal ensemble with 60+ integrated predictors.
    
    A massive ensemble combining heritage patterns, Markov chains, and temporal
    alignments. Tracks over 60 predictors simultaneously using a Hedge meta-learning
    algorithm for weight updates and selection.
    
    **Type**: Ensemble / Meta
    **Experts**: 60+ Tracked models
    """
    name = "Phantom Ensemble"

    def reset(self):
        self._m2i = {Move.ROCK: 0, Move.PAPER: 1, Move.SCISSORS: 2}
        self._i2m = {0: Move.ROCK, 1: Move.PAPER, 2: Move.SCISSORS}
        self._opp_hist = []
        self._my_hist = []
        self._my_wins = 0
        self._opp_wins = 0

        # === Predictor pools ===
        # Group A: History matching on opp history (depths 1-15)
        self._hm_depths = 15
        # Group B: History matching on my history (depths 1-15)
        # Group C: History matching on combined (depths 1-15)
        # = 45 history matchers
        self._n_hm = self._hm_depths * 3

        # Group D: Markov
        self._trans_opp1 = _np.ones((3, 3))
        self._trans_opp2 = _np.ones((9, 3))
        self._trans_opp3 = _np.ones((27, 3))
        self._trans_pair1 = _np.ones((9, 3))
        self._trans_pair2 = _np.ones((81, 3))
        self._trans_my1 = _np.ones((3, 3))
        self._n_markov = 6

        # Group E: Rotation (3 rotation + 3 anti-rotation)
        self._n_rotation = 6

        # Group F: Special (anti-freq, WLD, cycle)
        self._wld_trans = _np.ones((3, 3))
        self._n_special = 3

        self._n_total = self._n_hm + self._n_markov + self._n_rotation + self._n_special
        # = 45 + 6 + 6 + 3 = 60

        self._scores = _np.zeros(self._n_total)
        self._predictions = _np.full(self._n_total, -1, dtype=int)
        self._decay = 0.94
        self._opponent_name = ""
        self._opponent_history = []

    def set_match_context(self, opponent_name, opponent_history):
        self._opponent_name = opponent_name
        self._opponent_history = opponent_history
        if opponent_history:
            # Analyze tournament results to pre-bias predictors
            wins = sum(1 for r in opponent_history if r.get('result') == 'win')
            losses = sum(1 for r in opponent_history if r.get('result') == 'loss')
            wr = wins / max(len(opponent_history), 1)

            # Check scores for patterns
            opp_round_wins = 0
            opp_round_losses = 0
            for rec in opponent_history:
                score_str = rec.get('score', '')
                if '-' in score_str:
                    parts = score_str.split('-')
                    try:
                        opp_round_wins += int(parts[0])
                        opp_round_losses += int(parts[1])
                    except (ValueError, IndexError):
                        pass

            if wr > 0.7:
                # Very strong → boost Nash/rotation predictors, penalize simple
                base = self._n_hm + self._n_markov
                for i in range(self._n_rotation):
                    self._scores[base + i] += 2.0
            elif wr < 0.3:
                # Weak → boost exploitation (history matching + Markov)
                for i in range(min(10, self._n_hm)):
                    self._scores[i] += 1.5
                for i in range(self._n_markov):
                    self._scores[self._n_hm + i] += 1.5

            # Name-based heuristics
            name_lower = opponent_name.lower()
            if any(w in name_lower for w in ['always', 'constant', 'fixed']):
                # Boost frequency-based predictors
                for i in range(self._n_markov):
                    self._scores[self._n_hm + i] += 3.0
            elif any(w in name_lower for w in ['random', 'chaos', 'noise']):
                # Don't bother predicting — slight Nash bias
                pass
            elif any(w in name_lower for w in ['pattern', 'markov', 'sequence']):
                # Boost history matching
                for i in range(self._n_hm):
                    self._scores[i] += 1.0

    def _find_match(self, history, length):
        """Find last matching suffix of given length."""
        n = len(history)
        if n <= length:
            return -1
        pattern = history[-length:]
        for i in range(n - length - 1, -1, -1):
            if history[i:i+length] == pattern:
                pos = i + length
                if pos < n:
                    return history[pos]
        return -1

    def _generate_predictions(self):
        """Generate predictions from all 60 predictors."""
        oh = self._opp_hist
        mh = self._my_hist
        preds = self._predictions
        preds[:] = -1

        # Group A: History match on opp (indices 0..14)
        for d in range(1, self._hm_depths + 1):
            if len(oh) > d:
                preds[d - 1] = self._find_match(oh, d)

        # Group B: History match on my (indices 15..29)
        for d in range(1, self._hm_depths + 1):
            if len(mh) > d and len(oh) > d:
                match = self._find_match(mh, d)
                if match >= 0:
                    # Find what opponent did at that position
                    # Use the opp_hist at the same time step
                    for i in range(len(mh) - d - 1, -1, -1):
                        if mh[i:i+d] == mh[-d:]:
                            pos = i + d
                            if pos < len(oh):
                                preds[self._hm_depths + d - 1] = oh[pos]
                            break

        # Group C: Combined history match (indices 30..44)
        if oh and mh:
            combined = [m * 3 + o for m, o in zip(mh, oh)]
            for d in range(1, self._hm_depths + 1):
                if len(combined) > d:
                    match = self._find_match(combined, d)
                    if match >= 0:
                        # Need the opp move at that position
                        for i in range(len(combined) - d - 1, -1, -1):
                            if combined[i:i+d] == combined[-d:]:
                                pos = i + d
                                if pos < len(oh):
                                    preds[2 * self._hm_depths + d - 1] = oh[pos]
                                break

        base = self._n_hm  # 45

        # Group D: Markov chains (indices 45..50)
        if oh:
            # Order-1 opp
            row = self._trans_opp1[oh[-1]]
            preds[base] = int(_np.argmax(row))
            # Order-1 my conditioned
            row = self._trans_my1[mh[-1]] if mh else _np.ones(3)
            preds[base + 3] = int(_np.argmax(row))

        if len(oh) >= 2:
            state = oh[-2] * 3 + oh[-1]
            row = self._trans_opp2[state]
            preds[base + 1] = int(_np.argmax(row))
            # Pair
            if mh:
                pair = mh[-1] * 3 + oh[-1]
                row = self._trans_pair1[pair]
                preds[base + 4] = int(_np.argmax(row))

        if len(oh) >= 3:
            state = oh[-3] * 9 + oh[-2] * 3 + oh[-1]
            row = self._trans_opp3[state]
            preds[base + 2] = int(_np.argmax(row))
            if len(mh) >= 2:
                pair = (mh[-2] * 3 + oh[-2]) * 9 + (mh[-1] * 3 + oh[-1])
                if pair < 81:
                    row = self._trans_pair2[pair]
                    preds[base + 5] = int(_np.argmax(row))

        base += self._n_markov  # 51

        # Group E: Rotation (indices 51..56)
        if oh:
            for r in range(3):
                preds[base + r] = (oh[-1] + r) % 3
        if mh:
            for r in range(3):
                preds[base + 3 + r] = (mh[-1] + r) % 3

        base += self._n_rotation  # 57

        # Group F: Special (indices 57..59)
        # Anti-frequency
        if mh and len(mh) > 5:
            my_freq = _np.zeros(3)
            for m in mh:
                my_freq[m] += 1
            preds[base] = (int(_np.argmax(my_freq)) + 1) % 3

        # WLD conditional
        if mh and oh:
            m, o = mh[-1], oh[-1]
            if (m - o) % 3 == 1:
                outcome = 0
            elif (o - m) % 3 == 1:
                outcome = 1
            else:
                outcome = 2
            row = self._wld_trans[outcome]
            preds[base + 1] = int(_np.argmax(row))

        # Cycle detection
        if len(oh) >= 6:
            for period in [2, 3, 4, 5]:
                if len(oh) >= period * 2:
                    if oh[-period:] == oh[-period*2:-period]:
                        preds[base + 2] = oh[-period]
                        break

        return preds

    def choose(self, round_num, my_history, opp_history):
        if my_history and opp_history:
            m = self._m2i[my_history[-1]]
            o = self._m2i[opp_history[-1]]
            self._my_hist.append(m)
            self._opp_hist.append(o)

            if (m - o) % 3 == 1:
                self._my_wins += 1
            elif (o - m) % 3 == 1:
                self._opp_wins += 1

            # Score all predictors
            for i in range(self._n_total):
                pred = self._predictions[i]
                if pred >= 0:
                    if pred == o:
                        self._scores[i] += 1.0
                    elif (pred + 1) % 3 == o:
                        self._scores[i] -= 0.5
                    self._scores[i] *= self._decay

            # Update Markov tables
            if len(self._opp_hist) >= 2:
                prev = self._opp_hist[-2]
                self._trans_opp1[prev][o] += 1
                if self._my_hist:
                    self._trans_my1[self._my_hist[-2]][o] += 1
                    pair = self._my_hist[-2] * 3 + prev
                    self._trans_pair1[pair][o] += 1
            if len(self._opp_hist) >= 3:
                state = self._opp_hist[-3] * 3 + self._opp_hist[-2]
                self._trans_opp2[state][o] += 1
            if len(self._opp_hist) >= 4:
                state = self._opp_hist[-4] * 9 + self._opp_hist[-3] * 3 + self._opp_hist[-2]
                self._trans_opp3[state][o] += 1
                if len(self._my_hist) >= 3:
                    pair = (self._my_hist[-3] * 3 + self._opp_hist[-3]) * 9 + (self._my_hist[-2] * 3 + self._opp_hist[-2])
                    if pair < 81:
                        self._trans_pair2[pair][o] += 1

            # WLD
            if (m - o) % 3 == 1:
                self._wld_trans[0][o] += 1
            elif (o - m) % 3 == 1:
                self._wld_trans[1][o] += 1
            else:
                self._wld_trans[2][o] += 1

        # First round: random
        if round_num < 1:
            return self.rng.choice(MOVES)

        # Safety: if losing badly, go Nash
        if round_num > 20 and self._opp_wins - self._my_wins > 12:
            return self.rng.choice(MOVES)

        # Generate all predictions
        preds = self._generate_predictions()

        # Find best predictor(s)
        valid_mask = preds >= 0
        if not _np.any(valid_mask):
            return self.rng.choice(MOVES)

        valid_scores = self._scores[valid_mask]
        valid_preds = preds[valid_mask]

        # Softmax over scores (temperature = 0.3 for sharp selection)
        temp = 0.3
        centered = (valid_scores - valid_scores.max()) / max(temp, 0.01)
        centered = _np.clip(centered, -20, 0)
        weights = _np.exp(centered)
        weights /= weights.sum()

        # Weighted vote
        opp_probs = _np.zeros(3)
        for pred_val, w in zip(valid_preds, weights):
            opp_probs[int(pred_val)] += w

        predicted = int(_np.argmax(opp_probs))
        confidence = opp_probs[predicted]

        # Low confidence → add noise
        if confidence < 0.38:
            if self.rng.random() < 0.25:
                return self.rng.choice(MOVES)

        return self._i2m[(predicted + 1) % 3]


# ---------------------------------------------------------------------------
# 81: Decision Cascader — multi-depth prediction with cascading fallbacks
# ---------------------------------------------------------------------------

class DecisionCascader(Algorithm):
    """Multi-layer predictor with adaptive cascading fallbacks.
    
    Processes opponent behavior through a hierarchy of analysis levels (History,
    Markov, Balanced, etc.). Dynamically evaluates which depth level is currently
    winning and uses it to drive the strategy.
    
    **Type**: Meta-Strategy / Specialist
    **Logic**: Multi-depth Cascading
    """
    name = "Decision Cascader"

    def reset(self):
        self._m2i = {Move.ROCK: 0, Move.PAPER: 1, Move.SCISSORS: 2}
        self._i2m = {0: Move.ROCK, 1: Move.PAPER, 2: Move.SCISSORS}
        self._opp_hist = []
        self._my_hist = []
        self._n_levels = 5
        self._level_scores = _np.zeros(self._n_levels)
        self._level_preds = _np.full(self._n_levels, -1)
        self._trans1 = _np.ones((3, 3))  # order-1
        self._trans2 = _np.ones((9, 3))  # order-2
        self._trans3 = _np.ones((27, 3))  # order-3
        self._my_freq = _np.zeros(3)
        self._opp_freq = _np.zeros(3)
        self._my_wins = 0
        self._opp_wins = 0

    def set_match_context(self, opponent_name, opponent_history):
        if opponent_history:
            wins = sum(1 for m in opponent_history if m.get('result') == 'win')
            wr = wins / max(len(opponent_history), 1)
            if wr > 0.6:
                # Strong opponent → boost higher levels
                self._level_scores[3] += 2.0
                self._level_scores[4] += 3.0
            elif wr < 0.3:
                self._level_scores[0] += 2.0
                self._level_scores[1] += 2.0

    def _history_match(self, hist, max_search=15):
        """Find longest matching suffix in history."""
        for length in range(min(max_search, len(hist) - 1), 0, -1):
            pattern = hist[-length:]
            for i in range(len(hist) - length - 1, -1, -1):
                if hist[i:i+length] == pattern:
                    pos = i + length
                    if pos < len(hist):
                        return hist[pos]
        return -1

    def _level_predict(self, level):
        """Generate prediction for a given depth level."""
        oh = self._opp_hist
        mh = self._my_hist

        if level == 0:
            # Longest history match on opponent
            return self._history_match(oh)

        elif level == 1:
            # Markov chain (best order)
            best_confidence = 0
            best_pred = -1
            if oh:
                # Order 1
                row = self._trans1[oh[-1]]
                total = row.sum()
                if total > 3:
                    pred = int(_np.argmax(row))
                    conf = row[pred] / total
                    if conf > best_confidence:
                        best_confidence = conf
                        best_pred = pred

            if len(oh) >= 2:
                state = oh[-2] * 3 + oh[-1]
                row = self._trans2[state]
                total = row.sum()
                if total > 3:
                    pred = int(_np.argmax(row))
                    conf = row[pred] / total
                    if conf > best_confidence:
                        best_confidence = conf
                        best_pred = pred

            if len(oh) >= 3:
                state = oh[-3] * 9 + oh[-2] * 3 + oh[-1]
                row = self._trans3[state]
                total = row.sum()
                if total > 3:
                    pred = int(_np.argmax(row))
                    conf = row[pred] / total
                    if conf > best_confidence:
                        best_confidence = conf
                        best_pred = pred

            return best_pred

        elif level == 2:
            # Counter-strategy: opponent is counter-frequencing us
            if self._my_freq.sum() > 3:
                our_most = int(_np.argmax(self._my_freq))
                return (our_most + 1) % 3  # they counter our most

        elif level == 3:
            # Double-counter: opponent models us modeling them
            if self._opp_freq.sum() > 3:
                opp_most = int(_np.argmax(self._opp_freq))
                our_counter = (opp_most + 1) % 3
                # They know we counter, so they play what beats that
                their_counter = (our_counter + 1) % 3
                return their_counter

        elif level == 4:
            # Nash
            return self.rng.randint(0, 2)

        return -1

    def choose(self, round_num, my_history, opp_history):
        if my_history and opp_history:
            m = self._m2i[my_history[-1]]
            o = self._m2i[opp_history[-1]]
            self._opp_hist.append(o)
            self._my_hist.append(m)
            self._opp_freq[o] += 1
            self._my_freq[m] += 1

            if (m - o) % 3 == 1:
                self._my_wins += 1
            elif (o - m) % 3 == 1:
                self._opp_wins += 1

            # Update Markov tables
            if len(self._opp_hist) >= 2:
                self._trans1[self._opp_hist[-2]][o] += 1
            if len(self._opp_hist) >= 3:
                state = self._opp_hist[-3] * 3 + self._opp_hist[-2]
                self._trans2[state][o] += 1
            if len(self._opp_hist) >= 4:
                state = self._opp_hist[-4] * 9 + self._opp_hist[-3] * 3 + self._opp_hist[-2]
                self._trans3[state][o] += 1

            # Score levels
            for lv in range(self._n_levels):
                pred = self._level_preds[lv]
                if pred >= 0:
                    if pred == o:
                        self._level_scores[lv] += 1.0
                    else:
                        self._level_scores[lv] -= 0.3
                    self._level_scores[lv] *= 0.95

        if round_num < 2:
            return self.rng.choice(MOVES)

        # Generate predictions and pick best level
        for lv in range(self._n_levels):
            self._level_preds[lv] = self._level_predict(lv)

        best_level = int(_np.argmax(self._level_scores))
        pred = self._level_preds[best_level]

        if pred >= 0 and self._level_scores[best_level] > 0:
            return self._i2m[(pred + 1) % 3]

        # Cascade through levels
        for lv in range(self._n_levels):
            pred = self._level_preds[lv]
            if pred >= 0:
                return self._i2m[(pred + 1) % 3]

        return self.rng.choice(MOVES)


# ---------------------------------------------------------------------------
# 87: The Time Traveler — Echo State Network (Reservoir Computing)
# ---------------------------------------------------------------------------

class TheTimeTraveler(Algorithm):
    """Echo State Network (ESN) using Reservoir Computing.
    
    Projects game history into a high-dimensional non-linear state space (the
    reservoir) and trains a fast linear readout to predict the opponent's next
    move. Provides deep-learning style pattern recognition with minimal latency.
    
    **Type**: ML / Reservoir Computing
    **Model**: ESN (40-unit Reservoir)
    """
    name = "The Time Traveler"

    def reset(self):
        self._m2i = {Move.ROCK: 0, Move.PAPER: 1, Move.SCISSORS: 2}
        self._i2m = {0: Move.ROCK, 1: Move.PAPER, 2: Move.SCISSORS}
        self._oh = []
        self._mh = []

        # Reservoir params
        self._res_size = 40  # Small reservoir for speed
        self._spectral_radius = 0.9
        self._leak_rate = 0.3
        self._input_scaling = 0.5
        self._ridge_alpha = 0.1

        # Weights (fixed)
        # Use numpy for matrix generation, seeded from self.rng
        seed = self.rng.randint(0, 2**32 - 1)
        rs = _np.random.RandomState(seed)
        self._W_in = (rs.rand(self._res_size, 6) - 0.5) * 2 * self._input_scaling

        # Recurrent weights (sparse)
        W = rs.rand(self._res_size, self._res_size) - 0.5
        # Normalize spectral radius
        radius = max(abs(_np.linalg.eigvals(W)))
        self._W = W * (self._spectral_radius / radius)

        # State
        self._x = _np.zeros(self._res_size)

        # History buffers for training (sliding window)
        self._X_history = []
        self._Y_history = []
        self._window_size = 100

        # Readout weights (trained) - initialize randomly to break symmetry
        self._W_out = (rs.rand(3, self._res_size) - 0.5) * 0.1

    def _train_readout(self):
        """Update linear readout using Ridge Regression on recent history."""
        if len(self._X_history) < 10:
            return

        X = _np.array(self._X_history)  # (N, res_size)
        Y = _np.array(self._Y_history)  # (N, 3) - one-hot targets

        # Ridge Regression: W_out = Y.T * X * (X.T * X + alpha * I)^-1
        # Transpose for easier calculation: W_out matches shape (3, res_size)
        # Using pseudo-inverse or solve is better
        # For simplicity: W_out = (X.T X + alpha I)^-1 X.T Y

        N, D = X.shape
        # Add regularization
        XTX = X.T @ X + self._ridge_alpha * _np.eye(D)
        XTY = X.T @ Y

        try:
            # Use pseudo-inverse for better stability
            self._W_out = _np.linalg.pinv(XTX) @ XTY
            self._W_out = self._W_out.T
        except _np.linalg.LinAlgError:
            pass  # Keep old weights if singular

    def choose(self, round_num, my_history, opp_history):
        if my_history and opp_history:
            m = self._m2i[my_history[-1]]
            o_prev = self._m2i[opp_history[-1]]
            self._mh.append(m)
            self._oh.append(o_prev)

            # Input vector: one-hot my_move + one-hot opp_move
            u = _np.zeros(6)
            u[m] = 1
            u[3 + o_prev] = 1

            # Update reservoir state
            # x[t] = (1-a)*x[t-1] + a*tanh(W_in*u + W*x[t-1])
            pre_activation = self._W_in @ u + self._W @ self._x
            self._x = (1 - self._leak_rate) * self._x + self._leak_rate * _np.tanh(pre_activation)

            # Store for training (target is NEXT opponent move, but we don't know it yet)
            # Actually, standard ESN trains on x[t] -> y[t+1].
            # So at round t, we have x[t] and we observe o[t].
            # We want to predict o[t+1].
            # So we pair x[t-1] with target o[t].
            if round_num > 0:
                self._X_history.append(self._x_prev)
                target = _np.zeros(3)
                target[o_prev] = 1
                self._Y_history.append(target)

                if len(self._X_history) > self._window_size:
                    self._X_history.pop(0)
                    self._Y_history.pop(0)

                # Retrain every 10 rounds
                if round_num % 10 == 0:
                    self._train_readout()

            self._x_prev = self._x.copy()

        # Initial rounds
        if round_num < 15:
            self._x_prev = self._x.copy() # Needed for first training step
            return self.rng.choice(MOVES)

        # Predict next opponent move
        # y = W_out * x[t]
        y_pred = self._W_out @ self._x  # (3,)

        # Softmax-ish selection
        predicted = int(_np.argmax(y_pred))

        # Counter it
        return self._i2m[(predicted + 1) % 3]


# ---------------------------------------------------------------------------
# 88: The Collective — Boosting Ensemble
# ---------------------------------------------------------------------------

class TheCollective(Algorithm):
    """Boosting-inspired ensemble with adaptive instance weighting.
    
    An ensemble of weak learners (Markov, HM, Rotation, etc.) that focuses its
    learning on "hard" rounds where previous predictions failed. Dynamically
    boosts the influence of predictors that succeed where others stumble.
    
    **Type**: Ensemble / Meta
    **Technique**: Online Boosting
    """
    name = "The Collective"

    def reset(self):
        self._m2i = {Move.ROCK: 0, Move.PAPER: 1, Move.SCISSORS: 2}
        self._i2m = {0: Move.ROCK, 1: Move.PAPER, 2: Move.SCISSORS}
        self._oh = []
        self._mh = []

        # Weak learners
        # 1. Markov-1
        # 2. HM-depth-3
        # 3. Frequency (most frequent)
        # 4. Anti-Frequency
        # 5. Rotation
        # 6. Random (essential for breaking anti-prediction)
        self._n_predictors = 6
        self._predictor_weights = _np.ones(self._n_predictors)

        # Instance weights (one per round history)
        # We keep only last 50 rounds to be adaptive
        self._round_weights = []  # list of floats
        self._last_predictions = _np.zeros(self._n_predictors, dtype=int)

        # Predictor states
        self._trans1 = _np.ones((3, 3))

    def _get_weak_predictions(self):
        oh = self._oh
        mh = self._mh
        preds = _np.full(self._n_predictors, -1, dtype=int)

        # 1. Markov-1
        if oh:
            preds[0] = int(_np.argmax(self._trans1[oh[-1]]))

        # 2. HM-depth-3
        if len(oh) > 3:
            pattern = oh[-3:]
            for i in range(len(oh) - 4, -1, -1):
                if oh[i:i+3] == pattern:
                    if i+3 < len(oh):
                        preds[1] = oh[i+3]
                    break

        # 3. Frequency
        if oh:
            freq = _np.zeros(3)
            for m in oh: freq[m] += 1
            preds[2] = int(_np.argmax(freq))

        # 4. Anti-Frequency
        if len(mh) > 2:
            freq = _np.zeros(3)
            for m in mh: freq[m] += 1
            # Opponent counters my most frequent -> (argmax + 1)%3
            # I counter that -> +1 again
            preds[3] = (int(_np.argmax(freq)) + 2) % 3

        # 5. Rotation
        if oh:
            preds[4] = (oh[-1] + 1) % 3

        # 6. Random
        preds[5] = self.rng.randint(0, 2)

        return preds

    def choose(self, round_num, my_history, opp_history):
        if my_history and opp_history:
            m = self._m2i[my_history[-1]]
            o = self._m2i[opp_history[-1]]
            self._mh.append(m)
            self._oh.append(o)

            # Update Markov
            if len(self._oh) >= 2:
                self._trans1[self._oh[-2]][o] += 1

            # Update weights based on last round's predictions
            if self._last_predictions[0] != -1: # if we made predictions
                # Did the ensemble win efficiently?
                # Actually, "Boosting" updates predictor weights based on their error
                # weighted by instance difficulty.

                # Instance weight for this round
                # Start with 1.0. If we lost, increase it.
                w_t = 1.0
                if (m - o) % 3 != 1: # Lost or Draw
                    w_t = 2.0

                self._round_weights.append(w_t)
                if len(self._round_weights) > 50:
                    self._round_weights.pop(0)

                # Check which predictors were correct (predicted 'o')
                correct = (self._last_predictions == o)

                # Update predictor weights
                # Increase weight if correct, decrease if wrong
                # Multiplicative update
                for i in range(self._n_predictors):
                    if correct[i]:
                        self._predictor_weights[i] *= 1.1 * w_t
                    else:
                        self._predictor_weights[i] *= 0.9

                # Normalize weights to prevent explosion
                self._predictor_weights /= self._predictor_weights.sum()

        if round_num < 5:
            return self.rng.choice(MOVES)

        preds = self._get_weak_predictions()
        self._last_predictions = preds

        # Weighted Vote
        opp_probs = _np.zeros(3)
        for i in range(self._n_predictors):
            if preds[i] >= 0:
                opp_probs[preds[i]] += self._predictor_weights[i]

        predicted = int(_np.argmax(opp_probs))

        # Counter
        return self._i2m[(predicted + 1) % 3]


# ---------------------------------------------------------------------------
# 89: The Mirror World — Recursive Simulation
# ---------------------------------------------------------------------------

class TheMirrorWorld(Algorithm):
    """Recursive simulation engine using Level-K thinking.
    
    Models an opponent who is also modeling the player. Simulates multiple
    depths of recursive counter-strategies (Level 0 to Level 3) and dynamically
    selects the depth that best anticipates the opponent's current reasoning.
    
    **Type**: Psychological / Simulation
    **Logic**: Level-K Thinking
    """
    name = "The Mirror World"

    def reset(self):
        self._m2i = {Move.ROCK: 0, Move.PAPER: 1, Move.SCISSORS: 2}
        self._i2m = {0: Move.ROCK, 1: Move.PAPER, 2: Move.SCISSORS}
        self._oh = []
        self._mh = []
        self._last_moves = [0, 0, 0, 0]

        # Track score of each depth strategy
        self._depth_scores = _np.zeros(4) # Depths 0, 1, 2, 3
        self._decay = 0.9

        # Simple models for simulation
        self._my_trans = _np.ones((3, 3))
        self._opp_trans = _np.ones((3, 3))

    def _predict_move(self, hist, trans_table):
        """Predict next move from history using Markov."""
        if not hist: return self.rng.randint(0, 2)
        return int(_np.argmax(trans_table[hist[-1]]))

    def choose(self, round_num, my_history, opp_history):
        if my_history and opp_history:
            m = self._m2i[my_history[-1]]
            o = self._m2i[opp_history[-1]]
            self._mh.append(m)
            self._oh.append(o)

            # Update models
            if len(self._mh) >= 2:
                self._my_trans[self._mh[-2]][self._mh[-1]] += 1
            if len(self._oh) >= 2:
                self._opp_trans[self._oh[-2]][self._oh[-1]] += 1

            # Score depths
            # Re-calculate what each depth WOULD have played last round
            # to update scores
            # (Skipped for brevity/speed, only scoring based on current play is risky but ok)
            pass

        # Generate moves for each depth

        # Depth 0: Opponent is Random
        # Best response: Random (Nash)
        # Note: If opponent is truly random, any move is equal.
        # But we treat Depth 0 as "assume opponent plays random, so we play random"
        # Actually, "Level 0" usually means "I play random".
        d0_move = self.rng.randint(0, 2)

        # Depth 1: Level-1 Opponent
        # Opponent thinks I am Level-0 (Frequency/Markov biased).
        # Opponent predicts ME.
        pred_my_move = 0
        if self._mh:
             pred_my_move = self._predict_move(self._mh, self._my_trans)
        else:
             pred_my_move = self.rng.randint(0, 2)

        # Opponent plays counter to my predicted move
        opp_d1_move = (pred_my_move + 1) % 3

        # I play counter to that
        d1_move = (opp_d1_move + 1) % 3

        # Depth 2: Level-2 Opponent
        # Opponent thinks I am Level-1.
        # So Opponent expects me to play `d1_move`.
        # Opponent plays counter to `d1_move`.
        opp_d2_move = (d1_move + 1) % 3

        # I play counter to that
        d2_move = (opp_d2_move + 1) % 3

        # Depth 3: Level-3 Opponent
        # Opponent thinks I am Level-2.
        # Opponent expects me to play `d2_move`.
        opp_d3_move = (d2_move + 1) % 3

        # I play counter
        d3_move = (opp_d3_move + 1) % 3

        moves = [d0_move, d1_move, d2_move, d3_move]

        # Update scores (virtual play)
        if round_num > 0:
            actual_opp = self._oh[-1]
            # Scoring: did the depth's move win?
            for i in range(4):
                move = self._last_moves[i]
                if (move - actual_opp) % 3 == 1:
                    self._depth_scores[i] += 1.0
                elif (actual_opp - move) % 3 == 1:
                    self._depth_scores[i] -= 1.0
                self._depth_scores[i] *= self._decay

        self._last_moves = moves

        # Select best depth
        best_depth = int(_np.argmax(self._depth_scores))

        # Epsilon-greedy exploration
        if self.rng.random() < 0.1:
            best_depth = self.rng.randint(0, 3) # 0 to 3 inclusive

        return self._i2m[moves[best_depth]]


# ---------------------------------------------------------------------------
# 82: The Omniscient — ultimate competition algorithm
# ---------------------------------------------------------------------------

class TheOmniscient(Algorithm):
    """Apex ensemble with 79 high-fidelity predictors.
    
    An ultimate competition strategy featuring an expansive array of models
    including high-order Markov chains, deep history matching, and specialized
    outcome-conditional predictors. Synchronized via Hedge meta-learning.
    
    **Type**: Ensemble / Meta
    **Experts**: 79 Tracked models
    """
    name = "The Omniscient"

    def reset(self):
        self._m2i = {Move.ROCK: 0, Move.PAPER: 1, Move.SCISSORS: 2}
        self._i2m = {0: Move.ROCK, 1: Move.PAPER, 2: Move.SCISSORS}
        self._oh = []
        self._mh = []
        self._my_wins = 0
        self._opp_wins = 0

        # History matching: depth 1-20 * 3 = 60
        self._hm_depths = 20
        self._n_hm = self._hm_depths * 3

        # Markov transitions
        self._trans_opp1 = _np.ones((3, 3))
        self._trans_opp2 = _np.ones((9, 3))
        self._trans_opp3 = _np.ones((27, 3))
        self._trans_opp4 = _np.ones((81, 3))
        self._trans_pair1 = _np.ones((9, 3))
        self._trans_pair2 = _np.ones((81, 3))
        self._trans_my1 = _np.ones((3, 3))
        self._trans_my2 = _np.ones((9, 3))
        self._n_markov = 8

        self._n_rotation = 6

        self._wld_trans = _np.ones((3, 3))
        self._n_special = 5

        self._n_total = self._n_hm + self._n_markov + self._n_rotation + self._n_special

        self._scores = _np.zeros(self._n_total)
        self._predictions = _np.full(self._n_total, -1, dtype=int)
        self._decay = 0.93

        self._decay_freq = _np.zeros(3)
        self._freq_decay = 0.9

    def set_match_context(self, opponent_name, opponent_history):
        if not opponent_name and not opponent_history:
            return
        nl = (opponent_name or "").lower()

        if any(w in nl for w in ['always', 'constant', 'fixed', 'intentional']):
            for i in range(min(5, self._hm_depths)):
                self._scores[i] += 4.0
            base = self._n_hm
            for i in range(self._n_markov):
                self._scores[base + i] += 4.0
        elif any(w in nl for w in ['random', 'chaos', 'noise', 'pure']):
            base = self._n_hm + self._n_markov
            for i in range(self._n_rotation):
                self._scores[base + i] += 0.5
        elif any(w in nl for w in ['cycle', 'rotation', 'phase', 'bruijn']):
            for i in range(min(8, self._hm_depths)):
                self._scores[i] += 3.0
            base = self._n_hm + self._n_markov + self._n_rotation
            self._scores[base + 2] += 5.0
        elif any(w in nl for w in ['mirror', 'copy', 'tit', 'echo']):
            for i in range(self._hm_depths, 2 * self._hm_depths):
                self._scores[i] += 3.0
            base = self._n_hm
            self._scores[base + 4] += 4.0
            self._scores[base + 5] += 4.0
        elif any(w in nl for w in ['frequency', 'freq', 'decay', 'majority']):
            base = self._n_hm + self._n_markov + self._n_rotation
            self._scores[base] += 4.0
            self._scores[base + 4] += 4.0
        elif any(w in nl for w in ['pattern', 'markov', 'sequence', 'history',
                                   'historian', 'n-gram', 'gram']):
            for i in range(self._hm_depths):
                self._scores[i] += 2.0
            for i in range(2 * self._hm_depths, 3 * self._hm_depths):
                self._scores[i] += 2.0
        elif any(w in nl for w in ['meta', 'iocaine', 'hydra', 'ensemble',
                                   'phantom', 'prophet', 'neural', 'lstm',
                                   'bayes', 'geometry', 'scout', 'cascad',
                                   'omniscient']):
            base = self._n_hm + self._n_markov
            for i in range(self._n_rotation):
                self._scores[base + i] += 3.0
        elif any(w in nl for w in ['q-learner', 'thompson', 'ucb', 'gradient',
                                   'genetic', 'pid', 'hmm', 'regret']):
            for i in range(10, self._hm_depths):
                self._scores[i] += 2.0
            for i in range(2 * self._hm_depths, 3 * self._hm_depths):
                self._scores[i] += 1.5

        if opponent_history:
            wins = sum(1 for r in opponent_history if r.get('result') == 'win')
            total = max(len(opponent_history), 1)
            wr = wins / total
            if wr > 0.7:
                base = self._n_hm + self._n_markov
                for i in range(self._n_rotation):
                    self._scores[base + i] += 2.5
            elif wr < 0.25:
                for i in range(min(8, self._n_hm)):
                    self._scores[i] += 2.0
                base = self._n_hm
                for i in range(self._n_markov):
                    self._scores[base + i] += 2.0

    def _find_match(self, history, length):
        n = len(history)
        if n <= length:
            return -1
        pattern = history[-length:]
        for i in range(n - length - 1, -1, -1):
            if history[i:i + length] == pattern:
                pos = i + length
                if pos < n:
                    return history[pos]
        return -1

    def _generate_predictions(self):
        oh = self._oh
        mh = self._mh
        preds = self._predictions
        preds[:] = -1

        # Group A: HM on opp (0..19)
        for d in range(1, self._hm_depths + 1):
            if len(oh) > d:
                preds[d - 1] = self._find_match(oh, d)

        # Group B: HM on my → opp (20..39)
        for d in range(1, self._hm_depths + 1):
            if len(mh) > d and len(oh) > d:
                for i in range(len(mh) - d - 1, -1, -1):
                    if mh[i:i + d] == mh[-d:]:
                        pos = i + d
                        if pos < len(oh):
                            preds[self._hm_depths + d - 1] = oh[pos]
                        break

        # Group C: Combined HM (40..59)
        if oh and mh:
            combined = [m * 3 + o for m, o in zip(mh, oh)]
            for d in range(1, self._hm_depths + 1):
                if len(combined) > d:
                    for i in range(len(combined) - d - 1, -1, -1):
                        if combined[i:i + d] == combined[-d:]:
                            pos = i + d
                            if pos < len(oh):
                                preds[2 * self._hm_depths + d - 1] = oh[pos]
                            break

        base = self._n_hm

        # Group D: Markov (base..base+7)
        if oh:
            preds[base] = int(_np.argmax(self._trans_opp1[oh[-1]]))
            if mh:
                preds[base + 6] = int(_np.argmax(self._trans_my1[mh[-1]]))
        if len(oh) >= 2:
            preds[base + 1] = int(_np.argmax(self._trans_opp2[oh[-2] * 3 + oh[-1]]))
            if mh:
                pair = mh[-1] * 3 + oh[-1]
                preds[base + 4] = int(_np.argmax(self._trans_pair1[pair]))
            if len(mh) >= 2:
                preds[base + 7] = int(_np.argmax(self._trans_my2[mh[-2] * 3 + mh[-1]]))
        if len(oh) >= 3:
            st = oh[-3] * 9 + oh[-2] * 3 + oh[-1]
            if st < 27:
                preds[base + 2] = int(_np.argmax(self._trans_opp3[st]))
            if len(mh) >= 2:
                pair = (mh[-2] * 3 + oh[-2]) * 9 + (mh[-1] * 3 + oh[-1])
                if pair < 81:
                    preds[base + 5] = int(_np.argmax(self._trans_pair2[pair]))
        if len(oh) >= 4:
            state = oh[-4] * 27 + oh[-3] * 9 + oh[-2] * 3 + oh[-1]
            if state < 81:
                preds[base + 3] = int(_np.argmax(self._trans_opp4[state]))

        base += self._n_markov

        # Group E: Rotation
        if oh:
            for r in range(3):
                preds[base + r] = (oh[-1] + r) % 3
        if mh:
            for r in range(3):
                preds[base + 3 + r] = (mh[-1] + r) % 3

        base += self._n_rotation

        # Group F: Special
        if mh and len(mh) > 3:
            my_freq = _np.zeros(3)
            for m in mh:
                my_freq[m] += 1
            preds[base] = (int(_np.argmax(my_freq)) + 1) % 3

        if mh and oh:
            m_l, o_l = mh[-1], oh[-1]
            if (m_l - o_l) % 3 == 1:
                outcome = 0
            elif (o_l - m_l) % 3 == 1:
                outcome = 1
            else:
                outcome = 2
            preds[base + 1] = int(_np.argmax(self._wld_trans[outcome]))

        if len(oh) >= 6:
            for period in [2, 3, 4, 5]:
                if len(oh) >= period * 2:
                    if oh[-period:] == oh[-period * 2:-period]:
                        preds[base + 2] = oh[-period]
                        break

        if len(oh) >= 4:
            ctx = str(oh[-2]) + str(oh[-1])
            s = ''.join(str(x) for x in oh)
            best_c, best_n = 0, -1
            for nm in range(3):
                c = s.count(ctx + str(nm))
                if c > best_c:
                    best_c, best_n = c, nm
            if best_n >= 0:
                preds[base + 3] = best_n

        if oh:
            preds[base + 4] = (int(_np.argmax(self._decay_freq)) + 1) % 3

        return preds

    def choose(self, round_num, my_history, opp_history):
        if my_history and opp_history:
            m = self._m2i[my_history[-1]]
            o = self._m2i[opp_history[-1]]
            self._mh.append(m)
            self._oh.append(o)

            if (m - o) % 3 == 1:
                self._my_wins += 1
            elif (o - m) % 3 == 1:
                self._opp_wins += 1

            for i in range(self._n_total):
                pred = self._predictions[i]
                if pred >= 0:
                    if pred == o:
                        self._scores[i] += 1.0
                    elif (pred + 1) % 3 == o:
                        self._scores[i] -= 0.5
                    self._scores[i] *= self._decay

            # Update Markov
            if len(self._oh) >= 2:
                prev = self._oh[-2]
                self._trans_opp1[prev][o] += 1
                if len(self._mh) >= 2:
                    self._trans_my1[self._mh[-2]][o] += 1
                    pair = self._mh[-2] * 3 + prev
                    self._trans_pair1[pair][o] += 1
                    my_st = self._mh[-2] * 3 + self._mh[-1]
                    self._trans_my2[my_st][o] += 1
            if len(self._oh) >= 3:
                self._trans_opp2[self._oh[-3] * 3 + self._oh[-2]][o] += 1
            if len(self._oh) >= 4:
                st = self._oh[-4] * 9 + self._oh[-3] * 3 + self._oh[-2]
                if st < 27:
                    self._trans_opp3[st][o] += 1
                st4 = self._oh[-4] * 27 + self._oh[-3] * 9 + self._oh[-2] * 3 + self._oh[-1]
                if st4 < 81:
                    self._trans_opp4[st4 // 3][o] += 1
                if len(self._mh) >= 3:
                    pair = (self._mh[-3] * 3 + self._oh[-3]) * 9 + (self._mh[-2] * 3 + self._oh[-2])
                    if pair < 81:
                        self._trans_pair2[pair][o] += 1

            # WLD
            if (m - o) % 3 == 1:
                self._wld_trans[0][o] += 1
            elif (o - m) % 3 == 1:
                self._wld_trans[1][o] += 1
            else:
                self._wld_trans[2][o] += 1

            self._decay_freq *= self._freq_decay
            self._decay_freq[o] += 1.0

        if round_num < 1:
            return self.rng.choice(MOVES)

        # Safety: Nash fallback when losing badly
        if round_num > 15 and self._opp_wins - self._my_wins > 8:
            preds = self._generate_predictions()
            valid = preds >= 0
            if _np.any(valid):
                vs = self._scores[valid]
                best_idx = int(_np.argmax(vs))
                best_pred = preds[valid][best_idx]
                if vs[best_idx] > 0 and self.rng.random() > 0.4:
                    return self._i2m[(int(best_pred) + 1) % 3]
            return self.rng.choice(MOVES)

        preds = self._generate_predictions()
        valid_mask = preds >= 0
        if not _np.any(valid_mask):
            return self.rng.choice(MOVES)

        valid_scores = self._scores[valid_mask]
        valid_preds = preds[valid_mask]

        temp = 0.2
        centered = (valid_scores - valid_scores.max()) / max(temp, 0.01)
        centered = _np.clip(centered, -20, 0)
        weights = _np.exp(centered)
        weights /= weights.sum()

        opp_probs = _np.zeros(3)
        for pred_val, w in zip(valid_preds, weights):
            opp_probs[int(pred_val)] += w

        predicted = int(_np.argmax(opp_probs))
        confidence = opp_probs[predicted]

        if confidence < 0.36:
            if self.rng.random() < 0.15:
                return self.rng.choice(MOVES)



        return self._i2m[(predicted + 1) % 3]




# ---------------------------------------------------------------------------
# 83: The Doppelganger — Bayesian Online Change-Point Detection
# ---------------------------------------------------------------------------

class TheDoppelganger(Algorithm):
    """Change-point detector using Bayesian online inference.
    
    Identifies if and when an opponent switches their strategy mid-match. Uses
    Bayesian Online Change-Point Detection (BOCPD) to reset its models, ensuring
    it only learns from the most relevant recent behavior.
    
    **Type**: Bayesian / Adaptive
    **Logic**: BOCPD (Adams & MacKay)
    """
    name = "The Doppelganger"

    def reset(self):
        self._m2i = {Move.ROCK: 0, Move.PAPER: 1, Move.SCISSORS: 2}
        self._i2m = {0: Move.ROCK, 1: Move.PAPER, 2: Move.SCISSORS}
        self._oh = []
        self._mh = []

        # Change-point detection
        self._hazard = 1.0 / 25.0  # Expected run length ~ 25 rounds
        self._run_length_probs = _np.array([1.0])  # P(run_length = r)
        self._change_point = 0  # Index of last detected change point
        self._cp_threshold = 0.4  # Probability mass at r=0 to trigger CP

        # Markov models (reset after change point)
        self._trans1 = _np.ones((3, 3))
        self._trans2 = _np.ones((9, 3))
        self._pair1 = _np.ones((9, 3))

        # Full-history fallback
        self._full_trans1 = _np.ones((3, 3))

        # Predictor scoring (like Phantom)
        self._n_preds = 20
        self._scores = _np.zeros(self._n_preds)
        self._predictions = _np.full(self._n_preds, -1, dtype=int)
        self._decay = 0.94

    def _update_changepoint(self, obs):
        """Bayesian online change-point detection step."""
        n = len(self._run_length_probs)

        # Predictive probability under each run length
        pred_probs = _np.full(n, 1.0 / 3.0)  # Uniform prior for short runs

        # For runs > 3, use empirical frequency
        if n > 3:
            recent_start = max(0, len(self._oh) - n)
            recent = self._oh[recent_start:]
            if len(recent) > 3:
                freq = _np.zeros(3)
                for m in recent[-min(len(recent), 15):]:
                    freq[m] += 1
                freq += 0.5
                freq /= freq.sum()
                for r in range(3, n):
                    pred_probs[r] = freq[obs]

        # Growth probabilities
        growth = self._run_length_probs * pred_probs * (1 - self._hazard)

        # Change-point probability (all mass that "dies")
        cp_prob = (self._run_length_probs * pred_probs * self._hazard).sum()

        # New run length distribution
        new_probs = _np.zeros(n + 1)
        new_probs[0] = cp_prob
        new_probs[1:] = growth

        # Normalize
        total = new_probs.sum()
        if total > 0:
            new_probs /= total

        # Truncate to avoid unbounded growth (keep last 50)
        if len(new_probs) > 50:
            new_probs = new_probs[-50:]
            total = new_probs.sum()
            if total > 0:
                new_probs /= total

        self._run_length_probs = new_probs

        # Detect change point
        if new_probs[0] > self._cp_threshold:
            self._change_point = len(self._oh)
            self._trans1 = _np.ones((3, 3))
            self._trans2 = _np.ones((9, 3))
            self._pair1 = _np.ones((9, 3))

    def _find_match(self, history, length):
        n = len(history)
        if n <= length:
            return -1
        pattern = history[-length:]
        for i in range(n - length - 1, -1, -1):
            if history[i:i + length] == pattern:
                pos = i + length
                if pos < n:
                    return history[pos]
        return -1

    def _generate_predictions(self):
        preds = self._predictions
        preds[:] = -1

        # Use data since change point
        cp = self._change_point
        oh_recent = self._oh[cp:]
        mh_recent = self._mh[cp:]

        # History match on recent (depths 1-8)
        for d in range(1, 9):
            if len(oh_recent) > d:
                preds[d - 1] = self._find_match(oh_recent, d)

        # Markov on recent data
        if oh_recent:
            preds[8] = int(_np.argmax(self._trans1[oh_recent[-1]]))
        if len(oh_recent) >= 2:
            st = oh_recent[-2] * 3 + oh_recent[-1]
            preds[9] = int(_np.argmax(self._trans2[st]))
        if oh_recent and mh_recent:
            pair = mh_recent[-1] * 3 + oh_recent[-1]
            preds[10] = int(_np.argmax(self._pair1[pair]))

        # Full-history predictors (safety net)
        oh = self._oh
        mh = self._mh
        if oh:
            preds[11] = int(_np.argmax(self._full_trans1[oh[-1]]))
        for d in [1, 3, 5, 8]:
            if len(oh) > d:
                preds[11 + [1,3,5,8].index(d) + 1] = self._find_match(oh, d)

        # Rotation
        if oh:
            for r in range(3):
                preds[16 + r] = (oh[-1] + r) % 3

        # Anti-frequency
        if mh and len(mh) > 3:
            freq = _np.zeros(3)
            for m in mh:
                freq[m] += 1
            preds[19] = (int(_np.argmax(freq)) + 1) % 3

        return preds

    def choose(self, round_num, my_history, opp_history):
        if my_history and opp_history:
            m = self._m2i[my_history[-1]]
            o = self._m2i[opp_history[-1]]
            self._mh.append(m)
            self._oh.append(o)

            # Score predictors
            for i in range(self._n_preds):
                p = self._predictions[i]
                if p >= 0:
                    if p == o:
                        self._scores[i] += 1.0
                    elif (p + 1) % 3 == o:
                        self._scores[i] -= 0.5
                    self._scores[i] *= self._decay

            # Update Markov (recent window)
            cp = self._change_point
            oh_r = self._oh[cp:]
            if len(oh_r) >= 2:
                self._trans1[oh_r[-2]][o] += 1
                self._full_trans1[self._oh[-2]][o] += 1
            if len(oh_r) >= 3:
                self._trans2[oh_r[-3] * 3 + oh_r[-2]][o] += 1
            mh_r = self._mh[cp:]
            if len(mh_r) >= 2 and len(oh_r) >= 2:
                self._pair1[mh_r[-2] * 3 + oh_r[-2]][o] += 1

            # Change-point detection
            self._update_changepoint(o)

        if round_num < 1:
            return self.rng.choice(MOVES)

        preds = self._generate_predictions()
        valid = preds >= 0
        if not _np.any(valid):
            return self.rng.choice(MOVES)

        vs = self._scores[valid]
        vp = preds[valid]

        centered = (vs - vs.max()) / 0.3
        centered = _np.clip(centered, -20, 0)
        w = _np.exp(centered)
        w /= w.sum()

        opp_probs = _np.zeros(3)
        for pv, wt in zip(vp, w):
            opp_probs[int(pv)] += wt

        predicted = int(_np.argmax(opp_probs))
        confidence = opp_probs[predicted]

        if confidence < 0.38:
            if self.rng.random() < 0.2:
                return self.rng.choice(MOVES)

        return self._i2m[(predicted + 1) % 3]


# ---------------------------------------------------------------------------
# 84: The Void — Information-Theoretic Anti-Prediction
# ---------------------------------------------------------------------------

class TheVoid(Algorithm):
    """Information-theoretic anti-prediction engine.
    
    Monitors mutual information between the player's moves and the opponent's
    responses to determine if the player is being successfully modeled. Injects
    entropy when transparency is high and exploits predictability otherwise.
    
    **Type**: Information Theory / Adaptive
    **Logic**: Mutual Information Analysis
    """
    name = "The Void"

    def reset(self):
        self._m2i = {Move.ROCK: 0, Move.PAPER: 1, Move.SCISSORS: 2}
        self._i2m = {0: Move.ROCK, 1: Move.PAPER, 2: Move.SCISSORS}
        self._oh = []
        self._mh = []

        # MI tracking
        self._window = 20
        self._mi_threshold = 0.15  # bits
        self._being_read = False

        # Ensemble (same as Phantom-lite)
        self._hm_depths = 15
        self._n_hm = self._hm_depths * 3
        self._trans1 = _np.ones((3, 3))
        self._trans2 = _np.ones((9, 3))
        self._pair1 = _np.ones((9, 3))
        self._n_markov = 3
        self._n_rotation = 6
        self._wld_trans = _np.ones((3, 3))
        self._n_special = 3
        self._n_total = self._n_hm + self._n_markov + self._n_rotation + self._n_special
        self._scores = _np.zeros(self._n_total)
        self._predictions = _np.full(self._n_total, -1, dtype=int)
        self._decay = 0.94

    def _compute_mi(self):
        """Compute mutual information I(our_move; opp_next_move) in recent window."""
        if len(self._mh) < self._window + 1:
            return 0.0

        # Pairs: (our_move[t], opp_move[t+1])
        w = self._window
        my_recent = self._mh[-w - 1:-1]
        opp_next = self._oh[-w:]

        # Joint and marginal distributions
        joint = _np.zeros((3, 3))
        for m, o in zip(my_recent, opp_next):
            joint[m][o] += 1
        joint += 0.01  # smoothing
        joint /= joint.sum()

        p_my = joint.sum(axis=1)
        p_opp = joint.sum(axis=0)

        mi = 0.0
        for i in range(3):
            for j in range(3):
                if joint[i][j] > 0:
                    mi += joint[i][j] * _np.log2(joint[i][j] / (p_my[i] * p_opp[j]))
        return max(mi, 0.0)

    def _opp_entropy(self):
        """Compute opponent's recent move entropy."""
        if len(self._oh) < 10:
            return _np.log2(3)
        recent = self._oh[-20:]
        freq = _np.zeros(3)
        for m in recent:
            freq[m] += 1
        freq += 0.01
        freq /= freq.sum()
        return -sum(f * _np.log2(f) for f in freq if f > 0)

    def _find_match(self, history, length):
        n = len(history)
        if n <= length:
            return -1
        pattern = history[-length:]
        for i in range(n - length - 1, -1, -1):
            if history[i:i + length] == pattern:
                pos = i + length
                if pos < n:
                    return history[pos]
        return -1

    def _generate_predictions(self):
        oh = self._oh
        mh = self._mh
        preds = self._predictions
        preds[:] = -1

        for d in range(1, self._hm_depths + 1):
            if len(oh) > d:
                preds[d - 1] = self._find_match(oh, d)
        for d in range(1, self._hm_depths + 1):
            if len(mh) > d and len(oh) > d:
                for i in range(len(mh) - d - 1, -1, -1):
                    if mh[i:i + d] == mh[-d:]:
                        pos = i + d
                        if pos < len(oh):
                            preds[self._hm_depths + d - 1] = oh[pos]
                        break
        if oh and mh:
            combined = [m * 3 + o for m, o in zip(mh, oh)]
            for d in range(1, self._hm_depths + 1):
                if len(combined) > d:
                    for i in range(len(combined) - d - 1, -1, -1):
                        if combined[i:i + d] == combined[-d:]:
                            pos = i + d
                            if pos < len(oh):
                                preds[2 * self._hm_depths + d - 1] = oh[pos]
                            break

        base = self._n_hm
        if oh:
            preds[base] = int(_np.argmax(self._trans1[oh[-1]]))
        if len(oh) >= 2:
            preds[base + 1] = int(_np.argmax(self._trans2[oh[-2] * 3 + oh[-1]]))
        if oh and mh:
            preds[base + 2] = int(_np.argmax(self._pair1[mh[-1] * 3 + oh[-1]]))

        base += self._n_markov
        if oh:
            for r in range(3):
                preds[base + r] = (oh[-1] + r) % 3
        if mh:
            for r in range(3):
                preds[base + 3 + r] = (mh[-1] + r) % 3

        base += self._n_rotation
        if mh and len(mh) > 5:
            freq = _np.zeros(3)
            for m in mh:
                freq[m] += 1
            preds[base] = (int(_np.argmax(freq)) + 1) % 3
        if mh and oh:
            m_l, o_l = mh[-1], oh[-1]
            outcome = 0 if (m_l - o_l) % 3 == 1 else (1 if (o_l - m_l) % 3 == 1 else 2)
            preds[base + 1] = int(_np.argmax(self._wld_trans[outcome]))
        if len(oh) >= 6:
            for p in [2, 3, 4, 5]:
                if len(oh) >= p * 2 and oh[-p:] == oh[-p * 2:-p]:
                    preds[base + 2] = oh[-p]
                    break

        return preds

    def choose(self, round_num, my_history, opp_history):
        if my_history and opp_history:
            m = self._m2i[my_history[-1]]
            o = self._m2i[opp_history[-1]]
            self._mh.append(m)
            self._oh.append(o)

            for i in range(self._n_total):
                p = self._predictions[i]
                if p >= 0:
                    if p == o:
                        self._scores[i] += 1.0
                    elif (p + 1) % 3 == o:
                        self._scores[i] -= 0.5
                    self._scores[i] *= self._decay

            if len(self._oh) >= 2:
                self._trans1[self._oh[-2]][o] += 1
            if len(self._oh) >= 3:
                self._trans2[self._oh[-3] * 3 + self._oh[-2]][o] += 1
            if len(self._mh) >= 2 and len(self._oh) >= 2:
                self._pair1[self._mh[-2] * 3 + self._oh[-2]][o] += 1
            if (m - o) % 3 == 1:
                self._wld_trans[0][o] += 1
            elif (o - m) % 3 == 1:
                self._wld_trans[1][o] += 1
            else:
                self._wld_trans[2][o] += 1

        if round_num < 1:
            return self.rng.choice(MOVES)

        # Compute mutual information every 5 rounds
        if round_num % 5 == 0 and round_num >= self._window:
            mi = self._compute_mi()
            self._being_read = mi > self._mi_threshold

        # If being read → high entropy play
        if self._being_read:
            # Still try to predict, but inject more noise
            preds = self._generate_predictions()
            valid = preds >= 0
            if _np.any(valid):
                vs = self._scores[valid]
                best_idx = int(_np.argmax(vs))
                best_pred = preds[valid][best_idx]
                if vs[best_idx] > 2.0 and self.rng.random() > 0.5:
                    return self._i2m[(int(best_pred) + 1) % 3]
            return self.rng.choice(MOVES)

        # Normal exploitation mode
        preds = self._generate_predictions()
        valid = preds >= 0
        if not _np.any(valid):
            return self.rng.choice(MOVES)

        vs = self._scores[valid]
        vp = preds[valid]

        centered = (vs - vs.max()) / 0.3
        centered = _np.clip(centered, -20, 0)
        w = _np.exp(centered)
        w /= w.sum()

        opp_probs = _np.zeros(3)
        for pv, wt in zip(vp, w):
            opp_probs[int(pv)] += wt

        predicted = int(_np.argmax(opp_probs))
        confidence = opp_probs[predicted]
        if confidence < 0.38:
            if self.rng.random() < 0.25:
                return self.rng.choice(MOVES)

        return self._i2m[(predicted + 1) % 3]


# ---------------------------------------------------------------------------
# 85: The Architect — Strategy-Space Bandit
# ---------------------------------------------------------------------------

class TheArchitect(Algorithm):
    """Hierarchical bandit governing a library of complete strategies.
    
    Instead of individual moves, this meta-algorithm selects between 12 distinct
    high-level strategies (HM, Markov, WSLS, etc.) using Thompson Sampling. It
    commits to a strategy for the round and updates its beliefs based on success.
    
    **Type**: Bandit / Meta
    **Technique**: Thompson Sampling
    """
    name = "The Architect"

    def reset(self):
        self._m2i = {Move.ROCK: 0, Move.PAPER: 1, Move.SCISSORS: 2}
        self._i2m = {0: Move.ROCK, 1: Move.PAPER, 2: Move.SCISSORS}
        self._oh = []
        self._mh = []

        # 12 strategy arms
        self._n_strategies = 12
        self._alpha = _np.ones(self._n_strategies) * 2.0  # Prior successes
        self._beta = _np.ones(self._n_strategies) * 2.0   # Prior failures
        self._last_strategy = -1
        self._last_move = -1

        # Markov tables for strategies that need them
        self._trans1 = _np.ones((3, 3))
        self._trans2 = _np.ones((9, 3))

    def _strategy_move(self, strategy_id):
        """Execute a specific strategy and return the move (as int)."""
        oh = self._oh
        mh = self._mh

        if strategy_id == 0:
            # Counter opponent's last move
            if oh:
                return (oh[-1] + 1) % 3
            return self.rng.randint(0, 2)

        elif strategy_id == 1:
            # Counter opponent's most frequent
            if oh and len(oh) > 3:
                freq = _np.zeros(3)
                for m in oh:
                    freq[m] += 1
                return (int(_np.argmax(freq)) + 1) % 3
            return self.rng.randint(0, 2)

        elif strategy_id == 2:
            # Copy opponent (play same)
            if oh:
                return oh[-1]
            return self.rng.randint(0, 2)

        elif strategy_id == 3:
            # Markov-1 counter
            if oh:
                pred = int(_np.argmax(self._trans1[oh[-1]]))
                return (pred + 1) % 3
            return self.rng.randint(0, 2)

        elif strategy_id == 4:
            # Markov-2 counter
            if len(oh) >= 2:
                st = oh[-2] * 3 + oh[-1]
                pred = int(_np.argmax(self._trans2[st]))
                return (pred + 1) % 3
            return self.rng.randint(0, 2)

        elif strategy_id == 5:
            # De Bruijn counter (cycle through all 3)
            if oh:
                return (oh[-1] + 2) % 3
            return 0

        elif strategy_id == 6:
            # Win-stay Lose-shift
            if mh and oh:
                m, o = mh[-1], oh[-1]
                if (m - o) % 3 == 1:
                    return m  # Won → stay
                return (m + 1) % 3  # Lost/draw → shift
            return self.rng.randint(0, 2)

        elif strategy_id == 7:
            # Anti-frequency (counter their counter to our most-played)
            if mh and len(mh) > 3:
                freq = _np.zeros(3)
                for m in mh:
                    freq[m] += 1
                my_most = int(_np.argmax(freq))
                their_counter = (my_most + 1) % 3
                return (their_counter + 1) % 3
            return self.rng.randint(0, 2)

        elif strategy_id == 8:
            # Pattern match depth 3
            if len(oh) > 3:
                pattern = oh[-3:]
                for i in range(len(oh) - 4, -1, -1):
                    if oh[i:i + 3] == pattern:
                        pos = i + 3
                        if pos < len(oh):
                            return (oh[pos] + 1) % 3
            return self.rng.randint(0, 2)

        elif strategy_id == 9:
            # Pattern match depth 5
            if len(oh) > 5:
                pattern = oh[-5:]
                for i in range(len(oh) - 6, -1, -1):
                    if oh[i:i + 5] == pattern:
                        pos = i + 5
                        if pos < len(oh):
                            return (oh[pos] + 1) % 3
            return self.rng.randint(0, 2)

        elif strategy_id == 10:
            # Nash (uniform random)
            return self.rng.randint(0, 2)

        elif strategy_id == 11:
            # Anti-last-own: counter what beats our last move
            if mh:
                what_beats_me = (mh[-1] + 1) % 3
                return (what_beats_me + 1) % 3
            return self.rng.randint(0, 2)

        return self.rng.integers(3)

    def choose(self, round_num, my_history, opp_history):
        if my_history and opp_history:
            m = self._m2i[my_history[-1]]
            o = self._m2i[opp_history[-1]]
            self._mh.append(m)
            self._oh.append(o)

            # Update Markov tables
            if len(self._oh) >= 2:
                self._trans1[self._oh[-2]][o] += 1
            if len(self._oh) >= 3:
                self._trans2[self._oh[-3] * 3 + self._oh[-2]][o] += 1

            # Reward/punish last strategy
            if self._last_strategy >= 0:
                if (m - o) % 3 == 1:
                    self._alpha[self._last_strategy] += 1.0
                elif (o - m) % 3 == 1:
                    self._beta[self._last_strategy] += 1.0
                else:
                    self._alpha[self._last_strategy] += 0.3
                    self._beta[self._last_strategy] += 0.3

        # Thompson Sampling: sample from each Beta distribution
        samples = _np.array([
            self.rng.betavariate(self._alpha[i], self._beta[i])
            for i in range(self._n_strategies)
        ])

        # Select best strategy
        best = int(_np.argmax(samples))
        self._last_strategy = best

        move = self._strategy_move(best)
        self._last_move = move
        return self._i2m[int(move)]


# ---------------------------------------------------------------------------
# 86: Super Omniscient — Fixed + Enhanced Omniscient
# ---------------------------------------------------------------------------

class SuperOmniscient(Algorithm):
    """Refined apex ensemble with regime-shift awareness.
    
    An enhanced version of The Omniscient that incorporates change-point
    detection and CTW context modeling. Calibrated with more conservative meta-
    biasing and sharper selection temperatures for peak competitive performance.
    
    **Type**: Ensemble / Meta
    **Logic**: Dynamic Regime Adaptation
    """
    name = "Super Omniscient"

    def reset(self):
        self._m2i = {Move.ROCK: 0, Move.PAPER: 1, Move.SCISSORS: 2}
        self._i2m = {0: Move.ROCK, 1: Move.PAPER, 2: Move.SCISSORS}
        self._oh = []
        self._mh = []
        self._my_wins = 0
        self._opp_wins = 0

        # History match: depth 1-18 × 3 = 54
        self._hm_depths = 18
        self._n_hm = self._hm_depths * 3

        # Markov
        self._trans_opp1 = _np.ones((3, 3))
        self._trans_opp2 = _np.ones((9, 3))
        self._trans_opp3 = _np.ones((27, 3))
        self._trans_pair1 = _np.ones((9, 3))
        self._trans_pair2 = _np.ones((81, 3))
        self._trans_my1 = _np.ones((3, 3))
        self._n_markov = 6

        self._n_rotation = 6

        # Special: anti-freq, WLD, cycle, LZ-context, decay-freq,
        #          CTW-context, change-point-decay-freq
        self._wld_trans = _np.ones((3, 3))
        self._n_special = 7

        self._n_total = self._n_hm + self._n_markov + self._n_rotation + self._n_special
        # = 54 + 6 + 6 + 7 = 73

        self._scores = _np.zeros(self._n_total)
        self._predictions = _np.full(self._n_total, -1, dtype=int)
        self._decay = 0.94  # Same as Phantom

        self._decay_freq = _np.zeros(3)
        self._freq_decay = 0.9

        # Change-point detection (simplified)
        self._recent_wins = 0
        self._recent_losses = 0
        self._regime_shift_count = 0

        # CTW context tables (depth 2 and 3)
        self._ctw2 = {}  # {(oh[-2], oh[-1]): [counts for 0,1,2]}
        self._ctw3 = {}

    def set_match_context(self, opponent_name, opponent_history):
        if not opponent_name and not opponent_history:
            return
        nl = (opponent_name or "").lower()

        # Conservative biasing (+1.5 max, not +4)
        if any(w in nl for w in ['always', 'constant', 'fixed', 'intentional']):
            for i in range(min(5, self._hm_depths)):
                self._scores[i] += 1.5
            base = self._n_hm
            for i in range(self._n_markov):
                self._scores[base + i] += 1.5
        elif any(w in nl for w in ['random', 'chaos', 'noise', 'pure']):
            pass  # No biasing for random
        elif any(w in nl for w in ['cycle', 'rotation', 'phase', 'bruijn']):
            for i in range(min(6, self._hm_depths)):
                self._scores[i] += 1.0
        elif any(w in nl for w in ['mirror', 'copy', 'tit', 'echo']):
            for i in range(self._hm_depths, 2 * self._hm_depths):
                self._scores[i] += 1.0
        elif any(w in nl for w in ['pattern', 'markov', 'history', 'historian',
                                   'n-gram', 'sequence']):
            for i in range(self._hm_depths):
                self._scores[i] += 1.0
        elif any(w in nl for w in ['meta', 'iocaine', 'hydra', 'ensemble',
                                   'phantom', 'omniscient', 'cascad']):
            base = self._n_hm + self._n_markov
            for i in range(self._n_rotation):
                self._scores[base + i] += 1.5

        if opponent_history:
            wins = sum(1 for r in opponent_history if r.get('result') == 'win')
            total = max(len(opponent_history), 1)
            wr = wins / total
            if wr > 0.7:
                base = self._n_hm + self._n_markov
                for i in range(self._n_rotation):
                    self._scores[base + i] += 1.0
            elif wr < 0.25:
                for i in range(min(6, self._n_hm)):
                    self._scores[i] += 1.0

    def _find_match(self, history, length):
        n = len(history)
        if n <= length:
            return -1
        pattern = history[-length:]
        for i in range(n - length - 1, -1, -1):
            if history[i:i + length] == pattern:
                pos = i + length
                if pos < n:
                    return history[pos]
        return -1

    def _generate_predictions(self):
        oh = self._oh
        mh = self._mh
        preds = self._predictions
        preds[:] = -1

        for d in range(1, self._hm_depths + 1):
            if len(oh) > d:
                preds[d - 1] = self._find_match(oh, d)
        for d in range(1, self._hm_depths + 1):
            if len(mh) > d and len(oh) > d:
                for i in range(len(mh) - d - 1, -1, -1):
                    if mh[i:i + d] == mh[-d:]:
                        pos = i + d
                        if pos < len(oh):
                            preds[self._hm_depths + d - 1] = oh[pos]
                        break
        if oh and mh:
            combined = [m * 3 + o for m, o in zip(mh, oh)]
            for d in range(1, self._hm_depths + 1):
                if len(combined) > d:
                    for i in range(len(combined) - d - 1, -1, -1):
                        if combined[i:i + d] == combined[-d:]:
                            pos = i + d
                            if pos < len(oh):
                                preds[2 * self._hm_depths + d - 1] = oh[pos]
                            break

        base = self._n_hm
        if oh:
            preds[base] = int(_np.argmax(self._trans_opp1[oh[-1]]))
            if mh:
                preds[base + 5] = int(_np.argmax(self._trans_my1[mh[-1]]))
        if len(oh) >= 2:
            preds[base + 1] = int(_np.argmax(self._trans_opp2[oh[-2] * 3 + oh[-1]]))
            if mh:
                preds[base + 3] = int(_np.argmax(self._trans_pair1[mh[-1] * 3 + oh[-1]]))
        if len(oh) >= 3:
            st = oh[-3] * 9 + oh[-2] * 3 + oh[-1]
            if st < 27:
                preds[base + 2] = int(_np.argmax(self._trans_opp3[st]))
            if len(mh) >= 2:
                pair = (mh[-2] * 3 + oh[-2]) * 9 + (mh[-1] * 3 + oh[-1])
                if pair < 81:
                    preds[base + 4] = int(_np.argmax(self._trans_pair2[pair]))

        base += self._n_markov
        if oh:
            for r in range(3):
                preds[base + r] = (oh[-1] + r) % 3
        if mh:
            for r in range(3):
                preds[base + 3 + r] = (mh[-1] + r) % 3

        base += self._n_rotation

        # Anti-freq
        if mh and len(mh) > 3:
            freq = _np.zeros(3)
            for m in mh:
                freq[m] += 1
            preds[base] = (int(_np.argmax(freq)) + 1) % 3

        # WLD
        if mh and oh:
            m_l, o_l = mh[-1], oh[-1]
            outcome = 0 if (m_l - o_l) % 3 == 1 else (1 if (o_l - m_l) % 3 == 1 else 2)
            preds[base + 1] = int(_np.argmax(self._wld_trans[outcome]))

        # Cycle
        if len(oh) >= 6:
            for p in [2, 3, 4, 5]:
                if len(oh) >= p * 2 and oh[-p:] == oh[-p * 2:-p]:
                    preds[base + 2] = oh[-p]
                    break

        # LZ-context bigram
        if len(oh) >= 4:
            ctx = str(oh[-2]) + str(oh[-1])
            s = ''.join(str(x) for x in oh)
            bc, bn = 0, -1
            for nm in range(3):
                c = s.count(ctx + str(nm))
                if c > bc:
                    bc, bn = c, nm
            if bn >= 0:
                preds[base + 3] = bn

        # Decay-freq
        if oh:
            preds[base + 4] = (int(_np.argmax(self._decay_freq)) + 1) % 3

        # CTW depth-2 context
        if len(oh) >= 2:
            ctx2 = (oh[-2], oh[-1])
            if ctx2 in self._ctw2:
                counts = self._ctw2[ctx2]
                preds[base + 5] = int(_np.argmax(counts))

        # CTW depth-3 context
        if len(oh) >= 3:
            ctx3 = (oh[-3], oh[-2], oh[-1])
            if ctx3 in self._ctw3:
                counts = self._ctw3[ctx3]
                preds[base + 6] = int(_np.argmax(counts))

        return preds

    def choose(self, round_num, my_history, opp_history):
        if my_history and opp_history:
            m = self._m2i[my_history[-1]]
            o = self._m2i[opp_history[-1]]
            self._mh.append(m)
            self._oh.append(o)

            if (m - o) % 3 == 1:
                self._my_wins += 1
                self._recent_wins += 1
            elif (o - m) % 3 == 1:
                self._opp_wins += 1
                self._recent_losses += 1

            for i in range(self._n_total):
                pred = self._predictions[i]
                if pred >= 0:
                    if pred == o:
                        self._scores[i] += 1.0
                    elif (pred + 1) % 3 == o:
                        self._scores[i] -= 0.5
                    self._scores[i] *= self._decay

            # Update Markov
            if len(self._oh) >= 2:
                prev = self._oh[-2]
                self._trans_opp1[prev][o] += 1
                if len(self._mh) >= 2:
                    self._trans_my1[self._mh[-2]][o] += 1
                    pair = self._mh[-2] * 3 + prev
                    self._trans_pair1[pair][o] += 1
            if len(self._oh) >= 3:
                self._trans_opp2[self._oh[-3] * 3 + self._oh[-2]][o] += 1
            if len(self._oh) >= 4:
                st = self._oh[-4] * 9 + self._oh[-3] * 3 + self._oh[-2]
                if st < 27:
                    self._trans_opp3[st][o] += 1
                if len(self._mh) >= 3:
                    pair = (self._mh[-3] * 3 + self._oh[-3]) * 9 + (self._mh[-2] * 3 + self._oh[-2])
                    if pair < 81:
                        self._trans_pair2[pair][o] += 1
            # WLD
            if (m - o) % 3 == 1:
                self._wld_trans[0][o] += 1
            elif (o - m) % 3 == 1:
                self._wld_trans[1][o] += 1
            else:
                self._wld_trans[2][o] += 1

            self._decay_freq *= self._freq_decay
            self._decay_freq[o] += 1.0

            # CTW update
            if len(self._oh) >= 3:
                ctx2 = (self._oh[-3], self._oh[-2])
                if ctx2 not in self._ctw2:
                    self._ctw2[ctx2] = _np.ones(3)
                self._ctw2[ctx2][o] += 1
            if len(self._oh) >= 4:
                ctx3 = (self._oh[-4], self._oh[-3], self._oh[-2])
                if ctx3 not in self._ctw3:
                    self._ctw3[ctx3] = _np.ones(3)
                self._ctw3[ctx3][o] += 1

        # Change-point detection every 12 rounds
        if round_num > 0 and round_num % 12 == 0:
            if self._recent_losses > self._recent_wins + 4:
                # Regime shift detected — halve all scores to forget
                self._scores *= 0.5
                self._regime_shift_count += 1
            self._recent_wins = 0
            self._recent_losses = 0

        if round_num < 1:
            return self.rng.choice(MOVES)

        # Safety: Nash when losing badly
        if round_num > 20 and self._opp_wins - self._my_wins > 12:
            return self.rng.choice(MOVES)

        preds = self._generate_predictions()
        valid_mask = preds >= 0
        if not _np.any(valid_mask):
            return self.rng.choice(MOVES)

        valid_scores = self._scores[valid_mask]
        valid_preds = preds[valid_mask]

        # Phantom's proven temperature
        temp = 0.3
        centered = (valid_scores - valid_scores.max()) / max(temp, 0.01)
        centered = _np.clip(centered, -20, 0)
        weights = _np.exp(centered)
        weights /= weights.sum()

        opp_probs = _np.zeros(3)
        for pred_val, w in zip(valid_preds, weights):
            opp_probs[int(pred_val)] += w

        predicted = int(_np.argmax(opp_probs))
        confidence = opp_probs[predicted]

        # Phantom's proven noise parameters
        if confidence < 0.38:
            if self.rng.random() < 0.25:
                return self.rng.choice(MOVES)

        return self._i2m[(predicted + 1) % 3]


# ALL_ALGORITHM_CLASSES is now dynamically populated at the end of the file.

# ---------------------------------------------------------------------------
# 101: Mirror Opponent
# ---------------------------------------------------------------------------

class MirrorOpponent(Algorithm):
    """Simple mirroring strategy that plays the opponent's last move.
    
    A basic reactive algorithm that copies the opponent's most recent action.
    Effective against simple deterministic patterns but easily exploited by
    anti-mirroring strategies.
    
    **Type**: Reactive / Baseline
    **Logic**: Tit-for-Tat
    """
    name = "Mirror Opponent"
    def choose(self, round_num, my_history, opp_history):
        if not opp_history:
            return self.rng.choice(MOVES)
        return opp_history[-1]


# ---------------------------------------------------------------------------
# 102: Contrarian
# ---------------------------------------------------------------------------

class Contrarian(Algorithm):
    """The Contrarian – Rare move prioritizer.
    
    Tracks its own move history and deliberately selects the move it has used
    least recently. Designed to defeat opponents that model the player's
    frequency and expect a bias toward common moves.
    
    **Type**: Adaptive / Anti-Meta
    **Logic**: Least-Frequent Play
    """
    name = "Contrarian"

    def choose(self, round_num, my_history, opp_history):
        if not my_history:
            return self.rng.choice(MOVES)

        # Find our least-played move recently
        window = my_history[-30:] if len(my_history) >= 30 else my_history
        counts = Counter(window)

        # Ensure all moves counted
        for m in MOVES:
            if m not in counts:
                counts[m] = 0

        # Play our LEAST common move (the one opponents won't predict)
        least_common = min(MOVES, key=lambda m: counts[m])
        return least_common


# ---------------------------------------------------------------------------
# 103: Pattern Detector
# ---------------------------------------------------------------------------

class PatternDetector(Algorithm):
    """Pattern identification engine using variable-length history matching.
    
    Searches historical records for repeating sequences of moves (lengths 2 to 5).
    When a match is found, it plays the counter to the move that followed the
    previous occurrence of the pattern.
    
    **Type**: Pattern Matcher / Statistical
    **Logic**: Sequence Correlation
    """
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
# 90: The Singularity — The Ultimate Ensemble
# ---------------------------------------------------------------------------

class TheSingularity(Algorithm):
    """Apex meta-algorithm fusing high-power predictors.
    
    A sophisticated ensemble that integrates Super Omniscient for prediction,
    The Doppelganger for regime shift detection, and The Void for information-
    theoretic defense. Dynamically switches modes to maintain optimality.
    
    **Type**: Meta-Strategy / Ultimate
    **Logic**: Multi-Agent Fusion
    """
    name = "The Singularity"

    def reset(self):
        self._so = SuperOmniscient()
        self._dg = TheDoppelganger()
        self._void = TheVoid()
        self._so.reset()
        self._dg.reset()
        self._void.reset()

    def choose(self, round_num, my_history, opp_history):
        # Update and get moves from all sub-bots
        m_so = self._so.choose(round_num, my_history, opp_history)
        m_dg = self._dg.choose(round_num, my_history, opp_history)
        m_void = self._void.choose(round_num, my_history, opp_history) # Updates MI

        # 1. Defense Check (The Void)
        # Check internal metrics
        mi = self._void._compute_mi()
        if mi > 0.15: # High mutual information -> We are being read
            # Play The Void's move (which is likely Random or defensive)
            # Or play Mirror World logic (Counter-Counter) if we had it.
            # Singularity plays safe: Random (via Void)
            return m_void

        # 2. Regime Shift Check (The Doppelganger)
        # If Dopperganger detects a recent change, its confidence is higher than stale SO?
        # Access internal change point
        dg_cp = self._dg._change_point
        # If change point is very recent (within last 10 rounds), trust DG
        if len(opp_history) - dg_cp < 15 and len(opp_history) > 30:
            return m_dg

        # 3. Default: Super Omniscient Power
        return m_so


# ---------------------------------------------------------------------------
# 91: The Black Hole — Recursive Anti-Prediction
# ---------------------------------------------------------------------------

class TheBlackHole(Algorithm):
    """Recursive anti-prediction engine with defensive fallback.
    
    Combines information-theoretic monitoring with recursive simulation. If it
    detects it is being read, it switches to Level-K thinking to exploit the
    opponent's model; otherwise, it uses its internal ensemble for exploitation.
    
    **Type**: Information Theory / Simulation
    **Logic**: Recursive Countering
    """
    name = "The Black Hole"

    def reset(self):
        self._void = TheVoid()
        self._mirror = TheMirrorWorld()
        self._void.reset()
        self._mirror.reset()

    def choose(self, round_num, my_history, opp_history):
        m_void = self._void.choose(round_num, my_history, opp_history)
        m_mirror = self._mirror.choose(round_num, my_history, opp_history)

        # Check MI
        mi = self._void._compute_mi()

        if mi > 0.12: # Slightly aggressive threshold
            # They are reading us.
            # Void would play Random.
            # Black Hole plays Mirror World (exploits their prediction).
            return m_mirror

        # We are invisible/unpredictable
        return m_void


# ---------------------------------------------------------------------------
# 92: Phoenix — Adaptive Phantom + Tilt Detection
# ---------------------------------------------------------------------------

class Phoenix(SuperOmniscient):
    """Adaptive meta-strategy with integrated psychological tilt modeling.
    
    Extends Super Omniscient with specialized tilt predictors that activate during
    streaks. Captures and exploits psychological biases that emerge when an
    opponent is winning or losing consistently.
    
    **Type**: Meta-Strategy / Psychological
    **Logic**: Tilt-Bias Prediction
    """
    name = "Phoenix"

    def reset(self):
        super().reset()
        # Add Tilt Predictors
        # 1. Opp loss streak > 2
        # 2. Opp win streak > 2
        # 3. My loss streak > 2
        # 4. My win streak > 2
        self._n_tilt = 4
        self._n_total += self._n_tilt

        # Resize arrays
        self._scores = _np.zeros(self._n_total)
        self._predictions = _np.full(self._n_total, -1, dtype=int)

        # Track streaks
        self._opp_loss_streak = 0
        self._opp_win_streak = 0

        # Tilt conditional probs: [streak_len][next_move]
        self._tilt_opp_loss = _np.ones((10, 3))
        self._tilt_opp_win = _np.ones((10, 3))

    def _generate_predictions(self):
        # reuse parent predictions
        # SuperOmniscient logic is in _generate_predictions... but wait
        # Inheritance of _generate_predictions is tricky if I can't call super() easily inside it
        # because the parent method sets self._predictions directly and returns it.
        # I can call super()._generate_predictions(), get the array, append my predictions?
        # But parent method uses fixed self._n_total.

        # Correction: parent _generate_predictions uses loop ranges based on self._n_hm etc.
        # It fills self._predictions up to self._n_total - (extras).

        # To avoid complexity without full copy-paste, I will rely on the fact that
        # SuperOmniscient._generate_predictions fills indices 0 to X.
        # I increased _n_total. So I can just fill the end indices.

        # Run parent logic
        # Ideally I should copy the code, but 'enhance existing' implies reusing.
        # Using composition or full copy is safer.
        # Let's use Composition for Phoenix too, to avoid broken inheritance assumptions.
        # Actually, full copy-paste of SuperOmniscient logic + extras is safest.
        # But for brevity here, I'll attempt the 'extension' if safe.
        # SuperOmniscient uses specific indices? No, it uses 'base' offsets.
        # It calculates _n_total in reset.

        # Let's use Composition/Wrapper for Phoenix too.
        # Tilt override
        # If opp is on loose streak > 2, predict their move using specific bias
        if self._opp_loss_streak >= 2:
            # Predict based on what they usually do after losing with 'last'
            if opp_history:
                 last = self._base._m2i[opp_history[-1]]
                 probs = self._tilt_loss_track[last]
                 pred = int(_np.argmax(probs))
                 # If significant bias
                 if probs[pred] / probs.sum() > 0.6:
                     return self._base._i2m[(pred + 1) % 3]

        if self._opp_win_streak >= 2:
            if opp_history:
                last = self._base._m2i[opp_history[-1]]
                probs = self._tilt_win_track[last]
                pred = int(_np.argmax(probs))
                if probs[pred] / probs.sum() > 0.6:
                    return self._base._i2m[(pred + 1) % 3]

        return m_base

# Duplicate class definition removed
    name = "Phoenix"
    def reset(self):
        self._base = SuperOmniscient()
        self._base.reset()
        # Tilt tracking
        self._opp_loss_streak = 0
        self._opp_win_streak = 0
        self._tilt_loss_track = _np.ones((3, 3)) # [last_move][next_move] after loss
        self._tilt_win_track = _np.ones((3, 3))

    def choose(self, round_num, my_history, opp_history):
        # Base move
        m_base = self._base.choose(round_num, my_history, opp_history)

        # Update streaks
        if opp_history and len(opp_history) >= 2:
            last = self._base._m2i[opp_history[-1]]
            prev = self._base._m2i[opp_history[-2]]
            my_last = self._base._m2i[my_history[-1]]
            my_prev = self._base._m2i[my_history[-2]]

            # Did opp lose last round?
            # (My prev - Opp prev) % 3 == 1
            if (my_prev - prev) % 3 == 1:
                self._opp_loss_streak += 1
                self._opp_win_streak = 0
                # Update tracker: they played 'last' after losing
                self._tilt_loss_track[prev][last] += 1
            elif (prev - my_prev) % 3 == 1:
                self._opp_win_streak += 1
                self._opp_loss_streak = 0
                self._tilt_win_track[prev][last] += 1
            else:
                self._opp_loss_streak = 0
                self._opp_win_streak = 0

        # Tilt override
        # If opp is on loose streak > 2, predict their move using specific bias
        if self._opp_loss_streak >= 2:
            # Predict based on what they usually do after losing with 'last'
            last = self._base._m2i[opp_history[-1]]
            probs = self._tilt_loss_track[last]
            pred = int(_np.argmax(probs))
            # If significant bias
            if probs[pred] / probs.sum() > 0.6:
                return self._base._i2m[(pred + 1) % 3]

        if self._opp_win_streak >= 2:
            last = self._base._m2i[opp_history[-1]]
            probs = self._tilt_win_track[last]
            pred = int(_np.argmax(probs))
            if probs[pred] / probs.sum() > 0.6:
                return self._base._i2m[(pred + 1) % 3]

        return m_base

# Dynamically register all subclasses of Algorithm as available bots
def _get_all_subclasses(cls):
    all_subclasses = []
    for subclass in cls.__subclasses__():
        all_subclasses.append(subclass)
        all_subclasses.extend(_get_all_subclasses(subclass))
    return all_subclasses

# Filter and unique-ify
ALL_ALGORITHM_CLASSES = []
seen = set()
for cls in _get_all_subclasses(Algorithm):
    if hasattr(cls, 'name') and cls.name and cls != Algorithm and not cls.__name__.endswith('Base'):
        if cls.name not in seen:
            ALL_ALGORITHM_CLASSES.append(cls)
            seen.add(cls.name)





def get_all_algorithms() -> list[Algorithm]:
    """Return fresh instances of all algorithms."""
    return [cls() for cls in ALL_ALGORITHM_CLASSES]


def get_algorithm_by_name(name: str) -> Algorithm:
    """Get a single algorithm instance by name (case-insensitive)."""
    name_lower = name.lower()
    for cls in ALL_ALGORITHM_CLASSES:
        if cls.name.lower() == name_lower:
            return cls()
    available = ", ".join(cls.name for cls in ALL_ALGORITHM_CLASSES)
    raise ValueError(f"Unknown algorithm: '{name}'. Available: {available}")
