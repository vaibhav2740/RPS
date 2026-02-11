"""32 Rock-Paper-Scissors algorithms (20 baseline + 12 creative)."""

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


# ===========================================================================
#  CREATIVE ALGORITHMS (21-32)
# ===========================================================================


# ---------------------------------------------------------------------------
# 21: Decay Analyzer (exponential recency-weighted frequency)
# ---------------------------------------------------------------------------

class DecayAnalyzer(Algorithm):
    """Frequency analysis where recent moves matter exponentially more.

    Uses a decay factor (0.9) so the last 10-20 moves dominate the
    prediction, adapting much faster than vanilla FrequencyAnalyzer.
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
    """Finds the longest matching suffix in past history and predicts what
    the opponent played immediately after that same context.

    Inspired by Lempel-Ziv compression — the longer the matched context,
    the higher the predictive confidence looks.
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

    Thinks: "I played X → opponent expects me to play X again →
    opponent will play counter(X) → I should play counter(counter(X))."

    This creates a 2-level reasoning chain that beats naive counter-last
    strategies but loses to simple repeaters.
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
    """Copies the opponent's move from 3 rounds ago instead of last round.

    Creates a temporal echo — useful when opponents have cyclic patterns
    with period > 1, and confusing for strategies that only look at the
    immediate last move.
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
    """Feeds a predictable pattern (always Rock) for the first 30 rounds
    to bait the opponent into adapting, then abruptly switches to
    exploiting their adaptation.

    Opponents that adapt to "always Rock" will start playing Paper —
    the Trojan then switches to Scissors and shreds them.
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
    """Plays purely random until it has enough data to be statistically
    confident about the opponent's bias (chi-squared-like threshold).

    Once confident, switches to hard exploitation. Resets confidence
    check every 100 rounds in case opponent changed strategy.
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
    """Monitors its OWN move distribution entropy. If becoming too
    predictable (entropy drops below threshold), forces random play.
    Otherwise exploits the opponent.

    Ensures it never becomes easy to read while still adapting.
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
    """Assumes the opponent uses a "counter my last move" strategy.

    Thinks: "I played X → opponent plays counter(X) → so I should play
    counter(counter(X)) which actually equals the move that LOSES to X."

    But wait — that's what Reverse Psychologist does too. The difference:
    Second Guess also monitors whether the opponent IS actually countering,
    and falls back to frequency analysis if they're not.
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
    """Runs 5 independent sub-strategies each round and uses majority vote.

    Sub-strategies: counter-last, mirror, frequency, anti-cycle, random.
    The wisdom of crowds — no single strategy dominates, but the ensemble
    is robust against many different opponent types.
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
    """Alternates between two modes in timed phases:

    - Aggressive (40 rounds): Full exploitation — counter opponent's
      most common recent move.
    - Defensive (20 rounds): Pure randomness to reset opponent's
      model of us.

    The asymmetric timing means it spends more time exploiting than
    hiding, but the defensive bursts prevent easy counter-adaptation.
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
    """Walks through moves following a De Bruijn-like sequence that covers
    all possible 2-grams (RR, RP, RS, PR, PP, PS, SR, SP, SS).

    This makes it extremely hard for pattern detectors to find repeating
    subsequences, since every possible pair appears exactly once per cycle.
    Occasionally (10%) deviates to exploit if a strong bias is detected.
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
    """Inspired by the legendary "Iocaine Powder" RPS bot.

    Runs 6 meta-strategies simultaneously:
    1. Naive: predict opponent plays same as last
    2. Naive counter: predict opponent counters our last
    3. Naive counter-counter: one level deeper
    4-6: Same three but applied to OPPONENT's perspective

    Each meta-strategy is scored by how well it would have predicted
    the actual last move. The best-performing one is used.

    Named after the battle of wits in The Princess Bride —
    "I know that you know that I know..."
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
# Registry
# ---------------------------------------------------------------------------

ALL_ALGORITHM_CLASSES = [
    # Baseline (1-20)
    AlwaysRock, AlwaysPaper, AlwaysScissors,
    PureRandom, Cycle, MirrorOpponent,
    TitForTat, AntiTitForTat, FrequencyAnalyzer,
    MarkovPredictor, PatternDetector, WinStayLoseShift,
    MetaPredictor, NoiseStrategy, AdaptiveHybrid,
    LastMoveCounter, WeightedRandom, Punisher,
    Forgiver, ChaosStrategy,
    # Creative (21-32)
    DecayAnalyzer, Historian, ReversePsychologist,
    Echo, TrojanHorse, ReluctantGambler,
    EntropyGuardian, SecondGuess, MajorityRule,
    PhaseShifter, DeBruijnWalker, IocainePowder,
]


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

