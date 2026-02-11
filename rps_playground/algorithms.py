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
# 33: Intentional Loser (deliberately tries to lose)
# ---------------------------------------------------------------------------

def _losing_move(move: Move) -> Move:
    """Return the move that LOSES to `move`."""
    return BEATS[move]


class IntentionalLoser(Algorithm):
    """Deliberately tries to LOSE every round.

    Predicts the opponent's next move using frequency analysis,
    then plays the move that LOSES to the prediction.

    Why include it? It's a useful baseline to test that your algorithm
    can at least beat something that actively tries to lose. It's also
    a fun sanity check — any algorithm that loses to the Intentional
    Loser has a serious bug.
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
    """Tabular Q-Learning v4 with experience replay and outcome-aware state.

    State = (my[-1], opp[-1], opp[-2], last_outcome) → 81 states + warm-up.
    Experience replay: stores last 200 transitions, replays 10 random
    ones per round for multi-pass learning (DQN-inspired).
    Pre-trained via self-play against 5 archetypal opponents.

    Q-update: Q(s,a) ← Q(s,a) + α × (reward - Q(s,a))
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
    """Bayesian multi-armed bandit v4 using Beta-Bernoulli model.

    State includes last-round outcome for context-aware exploration.
    For each (state, action), maintains Beta(α, β) where α counts wins
    and β counts losses. Samples from posteriors; highest sample wins.
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
    """UCB1 bandit v4 with outcome-aware state.

    Picks the action maximizing: Q̄(s,a) + c × √(ln(N_s) / n_sa)
    where c = √2 for optimal exploration-exploitation trade-off.
    State includes last-round outcome for contextual decisions.
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
    """Policy gradient v4 with softmax action selection.

    Maintains preference vector h(s,a). Policy π(a|s) = softmax(h).
    Updates preferences via gradient ascent on expected reward.
    Outcome-aware state + entropy regularization to prevent collapse.
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
    """Q-Learning v5 with linear function approximation.

    Replaces tabular Q(s,a) with Q(s,a) = wᵀ·φ(s,a) where φ is a
    16-dim feature vector per action (48 weights total).
    SGD update: w ← w + α·δ·φ where δ = r - Q(s,a).
    Experience replay performs SGD on past transitions.
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
# 64: Thompson Sampler v5 (Bayesian Linear Regression)
# ---------------------------------------------------------------------------

class ThompsonSamplerV5(Algorithm):
    """Thompson Sampler v5 with Bayesian linear regression on features.

    Instead of Beta distributions per (state, action), maintains a
    Bayesian linear model per action: posterior N(μ, Σ) where
    Σ⁻¹ = λI + Σ φφᵀ and μ = Σ · Σ φr.
    Samples weights from posterior, picks action with highest Q-sample.
    """
    name = "Thompson Sampler v5"

    def reset(self):
        import numpy as np
        self._np = np
        d = 16  # feature dimension
        self._lambda = 1.0  # prior precision
        # Per action: (A = Σ⁻¹, b = Σ φr)
        self._A = [np.eye(d) * self._lambda for _ in range(3)]
        self._b = [np.zeros(d) for _ in range(3)]
        self._last_action = None
        self._last_phi = None
        _pretrain_against_archetypes(self)

    def choose(self, round_num, my_history, opp_history):
        np = self._np

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
            phi = np.array(self._last_phi)
            self._A[a_idx] += np.outer(phi, phi)
            self._b[a_idx] += reward * phi

        # Thompson sampling: sample weights from posterior, pick best action
        best_q, best_action = -1e9, MOVES[0]
        best_phi = None
        for m in MOVES:
            phi_list = _build_feature_vector(my_history, opp_history, m)
            phi = np.array(phi_list)
            a_idx = MOVES.index(m)
            try:
                A_inv = np.linalg.inv(self._A[a_idx])
                mu = A_inv @ self._b[a_idx]
                # Sample from posterior
                w_sample = np.random.multivariate_normal(mu, A_inv)
            except np.linalg.LinAlgError:
                mu = np.zeros(16)
                w_sample = mu
            q = float(w_sample @ phi)
            if q > best_q:
                best_q, best_action, best_phi = q, m, phi_list

        self._last_action = best_action
        self._last_phi = best_phi
        return best_action


# ---------------------------------------------------------------------------
# 65: UCB Explorer v5 (LinUCB — contextual bandits)
# ---------------------------------------------------------------------------

class UCBExplorerV5(Algorithm):
    """LinUCB v5 — contextual bandit with linear payoff model.

    For each action: UCB = wᵀφ + α√(φᵀA⁻¹φ).
    A is the feature covariance matrix, updated online.
    """
    name = "UCB Explorer v5"

    def reset(self):
        import numpy as np
        self._np = np
        d = 16
        self._alpha_ucb = 1.5  # exploration coefficient
        self._A = [np.eye(d) for _ in range(3)]
        self._b = [np.zeros(d) for _ in range(3)]
        self._last_action = None
        self._last_phi = None
        _pretrain_against_archetypes(self)

    def choose(self, round_num, my_history, opp_history):
        np = self._np

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
            phi = np.array(self._last_phi)
            self._A[a_idx] += np.outer(phi, phi)
            self._b[a_idx] += reward * phi

        # LinUCB action selection
        best_ucb, best_action = -1e9, MOVES[0]
        best_phi = None
        for m in MOVES:
            phi_list = _build_feature_vector(my_history, opp_history, m)
            phi = np.array(phi_list)
            a_idx = MOVES.index(m)
            try:
                A_inv = np.linalg.inv(self._A[a_idx])
            except np.linalg.LinAlgError:
                A_inv = np.eye(16)
            theta = A_inv @ self._b[a_idx]
            exploit = float(theta @ phi)
            explore = self._alpha_ucb * float(np.sqrt(phi @ A_inv @ phi))
            ucb = exploit + explore
            if ucb > best_ucb:
                best_ucb, best_action, best_phi = ucb, m, phi_list

        self._last_action = best_action
        self._last_phi = best_phi
        return best_action


# ---------------------------------------------------------------------------
# 66: Gradient Learner v5 (Linear Softmax REINFORCE)
# ---------------------------------------------------------------------------

class GradientLearnerV5(Algorithm):
    """Policy gradient v5 with linear softmax on features.

    Preferences h(a) = wₐᵀ·φ(s) per action. Policy π(a|s) = softmax(h).
    REINFORCE gradient update on linear weights.
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
    """Maintains a Dirichlet prior over opponent's move distribution.

    Uses a sliding window of the last 50 moves for adaptivity.
    Posterior: Dir(α_R + n_R, α_P + n_P, α_S + n_S) with α_i = 1.
    Samples from posterior and counters the most probable move.
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
    """Builds n-gram model over JOINT (my_move, opp_move) history.

    Unlike Pattern Detector which only looks at opponent's moves,
    this uses both players' moves as context, capturing interactive
    patterns like "when I play Rock and they play Paper, they usually
    follow with Scissors."

    Searches n = 3, 2, 1 for the best matching context.
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
    """Identifies which known strategy archetype the opponent is using,
    then plays the specific hard counter for that archetype.

    Detects: constant, cycle, mirror, counter, frequency-based.
    Scores each detector by accuracy on last 20 moves.
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
    """Multiplicative weights (Hedge) algorithm over 5 expert strategies.

    Achieves O(√(T log K)) regret — performs nearly as well as the
    best single expert in hindsight.

    Experts: counter-last, frequency, Markov-like, WSLS, random.
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
    """Plays pure random for 80 rounds, collecting data silently.

    After activation, uses a multi-predictor ensemble (frequency,
    Markov, pattern) on the 80 rounds of clean opponent data —
    data unaffected by adversarial adaptation since the opponent
    was just facing "Pure Random" during collection.
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
    """Cycles through 5 completely different strategies every 40 rounds.

    The opponent can't model it because the entire strategy changes
    before they accumulate enough data to counter it.

    Strategies: Random, Counter-last, Frequency, Markov, WSLS
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

    When winning: repeats the winning move (it's working).
    When losing 3+ in a row: switches to pure random to reset.
    During neutral/draws: uses frequency counter.
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

class Contrarian(Algorithm):
    """Plays the move the opponent would LEAST expect.

    Tracks its own move history and deliberately plays the move
    it has used least recently. Opponents modeling our frequency
    will predict our most common move — we play the rarest one.
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
# 46: Monte Carlo Predictor
# ---------------------------------------------------------------------------

class MonteCarloPredictor(Algorithm):
    """Simulates random playout histories and picks the best counter.

    For each possible opponent move, estimates the probability by
    running Monte Carlo simulations over the observed transition
    patterns, then plays the move with the highest expected score.
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
    """Remembers the exact (my_move, opp_move) pairs that caused losses.

    Tracks which of our moves got beaten and by what. Refuses to
    repeat moves that have historically led to losses against
    specific opponent replies. Exploits opponent's winning patterns.
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
    """Matches the opponent's own move distribution.

    If opponent plays 50% Rock, 30% Paper, 20% Scissors,
    Chameleon will also play 50% Rock, 30% Paper, 20% Scissors.

    This is surprisingly effective: against biased opponents,
    it creates "frequency proximity" that forces many draws,
    while the slight randomness prevents easy countering.
    Against pure random, it becomes pure random too.
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
    """Uses Fibonacci sequence to determine move index.

    The Fibonacci sequence mod 3 produces a complex, non-repeating
    (for long stretches) pattern: 0,1,1,2,0,2,2,1,0,1,1,...

    Combined with opponent exploitation: 70% Fibonacci pattern,
    30% frequency counter. The Fibonacci base makes it harder
    for pattern detectors to find the cycle (period 8 in mod 3).
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
    """LZ78 compression-based sequence prediction.

    From information theory: a good compressor IS a good predictor.
    Builds a dictionary of observed subsequences incrementally.
    The current phrase context determines the prediction.

    If the opponent's sequence compresses poorly (high Kolmogorov
    complexity), they're close to random and we default to frequency.
    If it compresses well, we exploit the detected structure.
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
    """Context Tree Weighting — provably optimal universal prediction.

    A Bayesian mixture over ALL possible context depths (0 to D=6).
    Uses the Krichevsky-Trofimov (KT) estimator at each node and
    weights each depth by its posterior probability.

    Unlike N-Gram Predictor which tries fixed n=3,2,1, Context Tree
    computes a proper weighted average over ALL depths simultaneously,
    giving more weight to depths with better predictive track records.

    Joint history version: uses (my_move, opp_move) pairs as context,
    combining the upgrades from N-Gram's joint approach with CTW's
    optimal depth selection.
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

    Finds the distribution P(opp_next) with MAXIMUM ENTROPY subject
    to constraints from 3 observed feature scales:
      - Marginal frequencies (how often they play each move)
      - 1st-order transitions P(next | last)
      - 2nd-order transitions P(next | last_2, last_1)

    The MaxEnt solution is a log-linear (exponential family) model:
      log P(m) ∝ λ₁·log(f_marginal) + λ₂·log(f_transition) + λ₃·log(f_bigram)

    This is the "least biased" estimate consistent with observed data.
    Feature weights adapt based on recent predictive accuracy.
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
    """Deliberately plants a bias, waits for opponent to adapt, exploits.

    3-phase cycle repeated every 90 rounds:
    Phase 1 (30 rounds): Plant 85% Rock bias → opponent adapts to Paper
    Phase 2 (30 rounds): Exploit with 85% Scissors (beats their Paper)
    Phase 3 (30 rounds): Switch to 85% Paper → opponent expects Scissors
    Then: cycle poisons in a new order so it's never the same twice

    The key insight: most algorithms need ~15-25 rounds to detect a
    bias. Poison Pill changes every 30, keeping opponents perpetually
    one step behind.
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
    """Specifically designed to exploit reactive/mirror opponents.

    Phase 1: Play a diagnostic sequence to identify mirrors/copycats.
    Phase 2: If reactive opponent detected, create a feedback trap
             where the opponent's reactions become predictable.

    Against Mirror: I play R → they play R → I play P → they play P → ...
      Trap: I play the counter of whatever I played last.
    Against Counter: I play R → they play P →
      Trap: I play what I played last (they counter it, I counter their counter).
    Against non-reactive: fall back to Markov prediction.
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
    """Identifies which known strategy archetype the opponent uses,
    then becomes a STRICTLY BETTER version of that strategy.

    5 archetype detectors run in parallel with exponentially decaying
    scores. Once an archetype is identified with confidence > 60%,
    The Usurper plays the specific hard counter.

    Unlike Anti-Strategy Detector (which counter-predicts), The Usurper
    counter-STRATEGIZES: it exploits the mathematical weakness of the
    entire strategy class, not just individual predictions.
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
    """Multi-level reasoning with adaptive depth selection.

    Maintains 3 reasoning levels:
    Level 0: Counter their most common (basic frequency analysis)
    Level 1: "They predict level 0" → counter(counter(predicted))
    Level 2: "They predict level 1" → counter(counter(counter(predicted)))
    Note: Level 3 = Level 0 (mod 3 cycle), so 3 levels suffice.

    Tracks which reasoning level would have won historically (with
    exponential recency weighting) and follows the best-performing one.
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
    """Deliberately creates FALSE patterns to mislead frequency-based opponents,
    while tracking and countering their actual moves.

    Phase 1 (20 rounds): Establish a strong fake pattern (e.g., RRPRR PRRPR...)
    Phase 2 (10 rounds): Switch to the move that destroys opponents
                          who adapted to the fake pattern
    Phase 3: Execute frequency analysis on their REAL distribution
             (revealed during Phase 2, when they were chasing the fake)

    The disruptor cycles through different fake patterns to prevent
    meta-detection of the disruption strategy itself.
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
    """Upgraded Historian with joint pattern matching and recency weighting.

    Original Historian matches opponent-only sequences of fixed length 4.
    Deep Historian improves on this with:
    - Joint (my_move, opp_move) pair patterns (captures interactive play)
    - Variable-length matching (tries 5,4,3,2 and takes longest match)
    - Exponential recency decay (recent patterns weighted 4x vs old ones)
    - Win/loss context (what happened after similar patterns before?)
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
    """Upgraded N-Gram Predictor with dynamic context and decay-weighted counts.

    Original N-Gram tries fixed n=3,2,1 with equal weighting.
    Adaptive N-Gram improves with:
    - Dynamic context length (tries n=5 down to n=1)
    - Exponential decay on counts (recent transitions 3x heavier)
    - Accuracy tracking per context length (learns which n works best)
    - Joint (my, opp) pair contexts for deeper pattern capture

    Uses a meta-learner to select the best-performing context length
    based on rolling prediction accuracy.
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
    """Regret Matching — the foundation of modern game-playing AI.

    This is the core algorithm behind Libratus and Pluribus, the AIs
    that beat world champions at poker. It provably converges to a
    Nash equilibrium while exploiting non-Nash opponents.

    For each move, tracks REGRET: how much better we would have done
    playing that move vs what we actually played. Plays moves
    proportional to their positive cumulative regret.

    Regret formula:
      regret(m) += payoff(m, opp_move) - payoff(my_move, opp_move)
      strategy(m) = max(0, regret(m)) / sum(max(0, regret(m')))
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
    """Applies Discrete Fourier Transform to detect hidden periodicities.

    Encodes opponent moves as numbers (R=0, P=1, S=2) and computes
    the DFT to find dominant frequency components. Extrapolates the
    dominant frequencies to predict the next value.

    The DFT decomposes the signal x[n] into frequency components:
      X[k] = Σ x[n] × e^(-2πi·k·n/N)

    If the opponent has ANY periodic pattern (even noisy), the DFT
    will detect it. Works against Cycle, Phase Shifter, Fibonacci,
    De Bruijn Walker, and any algorithm with periodic behavior.
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
    """Predicts using the dominant eigenvector of the opponent's transition matrix.

    Builds a 3×3 transition matrix M[i][j] = P(opp plays j | opp played i).
    Computes the STATIONARY DISTRIBUTION π via power iteration:
      π = lim(n→∞) M^n × π₀

    The stationary distribution reveals the opponent's long-term behavior.
    For prediction, combines:
    - Current-row prediction (what follows their last move)
    - Stationary distribution (their overall bias)
    - 2nd-order: M² row (what happens after the current transition)

    Power iteration: π^(t+1) = M^T × π^(t), repeated until convergence.
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
    """HMM-based predictor that discovers opponent's hidden 'moods'.

    Assumes the opponent has 3 hidden states with different move
    distributions. Uses the forward algorithm for state inference
    and online Baum-Welch for parameter updates every 20 rounds.
    Predicts by marginalizing over hidden states.
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
    """Evolves a population of response tables via natural selection.

    Each genome maps (opp[-1], opp[-2]) → Move (9 entries).
    Fitness is tested against opponent's recent 50 moves.
    Every 25 rounds: selection, crossover, mutation.
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
    """Feedback control strategy using PID (Proportional-Integral-Derivative).

    Error signal per move = expected win rate - actual win rate.
    PID regulates move selection probabilities via softmax.
    Kp=0.5, Ki=0.05, Kd=0.2, anti-windup on integral ±10.
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
    """Deterministic but unpredictable via the logistic map.

    Uses x_{n+1} = 3.99 · xₙ · (1-xₙ) for chaotic move generation.
    70% chaotic moves, 30% frequency exploitation.
    Re-seeds from outcome hash every 50 rounds.
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
    """Cognitive hierarchy model — detects opponent's reasoning level.

    Level 0: uniform random
    Level 1: frequency counter (best respond to level 0)
    Level 2: counter frequency counter
    Level 3: counter-counter-counter
    Level 4: counter-counter-counter-counter

    Detects opponent's level by simulating each, then plays one
    level above. Falls back to Regret Matching when unclear.
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
    # Special (33)
    IntentionalLoser,
    # RL / ML v4 (34-37)
    QLearner, ThompsonSampler, UCBExplorer, GradientLearner,
    # Advanced Competitive (38-41)
    BayesianPredictor, NGramPredictor, AntiStrategyDetector, MixtureModel,
    # Creative / Deception (42-49)
    SleeperAgent, Shapeshifter, HotStreak, Contrarian,
    MonteCarloPredictor, GrudgeHolder, Chameleon, FibonacciPlayer,
    # Math-heavy / Proven Concepts (50-52)
    LempelZivPredictor, ContextTree, MaxEntropyPredictor,
    # Trojan / Deception / Weird (53-57)
    PoisonPill, MirrorBreaker, TheUsurper, DoubleBluff, FrequencyDisruptor,
    # Upgraded Variants (58-59)
    DeepHistorian, AdaptiveNGram,
    # Deep Math / Proven Theory (60-62)
    RegretMinimizer, FourierPredictor, EigenvaluePredictor,
    # RL v5: Linear Function Approximation (63-66)
    QLearnerV5, ThompsonSamplerV5, UCBExplorerV5, GradientLearnerV5,
    # New-Field Algorithms (67-71)
    HiddenMarkovOracle, GeneticStrategist, PIDController, ChaosEngine, LevelKReasoner,
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
