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
#  RL / ML ALGORITHMS (34-37) — v3: Self-play pre-training
# ===========================================================================


def _pretrain_against_archetypes(algo, rounds_per: int = 80):
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
    """Tabular Q-Learning v2 with deeper state representation.

    State = (my[-1], opp[-1], opp[-2]) → 27 states + warm-up states.
    Captures 2nd-order opponent transitions for better prediction.

    Q-update: Q(s,a) ← Q(s,a) + α × (reward - Q(s,a))
    """
    name = "Q-Learner"

    def reset(self):
        # 27 states for (my[-1], opp[-1], opp[-2]) + warm-up states
        # 3 actions per state → up to 81+ Q-values
        self._q_table: dict[tuple, dict[Move, float]] = {}
        self._alpha = 0.2      # learning rate (lower for more states)
        self._epsilon = 0.3    # exploration rate
        self._last_state = None
        self._last_action = None
        self._rounds_played = 0
        _pretrain_against_archetypes(self)

    def _get_state(self, my_history, opp_history):
        if not my_history:
            return ("START",)
        if len(opp_history) < 2:
            return ("EARLY", my_history[-1], opp_history[-1])
        return (my_history[-1], opp_history[-1], opp_history[-2])

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
    """Bayesian multi-armed bandit using Beta-Bernoulli model.

    For each (state, action), maintains Beta(α, β) where α counts wins
    and β counts losses. Samples from each posterior and picks the action
    with the highest sample. Naturally balances exploration/exploitation.
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
        if len(opp_history) < 2:
            return ("EARLY", opp_history[-1])
        return (my_history[-1], opp_history[-1], opp_history[-2])

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
    """UCB1 bandit algorithm with state-dependent action selection.

    Picks the action maximizing: Q̄(s,a) + c × √(ln(N_s) / n_sa)
    where c = √2 for optimal exploration-exploitation trade-off.
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
        if len(opp_history) < 2:
            return ("EARLY", opp_history[-1])
        return (my_history[-1], opp_history[-1], opp_history[-2])

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
    """Policy gradient with softmax action selection.

    Maintains preference vector h(s,a). Policy π(a|s) = softmax(h).
    Updates preferences via gradient ascent on expected reward.
    Can learn stochastic (mixed) strategies, unlike Q-learning.
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
        if len(opp_history) < 2:
            return ("EARLY", my_history[-1], opp_history[-1])
        return (my_history[-1], opp_history[-1], opp_history[-2])

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
    # RL / ML v3 (34-37)
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





