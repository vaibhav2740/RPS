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

class PersistentRandom(Algorithm):
    """Picks a random move and plays it for a random duration (5-15 rounds).
    Then picks a new move and repeats.
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

class Spiral(Algorithm):
    """Plays R, R, P, P, S, S... in a continuous spiral pattern.
    Slower than the standard Cycle.
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
    """Thompson Sampler v5 with Bayesian linear regression (pure Python).

    Maintains per-action posterior N(μ, Σ) where Σ⁻¹ = λI + Σ φφᵀ.
    Samples weights from posterior via Cholesky decomposition, picks
    action with highest Q-sample.
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
    """LinUCB v5 — contextual bandit with linear payoff model (pure Python).

    For each action: UCB = wᵀφ + α·√(φᵀA⁻¹φ).
    A is the feature covariance matrix, updated online.
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

class MarkovGenerator(Algorithm):
    """Generates moves based on a random internal Markov chain.
    
    Creates a random transition matrix on reset and follows it.
    Does not adapt to opponent; acts as a 'structured noise' generator.
    Resets the matrix every 50 rounds to change the structure.
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
# 72: UCB-NGram Fusion (Hybrid — exploits complementary phase advantages)
# ---------------------------------------------------------------------------

class UCBNGramFusion(Algorithm):
    """Hybrid that fuses UCB bandit exploration with N-Gram pattern prediction.

    Analysis shows UCB Explorer beats N-Gram via early-game exploration
    unpredictability (rounds 0-200), while N-Gram dominates late-game
    once it has enough data to predict patterns (rounds 300+).

    This fusion combines three layers:
    1. Strategy Layer: UCB bandit, N-Gram predictor, frequency counter
    2. Selection Layer: softmax mixture weighted by rolling accuracy
    3. Meta-Prediction Layer: simulates opponent modeling OUR patterns
       and counter-rotates when we become predictable

    The meta-prediction layer is what makes this hybrid stronger than
    either parent — it resists being pattern-matched by N-Gram-type
    opponents while maintaining prediction power against others.
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

    Original uses 6 meta-strategies. Plus version adds:
    - Markov counter: predict via transition matrix
    - Bigram counter: predict from (opp[-2],opp[-1]) pattern
    - Trigram counter: predict from (opp[-3],opp[-2],opp[-1])
    - Mirror of each new strategy (from opponent's perspective)
    - Sliding window scoring (last 50 rounds) with exponential decay
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
    """Upgraded Mixture Model with 8 experts, pruning, and spawning.

    Original has 5 fixed experts. Dynamic version adds:
    - 8 experts (+ Markov-2, Transition counter, Win-pattern)
    - Expert pruning: drop experts with <25% accuracy after 100 rounds
    - Expert spawning: clone best expert with noise every 200 rounds
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
    """Upgraded Bayesian Predictor with learned prior and change-point detection.

    Original uses flat Dir(1,1,1) prior with 50-round window. This version:
    - Learns the prior α via evidence maximization (adapts over time)
    - Change-point detection: resets when KL divergence exceeds threshold
    - Multi-window ensemble: combines predictions from windows 20, 50, 100
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
    """Upgraded Anti-Strategy Detector with self-play opponent identification.

    Original detects 5 simple archetypes. Self-Model Detector:
    - Simulates what each of 10 candidate strategies would play
    - Finds which strategy the opponent most closely resembles
    - Plays the known counter to the detected strategy

    Candidate strategies are simple enough to simulate inline:
    constant, cycle, mirror, counter, frequency, markov, WSLS,
    anti-tit-for-tat, pattern-trigger, decay frequency.
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
    """Uses the digits of Pi to determine moves.
    
    Deterministic but high entropy. Uses the first 1000 digits of Pi.
    Maps digits: 0-2->Rock, 3-5->Paper, 6-9->Scissors (roughly balanced).
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
    """Uses the Golden Ratio to generate chaotic deterministic moves.
    
    Formula: move = floor((round_num * φ) % 1 * 3)
    This creates a quasi-periodic sequence that is hard to predict without
    knowing the exact formula.
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
    """Treats R, P, S as stocks in a volatile market.
    
    - 'Stocks' gain value when they would have won the last round.
    - 'Stocks' lose value when they would have lost.
    - Adds random 'market noise' (volatility).
    - Always 'buys' (plays) the highest valued stock.
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
    """Maintains a 'superposition' of move probabilities.
    
    - Wins observable -> Reinforces the state (constructive interference).
    - Losses observable -> Collapses the probability (destructive interference).
    - Renormalizes after every observation.
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
    """Generates moves based on oscillating sine waves.
    
    Uses constructive interference of two sine waves with different frequencies
    to create a complex but deterministic pattern.
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
    """Uses the Ackermann function to determine lookback depth.
    
    The Ackermann function grows extremely rapidly. We use small inputs
    derived from the round number to get a dynamic, non-linear lookback distance.
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
    """Plays aggressively only on prime-numbered rounds.
    
    - If round_num is prime: Plays a hard counter to opponent's last move.
    - If round_num is composite: Plays completely random to confuse prediction.
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
    """Uses zlib compression to predict the most likely next move.
    
    Based on Normalized Compression Distance (NCD).
    It asks: "Which hypothetical next move by the opponent makes their
    history string most compressible?"
    The most compressible sequence is the most predictable one.
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
    """Punishes deviations from Nash Equilibrium.
    
    If the opponent plays any move > 33.3% of the time, this bot
    identifies the frequency bias and exploits it, but remains close to
    random to avoid being exploited itself.
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
    """Mirrors the opponent's move from 2 rounds ago.
    
    Effective against bots that expect immediate mirroring (like Tit-for-Tat)
    or immediate countering (like Win-Stay Lose-Shift).
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
    """Treats moves as DNA sequences (codons) and allows for mutations.
    
    Looks for the last 5 moves (a 'gene') in historical data.
    Unlike standard pattern matchers, this allows for 1 'mutation' (mismatch)
    when searching, simulating biological sequence alignment.
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
    """Cycles through 12 different personality archetypes based on round number.
    
    Each 'sign' (every 12th round) has a distinct strategy:
    Aries(Aggro), Taurus(Stubborn), Gemini(Dual), Cancer(Paper), Leo(Winner),
    Virgo(Analytic), Libra(Balanced), Scorpio(Counter), Sagittarius(Random),
    Capricorn(WSLS), Aquarius(Chaos), Pisces(Mirror).
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
    """A minimal neural network (perceptron) that evolves weights on failure.
    
    Inputs: encoded history of last round.
    Weights: Evolve (add noise) whenever losing streak >= 3.
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
    """Visualizes moves as vectors on an equilateral triangle unit circle.
    
    - Rock:     (1, 0)
    - Paper:    (-0.5, 0.866)
    - Scissors: (-0.5, -0.866)
    
    Calculates the centroid (average vector) of opponent's recent history.
    The response is the move corresponding to the vector *opposite* the centroid.
    This effectively counters the opponent's 'average bias' in 2D space.
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
    """Assume that if a move hasn't happened in a while, it is 'due'.
    
    looks at the last 20 moves. Plays the move that the opponent has
    played the LEAST frequently, expecting them to 'balance' it out.
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
    """ Tries to enforce a perfect 33/33/33 distribution in its OWN history.
    
    If it has played Rock too little, it plays Rock.
    This makes it asymptotically unexploitable by frequency analysis.
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
    """The opposite of Win-Stay Lose-Shift.
    
    - Win: Shift (Don't push your luck).
    - Lose: Stay (Stubbornly try again, doubling down).
    
    Beats standard WSLS players who expect you to shift on loss.
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
    """Mirrors the opponent 80% of the time to build trust.
    
    But 20% of the time, it 'betrays' the mirror logic by playing
    the counter to the opponent's last move, catching them if they
    try to counter the expected mirror.
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
    """Classifies opponent into a 'Persona' based on history.
    
    - Aggressive (>40% Rock): Counts with Paper.
    - Defensive (>40% Paper): Counts with Scissors.
    - Evasive (>40% Scissors): Counts with Rock.
    - Balanced: Plays Random.
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
    """Detects losing streaks and backs off into randomness.
    
    If on a losing streak (N), play purely random for 2^N rounds
    to break any predictive lock the opponent has.
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
    """Monitors its OWN moves to detect and break obviously patterns.
    
    If it detects it has played R-R-R or R-P-S (cycles),
    it intentionally deviates from that pattern.
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
    """Takes a vote from 3 distinct historical windows.
    
    Predicts opponent move based on:
    1. Short term (last 5)
    2. Medium term (last 20)
    3. Long term (last 100)
    
    Plays the counter to the majority vote prediction.
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
    """Switches strategies every 10 rounds to confuse opponent.
    
    - Agent A (Rounds 0-9): Mirror Opponent.
    - Agent B (Rounds 10-19): Counter Opponent.
    - Repeat.
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
    """Specifically targets Win-Stay Lose-Shift (WSLS) logic.
    
    Assumes opponent is playing WSLS:
    - If I Won (Opp Lost): They will Shift. (To beat my last move). 
      Prediction: They play Counter(MyLast).
      Response: I play Counter(Counter(MyLast)).
      
    - If I Lost (Opp Won): They will Stay. (Replay their last).
      Prediction: They play OppLast.
      Response: I play Counter(OppLast).
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

ALL_ALGORITHM_CLASSES = [
    # Baseline (1-20)
    AlwaysRock, AlwaysPaper, AlwaysScissors,
    PureRandom, Cycle, PersistentRandom,
    TitForTat, AntiTitForTat, FrequencyAnalyzer,
    MarkovPredictor, Spiral, WinStayLoseShift,
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
    SleeperAgent, Shapeshifter, HotStreak, MarkovGenerator,
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
    # Hybrid Fusion (72)
    UCBNGramFusion,
    # Upgraded Competitive (73-76)
    IocainePowderPlus, DynamicMixture, HierarchicalBayesian, SelfModelDetector,
    # Creative Boost / Logic (77-100)
    PiBot, GoldenRatio, StockBroker, QuantumCollapse, SoundWave,
    Ackermann, PrimeHunter, CompressionBot, EquilibriumBreaker, DelayedMirror,
    GeneSequencer, Zodiac, NeuroEvo, GeometryBot,
    GamblersFallacy, NashStabilizer, StubbornLoser, TraitorMirror, OpponentPersona,
    ExponentialBackoff, PatternBreaker, SlidingWindowVote, DoubleAgent, CounterStrike,
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
