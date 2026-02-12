"""Competition-optimized bot — 'The Hydra'.

Designed for 100-round matches with opponent metadata.
Uses an 8-expert Hedge ensemble with opponent profiling.

Competition API:
    bot = CompetitionBot()
    move = bot.make_move(history, opponent_name, opponent_history)

Where:
    history: list of [your_move, opp_move] pairs (0=Rock, 1=Paper, 2=Scissors)
    opponent_name: str
    opponent_history: list of {"opponent": str, "result": str, "score": str}
    Returns: int (0=Rock, 1=Paper, 2=Scissors)
"""

import math
import random
from collections import Counter


# Move constants
ROCK, PAPER, SCISSORS = 0, 1, 2
MOVES = [ROCK, PAPER, SCISSORS]
COUNTER = {ROCK: PAPER, PAPER: SCISSORS, SCISSORS: ROCK}
BEATS = {ROCK: SCISSORS, PAPER: ROCK, SCISSORS: PAPER}  # BEATS[x] = what x beats


# ==========================================================================
# Expert Strategies — each returns a PREDICTED OPPONENT MOVE or None
# ==========================================================================

class _Expert:
    """Base class for experts."""
    def __init__(self):
        self.name = "base"

    def predict_opponent(self, my_moves, opp_moves):
        """Return predicted opponent move, or None if unsure."""
        return None


class FrequencyExpert(_Expert):
    """Counter opponent's most common move (last 20)."""
    def __init__(self):
        self.name = "frequency"

    def predict_opponent(self, my_moves, opp_moves):
        if not opp_moves:
            return None
        window = opp_moves[-20:]
        counts = Counter(window)
        return counts.most_common(1)[0][0]


class DecayFrequencyExpert(_Expert):
    """Exponentially weighted move frequency."""
    def __init__(self):
        self.name = "decay_freq"
        self._decay = 0.85

    def predict_opponent(self, my_moves, opp_moves):
        if not opp_moves:
            return None
        weights = [0.0, 0.0, 0.0]
        w = 1.0
        for m in reversed(opp_moves):
            weights[m] += w
            w *= self._decay
        return weights.index(max(weights))


class MarkovExpert(_Expert):
    """Order-1 + order-2 Markov chain prediction."""
    def __init__(self):
        self.name = "markov"

    def predict_opponent(self, my_moves, opp_moves):
        if len(opp_moves) < 2:
            return None

        # Order 2: (opp[-2], opp[-1]) → next
        if len(opp_moves) >= 3:
            key2 = (opp_moves[-2], opp_moves[-1])
            order2_counts = [0, 0, 0]
            for i in range(len(opp_moves) - 2):
                if (opp_moves[i], opp_moves[i+1]) == key2:
                    if i + 2 < len(opp_moves):
                        order2_counts[opp_moves[i+2]] += 1
            if sum(order2_counts) >= 2:
                return order2_counts.index(max(order2_counts))

        # Fallback: order 1
        key1 = opp_moves[-1]
        order1_counts = [0, 0, 0]
        for i in range(len(opp_moves) - 1):
            if opp_moves[i] == key1:
                order1_counts[opp_moves[i+1]] += 1
        if sum(order1_counts) >= 1:
            return order1_counts.index(max(order1_counts))

        return None


class NGramExpert(_Expert):
    """Trigram + 4-gram pattern matching on (my, opp) pairs."""
    def __init__(self):
        self.name = "ngram"

    def predict_opponent(self, my_moves, opp_moves):
        if len(opp_moves) < 4 or len(my_moves) < 4:
            return None

        pairs = list(zip(my_moves, opp_moves))

        # Try 4-gram first
        key4 = tuple(pairs[-3:])
        counts4 = [0, 0, 0]
        for i in range(len(pairs) - 3):
            if tuple(pairs[i:i+3]) == key4 and i + 3 < len(pairs):
                counts4[pairs[i+3][1]] += 1
        if sum(counts4) >= 2:
            return counts4.index(max(counts4))

        # Fallback: trigram
        key3 = tuple(pairs[-2:])
        counts3 = [0, 0, 0]
        for i in range(len(pairs) - 2):
            if tuple(pairs[i:i+2]) == key3 and i + 2 < len(pairs):
                counts3[pairs[i+2][1]] += 1
        if sum(counts3) >= 1:
            return counts3.index(max(counts3))

        return None


class AntiPatternExpert(_Expert):
    """Predicts opponent will counter our most-played move."""
    def __init__(self):
        self.name = "anti_pattern"

    def predict_opponent(self, my_moves, opp_moves):
        if len(my_moves) < 5:
            return None
        my_common = Counter(my_moves[-15:]).most_common(1)[0][0]
        # Opponent would play the counter to our most common
        return COUNTER[my_common]


class NashExpert(_Expert):
    """Uniform 1/3 — pure noise expert to prevent overfitting."""
    def __init__(self):
        self.name = "nash"

    def predict_opponent(self, my_moves, opp_moves):
        return random.choice(MOVES)


class IocaineExpert(_Expert):
    """Predict: opponent predicts our move via frequency, then counters it."""
    def __init__(self):
        self.name = "iocaine"

    def predict_opponent(self, my_moves, opp_moves):
        if len(my_moves) < 3:
            return None
        my_counts = Counter(my_moves[-15:])
        my_predicted = my_counts.most_common(1)[0][0]
        # Opponent counters our predicted move
        return COUNTER[my_predicted]


class BayesianTransitionExpert(_Expert):
    """Dirichlet-updated transition probabilities."""
    def __init__(self):
        self.name = "bayesian"

    def predict_opponent(self, my_moves, opp_moves):
        if len(opp_moves) < 2:
            return None

        alpha = {m: [1.0, 1.0, 1.0] for m in MOVES}
        for i in range(len(opp_moves) - 1):
            alpha[opp_moves[i]][opp_moves[i+1]] += 1.0

        last = opp_moves[-1]
        probs = alpha[last]
        return probs.index(max(probs))


# ==========================================================================
# Opponent Profiler
# ==========================================================================

def _profile_opponent(name, opp_history):
    """Classify opponent from name and tournament history."""
    name_lower = name.lower() if name else ""

    if any(k in name_lower for k in ['random', 'chaos', 'noise', 'uniform']):
        return 'RANDOM', 0.7
    if 'always rock' in name_lower or (name_lower == 'rock'):
        return 'STATIC_ROCK', 0.9
    if 'always paper' in name_lower or (name_lower == 'paper'):
        return 'STATIC_PAPER', 0.9
    if 'always scissors' in name_lower or (name_lower == 'scissors'):
        return 'STATIC_SCISSORS', 0.9
    if any(k in name_lower for k in ['copy', 'mirror', 'tit', 'copycat']):
        return 'REACTIVE', 0.6
    if any(k in name_lower for k in ['frequency', 'counter', 'freq']):
        return 'FREQUENCY', 0.6
    if any(k in name_lower for k in ['markov', 'predictor', 'pattern']):
        return 'PREDICTOR', 0.5
    if any(k in name_lower for k in ['meta', 'ensemble', 'hydra', 'mixture']):
        return 'META', 0.5

    if opp_history:
        total = len(opp_history)
        wins = sum(1 for m in opp_history if m.get('result') == 'win')
        win_rate = wins / max(total, 1)
        if win_rate > 0.75:
            return 'STRONG', 0.6
        if win_rate < 0.25:
            return 'WEAK', 0.6

    return 'UNKNOWN', 0.3


def _get_initial_bias(profile_type):
    """Return initial expert weight multipliers based on opponent profile."""
    biases = {
        'RANDOM':           {'nash': 5.0, 'frequency': 0.3, 'markov': 0.3},
        'STATIC_ROCK':      {'frequency': 5.0, 'decay_freq': 5.0},
        'STATIC_PAPER':     {'frequency': 5.0, 'decay_freq': 5.0},
        'STATIC_SCISSORS':  {'frequency': 5.0, 'decay_freq': 5.0},
        'REACTIVE':         {'anti_pattern': 3.0, 'iocaine': 3.0},
        'FREQUENCY':        {'iocaine': 3.0, 'anti_pattern': 2.0},
        'PREDICTOR':        {'anti_pattern': 2.0, 'iocaine': 2.0, 'nash': 1.5},
        'META':             {'nash': 3.0, 'bayesian': 1.5},
        'STRONG':           {'nash': 2.5, 'iocaine': 1.5, 'bayesian': 1.5},
        'WEAK':             {'frequency': 2.0, 'decay_freq': 2.0, 'markov': 2.0},
        'UNKNOWN':          {},
    }
    return biases.get(profile_type, {})


# ==========================================================================
# The Hydra — Competition Bot
# ==========================================================================

class CompetitionBot:
    """The Hydra — competition-optimized meta-algorithm.

    8-expert ensemble with Hedge (multiplicative weights).
    Exploits opponent_name and opponent_history for fast profiling.
    Designed for 100-round matches.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self._experts = [
            FrequencyExpert(),
            DecayFrequencyExpert(),
            MarkovExpert(),
            NGramExpert(),
            AntiPatternExpert(),
            NashExpert(),
            IocaineExpert(),
            BayesianTransitionExpert(),
        ]
        self._n_experts = len(self._experts)

        # Hedge learning rate: aggressive for 100 rounds
        self._eta = math.sqrt(8 * math.log(self._n_experts) / 100)

        # Expert weights
        self._weights = [1.0] * self._n_experts

        # Track expert predictions for credit assignment
        self._last_predictions = [None] * self._n_experts

        # Opponent profile
        self._profile = 'UNKNOWN'
        self._profile_confidence = 0.0
        self._profiled = False

        # Running score
        self._my_wins = 0
        self._opp_wins = 0

    def _update_expert_weights(self, opp_actual):
        """Reward experts whose predictions were correct."""
        for i in range(self._n_experts):
            pred = self._last_predictions[i]
            if pred is None:
                continue

            # Simple: +1 if prediction matches, -0.5 if wrong
            if pred == opp_actual:
                reward = 1.0
            else:
                reward = -0.5

            self._weights[i] *= math.exp(self._eta * reward)

        # Normalize
        total = sum(self._weights)
        if total > 0:
            self._weights = [w / total * self._n_experts for w in self._weights]

    def _detect_randomness(self, opp_moves):
        """Chi-squared test for uniform distribution."""
        if len(opp_moves) < 20:
            return False
        window = opp_moves[-30:]
        counts = Counter(window)
        n = len(window)
        expected = n / 3
        chi2 = sum((counts.get(m, 0) - expected) ** 2 / expected for m in MOVES)
        return chi2 < 3.0

    def make_move(self, history, opponent_name="", opponent_history=None):
        """Competition API: returns 0/1/2 for Rock/Paper/Scissors."""
        if opponent_history is None:
            opponent_history = []

        my_moves = [h[0] for h in history] if history else []
        opp_moves = [h[1] for h in history] if history else []
        round_num = len(history)

        # === Profile opponent on first call ===
        if not self._profiled:
            self._profile, self._profile_confidence = _profile_opponent(
                opponent_name, opponent_history
            )
            self._profiled = True
            biases = _get_initial_bias(self._profile)
            for i, expert in enumerate(self._experts):
                mult = biases.get(expert.name, 1.0)
                self._weights[i] *= mult

        # === Update weights from last round ===
        if opp_moves:
            opp_last = opp_moves[-1]
            self._update_expert_weights(opp_last)
            if history:
                my_last = my_moves[-1]
                if BEATS[my_last] == opp_last:
                    self._my_wins += 1
                elif BEATS[opp_last] == my_last:
                    self._opp_wins += 1

        # === Meta-layer overrides ===

        # Losing badly → Nash mixed strategy
        if round_num >= 30 and (self._opp_wins - self._my_wins) > 10:
            return random.choice(MOVES)

        # Detected random → Nash
        if round_num >= 20 and self._detect_randomness(opp_moves):
            return random.choice(MOVES)

        # === Opening strategy (rounds 0-2) ===
        if round_num < 3:
            if self._profile == 'STATIC_ROCK':
                return PAPER
            elif self._profile == 'STATIC_PAPER':
                return SCISSORS
            elif self._profile == 'STATIC_SCISSORS':
                return ROCK
            elif self._profile in ('RANDOM', 'META', 'STRONG'):
                return random.choice(MOVES)
            else:
                return [PAPER, SCISSORS, ROCK][round_num % 3]

        # === Get expert predictions ===
        predictions = []
        for i, expert in enumerate(self._experts):
            pred = expert.predict_opponent(my_moves, opp_moves)
            self._last_predictions[i] = pred
            predictions.append(pred)

        # === Weighted vote: vote for COUNTER(predicted_opp_move) ===
        vote_weights = [0.0, 0.0, 0.0]
        for i in range(self._n_experts):
            if predictions[i] is not None:
                move_to_play = COUNTER[predictions[i]]
                vote_weights[move_to_play] += self._weights[i]
            else:
                # Abstaining expert spreads weight uniformly
                for m in MOVES:
                    vote_weights[m] += self._weights[i] / 3

        # === Pick move: mostly deterministic, slight noise ===
        total = sum(vote_weights)
        if total == 0:
            return random.choice(MOVES)

        probs = [v / total for v in vote_weights]

        # Add small exploration noise (5%)
        noise = 0.05
        probs = [p * (1 - noise) + noise / 3 for p in probs]

        # Sample
        r = random.random()
        cumulative = 0.0
        for m in MOVES:
            cumulative += probs[m]
            if r <= cumulative:
                return m
        return MOVES[-1]


# ==========================================================================
# Standalone testing
# ==========================================================================

if __name__ == "__main__":
    def run_test(name, opp_fn, opp_name="Unknown", n_rounds=100, trials=10):
        total_w, total_l, total_d = 0, 0, 0
        for _ in range(trials):
            bot = CompetitionBot()
            history = []
            w, l, d = 0, 0, 0
            for rnd in range(n_rounds):
                move = bot.make_move(history, opp_name, [])
                opp_move = opp_fn(history, rnd)
                history.append([move, opp_move])
                if BEATS[move] == opp_move:
                    w += 1
                elif BEATS[opp_move] == move:
                    l += 1
                else:
                    d += 1
            total_w += w
            total_l += l
            total_d += d
        avg_w = total_w / trials
        avg_l = total_l / trials
        avg_d = total_d / trials
        print(f"vs {name:20s} ({n_rounds}r × {trials}t): {avg_w:.1f}W - {avg_l:.1f}L - {avg_d:.1f}D | "
              f"Win%={avg_w/(n_rounds)*100:.1f}%")

    # Always Rock
    run_test("Always Rock", lambda h, r: ROCK, "AlwaysRock")

    # Cycle R→P→S
    run_test("Cycle R→P→S", lambda h, r: [ROCK, PAPER, SCISSORS][r % 3], "Cycler")

    # Random
    run_test("Pure Random", lambda h, r: random.choice(MOVES), "RandomBot")

    # Mirror (copy last move)
    run_test("Mirror/Copycat",
             lambda h, r: h[-1][0] if h else ROCK,
             "Copycat")

    # Frequency Counter
    def freq_counter(h, r):
        if not h:
            return ROCK
        opp = [x[0] for x in h]  # opponent sees our moves as their opponent
        counts = Counter(opp)
        predicted = counts.most_common(1)[0][0]
        return COUNTER[predicted]
    run_test("Freq Counter", freq_counter, "FrequencyBot")

    # Win-Stay-Lose-Shift
    def wsls(h, r):
        if not h:
            return ROCK
        my_last = h[-1][1]  # WSLS's last move
        opp_last = h[-1][0]  # WSLS's opponent (us)
        if BEATS[my_last] == opp_last:  # WSLS won
            return my_last
        else:
            return COUNTER[my_last]
    run_test("Win-Stay-Lose-Shift", wsls, "WSLS")
