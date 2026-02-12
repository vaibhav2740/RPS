class TheMirrorWorld(Algorithm):
    """Recursive Simulation (Level-k thinking).

    Simulates an opponent model trying to predict US.
    Level 0: Random
    Level 1: Opponent counters my most likely move (Freq/Markov)
    Level 2: I counter Level 1
    Level 3: I counter Opponent countering Level 2

    Dynamically selects best depth based on performance.
    """
    name = "The Mirror World"

    def reset(self):
        self._m2i = {Move.ROCK: 0, Move.PAPER: 1, Move.SCISSORS: 2}
        self._i2m = {0: Move.ROCK, 1: Move.PAPER, 2: Move.SCISSORS}
        self._oh = []
        self._mh = []

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
    """Supercharged Phantom Ensemble with 79 predictors and deeper prediction.

    Architecture (built on Phantom Ensemble's proven design):
    - History matching at depths 1-20 on (opp, my, combined) = 60 predictors
    - Markov chains order 1-4 on (opp, pair, my) = 8 predictors
    - Rotation/anti-rotation = 6 predictors
    - Special (anti-freq, WLD, cycle, LZ-context, decay-freq) = 5 predictors
    Total: 79 predictors with Hedge meta-learning.
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

