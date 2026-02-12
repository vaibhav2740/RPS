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


