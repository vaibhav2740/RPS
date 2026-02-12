"""Flask web server for the RPS Algorithm Testing Playground."""

import json
import time
import uuid
import random
import queue
import threading
from flask import Flask, render_template, request, jsonify, Response

from .algorithms import get_algorithm_by_name, get_all_algorithms, ALL_ALGORITHM_CLASSES
from .tournament import head_to_head, one_vs_all, round_robin, competition_round_robin
from .engine import Move, determine_winner
from .stats import (
    compute_leaderboard,
    head_to_head_matrix,
    compute_elo_ratings,
    ELO_INITIAL,
    ELO_K_FACTOR,
    _elo_expected,
    _elo_update,
)

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Arena session storage (in-memory, single-player local tool)
# ---------------------------------------------------------------------------
_arena_sessions: dict[str, dict] = {}
_ARENA_MAX_SESSIONS = 100
_MOVE_MAP = {"rock": Move.ROCK, "paper": Move.PAPER, "scissors": Move.SCISSORS}
_MOVE_EMOJI = {Move.ROCK: "✊", Move.PAPER: "✋", Move.SCISSORS: "✌️"}

# ---------------------------------------------------------------------------
# Bot Categories
# ---------------------------------------------------------------------------
BOT_CATEGORIES = {
    # Baseline
    "Always Rock": "Baseline", "Always Paper": "Baseline", "Always Scissors": "Baseline",
    "Pure Random": "Baseline", "Cycle": "Baseline", "Spiral": "Baseline",
    "Persistent Random": "Baseline", "Stuttering Cycle": "Baseline",
    "Double Cycle": "Baseline", "Triple Cycle": "Baseline",
    "Alternating Mirror": "Baseline", "Mirror Opponent": "Baseline",
    "Tit-for-Tat": "Baseline", "Anti-Tit-for-Tat": "Baseline", "Echo": "Baseline",

    # Reactive
    "Copycat": "Reactive", "Last-Move Counter": "Reactive", "Second Guess": "Reactive",
    "Reverse Psychologist": "Reactive", "Immediate Counter": "Reactive",
    "Heuristic": "Reactive",
    "Delayed Counter": "Reactive", "Win-Reflect Lose-Random": "Reactive",
    "Win-Shift Lose-Stay": "Reactive", "Win-Stay-Lose-Shift": "Reactive",
    "Grudge Holder": "Reactive", "Stubborn": "Reactive", "Fickle": "Reactive",
    "Desperate": "Reactive", "Traitor Mirror": "Reactive", "Delayed Mirror": "Reactive",
    "Anti-Strategy Detector": "Reactive", "Mirror Breaker": "Reactive",

    # Frequency
    "Frequency Analyzer": "Frequency", "Decay Analyzer": "Frequency",
    "Windowed Frequency": "Frequency", "Weighted Random": "Frequency",
    "Biased Random": "Frequency", "Anti-Frequency": "Frequency",
    "Recent Frequency": "Frequency", "Long-Term Frequency": "Frequency",
    "Mode Player": "Frequency", "Anti-Mode Player": "Frequency",
    "Weighted Mirror": "Frequency", "Exponential Backoff": "Frequency",
    "Noise Strategy": "Frequency",

    # Pattern
    "Markov Predictor": "Pattern", "Pattern Matcher": "Pattern",
    "N-Gram Predictor": "Pattern", "Historian": "Pattern",
    "De Bruijn Walker": "Pattern", "Pattern Breaker": "Pattern",
    "Sequence Learner": "Pattern", "Transition Matrix": "Pattern",
    "Dual Markov": "Pattern", "Triple Markov": "Pattern",
    "Context Tree": "Pattern", "LZ78 Predictor": "Pattern",
    "Variable Markov": "Pattern", "Bayesian Transition": "Pattern",
    "Lempel-Ziv Predictor": "Pattern", "Markov Generator": "Pattern",
    "Deep Historian": "Pattern", "Adaptive N-Gram": "Pattern",
    "Pattern Detector": "Pattern", "Gene Sequencer": "Pattern",
    "Zodiac": "Pattern", "History Matcher": "Pattern",

    # Psychological
    "Gambler's Fallacy": "Psychological", "Reluctant Gambler": "Psychological",
    "Punisher": "Psychological", "Forgiver": "Psychological",
    "Naive Learner": "Psychological", "Overconfident": "Psychological",
    "Paranoid": "Psychological", "Psychotic": "Psychological",
    "Hot Streak": "Psychological", "Chameleon": "Psychological",
    "Poison Pill": "Psychological", "Double Bluff": "Psychological",
    "Opponent Persona": "Psychological", "Stubborn Loser": "Psychological",
    "Trojan Horse": "Psychological", "Intentional Loser": "Psychological",
    "Frequency Disruptor": "Psychological",

    # Mathematical
    "PiBot": "Mathematical", "Golden Ratio": "Mathematical",
    "Fibonacci": "Mathematical", "Fibonacci Player": "Mathematical",
    "Prime Hunter": "Mathematical", "Chaos Strategy": "Mathematical",
    "Entropy Guardian": "Mathematical", "Pseudo RNG": "Mathematical",
    "Linear Congruential": "Mathematical", "Xorshift": "Mathematical",
    "Mersenne Twister": "Mathematical", "Fourier Predictor": "Mathematical",
    "Eigenvalue Predictor": "Mathematical", "Chaos Engine": "Mathematical",
    "Ackermann": "Mathematical", "Quantum Collapse": "Mathematical",
    "Sound Wave": "Mathematical",

    # Ensemble & Meta
    "Meta-Predictor": "Ensemble", "Majority Rule": "Ensemble",
    "Adaptive Hybrid": "Ensemble", "The Hydra": "Ensemble",
    "Ensemble Voter": "Ensemble", "Weighted Ensemble": "Ensemble",
    "Dynamic Ensemble": "Ensemble", "Random Forest": "Ensemble",
    "Boosted Tree": "Ensemble", "Genetic Algorithm": "Ensemble",
    "Evolutionary Strategy": "Ensemble", "Expert Mixer": "Ensemble",
    "Switching Experts": "Ensemble", "Meta-Learner": "Ensemble",
    "Bayes Ensemble": "Ensemble", "Geometry Ensemble": "Ensemble",
    "Phantom Ensemble": "Ensemble", "The Collective": "Ensemble",
    "The Omniscient": "Ensemble", "Super Omniscient": "Ensemble",
    "The Singularity": "Ensemble", "Geometry Bot": "Ensemble",
    "Iocaine Powder": "Ensemble", "Iocaine Powder Plus": "Ensemble",
    "Mixture Model": "Ensemble", "MetaLearner": "Ensemble",

    # Adaptive
    "Thompson Sampling": "Adaptive", "UCB1": "Adaptive",
    "EXP3": "Adaptive", "Epsilon-Greedy": "Adaptive",
    "Softmax Explorer": "Adaptive", "Bayesian Bandit": "Adaptive",
    "Reinforcement Learner": "Adaptive", "Q-Learning (Simple)": "Adaptive",
    "Multi-Armed Bandit": "Adaptive", "Monte Carlo Predictor": "Adaptive",
    "Max Entropy Predictor": "Adaptive", "The Usurper": "Adaptive",
    "Regret Minimizer": "Adaptive", "Hidden Markov Oracle": "Adaptive",
    "Genetic Strategist": "Adaptive", "PID Controller": "Adaptive",
    "Level-k Reasoner": "Adaptive", "UCB-NGram Fusion": "Adaptive",
    "Dynamic Mixture": "Adaptive", "Hierarchical Bayesian": "Adaptive",
    "Self-Model Detector": "Adaptive", "Stock Broker": "Adaptive",
    "Compression Bot": "Adaptive", "Equilibrium Breaker": "Adaptive",
    "Neuro Evo": "Adaptive", "Nash Stabilizer": "Adaptive",
    "Sliding Window Vote": "Adaptive", "Double Agent": "Adaptive",
    "Counter Strike": "Adaptive", "Tournament Scout": "Adaptive",
    "Neural Prophet": "Adaptive", "LSTM Predictor": "Adaptive",
    "Decision Cascader": "Adaptive", "The Time Traveler": "Adaptive",
    "The Mirror World": "Adaptive", "The Doppelganger": "Adaptive",
    "The Void": "Adaptive", "The Architect": "Adaptive",
    "The Black Hole": "Adaptive", "Phoenix": "Adaptive",
    "Contrarian": "Adaptive", "Bayesian Predictor": "Adaptive",
    "Gradient Learner": "Adaptive", "Gradient Learner v5": "Adaptive",
    "Phase Shifter": "Adaptive", "Q-Learner": "Adaptive",
    "Q-Learner v5": "Adaptive", "Shapeshifter": "Adaptive",
    "Sleeper Agent": "Adaptive", "Thompson Sampler v5": "Adaptive",
    "UCB Explorer": "Adaptive", "UCB Explorer v5": "Adaptive"
}


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/algorithms")
def api_algorithms():
    """Return list of available algorithms with their categories and docstrings."""
    algos = []
    for cls in ALL_ALGORITHM_CLASSES:
        cat = BOT_CATEGORIES.get(cls.name, "Other")
        
        # Parse docstring
        import re
        doc = cls.__doc__ or ""
        parts = re.split(r'\n\s*\n', doc.strip(), 1)
        summary = parts[0].replace("\n", " ").strip() if parts else "No description available."
        details = parts[1].strip() if len(parts) > 1 else ""
        
        algos.append({
            "name": cls.name, 
            "category": cat,
            "summary": summary,
            "details": details
        })
    
    # Sort by category then name
    algos.sort(key=lambda x: (x["category"], x["name"]))
    return jsonify(algos)


# ---------------------------------------------------------------------------
# Arena endpoints — Human vs AI, round-by-round
# ---------------------------------------------------------------------------

@app.route("/api/arena/start", methods=["POST"])
def api_arena_start():
    """Create a new arena session with the chosen bot."""
    # Evict oldest sessions if at capacity
    while len(_arena_sessions) >= _ARENA_MAX_SESSIONS:
        oldest_key = min(_arena_sessions, key=lambda k: _arena_sessions[k]["created"])
        del _arena_sessions[oldest_key]

    data = request.get_json()
    bot_name = data.get("bot", "Always Rock")
    bot = get_algorithm_by_name(bot_name)
    bot.rng = random.Random()
    bot.reset()

    sid = uuid.uuid4().hex[:12]
    _arena_sessions[sid] = {
        "bot": bot,
        "bot_name": bot_name,
        "human_history": [],   # list[Move]
        "bot_history": [],     # list[Move]
        "rounds": [],          # [{human_move, bot_move, result}]
        "wins": 0,
        "losses": 0,
        "draws": 0,
        "streak": 0,           # positive = win streak, negative = loss streak
        "created": time.time(),
    }
    return jsonify({"session_id": sid, "bot_name": bot_name})


@app.route("/api/arena/play", methods=["POST"])
def api_arena_play():
    """Play one round: human submits a move, bot responds."""
    data = request.get_json()
    sid = data.get("session_id", "")
    human_move_str = data.get("move", "").lower()

    session = _arena_sessions.get(sid)
    if not session:
        return jsonify({"error": "Session not found. Start a new game."}), 404

    if human_move_str not in _MOVE_MAP:
        return jsonify({"error": f"Invalid move: {human_move_str}"}), 400

    human_move = _MOVE_MAP[human_move_str]
    round_num = len(session["human_history"])

    # Bot chooses (bot sees: its own history as "my_history", human history as "opp_history")
    bot_move = session["bot"].choose(
        round_num,
        session["bot_history"],    # bot's own past moves
        session["human_history"],  # opponent (human) past moves
    )

    # Determine outcome from human's perspective
    outcome = determine_winner(human_move, bot_move)  # 1=human wins, -1=bot wins, 0=draw

    # Update histories
    session["human_history"].append(human_move)
    session["bot_history"].append(bot_move)

    # Update score
    if outcome == 1:
        result_str = "win"
        session["wins"] += 1
        session["streak"] = max(session["streak"], 0) + 1
    elif outcome == -1:
        result_str = "loss"
        session["losses"] += 1
        session["streak"] = min(session["streak"], 0) - 1
    else:
        result_str = "draw"
        session["draws"] += 1
        session["streak"] = 0

    round_data = {
        "round": round_num + 1,
        "human_move": human_move.value,
        "bot_move": bot_move.value,
        "human_emoji": _MOVE_EMOJI[human_move],
        "bot_emoji": _MOVE_EMOJI[bot_move],
        "result": result_str,
    }
    session["rounds"].append(round_data)

    return jsonify({
        **round_data,
        "score": {
            "wins": session["wins"],
            "losses": session["losses"],
            "draws": session["draws"],
        },
        "streak": session["streak"],
        "total_rounds": round_num + 1,
    })


@app.route("/api/arena/reset", methods=["POST"])
def api_arena_reset():
    """Reset an arena session — same bot, fresh state."""
    data = request.get_json()
    sid = data.get("session_id", "")

    session = _arena_sessions.get(sid)
    if not session:
        return jsonify({"error": "Session not found."}), 404

    # Re-instantiate the bot
    bot = get_algorithm_by_name(session["bot_name"])
    bot.rng = random.Random()
    bot.reset()

    session["bot"] = bot
    session["human_history"] = []
    session["bot_history"] = []
    session["rounds"] = []
    session["wins"] = 0
    session["losses"] = 0
    session["draws"] = 0
    session["streak"] = 0
    session["created"] = time.time()

    return jsonify({"status": "reset", "bot_name": session["bot_name"]})


@app.route("/api/head-to-head", methods=["POST"])
def api_head_to_head():
    data = request.get_json()
    algo_a_name = data.get("algo_a", "Always Rock")
    algo_b_name = data.get("algo_b", "Always Scissors")
    rounds = data.get("rounds", 1000)
    seed = data.get("seed", None)

    algo_a = get_algorithm_by_name(algo_a_name)
    algo_b = get_algorithm_by_name(algo_b_name)

    result = head_to_head(algo_a, algo_b, rounds=rounds, seed=seed)

    # Compute Elo delta for this match
    ra, rb = ELO_INITIAL, ELO_INITIAL
    ea_exp = _elo_expected(ra, rb)
    eb_exp = _elo_expected(rb, ra)
    if result.a_wins > result.b_wins:
        sa, sb = 1.0, 0.0
    elif result.b_wins > result.a_wins:
        sa, sb = 0.0, 1.0
    else:
        sa, sb = 0.5, 0.5
    new_ra = _elo_update(ra, ea_exp, sa)
    new_rb = _elo_update(rb, eb_exp, sb)

    return jsonify({
        "match": result.to_dict(),
        "elo": {
            "a": {"rating": round(new_ra, 1), "delta": round(new_ra - ra, 1)},
            "b": {"rating": round(new_rb, 1), "delta": round(new_rb - rb, 1)},
        },
        "winner": (
            result.algo_a_name if result.a_wins > result.b_wins
            else result.algo_b_name if result.b_wins > result.a_wins
            else "DRAW"
        ),
    })


# ---------------------------------------------------------------------------
# Original POST endpoints (kept for backward compatibility)
# ---------------------------------------------------------------------------

@app.route("/api/tournament", methods=["POST"])
def api_tournament():
    data = request.get_json()
    rounds = data.get("rounds", 1000)
    seed = data.get("seed", None)

    algos = get_all_algorithms()
    results = round_robin(algos, rounds=rounds, seed=seed, parallel=True)
    leaderboard = compute_leaderboard(results)
    matrix = head_to_head_matrix(results)

    return jsonify({
        "leaderboard": [e.to_dict() for e in leaderboard],
        "matrix": matrix,
        "total_matches": len(results),
    })


@app.route("/api/one-vs-all", methods=["POST"])
def api_one_vs_all():
    data = request.get_json()
    algo_name = data.get("algo", "Always Rock")
    rounds = data.get("rounds", 1000)
    seed = data.get("seed", None)

    custom = get_algorithm_by_name(algo_name)
    pool = get_all_algorithms()
    results = one_vs_all(custom, pool, rounds=rounds, seed=seed, parallel=True)
    leaderboard = compute_leaderboard(results)

    matchups = []
    for r in results:
        if r.a_wins > r.b_wins:
            outcome = "WIN"
        elif r.b_wins > r.a_wins:
            outcome = "LOSS"
        else:
            outcome = "DRAW"
        matchups.append({
            "opponent": r.algo_b_name,
            "wins": r.a_wins,
            "losses": r.b_wins,
            "draws": r.draws,
            "outcome": outcome,
        })

    return jsonify({
        "algo": algo_name,
        "leaderboard": [e.to_dict() for e in leaderboard],
        "matchups": matchups,
    })


# ---------------------------------------------------------------------------
# SSE streaming endpoints for live progress tracking
# ---------------------------------------------------------------------------

def _sse_event(data: dict, event: str = "message") -> str:
    """Format a Server-Sent Event string."""
    payload = json.dumps(data)
    return f"event: {event}\ndata: {payload}\n\n"


@app.route("/api/tournament/stream")
def api_tournament_stream():
    """SSE endpoint that streams tournament progress."""
    rounds = request.args.get("rounds", 1000, type=int)
    seed = request.args.get("seed", None, type=int)

    def generate():
        progress_queue = queue.Queue()
        start_time = time.time()
        final_results = [None]  # mutable container for thread result

        def on_match_done(completed, total, result):
            elapsed = time.time() - start_time
            if completed > 0:
                eta = (elapsed / completed) * (total - completed)
            else:
                eta = 0
            # Determine winner of this match
            if result.a_wins > result.b_wins:
                winner = result.algo_a_name
            elif result.b_wins > result.a_wins:
                winner = result.algo_b_name
            else:
                winner = "DRAW"
            progress_queue.put({
                "completed": completed,
                "total": total,
                "match": f"{result.algo_a_name} vs {result.algo_b_name}",
                "winner": winner,
                "elapsed": round(elapsed, 1),
                "eta": round(eta, 1),
                "pct": round(completed / total * 100, 1),
            })

        def run_tournament():
            algos = get_all_algorithms()
            results = round_robin(
                algos, rounds=rounds, seed=seed,
                parallel=True, on_match_done=on_match_done,
            )
            final_results[0] = results
            progress_queue.put("DONE")

        thread = threading.Thread(target=run_tournament, daemon=True)
        thread.start()

        while True:
            try:
                item = progress_queue.get(timeout=60)
            except queue.Empty:
                # Send a keepalive
                yield ": keepalive\n\n"
                continue

            if item == "DONE":
                # Compute final results
                results = final_results[0]
                leaderboard = compute_leaderboard(results)
                matrix = head_to_head_matrix(results)
                elapsed = round(time.time() - start_time, 1)
                yield _sse_event({
                    "leaderboard": [e.to_dict() for e in leaderboard],
                    "matrix": matrix,
                    "total_matches": len(results),
                    "elapsed": elapsed,
                }, event="done")
                break
            else:
                yield _sse_event(item, event="progress")

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@app.route("/api/one-vs-all/stream")
def api_one_vs_all_stream():
    """SSE endpoint that streams one-vs-all progress."""
    algo_name = request.args.get("algo", "Always Rock")
    rounds = request.args.get("rounds", 1000, type=int)
    seed = request.args.get("seed", None, type=int)

    def generate():
        progress_queue = queue.Queue()
        start_time = time.time()
        final_results = [None]

        def on_match_done(completed, total, result):
            elapsed = time.time() - start_time
            if completed > 0:
                eta = (elapsed / completed) * (total - completed)
            else:
                eta = 0
            if result.a_wins > result.b_wins:
                winner = result.algo_a_name
            elif result.b_wins > result.a_wins:
                winner = result.algo_b_name
            else:
                winner = "DRAW"
            progress_queue.put({
                "completed": completed,
                "total": total,
                "match": f"{result.algo_a_name} vs {result.algo_b_name}",
                "winner": winner,
                "elapsed": round(elapsed, 1),
                "eta": round(eta, 1),
                "pct": round(completed / total * 100, 1),
            })

        def run_ova():
            custom = get_algorithm_by_name(algo_name)
            pool = get_all_algorithms()
            results = one_vs_all(
                custom, pool, rounds=rounds, seed=seed,
                parallel=True, on_match_done=on_match_done,
            )
            final_results[0] = results
            progress_queue.put("DONE")

        thread = threading.Thread(target=run_ova, daemon=True)
        thread.start()

        while True:
            try:
                item = progress_queue.get(timeout=60)
            except queue.Empty:
                yield ": keepalive\n\n"
                continue

            if item == "DONE":
                results = final_results[0]
                leaderboard = compute_leaderboard(results)
                matchups = []
                for r in results:
                    if r.a_wins > r.b_wins:
                        outcome = "WIN"
                    elif r.b_wins > r.a_wins:
                        outcome = "LOSS"
                    else:
                        outcome = "DRAW"
                    matchups.append({
                        "opponent": r.algo_b_name,
                        "wins": r.a_wins,
                        "losses": r.b_wins,
                        "draws": r.draws,
                        "outcome": outcome,
                    })
                elapsed = round(time.time() - start_time, 1)
                yield _sse_event({
                    "algo": algo_name,
                    "leaderboard": [e.to_dict() for e in leaderboard],
                    "matchups": matchups,
                    "elapsed": elapsed,
                }, event="done")
                break
            else:
                yield _sse_event(item, event="progress")

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@app.route("/api/competition/stream")
def api_competition_stream():
    """SSE endpoint for competition tournament (sequential, with metadata)."""
    rounds = request.args.get("rounds", 100, type=int)
    seed = request.args.get("seed", None, type=int)

    def generate():
        progress_queue = queue.Queue()
        start_time = time.time()
        final_results = [None]

        def on_match_done(completed, total, result):
            elapsed = time.time() - start_time
            if completed > 0:
                eta = (elapsed / completed) * (total - completed)
            else:
                eta = 0
            if result.a_wins > result.b_wins:
                winner = result.algo_a_name
            elif result.b_wins > result.a_wins:
                winner = result.algo_b_name
            else:
                winner = "DRAW"
            progress_queue.put({
                "completed": completed,
                "total": total,
                "match": f"{result.algo_a_name} vs {result.algo_b_name}",
                "winner": winner,
                "a_wins": result.a_wins,
                "b_wins": result.b_wins,
                "elapsed": round(elapsed, 1),
                "eta": round(eta, 1),
                "pct": round(completed / total * 100, 1),
            })

        def run_comp():
            algos = get_all_algorithms()
            results = competition_round_robin(
                algos, rounds=rounds, seed=seed,
                on_match_done=on_match_done,
            )
            final_results[0] = results
            progress_queue.put("DONE")

        thread = threading.Thread(target=run_comp, daemon=True)
        thread.start()

        while True:
            try:
                item = progress_queue.get(timeout=60)
            except queue.Empty:
                yield ": keepalive\n\n"
                continue

            if item == "DONE":
                results = final_results[0]
                leaderboard = compute_leaderboard(results)
                matrix = head_to_head_matrix(results)
                elapsed = round(time.time() - start_time, 1)
                yield _sse_event({
                    "leaderboard": [e.to_dict() for e in leaderboard],
                    "matrix": matrix,
                    "total_matches": len(results),
                    "elapsed": elapsed,
                }, event="done")
                break
            else:
                yield _sse_event(item, event="progress")

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


def main():
    print("\n🎮 RPS Playground Web UI")
    print("  → http://localhost:5000\n")
    app.run(debug=True, port=5000)


if __name__ == "__main__":
    main()
