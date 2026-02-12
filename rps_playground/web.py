"""Flask web server for the RPS Algorithm Testing Playground."""

import json
import time
import queue
import threading
from flask import Flask, render_template, request, jsonify, Response

from .algorithms import get_algorithm_by_name, get_all_algorithms, ALL_ALGORITHM_CLASSES
from .tournament import head_to_head, one_vs_all, round_robin, competition_round_robin
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


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/algorithms")
def api_algorithms():
    return jsonify([cls.name for cls in ALL_ALGORITHM_CLASSES])


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
