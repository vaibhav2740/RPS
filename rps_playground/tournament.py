"""Tournament modes: head-to-head, one-vs-all, full round-robin.

Supports parallel execution via ProcessPoolExecutor for multi-core speedup.
Supports optional on_match_done callback for live progress tracking.
"""

import os
from typing import Optional, Callable
from concurrent.futures import ProcessPoolExecutor, as_completed

from .engine import run_match, MatchResult
from .algorithms import Algorithm, get_all_algorithms, get_algorithm_by_name


# ---------------------------------------------------------------------------
# Worker function for parallel execution (must be top-level for pickling)
# ---------------------------------------------------------------------------

def _run_match_worker(
    algo_a_name: str,
    algo_b_name: str,
    rounds: int,
    seed: Optional[int],
    record_moves: bool = False,
) -> MatchResult:
    """Run a single match in a worker process.

    Creates fresh algorithm instances inside the worker to avoid pickling
    complex stateful objects across process boundaries.
    """
    algo_a = get_algorithm_by_name(algo_a_name)
    algo_b = get_algorithm_by_name(algo_b_name)
    return run_match(algo_a, algo_b, rounds=rounds, seed=seed, record_moves=record_moves)


# ---------------------------------------------------------------------------
# Tournament modes
# ---------------------------------------------------------------------------

def head_to_head(
    algo_a: Algorithm,
    algo_b: Algorithm,
    rounds: int = 1000,
    seed: Optional[int] = None,
) -> MatchResult:
    """Mode 1: Run a single match between two algorithms."""
    return run_match(algo_a, algo_b, rounds=rounds, seed=seed)


def one_vs_all(
    custom_algo: Algorithm,
    pool: Optional[list[Algorithm]] = None,
    rounds: int = 1000,
    seed: Optional[int] = None,
    parallel: bool = True,
    on_match_done: Optional[Callable[[int, int, MatchResult], None]] = None,
) -> list[MatchResult]:
    """Mode 2: Run a custom algorithm against every algorithm in the pool.

    Args:
        parallel: If True, run matches across multiple CPU cores.
        on_match_done: Optional callback(completed, total, result) called
                       after each match finishes. Used for progress tracking.
    """
    if pool is None:
        pool = get_all_algorithms()

    # Build list of (algo_a_name, algo_b_name, rounds, match_seed)
    jobs = []
    for i, opponent in enumerate(pool):
        match_seed = (seed * 1000 + i) if seed is not None else None
        jobs.append((custom_algo.name, opponent.name, rounds, match_seed))

    if parallel and len(jobs) > 1:
        return _run_parallel(jobs, record_moves=False, on_match_done=on_match_done)
    else:
        # Sequential fallback
        results = []
        for i, (algo_a_name, algo_b_name, rds, ms) in enumerate(jobs):
            result = _run_match_worker(algo_a_name, algo_b_name, rds, ms, record_moves=False)
            results.append(result)
            if on_match_done:
                on_match_done(i + 1, len(jobs), result)
        return results


def round_robin(
    algos: Optional[list[Algorithm]] = None,
    rounds: int = 1000,
    seed: Optional[int] = None,
    parallel: bool = True,
    on_match_done: Optional[Callable[[int, int, MatchResult], None]] = None,
) -> list[MatchResult]:
    """Mode 3: Full round-robin — every algorithm plays every other algorithm.

    Args:
        parallel: If True, run matches across multiple CPU cores.
        on_match_done: Optional callback(completed, total, result) called
                       after each match finishes. Used for progress tracking.
    """
    if algos is None:
        algos = get_all_algorithms()

    # Build list of all match jobs
    jobs = []
    match_idx = 0
    for i in range(len(algos)):
        for j in range(i + 1, len(algos)):
            match_seed = (seed * 10000 + match_idx) if seed is not None else None
            jobs.append((algos[i].name, algos[j].name, rounds, match_seed))
            match_idx += 1

    if parallel and len(jobs) > 1:
        return _run_parallel(jobs, record_moves=False, on_match_done=on_match_done)
    else:
        # Sequential fallback
        results = []
        for i, (algo_a_name, algo_b_name, rds, ms) in enumerate(jobs):
            result = _run_match_worker(algo_a_name, algo_b_name, rds, ms, record_moves=False)
            results.append(result)
            if on_match_done:
                on_match_done(i + 1, len(jobs), result)
        return results


# ---------------------------------------------------------------------------
# Parallel execution helper
# ---------------------------------------------------------------------------

def _run_parallel(
    jobs: list[tuple[str, str, int, Optional[int]]],
    record_moves: bool = False,
    on_match_done: Optional[Callable[[int, int, MatchResult], None]] = None,
) -> list[MatchResult]:
    """Run a batch of matches in parallel using ProcessPoolExecutor.

    Jobs are submitted preserving order. Uses all available CPU cores.
    Calls on_match_done(completed_count, total, result) as each future
    completes, enabling real-time progress tracking.
    """
    max_workers = min(os.cpu_count() or 4, len(jobs))
    total = len(jobs)

    # Use a dict to preserve original job ordering
    results: list[Optional[MatchResult]] = [None] * total
    completed = 0

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {}
        for idx, (a_name, b_name, rounds, mseed) in enumerate(jobs):
            future = executor.submit(
                _run_match_worker, a_name, b_name, rounds, mseed, record_moves
            )
            future_to_idx[future] = idx

        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            result = future.result()
            results[idx] = result
            completed += 1

            if on_match_done:
                on_match_done(completed, total, result)

    return results  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Competition tournament (sequential, with metadata)
# ---------------------------------------------------------------------------

def competition_round_robin(
    algos: Optional[list[Algorithm]] = None,
    rounds: int = 100,
    seed: Optional[int] = None,
    on_match_done: Optional[Callable[[int, int, MatchResult], None]] = None,
) -> list[MatchResult]:
    """Competition round-robin — every algo plays every other with metadata.

    Runs SEQUENTIALLY so that each algorithm can see its opponent's
    prior tournament results before the match begins. After each match,
    the tournament record is updated for both participants.

    Each algorithm receives via set_match_context():
        - opponent_name: str
        - opponent_history: list of {opponent, result, score, rounds}

    Args:
        on_match_done: Optional callback(completed, total, result)
    """
    from .engine import run_match_with_context

    if algos is None:
        algos = get_all_algorithms()

    # Tournament record: algo_name → list of past match results
    records: dict[str, list[dict]] = {algo.name: [] for algo in algos}

    # Build job list
    jobs = []
    match_idx = 0
    for i in range(len(algos)):
        for j in range(i + 1, len(algos)):
            match_seed = (seed * 10000 + match_idx) if seed is not None else None
            jobs.append((i, j, rounds, match_seed))
            match_idx += 1

    total = len(jobs)
    results = []

    for completed_idx, (i, j, rds, mseed) in enumerate(jobs):
        algo_a = algos[i].__class__()
        algo_b = algos[j].__class__()

        # Pass tournament context
        a_opp_history = records.get(algo_b.name, [])
        b_opp_history = records.get(algo_a.name, [])

        result = run_match_with_context(
            algo_a, algo_b,
            rounds=rds, seed=mseed,
            record_moves=False,
            a_tournament_history=a_opp_history,
            b_tournament_history=b_opp_history,
        )
        results.append(result)

        # Update tournament records for both participants
        if result.a_wins > result.b_wins:
            a_result, b_result = "win", "loss"
        elif result.b_wins > result.a_wins:
            a_result, b_result = "loss", "win"
        else:
            a_result, b_result = "draw", "draw"

        records[result.algo_a_name].append({
            "opponent": result.algo_b_name,
            "result": a_result,
            "score": f"{result.a_wins}-{result.b_wins}",
            "rounds": rds,
        })
        records[result.algo_b_name].append({
            "opponent": result.algo_a_name,
            "result": b_result,
            "score": f"{result.b_wins}-{result.a_wins}",
            "rounds": rds,
        })

        if on_match_done:
            on_match_done(completed_idx + 1, total, result)

    return results
