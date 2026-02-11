"""Tournament modes: head-to-head, one-vs-all, full round-robin."""

from typing import Optional
from .engine import run_match, MatchResult
from .algorithms import Algorithm, get_all_algorithms


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
) -> list[MatchResult]:
    """Mode 2: Run a custom algorithm against every algorithm in the pool."""
    if pool is None:
        pool = get_all_algorithms()
    results = []
    for i, opponent in enumerate(pool):
        # Derive a unique seed per match for reproducibility
        match_seed = (seed * 1000 + i) if seed is not None else None
        result = run_match(custom_algo, opponent, rounds=rounds, seed=match_seed)
        results.append(result)
    return results


def round_robin(
    algos: Optional[list[Algorithm]] = None,
    rounds: int = 1000,
    seed: Optional[int] = None,
) -> list[MatchResult]:
    """Mode 3: Full round-robin — every algorithm plays every other algorithm."""
    if algos is None:
        algos = get_all_algorithms()
    results = []
    match_idx = 0
    for i in range(len(algos)):
        for j in range(i + 1, len(algos)):
            match_seed = (seed * 10000 + match_idx) if seed is not None else None
            result = run_match(algos[i], algos[j], rounds=rounds, seed=match_seed)
            results.append(result)
            match_idx += 1
    return results
