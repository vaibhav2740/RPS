"""CLI entry point for the RPS Algorithm Testing Playground."""

import argparse
import sys

from .algorithms import get_algorithm_by_name, get_all_algorithms, ALL_ALGORITHM_CLASSES
from .tournament import head_to_head, one_vs_all, round_robin
from .stats import (
    compute_leaderboard,
    head_to_head_matrix,
    print_match_summary,
    print_leaderboard,
    print_h2h_matrix,
)
from .export import export_json, export_csv


def list_algorithms():
    """Print all available algorithm names."""
    print("\nAvailable Algorithms:")
    print("-" * 40)
    for i, cls in enumerate(ALL_ALGORITHM_CLASSES, 1):
        print(f"  {i:>2d}. {cls.name}")
    print()


def cmd_head_to_head(args):
    """Run head-to-head mode."""
    algo_a = get_algorithm_by_name(args.algo_a)
    algo_b = get_algorithm_by_name(args.algo_b)
    print(f"\n⚔️  Mode 1: Head-to-Head")
    print(f"  {algo_a.name} vs {algo_b.name}  |  {args.rounds} rounds"
          + (f"  |  seed={args.seed}" if args.seed is not None else ""))

    result = head_to_head(algo_a, algo_b, rounds=args.rounds, seed=args.seed)
    print_match_summary(result)

    if args.export and args.output:
        _export(args, [result])


def cmd_one_vs_all(args):
    """Run one-vs-all mode."""
    custom = get_algorithm_by_name(args.algo)
    pool = get_all_algorithms()
    print(f"\n🤖 Mode 2: One vs All")
    print(f"  {custom.name} vs {len(pool)} algorithms  |  {args.rounds} rounds"
          + (f"  |  seed={args.seed}" if args.seed is not None else ""))
    print()

    results = one_vs_all(custom, pool, rounds=args.rounds, seed=args.seed)

    # Print individual match summaries
    for r in results:
        print(f"  {r.algo_a_name:>22s} vs {r.algo_b_name:<22s}  →  "
              f"W:{r.a_wins:>4d}  L:{r.b_wins:>4d}  D:{r.draws:>4d}  "
              f"({'WIN' if r.a_wins > r.b_wins else 'LOSS' if r.b_wins > r.a_wins else 'DRAW'})")

    leaderboard = compute_leaderboard(results)
    print_leaderboard(leaderboard)

    if args.export and args.output:
        _export(args, results)


def cmd_tournament(args):
    """Run full round-robin tournament."""
    algos = get_all_algorithms()
    total_matches = len(algos) * (len(algos) - 1) // 2
    print(f"\n🏆 Mode 3: Full Round-Robin Tournament")
    print(f"  {len(algos)} algorithms  |  {total_matches} matches  |  {args.rounds} rounds each"
          + (f"  |  seed={args.seed}" if args.seed is not None else ""))
    print(f"  Running...", end="", flush=True)

    results = round_robin(algos, rounds=args.rounds, seed=args.seed)
    print(f" done! ({len(results)} matches played)")

    leaderboard = compute_leaderboard(results)
    print_leaderboard(leaderboard)

    # Head-to-head matrix
    names = [e.name for e in leaderboard]
    matrix = head_to_head_matrix(results)
    print_h2h_matrix(matrix, names)

    # Per-algorithm breakdown
    print_per_algorithm_breakdown(results, leaderboard)

    if args.export and args.output:
        _export(args, results)


def print_per_algorithm_breakdown(results: list, leaderboard: list):
    """Print detailed performance breakdown for each algorithm."""
    print("=" * 60)
    print("  Per-Algorithm Performance Breakdown")
    print("=" * 60)

    for entry in leaderboard:
        total_rounds = entry.total_wins + entry.total_losses + entry.total_draws
        print(f"\n  ▸ {entry.name}")
        print(f"    Matches: {entry.matches_played} played  "
              f"({entry.match_wins}W / {entry.match_losses}L / {entry.match_draws}D)")
        print(f"    Rounds:  {total_rounds} played  "
              f"({entry.total_wins}W / {entry.total_losses}L / {entry.total_draws}D)")
        print(f"    Elo: {entry.elo:.1f}  |  Round Win%: {entry.win_pct:.1f}%  |  Score: {entry.score}")
    print()


def _export(args, results):
    """Handle export based on CLI args."""
    fmt = args.export.lower()
    if fmt == "json":
        export_json(results, args.output)
    elif fmt == "csv":
        export_csv(results, args.output)
    else:
        print(f"  ✗ Unknown export format: {fmt}. Use 'json' or 'csv'.")


def main():
    parser = argparse.ArgumentParser(
        prog="rps_playground",
        description="🎮 Rock-Paper-Scissors Algorithm Testing Playground",
    )
    parser.add_argument("--list", action="store_true", help="List all available algorithms")

    subparsers = parser.add_subparsers(dest="command")

    # head-to-head
    h2h = subparsers.add_parser("head-to-head", help="Mode 1: Algo A vs Algo B")
    h2h.add_argument("--algo-a", required=True, help="Name of algorithm A")
    h2h.add_argument("--algo-b", required=True, help="Name of algorithm B")
    h2h.add_argument("--rounds", type=int, default=1000, help="Number of rounds (default: 1000)")
    h2h.add_argument("--seed", type=int, default=None, help="RNG seed for reproducibility")
    h2h.add_argument("--export", choices=["json", "csv"], help="Export format")
    h2h.add_argument("--output", help="Export file path")

    # one-vs-all
    ova = subparsers.add_parser("one-vs-all", help="Mode 2: Custom algo vs all 20 bots")
    ova.add_argument("--algo", required=True, help="Name of the custom algorithm")
    ova.add_argument("--rounds", type=int, default=1000, help="Number of rounds (default: 1000)")
    ova.add_argument("--seed", type=int, default=None, help="RNG seed for reproducibility")
    ova.add_argument("--export", choices=["json", "csv"], help="Export format")
    ova.add_argument("--output", help="Export file path")

    # tournament
    trn = subparsers.add_parser("tournament", help="Mode 3: Full round-robin tournament")
    trn.add_argument("--rounds", type=int, default=1000, help="Number of rounds per match (default: 1000)")
    trn.add_argument("--seed", type=int, default=None, help="RNG seed for reproducibility")
    trn.add_argument("--export", choices=["json", "csv"], help="Export format")
    trn.add_argument("--output", help="Export file path")

    args = parser.parse_args()

    if args.list:
        list_algorithms()
        return

    if args.command == "head-to-head":
        cmd_head_to_head(args)
    elif args.command == "one-vs-all":
        cmd_one_vs_all(args)
    elif args.command == "tournament":
        cmd_tournament(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
