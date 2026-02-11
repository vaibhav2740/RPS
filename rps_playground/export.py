"""Export match results to JSON or CSV."""

import json
import csv
from pathlib import Path
from .engine import MatchResult
from .stats import compute_leaderboard, head_to_head_matrix


def export_json(results: list[MatchResult], path: str):
    """Export results to a JSON file."""
    leaderboard = compute_leaderboard(results)
    h2h = head_to_head_matrix(results)

    data = {
        "matches": [r.to_dict() for r in results],
        "leaderboard": [e.to_dict() for e in leaderboard],
        "head_to_head_matrix": h2h,
    }

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  ✓ Results exported to {out}")


def export_csv(results: list[MatchResult], path: str):
    """Export leaderboard to a CSV file."""
    leaderboard = compute_leaderboard(results)

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "rank", "name", "score", "matches_played",
        "match_wins", "match_losses", "match_draws",
        "total_round_wins", "total_round_losses", "total_round_draws",
        "round_win_pct", "avg_score_per_match",
    ]

    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, entry in enumerate(leaderboard, 1):
            row = entry.to_dict()
            row["rank"] = i
            writer.writerow(row)
    print(f"  ✓ Leaderboard exported to {out}")
