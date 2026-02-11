"""Stats computation and pretty-printing for RPS matches."""

from dataclasses import dataclass, field
from collections import Counter, defaultdict
from .engine import MatchResult

# ---------------------------------------------------------------------------
# Elo Rating System
# ---------------------------------------------------------------------------

ELO_INITIAL = 1500
ELO_K_FACTOR = 32


def _elo_expected(rating_a: float, rating_b: float) -> float:
    """Expected score for player A given both ratings."""
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def _elo_update(rating: float, expected: float, actual: float, k: float = ELO_K_FACTOR) -> float:
    """Update a single Elo rating."""
    return rating + k * (actual - expected)


def compute_elo_ratings(
    results: list[MatchResult],
    initial: float = ELO_INITIAL,
    k: float = ELO_K_FACTOR,
) -> dict[str, float]:
    """Compute Elo ratings by processing match results in order.

    Match outcome mapping:
        win  → actual score = 1.0
        draw → actual score = 0.5
        loss → actual score = 0.0
    """
    ratings: dict[str, float] = defaultdict(lambda: initial)

    for r in results:
        ra = ratings[r.algo_a_name]
        rb = ratings[r.algo_b_name]

        ea = _elo_expected(ra, rb)
        eb = _elo_expected(rb, ra)

        if r.a_wins > r.b_wins:
            sa, sb = 1.0, 0.0
        elif r.b_wins > r.a_wins:
            sa, sb = 0.0, 1.0
        else:
            sa, sb = 0.5, 0.5

        ratings[r.algo_a_name] = _elo_update(ra, ea, sa, k)
        ratings[r.algo_b_name] = _elo_update(rb, eb, sb, k)

    return dict(ratings)


@dataclass
class LeaderboardEntry:
    """Aggregated stats for one algorithm across multiple matches."""
    name: str
    total_wins: int = 0
    total_losses: int = 0
    total_draws: int = 0
    matches_played: int = 0
    elo: float = ELO_INITIAL

    @property
    def score(self) -> int:
        """3 points for a match win, 1 for a draw, 0 for loss."""
        return self.match_wins * 3 + self.match_draws * 1

    @property
    def match_wins(self) -> int:
        """Matches won (more round wins than opponent)."""
        return self._match_wins

    @property
    def match_losses(self) -> int:
        return self._match_losses

    @property
    def match_draws(self) -> int:
        return self._match_draws

    @property
    def win_pct(self) -> float:
        total = self.total_wins + self.total_losses + self.total_draws
        return (self.total_wins / total * 100) if total else 0.0

    @property
    def avg_score_per_match(self) -> float:
        return (self.score / self.matches_played) if self.matches_played else 0.0

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "elo": round(self.elo, 1),
            "matches_played": self.matches_played,
            "match_wins": self.match_wins,
            "match_losses": self.match_losses,
            "match_draws": self.match_draws,
            "score": self.score,
            "total_round_wins": self.total_wins,
            "total_round_losses": self.total_losses,
            "total_round_draws": self.total_draws,
            "round_win_pct": round(self.win_pct, 2),
            "avg_score_per_match": round(self.avg_score_per_match, 2),
        }


def compute_leaderboard(results: list[MatchResult]) -> list[LeaderboardEntry]:
    """Build a sorted leaderboard from match results, including Elo ratings."""
    entries: dict[str, LeaderboardEntry] = {}

    # Track match-level outcomes
    match_wins_count: dict[str, int] = defaultdict(int)
    match_losses_count: dict[str, int] = defaultdict(int)
    match_draws_count: dict[str, int] = defaultdict(int)

    for r in results:
        for name in [r.algo_a_name, r.algo_b_name]:
            if name not in entries:
                entries[name] = LeaderboardEntry(name=name)

        ea = entries[r.algo_a_name]
        eb = entries[r.algo_b_name]

        ea.total_wins += r.a_wins
        ea.total_losses += r.b_wins
        ea.total_draws += r.draws
        ea.matches_played += 1

        eb.total_wins += r.b_wins
        eb.total_losses += r.a_wins
        eb.total_draws += r.draws
        eb.matches_played += 1

        # Determine match winner
        if r.a_wins > r.b_wins:
            match_wins_count[r.algo_a_name] += 1
            match_losses_count[r.algo_b_name] += 1
        elif r.b_wins > r.a_wins:
            match_wins_count[r.algo_b_name] += 1
            match_losses_count[r.algo_a_name] += 1
        else:
            match_draws_count[r.algo_a_name] += 1
            match_draws_count[r.algo_b_name] += 1

    for name, entry in entries.items():
        entry._match_wins = match_wins_count.get(name, 0)
        entry._match_losses = match_losses_count.get(name, 0)
        entry._match_draws = match_draws_count.get(name, 0)

    # Compute Elo ratings
    elo_ratings = compute_elo_ratings(results)
    for name, entry in entries.items():
        entry.elo = elo_ratings.get(name, ELO_INITIAL)

    leaderboard = sorted(entries.values(), key=lambda e: (-e.elo, -e.score, -e.win_pct))
    return leaderboard


def head_to_head_matrix(results: list[MatchResult]) -> dict[str, dict[str, str]]:
    """Build an NxN head-to-head result matrix.

    Returns: {algo_a: {algo_b: "W" | "L" | "D"}}
    """
    matrix: dict[str, dict[str, str]] = defaultdict(dict)
    for r in results:
        if r.a_wins > r.b_wins:
            matrix[r.algo_a_name][r.algo_b_name] = "W"
            matrix[r.algo_b_name][r.algo_a_name] = "L"
        elif r.b_wins > r.a_wins:
            matrix[r.algo_a_name][r.algo_b_name] = "L"
            matrix[r.algo_b_name][r.algo_a_name] = "W"
        else:
            matrix[r.algo_a_name][r.algo_b_name] = "D"
            matrix[r.algo_b_name][r.algo_a_name] = "D"
    return dict(matrix)


# ---------------------------------------------------------------------------
# Pretty-printing
# ---------------------------------------------------------------------------

def print_match_summary(result: MatchResult):
    """Print a detailed summary of a single match."""
    # Compute Elo change for this single match
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
    delta_a = new_ra - ra
    delta_b = new_rb - rb

    print("=" * 60)
    print(f"  {result.algo_a_name}  vs  {result.algo_b_name}")
    print(f"  Rounds: {result.rounds}")
    print("=" * 60)
    print(f"  {'':20s} {'A':>10s} {'B':>10s}")
    print(f"  {'Wins':20s} {result.a_wins:>10d} {result.b_wins:>10d}")
    print(f"  {'Losses':20s} {result.b_wins:>10d} {result.a_wins:>10d}")
    print(f"  {'Draws':20s} {result.draws:>10d} {result.draws:>10d}")
    print(f"  {'Win %':20s} {result.a_win_pct:>9.1f}% {result.b_win_pct:>9.1f}%")
    print(f"  {'Most Common Move':20s} {result.a_most_common_move:>10s} {result.b_most_common_move:>10s}")
    sign_a = '+' if delta_a >= 0 else ''
    sign_b = '+' if delta_b >= 0 else ''
    print(f"  {'Elo':20s} {new_ra:>7.1f}({sign_a}{delta_a:.1f}) {new_rb:>7.1f}({sign_b}{delta_b:.1f})")
    print()
    print(f"  A move distribution: {result.a_move_distribution}")
    print(f"  B move distribution: {result.b_move_distribution}")

    if result.a_wins > result.b_wins:
        winner = result.algo_a_name
    elif result.b_wins > result.a_wins:
        winner = result.algo_b_name
    else:
        winner = "DRAW"
    print(f"\n  ★ Winner: {winner}")
    print("=" * 60)


def print_leaderboard(leaderboard: list[LeaderboardEntry]):
    """Print a formatted leaderboard table."""
    print()
    print("=" * 105)
    print(f"  {'#':>3s}  {'Algorithm':<22s} {'Elo':>7s} {'Score':>6s} {'MW':>4s} {'ML':>4s} {'MD':>4s} "
          f"{'RndW':>6s} {'RndL':>6s} {'RndD':>6s} {'Win%':>7s}")
    print("-" * 105)
    for i, e in enumerate(leaderboard, 1):
        print(f"  {i:>3d}  {e.name:<22s} {e.elo:>7.1f} {e.score:>6d} {e.match_wins:>4d} "
              f"{e.match_losses:>4d} {e.match_draws:>4d} "
              f"{e.total_wins:>6d} {e.total_losses:>6d} {e.total_draws:>6d} "
              f"{e.win_pct:>6.1f}%")
    print("=" * 105)
    print(f"  Elo = Elo rating (K={ELO_K_FACTOR}, start={ELO_INITIAL})")
    print(f"  MW=Match Wins  ML=Match Losses  MD=Match Draws")
    print(f"  RndW/RndL/RndD = Total Round Wins/Losses/Draws")
    print(f"  Score = MW×3 + MD×1  |  Sorted by Elo")
    print()


def print_h2h_matrix(matrix: dict[str, dict[str, str]], names: list[str]):
    """Print the head-to-head matrix."""
    # Abbreviate names for width
    abbrs = []
    for i, n in enumerate(names):
        abbrs.append(f"{i+1:>2d}")

    print()
    print("Head-to-Head Matrix (W=Win, L=Loss, D=Draw):")
    print()

    # Legend
    for i, n in enumerate(names):
        print(f"  {i+1:>2d} = {n}")
    print()

    # Header
    header = f"  {'':>22s} " + " ".join(f"{a:>3s}" for a in abbrs)
    print(header)
    print("  " + "-" * (22 + 1 + 4 * len(names)))

    for i, name in enumerate(names):
        row = f"  {name:>22s} "
        for j, opp in enumerate(names):
            if name == opp:
                row += "  - "
            else:
                result = matrix.get(name, {}).get(opp, "?")
                row += f"  {result} "
        print(row)
    print()
