"""
Feature Engineering for Tennis Betting Model.

Builds a rich feature matrix from the ATP match history with
NO look-ahead bias — every feature is computed using only
matches that occurred strictly BEFORE the target match.

Feature groups
--------------
1.  Elo ratings (global + surface-specific)
2.  ATP ranking & points
3.  Recent form (win rate, last 20 matches overall + surface-specific)
4.  Service stats (ace%, DF%, 1stIn%, 1stWon%, 2ndWon%, break-point stats)
5.  Return stats (derived from opponent service data)
6.  Head-to-head record (overall + surface)
7.  Physical / biographical (age, height, handedness)
8.  Context (surface dummies, round, tournament level, best-of, days-rest)

The feature matrix is from an arbitrary perspective (player_1 vs player_2)
with the target label being 1 when player_1 (= the match winner in the raw
data) wins.  During training we also randomly flip pairs to avoid the model
learning that "player_1 always wins".
"""
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from config import FORM_WINDOW, SURFACE_FORM_WINDOW, H2H_WINDOW, SURFACES


# ── Rolling player stats accumulator ──────────────────────────────────────────

class PlayerStatsAccumulator:
    """
    Maintains a per-player chronological match history and computes
    rolling statistics on demand for any date.

    Usage (chronological, no look-ahead):
        acc = PlayerStatsAccumulator()
        for match in matches_sorted_by_date:
            stats_p1 = acc.get_stats(p1_id, match.date, match.surface)
            stats_p2 = acc.get_stats(p2_id, match.date, match.surface)
            acc.update(match)   # add AFTER querying
    """

    def __init__(self):
        # {player_id: list of match dicts (chronological)}
        self._history: Dict[int, List[dict]] = defaultdict(list)

    def update(self, player_id: int, match: dict):
        """Append a match result to the player's history."""
        self._history[player_id].append(match)

    def get_stats(
        self, player_id: int, surface: str, window: int = FORM_WINDOW,
        surf_window: int = SURFACE_FORM_WINDOW
    ) -> dict:
        """
        Return rolling stats for player_id using all matches already stored
        (i.e., matches strictly before the current one).
        """
        history = self._history[player_id]
        n_total = len(history)

        if n_total == 0:
            return _empty_stats(surface)

        # Overall recent form (last `window` matches)
        recent = history[-window:]
        # Surface-specific recent form
        surf_recent = [m for m in history if m["surface"] == surface][-surf_window:]

        # ── Win rates ──────────────────────────────────────────────────────────
        win_rate  = np.mean([m["won"] for m in recent])
        surf_wr   = np.mean([m["won"] for m in surf_recent]) if surf_recent else np.nan

        # ── Service stats ──────────────────────────────────────────────────────
        svc = _aggregate_service(recent)
        svc_surf = _aggregate_service(surf_recent) if surf_recent else _empty_service()

        # ── Fatigue proxy: days since last match ───────────────────────────────
        last_date = history[-1]["date"] if history else None

        return {
            "n_matches":         n_total,
            "n_recent":          len(recent),
            "win_rate":          win_rate,
            "surf_n":            len(surf_recent),
            "surf_win_rate":     surf_wr,
            "last_match_date":   last_date,
            **svc,
            "surf_ace_rate":     svc_surf.get("ace_rate"),
            "surf_first_serve":  svc_surf.get("first_serve_pct"),
            "surf_first_won":    svc_surf.get("first_serve_won"),
        }


def _empty_service() -> dict:
    return {
        "ace_rate":        np.nan,
        "df_rate":         np.nan,
        "first_serve_pct": np.nan,
        "first_serve_won": np.nan,
        "second_serve_won":np.nan,
        "bp_conv_rate":    np.nan,
        "bp_saved_rate":   np.nan,
        "sv_games_won":    np.nan,
        "ret_pts_won":     np.nan,
    }


def _empty_stats(surface: str) -> dict:
    return {
        "n_matches": 0, "n_recent": 0,
        "win_rate": np.nan, "surf_n": 0, "surf_win_rate": np.nan,
        "last_match_date": None,
        "surf_ace_rate": np.nan, "surf_first_serve": np.nan, "surf_first_won": np.nan,
        **_empty_service(),
    }


def _safe_div(num, den):
    if den and not np.isnan(den) and den > 0:
        return num / den
    return np.nan


def _aggregate_service(matches: List[dict]) -> dict:
    """Aggregate service stats across a list of match dicts."""
    if not matches:
        return _empty_service()

    svpt = sum(m.get("svpt", 0) or 0 for m in matches)
    ace  = sum(m.get("ace",  0) or 0 for m in matches)
    df   = sum(m.get("df",   0) or 0 for m in matches)
    first_in   = sum(m.get("first_in", 0) or 0 for m in matches)
    first_won  = sum(m.get("first_won", 0) or 0 for m in matches)
    second_won = sum(m.get("second_won", 0) or 0 for m in matches)
    bp_conv    = sum(m.get("bp_won", 0) or 0 for m in matches)
    bp_opp     = sum(m.get("bp_faced_opp", 0) or 0 for m in matches)   # bp we faced on return
    bp_saved   = sum(m.get("bp_saved", 0) or 0 for m in matches)
    bp_faced   = sum(m.get("bp_faced", 0) or 0 for m in matches)
    sv_gms     = sum(m.get("sv_gms", 0) or 0 for m in matches)
    sv_gms_won = sum(m.get("sv_gms_won", 0) or 0 for m in matches)
    ret_pts_won= sum(m.get("ret_pts_won", 0) or 0 for m in matches)
    ret_pts    = sum(m.get("ret_pts", 0) or 0 for m in matches)

    second_svpt = max(svpt - first_in, 0)
    return {
        "ace_rate":         _safe_div(ace,        svpt),
        "df_rate":          _safe_div(df,         svpt),
        "first_serve_pct":  _safe_div(first_in,   svpt),
        "first_serve_won":  _safe_div(first_won,  first_in),
        "second_serve_won": _safe_div(second_won, second_svpt),
        "bp_conv_rate":     _safe_div(bp_conv,    bp_opp),
        "bp_saved_rate":    _safe_div(bp_saved,   bp_faced),
        "sv_games_won":     _safe_div(sv_gms_won, sv_gms),
        "ret_pts_won":      _safe_div(ret_pts_won,ret_pts),
    }


def _match_to_winner_dict(row) -> dict:
    """Extract winner's stats from a match row."""
    svpt = row.get("w_svpt") or 0
    return {
        "won": True,
        "surface": row["surface"],
        "date": row["tourney_date"],
        "opponent_rank": row.get("loser_rank"),
        "svpt":       _v(row, "w_svpt"),
        "ace":        _v(row, "w_ace"),
        "df":         _v(row, "w_df"),
        "first_in":   _v(row, "w_1stIn"),
        "first_won":  _v(row, "w_1stWon"),
        "second_won": _v(row, "w_2ndWon"),
        "sv_gms":     _v(row, "w_SvGms"),
        "sv_gms_won": _v(row, "w_SvGms"),   # approximation: all sv gms held
        "bp_saved":   _v(row, "w_bpSaved"),
        "bp_faced":   _v(row, "w_bpFaced"),
        # Return stats = loser's service stats from our perspective
        "bp_won":      _v(row, "l_bpFaced"),    # we converted opp bp chances
        "bp_faced_opp":_v(row, "l_bpFaced"),
        "ret_pts_won": _v(row, "l_svpt") - _v(row, "l_1stWon") - _v(row, "l_2ndWon") if _v(row, "l_svpt") else np.nan,
        "ret_pts":     _v(row, "l_svpt"),
    }


def _match_to_loser_dict(row) -> dict:
    """Extract loser's stats from a match row."""
    return {
        "won": False,
        "surface": row["surface"],
        "date": row["tourney_date"],
        "opponent_rank": row.get("winner_rank"),
        "svpt":       _v(row, "l_svpt"),
        "ace":        _v(row, "l_ace"),
        "df":         _v(row, "l_df"),
        "first_in":   _v(row, "l_1stIn"),
        "first_won":  _v(row, "l_1stWon"),
        "second_won": _v(row, "l_2ndWon"),
        "sv_gms":     _v(row, "l_SvGms"),
        "sv_gms_won": 0,
        "bp_saved":   _v(row, "l_bpSaved"),
        "bp_faced":   _v(row, "l_bpFaced"),
        "bp_won":      _v(row, "w_bpFaced"),
        "bp_faced_opp":_v(row, "w_bpFaced"),
        "ret_pts_won": _v(row, "w_svpt") - _v(row, "w_1stWon") - _v(row, "w_2ndWon") if _v(row, "w_svpt") else np.nan,
        "ret_pts":     _v(row, "w_svpt"),
    }


def _v(row, col, default=np.nan):
    val = row.get(col)
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return default
    return float(val)


# ── H2H tracker ───────────────────────────────────────────────────────────────

class H2HTracker:
    """
    Maintains a head-to-head record between every pair of players.
    Records are stored from both players' perspective.
    Query returns stats for player_a vs player_b strictly before `date`.
    """
    def __init__(self):
        # {(p1_id, p2_id): [(date, surface, p1_won), ...]}
        self._records: Dict[Tuple, List] = defaultdict(list)

    def update(self, winner_id: int, loser_id: int, date, surface: str):
        key_w = (winner_id, loser_id)
        key_l = (loser_id, winner_id)
        self._records[key_w].append((date, surface, True))
        self._records[key_l].append((date, surface, False))

    def get_h2h(self, p1_id: int, p2_id: int, surface: str) -> dict:
        records = self._records.get((p1_id, p2_id), [])
        recent  = records[-H2H_WINDOW:]

        if not recent:
            return {"h2h_n": 0, "h2h_win_rate": 0.5, "h2h_surf_n": 0, "h2h_surf_win_rate": 0.5}

        surf_rec = [r for r in recent if r[1] == surface]
        return {
            "h2h_n":            len(recent),
            "h2h_win_rate":     np.mean([r[2] for r in recent]),
            "h2h_surf_n":       len(surf_rec),
            "h2h_surf_win_rate":np.mean([r[2] for r in surf_rec]) if surf_rec else 0.5,
        }


# ── Main feature-building function ────────────────────────────────────────────

def build_feature_matrix(matches: pd.DataFrame) -> pd.DataFrame:
    """
    Build the training feature matrix from the pre-processed, Elo-annotated
    matches DataFrame.

    Steps:
      1. Sort chronologically
      2. For each match, query stats accumulated so far (no look-ahead)
      3. Update accumulators after querying
      4. Randomly flip player order (50% of rows) to prevent player-order bias

    Returns a DataFrame where each row is one match and the target column
    `p1_wins` indicates whether player_1 (= winner in ~50% of rows) won.
    """
    matches = matches.sort_values("tourney_date").reset_index(drop=True)

    acc   = PlayerStatsAccumulator()
    h2h   = H2HTracker()
    rows  = []
    rng   = np.random.default_rng(42)

    for idx, row in matches.iterrows():
        date     = row["tourney_date"]
        surface  = row["surface"]
        wid      = int(row["winner_id"])
        lid      = int(row["loser_id"])

        # ── Retrieve pre-match stats (no look-ahead) ───────────────────────────
        w_stats  = acc.get_stats(wid, surface)
        l_stats  = acc.get_stats(lid, surface)
        h2h_as_w = h2h.get_h2h(wid, lid, surface)   # from winner's perspective
        h2h_as_l = h2h.get_h2h(lid, wid, surface)   # from loser's perspective

        # ── Elo features ───────────────────────────────────────────────────────
        w_elo        = row.get("w_elo_before", 1500.0)
        l_elo        = row.get("l_elo_before", 1500.0)
        w_surf_elo   = row.get("w_surf_elo_before", 1500.0)
        l_surf_elo   = row.get("l_surf_elo_before", 1500.0)

        # ── Ranking features ───────────────────────────────────────────────────
        w_rank  = row.get("winner_rank")
        l_rank  = row.get("loser_rank")
        w_pts   = row.get("winner_rank_points")
        l_pts   = row.get("loser_rank_points")

        # ── Days rest ──────────────────────────────────────────────────────────
        w_last = w_stats.get("last_match_date")
        l_last = l_stats.get("last_match_date")
        w_rest = (date - w_last).days if w_last is not None else 30
        l_rest = (date - l_last).days if l_last is not None else 30

        # ── Randomly assign player 1 / player 2 ───────────────────────────────
        flip = rng.random() < 0.5
        if flip:
            p1_id, p2_id = lid, wid
            p1_stats, p2_stats = l_stats, w_stats
            p1_elo, p2_elo     = l_elo, w_elo
            p1_surf, p2_surf   = l_surf_elo, w_surf_elo
            p1_rank, p2_rank   = l_rank, w_rank
            p1_pts,  p2_pts    = l_pts,  w_pts
            p1_rest, p2_rest   = l_rest, w_rest
            p1_age,  p2_age    = row.get("loser_age"),  row.get("winner_age")
            p1_ht,   p2_ht     = row.get("loser_ht"),   row.get("winner_ht")
            p1_hand, p2_hand   = row.get("loser_hand"), row.get("winner_hand")
            p1_h2h = h2h_as_l;  p2_h2h = h2h_as_w
            target = 0   # player_1 (=loser) did NOT win
        else:
            p1_id, p2_id = wid, lid
            p1_stats, p2_stats = w_stats, l_stats
            p1_elo, p2_elo     = w_elo, l_elo
            p1_surf, p2_surf   = w_surf_elo, l_surf_elo
            p1_rank, p2_rank   = w_rank, l_rank
            p1_pts,  p2_pts    = w_pts,  l_pts
            p1_rest, p2_rest   = w_rest, l_rest
            p1_age,  p2_age    = row.get("winner_age"), row.get("loser_age")
            p1_ht,   p2_ht     = row.get("winner_ht"),  row.get("loser_ht")
            p1_hand, p2_hand   = row.get("winner_hand"),row.get("loser_hand")
            p1_h2h = h2h_as_w;  p2_h2h = h2h_as_l
            target = 1   # player_1 (=winner) won

        # ── Assemble feature row ───────────────────────────────────────────────
        feat = {
            "match_idx": idx,
            "date": date,
            "surface": surface,
            "tourney_level": row.get("tourney_level", "A"),
            "round_num": row.get("round_num", 3),
            "best_of": row.get("best_of", 3),

            # Elo
            "p1_elo":            p1_elo,
            "p2_elo":            p2_elo,
            "elo_diff":          p1_elo - p2_elo,
            "p1_surf_elo":       p1_surf,
            "p2_surf_elo":       p2_surf,
            "surf_elo_diff":     p1_surf - p2_surf,

            # Rank
            "p1_rank":           p1_rank if p1_rank else 500,
            "p2_rank":           p2_rank if p2_rank else 500,
            "rank_diff":         (_safe_log_rank(p1_rank) - _safe_log_rank(p2_rank)),
            "p1_rank_pts":       p1_pts if p1_pts else 0,
            "p2_rank_pts":       p2_pts if p2_pts else 0,
            "rank_pts_diff":     (p1_pts or 0) - (p2_pts or 0),

            # Form
            "p1_win_rate":       p1_stats.get("win_rate"),
            "p2_win_rate":       p2_stats.get("win_rate"),
            "win_rate_diff":     _diff(p1_stats.get("win_rate"), p2_stats.get("win_rate")),
            "p1_surf_win_rate":  p1_stats.get("surf_win_rate"),
            "p2_surf_win_rate":  p2_stats.get("surf_win_rate"),
            "surf_win_rate_diff":_diff(p1_stats.get("surf_win_rate"), p2_stats.get("surf_win_rate")),

            # Service stats
            "p1_ace_rate":       p1_stats.get("ace_rate"),
            "p2_ace_rate":       p2_stats.get("ace_rate"),
            "ace_diff":          _diff(p1_stats.get("ace_rate"),         p2_stats.get("ace_rate")),
            "p1_df_rate":        p1_stats.get("df_rate"),
            "p2_df_rate":        p2_stats.get("df_rate"),
            "df_diff":           _diff(p1_stats.get("df_rate"),          p2_stats.get("df_rate")),
            "p1_first_serve":    p1_stats.get("first_serve_pct"),
            "p2_first_serve":    p2_stats.get("first_serve_pct"),
            "first_serve_diff":  _diff(p1_stats.get("first_serve_pct"),  p2_stats.get("first_serve_pct")),
            "p1_first_won":      p1_stats.get("first_serve_won"),
            "p2_first_won":      p2_stats.get("first_serve_won"),
            "first_won_diff":    _diff(p1_stats.get("first_serve_won"),  p2_stats.get("first_serve_won")),
            "p1_second_won":     p1_stats.get("second_serve_won"),
            "p2_second_won":     p2_stats.get("second_serve_won"),
            "second_won_diff":   _diff(p1_stats.get("second_serve_won"), p2_stats.get("second_serve_won")),
            "p1_bp_conv":        p1_stats.get("bp_conv_rate"),
            "p2_bp_conv":        p2_stats.get("bp_conv_rate"),
            "bp_conv_diff":      _diff(p1_stats.get("bp_conv_rate"),     p2_stats.get("bp_conv_rate")),
            "p1_bp_saved":       p1_stats.get("bp_saved_rate"),
            "p2_bp_saved":       p2_stats.get("bp_saved_rate"),
            "bp_saved_diff":     _diff(p1_stats.get("bp_saved_rate"),    p2_stats.get("bp_saved_rate")),
            "p1_ret_won":        p1_stats.get("ret_pts_won"),
            "p2_ret_won":        p2_stats.get("ret_pts_won"),
            "ret_won_diff":      _diff(p1_stats.get("ret_pts_won"),      p2_stats.get("ret_pts_won")),

            # H2H
            "h2h_n":             p1_h2h["h2h_n"],
            "h2h_win_rate":      p1_h2h["h2h_win_rate"],
            "h2h_surf_n":        p1_h2h["h2h_surf_n"],
            "h2h_surf_win_rate": p1_h2h["h2h_surf_win_rate"],

            # Physical
            "p1_age":            p1_age,
            "p2_age":            p2_age,
            "age_diff":          _diff(p1_age, p2_age),
            "p1_height":         p1_ht,
            "p2_height":         p2_ht,
            "height_diff":       _diff(p1_ht, p2_ht),
            "p1_right_handed":   1 if p1_hand == "R" else (0 if p1_hand == "L" else np.nan),
            "p2_right_handed":   1 if p2_hand == "R" else (0 if p2_hand == "L" else np.nan),

            # Context
            "days_rest_diff":    p1_rest - p2_rest,
            "p1_days_rest":      p1_rest,
            "p2_days_rest":      p2_rest,

            # Surface dummies
            "surf_hard":   1 if surface == "Hard"   else 0,
            "surf_clay":   1 if surface == "Clay"   else 0,
            "surf_grass":  1 if surface == "Grass"  else 0,
            "surf_carpet": 1 if surface == "Carpet" else 0,

            # Grand Slam flag
            "is_grand_slam": 1 if row.get("tourney_level") == "G" else 0,
            "is_masters":    1 if row.get("tourney_level") == "M" else 0,

            # Stored IDs (for lookup, not used as features)
            "p1_id": p1_id,
            "p2_id": p2_id,
            "p1_wins": target,
        }

        rows.append(feat)

        # ── Update accumulators AFTER querying (no look-ahead) ─────────────────
        acc.update(wid, _match_to_winner_dict(row))
        acc.update(lid, _match_to_loser_dict(row))
        h2h.update(wid, lid, date, surface)

    df = pd.DataFrame(rows)
    print(f"Feature matrix: {len(df):,} rows × {len(df.columns)} columns")
    return df


def _diff(a, b):
    if a is None or b is None:
        return np.nan
    if isinstance(a, float) and np.isnan(a):
        return np.nan
    if isinstance(b, float) and np.isnan(b):
        return np.nan
    return a - b


def _safe_log_rank(rank):
    if rank is None or (isinstance(rank, float) and np.isnan(rank)):
        return np.log(500)
    return np.log(max(float(rank), 1))


# ── Feature column definitions ─────────────────────────────────────────────────

FEATURE_COLS = [
    # Elo
    "elo_diff", "surf_elo_diff", "p1_elo", "p2_elo",
    # Rank
    "rank_diff", "rank_pts_diff",
    # Form
    "win_rate_diff", "surf_win_rate_diff", "p1_win_rate", "p2_win_rate",
    # Service
    "ace_diff", "df_diff", "first_serve_diff", "first_won_diff",
    "second_won_diff", "bp_conv_diff", "bp_saved_diff", "ret_won_diff",
    # H2H
    "h2h_win_rate", "h2h_surf_win_rate", "h2h_n",
    # Physical
    "age_diff", "height_diff",
    # Context
    "days_rest_diff", "round_num", "best_of",
    "surf_hard", "surf_clay", "surf_grass", "surf_carpet",
    "is_grand_slam", "is_masters",
]

TARGET_COL = "p1_wins"


def get_feature_cols() -> List[str]:
    return [c for c in FEATURE_COLS]


def prepare_X_y(df: pd.DataFrame):
    """Return X (feature array) and y (target) from the feature DataFrame."""
    X = df[FEATURE_COLS].copy()
    y = df[TARGET_COL].values
    return X, y


def impute_features(X: pd.DataFrame) -> pd.DataFrame:
    """Fill NaN with column medians (fitted on training data)."""
    return X.fillna(X.median(numeric_only=True))


def fit_impute(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """Compute medians on train, apply to both train and test."""
    medians = X_train.median(numeric_only=True)
    return X_train.fillna(medians), X_test.fillna(medians), medians
