"""
Prediction Interface for the Tennis Betting Model.

Usage (CLI interactive mode):
    python predict.py

Usage (programmatic):
    from tennis_betting.predict import predict_match
    result = predict_match("Jannik Sinner", "Carlos Alcaraz", surface="Hard",
                           odds_p1=1.85, odds_p2=2.05)

The predictor:
  1. Loads the trained XGBoost model and imputation medians
  2. Looks up each player's current Elo + rolling stats from historical data
  3. Builds the feature vector for the requested match-up
  4. Returns win probabilities and a betting recommendation
"""
import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from typing import Optional, Dict, Tuple
from difflib import get_close_matches

# Allow running this file directly from scripts/tennis_betting/
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)

from config import (
    MODELS_DIR, DATA_DIR, SURFACES, FORM_WINDOW,
    SURFACE_FORM_WINDOW, ELO_INITIAL,
)
from feature_engineering import (
    FEATURE_COLS, PlayerStatsAccumulator, H2HTracker,
    _match_to_winner_dict, _match_to_loser_dict, _diff, _safe_log_rank,
)
from betting_engine import analyse_bet, print_bet_card


# ── Player registry ───────────────────────────────────────────────────────────

class PlayerRegistry:
    """
    Holds the current Elo ratings, rolling stats, and name mapping
    for all players seen in the training data.

    Built once from the combined match CSV; cached as a pickle.
    """

    def __init__(self):
        self._elo_global: Dict[int, float]            = {}
        self._elo_surface: Dict[Tuple[int, str], float] = {}
        self._player_stats: Dict[int, dict]           = {}
        self._name_to_id: Dict[str, int]              = {}
        self._id_to_name: Dict[int, str]              = {}
        self._h2h: H2HTracker                         = H2HTracker()
        self._acc: PlayerStatsAccumulator             = PlayerStatsAccumulator()

    def build(self, matches: pd.DataFrame):
        """Process all matches and compute current player states."""
        from elo_calculator import compute_elo_ratings, ELO_INITIAL

        print("Building player registry from match history …")
        matches, elo_global, elo_surface = compute_elo_ratings(matches)
        self._elo_global  = elo_global
        self._elo_surface = elo_surface

        # Build name ↔ id mappings
        for _, row in matches.iterrows():
            wid = int(row["winner_id"]); lid = int(row["loser_id"])
            wn  = str(row["winner_name"]); ln = str(row["loser_name"])
            self._name_to_id[wn.lower()] = wid
            self._name_to_id[ln.lower()] = lid
            self._id_to_name[wid]        = wn
            self._id_to_name[lid]        = ln

        # Replay matches to accumulate stats
        acc = PlayerStatsAccumulator()
        h2h = H2HTracker()
        for _, row in matches.sort_values("tourney_date").iterrows():
            wid = int(row["winner_id"]); lid = int(row["loser_id"])
            acc.update(wid, _match_to_winner_dict(row))
            acc.update(lid, _match_to_loser_dict(row))
            h2h.update(wid, lid, row["tourney_date"], row["surface"])

        self._acc = acc
        self._h2h = h2h
        print(f"  Registry: {len(self._name_to_id):,} players")

    def resolve_name(self, name: str) -> Optional[int]:
        """Fuzzy-match a player name and return their ID."""
        key = name.strip().lower()
        if key in self._name_to_id:
            return self._name_to_id[key]
        candidates = list(self._name_to_id.keys())
        close = get_close_matches(key, candidates, n=1, cutoff=0.6)
        if close:
            matched_name = close[0]
            pid = self._name_to_id[matched_name]
            print(f"  [INFO] '{name}' matched to '{self._id_to_name[pid]}'")
            return pid
        return None

    def get_elo(self, pid: int, surface: str) -> Tuple[float, float]:
        """Return (global_elo, surface_elo) for player id."""
        g = self._elo_global.get(pid, ELO_INITIAL)
        s = self._elo_surface.get((pid, surface), ELO_INITIAL)
        return g, s

    def get_stats(self, pid: int, surface: str) -> dict:
        """Return rolling stats for player using all available history."""
        # Use a future date so all matches are included
        return self._acc.get_stats(pid, surface)

    def get_h2h(self, p1_id: int, p2_id: int, surface: str) -> dict:
        return self._h2h.get_h2h(p1_id, p2_id, surface)

    def top_players(self, n: int = 20) -> pd.DataFrame:
        """Return top-N players by current global Elo."""
        rows = []
        for pid, elo in sorted(self._elo_global.items(), key=lambda x: -x[1])[:n]:
            rows.append({
                "rank":      len(rows) + 1,
                "name":      self._id_to_name.get(pid, f"#{pid}"),
                "elo":       round(elo, 1),
                "elo_hard":  round(self._elo_surface.get((pid, "Hard"),  ELO_INITIAL), 1),
                "elo_clay":  round(self._elo_surface.get((pid, "Clay"),  ELO_INITIAL), 1),
                "elo_grass": round(self._elo_surface.get((pid, "Grass"), ELO_INITIAL), 1),
            })
        return pd.DataFrame(rows)

    def save(self, path: str):
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> "PlayerRegistry":
        return joblib.load(path)


# ── Feature builder for a single match ───────────────────────────────────────

def build_match_features(
    registry: "PlayerRegistry",
    p1_id: int,
    p2_id: int,
    surface: str,
    tourney_level: str = "A",
    round_num: int = 4,
    best_of: int = 3,
    p1_rank: Optional[float] = None,
    p2_rank: Optional[float] = None,
    p1_rank_pts: Optional[float] = None,
    p2_rank_pts: Optional[float] = None,
) -> pd.DataFrame:
    """
    Build a one-row feature DataFrame for a match to be predicted.
    Mirrors the structure of feature_engineering.build_feature_matrix rows.
    """
    p1_g, p1_s = registry.get_elo(p1_id, surface)
    p2_g, p2_s = registry.get_elo(p2_id, surface)

    p1_stats = registry.get_stats(p1_id, surface)
    p2_stats = registry.get_stats(p2_id, surface)
    h2h_p1   = registry.get_h2h(p1_id, p2_id, surface)

    def _s(d, k, default=np.nan):
        v = d.get(k, default)
        return v if v is not None else default

    r1 = p1_rank or 500.0
    r2 = p2_rank or 500.0
    rp1 = p1_rank_pts or 0.0
    rp2 = p2_rank_pts or 0.0

    feat = {
        "elo_diff":          p1_g - p2_g,
        "surf_elo_diff":     p1_s - p2_s,
        "p1_elo":            p1_g,
        "p2_elo":            p2_g,
        "rank_diff":         _safe_log_rank(r1) - _safe_log_rank(r2),
        "rank_pts_diff":     rp1 - rp2,
        "p1_win_rate":       _s(p1_stats, "win_rate"),
        "p2_win_rate":       _s(p2_stats, "win_rate"),
        "win_rate_diff":     _diff(_s(p1_stats, "win_rate"), _s(p2_stats, "win_rate")),
        "p1_surf_win_rate":  _s(p1_stats, "surf_win_rate"),
        "p2_surf_win_rate":  _s(p2_stats, "surf_win_rate"),
        "surf_win_rate_diff":_diff(_s(p1_stats, "surf_win_rate"), _s(p2_stats, "surf_win_rate")),
        "ace_diff":          _diff(_s(p1_stats, "ace_rate"), _s(p2_stats, "ace_rate")),
        "df_diff":           _diff(_s(p1_stats, "df_rate"),  _s(p2_stats, "df_rate")),
        "first_serve_diff":  _diff(_s(p1_stats, "first_serve_pct"), _s(p2_stats, "first_serve_pct")),
        "first_won_diff":    _diff(_s(p1_stats, "first_serve_won"), _s(p2_stats, "first_serve_won")),
        "second_won_diff":   _diff(_s(p1_stats, "second_serve_won"),_s(p2_stats, "second_serve_won")),
        "bp_conv_diff":      _diff(_s(p1_stats, "bp_conv_rate"), _s(p2_stats, "bp_conv_rate")),
        "bp_saved_diff":     _diff(_s(p1_stats, "bp_saved_rate"),_s(p2_stats, "bp_saved_rate")),
        "ret_won_diff":      _diff(_s(p1_stats, "ret_pts_won"), _s(p2_stats, "ret_pts_won")),
        "h2h_n":             h2h_p1["h2h_n"],
        "h2h_win_rate":      h2h_p1["h2h_win_rate"],
        "h2h_surf_n":        h2h_p1["h2h_surf_n"],
        "h2h_surf_win_rate": h2h_p1["h2h_surf_win_rate"],
        "age_diff":          np.nan,
        "height_diff":       np.nan,
        "days_rest_diff":    0.0,
        "round_num":         round_num,
        "best_of":           best_of,
        "surf_hard":         1 if surface == "Hard"   else 0,
        "surf_clay":         1 if surface == "Clay"   else 0,
        "surf_grass":        1 if surface == "Grass"  else 0,
        "surf_carpet":       1 if surface == "Carpet" else 0,
        "is_grand_slam":     1 if tourney_level == "G" else 0,
        "is_masters":        1 if tourney_level == "M" else 0,
    }
    return pd.DataFrame([feat])[FEATURE_COLS]


# ── Main predict function ─────────────────────────────────────────────────────

def predict_match(
    player1: str,
    player2: str,
    surface: str = "Hard",
    tourney_level: str = "A",
    round_name: str = "QF",
    best_of: int = 3,
    odds_p1: Optional[float] = None,
    odds_p2: Optional[float] = None,
    registry: Optional["PlayerRegistry"] = None,
    model=None,
    medians=None,
) -> Dict:
    """
    Predict the outcome of a tennis match.

    Parameters
    ----------
    player1, player2 : player names (fuzzy-matched to database)
    surface          : "Hard" | "Clay" | "Grass" | "Carpet"
    tourney_level    : "G" | "M" | "A" | "F" | "D" | "C"
    round_name       : "R128" | "R64" | "R32" | "R16" | "QF" | "SF" | "F"
    best_of          : 3 or 5
    odds_p1, odds_p2 : bookmaker decimal odds (optional, for bet analysis)

    Returns dict with prediction and optional betting recommendation.
    """
    from config import ROUND_MAP
    from model_training import load_model, predict_proba

    # Load model if not provided
    if model is None or medians is None:
        model, medians = load_model()

    # Load registry if not provided
    if registry is None:
        reg_path = os.path.join(MODELS_DIR, "player_registry.pkl")
        if os.path.exists(reg_path):
            registry = PlayerRegistry.load(reg_path)
        else:
            raise FileNotFoundError(
                "Player registry not found. Run the pipeline first."
            )

    # Resolve player names
    p1_id = registry.resolve_name(player1)
    p2_id = registry.resolve_name(player2)

    p1_name = registry._id_to_name.get(p1_id, player1) if p1_id else player1
    p2_name = registry._id_to_name.get(p2_id, player2) if p2_id else player2

    if p1_id is None:
        print(f"  [WARN] '{player1}' not found in database — using default Elo")
    if p2_id is None:
        print(f"  [WARN] '{player2}' not found in database — using default Elo")

    # Use dummy IDs for unknown players
    if p1_id is None:
        p1_id = -1
    if p2_id is None:
        p2_id = -2

    # Build features
    round_num = ROUND_MAP.get(round_name.upper(), 4)
    X = build_match_features(
        registry, p1_id, p2_id, surface,
        tourney_level=tourney_level,
        round_num=round_num,
        best_of=best_of,
    )

    # Predict
    prob_p1 = float(predict_proba(model, medians, X)[0])
    prob_p2 = 1.0 - prob_p1

    # Elo-only baseline
    p1_g, p1_s = registry.get_elo(p1_id, surface)
    p2_g, p2_s = registry.get_elo(p2_id, surface)
    elo_prob_p1 = 1.0 / (1.0 + 10 ** ((p2_g - p1_g) / 400.0))
    surf_prob_p1= 1.0 / (1.0 + 10 ** ((p2_s - p1_s) / 400.0))

    result = {
        "player1":       p1_name,
        "player2":       p2_name,
        "surface":       surface,
        "tourney_level": tourney_level,
        "round":         round_name,
        "best_of":       best_of,
        "p1_win_prob":   round(prob_p1, 4),
        "p2_win_prob":   round(prob_p2, 4),
        "elo_p1_win_prob":   round(elo_prob_p1, 4),
        "surf_elo_p1_win_prob": round(surf_prob_p1, 4),
        "p1_elo":        round(p1_g, 1),
        "p2_elo":        round(p2_g, 1),
        "p1_surf_elo":   round(p1_s, 1),
        "p2_surf_elo":   round(p2_s, 1),
    }

    # Betting analysis
    if odds_p1 is not None:
        bet = analyse_bet(
            model_prob_p1=prob_p1,
            decimal_odds_p1=odds_p1,
            decimal_odds_p2=odds_p2,
            player_name_p1=p1_name,
            player_name_p2=p2_name,
        )
        result["bet"] = bet

    return result


def print_prediction(result: Dict):
    """Pretty-print a match prediction."""
    print(f"\n{'═'*60}")
    print(f"  MATCH PREDICTION  —  {result['surface']} / {result['round']}")
    print(f"{'═'*60}")

    p1 = result["player1"]; p2 = result["player2"]
    lw = max(len(p1), len(p2), 22)

    def pct(x): return f"{x*100:.1f}%"

    print(f"\n  {'Player':<{lw}}  {'ML Model':>9}  {'Elo':>9}  {'Surf Elo':>9}  {'Global Elo':>10}")
    print(f"  {'─'*58}")
    print(f"  {p1:<{lw}}  {pct(result['p1_win_prob']):>9}  "
          f"{pct(result['elo_p1_win_prob']):>9}  "
          f"{pct(result['surf_elo_p1_win_prob']):>9}  "
          f"{result['p1_elo']:>10.0f}")
    print(f"  {p2:<{lw}}  {pct(result['p2_win_prob']):>9}  "
          f"{pct(1-result['elo_p1_win_prob']):>9}  "
          f"{pct(1-result['surf_elo_p1_win_prob']):>9}  "
          f"{result['p2_elo']:>10.0f}")

    winner = p1 if result["p1_win_prob"] > 0.5 else p2
    conf   = max(result["p1_win_prob"], result["p2_win_prob"])
    print(f"\n  Predicted winner: {winner}  ({pct(conf)} confidence)")

    if "bet" in result:
        print_bet_card(result["bet"], surface=result["surface"])


# ── Interactive CLI ───────────────────────────────────────────────────────────

def _prompt(msg: str, default: str = "") -> str:
    val = input(f"  {msg}" + (f" [{default}]" if default else "") + ": ").strip()
    return val if val else default


def interactive_predict(registry, model, medians):
    """Run the interactive prediction CLI."""
    print("\n" + "═" * 60)
    print("  TENNIS BETTING MODEL — MATCH PREDICTOR")
    print("=" * 60)
    print("\n  Top players by current Elo:")
    top = registry.top_players(20)
    print(top.to_string(index=False))
    print()

    while True:
        print("\n" + "─" * 60)
        p1 = _prompt("Player 1 name (or 'q' to quit)")
        if p1.lower() in ("q", "quit", "exit"):
            break
        p2 = _prompt("Player 2 name")

        surface = _prompt("Surface (Hard/Clay/Grass/Carpet)", "Hard")
        if surface not in SURFACES:
            surface = "Hard"

        level = _prompt("Level (G=GrandSlam, M=Masters, A=ATP500/250, C=Challenger)", "A")
        round_name = _prompt("Round (R128/R64/R32/R16/QF/SF/F)", "QF")
        best_of = int(_prompt("Best of (3 or 5)", "3"))

        odds_p1_str = _prompt(f"Decimal odds for {p1} (or Enter to skip)", "")
        odds_p2_str = _prompt(f"Decimal odds for {p2} (or Enter to skip)", "")

        odds_p1 = float(odds_p1_str) if odds_p1_str else None
        odds_p2 = float(odds_p2_str) if odds_p2_str else None

        result = predict_match(
            player1=p1, player2=p2,
            surface=surface, tourney_level=level,
            round_name=round_name, best_of=best_of,
            odds_p1=odds_p1, odds_p2=odds_p2,
            registry=registry, model=model, medians=medians,
        )
        print_prediction(result)

        again = _prompt("Predict another match? (y/n)", "y")
        if again.lower() != "y":
            break


if __name__ == "__main__":
    # Standalone mode: load model + registry and run interactive predictor
    from model_training import load_model
    model, medians = load_model()
    reg_path = os.path.join(MODELS_DIR, "player_registry.pkl")
    registry = PlayerRegistry.load(reg_path)
    interactive_predict(registry, model, medians)
