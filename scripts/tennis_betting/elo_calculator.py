"""
Elo Rating Calculator for Tennis.

Computes three types of Elo for each player:
  1. Global Elo — updated after every match, regardless of surface
  2. Surface Elo — separate ratings for Hard, Clay, Grass, Carpet
  3. Recent Elo — a time-decayed variant emphasising last 12 months

K-factor is scaled by tournament importance (Grand Slams update more than Challengers).

The Elo BEFORE each match is stored so that features built later
have no look-ahead bias.
"""
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional

from config import (
    ELO_INITIAL, ELO_K_FACTOR, ELO_SURFACE_K,
    ELO_WEIGHT_LEVEL, SURFACES,
)


# ── Core Elo math ──────────────────────────────────────────────────────────────

def expected_score(rating_a: float, rating_b: float) -> float:
    """Logistic expected win probability for player A."""
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))


def update_elo(rating: float, expected: float, actual: float, k: float) -> float:
    return rating + k * (actual - expected)


# ── Main computation ───────────────────────────────────────────────────────────

def compute_elo_ratings(matches: pd.DataFrame) -> pd.DataFrame:
    """
    Process matches chronologically and compute pre-match Elo ratings.

    Parameters
    ----------
    matches : DataFrame
        Preprocessed ATP matches (must contain tourney_date, winner_id,
        loser_id, surface, tourney_level).

    Returns
    -------
    DataFrame
        Same matches with added columns:
          w_elo_before, l_elo_before                 (global pre-match Elo)
          w_surf_elo_before, l_surf_elo_before        (surface Elo)
          w_elo_after, l_elo_after                    (global post-match Elo)
          elo_diff, surf_elo_diff                     (winner minus loser, pre-match)
    """
    matches = matches.copy().sort_values("tourney_date").reset_index(drop=True)

    # Current ratings dictionaries
    elo_global: Dict[int, float]          = {}
    elo_surface: Dict[Tuple[int, str], float] = {}

    pre_w_elo, pre_l_elo         = [], []
    pre_w_surf, pre_l_surf       = [], []
    post_w_elo, post_l_elo       = [], []

    for _, row in matches.iterrows():
        wid  = int(row["winner_id"])
        lid  = int(row["loser_id"])
        surf = row["surface"]
        lvl  = row.get("tourney_level", "A")

        # K-factor scaled by level
        k_mult = ELO_WEIGHT_LEVEL.get(str(lvl), 1.0)
        k_glob = ELO_K_FACTOR   * k_mult
        k_surf = ELO_SURFACE_K  * k_mult

        # Retrieve current (pre-match) ratings
        w_g  = elo_global.get(wid, ELO_INITIAL)
        l_g  = elo_global.get(lid, ELO_INITIAL)
        w_s  = elo_surface.get((wid, surf), ELO_INITIAL)
        l_s  = elo_surface.get((lid, surf), ELO_INITIAL)

        pre_w_elo.append(w_g);   pre_l_elo.append(l_g)
        pre_w_surf.append(w_s);  pre_l_surf.append(l_s)

        # Global Elo update
        e_w = expected_score(w_g, l_g)
        new_w_g = update_elo(w_g, e_w,   1.0, k_glob)
        new_l_g = update_elo(l_g, 1-e_w, 0.0, k_glob)
        elo_global[wid] = new_w_g
        elo_global[lid] = new_l_g
        post_w_elo.append(new_w_g); post_l_elo.append(new_l_g)

        # Surface Elo update
        e_w_s = expected_score(w_s, l_s)
        elo_surface[(wid, surf)] = update_elo(w_s, e_w_s,   1.0, k_surf)
        elo_surface[(lid, surf)] = update_elo(l_s, 1-e_w_s, 0.0, k_surf)

    matches["w_elo_before"]    = pre_w_elo
    matches["l_elo_before"]    = pre_l_elo
    matches["w_surf_elo_before"] = pre_w_surf
    matches["l_surf_elo_before"] = pre_l_surf
    matches["w_elo_after"]     = post_w_elo
    matches["l_elo_after"]     = post_l_elo

    matches["elo_diff"]        = matches["w_elo_before"] - matches["l_elo_before"]
    matches["surf_elo_diff"]   = matches["w_surf_elo_before"] - matches["l_surf_elo_before"]

    return matches, elo_global, elo_surface


def get_current_elo(matches: pd.DataFrame) -> Dict[int, Dict]:
    """
    Return the most recent Elo ratings for every player.
    Used at prediction time when a player's pre-match rating is needed.

    Returns dict: {player_id: {"global": float, "Hard": float, "Clay": float, "Grass": float, "Carpet": float}}
    """
    _, elo_global, elo_surface = compute_elo_ratings(matches)

    result: Dict[int, Dict] = {}
    all_players = set(elo_global.keys())
    for pid in all_players:
        entry = {"global": elo_global.get(pid, ELO_INITIAL)}
        for surf in SURFACES:
            entry[surf] = elo_surface.get((pid, surf), ELO_INITIAL)
        result[pid] = entry

    return result


def elo_summary(matches: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """
    Print the current top-N players by Elo rating.
    Requires matches to have been processed by compute_elo_ratings.
    """
    _, elo_global, elo_surface = compute_elo_ratings(matches)

    # Build a name lookup
    name_lookup: Dict[int, str] = {}
    for col_id, col_name in [("winner_id", "winner_name"), ("loser_id", "loser_name")]:
        if col_id in matches.columns and col_name in matches.columns:
            latest = matches.sort_values("tourney_date").drop_duplicates(col_id, keep="last")
            for _, row in latest.iterrows():
                pid = int(row[col_id])
                if pid not in name_lookup:
                    name_lookup[pid] = str(row[col_name])

    rows = []
    for pid, g_elo in sorted(elo_global.items(), key=lambda x: -x[1])[:top_n]:
        rows.append({
            "player_id": pid,
            "name":      name_lookup.get(pid, f"ID#{pid}"),
            "elo":       round(g_elo, 1),
            "elo_hard":  round(elo_surface.get((pid, "Hard"),   ELO_INITIAL), 1),
            "elo_clay":  round(elo_surface.get((pid, "Clay"),   ELO_INITIAL), 1),
            "elo_grass": round(elo_surface.get((pid, "Grass"),  ELO_INITIAL), 1),
        })
    return pd.DataFrame(rows)


def elo_win_probability(elo_a: float, elo_b: float) -> float:
    """Return win probability for player with elo_a against elo_b."""
    return expected_score(elo_a, elo_b)
