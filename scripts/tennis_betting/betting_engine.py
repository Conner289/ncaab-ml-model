"""
Betting Engine for the Tennis Betting Model.

Implements:
  - Implied probability from decimal / American / fractional odds
  - Expected Value (EV) calculation
  - Kelly Criterion bet sizing (full and fractional)
  - Value bet detection & filtering
  - Bankroll management with configurable fraction-Kelly

Glossary
--------
decimal_odds  : e.g. 2.10 means €1 stake returns €2.10 total (€1.10 profit)
implied_prob  : 1 / decimal_odds  (bookmaker's probability estimate)
model_prob    : probability predicted by the ML model
edge          : model_prob - implied_prob  (positive = value bet)
EV            : expected profit per unit staked = edge * (decimal_odds - 1)
Kelly f       : edge / (decimal_odds - 1)  (full Kelly stake fraction)
"""
from typing import Optional, Dict
import numpy as np

from config import (
    MIN_EDGE, KELLY_FRACTION, BANKROLL,
    MIN_DECIMAL_ODDS, MAX_DECIMAL_ODDS,
)


# ── Odds conversions ───────────────────────────────────────────────────────────

def american_to_decimal(american: float) -> float:
    """Convert American (moneyline) odds to decimal odds."""
    if american > 0:
        return 1.0 + american / 100.0
    else:
        return 1.0 + 100.0 / abs(american)


def fractional_to_decimal(numerator: float, denominator: float) -> float:
    """Convert fractional odds (e.g. 7/4) to decimal."""
    return 1.0 + numerator / denominator


def implied_probability(decimal_odds: float) -> float:
    """Convert decimal odds to implied win probability (raw, no margin removal)."""
    if decimal_odds <= 1.0:
        return 1.0
    return 1.0 / decimal_odds


def remove_vig(odds_p1: float, odds_p2: float):
    """
    Remove the bookmaker's overround from a two-way market.
    Returns (true_prob_p1, true_prob_p2, overround).
    """
    raw_p1 = implied_probability(odds_p1)
    raw_p2 = implied_probability(odds_p2)
    overround = raw_p1 + raw_p2     # typically 1.02–1.10
    true_p1 = raw_p1 / overround
    true_p2 = raw_p2 / overround
    return true_p1, true_p2, overround


# ── EV & Kelly ─────────────────────────────────────────────────────────────────

def expected_value(model_prob: float, decimal_odds: float) -> float:
    """
    Expected profit per €1 staked.
    EV = model_prob * (decimal_odds - 1) - (1 - model_prob)
       = model_prob * decimal_odds - 1
    """
    return model_prob * decimal_odds - 1.0


def kelly_fraction(model_prob: float, decimal_odds: float) -> float:
    """
    Full Kelly fraction of bankroll to stake.
    f* = (model_prob * (decimal_odds - 1) - (1 - model_prob)) / (decimal_odds - 1)
       = edge / (decimal_odds - 1)
    Returns 0 if negative (no bet).
    """
    b = decimal_odds - 1.0   # net odds
    if b <= 0:
        return 0.0
    q = 1.0 - model_prob
    f = (model_prob * b - q) / b
    return max(f, 0.0)


def recommended_stake(
    model_prob: float,
    decimal_odds: float,
    bankroll: float = BANKROLL,
    kelly_frac: float = KELLY_FRACTION,
    max_stake_pct: float = 0.10,   # hard cap: never bet more than 10% of bankroll
) -> float:
    """
    Fractional Kelly bet size in currency units.
    Capped at max_stake_pct of bankroll to limit variance.
    """
    f = kelly_fraction(model_prob, decimal_odds)
    stake = bankroll * f * kelly_frac
    return min(stake, bankroll * max_stake_pct)


# ── Value bet analysis ─────────────────────────────────────────────────────────

def analyse_bet(
    model_prob_p1: float,
    decimal_odds_p1: float,
    decimal_odds_p2: Optional[float] = None,
    bankroll: float = BANKROLL,
    player_name_p1: str = "Player 1",
    player_name_p2: str = "Player 2",
    min_edge: float = MIN_EDGE,
) -> Dict:
    """
    Full betting analysis for one match.

    Parameters
    ----------
    model_prob_p1   : model's win probability for player 1
    decimal_odds_p1 : bookmaker's decimal odds for player 1
    decimal_odds_p2 : bookmaker's decimal odds for player 2 (optional)
    bankroll        : current bankroll
    min_edge        : minimum edge to recommend a bet

    Returns a dict with:
      - recommendation: "BET P1" | "BET P2" | "NO BET"
      - player to bet, edge, EV, stake, implied prob, model prob
    """
    model_prob_p2 = 1.0 - model_prob_p1

    # ── Player 1 side ──────────────────────────────────────────────────────────
    imp_p1  = implied_probability(decimal_odds_p1)
    edge_p1 = model_prob_p1 - imp_p1
    ev_p1   = expected_value(model_prob_p1, decimal_odds_p1)

    # ── Player 2 side ──────────────────────────────────────────────────────────
    if decimal_odds_p2:
        imp_p2  = implied_probability(decimal_odds_p2)
        edge_p2 = model_prob_p2 - imp_p2
        ev_p2   = expected_value(model_prob_p2, decimal_odds_p2)
        overround = imp_p1 + imp_p2
    else:
        # Derive p2 odds from p1 (assuming standard ~5% vig)
        fair_p2 = model_prob_p2
        decimal_odds_p2 = (1.0 / fair_p2) * 0.95 if fair_p2 > 0 else None
        imp_p2  = 1.0 - imp_p1
        edge_p2 = model_prob_p2 - imp_p2
        ev_p2   = expected_value(model_prob_p2, decimal_odds_p2) if decimal_odds_p2 else 0
        overround = None

    # ── Decision ───────────────────────────────────────────────────────────────
    bet_on = None
    if edge_p1 >= min_edge and decimal_odds_p1 >= MIN_DECIMAL_ODDS and decimal_odds_p1 <= MAX_DECIMAL_ODDS:
        if ev_p1 > 0:
            bet_on = "p1"
    if edge_p2 >= min_edge and decimal_odds_p2 and decimal_odds_p2 >= MIN_DECIMAL_ODDS and decimal_odds_p2 <= MAX_DECIMAL_ODDS:
        if ev_p2 > 0:
            if bet_on is None or ev_p2 > ev_p1:
                bet_on = "p2"

    if bet_on == "p1":
        stake   = recommended_stake(model_prob_p1, decimal_odds_p1, bankroll)
        rec     = f"BET {player_name_p1}"
        best_ev = ev_p1; best_edge = edge_p1; best_odds = decimal_odds_p1
    elif bet_on == "p2":
        stake   = recommended_stake(model_prob_p2, decimal_odds_p2, bankroll)
        rec     = f"BET {player_name_p2}"
        best_ev = ev_p2; best_edge = edge_p2; best_odds = decimal_odds_p2
    else:
        stake   = 0.0
        rec     = "NO BET"
        best_ev = max(ev_p1, ev_p2 if decimal_odds_p2 else -99)
        best_edge = max(edge_p1, edge_p2)
        best_odds = decimal_odds_p1

    return {
        "recommendation":   rec,
        "bet_on":           bet_on,
        # P1
        "p1_name":          player_name_p1,
        "p1_model_prob":    round(model_prob_p1, 4),
        "p1_odds":          round(decimal_odds_p1, 3),
        "p1_implied_prob":  round(imp_p1, 4),
        "p1_edge":          round(edge_p1, 4),
        "p1_ev":            round(ev_p1, 4),
        # P2
        "p2_name":          player_name_p2,
        "p2_model_prob":    round(model_prob_p2, 4),
        "p2_odds":          round(decimal_odds_p2, 3) if decimal_odds_p2 else None,
        "p2_implied_prob":  round(imp_p2, 4),
        "p2_edge":          round(edge_p2, 4),
        "p2_ev":            round(ev_p2, 4),
        # Bet
        "overround":        round(overround, 4) if overround else None,
        "stake":            round(stake, 2),
        "bankroll":         bankroll,
    }


def print_bet_card(analysis: Dict, surface: str = "", tournament: str = ""):
    """Pretty-print a betting analysis card to stdout."""
    sep = "─" * 60

    print(f"\n{'═' * 60}")
    if tournament:
        print(f"  {tournament}  [{surface}]")
    print(f"{'═' * 60}")

    def pct(x): return f"{x * 100:.1f}%"
    def odds_str(x): return f"{x:.2f}" if x else "N/A"

    p1 = analysis["p1_name"]
    p2 = analysis["p2_name"]
    longest = max(len(p1), len(p2), 20)

    print(f"  {'Player':<{longest}}  {'Model%':>7}  {'Odds':>6}  {'Implied%':>9}  {'Edge':>7}  {'EV/unit':>8}")
    print(f"  {sep}")
    p1_edge_sign = "▲" if analysis["p1_edge"] >= MIN_EDGE else " "
    p2_edge_sign = "▲" if analysis["p2_edge"] >= MIN_EDGE else " "

    print(f"  {p1:<{longest}}  {pct(analysis['p1_model_prob']):>7}  {odds_str(analysis['p1_odds']):>6}  "
          f"{pct(analysis['p1_implied_prob']):>9}  {p1_edge_sign}{pct(analysis['p1_edge']):>6}  "
          f"{analysis['p1_ev']:>+8.4f}")
    print(f"  {p2:<{longest}}  {pct(analysis['p2_model_prob']):>7}  {odds_str(analysis['p2_odds']):>6}  "
          f"{pct(analysis['p2_implied_prob']):>9}  {p2_edge_sign}{pct(analysis['p2_edge']):>6}  "
          f"{analysis['p2_ev']:>+8.4f}")

    if analysis.get("overround"):
        print(f"\n  Overround: {analysis['overround']:.3f} ({(analysis['overround']-1)*100:.1f}% vig)")

    print(f"\n  {'─'*56}")
    rec = analysis["recommendation"]
    if analysis["bet_on"]:
        print(f"  RECOMMENDATION  :  ✓ {rec}")
        print(f"  STAKE           :  ${analysis['stake']:.2f} "
              f"(bankroll: ${analysis['bankroll']:.2f})")
    else:
        print(f"  RECOMMENDATION  :  ✗ {rec}  (insufficient edge)")
    print(f"{'═' * 60}\n")


# ── Batch screening ───────────────────────────────────────────────────────────

def screen_slate(matches: list) -> list:
    """
    Screen a slate of matches for value bets.

    matches is a list of dicts with keys:
      p1_name, p2_name, model_prob_p1, odds_p1, odds_p2, surface, tournament

    Returns a sorted list of value bets.
    """
    value_bets = []
    for m in matches:
        a = analyse_bet(
            model_prob_p1=m["model_prob_p1"],
            decimal_odds_p1=m["odds_p1"],
            decimal_odds_p2=m.get("odds_p2"),
            player_name_p1=m.get("p1_name", "P1"),
            player_name_p2=m.get("p2_name", "P2"),
        )
        if a["bet_on"]:
            a["surface"]    = m.get("surface", "")
            a["tournament"] = m.get("tournament", "")
            value_bets.append(a)

    # Sort by EV descending
    best_ev_key = lambda x: max(x["p1_ev"], x["p2_ev"])
    return sorted(value_bets, key=best_ev_key, reverse=True)
