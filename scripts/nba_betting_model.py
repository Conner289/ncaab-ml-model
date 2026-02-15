"""
NBA Game Outcome Prediction & Value Betting Model
===================================================
Predicts NBA home team win/loss probabilities and identifies value bets
by comparing model-estimated probabilities against bookmaker odds.

DISCLAIMER: This script is for EDUCATIONAL PURPOSES ONLY. Sports betting
involves significant financial risk. Past performance does not guarantee
future results. No model can reliably predict game outcomes. Please
gamble responsibly and within your means. Never bet more than you can
afford to lose.
"""

import os
import sys
import warnings
from typing import Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# 1. DATA LOADING
# ---------------------------------------------------------------------------


def generate_sample_data(n_games: int = 2000, seed: int = 42) -> pd.DataFrame:
    """Generate a synthetic NBA-like dataset for demonstration.

    Creates realistic-looking game data with correlated features so the model
    has signal to learn from.  Use this when no real CSV is available.
    """
    rng = np.random.default_rng(seed)

    teams = [
        "BOS", "MIL", "PHI", "CLE", "NYK", "BKN", "MIA", "ATL",
        "CHI", "TOR", "IND", "ORL", "WAS", "DET", "CHA",
        "DEN", "MEM", "SAC", "PHX", "LAC", "DAL", "MIN",
        "NOP", "LAL", "GSW", "OKC", "POR", "UTA", "SAS", "HOU",
    ]

    # Assign each team a hidden "strength" rating (70-115 scale)
    team_strength = {t: rng.uniform(70, 115) for t in teams}

    records = []
    for _ in range(n_games):
        home, away = rng.choice(teams, size=2, replace=False)
        h_str = team_strength[home]
        a_str = team_strength[away]

        # Home-court advantage: ~3 points
        home_advantage = 3.0

        # Simulate box-score stats from strength + noise
        home_pts = h_str + home_advantage + rng.normal(0, 10)
        away_pts = a_str + rng.normal(0, 10)

        home_reb = 42 + (h_str - 90) * 0.15 + rng.normal(0, 4)
        away_reb = 42 + (a_str - 90) * 0.15 + rng.normal(0, 4)
        home_ast = 24 + (h_str - 90) * 0.12 + rng.normal(0, 3)
        away_ast = 24 + (a_str - 90) * 0.12 + rng.normal(0, 3)
        home_tov = 14 - (h_str - 90) * 0.05 + rng.normal(0, 2)
        away_tov = 14 - (a_str - 90) * 0.05 + rng.normal(0, 2)
        home_fg_pct = 0.46 + (h_str - 90) * 0.002 + rng.normal(0, 0.03)
        away_fg_pct = 0.46 + (a_str - 90) * 0.002 + rng.normal(0, 0.03)
        home_3p_pct = 0.36 + (h_str - 90) * 0.001 + rng.normal(0, 0.04)
        away_3p_pct = 0.36 + (a_str - 90) * 0.001 + rng.normal(0, 0.04)
        home_ft_pct = 0.78 + rng.normal(0, 0.04)
        away_ft_pct = 0.78 + rng.normal(0, 0.04)

        # Recent form: probability of winning each of last-5 games is
        # proportional to strength
        home_last5_wins = int(rng.binomial(5, np.clip(h_str / 130, 0.15, 0.85)))
        away_last5_wins = int(rng.binomial(5, np.clip(a_str / 130, 0.15, 0.85)))

        # Opponent stats (season averages allowed)
        home_opp_pts = 110 - (h_str - 90) * 0.3 + rng.normal(0, 3)
        away_opp_pts = 110 - (a_str - 90) * 0.3 + rng.normal(0, 3)

        home_win = int(home_pts > away_pts)

        records.append(
            {
                "home_team": home,
                "away_team": away,
                "home_points": round(home_pts, 1),
                "away_points": round(away_pts, 1),
                "home_rebounds": round(home_reb, 1),
                "away_rebounds": round(away_reb, 1),
                "home_assists": round(home_ast, 1),
                "away_assists": round(away_ast, 1),
                "home_turnovers": round(home_tov, 1),
                "away_turnovers": round(away_tov, 1),
                "home_fg_pct": round(np.clip(home_fg_pct, 0.30, 0.60), 3),
                "away_fg_pct": round(np.clip(away_fg_pct, 0.30, 0.60), 3),
                "home_3p_pct": round(np.clip(home_3p_pct, 0.20, 0.50), 3),
                "away_3p_pct": round(np.clip(away_3p_pct, 0.20, 0.50), 3),
                "home_ft_pct": round(np.clip(home_ft_pct, 0.60, 0.95), 3),
                "away_ft_pct": round(np.clip(away_ft_pct, 0.60, 0.95), 3),
                "home_last5_wins": home_last5_wins,
                "away_last5_wins": away_last5_wins,
                "home_opp_pts_avg": round(home_opp_pts, 1),
                "away_opp_pts_avg": round(away_opp_pts, 1),
                "home_win": home_win,
            }
        )

    return pd.DataFrame(records)


def load_data(csv_path: str = "nba_games.csv") -> pd.DataFrame:
    """Load NBA game data from CSV, falling back to synthetic data.

    Expected CSV columns (at minimum):
        home_team, away_team, home_points, away_points,
        home_rebounds, away_rebounds, home_assists, away_assists,
        home_turnovers, away_turnovers, home_fg_pct, away_fg_pct,
        home_3p_pct, away_3p_pct, home_ft_pct, away_ft_pct,
        home_last5_wins, away_last5_wins,
        home_opp_pts_avg, away_opp_pts_avg, home_win
    """
    if os.path.isfile(csv_path):
        print(f"[INFO] Loading data from {csv_path}")
        df = pd.read_csv(csv_path)
        # Ensure the target column exists
        if "home_win" not in df.columns and "home_points" in df.columns:
            df["home_win"] = (df["home_points"] > df["away_points"]).astype(int)
        return df

    print("[INFO] CSV not found. Generating synthetic sample data (2000 games).")
    print("       To use real data, place 'nba_games.csv' in the working directory.\n")
    return generate_sample_data(n_games=2000)


# ---------------------------------------------------------------------------
# 2. PREPROCESSING & FEATURE ENGINEERING
# ---------------------------------------------------------------------------

# Features that the model will use (all differentials + individual form stats)
DIFFERENTIAL_FEATURES = [
    "pts_diff",
    "reb_diff",
    "ast_diff",
    "tov_diff",
    "fg_pct_diff",
    "three_pct_diff",
    "ft_pct_diff",
    "opp_pts_diff",
    "form_diff",
]

EXTRA_FEATURES = [
    "home_last5_wins",
    "away_last5_wins",
]

ALL_FEATURES = DIFFERENTIAL_FEATURES + EXTRA_FEATURES


def preprocess(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Clean the data and engineer differential features.

    Returns:
        X: DataFrame of features
        y: Series of target labels (1 = home win, 0 = home loss)
    """
    df = df.copy()

    # --- Handle missing values ---
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    # --- Feature engineering: home - away differentials ---
    df["pts_diff"] = df["home_points"] - df["away_points"]
    df["reb_diff"] = df["home_rebounds"] - df["away_rebounds"]
    df["ast_diff"] = df["home_assists"] - df["away_assists"]
    df["tov_diff"] = df["home_turnovers"] - df["away_turnovers"]
    df["fg_pct_diff"] = df["home_fg_pct"] - df["away_fg_pct"]
    df["three_pct_diff"] = df["home_3p_pct"] - df["away_3p_pct"]
    df["ft_pct_diff"] = df["home_ft_pct"] - df["away_ft_pct"]
    df["opp_pts_diff"] = df["home_opp_pts_avg"] - df["away_opp_pts_avg"]
    df["form_diff"] = df["home_last5_wins"] - df["away_last5_wins"]

    X = df[ALL_FEATURES].copy()
    y = df["home_win"].copy()

    return X, y


def split_and_scale(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.20,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """Split into train/test and scale features.

    Returns:
        X_train, X_test, y_train, y_test, fitted_scaler
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    return X_train_s, X_test_s, y_train.values, y_test.values, scaler


# ---------------------------------------------------------------------------
# 3. MODEL TRAINING & EVALUATION
# ---------------------------------------------------------------------------


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_type: str = "random_forest",
) -> object:
    """Train a classifier with cross-validated hyperparameter search.

    Supported model_type values:
        - 'logistic_regression'
        - 'random_forest'  (default)
        - 'gradient_boosting'
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    if model_type == "logistic_regression":
        model = LogisticRegression(max_iter=1000, random_state=42)
        param_grid = {
            "C": [0.01, 0.1, 1, 10],
            "penalty": ["l2"],
        }
    elif model_type == "gradient_boosting":
        model = GradientBoostingClassifier(random_state=42)
        param_grid = {
            "n_estimators": [100, 200],
            "max_depth": [3, 5],
            "learning_rate": [0.05, 0.1],
        }
    else:  # random_forest
        model = RandomForestClassifier(random_state=42)
        param_grid = {
            "n_estimators": [100, 200],
            "max_depth": [5, 10, None],
            "min_samples_split": [2, 5],
        }

    search = GridSearchCV(
        model,
        param_grid,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1,
        verbose=0,
    )
    search.fit(X_train, y_train)

    print(f"  Best params : {search.best_params_}")
    print(f"  Best CV AUC : {search.best_score_:.4f}")

    return search.best_estimator_


def evaluate_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    """Evaluate the model and return a dict of metrics."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob),
    }

    print("\n  === Test-Set Evaluation ===")
    print(f"  Accuracy  : {metrics['accuracy']:.4f}")
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}")
    print(f"  ROC AUC   : {metrics['roc_auc']:.4f}")
    print()
    print(classification_report(y_test, y_pred, target_names=["Away Win", "Home Win"]))

    return metrics


def train_and_evaluate(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> Tuple[object, dict]:
    """Train all three model types, pick the best by ROC AUC, and return it."""
    best_model = None
    best_metrics = {"roc_auc": 0.0}
    best_name = ""

    for name in ["logistic_regression", "random_forest", "gradient_boosting"]:
        print(f"\n--- Training: {name} ---")
        model = train_model(X_train, y_train, model_type=name)
        metrics = evaluate_model(model, X_test, y_test)
        if metrics["roc_auc"] > best_metrics["roc_auc"]:
            best_model = model
            best_metrics = metrics
            best_name = name

    print(f"\n*** Best model: {best_name} (AUC={best_metrics['roc_auc']:.4f}) ***\n")
    return best_model, best_metrics


# ---------------------------------------------------------------------------
# 4. BETTING SIMULATION
# ---------------------------------------------------------------------------


def american_to_decimal(american_odds: int) -> float:
    """Convert American odds to decimal odds.

    Examples:
        -150  ->  1.6667   (risk 150 to win 100)
        +130  ->  2.30     (risk 100 to win 130)
    """
    if american_odds < 0:
        return 1 + (100 / abs(american_odds))
    else:
        return 1 + (american_odds / 100)


def calculate_ev(prob_win: float, decimal_odds: float) -> float:
    """Calculate expected value of a $1 bet.

    EV = (prob_win * net_profit_if_win) - (prob_loss * stake)
       = prob_win * (decimal_odds - 1) - (1 - prob_win)
    """
    return prob_win * (decimal_odds - 1) - (1 - prob_win)


def implied_probability(decimal_odds: float) -> float:
    """Calculate the bookmaker's implied probability from decimal odds."""
    return 1.0 / decimal_odds


def simulate_bet(
    model,
    scaler: StandardScaler,
    game_features: dict,
    home_odds_american: int = -150,
    away_odds_american: int = 130,
) -> dict:
    """Run a single-game betting simulation.

    Parameters:
        model:                trained classifier
        scaler:               fitted StandardScaler
        game_features:        dict with keys matching ALL_FEATURES
        home_odds_american:   American odds for home win (e.g., -150)
        away_odds_american:   American odds for away win (e.g., +130)

    Returns:
        dict with probabilities, EVs, and recommendation
    """
    feature_values = np.array([[game_features[f] for f in ALL_FEATURES]])
    feature_values_scaled = scaler.transform(feature_values)

    prob_home = model.predict_proba(feature_values_scaled)[0][1]
    prob_away = 1 - prob_home

    home_decimal = american_to_decimal(home_odds_american)
    away_decimal = american_to_decimal(away_odds_american)

    ev_home = calculate_ev(prob_home, home_decimal)
    ev_away = calculate_ev(prob_away, away_decimal)

    implied_home = implied_probability(home_decimal)
    implied_away = implied_probability(away_decimal)

    # Determine recommendation
    recommendations = []
    if ev_home > 0:
        recommendations.append(
            f"BET HOME (EV={ev_home:+.4f}, model={prob_home:.1%} vs implied={implied_home:.1%})"
        )
    if ev_away > 0:
        recommendations.append(
            f"BET AWAY (EV={ev_away:+.4f}, model={prob_away:.1%} vs implied={implied_away:.1%})"
        )
    if not recommendations:
        recommendations.append("NO VALUE BET — pass on this game")

    return {
        "prob_home_win": prob_home,
        "prob_away_win": prob_away,
        "home_odds_decimal": home_decimal,
        "away_odds_decimal": away_decimal,
        "implied_prob_home": implied_home,
        "implied_prob_away": implied_away,
        "ev_home": ev_home,
        "ev_away": ev_away,
        "recommendations": recommendations,
    }


def simulate_betting_on_test_set(
    model,
    scaler: StandardScaler,
    X_test: np.ndarray,
    y_test: np.ndarray,
    bankroll: float = 1000.0,
    bet_fraction: float = 0.02,
) -> dict:
    """Back-test the model over the test set with flat-fraction Kelly staking.

    For each game in the test set, generates random plausible odds, then
    bets only when EV > 0.  Returns summary statistics.
    """
    rng = np.random.default_rng(99)
    balance = bankroll
    n_bets = 0
    n_wins = 0
    total_wagered = 0.0

    probas = model.predict_proba(X_test)[:, 1]

    for i in range(len(X_test)):
        prob_home = probas[i]

        # Simulate a market with ~5% vig
        true_home_dec = 1.0 / prob_home if prob_home > 0.05 else 20.0
        vig_shift = rng.uniform(0.90, 0.97)  # bookmaker edge
        offered_home_dec = true_home_dec * vig_shift

        ev = calculate_ev(prob_home, offered_home_dec)

        if ev > 0:
            stake = balance * bet_fraction
            total_wagered += stake
            n_bets += 1
            actual_home_win = y_test[i] == 1

            if actual_home_win:
                balance += stake * (offered_home_dec - 1)
                n_wins += 1
            else:
                balance -= stake

    roi = ((balance - bankroll) / total_wagered * 100) if total_wagered > 0 else 0
    return {
        "starting_bankroll": bankroll,
        "ending_bankroll": round(balance, 2),
        "profit": round(balance - bankroll, 2),
        "total_bets": n_bets,
        "winning_bets": n_wins,
        "win_rate": round(n_wins / n_bets * 100, 1) if n_bets else 0,
        "roi_pct": round(roi, 2),
    }


# ---------------------------------------------------------------------------
# 5. PREDICTION ON NEW GAMES
# ---------------------------------------------------------------------------


def predict_game(
    model,
    scaler: StandardScaler,
    home_team: str,
    away_team: str,
    game_features: dict,
    home_odds: int = -150,
    away_odds: int = 130,
) -> None:
    """Pretty-print a prediction and betting recommendation for one game."""
    result = simulate_bet(model, scaler, game_features, home_odds, away_odds)

    print("=" * 60)
    print(f"  {home_team} (Home) vs {away_team} (Away)")
    print("=" * 60)
    print(f"  Model probability — Home win : {result['prob_home_win']:.1%}")
    print(f"  Model probability — Away win : {result['prob_away_win']:.1%}")
    print(f"  Bookmaker implied  — Home    : {result['implied_prob_home']:.1%}")
    print(f"  Bookmaker implied  — Away    : {result['implied_prob_away']:.1%}")
    print(f"  EV (Home bet)                : {result['ev_home']:+.4f}")
    print(f"  EV (Away bet)                : {result['ev_away']:+.4f}")
    print()
    for rec in result["recommendations"]:
        print(f"  >> {rec}")
    print("=" * 60)
    print()


# ---------------------------------------------------------------------------
# 6. MAIN ENTRY POINT
# ---------------------------------------------------------------------------


def main() -> None:
    print("=" * 60)
    print("  NBA Betting Prediction Model")
    print("  (Educational purposes only — gamble responsibly)")
    print("=" * 60)
    print()

    # --- Step 1: Load data ---
    csv_path = "nba_games.csv"
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    df = load_data(csv_path)
    print(f"  Loaded {len(df)} games  |  Home win rate: {df['home_win'].mean():.1%}\n")

    # --- Step 2: Preprocess ---
    X, y = preprocess(df)
    X_train, X_test, y_train, y_test, scaler = split_and_scale(X, y)
    print(f"  Train: {len(X_train)} games  |  Test: {len(X_test)} games\n")

    # --- Step 3: Train & evaluate all models, pick best ---
    best_model, best_metrics = train_and_evaluate(X_train, X_test, y_train, y_test)

    # --- Step 4: Back-test betting simulation on test set ---
    print("\n--- Betting Back-Test (test set) ---")
    bt = simulate_betting_on_test_set(best_model, scaler, X_test, y_test)
    print(f"  Starting bankroll : ${bt['starting_bankroll']:,.2f}")
    print(f"  Ending bankroll   : ${bt['ending_bankroll']:,.2f}")
    print(f"  Profit / Loss     : ${bt['profit']:+,.2f}")
    print(f"  Total bets placed : {bt['total_bets']}")
    print(f"  Win rate          : {bt['win_rate']}%")
    print(f"  ROI               : {bt['roi_pct']}%")
    print()

    # --- Step 5: Example single-game prediction ---
    print("\n--- Example Prediction ---")
    example_features = {
        "pts_diff": 5.2,        # Home team averages 5.2 more PPG
        "reb_diff": 2.1,        # Home team averages 2.1 more RPG
        "ast_diff": 1.5,        # Home team averages 1.5 more APG
        "tov_diff": -0.8,       # Home team has fewer turnovers (good)
        "fg_pct_diff": 0.02,    # Home team shoots 2% better from field
        "three_pct_diff": 0.01, # Home team shoots 1% better from three
        "ft_pct_diff": 0.03,    # Home team shoots 3% better from FT line
        "opp_pts_diff": -3.0,   # Home team allows 3 fewer PPG (better D)
        "form_diff": 2,         # Home team won 2 more of last 5 games
        "home_last5_wins": 4,   # Home team won 4 of last 5
        "away_last5_wins": 2,   # Away team won 2 of last 5
    }

    predict_game(
        model=best_model,
        scaler=scaler,
        home_team="Boston Celtics",
        away_team="Los Angeles Lakers",
        game_features=example_features,
        home_odds=-150,   # Favored at -150
        away_odds=130,    # Underdog at +130
    )

    # Second example: closer matchup
    close_features = {
        "pts_diff": 0.5,
        "reb_diff": -0.3,
        "ast_diff": 0.2,
        "tov_diff": 0.1,
        "fg_pct_diff": 0.005,
        "three_pct_diff": -0.005,
        "ft_pct_diff": 0.01,
        "opp_pts_diff": -0.5,
        "form_diff": 0,
        "home_last5_wins": 3,
        "away_last5_wins": 3,
    }

    predict_game(
        model=best_model,
        scaler=scaler,
        home_team="Denver Nuggets",
        away_team="Phoenix Suns",
        game_features=close_features,
        home_odds=-110,
        away_odds=-110,
    )

    print("=" * 60)
    print("  REMINDER: This model is for educational purposes only.")
    print("  Never risk money you cannot afford to lose.")
    print("=" * 60)


if __name__ == "__main__":
    main()
