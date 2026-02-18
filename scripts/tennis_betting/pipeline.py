"""
End-to-End Tennis Betting Model Pipeline.

Stages
------
1. DATA        — Download ATP matches (2015-2025) + historical odds
2. ELO         — Compute global & surface Elo ratings for all players
3. FEATURES    — Build no-look-ahead feature matrix
4. TRAIN       — Train XGBoost model with time-based validation
5. REGISTRY    — Build & persist the player registry for prediction
6. DEMO        — Show current Elo leaderboard + example predictions

Run:
    cd scripts/tennis_betting && python pipeline.py
Or with forced re-download:
    python pipeline.py --force-download
"""
import os
import sys
import json
import time
import argparse

# Allow imports from this directory
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)

from config import MODELS_DIR, RESULTS_DIR, DATA_DIR
from data_acquisition import load_all_data
from elo_calculator import compute_elo_ratings, elo_summary
from feature_engineering import build_feature_matrix
from model_training import train_model, load_model
from predict import PlayerRegistry, predict_match, print_prediction


def banner(text: str):
    print("\n" + "═" * 60)
    print(f"  {text}")
    print("═" * 60)


def run_pipeline(force_download: bool = False, skip_train: bool = False):
    t0 = time.time()

    os.makedirs(MODELS_DIR,  exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(DATA_DIR,    exist_ok=True)

    # ── Stage 1: Data ──────────────────────────────────────────────────────────
    banner("STAGE 1 / 5 — DATA ACQUISITION")
    matches = load_all_data(force_download=force_download)

    # ── Stage 2: Elo ──────────────────────────────────────────────────────────
    banner("STAGE 2 / 5 — ELO RATING COMPUTATION")
    matches_with_elo, elo_global, elo_surface = compute_elo_ratings(matches)
    print(f"Elo computed for {len(elo_global):,} players")

    # Save Elo table
    elo_df = elo_summary(matches_with_elo, top_n=50)
    elo_path = os.path.join(RESULTS_DIR, "elo_rankings.csv")
    elo_df.to_csv(elo_path, index=False)
    print(f"Elo rankings saved → {elo_path}")

    print("\nTop 20 players by current Elo:")
    print(elo_df.head(20).to_string(index=False))

    # ── Stage 3: Features ─────────────────────────────────────────────────────
    banner("STAGE 3 / 5 — FEATURE ENGINEERING")
    feature_df = build_feature_matrix(matches_with_elo)

    feat_path = os.path.join(DATA_DIR, "tennis_features.csv")
    feature_df.to_csv(feat_path, index=False)
    print(f"Feature matrix saved → {feat_path}")
    print(f"Target balance: {feature_df['p1_wins'].mean():.3f} "
          f"(should be ~0.50 due to random flip)")

    # ── Stage 4: Training ─────────────────────────────────────────────────────
    if not skip_train:
        banner("STAGE 4 / 5 — MODEL TRAINING")
        model, medians, metrics = train_model(
            feature_df,
            calibrate=True,
            model_type="xgb",
        )
    else:
        banner("STAGE 4 / 5 — MODEL TRAINING (skipped — loading existing)")
        model, medians = load_model()
        metrics = {}

    # ── Stage 5: Player Registry ───────────────────────────────────────────────
    banner("STAGE 5 / 5 — BUILDING PLAYER REGISTRY")
    registry = PlayerRegistry()
    registry.build(matches_with_elo)
    reg_path = os.path.join(MODELS_DIR, "player_registry.pkl")
    registry.save(reg_path)
    print(f"Player registry saved → {reg_path}")

    # ── Demo predictions ───────────────────────────────────────────────────────
    banner("DEMO — EXAMPLE PREDICTIONS (2026 Season)")
    _run_demo_predictions(registry, model, medians)

    elapsed = time.time() - t0
    print(f"\n{'═'*60}")
    print(f"  Pipeline complete in {elapsed/60:.1f} minutes")
    print(f"  Model → {MODELS_DIR}/tennis_model.pkl")
    print(f"  Use predict.py for interactive predictions")
    print(f"{'═'*60}\n")


def _run_demo_predictions(registry, model, medians):
    """
    Demo predictions for high-profile current ATP match-ups.
    Uses real current top players based on 2025/2026 ATP season.
    """
    demo_matches = [
        {
            "p1": "Jannik Sinner", "p2": "Carlos Alcaraz",
            "surface": "Hard", "level": "G", "round": "F", "best_of": 5,
            "tournament": "Australian Open 2026 Final",
            "odds_p1": 1.80, "odds_p2": 2.10,
        },
        {
            "p1": "Novak Djokovic", "p2": "Alexander Zverev",
            "surface": "Clay", "level": "G", "round": "SF", "best_of": 5,
            "tournament": "Roland Garros 2026 Semi-Final",
            "odds_p1": 1.65, "odds_p2": 2.40,
        },
        {
            "p1": "Daniil Medvedev", "p2": "Stefanos Tsitsipas",
            "surface": "Hard", "level": "M", "round": "QF", "best_of": 3,
            "tournament": "Indian Wells Masters 2026",
            "odds_p1": 1.75, "odds_p2": 2.20,
        },
        {
            "p1": "Carlos Alcaraz", "p2": "Holger Rune",
            "surface": "Grass", "level": "G", "round": "SF", "best_of": 5,
            "tournament": "Wimbledon 2026 Semi-Final",
            "odds_p1": 1.40, "odds_p2": 3.10,
        },
        {
            "p1": "Jannik Sinner", "p2": "Taylor Fritz",
            "surface": "Hard", "level": "A", "round": "SF", "best_of": 3,
            "tournament": "Miami Open 2026",
            "odds_p1": 1.55, "odds_p2": 2.60,
        },
    ]

    all_results = []
    for m in demo_matches:
        print(f"\n  {m['tournament']}:")
        print(f"    {m['p1']} vs {m['p2']}  [{m['surface']}]")
        try:
            result = predict_match(
                player1=m["p1"], player2=m["p2"],
                surface=m["surface"], tourney_level=m.get("level", "A"),
                round_name=m.get("round", "QF"), best_of=m.get("best_of", 3),
                odds_p1=m.get("odds_p1"), odds_p2=m.get("odds_p2"),
                registry=registry, model=model, medians=medians,
            )
            print_prediction(result)
            all_results.append({
                "tournament": m["tournament"],
                "player1": result["player1"],
                "player2": result["player2"],
                "surface": result["surface"],
                "p1_win_prob": result["p1_win_prob"],
                "p2_win_prob": result["p2_win_prob"],
                "predicted_winner": result["player1"] if result["p1_win_prob"] > 0.5 else result["player2"],
                "confidence": max(result["p1_win_prob"], result["p2_win_prob"]),
                "recommendation": result.get("bet", {}).get("recommendation", "N/A"),
            })
        except Exception as e:
            print(f"    [WARN] Prediction failed: {e}")

    # Save demo results
    if all_results:
        import pandas as pd
        out_path = os.path.join(RESULTS_DIR, "demo_predictions.csv")
        pd.DataFrame(all_results).to_csv(out_path, index=False)
        print(f"\nDemo predictions saved → {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Tennis Betting Model Pipeline")
    parser.add_argument("--force-download", action="store_true",
                        help="Force re-download of all data files")
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip model training (use existing saved model)")
    parser.add_argument("--predict", action="store_true",
                        help="After pipeline, launch interactive predictor")
    args = parser.parse_args()

    run_pipeline(
        force_download=args.force_download,
        skip_train=args.skip_train,
    )

    if args.predict:
        from predict import interactive_predict
        model, medians = load_model()
        reg_path = os.path.join(MODELS_DIR, "player_registry.pkl")
        registry = PlayerRegistry.load(reg_path)
        interactive_predict(registry, model, medians)


if __name__ == "__main__":
    main()
