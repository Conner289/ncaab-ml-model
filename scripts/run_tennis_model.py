#!/usr/bin/env python3
"""
Tennis Betting Model — Top-Level Runner

Usage:
    # Full pipeline (download data → train → demo predictions)
    python scripts/run_tennis_model.py

    # Force fresh data download
    python scripts/run_tennis_model.py --force-download

    # Skip training, use existing model
    python scripts/run_tennis_model.py --skip-train

    # After pipeline, launch interactive predictor
    python scripts/run_tennis_model.py --predict

    # Interactive predictor only (model already trained)
    python scripts/run_tennis_model.py --predict-only

    # Show current Elo leaderboard only
    python scripts/run_tennis_model.py --leaderboard
"""
import os
import sys
import argparse

# Make sure the tennis_betting package is on the path
_SCRIPTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tennis_betting")
sys.path.insert(0, _SCRIPTS_DIR)


def main():
    parser = argparse.ArgumentParser(
        description="Tennis Betting Model — automated ATP match predictor"
    )
    parser.add_argument("--force-download", action="store_true",
                        help="Re-download all data from source (ignore cache)")
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip model training; use existing saved model")
    parser.add_argument("--predict", action="store_true",
                        help="Launch interactive predictor after pipeline")
    parser.add_argument("--predict-only", action="store_true",
                        help="Launch interactive predictor without running pipeline")
    parser.add_argument("--leaderboard", action="store_true",
                        help="Print current Elo leaderboard and exit")
    args = parser.parse_args()

    if args.leaderboard:
        _show_leaderboard()
        return

    if args.predict_only:
        _interactive_predict_only()
        return

    # Full pipeline
    from pipeline import run_pipeline
    run_pipeline(
        force_download=args.force_download,
        skip_train=args.skip_train,
    )

    if args.predict:
        _interactive_predict_only()


def _show_leaderboard():
    """Load saved Elo rankings and display them."""
    from config import RESULTS_DIR
    import pandas as pd

    elo_path = os.path.join(RESULTS_DIR, "elo_rankings.csv")
    if not os.path.exists(elo_path):
        print("No Elo rankings found. Run the pipeline first.")
        sys.exit(1)

    df = pd.read_csv(elo_path)
    print("\nCurrent ATP Elo Rankings (based on historical data):")
    print("=" * 65)
    print(df.head(30).to_string(index=False))
    print()


def _interactive_predict_only():
    """Load existing model + registry and launch interactive predictor."""
    from config import MODELS_DIR
    from model_training import load_model
    from predict import PlayerRegistry, interactive_predict

    model, medians = load_model()
    reg_path = os.path.join(MODELS_DIR, "player_registry.pkl")
    if not os.path.exists(reg_path):
        print("Player registry not found. Run the full pipeline first.")
        sys.exit(1)
    registry = PlayerRegistry.load(reg_path)
    interactive_predict(registry, model, medians)


if __name__ == "__main__":
    main()
