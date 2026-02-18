"""
Model Training for the Tennis Betting Model.

Strategy
--------
* Time-based train/test split: train on TRAIN_START_YEAR–TRAIN_END_YEAR,
  hold out TEST_START_YEAR+ for evaluation (no data leakage).
* Primary model: XGBoost gradient-boosted trees (best for tabular sport data).
* Probability calibration: isotonic regression to fix over-confidence.
* Evaluation metrics: accuracy, ROC-AUC, Brier score, log-loss,
  and a simulated flat-stake return-on-investment using model edge.
* Feature importance is saved for interpretability.
"""
import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, log_loss,
    brier_score_loss, classification_report,
)

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("[WARN] XGBoost not installed. Falling back to GradientBoosting.")

from config import (
    TRAIN_END_YEAR, TEST_START_YEAR, RANDOM_STATE, CV_FOLDS,
    XGBOOST_PARAMS, MODELS_DIR, RESULTS_DIR,
)
from feature_engineering import FEATURE_COLS, TARGET_COL, fit_impute


# ── Data splitting ─────────────────────────────────────────────────────────────

def time_split(feature_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split feature matrix into train / test by year.
    Train: up to TRAIN_END_YEAR (inclusive).
    Test : TEST_START_YEAR onwards.
    """
    train = feature_df[feature_df["date"].dt.year <= TRAIN_END_YEAR]
    test  = feature_df[feature_df["date"].dt.year >= TEST_START_YEAR]
    print(f"Train: {len(train):,} matches | Test: {len(test):,} matches")
    print(f"  Train: {train['date'].min().date()} → {train['date'].max().date()}")
    if len(test):
        print(f"  Test : {test['date'].min().date()} → {test['date'].max().date()}")
    return train, test


# ── Model builders ─────────────────────────────────────────────────────────────

def _build_xgb() -> object:
    params = {k: v for k, v in XGBOOST_PARAMS.items()
              if k != "use_label_encoder"}  # deprecated param
    params.pop("use_label_encoder", None)
    return XGBClassifier(**params, verbosity=0)


def _build_lr() -> object:
    return LogisticRegression(
        C=1.0, max_iter=1000, solver="lbfgs", random_state=RANDOM_STATE
    )


def _build_rf() -> object:
    return RandomForestClassifier(
        n_estimators=300, max_depth=10,
        min_samples_leaf=5, random_state=RANDOM_STATE, n_jobs=-1
    )


def _build_gb() -> object:
    return GradientBoostingClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, random_state=RANDOM_STATE
    )


# ── Training ───────────────────────────────────────────────────────────────────

def train_model(
    feature_df: pd.DataFrame,
    calibrate: bool = True,
    model_type: str = "xgb",
) -> Tuple[object, np.ndarray, dict]:
    """
    Train the primary tennis match-outcome model.

    Parameters
    ----------
    feature_df : processed feature matrix (output of build_feature_matrix)
    calibrate  : whether to apply isotonic probability calibration
    model_type : "xgb" | "lr" | "rf" | "gb"

    Returns
    -------
    model         : trained (calibrated) classifier
    medians       : imputation medians from training set
    eval_metrics  : dict of evaluation metrics on held-out test set
    """
    os.makedirs(MODELS_DIR,  exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ── Split ──────────────────────────────────────────────────────────────────
    train_df, test_df = time_split(feature_df)

    X_train_raw = train_df[FEATURE_COLS]
    X_test_raw  = test_df[FEATURE_COLS]  if len(test_df) else None
    y_train     = train_df[TARGET_COL].values
    y_test      = test_df[TARGET_COL].values if len(test_df) else None

    # ── Impute ─────────────────────────────────────────────────────────────────
    if X_test_raw is not None and len(X_test_raw):
        X_train, X_test, medians = fit_impute(X_train_raw, X_test_raw)
    else:
        medians = X_train_raw.median(numeric_only=True)
        X_train = X_train_raw.fillna(medians)
        X_test  = None

    # ── Build base model ───────────────────────────────────────────────────────
    if model_type == "xgb" and XGBOOST_AVAILABLE:
        base = _build_xgb()
    elif model_type == "rf":
        base = _build_rf()
    elif model_type == "lr":
        base = _build_lr()
    else:
        base = _build_gb()

    print(f"\nTraining {base.__class__.__name__} on {len(X_train):,} samples …")

    # ── Cross-validation on training set ──────────────────────────────────────
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_aucs = cross_val_score(base, X_train, y_train, cv=cv,
                              scoring="roc_auc", n_jobs=-1)
    print(f"  CV ROC-AUC: {cv_aucs.mean():.4f} ± {cv_aucs.std():.4f}")

    # ── Final fit on all training data ────────────────────────────────────────
    if calibrate:
        model = CalibratedClassifierCV(base, method="isotonic", cv=3)
    else:
        model = base

    model.fit(X_train, y_train)

    # ── Evaluate on test set ──────────────────────────────────────────────────
    eval_metrics: Dict = {
        "cv_auc_mean": float(cv_aucs.mean()),
        "cv_auc_std":  float(cv_aucs.std()),
        "model_type":  model_type,
        "n_train":     int(len(X_train)),
        "n_test":      int(len(X_test)) if X_test is not None else 0,
        "calibrated":  calibrate,
        "feature_cols": FEATURE_COLS,
    }

    if X_test is not None and y_test is not None and len(X_test) > 0:
        y_pred     = model.predict(X_test)
        y_prob     = model.predict_proba(X_test)[:, 1]

        acc  = accuracy_score(y_test, y_pred)
        auc  = roc_auc_score(y_test, y_prob)
        brier= brier_score_loss(y_test, y_prob)
        ll   = log_loss(y_test, y_prob)

        print(f"\n  Test-set results:")
        print(f"    Accuracy  : {acc:.4f}")
        print(f"    ROC-AUC   : {auc:.4f}")
        print(f"    Brier     : {brier:.4f}")
        print(f"    Log-loss  : {ll:.4f}")

        # ── Simulated flat-stake betting ROI ───────────────────────────────────
        roi = _simulate_flat_stake(y_test, y_prob, test_df)
        print(f"    Flat-stake ROI (edge>4%): {roi:.2%}")

        eval_metrics.update({
            "test_accuracy": float(acc),
            "test_auc":      float(auc),
            "test_brier":    float(brier),
            "test_logloss":  float(ll),
            "flat_stake_roi":float(roi),
        })
        print("\n" + classification_report(y_test, y_pred, target_names=["P2 wins", "P1 wins"]))

        # ── Feature importance ─────────────────────────────────────────────────
        fi = _get_feature_importance(model, FEATURE_COLS)
        if fi:
            eval_metrics["feature_importance"] = fi
            print("\n  Top-15 features:")
            for feat, imp in list(fi.items())[:15]:
                bar = "█" * int(imp * 400)
                print(f"    {feat:<30} {imp:.4f}  {bar}")

    # ── Persist ───────────────────────────────────────────────────────────────
    model_path   = os.path.join(MODELS_DIR,  "tennis_model.pkl")
    medians_path = os.path.join(MODELS_DIR,  "tennis_medians.pkl")
    results_path = os.path.join(RESULTS_DIR, "tennis_eval.json")

    joblib.dump(model,   model_path)
    joblib.dump(medians, medians_path)
    with open(results_path, "w") as f:
        json.dump({k: (v if not isinstance(v, np.ndarray) else v.tolist())
                   for k, v in eval_metrics.items()}, f, indent=2)

    print(f"\nModel saved   → {model_path}")
    print(f"Results saved → {results_path}")

    return model, medians, eval_metrics


def _simulate_flat_stake(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    test_df: pd.DataFrame,
    min_edge: float = 0.04,
    stake: float = 1.0,
) -> float:
    """
    Simulate flat-stake betting: bet on p1 when model probability >
    implied probability (from sportsbook odds) by at least min_edge.

    If odds columns are available use them; otherwise fall back to
    a synthetic closing line derived from Elo probability.
    """
    profits = []
    for i in range(len(y_true)):
        model_prob = float(y_prob[i])

        # Try real odds first (B365W = decimal odds for p1)
        if "B365W" in test_df.columns:
            row = test_df.iloc[i]
            odds = row.get("B365W")
            if pd.notna(odds) and float(odds) > 1.0:
                implied = 1.0 / float(odds)
                edge = model_prob - implied
                if abs(edge) >= min_edge:
                    if edge > 0:   # bet on p1
                        profit = stake * (float(odds) - 1) if y_true[i] == 1 else -stake
                    else:          # bet on p2
                        odds2 = row.get("B365L")
                        if pd.notna(odds2) and float(odds2) > 1.0:
                            profit = stake * (float(odds2) - 1) if y_true[i] == 0 else -stake
                        else:
                            continue
                    profits.append(profit)
                continue

        # Fall back to Elo-based synthetic odds
        elo_diff = test_df["elo_diff"].iloc[i] if "elo_diff" in test_df.columns else 0
        p_elo    = 1.0 / (1.0 + 10 ** (-float(elo_diff) / 400.0))
        edge     = model_prob - p_elo
        if abs(edge) >= min_edge:
            fair_odds = 1.0 / p_elo if edge > 0 else 1.0 / (1.0 - p_elo)
            vig_odds  = fair_odds * 0.95
            if vig_odds > 1.0:
                if edge > 0:
                    profit = stake * (vig_odds - 1) if y_true[i] == 1 else -stake
                else:
                    fair_p2 = 1.0 - p_elo
                    odds2   = (1.0 / fair_p2) * 0.95
                    profit  = stake * (odds2 - 1) if y_true[i] == 0 else -stake
                profits.append(profit)

    if not profits:
        return 0.0
    total_staked = len(profits) * stake
    return sum(profits) / total_staked if total_staked > 0 else 0.0


def _get_feature_importance(model, feature_cols) -> Optional[Dict]:
    """Extract and sort feature importances from the model."""
    # Try unwrapping CalibratedClassifierCV
    base = model
    if hasattr(model, "calibrated_classifiers_"):
        estimators = [c.estimator for c in model.calibrated_classifiers_]
        base = estimators[0] if estimators else model
    elif hasattr(model, "base_estimator"):
        base = model.base_estimator
    elif hasattr(model, "estimator"):
        base = model.estimator

    importance = None
    if hasattr(base, "feature_importances_"):
        importance = base.feature_importances_
    elif hasattr(base, "coef_"):
        importance = np.abs(base.coef_[0])

    if importance is None:
        return None

    fi = dict(zip(feature_cols, importance.tolist()))
    fi = dict(sorted(fi.items(), key=lambda x: -x[1]))
    return fi


# ── Loading ───────────────────────────────────────────────────────────────────

def load_model():
    """Load the trained model and imputation medians from disk."""
    model_path   = os.path.join(MODELS_DIR, "tennis_model.pkl")
    medians_path = os.path.join(MODELS_DIR, "tennis_medians.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No trained model found at {model_path}. "
            "Run the pipeline first: python pipeline.py"
        )

    model   = joblib.load(model_path)
    medians = joblib.load(medians_path)
    return model, medians


def predict_proba(model, medians: pd.Series, X: pd.DataFrame) -> np.ndarray:
    """Apply saved medians then return class-1 probabilities."""
    X_imp = X[FEATURE_COLS].fillna(medians)
    return model.predict_proba(X_imp)[:, 1]
