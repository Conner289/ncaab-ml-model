"""
Configuration for the Tennis Betting Model.

Data sources:
  - ATP match data: Jeff Sackmann's GitHub (real, authoritative data)
    https://github.com/JeffSackmann/tennis_atp
  - Historical odds: tennis-data.co.uk (free historical sportsbook odds)
  - Current rankings: Jeff Sackmann's GitHub rankings files
"""
import os

# ── Directories ────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR   = os.path.join(BASE_DIR, "data", "tennis")
MODELS_DIR = os.path.join(BASE_DIR, "models", "tennis")
RESULTS_DIR= os.path.join(BASE_DIR, "results", "tennis")

# ── Data Sources ───────────────────────────────────────────────────────────────
# Jeff Sackmann's ATP dataset (real match data, updated regularly)
ATP_MATCHES_URL  = "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_{year}.csv"
ATP_RANKINGS_URL = "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_rankings_current.csv"
ATP_PLAYERS_URL  = "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_players.csv"

# Historical betting odds from tennis-data.co.uk (free Excel/CSV files)
ODDS_URL_PATTERNS = [
    "http://www.tennis-data.co.uk/{year}/{year}.xlsx",
    "http://www.tennis-data.co.uk/{year}/atp.xlsx",
]

# ── Data Settings ──────────────────────────────────────────────────────────────
TRAIN_START_YEAR = 2015     # Start of historical data for training
TRAIN_END_YEAR   = 2023     # Last full training year
TEST_START_YEAR  = 2024     # Hold-out test period
CURRENT_YEAR     = 2025     # Most recent data to fetch

YEARS_TO_FETCH   = list(range(TRAIN_START_YEAR, CURRENT_YEAR + 1))

# Minimum matches for a player to be considered reliable
MIN_MATCHES_THRESHOLD = 10

# ── Elo Settings ───────────────────────────────────────────────────────────────
ELO_INITIAL        = 1500   # Starting Elo for new players
ELO_K_FACTOR       = 32     # Global Elo K-factor
ELO_SURFACE_K      = 24     # Surface-specific K-factor (lower = slower learning)
ELO_WEIGHT_LEVEL   = {      # K-factor multiplier by tournament level
    "G": 1.5,   # Grand Slam
    "M": 1.3,   # Masters 1000
    "F": 1.2,   # Tour Finals / ATP Finals
    "A": 1.0,   # ATP 500 / 250
    "D": 0.8,   # Davis Cup
    "C": 0.7,   # Challenger
    "S": 0.5,   # Satellite / ITF
    "O": 1.2,   # Olympics
}

# ── Feature Settings ───────────────────────────────────────────────────────────
FORM_WINDOW         = 20    # Matches for recent form calculation
SURFACE_FORM_WINDOW = 15    # Matches for surface-specific form
H2H_WINDOW          = 30    # Max H2H matches to look back

# Surfaces to track separately
SURFACES = ["Hard", "Clay", "Grass", "Carpet"]

# ── Model Settings ─────────────────────────────────────────────────────────────
RANDOM_STATE = 42
CV_FOLDS     = 5

XGBOOST_PARAMS = {
    "n_estimators":    400,
    "max_depth":       5,
    "learning_rate":   0.05,
    "subsample":       0.8,
    "colsample_bytree":0.8,
    "min_child_weight":3,
    "gamma":           0.1,
    "reg_alpha":       0.1,
    "reg_lambda":      1.0,
    "use_label_encoder": False,
    "eval_metric":     "logloss",
    "random_state":    RANDOM_STATE,
    "n_jobs":          -1,
}

# ── Betting Settings ───────────────────────────────────────────────────────────
MIN_EDGE            = 0.04   # Minimum edge (4%) to flag as value bet
KELLY_FRACTION      = 0.25   # Fraction of full Kelly to use (quarter-Kelly)
BANKROLL            = 1000   # Starting bankroll (USD)
MIN_DECIMAL_ODDS    = 1.30   # Skip very short-price favourites
MAX_DECIMAL_ODDS    = 15.0   # Skip extreme longshots

# Tournament level display names
LEVEL_NAMES = {
    "G": "Grand Slam",
    "M": "Masters 1000",
    "F": "Tour Finals",
    "A": "ATP 500/250",
    "D": "Davis Cup",
    "C": "Challenger",
    "S": "Satellite/ITF",
    "O": "Olympics",
}

# Round encoding (lower = earlier round)
ROUND_MAP = {
    "R128": 1, "R64": 2, "R32": 3, "R16": 4,
    "QF": 5, "SF": 6, "F": 7,
    "RR": 4,   # Round Robin (Tour Finals)
    "BR": 5,   # Bronze medal match
}

# Surface short codes for display
SURFACE_SHORT = {"Hard": "H", "Clay": "C", "Grass": "G", "Carpet": "P"}
