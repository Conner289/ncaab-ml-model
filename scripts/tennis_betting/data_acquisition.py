"""
Data Acquisition for Tennis Betting Model.

Downloads real ATP match data from Jeff Sackmann's GitHub repository
and historical betting odds from tennis-data.co.uk.

Jeff Sackmann's dataset contains:
  - All ATP matches from 1968 onwards with full stats
  - Current ATP rankings
  - Player biographical information (name, hand, DOB, country)
"""
import os
import time
import requests
import pandas as pd
import numpy as np
from io import StringIO, BytesIO
from typing import List, Optional

from config import (
    ATP_MATCHES_URL, ATP_RANKINGS_URL, ATP_PLAYERS_URL,
    ODDS_URL_PATTERNS, YEARS_TO_FETCH, DATA_DIR,
    TRAIN_START_YEAR, CURRENT_YEAR,
)


# ── HTTP helpers ───────────────────────────────────────────────────────────────

def _get(url: str, timeout: int = 30, retries: int = 3) -> Optional[requests.Response]:
    """GET with retry/backoff."""
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=timeout)
            if r.status_code == 200:
                return r
        except requests.RequestException as e:
            if attempt == retries - 1:
                print(f"    [WARN] Failed to fetch {url}: {e}")
        time.sleep(1.5 ** attempt)
    return None


# ── ATP Match Data ─────────────────────────────────────────────────────────────

def download_atp_matches(years: Optional[List[int]] = None, force: bool = False) -> pd.DataFrame:
    """
    Download ATP match data from Jeff Sackmann's GitHub.

    Each row is one completed ATP match with full stats:
    winner/loser IDs, names, ranks, scores, ace/df/serve stats, surface, round, etc.

    Returns a combined DataFrame sorted by tourney_date.
    """
    if years is None:
        years = YEARS_TO_FETCH

    os.makedirs(DATA_DIR, exist_ok=True)
    all_dfs: List[pd.DataFrame] = []

    print(f"Downloading ATP matches for {years[0]}-{years[-1]}...")
    for year in years:
        cache = os.path.join(DATA_DIR, f"atp_matches_{year}.csv")
        if os.path.exists(cache) and not force:
            df = pd.read_csv(cache, low_memory=False)
            print(f"  {year}: {len(df):>5} matches (cached)")
        else:
            url = ATP_MATCHES_URL.format(year=year)
            r = _get(url)
            if r is None:
                print(f"  {year}: SKIPPED (download failed)")
                continue
            df = pd.read_csv(StringIO(r.text), low_memory=False)
            df.to_csv(cache, index=False)
            print(f"  {year}: {len(df):>5} matches (downloaded)")
            time.sleep(0.4)  # Be polite to GitHub

        df["_year"] = year
        all_dfs.append(df)

    if not all_dfs:
        raise RuntimeError("No ATP match data could be downloaded. Check network connection.")

    combined = pd.concat(all_dfs, ignore_index=True)
    print(f"  Total: {len(combined):,} raw matches\n")
    return combined


def download_atp_players(force: bool = False) -> pd.DataFrame:
    """
    Download ATP player biographical data (name, hand, DOB, country).
    Columns: player_id, first_name, last_name, hand, dob, ioc
    """
    cache = os.path.join(DATA_DIR, "atp_players.csv")
    if os.path.exists(cache) and not force:
        return pd.read_csv(cache, low_memory=False)

    os.makedirs(DATA_DIR, exist_ok=True)
    r = _get(ATP_PLAYERS_URL)
    if r is None:
        if os.path.exists(cache):
            return pd.read_csv(cache, low_memory=False)
        return pd.DataFrame()

    df = pd.read_csv(StringIO(r.text), low_memory=False)
    df["full_name"] = df["name_first"].fillna("") + " " + df["name_last"].fillna("")
    df["full_name"] = df["full_name"].str.strip()
    df.to_csv(cache, index=False)
    print(f"Players: {len(df):,} entries downloaded")
    return df


def download_current_rankings(force: bool = False) -> pd.DataFrame:
    """
    Download the latest ATP rankings from Jeff Sackmann's GitHub.
    Columns: ranking_date, rank, player, points
    """
    cache = os.path.join(DATA_DIR, "atp_rankings_current.csv")
    if os.path.exists(cache) and not force:
        df = pd.read_csv(cache)
        print(f"Rankings: {len(df):,} entries (cached, {df['ranking_date'].max()})")
        return df

    os.makedirs(DATA_DIR, exist_ok=True)
    r = _get(ATP_RANKINGS_URL)
    if r is None:
        if os.path.exists(cache):
            return pd.read_csv(cache)
        return pd.DataFrame()

    df = pd.read_csv(StringIO(r.text))
    df.to_csv(cache, index=False)
    print(f"Rankings: {len(df):,} entries downloaded")
    return df


def download_historical_odds(years: Optional[List[int]] = None, force: bool = False) -> Optional[pd.DataFrame]:
    """
    Download historical ATP betting odds from tennis-data.co.uk.
    Includes closing odds from Bet365, Pinnacle, and others.
    Returns None if no odds files can be fetched.
    """
    if years is None:
        years = list(range(2016, CURRENT_YEAR))   # odds site is slightly behind

    os.makedirs(DATA_DIR, exist_ok=True)
    all_dfs: List[pd.DataFrame] = []

    print("Downloading historical betting odds...")
    for year in years:
        cache = os.path.join(DATA_DIR, f"odds_{year}.csv")
        if os.path.exists(cache) and not force:
            df = pd.read_csv(cache, low_memory=False)
            print(f"  {year}: {len(df):>5} rows (cached)")
            all_dfs.append(df)
            continue

        downloaded = False
        for pattern in ODDS_URL_PATTERNS:
            url = pattern.format(year=year)
            r = _get(url, timeout=25)
            if r is None:
                continue
            try:
                df = pd.read_excel(BytesIO(r.content))
                df["_year"] = year
                df.to_csv(cache, index=False)
                print(f"  {year}: {len(df):>5} rows (downloaded)")
                all_dfs.append(df)
                downloaded = True
                break
            except Exception as e:
                continue

        if not downloaded:
            print(f"  {year}: SKIPPED (not available)")
        time.sleep(0.8)

    if not all_dfs:
        print("  No odds data available — model will run without betting-line calibration.\n")
        return None

    combined = pd.concat(all_dfs, ignore_index=True)
    print(f"  Total: {len(combined):,} odds rows\n")
    return combined


# ── Preprocessing ──────────────────────────────────────────────────────────────

def preprocess_matches(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and standardise the raw ATP match DataFrame.
    - Parse dates
    - Filter valid surfaces and complete matches
    - Add helper columns (retirement flag, tournament category, round number)
    - Keep all stat columns, coercing to numeric
    """
    df = df.copy()

    # Parse date (format: YYYYMMDD integer)
    df["tourney_date"] = pd.to_datetime(
        df["tourney_date"].astype(str), format="%Y%m%d", errors="coerce"
    )
    df = df.dropna(subset=["tourney_date"])

    # Required columns
    req = ["winner_id", "loser_id", "winner_name", "loser_name", "surface", "tourney_date"]
    df = df.dropna(subset=[c for c in req if c in df.columns])

    # Valid surfaces only
    df = df[df["surface"].isin(["Hard", "Clay", "Grass", "Carpet"])].copy()

    # Filter to main tour (exclude Futures/ITF for training quality)
    if "tourney_level" in df.columns:
        df = df[df["tourney_level"].isin(["G", "M", "A", "F", "D", "C", "O"])].copy()

    # Retirement / walkover flag
    df["is_retirement"] = df["score"].fillna("").str.contains(r"RET|W/O", regex=True)

    # Tournament category
    level_map = {
        "G": "Grand Slam", "M": "Masters 1000", "F": "Tour Finals",
        "A": "ATP 500/250", "D": "Davis Cup", "C": "Challenger",
        "S": "Satellite", "O": "Olympics",
    }
    level_weight = {"G": 4, "M": 3, "F": 3, "A": 2, "D": 1, "C": 1, "S": 0, "O": 2}
    if "tourney_level" in df.columns:
        df["tourney_category"] = df["tourney_level"].map(level_map).fillna("Other")
        df["level_weight"]     = df["tourney_level"].map(level_weight).fillna(1).astype(float)
    else:
        df["tourney_category"] = "Unknown"
        df["level_weight"]     = 1.0

    # Round numeric encoding
    from config import ROUND_MAP
    if "round" in df.columns:
        df["round_num"] = df["round"].map(ROUND_MAP).fillna(3).astype(int)
    else:
        df["round_num"] = 3

    # best_of numeric
    if "best_of" in df.columns:
        df["best_of"] = pd.to_numeric(df["best_of"], errors="coerce").fillna(3).astype(int)
    else:
        df["best_of"] = 3
    # Grand Slams beyond RR are always best-of-5
    if "tourney_level" in df.columns:
        df.loc[df["tourney_level"] == "G", "best_of"] = 5

    # Coerce all stat columns to numeric
    stat_cols = [
        "w_ace", "w_df", "w_svpt", "w_1stIn", "w_1stWon", "w_2ndWon",
        "w_SvGms", "w_bpSaved", "w_bpFaced",
        "l_ace", "l_df", "l_svpt", "l_1stIn", "l_1stWon", "l_2ndWon",
        "l_SvGms", "l_bpSaved", "l_bpFaced",
        "winner_rank", "winner_rank_points", "loser_rank", "loser_rank_points",
        "winner_age", "loser_age", "winner_ht", "loser_ht",
        "minutes",
    ]
    for col in stat_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Sort chronologically
    df = df.sort_values("tourney_date").reset_index(drop=True)

    print(f"Preprocessed: {len(df):,} matches | "
          f"{df['tourney_date'].min().date()} → {df['tourney_date'].max().date()}")
    if "tourney_category" in df.columns:
        cats = df["tourney_category"].value_counts()
        print("  " + " | ".join(f"{k}: {v}" for k, v in cats.items()))

    return df


def merge_odds(matches: pd.DataFrame, odds: Optional[pd.DataFrame]) -> pd.DataFrame:
    """
    Attempt to join historical odds onto match data.
    Odds file columns vary; try common naming conventions.
    Returns matches with appended odds columns (B365W, B365L, PSW, PSL).
    """
    if odds is None or odds.empty:
        return matches

    # Standardise odds column names
    rename_candidates = {
        "Winner": "odds_winner_name",
        "Loser":  "odds_loser_name",
        "Date":   "odds_date",
        "B365W":  "B365W", "B365L": "B365L",
        "PSW":    "PSW",   "PSL":   "PSL",
        "MaxW":   "MaxW",  "MaxL":  "MaxL",
        "AvgW":   "AvgW",  "AvgL":  "AvgL",
    }
    odds = odds.rename(columns={k: v for k, v in rename_candidates.items() if k in odds.columns})

    if "odds_date" not in odds.columns:
        return matches

    odds["odds_date"] = pd.to_datetime(odds["odds_date"], errors="coerce", dayfirst=True)
    odds = odds.dropna(subset=["odds_date", "odds_winner_name", "odds_loser_name"])

    # Merge on date + winner + loser (approximate)
    merged = matches.merge(
        odds[["odds_date", "odds_winner_name", "odds_loser_name",
              "B365W", "B365L", "PSW", "PSL", "MaxW", "MaxL", "AvgW", "AvgL"]
              + [c for c in ["B365W","B365L","PSW","PSL","MaxW","MaxL","AvgW","AvgL"] if c in odds.columns]],
        left_on=["tourney_date", "winner_name", "loser_name"],
        right_on=["odds_date",   "odds_winner_name", "odds_loser_name"],
        how="left",
    )
    merged = merged.drop(columns=["odds_date", "odds_winner_name", "odds_loser_name"], errors="ignore")
    n_matched = merged["B365W"].notna().sum() if "B365W" in merged.columns else 0
    print(f"Odds merge: {n_matched:,} / {len(merged):,} matches have odds data")
    return merged


def load_all_data(force_download: bool = False) -> pd.DataFrame:
    """
    Full data loading pipeline:
      1. Download ATP matches (2015-2025)
      2. Download player bios
      3. Download historical odds
      4. Preprocess & merge everything

    Returns a clean, enriched match DataFrame ready for feature engineering.
    """
    print("=" * 60)
    print("  TENNIS BETTING MODEL — DATA ACQUISITION")
    print("=" * 60)

    # 1. Match data
    raw_matches = download_atp_matches(force=force_download)

    # 2. Players
    players = download_atp_players(force=force_download)

    # 3. Historical odds
    odds = download_historical_odds(force=force_download)

    # 4. Preprocess
    matches = preprocess_matches(raw_matches)

    # 5. Attach odds
    matches = merge_odds(matches, odds)

    # 6. Save combined
    out = os.path.join(DATA_DIR, "atp_matches_combined.csv")
    matches.to_csv(out, index=False)
    print(f"\nCombined dataset saved → {out}")

    return matches
