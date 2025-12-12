# NCAAB Model Data Sources

This document describes the data sources used in this project.

## 1. Game-Level Data (2002-2021)

**Source**: `cbb_games_info_2002_2021.csv` from [saiemgilani/pbp-data](https://github.com/saiemgilani/pbp-data)

**Description**: Contains game-level information for Men's College Basketball games from 2002-2021.

**Key Columns**:
- `id`: Unique game identifier
- `date`: Game date
- `home.id`, `home.name`, `home.displayName`: Home team information
- `away.id`, `away.name`, `away.displayName`: Away team information
- `season`: Season year
- `neutralSite`: Boolean indicating if the game was played at a neutral site
- `conferenceCompetition`: Boolean indicating if the game was a conference game

**Limitations**:
- Does not contain final scores (need to extract from play-by-play or other sources)
- Does not contain pre-game betting odds (spreads, totals, moneylines)

## 2. Team-Level Advanced Statistics (2013-2025)

**Source**: `cbb.csv` (and individual season files) from [Kaggle - andrewsundberg/college-basketball-dataset](https://www.kaggle.com/datasets/andrewsundberg/college-basketball-dataset)

**Description**: Contains season-level advanced statistics for Division I college basketball teams from 2013-2025, scraped from [barttorvik.com](http://barttorvik.com/trank.php#).

**Key Columns**:
- `TEAM`: Team name
- `CONF`: Conference
- `G`: Games played
- `W`: Wins
- `ADJOE`: Adjusted Offensive Efficiency
- `ADJDE`: Adjusted Defensive Efficiency
- `BARTHAG`: Power Rating (probability of beating an average D1 team)
- `EFG_O`, `EFG_D`: Effective Field Goal Percentage (Offense/Defense)
- `TOR`, `TORD`: Turnover Rate (Allowed/Committed)
- `ORB`, `DRB`: Offensive/Defensive Rebound Rate
- `FTR`, `FTRD`: Free Throw Rate (Shot/Allowed)
- `2P_O`, `2P_D`: Two-Point Shooting Percentage (Offense/Defense)
- `3P_O`, `3P_D`: Three-Point Shooting Percentage (Offense/Defense)
- `ADJ_T`: Adjusted Tempo
- `WAB`: Wins Above Bubble
- `POSTSEASON`: Tournament result
- `SEED`: Tournament seed
- `YEAR`: Season year

**Limitations**:
- Team-level data (not game-level)
- Does not contain game-by-game results or betting odds

## Data Integration Strategy

To build a comprehensive model, we need to:

1. **Merge the two datasets** by matching team names and seasons.
2. **Obtain game-level scores** from the `cbb_games_info_2002_2021.csv` (may require parsing play-by-play data or finding an alternative source).
3. **Obtain historical betting odds** (spreads, totals, moneylines) from a third-party source such as:
   - The Odds API (historical endpoint - paid)
   - BigDataBall (paid)
   - Manual scraping from historical odds websites

## Next Steps

1. Create a unified game-level dataset that includes:
   - Game date, home/away teams
   - Final scores
   - Pre-game advanced statistics for both teams (from the Kaggle dataset)
   - Pre-game betting odds (if available)

2. Engineer features such as:
   - Score differential
   - Efficiency margin (ADJOE - ADJDE for each team)
   - Recent form (last 5-10 games)
   - Head-to-head records
   - Home/away/neutral splits

3. Build and train models for three targets:
   - Game winner (binary classification)
   - Point spread cover (binary classification)
   - Over/Under (binary classification)
