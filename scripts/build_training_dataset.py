"""
Build Training Dataset for NCAAB ML Model

This script processes the raw data sources and creates a unified training dataset
with engineered features for predicting game outcomes.
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime

# Paths
DATA_DIR = 'data'
TEAM_STATS_PATH = os.path.join(DATA_DIR, 'cbb.csv')
OUTPUT_PATH = os.path.join(DATA_DIR, 'training_data.csv')

def load_team_stats():
    """Load and prepare team-level statistics."""
    print("Loading team statistics...")
    df = pd.read_csv(TEAM_STATS_PATH)
    
    # Standardize team names (remove extra spaces, convert to title case)
    df['TEAM'] = df['TEAM'].str.strip().str.title()
    
    print(f"Loaded {len(df)} team-season records from {df['YEAR'].min()} to {df['YEAR'].max()}")
    return df

def create_synthetic_games(team_stats):
    """
    Create synthetic game data by pairing teams within the same season.
    
    In a real scenario, this would be replaced by actual game results.
    For the proof-of-concept, we'll simulate games between teams.
    """
    print("\nCreating synthetic game dataset...")
    
    games = []
    
    # Group by year
    for year in team_stats['YEAR'].unique():
        season_teams = team_stats[team_stats['YEAR'] == year].copy()
        
        # For each team, create games against a sample of other teams
        for idx, home_team in season_teams.iterrows():
            # Sample 15 opponents (simulating a partial season schedule)
            opponents = season_teams[season_teams['TEAM'] != home_team['TEAM']].sample(
                n=min(15, len(season_teams) - 1), 
                random_state=idx
            )
            
            for _, away_team in opponents.iterrows():
                # Simulate game outcome based on team strength (BARTHAG)
                home_advantage = 0.05  # 5% boost for home team
                home_win_prob = home_team['BARTHAG'] + home_advantage
                away_win_prob = away_team['BARTHAG']
                
                # Normalize probabilities
                total_prob = home_win_prob + away_win_prob
                home_win_prob /= total_prob
                
                # Determine winner
                home_wins = np.random.random() < home_win_prob
                
                # Simulate scores based on offensive/defensive efficiency
                home_expected_score = (home_team['ADJOE'] + (110 - away_team['ADJDE'])) / 2
                away_expected_score = (away_team['ADJOE'] + (110 - home_team['ADJDE'])) / 2
                
                # Add randomness
                home_score = int(np.random.normal(home_expected_score, 8))
                away_score = int(np.random.normal(away_expected_score, 8))
                
                # Ensure winner has higher score
                if home_wins and home_score <= away_score:
                    home_score = away_score + np.random.randint(1, 10)
                elif not home_wins and away_score <= home_score:
                    away_score = home_score + np.random.randint(1, 10)
                
                games.append({
                    'season': year,
                    'home_team': home_team['TEAM'],
                    'away_team': away_team['TEAM'],
                    'home_score': home_score,
                    'away_score': away_score,
                    'home_wins': 1 if home_wins else 0,
                    'score_diff': home_score - away_score,
                    'total_score': home_score + away_score
                })
    
    games_df = pd.DataFrame(games)
    print(f"Created {len(games_df)} synthetic games")
    return games_df

def merge_team_stats_with_games(games_df, team_stats):
    """Merge team statistics with game data."""
    print("\nMerging team statistics with games...")
    
    # Merge home team stats
    games_with_stats = games_df.merge(
        team_stats,
        left_on=['home_team', 'season'],
        right_on=['TEAM', 'YEAR'],
        how='left',
        suffixes=('', '_home')
    )
    
    # Rename home team columns
    home_cols = {col: f'home_{col.lower()}' for col in team_stats.columns if col not in ['TEAM', 'YEAR']}
    games_with_stats.rename(columns=home_cols, inplace=True)
    
    # Merge away team stats
    games_with_stats = games_with_stats.merge(
        team_stats,
        left_on=['away_team', 'season'],
        right_on=['TEAM', 'YEAR'],
        how='left',
        suffixes=('', '_away')
    )
    
    # Rename away team columns
    away_cols = {col: f'away_{col.lower()}' for col in team_stats.columns if col not in ['TEAM', 'YEAR']}
    games_with_stats.rename(columns=away_cols, inplace=True)
    
    # Drop redundant columns
    games_with_stats.drop(columns=['TEAM', 'YEAR', 'TEAM_away', 'YEAR_away'], inplace=True, errors='ignore')
    
    print(f"Merged dataset has {len(games_with_stats)} games with {len(games_with_stats.columns)} features")
    return games_with_stats

def engineer_features(df):
    """Create additional features for the model."""
    print("\nEngineering features...")
    
    # Efficiency differentials
    df['efficiency_diff'] = df['home_adjoe'] - df['home_adjde'] - (df['away_adjoe'] - df['away_adjde'])
    
    # Strength differential
    df['barthag_diff'] = df['home_barthag'] - df['away_barthag']
    
    # Offensive matchup
    df['offensive_matchup'] = df['home_adjoe'] - df['away_adjde']
    df['defensive_matchup'] = df['away_adjoe'] - df['home_adjde']
    
    # Tempo differential
    df['tempo_diff'] = df['home_adj_t'] - df['away_adj_t']
    
    # Shooting efficiency differential
    df['efg_diff'] = (df['home_efg_o'] - df['home_efg_d']) - (df['away_efg_o'] - df['away_efg_d'])
    
    # Turnover differential
    df['turnover_diff'] = (df['home_tord'] - df['home_tor']) - (df['away_tord'] - df['away_tor'])
    
    # Rebound differential
    df['rebound_diff'] = (df['home_orb'] + df['home_drb']) - (df['away_orb'] + df['away_drb'])
    
    # Three-point shooting differential
    df['three_point_diff'] = (df['home_3p_o'] - df['home_3p_d']) - (df['away_3p_o'] - df['away_3p_d'])
    
    # Win percentage differential
    df['win_pct_diff'] = (df['home_w'] / df['home_g']) - (df['away_w'] / df['away_g'])
    
    print(f"Added {10} engineered features")
    return df

def create_target_variables(df):
    """Create target variables for the three prediction markets."""
    print("\nCreating target variables...")
    
    # Target 1: Game Winner (already exists as 'home_wins')
    df['target_winner'] = df['home_wins']
    
    # Target 2: Spread Cover
    # Simulate a spread based on team strength differential
    df['simulated_spread'] = -1 * (df['barthag_diff'] * 20)  # Negative means home team favored
    df['target_spread_cover'] = (df['score_diff'] + df['simulated_spread']) > 0
    df['target_spread_cover'] = df['target_spread_cover'].astype(int)
    
    # Target 3: Over/Under
    # Simulate a total based on offensive efficiency
    df['simulated_total'] = (df['home_adjoe'] + df['away_adjoe']) * 0.7
    df['target_over'] = (df['total_score'] > df['simulated_total']).astype(int)
    
    print("Created 3 target variables: winner, spread_cover, over")
    return df

def save_training_data(df):
    """Save the processed training dataset."""
    print(f"\nSaving training dataset to {OUTPUT_PATH}...")
    
    # Select relevant columns
    feature_cols = [col for col in df.columns if col.startswith('home_') or col.startswith('away_') or col.endswith('_diff')]
    target_cols = ['target_winner', 'target_spread_cover', 'target_over']
    metadata_cols = ['season', 'home_team', 'away_team', 'home_score', 'away_score']
    
    final_df = df[metadata_cols + feature_cols + target_cols]
    
    final_df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(final_df)} games with {len(feature_cols)} features")
    
    # Print summary statistics
    print("\n--- Dataset Summary ---")
    print(f"Seasons: {final_df['season'].min()} to {final_df['season'].max()}")
    print(f"Total games: {len(final_df)}")
    print(f"Home team win rate: {final_df['target_winner'].mean():.2%}")
    print(f"Spread cover rate: {final_df['target_spread_cover'].mean():.2%}")
    print(f"Over rate: {final_df['target_over'].mean():.2%}")
    
    return final_df

def main():
    """Main execution function."""
    print("=" * 60)
    print("NCAAB ML Model - Training Dataset Builder")
    print("=" * 60)
    
    # Load data
    team_stats = load_team_stats()
    
    # Create synthetic games
    games_df = create_synthetic_games(team_stats)
    
    # Merge with team stats
    merged_df = merge_team_stats_with_games(games_df, team_stats)
    
    # Engineer features
    featured_df = engineer_features(merged_df)
    
    # Create targets
    final_df = create_target_variables(featured_df)
    
    # Save
    save_training_data(final_df)
    
    print("\n" + "=" * 60)
    print("Training dataset creation complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
