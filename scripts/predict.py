"""
Make Predictions Using Trained NCAAB Models

This script loads the trained models and makes predictions for new games.
"""

import pandas as pd
import numpy as np
import joblib
import os
import argparse

# Paths
MODELS_DIR = 'models'

def load_models():
    """Load all trained models and scalers."""
    print("Loading models...")
    
    models = {
        'winner': {
            'model': joblib.load(os.path.join(MODELS_DIR, 'model_winner.pkl')),
            'scaler': joblib.load(os.path.join(MODELS_DIR, 'scaler_winner.pkl'))
        },
        'spread': {
            'model': joblib.load(os.path.join(MODELS_DIR, 'model_spread.pkl')),
            'scaler': joblib.load(os.path.join(MODELS_DIR, 'scaler_spread.pkl'))
        },
        'over': {
            'model': joblib.load(os.path.join(MODELS_DIR, 'model_over.pkl')),
            'scaler': joblib.load(os.path.join(MODELS_DIR, 'scaler_over.pkl'))
        }
    }
    
    print("Models loaded successfully")
    return models

def prepare_game_features(home_stats, away_stats):
    """
    Prepare features for a single game given home and away team statistics.
    
    Args:
        home_stats: Dictionary of home team statistics
        away_stats: Dictionary of away team statistics
    
    Returns:
        DataFrame with engineered features
    """
    # Create base features
    features = {}
    
    # Add home team stats
    for key, value in home_stats.items():
        features[f'home_{key.lower()}'] = value
    
    # Add away team stats
    for key, value in away_stats.items():
        features[f'away_{key.lower()}'] = value
    
    # Engineer differential features
    features['efficiency_diff'] = (home_stats['ADJOE'] - home_stats['ADJDE']) - (away_stats['ADJOE'] - away_stats['ADJDE'])
    features['barthag_diff'] = home_stats['BARTHAG'] - away_stats['BARTHAG']
    features['offensive_matchup'] = home_stats['ADJOE'] - away_stats['ADJDE']
    features['defensive_matchup'] = away_stats['ADJOE'] - home_stats['ADJDE']
    features['tempo_diff'] = home_stats['ADJ_T'] - away_stats['ADJ_T']
    features['efg_diff'] = (home_stats['EFG_O'] - home_stats['EFG_D']) - (away_stats['EFG_O'] - away_stats['EFG_D'])
    features['turnover_diff'] = (home_stats['TORD'] - home_stats['TOR']) - (away_stats['TORD'] - away_stats['TOR'])
    features['rebound_diff'] = (home_stats['ORB'] + home_stats['DRB']) - (away_stats['ORB'] + away_stats['DRB'])
    features['three_point_diff'] = (home_stats['3P_O'] - home_stats['3P_D']) - (away_stats['3P_O'] - away_stats['3P_D'])
    features['win_pct_diff'] = (home_stats['W'] / home_stats['G']) - (away_stats['W'] / away_stats['G'])
    
    return pd.DataFrame([features])

def predict_game(models, home_stats, away_stats):
    """
    Make predictions for a single game.
    
    Args:
        models: Dictionary of loaded models
        home_stats: Dictionary of home team statistics
        away_stats: Dictionary of away team statistics
    
    Returns:
        Dictionary of predictions
    """
    # Prepare features
    features_df = prepare_game_features(home_stats, away_stats)
    
    predictions = {}
    
    # Predict for each market
    for market_name, model_dict in models.items():
        # Scale features
        features_scaled = model_dict['scaler'].transform(features_df)
        
        # Make prediction
        prediction = model_dict['model'].predict(features_scaled)[0]
        probability = model_dict['model'].predict_proba(features_scaled)[0] if hasattr(model_dict['model'], 'predict_proba') else None
        
        predictions[market_name] = {
            'prediction': int(prediction),
            'probability': probability.tolist() if probability is not None else None
        }
    
    return predictions

def format_predictions(home_team, away_team, predictions):
    """Format predictions for display."""
    print("\n" + "=" * 60)
    print(f"PREDICTIONS: {home_team} vs {away_team}")
    print("=" * 60)
    
    # Game Winner
    winner_pred = predictions['winner']
    winner = home_team if winner_pred['prediction'] == 1 else away_team
    winner_prob = winner_pred['probability'][1] if winner_pred['probability'] else None
    
    print(f"\n🏀 Game Winner: {winner}")
    if winner_prob:
        print(f"   Confidence: {winner_prob:.2%}")
    
    # Spread Cover
    spread_pred = predictions['spread']
    spread_result = "Home team covers" if spread_pred['prediction'] == 1 else "Away team covers"
    spread_prob = spread_pred['probability'][1] if spread_pred['probability'] else None
    
    print(f"\n📊 Spread: {spread_result}")
    if spread_prob:
        print(f"   Confidence: {spread_prob:.2%}")
    
    # Over/Under
    over_pred = predictions['over']
    over_result = "Over" if over_pred['prediction'] == 1 else "Under"
    over_prob = over_pred['probability'][1] if over_pred['probability'] else None
    
    print(f"\n🎯 Total: {over_result}")
    if over_prob:
        print(f"   Confidence: {over_prob:.2%}")
    
    print("\n" + "=" * 60)

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Predict NCAAB game outcomes')
    parser.add_argument('--home-team', type=str, help='Home team name')
    parser.add_argument('--away-team', type=str, help='Away team name')
    args = parser.parse_args()
    
    # Load models
    models = load_models()
    
    # Example prediction (replace with actual team stats)
    print("\n" + "=" * 60)
    print("EXAMPLE PREDICTION")
    print("=" * 60)
    print("\nNote: This is a demonstration using sample team statistics.")
    print("In production, you would fetch current team stats from a data source.")
    
    # Sample team statistics (Duke vs North Carolina)
    home_stats = {
        'TEAM': 'Duke',
        'G': 30,
        'W': 25,
        'ADJOE': 120.5,
        'ADJDE': 92.3,
        'BARTHAG': 0.95,
        'EFG_O': 55.2,
        'EFG_D': 46.8,
        'TOR': 16.5,
        'TORD': 19.2,
        'ORB': 32.1,
        'DRB': 25.4,
        'FTR': 32.5,
        'FTRD': 28.3,
        '2P_O': 54.3,
        '2P_D': 45.2,
        '3P_O': 37.8,
        '3P_D': 32.1,
        'ADJ_T': 68.5,
        'WAB': 8.2
    }
    
    away_stats = {
        'TEAM': 'North Carolina',
        'G': 30,
        'W': 23,
        'ADJOE': 118.2,
        'ADJDE': 94.1,
        'BARTHAG': 0.92,
        'EFG_O': 53.8,
        'EFG_D': 48.2,
        'TOR': 17.2,
        'TORD': 18.5,
        'ORB': 35.2,
        'DRB': 23.8,
        'FTR': 30.1,
        'FTRD': 29.5,
        '2P_O': 52.9,
        '2P_D': 46.8,
        '3P_O': 35.2,
        '3P_D': 34.5,
        'ADJ_T': 70.2,
        'WAB': 7.1
    }
    
    # Make predictions
    predictions = predict_game(models, home_stats, away_stats)
    
    # Format and display
    format_predictions(home_stats['TEAM'], away_stats['TEAM'], predictions)
    
    print("\n📝 To use this script with your own data:")
    print("   1. Update the home_stats and away_stats dictionaries")
    print("   2. Or integrate with a live data API")
    print("   3. Run: python scripts/predict.py")

if __name__ == "__main__":
    main()
