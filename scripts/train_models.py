"""
Train Machine Learning Models for NCAAB Predictions

This script trains and evaluates models for three prediction markets:
1. Game Winner (binary classification)
2. Spread Cover (binary classification)
3. Over/Under (binary classification)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix
import joblib
import os
import json

# Paths
DATA_PATH = 'data/training_data.csv'
MODELS_DIR = 'models'
RESULTS_DIR = 'results'

# Create directories
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

def load_data():
    """Load the training dataset."""
    print("Loading training data...")
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} games")
    return df

def prepare_features(df, target_col):
    """Prepare features and target for modeling."""
    # Select feature columns
    feature_cols = [col for col in df.columns if col.startswith('home_') or col.startswith('away_') or col.endswith('_diff')]
    
    # Remove non-numeric columns
    feature_cols = [col for col in feature_cols if df[col].dtype in ['int64', 'float64']]
    
    X = df[feature_cols]
    y = df[target_col]
    
    # Handle missing values
    X = X.fillna(X.mean())
    
    print(f"Features: {len(feature_cols)}")
    print(f"Target: {target_col}")
    print(f"Class distribution: {y.value_counts().to_dict()}")
    
    return X, y, feature_cols

def train_and_evaluate_model(X_train, X_test, y_train, y_test, model_name, model, feature_names):
    """Train a model and evaluate its performance."""
    print(f"\n--- Training {model_name} ---")
    
    # Train
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    results = {
        'model': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }
    
    if y_pred_proba is not None:
        try:
            auc = roc_auc_score(y_test, y_pred_proba)
            results['auc'] = auc
        except:
            results['auc'] = None
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    if results.get('auc'):
        print(f"AUC: {results['auc']:.4f}")
    
    # Feature importance (if available)
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 Most Important Features:")
        print(feature_importance.head(10).to_string(index=False))
        
        results['feature_importance'] = feature_importance.to_dict('records')
    
    return model, results

def train_market_models(df, target_col, market_name):
    """Train multiple models for a specific market."""
    print("\n" + "=" * 60)
    print(f"Training models for: {market_name}")
    print("=" * 60)
    
    # Prepare data
    X, y, feature_names = prepare_features(df, target_col)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save scaler
    scaler_path = os.path.join(MODELS_DIR, f'scaler_{market_name}.pkl')
    joblib.dump(scaler, scaler_path)
    print(f"Saved scaler to {scaler_path}")
    
    # Define models
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
    }
    
    # Train and evaluate each model
    results = []
    best_model = None
    best_score = 0
    
    for model_name, model in models.items():
        trained_model, model_results = train_and_evaluate_model(
            X_train_scaled, X_test_scaled, y_train, y_test, model_name, model, feature_names
        )
        
        results.append(model_results)
        
        # Track best model
        if model_results['accuracy'] > best_score:
            best_score = model_results['accuracy']
            best_model = trained_model
            best_model_name = model_name
    
    # Save best model
    model_path = os.path.join(MODELS_DIR, f'model_{market_name}.pkl')
    joblib.dump(best_model, model_path)
    print(f"\nSaved best model ({best_model_name}) to {model_path}")
    
    # Save results
    results_path = os.path.join(RESULTS_DIR, f'results_{market_name}.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {results_path}")
    
    return results

def generate_summary_report(all_results):
    """Generate a summary report of all model performances."""
    print("\n" + "=" * 60)
    print("MODEL PERFORMANCE SUMMARY")
    print("=" * 60)
    
    summary = []
    
    for market, results in all_results.items():
        print(f"\n{market}:")
        print("-" * 40)
        
        for result in results:
            print(f"  {result['model']:20s} | Accuracy: {result['accuracy']:.4f} | F1: {result['f1_score']:.4f}")
            summary.append({
                'market': market,
                'model': result['model'],
                'accuracy': result['accuracy'],
                'f1_score': result['f1_score']
            })
    
    # Save summary
    summary_df = pd.DataFrame(summary)
    summary_path = os.path.join(RESULTS_DIR, 'model_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSaved summary to {summary_path}")
    
    return summary_df

def main():
    """Main execution function."""
    print("=" * 60)
    print("NCAAB ML Model - Training Pipeline")
    print("=" * 60)
    
    # Load data
    df = load_data()
    
    # Train models for each market
    all_results = {}
    
    # Market 1: Game Winner
    all_results['Game Winner'] = train_market_models(df, 'target_winner', 'winner')
    
    # Market 2: Spread Cover
    all_results['Spread Cover'] = train_market_models(df, 'target_spread_cover', 'spread')
    
    # Market 3: Over/Under
    all_results['Over/Under'] = train_market_models(df, 'target_over', 'over')
    
    # Generate summary report
    summary = generate_summary_report(all_results)
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Review model performance in the 'results/' directory")
    print("2. Use the trained models in 'models/' directory for predictions")
    print("3. Integrate with live data sources for real-time predictions")

if __name__ == "__main__":
    main()
