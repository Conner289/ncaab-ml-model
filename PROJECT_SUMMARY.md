# NCAAB ML Model - Project Summary

## Overview

This document provides a comprehensive summary of the Men's College Basketball machine learning model project, including what has been built, how to use it, and recommended next steps for production deployment.

## What Has Been Built

### 1. Data Pipeline

The project includes a complete data acquisition and processing pipeline that:

- Downloads historical NCAAB team statistics from Kaggle (2013-2024)
- Processes 3,885 team-season records with 20+ advanced metrics per team
- Engineers 10+ differential features for game-level predictions
- Creates a unified training dataset with 58,275 synthetic games

**Key Features Engineered:**
- Efficiency differential (net offensive/defensive efficiency gap)
- BARTHAG differential (power rating gap)
- Offensive and defensive matchup scores
- Tempo, shooting, turnover, and rebound differentials
- Win percentage differential

### 2. Machine Learning Models

Three separate models have been trained for the three betting markets:

| Market | Model Type | Accuracy | F1 Score | File |
|--------|-----------|----------|----------|------|
| **Game Winner** | Logistic Regression | 100.0% | 100.0% | `models/model_winner.pkl` |
| **Spread Cover** | Gradient Boosting | 99.9% | 99.9% | `models/model_spread.pkl` |
| **Over/Under** | Logistic Regression | 99.8% | 97.8% | `models/model_over.pkl` |

Each model is accompanied by a StandardScaler for feature normalization.

**Important Note:** These performance metrics are based on synthetic game data created from team statistics. In production, the model should be retrained with actual game results and historical betting odds.

### 3. Scripts and Tools

The project includes three main Python scripts:

1. **`scripts/build_training_dataset.py`**
   - Loads team statistics from multiple seasons
   - Creates synthetic game matchups
   - Engineers features and target variables
   - Outputs `data/training_data.csv`

2. **`scripts/train_models.py`**
   - Trains multiple model types (Logistic Regression, Random Forest, Gradient Boosting)
   - Evaluates performance with multiple metrics
   - Saves the best model for each market
   - Generates detailed performance reports

3. **`scripts/predict.py`**
   - Loads trained models
   - Makes predictions for new games
   - Includes example predictions

### 4. GitHub Repository

All code has been pushed to GitHub:
- Repository: https://github.com/Conner289/ncaab-ml-model
- Includes comprehensive README with installation and usage instructions
- Includes data source documentation
- Includes trained models and evaluation results

### 5. Automation (Partially Complete)

A GitHub Actions workflow has been created for automated model retraining:
- File: `.github/workflows/retrain_model.yml`
- Scheduled to run weekly during basketball season
- Automatically downloads latest data, retrains models, and commits updates

**Note:** Due to GitHub permissions, you will need to manually add this workflow file through the GitHub web interface.

## How to Use the Model

### Quick Start

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Conner289/ncaab-ml-model.git
   cd ncaab-ml-model
   ```

2. **Set up the environment:**
   ```bash
   python3.11 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Make predictions:**
   ```bash
   python scripts/predict.py
   ```

### Making Predictions for Real Games

To make predictions for actual games, you need to:

1. Obtain current season statistics for both teams
2. Update the `home_stats` and `away_stats` dictionaries in `scripts/predict.py`
3. Run the prediction script

**Example:**
```python
home_stats = {
    'TEAM': 'Duke',
    'G': 25,
    'W': 20,
    'ADJOE': 120.5,
    'ADJDE': 92.3,
    'BARTHAG': 0.95,
    # ... (all other required stats)
}

away_stats = {
    'TEAM': 'North Carolina',
    'G': 25,
    'W': 18,
    'ADJOE': 118.2,
    'ADJDE': 94.1,
    'BARTHAG': 0.92,
    # ... (all other required stats)
}
```

## Current Limitations

### 1. Synthetic Game Data

The model is currently trained on synthetic game outcomes generated from team statistics, not actual historical game results. This means the performance metrics are artificially high.

**Impact:** The model will likely perform worse on real games than the training metrics suggest.

**Solution:** Integrate actual historical game results from sources like ESPN API, sports-reference.com, or The Odds API.

### 2. No Historical Betting Odds

The model does not have access to historical betting lines (spreads, totals, moneylines), which are critical for training accurate betting market models.

**Impact:** The model predicts raw outcomes (who will win, by how much) rather than predicting against the market.

**Solution:** Purchase historical odds data from The Odds API, BigDataBall, or SportsInsights, or scrape from public sources.

### 3. No Temporal Features

The model uses season-level statistics but does not account for:
- Recent form (last 5-10 games)
- Win/loss streaks
- Days of rest
- Time of season (early vs. late)

**Impact:** The model may miss important momentum and fatigue factors.

**Solution:** Add rolling window features and time-based indicators.

### 4. No Player-Level Data

The model uses team-level statistics only and does not incorporate:
- Player injuries
- Lineup changes
- Individual player matchups

**Impact:** The model cannot adjust for key player absences or lineup changes.

**Solution:** Integrate player-level data and injury reports.

## Recommended Next Steps

### Phase 1: Data Integration (High Priority)

1. **Integrate Actual Game Results**
   - Source: ESPN API, sports-reference.com, or saiemgilani/hoopR
   - Replace synthetic games with real historical scores
   - Retrain models on actual outcomes

2. **Add Historical Betting Odds**
   - Source: The Odds API (historical endpoint), BigDataBall, or SportsInsights
   - Add pre-game spreads, totals, and moneylines as features
   - Train models to predict against the market

**Estimated Cost:** $10-50 for historical odds data (one-time purchase)

### Phase 2: Feature Enhancement (Medium Priority)

1. **Add Temporal Features**
   - Calculate rolling averages (last 5, 10 games)
   - Add win/loss streak indicators
   - Include days of rest

2. **Add Contextual Features**
   - Home/away/neutral splits
   - Conference game indicator
   - Rivalry game indicator
   - Tournament vs. regular season

3. **Add Advanced Metrics**
   - Four Factors (shooting, turnovers, rebounding, free throws)
   - Strength of schedule adjustments
   - Opponent-adjusted statistics

### Phase 3: Model Improvement (Medium Priority)

1. **Hyperparameter Tuning**
   - Use GridSearchCV or RandomizedSearchCV
   - Optimize for log loss or Brier score (better for probability predictions)

2. **Ensemble Methods**
   - Combine predictions from multiple models
   - Use stacking or blending techniques

3. **Calibration**
   - Calibrate probability outputs for better confidence estimates
   - Use Platt scaling or isotonic regression

### Phase 4: Deployment (Low Priority)

1. **Build a Web Application**
   - Create a Flask or FastAPI backend
   - Build a simple frontend for predictions
   - Deploy to Heroku, AWS, or DigitalOcean

2. **Integrate Live Data**
   - Connect to ESPN API or similar for current season stats
   - Automate daily prediction generation
   - Send predictions via email or Telegram bot

3. **Track Performance**
   - Log all predictions and actual outcomes
   - Calculate ROI and other betting metrics
   - Continuously monitor model performance

### Phase 5: Advanced Features (Future)

1. **Player-Level Modeling**
   - Integrate player statistics and injury data
   - Model lineup effects

2. **Live In-Game Predictions**
   - Use play-by-play data
   - Update predictions in real-time

3. **Multi-Sport Expansion**
   - Apply the same framework to NBA, NFL, etc.

## Cost Estimates

| Item | Cost | Priority |
|------|------|----------|
| Historical Odds Data (one-time) | $10-50 | High |
| The Odds API (monthly, for live odds) | $10-30/month | Medium |
| Web Hosting (Heroku/DigitalOcean) | $5-15/month | Low |
| Domain Name | $10-15/year | Low |

**Total Initial Investment:** $20-80
**Total Monthly Cost (if deployed):** $15-45/month

## Performance Tracking

To evaluate the model in production, track these metrics:

1. **Accuracy**: Percentage of correct predictions
2. **ROI**: Return on investment (assuming flat betting)
3. **Sharpe Ratio**: Risk-adjusted return
4. **Log Loss**: Probability calibration quality
5. **Brier Score**: Accuracy of probability predictions

## Disclaimer

This model is for educational and research purposes only. Sports betting involves financial risk, and this model does not guarantee profitable outcomes. Always gamble responsibly and never bet more than you can afford to lose.

## Contact and Support

For questions, issues, or collaboration:
- GitHub: https://github.com/Conner289/ncaab-ml-model
- Open an issue on GitHub for bug reports or feature requests

---

**Project Status:** Proof-of-Concept Complete ✅

**Next Milestone:** Integrate actual game results and historical odds data

**Estimated Time to Production:** 2-4 weeks (with data integration)
