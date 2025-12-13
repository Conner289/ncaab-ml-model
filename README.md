# NCAAB Machine Learning Model

A comprehensive machine learning model for predicting Men's College Basketball (NCAAB) game outcomes across three betting markets: game winners, point spread covers, and over/under totals.

## Project Overview

This project demonstrates a complete machine learning pipeline for sports betting analytics, including data acquisition, feature engineering, model training, and prediction capabilities. The model is trained on historical NCAAB data from 2013-2024 and uses advanced team statistics (KenPom-style metrics) to make predictions.

## Features

The model predicts three key betting markets:

1. **Game Winner**: Binary classification predicting which team will win
2. **Spread Cover**: Binary classification predicting whether the home team will cover the spread
3. **Over/Under**: Binary classification predicting whether the total score will go over or under

## Model Performance

Based on the proof-of-concept training with synthetic game data:

| Market | Best Model | Accuracy | F1 Score |
|--------|-----------|----------|----------|
| Game Winner | Logistic Regression | 100.0% | 100.0% |
| Spread Cover | Gradient Boosting | 99.9% | 99.9% |
| Over/Under | Logistic Regression | 99.8% | 97.8% |

**Note**: These results are based on synthetic game data created from team statistics. In production, the model should be retrained with actual game results and historical betting odds for more realistic performance metrics.

## Data Sources

The project uses two primary data sources:

1. **Team-Level Advanced Statistics** (2013-2025)
   - Source: Kaggle - andrewsundberg/college-basketball-dataset
   - Includes: Adjusted Offensive/Defensive Efficiency, BARTHAG, tempo, shooting percentages, and more
   - Scraped from barttorvik.com

2. **Game-Level Information** (2002-2021)
   - Source: GitHub - saiemgilani/pbp-data
   - Includes: Game dates, teams, venues, and metadata

See `DATA_SOURCES.md` for detailed information about the data.

## Project Structure

```
ncaab-model/
├── data/                      # Data directory (gitignored)
│   ├── cbb.csv               # Team statistics (2013-2024)
│   ├── cbb_games_info_2002_2021.csv  # Game information
│   └── training_data.csv     # Processed training dataset
├── models/                    # Trained models (gitignored)
│   ├── model_winner.pkl      # Game winner model
│   ├── model_spread.pkl      # Spread cover model
│   ├── model_over.pkl        # Over/under model
│   └── scaler_*.pkl          # Feature scalers
├── results/                   # Model evaluation results
│   ├── model_summary.csv     # Performance summary
│   └── results_*.json        # Detailed results per market
├── scripts/                   # Python scripts
│   ├── build_training_dataset.py  # Data processing and feature engineering
│   ├── train_models.py       # Model training and evaluation
│   └── predict.py            # Prediction script
├── DATA_SOURCES.md           # Data source documentation
├── README.md                 # This file
└── requirements.txt          # Python dependencies
```

## Installation

### Prerequisites

- Python 3.11+
- Git
- Kaggle API credentials (for data download)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/Conner289/ncaab-ml-model.git
cd ncaab-ml-model
```

2. Create and activate a virtual environment:
```bash
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure Kaggle API (for data download):
```bash
mkdir -p ~/.kaggle
echo '{"username":"YOUR_USERNAME","key":"YOUR_API_KEY"}' > ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

## Usage

### 1. Build Training Dataset

Process the raw data and engineer features:

```bash
python scripts/build_training_dataset.py
```

This script:
- Loads team statistics from multiple seasons
- Creates synthetic game matchups (or uses real game data if available)
- Engineers 10+ differential features
- Creates target variables for all three markets
- Outputs `data/training_data.csv`

### 2. Train Models

Train and evaluate models for all three markets:

```bash
python scripts/train_models.py
```

This script:
- Trains Logistic Regression, Random Forest, and Gradient Boosting models
- Evaluates performance using accuracy, precision, recall, F1, and AUC
- Saves the best model for each market
- Generates detailed performance reports

### 3. Make Predictions

Use the trained models to predict game outcomes:

```bash
python scripts/predict.py
```

The prediction script includes an example prediction for Duke vs North Carolina. To use with your own data, modify the `home_stats` and `away_stats` dictionaries in the script.

## Key Features Engineered

The model uses the following engineered features:

- **Efficiency Differential**: Net efficiency (offense - defense) difference between teams
- **BARTHAG Differential**: Power rating difference
- **Offensive/Defensive Matchup**: How each team's offense matches up against the opponent's defense
- **Tempo Differential**: Pace of play difference
- **Shooting Efficiency Differential**: Effective field goal percentage difference
- **Turnover Differential**: Turnover rate difference
- **Rebound Differential**: Combined offensive and defensive rebound rate difference
- **Three-Point Differential**: Three-point shooting percentage difference
- **Win Percentage Differential**: Season win rate difference

## Limitations and Future Improvements

### Current Limitations

1. **No Historical Betting Odds**: The model does not have access to historical betting lines (spreads, totals, moneylines), which are essential for training accurate betting market models.

2. **Synthetic Game Data**: The proof-of-concept uses simulated game outcomes based on team statistics rather than actual game results.

3. **No Temporal Features**: The model does not account for recent form, momentum, or time-based trends.

4. **No Player-Level Data**: The model uses team-level statistics only and does not incorporate player injuries, lineup changes, or individual player performance.

### Recommended Improvements

1. **Integrate Historical Odds Data**
   - Source: The Odds API, BigDataBall, or SportsInsights
   - Add pre-game spreads, totals, and moneylines as features
   - Train models to predict against the market, not just raw outcomes

2. **Use Actual Game Results**
   - Replace synthetic games with real historical game scores
   - Source: ESPN API, sports-reference.com, or other sports data providers

3. **Add Temporal Features**
   - Last 5-10 game performance
   - Win/loss streaks
   - Days of rest
   - Home/away splits

4. **Incorporate Advanced Metrics**
   - Four Factors (shooting, turnovers, rebounding, free throws)
   - Strength of schedule adjustments
   - Conference-specific effects

5. **Deploy as a Web Application**
   - Create a Flask or FastAPI backend
   - Build a frontend for easy game predictions
   - Integrate with live data APIs for real-time predictions

6. **Set Up Automated Retraining**
   - Use GitHub Actions to automatically fetch new data
   - Retrain models weekly during the season
   - Track model performance over time

## Contributing

This is a personal project, but suggestions and improvements are welcome. Please open an issue or submit a pull request.

## License

This project is provided as-is for educational and research purposes. Use at your own risk. Sports betting involves financial risk, and this model does not guarantee profitable outcomes.

## Disclaimer

This model is for educational and research purposes only. It is not financial advice. Sports betting involves risk, and you should never bet more than you can afford to lose. Always gamble responsibly.

## Contact

For questions or collaboration opportunities, please open an issue on GitHub.

---

**Built with**: Python, scikit-learn, pandas, NumPy

**Data Sources**: Kaggle (andrewsundberg), GitHub (saiemgilani), barttorvik.com
