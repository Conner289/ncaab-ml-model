# NCAAB Machine Learning Model

This repository contains the code and data pipeline for a comprehensive machine learning model designed to predict outcomes for Men's College Basketball (NCAAB) games.

## Project Goal

The primary goal of this project is to develop a robust predictive model capable of forecasting three key betting markets:

1.  **Game Winner** (Moneyline)
2.  **Point Spread** (Against the Spread - ATS)
3.  **Over/Under** (Total Points)

## Technology Stack

*   **Language**: Python
*   **Libraries**: pandas, scikit-learn, numpy, requests
*   **Data Sources**: Historical game data and statistics from public sources (e.g., Kaggle, NCAA API wrappers) and historical odds data (e.g., The Odds API).
*   **Automation**: GitHub Actions for scheduled data refresh and model retraining.

## Project Structure

| Directory/File | Purpose |
| :--- | :--- |
| `data/` | Stores raw and processed historical data. |
| `scripts/` | Contains Python scripts for data acquisition, cleaning, and feature engineering. |
| `model/` | Contains the trained machine learning models and model training scripts. |
| `notebooks/` | Optional: For exploratory data analysis (EDA) and initial model prototyping. |
| `predictions/` | Stores the final output of the model (daily predictions). |
| `README.md` | This file. |
| `requirements.txt` | Lists all necessary Python dependencies. |

## Getting Started

1.  Clone the repository: `git clone https://github.com/Conner289/ncaab-ml-model.git`
2.  Install dependencies: `pip install -r requirements.txt`
3.  Run the full pipeline: `python scripts/run_pipeline.py` (Once developed)

## Data Pipeline Overview

1.  **Acquisition**: Fetch historical game results, team statistics, and historical odds.
2.  **ETL**: Clean data, calculate advanced metrics (e.g., Adjusted Efficiency Margin, Pace), and merge all data sources.
3.  **Feature Engineering**: Create game-level features (e.g., Team A's last 5 games performance vs. Team B's last 5 games performance).
4.  **Modeling**: Train separate models for Moneyline, Spread, and Total.
5.  **Prediction**: Generate daily predictions for upcoming games.
