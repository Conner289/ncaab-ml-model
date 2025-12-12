import pandas as pd
import os

DATA_PATH = os.path.join('data', 'cbb_games_info_2002_2021.csv')

def inspect_data():
    """Reads the CSV and prints column names and info."""
    print(f"Attempting to read data from: {DATA_PATH}")
    try:
        # Read only the header and a few rows to save time
        df = pd.read_csv(DATA_PATH, nrows=5)
        
        print("\n--- Column Names ---")
        for col in df.columns:
            print(col)
            
        print("\n--- Data Info (First 5 rows) ---")
        df.info()
        
        print("\n--- Sample Data ---")
        print(df.head())
        
    except FileNotFoundError:
        print(f"Error: File not found at {DATA_PATH}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    inspect_data()
