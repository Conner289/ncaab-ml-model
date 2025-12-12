import pandas as pd
import os

DATA_PATH = os.path.join('data', 'cbb_games_info_2002_2021.csv')

def print_all_columns():
    """Reads the CSV and prints all column names."""
    print(f"Attempting to read data from: {DATA_PATH}")
    try:
        # Read only the header
        df = pd.read_csv(DATA_PATH, nrows=0)
        
        print("\n--- ALL COLUMN NAMES ---")
        for i, col in enumerate(df.columns):
            print(f"{i+1}: {col}")
            
    except FileNotFoundError:
        print(f"Error: File not found at {DATA_PATH}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    print_all_columns()
