import pandas as pd
import os

## Path
RAW_PATH = "data/raw/supply_chain_data.csv"
PROCESSED_PATH = "data/processed/supply_chain_clean.csv"

def load_raw_data():
    df = pd.read_csv(RAW_PATH)
    print(f"✅ Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

def clean_data(df):
    # Show nulls
    print(f"\n📊 Null values:\n{df.isnull().sum()}")

     # Drop nulls
    df = df.dropna()
    print(f"\n✅ After dropping nulls: {df.shape[0]} rows")

     # Clean column names
    df.columns = (df.columns
                  .str.strip()
                  .str.lower()
                  .str.replace(' ', '_'))
    print(f"\n✅ Column names cleaned")
    print(f"\n📋 Columns:\n{list(df.columns)}")

    return df

def save_data(df):
    os.makedirs("data/processed", exist_ok=True)
    df.to_csv(PROCESSED_PATH, index=False)
    print(f"\n✅ Saved to: {PROCESSED_PATH}")

if __name__ == "__main__":
    df = load_raw_data()
    df = clean_data(df)
    save_data(df)