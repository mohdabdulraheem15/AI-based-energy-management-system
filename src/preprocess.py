"""
Block 3 — Data Ingestion & Preprocessing
Downloads the REFIT dataset from Kaggle and prepares it for ML pipeline.
"""

import os
import pandas as pd
import kagglehub


# ─────────────────────────────────────────────────────────────
# STEP 1: Download dataset from Kaggle
# ─────────────────────────────────────────────────────────────
def download_dataset():
    """Downloads the REFIT UK Electrical Load dataset via kagglehub."""
    print("⬇️  Downloading dataset from Kaggle...")
    path = kagglehub.dataset_download("kyleahmurphy/uk-electrical-load")
    print(f"✅ Dataset downloaded to: {path}")
    return path


# ─────────────────────────────────────────────────────────────
# STEP 2: List available house files
# ─────────────────────────────────────────────────────────────
def list_house_files(dataset_path):
    """Lists all CSV house files in the downloaded dataset folder."""
    files = sorted([f for f in os.listdir(dataset_path) if f.endswith('.csv')])
    print(f"\n📂 Found {len(files)} CSV files:")
    for f in files:
        print(f"   → {f}")
    return files


# ─────────────────────────────────────────────────────────────
# STEP 3: Load a single house CSV
# ─────────────────────────────────────────────────────────────
# REFIT Cleaned CSV structure:
#   Time        - Timestamp (DD/MM/YYYY HH:MM:SS)
#   Aggregate   - Total house power consumption (Watts)
#   Appliance1  - e.g., Fridge
#   Appliance2  - e.g., Freezer
#   Appliance3  - e.g., Washing Machine
#   Appliance4  - e.g., Dishwasher
#   Appliance5  - e.g., Tumble Dryer
#   Appliance6  - e.g., Computer
#   Appliance7  - e.g., Television
#   Appliance8  - e.g., Electric Shower
#   Appliance9  - e.g., Microwave
#   Issues      - Flag: 1 if sum of appliances > aggregate (sensor issue)

def load_house(dataset_path, house_file="House_1.csv"):
    filepath = os.path.join(dataset_path, house_file)

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"❌ File not found: {filepath}")

    df = pd.read_csv(filepath)

    # Parse Time — handle both string and Unix timestamp formats
    if 'Time' in df.columns:
        df['Time'] = pd.to_datetime(df['Time'], format='ISO8601')
    elif 'Unix' in df.columns:
        df['Time'] = pd.to_datetime(df['Unix'], unit='s')

    # Drop Unix column (not needed for ML)
    if 'Unix' in df.columns:
        df = df.drop(columns=['Unix'])

    return df   # ✅ THIS LINE FIXES EVERYTHING


# ─────────────────────────────────────────────────────────────
# STEP 4: Clean the DataFrame
# ─────────────────────────────────────────────────────────────
def clean(df):
    """
    Removes invalid rows, fills missing appliance values,
    and drops metadata columns not needed for ML.
    """
    initial_rows = len(df)

    # Drop rows where Aggregate is 0 or NaN (meter offline / invalid)
    df = df[df['Aggregate'] > 0].dropna(subset=['Aggregate'])

    # Fill missing appliance readings with 0 (sensor not reporting = off)
    appliance_cols = [c for c in df.columns if 'Appliance' in c]
    df[appliance_cols] = df[appliance_cols].fillna(0)

    # Remove negative appliance values (sensor errors)
    for col in appliance_cols:
        df[col] = df[col].clip(lower=0)

    # Drop the Issues flag column (not useful for ML)
    if 'Issues' in df.columns:
        df = df.drop(columns=['Issues'])

    # Sort chronologically
    df = df.sort_values('Time').reset_index(drop=True)

    removed = initial_rows - len(df)
    print(f"\n🧹 Cleaned: removed {removed:,} invalid rows → {len(df):,} rows remaining")
    return df


# ─────────────────────────────────────────────────────────────
# STEP 5: Add time-based features
# ─────────────────────────────────────────────────────────────
def add_time_features(df):
    """Extracts temporal features from the Time column."""
    df = df.copy()
    df['hour']        = df['Time'].dt.hour
    df['minute']      = df['Time'].dt.minute
    df['day_of_week'] = df['Time'].dt.dayofweek   # 0=Monday, 6=Sunday
    df['month']       = df['Time'].dt.month
    df['is_weekend']  = (df['day_of_week'] >= 5).astype(int)
    # Night: 11 PM to 5 AM
    df['is_night']    = ((df['hour'] >= 23) | (df['hour'] < 5)).astype(int)
    # Peak hours: 7–9 AM and 5–9 PM
    df['is_peak']     = (
        ((df['hour'] >= 7) & (df['hour'] <= 9)) |
        ((df['hour'] >= 17) & (df['hour'] <= 21))
    ).astype(int)
    return df


# ─────────────────────────────────────────────────────────────
# STEP 6: Downsample from 8-second to 1-minute intervals
# ─────────────────────────────────────────────────────────────
def downsample(df, freq='1min'):
    """
    Resamples from 8-second intervals to a coarser frequency.
    This reduces ~10M rows to ~500K for House_1, making ML feasible.
    """
    df = df.copy()
    df['Time'] = pd.to_datetime(df['Time'])
    df = df.set_index('Time')
    numeric_cols = df.select_dtypes(include='number').columns
    df = df[numeric_cols].resample(freq).mean()
    df = df.reset_index()
    df = df.dropna(subset=['Aggregate'])
    print(f"⏬ Downsampled to {freq} intervals → {len(df):,} rows")
    return df


# ─────────────────────────────────────────────────────────────
# MAIN PIPELINE — used by all other modules
# ─────────────────────────────────────────────────────────────
def load_and_prepare(house_file="House_1.csv", resample=True, freq='1min', dataset_path=None):
    """
    Full preprocessing pipeline:
      1. Download (or use cached) dataset
      2. Load specified house file
      3. Clean data
      4. Optionally downsample
      5. Add time features
    
    Returns a clean, feature-rich DataFrame ready for ML.
    """
    if dataset_path is None:
        dataset_path = download_dataset()

    list_house_files(dataset_path)
    df = load_house(dataset_path, house_file)
    df = clean(df)

    if resample:
        df = downsample(df, freq=freq)

    df = add_time_features(df)

    print(f"\n✅ Pipeline complete. Final shape: {df.shape}")
    print(df[['Time', 'Aggregate', 'hour', 'is_night', 'is_peak']].head(3))
    return df, dataset_path


# ─────────────────────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df, path = load_and_prepare("House_1.csv", resample=True)
    print("\n📋 Column list:", list(df.columns))
    print(df.dtypes)
