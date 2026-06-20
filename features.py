"""
Block 4 — Feature Engineering
Generates statistical and behavioral features from raw consumption data.
These features are used by both the Isolation Forest and K-Means models.
"""

import pandas as pd
import numpy as np


# ─────────────────────────────────────────────────────────────
# ROLLING STATISTICAL FEATURES
# ─────────────────────────────────────────────────────────────
def add_rolling_features(df, window=30):
    """
    Computes rolling mean, std, and deviation over a sliding window.
    
    Args:
        df     : DataFrame with 'Aggregate' column
        window : Number of rows for rolling computation (default=30 → 30 mins at 1-min freq)
    
    New columns added:
        rolling_mean  - Baseline (expected) consumption
        rolling_std   - Variability in baseline
        deviation     - How far current reading is from baseline
        z_score       - Deviation normalized by std (units of std deviation)
    """
    df = df.copy()
    df['rolling_mean'] = df['Aggregate'].rolling(window=window, min_periods=5).mean()
    df['rolling_std']  = df['Aggregate'].rolling(window=window, min_periods=5).std()

    # Avoid division by zero with a small epsilon
    df['deviation'] = df['Aggregate'] - df['rolling_mean']
    df['z_score']   = df['deviation'] / (df['rolling_std'] + 1e-5)

    return df


# ─────────────────────────────────────────────────────────────
# RATE OF CHANGE FEATURES
# ─────────────────────────────────────────────────────────────
def add_rate_features(df):
    """
    Computes how fast the consumption is changing.
    A sudden large jump may indicate an appliance switching on unexpectedly.
    
    New columns added:
        diff_1    - Change from previous reading
        diff_5    - Change over last 5 readings (5 mins at 1-min freq)
        pct_change - Percentage change from previous reading
    """
    df = df.copy()
    df['diff_1']     = df['Aggregate'].diff(1)
    df['diff_5']     = df['Aggregate'].diff(5)
    df['pct_change'] = df['Aggregate'].pct_change(1) * 100   # in %

    # Cap extreme values caused by sensor restarts
    df['diff_1']     = df['diff_1'].clip(-5000, 5000)
    df['diff_5']     = df['diff_5'].clip(-5000, 5000)
    df['pct_change'] = df['pct_change'].clip(-500, 500)

    return df


# ─────────────────────────────────────────────────────────────
# APPLIANCE-LEVEL FEATURES
# ─────────────────────────────────────────────────────────────
def add_appliance_features(df):
    """
    Computes aggregate-level features from individual appliance columns.
    
    New columns added:
        appliance_sum     - Sum of all monitored appliance loads
        unaccounted_load  - Aggregate minus appliance sum (unmonitored devices)
        active_appliances - Count of appliances currently drawing > 10W
    """
    appliance_cols = [c for c in df.columns if 'Appliance' in c]

    if not appliance_cols:
        print("⚠️  No appliance columns found — skipping appliance features")
        return df

    df = df.copy()
    df['appliance_sum']    = df[appliance_cols].sum(axis=1)
    df['unaccounted_load'] = (df['Aggregate'] - df['appliance_sum']).clip(lower=0)
    df['active_appliances'] = (df[appliance_cols] > 10).sum(axis=1)

    return df


# ─────────────────────────────────────────────────────────────
# DAILY PATTERN FEATURES
# ─────────────────────────────────────────────────────────────
def add_daily_pattern_features(df):
    """
    Computes what the 'typical' consumption is for this time-of-day.
    Useful for detecting anomalies relative to historical patterns.
    
    New columns added:
        hourly_avg    - Average consumption for this hour across all days
        above_hourly  - Whether current reading is above the hourly average
    """
    df = df.copy()

    if 'hour' not in df.columns:
        df['hour'] = df['Time'].dt.hour

    hourly_avg_map = df.groupby('hour')['Aggregate'].mean()
    df['hourly_avg']   = df['hour'].map(hourly_avg_map)
    df['above_hourly'] = (df['Aggregate'] > df['hourly_avg']).astype(int)

    return df


# ─────────────────────────────────────────────────────────────
# COMBINED FEATURE PIPELINE
# ─────────────────────────────────────────────────────────────
def engineer_features(df, window=30):
    """
    Runs the complete feature engineering pipeline.
    Call this after preprocess.load_and_prepare().
    
    Args:
        df     : Cleaned DataFrame from preprocessing
        window : Rolling window size (default=30)
    
    Returns:
        DataFrame with all engineered features added.
    """
    print("🔧 Engineering features...")

    df = add_rolling_features(df, window=window)
    df = add_rate_features(df)
    df = add_appliance_features(df)
    df = add_daily_pattern_features(df)

    # Drop rows where rolling features are NaN (first `window` rows)
    before = len(df)
    df = df.dropna(subset=['rolling_mean', 'rolling_std']).reset_index(drop=True)
    print(f"   Dropped {before - len(df)} rows due to rolling window warmup")

    print(f"✅ Feature engineering complete. Shape: {df.shape}")
    print(f"   New features: rolling_mean, rolling_std, deviation, z_score,")
    print(f"                 diff_1, diff_5, pct_change,")
    print(f"                 appliance_sum, unaccounted_load, active_appliances,")
    print(f"                 hourly_avg, above_hourly")

    return df


# ─────────────────────────────────────────────────────────────
# Feature columns used by ML models
# ─────────────────────────────────────────────────────────────
ISOLATION_FOREST_FEATURES = [
    'Aggregate', 'rolling_mean', 'rolling_std',
    'deviation', 'z_score', 'hour', 'diff_1', 'pct_change'
]

KMEANS_FEATURES = [
    'Aggregate', 'hour', 'rolling_mean', 'is_weekend'
]


# ─────────────────────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, '/home/claude/energy_ai')
    from src.preprocess import load_and_prepare

    df, path = load_and_prepare("House_1.csv", resample=True)
    df = engineer_features(df)
    print("\n📋 Sample features:")
    print(df[ISOLATION_FOREST_FEATURES].head())