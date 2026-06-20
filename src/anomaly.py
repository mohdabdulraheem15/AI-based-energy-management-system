"""
Block 5 — Anomaly Detection using Isolation Forest
Trains an unsupervised ML model to detect unusual energy consumption patterns.
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from src.features import ISOLATION_FOREST_FEATURES


# ─────────────────────────────────────────────────────────────
# MODEL PATHS
# ─────────────────────────────────────────────────────────────
MODEL_DIR       = os.path.join(os.path.dirname(__file__), '..', 'models')
ISO_MODEL_PATH  = os.path.join(MODEL_DIR, 'iso_forest.pkl')
SCALER_PATH     = os.path.join(MODEL_DIR, 'iso_scaler.pkl')


# ─────────────────────────────────────────────────────────────
# HELPER: Get available feature columns
# ─────────────────────────────────────────────────────────────
def get_available_features(df):
    """Returns only the feature columns that exist in the DataFrame."""
    return [col for col in ISOLATION_FOREST_FEATURES if col in df.columns]


# ─────────────────────────────────────────────────────────────
# TRAIN
# ─────────────────────────────────────────────────────────────
def train_isolation_forest(df, contamination=0.05, save=True):
    """
    Trains an Isolation Forest on the feature-engineered DataFrame.
    
    Args:
        df            : DataFrame with engineered features
        contamination : Expected proportion of anomalies (0.01–0.15)
                        0.05 = ~5% of readings are expected anomalies
        save          : Whether to save the model to disk
    
    Returns:
        Trained IsolationForest model, fitted StandardScaler
    
    How Isolation Forest works:
        - Randomly partitions data using decision trees
        - Anomalous points are isolated faster (shorter paths)
        - No labels needed — fully unsupervised
    """
    print("\n🌲 Training Isolation Forest...")

    feature_cols = get_available_features(df)
    print(f"   Features used: {feature_cols}")

    X = df[feature_cols].values

    # Scale features so no single feature dominates
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(
        n_estimators=200,        # Number of trees (more = better, slower)
        contamination=contamination,
        max_samples='auto',      # Auto-selects sample size per tree
        random_state=42,
        n_jobs=-1                # Use all CPU cores
    )
    model.fit(X_scaled)

    if save:
        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump(model, ISO_MODEL_PATH)
        joblib.dump(scaler, SCALER_PATH)
        print(f"   💾 Model saved to: {ISO_MODEL_PATH}")

    print(f"✅ Isolation Forest trained on {len(df):,} samples")
    return model, scaler


# ─────────────────────────────────────────────────────────────
# PREDICT
# ─────────────────────────────────────────────────────────────
def predict_anomalies(df, model=None, scaler=None):
    """
    Applies the trained Isolation Forest to label each row.
    
    New columns added:
        anomaly_score : Raw decision score (more negative = more anomalous)
        is_anomaly    : 1 if anomaly detected, 0 if normal
    
    Loads model from disk if not provided.
    """
    if model is None:
        if not os.path.exists(ISO_MODEL_PATH):
            raise FileNotFoundError(
                "❌ No trained model found. Run train_isolation_forest() first."
            )
        model  = joblib.load(ISO_MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        print("📂 Loaded Isolation Forest model from disk")

    feature_cols = get_available_features(df)
    X = df[feature_cols].fillna(0).values
    X_scaled = scaler.transform(X)

    # predict() returns: -1 = anomaly, 1 = normal
    labels = model.predict(X_scaled)
    scores = model.decision_function(X_scaled)  # Lower = more anomalous

    df = df.copy()
    df['anomaly_score'] = scores
    df['is_anomaly']    = (labels == -1).astype(int)

    n_anomalies = df['is_anomaly'].sum()
    pct = n_anomalies / len(df) * 100
    print(f"🔍 Anomalies detected: {n_anomalies:,} / {len(df):,} ({pct:.1f}%)")

    return df


# ─────────────────────────────────────────────────────────────
# LOAD OR TRAIN
# ─────────────────────────────────────────────────────────────
def load_or_train(df, contamination=0.05):
    """
    Loads existing model if available, otherwise trains a new one.
    Use this in the Streamlit app to avoid retraining on every refresh.
    """
    if os.path.exists(ISO_MODEL_PATH) and os.path.exists(SCALER_PATH):
        print("📂 Found existing Isolation Forest model — loading from disk")
        model  = joblib.load(ISO_MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
    else:
        model, scaler = train_isolation_forest(df, contamination=contamination)
    return model, scaler


# ─────────────────────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, '/home/claude/energy_ai')
    from src.preprocess import load_and_prepare
    from src.features import engineer_features

    df, path = load_and_prepare("House_1.csv", resample=True)
    df = engineer_features(df)

    model, scaler = train_isolation_forest(df)
    df = predict_anomalies(df, model, scaler)

    print("\n📋 Sample anomaly predictions:")
    print(df[['Time', 'Aggregate', 'is_anomaly', 'anomaly_score']].head(10))

    anomalies = df[df['is_anomaly'] == 1]
    print(f"\n⚠️  Sample anomalies:")
    print(anomalies[['Time', 'Aggregate', 'hour', 'deviation']].head(5))
