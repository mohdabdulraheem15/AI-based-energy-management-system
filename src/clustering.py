"""
Block 6 — Pattern Recognition using K-Means Clustering
Groups consumption readings into behavioral patterns (e.g., night-low, daytime, peak).
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from src.features import KMEANS_FEATURES


# ─────────────────────────────────────────────────────────────
# MODEL PATHS
# ─────────────────────────────────────────────────────────────
MODEL_DIR        = os.path.join(os.path.dirname(__file__), '..', 'models')
KMEANS_PATH      = os.path.join(MODEL_DIR, 'kmeans.pkl')
KM_SCALER_PATH   = os.path.join(MODEL_DIR, 'km_scaler.pkl')

# Human-readable cluster labels (assigned after inspecting centroids)
# These are defaults — you can re-label after seeing your data's clusters
CLUSTER_LABELS = {
    0: "🌙 Night Low",
    1: "🏠 Daytime Moderate",
    2: "⚡ Peak Usage",
    3: "💤 Standby / Idle"
}


# ─────────────────────────────────────────────────────────────
# HELPER
# ─────────────────────────────────────────────────────────────
def get_available_features(df):
    return [col for col in KMEANS_FEATURES if col in df.columns]


# ─────────────────────────────────────────────────────────────
# FIND OPTIMAL K (Elbow Method)
# ─────────────────────────────────────────────────────────────
def find_optimal_k(df, k_range=range(2, 9)):
    """
    Computes inertia for different K values to help choose the best K.
    Plot the result to find the 'elbow' — where adding more clusters
    stops improving much.
    
    Returns dict of {k: inertia}
    """
    feature_cols = get_available_features(df)
    X = df[feature_cols].fillna(0).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    inertias = {}
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        inertias[k] = km.inertia_
        print(f"   K={k}: inertia={km.inertia_:.0f}")

    return inertias


# ─────────────────────────────────────────────────────────────
# TRAIN
# ─────────────────────────────────────────────────────────────
def train_kmeans(df, n_clusters=4, save=True):
    """
    Trains K-Means clustering to learn typical usage patterns.
    
    Args:
        df         : Feature-engineered DataFrame
        n_clusters : Number of clusters (4 works well for REFIT)
        save       : Save model to disk
    
    Returns:
        Trained KMeans model, fitted StandardScaler
    
    How K-Means works:
        - Groups data points into K clusters
        - Each cluster = a typical consumption pattern
        - Points far from their cluster center = unusual behavior
    """
    print(f"\n🔵 Training K-Means with K={n_clusters}...")

    feature_cols = get_available_features(df)
    print(f"   Features: {feature_cols}")

    X = df[feature_cols].fillna(0).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10,             # Run 10 times with different seeds, pick best
        max_iter=300
    )
    model.fit(X_scaled)

    if save:
        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump(model, KMEANS_PATH)
        joblib.dump(scaler, KM_SCALER_PATH)
        print(f"   💾 Model saved to: {KMEANS_PATH}")

    # Print cluster summary
    print(f"\n📊 Cluster centroids (in original scale):")
    centers_scaled = model.cluster_centers_
    centers_original = scaler.inverse_transform(centers_scaled)
    for i, center in enumerate(centers_original):
        label = CLUSTER_LABELS.get(i, f"Cluster {i}")
        vals  = dict(zip(feature_cols, center.round(1)))
        print(f"   {label}: {vals}")

    print(f"\n✅ K-Means trained on {len(df):,} samples")
    return model, scaler


# ─────────────────────────────────────────────────────────────
# PREDICT
# ─────────────────────────────────────────────────────────────
def assign_clusters(df, model=None, scaler=None):
    """
    Assigns each row to a cluster and computes distance from cluster center.
    
    New columns added:
        cluster          : Cluster ID (0 to n_clusters-1)
        cluster_label    : Human-readable cluster name
        cluster_distance : Distance from cluster center (higher = more unusual)
    """
    if model is None:
        if not os.path.exists(KMEANS_PATH):
            raise FileNotFoundError(
                "❌ No trained K-Means model found. Run train_kmeans() first."
            )
        model  = joblib.load(KMEANS_PATH)
        scaler = joblib.load(KM_SCALER_PATH)
        print("📂 Loaded K-Means model from disk")

    feature_cols = get_available_features(df)
    X = df[feature_cols].fillna(0).values
    X_scaled = scaler.transform(X)

    df = df.copy()
    df['cluster'] = model.predict(X_scaled)
    df['cluster_label'] = df['cluster'].map(CLUSTER_LABELS).fillna("Unknown")

    # Euclidean distance from assigned cluster center
    centers = model.cluster_centers_
    distances = []
    for i, row in enumerate(X_scaled):
        c = df['cluster'].iloc[i]
        dist = np.linalg.norm(row - centers[c])
        distances.append(dist)
    df['cluster_distance'] = distances

    print(f"\n📊 Cluster distribution:")
    dist = df.groupby(['cluster', 'cluster_label']).size().reset_index(name='count')
    dist['pct'] = (dist['count'] / len(df) * 100).round(1)
    print(dist.to_string(index=False))

    return df


# ─────────────────────────────────────────────────────────────
# AUTO-LABEL CLUSTERS by centroid characteristics
# ─────────────────────────────────────────────────────────────
def auto_label_clusters(model, scaler, feature_cols):
    """
    Automatically assigns labels to clusters based on centroid values.
    Sorts clusters by Aggregate consumption to assign meaningful names.
    
    Returns dict mapping cluster_id → label string
    """
    centers = scaler.inverse_transform(model.cluster_centers_)
    agg_idx = feature_cols.index('Aggregate') if 'Aggregate' in feature_cols else 0
    hour_idx = feature_cols.index('hour') if 'hour' in feature_cols else None

    # Sort by aggregate load
    sorted_by_load = sorted(range(len(centers)), key=lambda i: centers[i][agg_idx])

    auto_labels = {}
    for rank, cluster_id in enumerate(sorted_by_load):
        agg  = centers[cluster_id][agg_idx]
        hour = centers[cluster_id][hour_idx] if hour_idx else 12

        if rank == 0:
            label = "💤 Standby / Idle"
        elif rank == len(sorted_by_load) - 1:
            label = "⚡ Peak Usage"
        elif hour < 8 or hour >= 22:
            label = "🌙 Night Low"
        else:
            label = "🏠 Daytime Moderate"

        auto_labels[cluster_id] = label

    return auto_labels


# ─────────────────────────────────────────────────────────────
# LOAD OR TRAIN
# ─────────────────────────────────────────────────────────────
def load_or_train(df, n_clusters=4):
    """Loads existing model if available, otherwise trains a new one."""
    if os.path.exists(KMEANS_PATH) and os.path.exists(KM_SCALER_PATH):
        print("📂 Found existing K-Means model — loading from disk")
        model  = joblib.load(KMEANS_PATH)
        scaler = joblib.load(KM_SCALER_PATH)
    else:
        model, scaler = train_kmeans(df, n_clusters=n_clusters)
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

    model, scaler = train_kmeans(df)
    df = assign_clusters(df, model, scaler)

    print("\n📋 Cluster sample:")
    print(df[['Time', 'Aggregate', 'hour', 'cluster', 'cluster_label', 'cluster_distance']].head(10))
