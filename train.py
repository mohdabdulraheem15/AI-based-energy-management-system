"""
Block 10 — Offline Training Script
Run this ONCE before launching the Streamlit app to pre-train and save models.
This avoids training delay when the dashboard first loads.

Usage:
    python train.py
    python train.py --house House_2.csv --contamination 0.04
"""

import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from src.preprocess import load_and_prepare
from src.features   import engineer_features
from src.anomaly    import train_isolation_forest, predict_anomalies
from src.clustering import train_kmeans, assign_clusters
from src.rules      import get_summary, generate_all_alerts


def train_pipeline(house_file="House_1.csv", contamination=0.05, n_clusters=4):
    print("=" * 60)
    print("  AI Energy Management System — Training Pipeline")
    print("=" * 60)

    # ── Step 1: Load & preprocess ──────────────────────────────
    print("\n[1/5] Loading and preprocessing data...")
    df, dataset_path = load_and_prepare(house_file, resample=True)

    # ── Step 2: Feature engineering ───────────────────────────
    print("\n[2/5] Engineering features...")
    df = engineer_features(df)

    # ── Step 3: Train Isolation Forest ────────────────────────
    print(f"\n[3/5] Training Isolation Forest (contamination={contamination})...")
    iso_model, iso_scaler = train_isolation_forest(df, contamination=contamination, save=True)
    df = predict_anomalies(df, iso_model, iso_scaler)

    # ── Step 4: Train K-Means ─────────────────────────────────
    print(f"\n[4/5] Training K-Means (k={n_clusters})...")
    km_model, km_scaler = train_kmeans(df, n_clusters=n_clusters, save=True)
    df = assign_clusters(df, km_model, km_scaler)

    # ── Step 5: Generate alert summary ────────────────────────
    print("\n[5/5] Generating alert summary...")
    summary = get_summary(df)
    alerts  = generate_all_alerts(df)

    # ── Final report ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  ✅ TRAINING COMPLETE")
    print("=" * 60)
    print(f"\n📊 Dataset Summary:")
    print(f"   House          : {house_file}")
    print(f"   Total readings : {summary['total_readings']:,}")
    print(f"   Date range     : {df['Time'].min()} → {df['Time'].max()}")
    print(f"   Avg consumption: {summary['avg_consumption_w']} W")
    print(f"   Peak consumption: {summary['peak_consumption_w']} W")
    print(f"   Estimated usage: {summary['estimated_kwh']} kWh")
    print(f"\n🚨 Anomaly Detection:")
    print(f"   Anomalies found : {summary['anomaly_count']:,} ({summary['anomaly_pct']}%)")
    print(f"   Night anomalies : {summary['night_anomalies']}")
    print(f"   High consumption: {summary['high_consumption']}")
    print(f"\n🔔 Alerts Generated: {len(alerts)}")

    # Show top 5 critical alerts
    critical = [a for a in alerts if a.severity == 'critical']
    if critical:
        print(f"\n🔴 Top Critical Alerts:")
        for a in critical[:5]:
            print(f"   {a.timestamp} | {a.message}")

    print(f"\n💾 Models saved to: models/")
    print(f"   → iso_forest.pkl + iso_scaler.pkl")
    print(f"   → kmeans.pkl     + km_scaler.pkl")
    print(f"\n▶️  Now run: streamlit run app.py")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train AI Energy Management Models")
    parser.add_argument('--house',         default='House_1.csv', help='House CSV filename')
    parser.add_argument('--contamination', default=0.05, type=float, help='Isolation Forest contamination rate')
    parser.add_argument('--clusters',      default=4,    type=int,   help='Number of K-Means clusters')
    args = parser.parse_args()

    train_pipeline(
        house_file    = args.house,
        contamination = args.contamination,
        n_clusters    = args.clusters
    )