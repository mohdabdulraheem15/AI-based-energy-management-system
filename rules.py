"""
Block 7 — Rule-Based Reasoning Layer
Converts raw ML outputs into human-readable alerts and actionable recommendations.
This is the 'explainability' layer — it tells the user WHY something is flagged.
"""

import pandas as pd
from dataclasses import dataclass, field
from typing import List


# ─────────────────────────────────────────────────────────────
# ALERT DATA STRUCTURE
# ─────────────────────────────────────────────────────────────
@dataclass
class Alert:
    timestamp: str
    severity: str          # 'critical', 'warning', 'info'
    message: str
    recommendation: str
    value: float = 0.0
    icon: str = "⚠️"


# ─────────────────────────────────────────────────────────────
# THRESHOLDS — tune these per household
# ─────────────────────────────────────────────────────────────
THRESHOLDS = {
    'high_consumption_w'    : 3000,   # Watts — flag overall high usage
    'spike_multiplier'      : 2.0,    # x times rolling_std = spike
    'night_hours_start'     : 23,
    'night_hours_end'       : 5,
    'z_score_critical'      : 3.5,    # z-score threshold for critical alert
    'z_score_warning'       : 2.0,    # z-score threshold for warning
    'cluster_distance_high' : 2.5,    # Distance from cluster center threshold
    'standby_waste_w'       : 100,    # Standby power considered wasteful
}


# ─────────────────────────────────────────────────────────────
# CORE RULE ENGINE
# ─────────────────────────────────────────────────────────────
def generate_alert(row) -> tuple[List[str], List[str]]:
    """
    Takes a single DataFrame row and returns:
        alerts          : List of alert message strings
        recommendations : List of actionable suggestion strings
    
    Rules applied (in order of severity):
        R1 - Critical anomaly (very high z-score)
        R2 - Night-time anomaly
        R3 - Consumption spike above baseline
        R4 - High absolute consumption
        R5 - Unusual cluster behavior
        R6 - Standby power waste
    """
    alerts = []
    recommendations = []

    # Safely get values with defaults
    agg         = row.get('Aggregate', 0)
    z_score     = row.get('z_score', 0)
    deviation   = row.get('deviation', 0)
    rolling_std = row.get('rolling_std', 1)
    is_night    = row.get('is_night', 0)
    is_anomaly  = row.get('is_anomaly', 0)
    cluster_dist = row.get('cluster_distance', 0)
    hour        = row.get('hour', 12)

    # ── R1: Critical anomaly by z-score ──────────────────────
    if abs(z_score) > THRESHOLDS['z_score_critical']:
        alerts.append(f"🔴 CRITICAL: Consumption is {abs(z_score):.1f} standard deviations from baseline")
        recommendations.append("Immediate check recommended — inspect all high-power appliances.")

    # ── R2: Anomaly during night hours ────────────────────────
    elif is_anomaly and is_night:
        alerts.append(f"🌙 Unusual night-time consumption detected ({agg:.0f}W at {int(hour):02d}:00)")
        recommendations.append("Check if any appliance was accidentally left on overnight (oven, heating, etc.).")

    # ── R3: Spike above rolling baseline ─────────────────────
    elif is_anomaly and deviation > THRESHOLDS['spike_multiplier'] * rolling_std:
        alerts.append(f"🔺 Sudden spike: {agg:.0f}W vs baseline {(agg - deviation):.0f}W (+{deviation:.0f}W)")
        recommendations.append("A high-power appliance (kettle, washing machine, dryer) may have turned on.")

    # ── R4: General anomaly flagged by Isolation Forest ───────
    elif is_anomaly:
        alerts.append(f"⚠️ Unusual usage pattern detected by AI model ({agg:.0f}W)")
        recommendations.append("Review appliance schedule or check for unexpected activity.")

    # ── R5: High absolute consumption ────────────────────────
    if agg > THRESHOLDS['high_consumption_w']:
        alerts.append(f"💡 High overall usage: {agg:.0f}W (above {THRESHOLDS['high_consumption_w']}W threshold)")
        recommendations.append("Turn off standby devices and unnecessary appliances to reduce consumption.")

    # ── R6: Unusual cluster behavior (far from any pattern) ───
    if cluster_dist > THRESHOLDS['cluster_distance_high'] and is_anomaly:
        alerts.append("🔍 Behavior does not match any known usage pattern")
        recommendations.append("Possible unauthorized appliance or device malfunction. Verify manually.")

    # ── R7: Standby waste during night ────────────────────────
    if is_night and not is_anomaly and agg > THRESHOLDS['standby_waste_w']:
        recommendations.append(f"💤 Standby power ({agg:.0f}W overnight) — unplug TVs, chargers, and computers.")

    return alerts, recommendations


# ─────────────────────────────────────────────────────────────
# BATCH PROCESSING — run on entire DataFrame
# ─────────────────────────────────────────────────────────────
def generate_all_alerts(df) -> List[Alert]:
    """
    Processes the entire DataFrame and returns a list of Alert objects.
    Only returns rows with at least one alert.
    
    Returns:
        List of Alert objects, sorted by timestamp (most recent first)
    """
    all_alerts = []

    for _, row in df.iterrows():
        alerts, recs = generate_alert(row)
        if not alerts:
            continue

        ts = row['Time'].strftime('%Y-%m-%d %H:%M') if hasattr(row['Time'], 'strftime') else str(row['Time'])

        for alert_msg, rec in zip(alerts, recs):
            # Determine severity from alert message prefix
            if '🔴' in alert_msg:
                severity = 'critical'
            elif '🔺' in alert_msg or '🌙' in alert_msg:
                severity = 'warning'
            else:
                severity = 'info'

            all_alerts.append(Alert(
                timestamp      = ts,
                severity       = severity,
                message        = alert_msg,
                recommendation = rec,
                value          = row.get('Aggregate', 0),
                icon           = alert_msg.split()[0]
            ))

    all_alerts.sort(key=lambda a: a.timestamp, reverse=True)
    print(f"📋 Generated {len(all_alerts)} alerts from {df['is_anomaly'].sum()} anomalies")
    return all_alerts


# ─────────────────────────────────────────────────────────────
# SUMMARY STATISTICS FOR DASHBOARD
# ─────────────────────────────────────────────────────────────
def get_summary(df) -> dict:
    """
    Returns a summary dictionary for displaying in the dashboard.
    """
    anomalies = df[df['is_anomaly'] == 1]
    night_anomalies = anomalies[anomalies['is_night'] == 1]

    return {
        'total_readings'     : len(df),
        'anomaly_count'      : int(df['is_anomaly'].sum()),
        'anomaly_pct'        : round(df['is_anomaly'].mean() * 100, 2),
        'avg_consumption_w'  : round(df['Aggregate'].mean(), 1),
        'peak_consumption_w' : round(df['Aggregate'].max(), 1),
        'night_anomalies'    : len(night_anomalies),
        'high_consumption'   : int((df['Aggregate'] > THRESHOLDS['high_consumption_w']).sum()),
        'estimated_kwh'      : round(df['Aggregate'].sum() / 1000 / 60, 2),  # Watt-min → kWh
    }


# ─────────────────────────────────────────────────────────────
# ENERGY EFFICIENCY RECOMMENDATIONS (static, always shown)
# ─────────────────────────────────────────────────────────────
EFFICIENCY_TIPS = [
    "🌡️  Set your thermostat 1°C lower to save ~10% on heating bills.",
    "🔌  Unplug phone chargers, TVs and microwaves when not in use.",
    "🌙  Run washing machines and dishwashers at night (off-peak rates).",
    "💡  Switch remaining incandescent bulbs to LED — saves ~80% energy.",
    "🚿  Reduce hot water usage — water heating is ~15% of home energy.",
    "❄️   Defrost your freezer regularly — ice buildup increases energy use by 30%.",
    "🪟  Use curtains to retain heat in winter and block heat in summer.",
    "📊  Monitor your top 3 consuming appliances and set usage targets.",
]


# ─────────────────────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, '/home/claude/energy_ai')
    from src.preprocess import load_and_prepare
    from src.features import engineer_features
    from src.anomaly import train_isolation_forest, predict_anomalies
    from src.clustering import train_kmeans, assign_clusters

    df, path = load_and_prepare("House_1.csv", resample=True)
    df = engineer_features(df)

    model_iso, scaler_iso = train_isolation_forest(df)
    df = predict_anomalies(df, model_iso, scaler_iso)

    model_km, scaler_km = train_kmeans(df)
    df = assign_clusters(df, model_km, scaler_km)

    alerts = generate_all_alerts(df)
    print(f"\n🚨 Top 5 Alerts:")
    for a in alerts[:5]:
        print(f"  [{a.severity.upper()}] {a.timestamp} — {a.message}")
        print(f"    → {a.recommendation}")

    summary = get_summary(df)
    print("\n📊 Summary:", summary)