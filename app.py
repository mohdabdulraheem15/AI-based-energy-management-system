"""
Block 9 — Streamlit Dashboard
AI-Based Energy Management System — Interactive UI
Run with: streamlit run app.py
"""

import os
import sys
import time
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ── Path setup ────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from src.preprocess  import load_and_prepare
from src.features    import engineer_features
from src.anomaly     import load_or_train as iso_load_or_train, predict_anomalies
from src.clustering  import load_or_train as km_load_or_train, assign_clusters
from src.rules       import generate_alert, generate_all_alerts, get_summary, EFFICIENCY_TIPS
from src.simulator   import StreamlitSimulator, PLAYBACK_SPEEDS


# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title  = "⚡ Energy AI Dashboard",
    page_icon   = "⚡",
    layout      = "wide",
    initial_sidebar_state = "expanded"
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #0e1117; }

    /* Metric cards */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #1a1f2e, #252a3a);
        border: 1px solid #2d3548;
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 4px 16px rgba(0,0,0,0.3);
    }

    /* Alert boxes */
    .alert-critical {
        background: linear-gradient(90deg, #3d0000, #1a0000);
        border-left: 4px solid #ff4444;
        padding: 10px 16px;
        border-radius: 6px;
        margin: 6px 0;
        color: #ffcccc;
    }
    .alert-warning {
        background: linear-gradient(90deg, #3d2200, #1a1000);
        border-left: 4px solid #ff8800;
        padding: 10px 16px;
        border-radius: 6px;
        margin: 6px 0;
        color: #ffe0aa;
    }
    .alert-info {
        background: linear-gradient(90deg, #003d3d, #001a1a);
        border-left: 4px solid #00aaaa;
        padding: 10px 16px;
        border-radius: 6px;
        margin: 6px 0;
        color: #aaeeff;
    }
    .rec-box {
        background: #1a2535;
        border-left: 4px solid #4488ff;
        padding: 8px 14px;
        border-radius: 4px;
        margin: 4px 0 4px 20px;
        color: #99bbff;
        font-size: 0.9em;
    }
    .tip-box {
        background: #1a2530;
        border: 1px solid #2d4060;
        border-radius: 8px;
        padding: 10px 14px;
        margin: 4px 0;
        color: #aaccee;
        font-size: 0.9em;
    }
    .section-header {
        font-size: 1.1em;
        font-weight: 600;
        color: #88aaee;
        padding: 8px 0 4px 0;
        border-bottom: 1px solid #2d3548;
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# SIDEBAR — Controls
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚡ Energy AI")
    st.markdown("---")

    # House selector
    house_options = [f"House_{i}.csv" for i in range(1, 21)]
    selected_house = st.selectbox("🏠 Select House", house_options, index=0)

    st.markdown("---")

    # Simulation controls
    st.markdown("### 🎮 Simulation Controls")
    auto_play   = st.toggle("▶️  Auto-Play", value=False)
    speed_label = st.select_slider("Speed", options=list(PLAYBACK_SPEEDS.keys()), value="▶️  Normal")
    window_size = st.slider("📊 Window Size (rows)", min_value=50, max_value=500, value=200, step=50)

    st.markdown("---")

    # Model settings
    st.markdown("### 🤖 Model Settings")
    contamination = st.slider("Anomaly Sensitivity", min_value=0.01, max_value=0.15,
                               value=0.05, step=0.01,
                               help="Higher = more anomalies detected")
    n_clusters    = st.slider("K-Means Clusters", min_value=2, max_value=8, value=4)
    retrain       = st.button("🔄 Retrain Models")

    st.markdown("---")
    st.markdown("### 📂 Dataset")
    st.caption("REFIT UK Smart Meter Dataset\n20 houses · 8-second intervals · 2 years")


# ─────────────────────────────────────────────────────────────
# DATA LOADING & CACHING
# ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="📥 Loading and processing dataset...")
def load_data(house_file):
    """Cached data loader — only re-runs when house_file changes."""
    df, dataset_path = load_and_prepare(house_file, resample=True)
    df = engineer_features(df)
    return df, dataset_path


@st.cache_resource(show_spinner="🤖 Training ML models...")
def get_models(house_file, contamination, n_clusters, _df):
    """Cached model trainer — only re-runs when parameters change."""
    iso_model, iso_scaler = iso_load_or_train(_df, contamination=contamination)
    km_model,  km_scaler  = km_load_or_train(_df, n_clusters=n_clusters)
    return iso_model, iso_scaler, km_model, km_scaler


# ─────────────────────────────────────────────────────────────
# FORCE RETRAIN
# ─────────────────────────────────────────────────────────────
if retrain:
    # Delete saved models so load_or_train retrains
    for f in ['models/iso_forest.pkl', 'models/iso_scaler.pkl',
              'models/kmeans.pkl', 'models/km_scaler.pkl']:
        if os.path.exists(f):
            os.remove(f)
    st.cache_resource.clear()
    st.success("✅ Models cleared — will retrain on next load")
    st.rerun()


# ─────────────────────────────────────────────────────────────
# LOAD DATA AND MODELS
# ─────────────────────────────────────────────────────────────
with st.spinner("Loading data..."):
    df_full, dataset_path = load_data(selected_house)

with st.spinner("Loading/training models..."):
    iso_model, iso_scaler, km_model, km_scaler = get_models(
        selected_house, contamination, n_clusters, df_full
    )

# Apply predictions to full dataset
df_full = predict_anomalies(df_full, iso_model, iso_scaler)
df_full = assign_clusters(df_full, km_model, km_scaler)


# ─────────────────────────────────────────────────────────────
# SIMULATOR STATE
# ─────────────────────────────────────────────────────────────
sim_key = f"sim_{selected_house}"
if sim_key not in st.session_state or retrain:
    st.session_state[sim_key] = StreamlitSimulator(df_full, window=window_size, step=5)

simulator = st.session_state[sim_key]
simulator.window = window_size


# ─────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────
col_title, col_status = st.columns([3, 1])
with col_title:
    st.markdown(f"# ⚡ Energy Management Dashboard")
    st.caption(f"Monitoring: **{selected_house.replace('.csv', '')}** · REFIT UK Dataset")
with col_status:
    st.metric("Dataset Size", f"{len(df_full):,} rows")
    st.progress(simulator.progress(), text=f"Progress: {simulator.progress():.0%}")


# ─────────────────────────────────────────────────────────────
# ADVANCE SIMULATOR
# ─────────────────────────────────────────────────────────────
if auto_play:
    time.sleep(PLAYBACK_SPEEDS[speed_label])
    st.rerun()

col_prev, col_next, col_reset = st.columns([1, 1, 4])
with col_prev:
    if st.button("⏮ Reset"):
        simulator.reset()
        st.rerun()
with col_next:
    if st.button("⏭ Next Step") or auto_play:
        pass  # Simulator already advanced via auto_play loop above

# Get current window
df_view    = simulator.next()
latest_row = simulator.get_latest_row()


# ─────────────────────────────────────────────────────────────
# METRICS ROW
# ─────────────────────────────────────────────────────────────
st.markdown("---")
summary = get_summary(df_view)

m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("⚡ Current Power",  f"{latest_row['Aggregate']:.0f} W",
           delta=f"{latest_row.get('deviation', 0):+.0f}W vs baseline")
m2.metric("📊 Avg (Window)",   f"{summary['avg_consumption_w']} W")
m3.metric("📈 Peak (Window)",  f"{summary['peak_consumption_w']} W")
m4.metric("🚨 Anomalies",      f"{summary['anomaly_count']}",
           delta=f"{summary['anomaly_pct']}% of readings",
           delta_color="inverse")
m5.metric("🌙 Night Alerts",   f"{summary['night_anomalies']}")
m6.metric("💡 Est. Usage",     f"{summary['estimated_kwh']} kWh")


# ─────────────────────────────────────────────────────────────
# MAIN CHART — Consumption + Baseline + Anomalies
# ─────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="section-header">📈 Real-Time Energy Consumption</div>', unsafe_allow_html=True)

anomalies_view = df_view[df_view['is_anomaly'] == 1]

fig_main = go.Figure()

# Baseline band (rolling mean ± std)
if 'rolling_mean' in df_view.columns and 'rolling_std' in df_view.columns:
    fig_main.add_trace(go.Scatter(
        x    = pd.concat([df_view['Time'], df_view['Time'].iloc[::-1]]),
        y    = pd.concat([
            df_view['rolling_mean'] + df_view['rolling_std'],
            (df_view['rolling_mean'] - df_view['rolling_std']).iloc[::-1]
        ]),
        fill      = 'toself',
        fillcolor = 'rgba(68,136,255,0.1)',
        line      = dict(color='rgba(0,0,0,0)'),
        name      = 'Normal Range',
        hoverinfo = 'skip'
    ))

# Baseline (rolling mean)
fig_main.add_trace(go.Scatter(
    x     = df_view['Time'],
    y     = df_view['rolling_mean'],
    mode  = 'lines',
    name  = '📏 Baseline',
    line  = dict(color='#4488ff', width=1.5, dash='dash'),
    opacity = 0.7
))

# Actual consumption
fig_main.add_trace(go.Scatter(
    x    = df_view['Time'],
    y    = df_view['Aggregate'],
    mode = 'lines',
    name = '⚡ Consumption (W)',
    line = dict(color='#00ddaa', width=2),
    fill = 'tonexty' if False else None,
))

# Anomaly markers
if len(anomalies_view) > 0:
    fig_main.add_trace(go.Scatter(
        x      = anomalies_view['Time'],
        y      = anomalies_view['Aggregate'],
        mode   = 'markers',
        name   = '🚨 Anomaly',
        marker = dict(color='#ff4444', size=8, symbol='x',
                      line=dict(color='#ffffff', width=1))
    ))

fig_main.update_layout(
    height          = 380,
    paper_bgcolor   = 'rgba(0,0,0,0)',
    plot_bgcolor    = 'rgba(14,17,23,0.8)',
    font            = dict(color='#cccccc'),
    legend          = dict(orientation='h', yanchor='bottom', y=1.02),
    xaxis           = dict(gridcolor='#1a2030', title='Time'),
    yaxis           = dict(gridcolor='#1a2030', title='Power (Watts)'),
    margin          = dict(l=0, r=0, t=20, b=0),
    hovermode       = 'x unified'
)
st.plotly_chart(fig_main, use_container_width=True)


# ─────────────────────────────────────────────────────────────
# ROW 2: Cluster Chart + Appliance Breakdown
# ─────────────────────────────────────────────────────────────
col_left, col_right = st.columns([1, 1])

with col_left:
    st.markdown('<div class="section-header">🔵 Usage Pattern Clusters</div>', unsafe_allow_html=True)

    cluster_colors = {0: '#4488ff', 1: '#44cc88', 2: '#ff8844', 3: '#aa44ff'}
    fig_cluster = go.Figure()

    for cluster_id in df_view['cluster'].unique():
        subset = df_view[df_view['cluster'] == cluster_id]
        label  = subset['cluster_label'].iloc[0] if len(subset) > 0 else f"Cluster {cluster_id}"
        fig_cluster.add_trace(go.Scatter(
            x      = subset['Time'],
            y      = subset['Aggregate'],
            mode   = 'markers',
            name   = label,
            marker = dict(
                color   = cluster_colors.get(int(cluster_id), '#888888'),
                size    = 4,
                opacity = 0.7
            )
        ))

    fig_cluster.update_layout(
        height        = 280,
        paper_bgcolor = 'rgba(0,0,0,0)',
        plot_bgcolor  = 'rgba(14,17,23,0.8)',
        font          = dict(color='#cccccc'),
        xaxis         = dict(gridcolor='#1a2030'),
        yaxis         = dict(gridcolor='#1a2030', title='Watts'),
        margin        = dict(l=0, r=0, t=10, b=0),
        legend        = dict(font=dict(size=10))
    )
    st.plotly_chart(fig_cluster, use_container_width=True)

with col_right:
    st.markdown('<div class="section-header">🏠 Appliance Load Breakdown</div>', unsafe_allow_html=True)

    appliance_cols = [c for c in df_view.columns if 'Appliance' in c]
    if appliance_cols:
        appliance_avgs = df_view[appliance_cols].mean().sort_values(ascending=False)
        appliance_avgs = appliance_avgs[appliance_avgs > 1]   # Filter near-zero

        fig_app = go.Figure(go.Bar(
            x         = appliance_avgs.values,
            y         = appliance_avgs.index,
            orientation = 'h',
            marker_color = px.colors.sequential.Teal,
            text      = [f"{v:.0f}W" for v in appliance_avgs.values],
            textposition = 'outside'
        ))
        fig_app.update_layout(
            height        = 280,
            paper_bgcolor = 'rgba(0,0,0,0)',
            plot_bgcolor  = 'rgba(14,17,23,0.8)',
            font          = dict(color='#cccccc'),
            xaxis         = dict(gridcolor='#1a2030', title='Avg Watts'),
            yaxis         = dict(gridcolor='#1a2030'),
            margin        = dict(l=0, r=40, t=10, b=0)
        )
        st.plotly_chart(fig_app, use_container_width=True)
    else:
        st.info("No appliance-level data available for this house.")


# ─────────────────────────────────────────────────────────────
# ROW 3: Hourly Pattern Heatmap
# ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">🕐 Hourly Consumption Heatmap</div>', unsafe_allow_html=True)

if 'hour' in df_view.columns and 'day_of_week' in df_view.columns:
    pivot = df_view.pivot_table(
        values='Aggregate', index='day_of_week', columns='hour', aggfunc='mean'
    )
    day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    pivot.index = [day_names[i] for i in pivot.index if i < len(day_names)]

    fig_heat = go.Figure(go.Heatmap(
        z          = pivot.values,
        x          = [f"{h:02d}:00" for h in pivot.columns],
        y          = pivot.index,
        colorscale = 'Teal',
        colorbar   = dict(title='Watts')
    ))
    fig_heat.update_layout(
        height        = 220,
        paper_bgcolor = 'rgba(0,0,0,0)',
        plot_bgcolor  = 'rgba(14,17,23,0.8)',
        font          = dict(color='#cccccc'),
        margin        = dict(l=0, r=0, t=10, b=0),
        xaxis         = dict(title='Hour of Day')
    )
    st.plotly_chart(fig_heat, use_container_width=True)


# ─────────────────────────────────────────────────────────────
# ROW 4: Alerts Panel + Efficiency Tips
# ─────────────────────────────────────────────────────────────
col_alerts, col_tips = st.columns([3, 2])

with col_alerts:
    st.markdown('<div class="section-header">🚨 Active Alerts & Recommendations</div>', unsafe_allow_html=True)

    recent_anomalies = df_view[df_view['is_anomaly'] == 1].tail(10)

    if len(recent_anomalies) == 0:
        st.success("✅ No anomalies detected in current window")
    else:
        for _, row in recent_anomalies.tail(6).iterrows():
            alerts_list, recs_list = generate_alert(row)
            ts = row['Time'].strftime('%H:%M %d %b') if hasattr(row['Time'], 'strftime') else str(row['Time'])

            for alert_msg in alerts_list:
                severity = 'critical' if '🔴' in alert_msg else ('warning' if ('🔺' in alert_msg or '🌙' in alert_msg) else 'info')
                st.markdown(
                    f'<div class="alert-{severity}"><strong>{ts}</strong> — {alert_msg}</div>',
                    unsafe_allow_html=True
                )

            for rec in recs_list:
                st.markdown(f'<div class="rec-box">↳ {rec}</div>', unsafe_allow_html=True)

with col_tips:
    st.markdown('<div class="section-header">💡 Energy Efficiency Tips</div>', unsafe_allow_html=True)
    for tip in EFFICIENCY_TIPS[:6]:
        st.markdown(f'<div class="tip-box">{tip}</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "⚡ AI Energy Management System · REFIT UK Dataset · "
    "Isolation Forest + K-Means + Rule-Based Reasoning"
)

# ── Auto-refresh trigger ──────────────────────────────────────
if auto_play:
    delay = PLAYBACK_SPEEDS[speed_label]
    time.sleep(max(delay, 0.05))
    st.rerun()