# ⚡ AI-Based Energy Management System

> A machine learning powered dashboard that monitors home electricity consumption, detects anomalies, and provides intelligent energy-saving recommendations — built on real UK smart meter data.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-red)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3+-orange)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📌 Overview

Traditional energy monitoring systems only show you how much power you're using. This system goes further — it **learns** your normal consumption patterns, **detects** when something unusual happens, **explains** why it flagged it, and **recommends** what to do about it.

Built using the **REFIT UK Smart Meter Dataset** (20 real households, 2 years of 8-second interval data), the system combines statistical modeling, unsupervised machine learning, and rule-based reasoning into a single interactive dashboard.

---

## 🎯 Key Features

- 🔍 **Anomaly Detection** — Isolation Forest flags unusual consumption without needing labeled data
- 🔵 **Pattern Recognition** — K-Means clusters usage into behavioral groups (night, daytime, peak, idle)
- 🚨 **Smart Alerts** — Rule engine explains *why* something is flagged in plain English
- 💡 **Recommendations** — Actionable suggestions generated per anomaly
- 📊 **Live Dashboard** — Interactive charts, heatmaps, and appliance breakdowns
- ▶️ **Real-Time Simulation** — Replays historical data to mimic a live smart meter
- 🏠 **Multi-House Support** — Switch between any of the 20 households

---

## 🧠 System Architecture

```
Raw CSV Data
     │
     ▼
┌─────────────────┐
│  Preprocessing  │  → Clean, resample 8s → 1min, parse timestamps
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    Features     │  → Rolling mean/std, z-score, rate-of-change, appliance stats
└────────┬────────┘
         │
         ├──────────────────────┐
         ▼                      ▼
┌─────────────────┐   ┌─────────────────┐
│ Isolation Forest│   │    K-Means      │
│ Anomaly Detection   │ Pattern Clusters│
└────────┬────────┘   └────────┬────────┘
         │                      │
         └──────────┬───────────┘
                    ▼
         ┌─────────────────┐
         │  Rule Engine    │  → Converts ML output to human-readable alerts
         └────────┬────────┘
                  │
                  ▼
         ┌─────────────────┐
         │ Streamlit UI    │  → Charts, alerts, heatmap, recommendations
         └─────────────────┘
```

---

## 📁 Project Structure

```
energy_ai/
│
├── app.py                  ← Streamlit dashboard (run this to launch UI)
├── train.py                ← Offline training script (run this first)
├── requirements.txt        ← Python dependencies
├── README.md
│
├── src/                    ← Core modules
│   ├── __pycache__
│   ├── preprocess.py       ← Data loading, cleaning, resampling
│   ├── features.py         ← Feature engineering
│   ├── anomaly.py          ← Isolation Forest model
│   ├── clustering.py       ← K-Means clustering
│   ├── rules.py            ← Alert & recommendation engine
│   └── simulator.py        ← Real-time data streaming
│
└── models/                 ← Auto-created after training
    ├── iso_forest.pkl
    ├── iso_scaler.pkl
    ├── kmeans.pkl
    └── km_scaler.pkl
```

> ⚠️ **Important:** `app.py` and `train.py` must be in the **root** `energy_ai/` folder, NOT inside `src/`.

---

## 🚀 Quick Start

### 1. Navigate to project folder

```bash
cd energy_ai
```

### 2. Create virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Train models

```bash
python train.py
```

This will:
- Download the REFIT dataset from Kaggle automatically (~886MB)
- Clean and resample the data (8-second → 1-minute intervals)
- Train Isolation Forest + K-Means models
- Save models to `models/`
- Print a full anomaly summary report

### 5. Launch dashboard

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`

---

## 📊 Dataset

**REFIT UK Electrical Load** — [kaggle.com/kyleahmurphy/uk-electrical-load](https://www.kaggle.com/datasets/kyleahmurphy/uk-electrical-load)

| Property | Value |
|---|---|
| Houses | 20 UK households |
| Duration | ~2 years (2013–2015) |
| Interval | 8 seconds |
| Channels | Aggregate + 9 appliances per house |
| Raw Rows | ~7 million per house |
| After Resampling | ~500K rows per house |
| Size | ~886 MB |

**Appliances monitored per house:**
Fridge, Freezer, Washing Machine, Dishwasher, Tumble Dryer, Computer, Television, Electric Shower, Microwave

---

## 🤖 ML Models

### Isolation Forest (Anomaly Detection)
- Unsupervised — no labeled data required
- Isolates anomalies by randomly partitioning data
- Points that are isolated faster = more anomalous
- Contamination rate tunable via dashboard slider (default: 5%)

### K-Means Clustering (Pattern Recognition)
- Groups consumption into K behavioral clusters
- Default clusters: Night Low, Daytime Moderate, Peak Usage, Standby/Idle
- Distance from cluster center used as an additional anomaly signal

---

## 🚨 Alert Types

| Alert | Trigger |
|---|---|
| 🔴 Critical | Z-score > 3.5 standard deviations from baseline |
| 🌙 Night Anomaly | Unusual consumption detected between 11PM–5AM |
| 🔺 Spike | Sudden jump more than 2× rolling standard deviation |
| ⚠️ AI Flagged | Isolation Forest labels as anomaly |
| 🔍 Unknown Pattern | Far from all known K-Means cluster centers |
| 💡 High Usage | Absolute consumption above 3000W threshold |

---

## ⚙️ Configuration

Tune thresholds in `src/rules.py`:

```python
THRESHOLDS = {
    'high_consumption_w'    : 3000,   # Watts — high usage alert
    'spike_multiplier'      : 2.0,    # x rolling_std = spike
    'z_score_critical'      : 3.5,    # critical alert threshold
    'cluster_distance_high' : 2.5,    # unknown pattern threshold
    'standby_waste_w'       : 100,    # overnight standby waste
}
```

Dashboard sliders (no code editing needed):
- **Anomaly Sensitivity** — controls Isolation Forest contamination (0.01–0.15)
- **K-Means Clusters** — number of usage pattern groups (2–8)
- **Window Size** — number of data points shown at once (50–500)
- **Playback Speed** — Slow / Normal / Fast / Turbo

---

## 📈 Dashboard Sections

| Section | Description |
|---|---|
| Metrics Row | Current power, avg, peak, anomaly count, estimated kWh |
| Consumption Chart | Live line chart with baseline band and anomaly markers |
| Cluster Chart | Color-coded usage pattern scatter plot |
| Appliance Breakdown | Horizontal bar chart of average appliance loads |
| Hourly Heatmap | Consumption patterns by hour and day of week |
| Alerts Panel | Recent anomalies with explanations and recommendations |
| Efficiency Tips | Static energy-saving best practices |

---

## 🛠️ Tech Stack

| Library | Purpose |
|---|---|
| `pandas` | Data loading, cleaning, resampling |
| `numpy` | Numerical computations |
| `scikit-learn` | Isolation Forest, K-Means, StandardScaler |
| `streamlit` | Interactive web dashboard |
| `plotly` | Charts, heatmaps, scatter plots |
| `joblib` | Model serialization |
| `kagglehub` | Dataset download from Kaggle |

---

## 🐛 Common Issues & Fixes

| Error | Fix |
|---|---|
| `infer_datetime_format` TypeError | Removed in pandas 2.0 — use `pd.to_datetime(df['Time'], format='ISO8601')` |
| `Only valid with DatetimeIndex` | Call `pd.to_datetime()` before `.set_index('Time')` |
| `NoneType has no len()` | `load_house()` missing `return df` at the end |
| `can't open file [train.py]` | Don't copy markdown links — type `python train.py` manually |
| `python -m src.train` not working | `train.py` must be in root folder, run as `python train.py` |

---

## 💡 Conclusion

This project demonstrates how machine learning and data analytics can be applied to smart energy systems to improve efficiency, detect anomalies early, and provide actionable insights — all without requiring any labeled training data.

---

## 📄 License

MIT License — free to use, modify, and distribute.
