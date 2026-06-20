"""
Block 8 — Real-Time Data Simulator
Streams the dataset row-by-row to simulate live smart meter readings.
Used by the Streamlit dashboard to create a real-time monitoring experience.
"""

import time
import pandas as pd
import numpy as np
from typing import Generator


# ─────────────────────────────────────────────────────────────
# CORE STREAM GENERATOR
# ─────────────────────────────────────────────────────────────
def simulate_stream(df: pd.DataFrame, delay: float = 0.1) -> Generator:
    """
    Yields one row at a time from the DataFrame with an optional delay.
    
    Args:
        df    : Fully processed DataFrame (with features + predictions)
        delay : Seconds to wait between rows (0 = as fast as possible)
    
    Yields:
        pd.Series — one row at a time
    
    Usage in Streamlit:
        for row in simulate_stream(df, delay=0.1):
            # update your chart with row
    """
    for _, row in df.iterrows():
        yield row
        if delay > 0:
            time.sleep(delay)


# ─────────────────────────────────────────────────────────────
# WINDOWED STREAM — returns growing window of data
# ─────────────────────────────────────────────────────────────
def simulate_window_stream(df: pd.DataFrame, window_size: int = 100, step: int = 1, delay: float = 0.2):
    """
    Yields a sliding window DataFrame that grows over time,
    then slides forward once it reaches window_size.
    
    Args:
        df          : Processed DataFrame
        window_size : Number of rows to show at a time
        step        : How many rows to advance per iteration
        delay       : Seconds between updates
    
    Yields:
        pd.DataFrame — subset of df representing current 'window'
    
    Usage in Streamlit:
        for window_df in simulate_window_stream(df, window_size=200):
            fig = plot_consumption(window_df)
            chart_placeholder.plotly_chart(fig)
    """
    total = len(df)
    idx   = 0

    while idx < total:
        end = min(idx + window_size, total)
        yield df.iloc[max(0, end - window_size):end]
        idx += step
        if delay > 0:
            time.sleep(delay)


# ─────────────────────────────────────────────────────────────
# BATCH STREAM — yields chunks for processing
# ─────────────────────────────────────────────────────────────
def simulate_batch_stream(df: pd.DataFrame, batch_size: int = 10, delay: float = 0.5):
    """
    Yields batches of rows for processing.
    Useful when you want to update every N readings.
    
    Yields:
        pd.DataFrame — batch of rows
    """
    for start in range(0, len(df), batch_size):
        yield df.iloc[start:start + batch_size]
        if delay > 0:
            time.sleep(delay)


# ─────────────────────────────────────────────────────────────
# STREAMLIT-SPECIFIC: Stateful iterator using session state
# ─────────────────────────────────────────────────────────────
class StreamlitSimulator:
    """
    A stateful simulator designed for use with Streamlit's session state.
    Stores the current position and advances on each call.
    
    Usage in app.py:
        if 'simulator' not in st.session_state:
            st.session_state.simulator = StreamlitSimulator(df, window=200)
        
        window_df = st.session_state.simulator.next()
        # render window_df
    """
    def __init__(self, df: pd.DataFrame, window: int = 200, step: int = 5):
        self.df     = df.reset_index(drop=True)
        self.window = window
        self.step   = step
        self.pos    = window   # Start after first full window

    def next(self) -> pd.DataFrame:
        """Returns the current window and advances the pointer."""
        start = max(0, self.pos - self.window)
        end   = self.pos
        window_df = self.df.iloc[start:end]

        # Advance (loop back to start when done)
        self.pos += self.step
        if self.pos > len(self.df):
            self.pos = self.window

        return window_df

    def get_latest_row(self) -> pd.Series:
        """Returns only the most recent single row."""
        idx = min(self.pos - 1, len(self.df) - 1)
        return self.df.iloc[idx]

    def progress(self) -> float:
        """Returns progress as 0.0–1.0."""
        return min(self.pos / len(self.df), 1.0)

    def reset(self):
        """Resets simulation to the beginning."""
        self.pos = self.window


# ─────────────────────────────────────────────────────────────
# PLAYBACK SPEEDS
# ─────────────────────────────────────────────────────────────
PLAYBACK_SPEEDS = {
    "🐢 Slow"   : 0.5,
    "▶️  Normal" : 0.15,
    "⚡ Fast"   : 0.05,
    "🚀 Turbo"  : 0.0
}


# ─────────────────────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, '/home/claude/energy_ai')
    from src.preprocess import load_and_prepare
    from src.features import engineer_features
    from src.anomaly import train_isolation_forest, predict_anomalies

    df, path = load_and_prepare("House_1.csv", resample=True)
    df = engineer_features(df)
    model, scaler = train_isolation_forest(df)
    df = predict_anomalies(df, model, scaler)

    print("▶️  Simulating first 5 rows...")
    for i, row in enumerate(simulate_stream(df, delay=0)):
        print(f"  {row['Time']} | {row['Aggregate']:.0f}W | anomaly={int(row['is_anomaly'])}")
        if i >= 4:
            break

    print("\n▶️  Testing window stream (3 windows of 50)...")
    for i, window in enumerate(simulate_window_stream(df, window_size=50, step=50, delay=0)):
        print(f"  Window {i+1}: rows {len(window)}, avg={window['Aggregate'].mean():.0f}W")
        if i >= 2:
            break

    print("\n▶️  Testing StreamlitSimulator...")
    sim = StreamlitSimulator(df, window=100, step=10)
    for i in range(3):
        w = sim.next()
        print(f"  Step {i+1}: {len(w)} rows, progress={sim.progress():.1%}")
