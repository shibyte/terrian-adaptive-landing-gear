"""
app/dashboard.py
------------------
Streamlit dashboard for the Single-Leg Terrain-Adaptive Landing Gear.

Run with:
    streamlit run app/dashboard.py

This app:
    1. Simulates MPU6050 sensor data (sensors/simulator.py) for a chosen
       terrain type (swap for real ESP32 data later -- see simulator.py).
    2. Computes stability score / landing risk / recommended action with
       simple rule-based logic (utils/calculations.py).
    3. Optionally runs a trained Random Forest model (ml/predict.py) to
       guess the terrain type purely from sensor readings.
    4. Plots live history with Plotly and lets you download the session's
       data as CSV.
"""

from __future__ import annotations
import math
import os
import sys
import time

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Make sibling packages (sensors, ml, utils) importable regardless of
# where streamlit is launched from.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sensors.simulator import MPU6050Simulator, TERRAIN_TYPES, LEG_L1, LEG_L2  # noqa: E402
from utils.calculations import compute_all, leg_geometry  # noqa: E402
from ml.predict import try_load_predictor  # noqa: E402
from ml.train_model import train_and_save  # noqa: E402


def render_leg_diagram(latest: pd.Series) -> go.Figure:
    """
    Build a simple 2D diagram of the single-leg mechanism: body/hip -> upper
    leg -> knee -> lower leg -> foot -> terrain, using the current reading's
    leg_angle / knee_angle / terrain_slope / foot_contact.
    """
    geometry = leg_geometry(latest["leg_angle"], latest["knee_angle"], l1=LEG_L1, l2=LEG_L2)
    hip, knee, foot = geometry["hip"], geometry["knee"], geometry["foot"]

    contact = bool(latest["foot_contact"])
    foot_color = "#22c55e" if contact else "#ef4444"  # green = contact, red = no contact

    slope_rad = math.radians(float(latest.get("terrain_slope", 0.0)))
    ground_half_len = 1.2
    gx0 = foot[0] - ground_half_len * math.cos(slope_rad)
    gy0 = foot[1] - ground_half_len * math.sin(slope_rad)
    gx1 = foot[0] + ground_half_len * math.cos(slope_rad)
    gy1 = foot[1] + ground_half_len * math.sin(slope_rad)

    fig = go.Figure()

    # Terrain line
    fig.add_trace(go.Scatter(
        x=[gx0, gx1], y=[gy0, gy1], mode="lines",
        line=dict(color="#92400e", width=6), hoverinfo="skip", showlegend=False,
    ))
    # Upper leg (hip -> knee)
    fig.add_trace(go.Scatter(
        x=[hip[0], knee[0]], y=[hip[1], knee[1]], mode="lines+markers",
        line=dict(color="#3b82f6", width=9), marker=dict(size=10, color="#1e3a8a"),
        hoverinfo="skip", showlegend=False,
    ))
    # Lower leg (knee -> foot)
    fig.add_trace(go.Scatter(
        x=[knee[0], foot[0]], y=[knee[1], foot[1]], mode="lines+markers",
        line=dict(color="#60a5fa", width=9), marker=dict(size=10, color="#1e40af"),
        hoverinfo="skip", showlegend=False,
    ))
    # Foot (colored by contact state)
    fig.add_trace(go.Scatter(
        x=[foot[0]], y=[foot[1]], mode="markers",
        marker=dict(size=20, color=foot_color, line=dict(width=2, color="black")),
        name="Foot", hoverinfo="skip", showlegend=False,
    ))
    # Body / hip
    fig.add_trace(go.Scatter(
        x=[hip[0]], y=[hip[1]], mode="markers",
        marker=dict(size=22, symbol="square", color="#111827"),
        name="Body", hoverinfo="skip", showlegend=False,
    ))

    fig.update_layout(
        height=380,
        title=f"Leg pose — {latest['terrain']} ({'Foot Contact' if contact else 'No Contact'})",
        xaxis=dict(range=[-2.2, 2.2], visible=False),
        yaxis=dict(range=[-2.3, 0.6], visible=False, scaleanchor="x", scaleratio=1),
        margin=dict(l=10, r=10, t=40, b=10),
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig

MAX_HISTORY_ROWS = 300  # cap how many rows we keep/plot, so the app stays fast

st.set_page_config(page_title="Terrain-Adaptive Landing Gear", layout="wide")


# ----------------------------------------------------------------------
# Session state setup
# ----------------------------------------------------------------------
def init_state():
    defaults = {
        "running": False,
        "history": pd.DataFrame(),
        "terrain": "Flat",
        "step": 0,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

    if "simulator" not in st.session_state:
        st.session_state.simulator = MPU6050Simulator(terrain=st.session_state.terrain)


init_state()

# ----------------------------------------------------------------------
# Sidebar controls
# ----------------------------------------------------------------------
st.sidebar.title("Controls")

selected_terrain = st.sidebar.selectbox(
    "Terrain type", TERRAIN_TYPES, index=TERRAIN_TYPES.index(st.session_state.terrain)
)
if selected_terrain != st.session_state.terrain:
    st.session_state.terrain = selected_terrain
    st.session_state.simulator.set_terrain(selected_terrain)

refresh_rate = st.sidebar.slider("Update interval (seconds)", 0.1, 2.0, 0.5, 0.1)

col_a, col_b = st.sidebar.columns(2)
if col_a.button("▶ Start", use_container_width=True):
    st.session_state.running = True
if col_b.button("■ Stop", use_container_width=True):
    st.session_state.running = False

if st.sidebar.button("🗑 Clear history", use_container_width=True):
    st.session_state.history = pd.DataFrame()
    st.session_state.step = 0

st.sidebar.markdown("---")
st.sidebar.subheader("Terrain ML Model")

predictor = try_load_predictor()
if predictor is None:
    st.sidebar.warning("No trained model found yet.")
    if st.sidebar.button("Train model now", use_container_width=True):
        with st.spinner("Training Random Forest on simulated data..."):
            train_and_save()
        st.rerun()
else:
    st.sidebar.success(f"Model loaded (test accuracy: {predictor._bundle['accuracy']:.1%})")

st.sidebar.markdown("---")
if not st.session_state.history.empty:
    csv_bytes = st.session_state.history.to_csv(index=False).encode("utf-8")
    st.sidebar.download_button(
        "⬇ Download session CSV",
        data=csv_bytes,
        file_name="landing_gear_session.csv",
        mime="text/csv",
        use_container_width=True,
    )

# ----------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------
st.title(" Single-Leg Terrain-Adaptive Landing Gear")
st.caption(
    "Live dashboard powered by simulated MPU6050 sensor data. "
    "Swap `sensors/simulator.py` for real ESP32 readings later without changing this file."
)

status_col1, status_col2, status_col3 = st.columns(3)
status_col1.metric("Simulation status", "🟢 Running" if st.session_state.running else "🔴 Stopped")
status_col2.metric("Current terrain (selected)", st.session_state.terrain)
status_col3.metric("Readings collected", len(st.session_state.history))

# ----------------------------------------------------------------------
# Generate a new reading (if running)
# ----------------------------------------------------------------------
if st.session_state.running:
    raw_row = st.session_state.simulator.read()
    row = compute_all(raw_row)
    row["step"] = st.session_state.step
    st.session_state.step += 1

    if predictor is not None:
        try:
            pred_terrain, proba = predictor.predict_one(raw_row, return_proba=True)
            row["predicted_terrain"] = pred_terrain
            row["predicted_confidence"] = round(float(max(proba.values())), 3)
        except Exception:
            row["predicted_terrain"] = None
            row["predicted_confidence"] = None
    else:
        row["predicted_terrain"] = None
        row["predicted_confidence"] = None

    new_row_df = pd.DataFrame([row])
    st.session_state.history = pd.concat([st.session_state.history, new_row_df], ignore_index=True)
    if len(st.session_state.history) > MAX_HISTORY_ROWS:
        st.session_state.history = st.session_state.history.iloc[-MAX_HISTORY_ROWS:].reset_index(drop=True)

history = st.session_state.history

# ----------------------------------------------------------------------
# Live values
# ----------------------------------------------------------------------
if not history.empty:
    latest = history.iloc[-1]

    st.subheader(" Live Sensor Values")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accel X (m/s²)", f"{latest['accel_x']:.2f}")
    c2.metric("Accel Y (m/s²)", f"{latest['accel_y']:.2f}")
    c3.metric("Accel Z (m/s²)", f"{latest['accel_z']:.2f}")
    c4.metric("Vibration", f"{latest['vibration']:.2f}")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Gyro X (°/s)", f"{latest['gyro_x']:.2f}")
    c6.metric("Gyro Y (°/s)", f"{latest['gyro_y']:.2f}")
    c7.metric("Gyro Z (°/s)", f"{latest['gyro_z']:.2f}")
    c8.metric("Impact", f"{latest['impact']:.2f}")

    st.subheader(" Leg & Body State")
    d1, d2, d3, d4, d5, d6 = st.columns(6)
    d1.metric("Body Pitch (°)", f"{latest['pitch']:.2f}")
    d2.metric("Body Roll (°)", f"{latest['roll']:.2f}")
    d3.metric("Leg Angle (°)", f"{latest['leg_angle']:.2f}")
    d4.metric("Knee Angle (°)", f"{latest['knee_angle']:.2f}")
    d5.metric("Foot Contact", "✅ Yes" if latest["foot_contact"] else "❌ No")
    d6.metric("Terrain Slope (°)", f"{latest['terrain_slope']:.2f}")

    st.subheader("Single-Leg Visualization")
    st.plotly_chart(render_leg_diagram(latest), use_container_width=True)

    st.subheader("Stability & Risk")
    e1, e2, e3 = st.columns(3)
    e1.metric("Stability Score", f"{latest['stability_score']:.1f} / 100")

    risk_emoji = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}
    e2.metric("Landing Risk", f"{risk_emoji.get(latest['landing_risk'], '')} {latest['landing_risk']}")

    if latest["predicted_terrain"] is not None:
        e3.metric(
            "ML Predicted Terrain",
            latest["predicted_terrain"],
            help=f"Confidence: {latest['predicted_confidence']:.1%}",
        )
    else:
        e3.metric("ML Predicted Terrain", "— (train model)")

    st.info(f"**Recommended action:** {latest['recommended_action']}")

    # ------------------------------------------------------------------
    # Plotly charts
    # ------------------------------------------------------------------
    st.subheader(" Live Graphs")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Acceleration", "Gyroscope", "Pitch / Roll / Leg Angle", "Stability & Risk"]
    )

    with tab1:
        fig = go.Figure()
        for col, name in [("accel_x", "Accel X"), ("accel_y", "Accel Y"), ("accel_z", "Accel Z")]:
            fig.add_trace(go.Scatter(x=history["step"], y=history[col], mode="lines", name=name))
        fig.update_layout(xaxis_title="Step", yaxis_title="m/s²", height=350)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        fig = go.Figure()
        for col, name in [("gyro_x", "Gyro X"), ("gyro_y", "Gyro Y"), ("gyro_z", "Gyro Z")]:
            fig.add_trace(go.Scatter(x=history["step"], y=history[col], mode="lines", name=name))
        fig.update_layout(xaxis_title="Step", yaxis_title="°/s", height=350)
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        fig = go.Figure()
        for col, name in [("pitch", "Pitch"), ("roll", "Roll"), ("leg_angle", "Leg Angle"), ("knee_angle", "Knee Angle")]:
            fig.add_trace(go.Scatter(x=history["step"], y=history[col], mode="lines", name=name))
        fig.update_layout(xaxis_title="Step", yaxis_title="degrees", height=350)
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=history["step"], y=history["stability_score"], mode="lines", name="Stability Score"))
        fig.add_trace(go.Scatter(x=history["step"], y=history["impact"], mode="lines", name="Impact"))
        fig.add_trace(go.Scatter(x=history["step"], y=history["vibration"], mode="lines", name="Vibration"))
        fig.update_layout(xaxis_title="Step", yaxis_title="value", height=350)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Recent Data")
    st.dataframe(history.tail(15), use_container_width=True)

else:
    st.warning("No data yet. Choose a terrain in the sidebar and click **▶ Start** to begin the simulation.")

# ----------------------------------------------------------------------
# Auto-refresh loop while running
# ----------------------------------------------------------------------
if st.session_state.running:
    time.sleep(refresh_rate)
    st.rerun()