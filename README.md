# 🦿 Single-Leg Terrain-Adaptive Landing Gear

A working software prototype (v1) for a single-leg terrain-adaptive landing
gear system. Since the physical hardware (ESP32 + MPU6050) isn't ready yet,
this version runs on **simulated sensor data** — but it's built so the
simulator can be swapped for real hardware later with almost no code changes.

## What it does

- Simulates an MPU6050 IMU (accelerometer + gyroscope) plus derived leg/body
  signals (pitch, roll, leg angle, foot contact, impact, vibration) for 5
  terrain types: **Flat, Raised, Sloped, Uneven, Rocky**.
- Computes **stability score** and **landing risk** using simple, tunable
  rule-based formulas (no ML needed for this part).
- Trains a **Random Forest** classifier that predicts terrain type purely
  from sensor readings, and uses it live in the dashboard.
- Visualizes everything in a **Streamlit** dashboard with live metrics,
  Plotly graphs, a recommended action, and CSV export.

## Project structure

```text
app/dashboard.py        Streamlit dashboard (the app you run)
sensors/simulator.py    Simulated MPU6050 sensor data generator
ml/train_model.py       Trains & saves the Random Forest terrain classifier
ml/predict.py           Loads the saved model and predicts terrain
utils/calculations.py   Rule-based stability score / risk / action logic
data/                   Generated training data (CSV) lands here
models/                 Trained model (terrain_model.pkl) lands here
tests/                  Pytest unit tests for all modules
requirements.txt        Python dependencies
```

## Setup

```bash
pip install -r requirements.txt
```

## 1. Train the terrain-classification model

This generates simulated training data and saves a trained model to
`models/terrain_model.pkl` (also saves the dataset to `data/training_data.csv`):

```bash
python ml/train_model.py
```

You should see test accuracy printed (typically ~95%+ since simulated
terrain profiles are cleanly separated). You can also train from inside the
dashboard sidebar if you skip this step — it'll offer a "Train model now" button.

## 2. Run the dashboard

```bash
streamlit run app/dashboard.py
```

Then in the browser:
1. Pick a **terrain type** in the sidebar.
2. Click **▶ Start** to begin streaming simulated sensor data.
3. Watch live sensor values, leg angle/contact, stability score, landing
   risk, the ML-predicted terrain, and Plotly graphs update.
4. Click **■ Stop** anytime, or **🗑 Clear history** to reset.
5. Use **⬇ Download session CSV** to export everything collected so far.

## 3. Run the tests

```bash
pytest tests/ -v
```

## Swapping in real hardware later

`sensors/simulator.py`'s `MPU6050Simulator.read()` returns a flat dict like:

```python
{
    "accel_x": ..., "accel_y": ..., "accel_z": ...,
    "gyro_x": ..., "gyro_y": ..., "gyro_z": ...,
    "pitch": ..., "roll": ...,
    "leg_angle": ..., "foot_contact": ...,
    "impact": ..., "vibration": ..., "terrain": ...,
}
```

When the ESP32 + MPU6050 hardware is ready, create a class with the same
`.read()` method (e.g. `ESP32SensorReader`, reading over serial/BLE/Wi-Fi)
that returns a dict with the same keys. Swap it in at the top of
`app/dashboard.py` — none of `utils/calculations.py`, `ml/predict.py`, or the
rest of the dashboard need to change, since they only depend on the dict
shape, not on where the data came from.

Note: real hardware won't give you ground-truth `terrain` labels for free —
you'll need to label a data-collection session per terrain type (e.g. walk
the leg over each terrain and tag the CSV rows) before retraining
`ml/train_model.py` on real data instead of simulated data.

## Notes on the rule-based calculations

`utils/calculations.py` intentionally keeps stability/risk as simple,
readable formulas (tilt + vibration + impact penalties) rather than ML,
so they're easy to hand-tune once you have real-world impact/vibration
ranges from actual hardware tests. The Random Forest model is used only
for **terrain classification**, not for stability or risk scoring.
