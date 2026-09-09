"""
ml/train_model.py
-------------------
Trains a Random Forest classifier that predicts TERRAIN TYPE from a
sensor reading (accel/gyro/pitch/roll/leg_angle/vibration/impact/contact).

Since we don't have real hardware yet, training data is generated using
the same simulator used by the live dashboard (sensors/simulator.py).
When real ESP32 + MPU6050 data becomes available, simply replace the
`generate_training_data()` function with one that loads a CSV of labeled
real sensor logs -- the rest of this script (feature prep, training,
saving) stays the same.

Run directly:
    python ml/train_model.py

Produces:
    models/terrain_model.pkl   (trained RandomForestClassifier + metadata)
    data/training_data.csv     (the generated dataset, for inspection)
"""

from __future__ import annotations
import os
import sys
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Allow running this file directly (python ml/train_model.py) as well as
# as part of the package (python -m ml.train_model).
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sensors.simulator import MPU6050Simulator, TERRAIN_TYPES  # noqa: E402

FEATURE_COLUMNS = [
    "accel_x", "accel_y", "accel_z",
    "gyro_x", "gyro_y", "gyro_z",
    "pitch", "roll",
    "leg_angle", "knee_angle", "vibration", "impact",
]
# NOTE: 'terrain_height' and 'terrain_slope' are deliberately NOT used as
# features. They're ground-truth terrain properties that a real ESP32 +
# MPU6050 leg can't measure directly -- only their effect on leg_angle,
# knee_angle, impact, and vibration is observable. Keeping them out of
# FEATURE_COLUMNS means this model will still make sense once trained on
# real hardware data later. They're still saved in the CSV for inspection
# and used by the dashboard's leg visualization.

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "training_data.csv")
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "terrain_model.pkl")


def generate_training_data(samples_per_terrain: int = 400, seed: int = 42) -> pd.DataFrame:
    """Generate a labeled dataset by sampling the simulator for every terrain type."""
    rows = []
    for i, terrain in enumerate(TERRAIN_TYPES):
        sim = MPU6050Simulator(terrain=terrain, seed=seed + i)
        rows.extend(sim.read_batch(samples_per_terrain))
    df = pd.DataFrame(rows)
    # foot_contact is boolean -> convert to int so it can be used as a feature later if desired
    df["foot_contact"] = df["foot_contact"].astype(int)
    return df


def train_and_save(df: pd.DataFrame | None = None) -> dict:
    """Train the Random Forest terrain classifier and save it to models/terrain_model.pkl."""
    if df is None:
        df = generate_training_data()

    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    df.to_csv(DATA_PATH, index=False)

    X = df[FEATURE_COLUMNS]
    y = df["terrain"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    bundle = {
        "model": model,
        "feature_columns": FEATURE_COLUMNS,
        "classes": list(model.classes_),
        "accuracy": acc,
    }
    joblib.dump(bundle, MODEL_PATH)

    print(f"Trained RandomForestClassifier on {len(df)} samples")
    print(f"Test accuracy: {acc:.4f}")
    print(report)
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Training data saved to: {DATA_PATH}")

    return bundle


if __name__ == "__main__":
    train_and_save()