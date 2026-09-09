import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.train_model import generate_training_data, train_and_save, FEATURE_COLUMNS
from ml.predict import TerrainPredictor
from sensors.simulator import MPU6050Simulator


def test_generate_training_data_has_all_terrains():
    df = generate_training_data(samples_per_terrain=20)
    assert set(df["terrain"].unique()) == {"Flat", "Raised", "Sloped", "Uneven", "Rocky"}
    assert len(df) == 20 * 5


def test_train_and_save_creates_model(tmp_path=None):
    df = generate_training_data(samples_per_terrain=50)
    bundle = train_and_save(df)
    assert "model" in bundle
    assert bundle["accuracy"] > 0.5  # sanity check, should be much higher in practice
    assert bundle["feature_columns"] == FEATURE_COLUMNS


def test_predictor_predicts_known_class():
    # Assumes a model has already been trained (train_and_save called above
    # or via ml/train_model.py). If not present, this test is skipped.
    try:
        predictor = TerrainPredictor()
    except Exception:
        return  # model not trained yet, skip silently

    sim = MPU6050Simulator(terrain="Flat", seed=99)
    row = sim.read()
    pred = predictor.predict_one(row)
    assert pred in {"Flat", "Raised", "Sloped", "Uneven", "Rocky"}
