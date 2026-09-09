"""
ml/predict.py
--------------
Loads the trained Random Forest model (models/terrain_model.pkl) and
predicts terrain type from a single sensor reading or a batch of readings.

Usage
-----
    from ml.predict import TerrainPredictor

    predictor = TerrainPredictor()
    terrain = predictor.predict_one(sensor_row)          # -> "Rocky"
    terrain, proba = predictor.predict_one(sensor_row, return_proba=True)
"""

from __future__ import annotations
import os
import joblib
import pandas as pd

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "terrain_model.pkl")


class ModelNotTrainedError(RuntimeError):
    """Raised when terrain_model.pkl doesn't exist yet."""


class TerrainPredictor:
    def __init__(self, model_path: str = MODEL_PATH):
        self.model_path = model_path
        self._bundle = None
        self._load()

    def _load(self):
        if not os.path.exists(self.model_path):
            raise ModelNotTrainedError(
                f"No trained model found at '{self.model_path}'. "
                "Run `python ml/train_model.py` first."
            )
        self._bundle = joblib.load(self.model_path)

    @property
    def is_ready(self) -> bool:
        return self._bundle is not None

    def predict_one(self, row: dict, return_proba: bool = False):
        """Predict terrain for a single sensor reading (dict)."""
        feature_columns = self._bundle["feature_columns"]
        model = self._bundle["model"]

        x = pd.DataFrame([{col: row.get(col, 0) for col in feature_columns}])
        pred = model.predict(x)[0]

        if return_proba:
            proba = model.predict_proba(x)[0]
            proba_dict = dict(zip(model.classes_, proba))
            return pred, proba_dict
        return pred

    def predict_batch(self, df: pd.DataFrame) -> pd.Series:
        """Predict terrain for many sensor readings (DataFrame)."""
        feature_columns = self._bundle["feature_columns"]
        model = self._bundle["model"]
        x = df[feature_columns]
        return pd.Series(model.predict(x), index=df.index)


def try_load_predictor():
    """Return a TerrainPredictor, or None if the model hasn't been trained yet."""
    try:
        return TerrainPredictor()
    except ModelNotTrainedError:
        return None
