"""MODEL 3 - Threshold rainfall prediction: P(rainfall >= T) for several T."""
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from .config import (FEATURES, REGIME_PROB_COLS, THRESHOLDS, DECISION_CUTOFF,
                     MODEL_DIR, RANDOM_STATE)

MODEL_PATH = MODEL_DIR / "threshold_model.joblib"
INPUT_COLS = FEATURES + REGIME_PROB_COLS + ["corrected_rain"]


class ThresholdPredictor:
    """One binary classifier per threshold (10, 30, 50, 100 mm)."""

    def __init__(self):
        self.models = {}       # threshold -> fitted classifier or constant float

    @staticmethod
    def build_estimator():
        return HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.06, max_depth=5,
            random_state=RANDOM_STATE,
        )

    def fit(self, X: pd.DataFrame, y_actual):
        y_actual = np.asarray(y_actual)
        for t in THRESHOLDS:
            y = (y_actual >= t).astype(int)
            if y.min() == y.max():            # only one class in data
                self.models[t] = float(y[0])
            else:
                self.models[t] = self.build_estimator().fit(X[INPUT_COLS], y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        """DataFrame with a column per threshold: probability in [0, 1]."""
        out = {}
        for t, m in self.models.items():
            if isinstance(m, float):
                out[t] = np.full(len(X), m)
            else:
                out[t] = m.predict_proba(X[INPUT_COLS])[:, 1]
        return pd.DataFrame(out, index=X.index)

    def predict_binary(self, X: pd.DataFrame, cutoff=DECISION_CUTOFF) -> pd.DataFrame:
        return (self.predict_proba(X) >= cutoff).astype(int)

    def save(self, path=MODEL_PATH):
        joblib.dump(self.models, path)

    @classmethod
    def load(cls, path=MODEL_PATH):
        obj = cls()
        obj.models = joblib.load(path)
        return obj
