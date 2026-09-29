"""MODEL 2 - Bias correction: raw NWP rainfall -> corrected rainfall (mm)."""
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor

from .config import FEATURES, REGIME_PROB_COLS, MODEL_DIR, RANDOM_STATE

MODEL_PATH = MODEL_DIR / "bias_correction.joblib"
INPUT_COLS = FEATURES + REGIME_PROB_COLS   # NWP features + regime probabilities


class BiasCorrector:
    def __init__(self):
        self.model = self.build_estimator()

    @staticmethod
    def build_estimator():
        # log1p target: rainfall is heavily skewed, this stabilises training
        return TransformedTargetRegressor(
            regressor=HistGradientBoostingRegressor(
                max_iter=300, learning_rate=0.06, max_depth=6,
                random_state=RANDOM_STATE,
            ),
            func=np.log1p, inverse_func=np.expm1,
        )

    def fit(self, X: pd.DataFrame, y_actual):
        self.model.fit(X[INPUT_COLS], y_actual)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.clip(self.model.predict(X[INPUT_COLS]), 0, None)

    def save(self, path=MODEL_PATH):
        joblib.dump(self.model, path)

    @classmethod
    def load(cls, path=MODEL_PATH):
        obj = cls()
        obj.model = joblib.load(path)
        return obj
