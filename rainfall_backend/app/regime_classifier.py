"""MODEL 1 - Weather regime classifier (active monsoon, break, depression...)."""
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from .config import FEATURES, REGIMES, REGIME_LABELS, MODEL_DIR, RANDOM_STATE

MODEL_PATH = MODEL_DIR / "regime_classifier.joblib"


class RegimeClassifier:
    def __init__(self):
        self.model = self.build_estimator()

    @staticmethod
    def build_estimator():
        return RandomForestClassifier(
            n_estimators=200, min_samples_leaf=2,
            random_state=RANDOM_STATE, n_jobs=-1,
        )

    def fit(self, X: pd.DataFrame, y):
        self.model.fit(X[FEATURES], y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        """Probability per regime, columns ordered as config.REGIMES."""
        proba = pd.DataFrame(
            self.model.predict_proba(X[FEATURES]),
            columns=self.model.classes_, index=X.index,
        )
        return proba.reindex(columns=REGIMES, fill_value=0.0)

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """Returns regime key, display label, confidence (0-100)."""
        proba = self.predict_proba(X)
        best = proba.idxmax(axis=1)
        return pd.DataFrame({
            "regime": best,
            "regime_label": best.map(REGIME_LABELS),
            "confidence": (proba.max(axis=1) * 100).round(1),
        }, index=X.index)

    def feature_importance(self) -> dict:
        imp = self.model.feature_importances_
        return dict(sorted(zip(FEATURES, imp.round(4).tolist()), key=lambda kv: -kv[1]))

    def save(self, path=MODEL_PATH):
        joblib.dump(self.model, path)

    @classmethod
    def load(cls, path=MODEL_PATH):
        obj = cls()
        obj.model = joblib.load(path)
        return obj
