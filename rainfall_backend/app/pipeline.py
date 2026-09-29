"""
Pipeline: trains and chains the 3 models.

  features -> Model 1 (regime) -> Model 2 (bias correction) -> Model 3 (threshold)
"""
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_predict, StratifiedKFold, KFold

from .config import (FEATURES, REGIMES, REGIME_LABELS, REGIME_PROB_COLS, THRESHOLDS,
                     ALERT_THRESHOLD, DECISION_CUTOFF, FEATURE_DEFAULTS,
                     MODEL_DIR, OUTPUT_DIR, RANDOM_STATE)
from .data_generator import load_training_data, normalize_columns
from .regime_classifier import RegimeClassifier, MODEL_PATH as REGIME_PATH
from .bias_correction import BiasCorrector, MODEL_PATH as BIAS_PATH
from .threshold_model import ThresholdPredictor, MODEL_PATH as THR_PATH
from . import metrics as M

METRICS_PATH = MODEL_DIR / "metrics.json"
TEST_PRED_PATH = OUTPUT_DIR / "test_predictions.csv"


class RainfallPipeline:
    def __init__(self):
        self.regime = None
        self.bias = None
        self.threshold = None

    # ------------------------------------------------------------------ train
    def train(self, df: pd.DataFrame | None = None) -> dict:
        df = load_training_data() if df is None else df
        train, test = train_test_split(df, test_size=0.2, random_state=RANDOM_STATE,
                                       stratify=df["regime"])
        train, test = train.reset_index(drop=True), test.reset_index(drop=True)

        # Model 1: regime classifier. Out-of-fold probabilities for the training
        # set so downstream models see realistic (not overfit) regime inputs.
        self.regime = RegimeClassifier().fit(train, train["regime"])
        oof = cross_val_predict(
            RegimeClassifier.build_estimator(), train[FEATURES], train["regime"],
            cv=StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE),
            method="predict_proba")
        train_p = pd.DataFrame(oof, columns=sorted(train["regime"].unique()))
        train_p = train_p.reindex(columns=REGIMES, fill_value=0.0)
        train_p.columns = REGIME_PROB_COLS
        Xtr = pd.concat([train[FEATURES], train_p], axis=1)

        # Model 2: bias correction
        self.bias = BiasCorrector().fit(Xtr, train["actual_rain"])
        oof_corr = cross_val_predict(
            BiasCorrector.build_estimator(), Xtr[FEATURES + REGIME_PROB_COLS],
            train["actual_rain"], cv=KFold(5, shuffle=True, random_state=RANDOM_STATE))
        Xtr["corrected_rain"] = np.clip(oof_corr, 0, None)

        # Model 3: threshold classifiers
        self.threshold = ThresholdPredictor().fit(Xtr, train["actual_rain"])

        # Evaluate on held-out test set
        result = self.predict(test)
        result["actual_rain"] = test["actual_rain"].values
        result["true_regime"] = test["regime"].values
        result["nwp_rain"] = test["nwp_rain"].values
        report = self._evaluate(result)
        report["regime_feature_importance"] = self.regime.feature_importance()
        report["n_train"], report["n_test"] = len(train), len(test)

        self.save()
        result.to_csv(TEST_PRED_PATH, index=False)
        METRICS_PATH.write_text(json.dumps(report, indent=2))
        return report

    def _evaluate(self, r: pd.DataFrame) -> dict:
        obs = r["actual_rain"]
        t = ALERT_THRESHOLD
        nwp_rmse, ai_rmse = M.rmse(obs, r["nwp_rain"]), M.rmse(obs, r["corrected_rain"])
        nwp_mae, ai_mae = M.mae(obs, r["nwp_rain"]), M.mae(obs, r["corrected_rain"])
        nwp_cat = M.categorical_scores(obs >= t, r["nwp_rain"] >= t)
        ai_cat = M.categorical_scores(obs >= t, r[f"prob_{t}"] >= DECISION_CUTOFF)
        return {
            "alert_threshold_mm": t,
            "regime_accuracy": round(float((r["regime"] == r["true_regime"]).mean()), 3),
            "RMSE": {"nwp": round(nwp_rmse, 2), "ai": round(ai_rmse, 2),
                     "change_pct": M.pct_change(nwp_rmse, ai_rmse)},
            "MAE": {"nwp": round(nwp_mae, 2), "ai": round(ai_mae, 2),
                    "change_pct": M.pct_change(nwp_mae, ai_mae)},
            **{k: {"nwp": nwp_cat[k], "ai": ai_cat[k],
                   "change_pct": M.pct_change(nwp_cat[k], ai_cat[k])}
               for k in ("POD", "CSI", "FAR")},
        }

    # --------------------------------------------------------------- inference
    @staticmethod
    def prepare(df: pd.DataFrame) -> pd.DataFrame:
        """Normalise column names and fill missing optional features."""
        df = normalize_columns(df.copy())
        if "nwp_rain" not in df:
            raise ValueError("Input needs an NWP rainfall column (nwp_rain / NWP / Forecast).")
        for col, default in FEATURE_DEFAULTS.items():
            if col not in df:
                df[col] = default
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(default)
        df["nwp_rain"] = pd.to_numeric(df["nwp_rain"], errors="coerce").fillna(0.0)
        return df.reset_index(drop=True)

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.prepare(df)
        reg = self.regime.predict(df)
        proba = self.regime.predict_proba(df)
        proba.columns = REGIME_PROB_COLS

        X = pd.concat([df[FEATURES], proba], axis=1)
        corrected = self.bias.predict(X)
        X["corrected_rain"] = corrected
        thr = self.threshold.predict_proba(X)

        out = pd.concat([df[FEATURES], reg, proba], axis=1)
        out["corrected_rain"] = corrected.round(2)
        out["bias_offset"] = (corrected - df["nwp_rain"].values).round(2)
        for t in THRESHOLDS:
            out[f"prob_{t}"] = thr[t].round(4)
        return out

    def predict_one(self, features: dict) -> dict:
        row = self.predict(pd.DataFrame([features])).iloc[0]
        return {
            "input": {k: float(row[k]) for k in FEATURES},
            "regime": {
                "name": row["regime"], "label": row["regime_label"],
                "confidence": float(row["confidence"]),
                "probabilities": {REGIME_LABELS[r]: round(float(row[f"p_{r}"]) * 100, 1)
                                  for r in REGIMES},
            },
            "bias_correction": {
                "raw_nwp": round(float(row["nwp_rain"]), 2),
                "corrected": float(row["corrected_rain"]),
                "offset": float(row["bias_offset"]),
            },
            "threshold": {
                "probabilities": {f">{t}mm": round(float(row[f"prob_{t}"]) * 100, 1)
                                  for t in THRESHOLDS},
                "alert_threshold_mm": ALERT_THRESHOLD,
                "alert": bool(row[f"prob_{ALERT_THRESHOLD}"] >= DECISION_CUTOFF),
            },
        }

    # -------------------------------------------------------------- persistence
    def save(self):
        self.regime.save(REGIME_PATH)
        self.bias.save(BIAS_PATH)
        self.threshold.save(THR_PATH)

    @classmethod
    def load(cls):
        p = cls()
        p.regime = RegimeClassifier.load(REGIME_PATH)
        p.bias = BiasCorrector.load(BIAS_PATH)
        p.threshold = ThresholdPredictor.load(THR_PATH)
        return p

    @staticmethod
    def models_exist() -> bool:
        return all(p.exists() for p in (REGIME_PATH, BIAS_PATH, THR_PATH))
