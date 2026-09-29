"""
FastAPI app. Run:  uvicorn app.main:app --reload --port 8000

Endpoints (all under /api):
  GET  /health
  POST /api/regime          Model 1 only
  POST /api/bias-correct    Models 1+2
  POST /api/threshold       Models 1+2+3 (probabilities)
  POST /api/predict         full pipeline (regime + corrected + thresholds)
  POST /api/predict/batch   upload CSV/Excel -> per-row predictions + chart data
  POST /api/visualize       full pipeline + base64 PNG plots
  GET  /api/metrics         validation metrics (RMSE/MAE/POD/CSI/FAR)
  GET  /api/plots/{name}.png   scatter | metrics
  POST /api/train           retrain on data/training_data.csv (or synthetic)
"""
import io
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from . import visualization as viz
from .config import ALLOWED_ORIGINS, ALERT_THRESHOLD, THRESHOLDS
from .pipeline import RainfallPipeline, METRICS_PATH, TEST_PRED_PATH
from .schemas import PredictRequest
import json

state: dict = {"pipe": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    if RainfallPipeline.models_exist():
        state["pipe"] = RainfallPipeline.load()
    else:                                   # first run: train automatically
        pipe = RainfallPipeline()
        pipe.train()
        state["pipe"] = pipe
    yield


app = FastAPI(title="AeroCorrect AI Backend", version="1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"], allow_headers=["*"],
)


def pipe() -> RainfallPipeline:
    if state["pipe"] is None:
        raise HTTPException(503, "Models not loaded")
    return state["pipe"]


@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": state["pipe"] is not None}


@app.post("/api/regime")
def regime(req: PredictRequest):
    r = pipe().predict_one(req.features())
    return {"input": r["input"], "regime": r["regime"]}


@app.post("/api/bias-correct")
def bias_correct(req: PredictRequest):
    r = pipe().predict_one(req.features())
    return {"regime": r["regime"], "bias_correction": r["bias_correction"]}


@app.post("/api/threshold")
def threshold(req: PredictRequest):
    r = pipe().predict_one(req.features())
    return {"bias_correction": r["bias_correction"], "threshold": r["threshold"]}


@app.post("/api/predict")
def predict(req: PredictRequest):
    return pipe().predict_one(req.features())


@app.post("/api/predict/batch")
async def predict_batch(file: UploadFile = File(...)):
    raw = await file.read()
    try:
        df = (pd.read_excel(io.BytesIO(raw)) if file.filename.lower().endswith((".xlsx", ".xls"))
              else pd.read_csv(io.BytesIO(raw)))
        out = pipe().predict(df)
    except Exception as e:
        raise HTTPException(400, f"Could not process file: {e}")

    rows = out.head(500)
    result = {
        "rows": len(out),
        "summary": {
            "mean_raw_nwp": round(float(out["nwp_rain"].mean()), 2),
            "mean_corrected": round(float(out["corrected_rain"].mean()), 2),
            f"mean_prob_{ALERT_THRESHOLD}mm": round(float(out[f"prob_{ALERT_THRESHOLD}"].mean()) * 100, 1),
        },
        "predictions": json.loads(rows.round(3).to_json(orient="records")),
    }
    actual_col = next((c for c in df.columns if c.lower() in
                       ("actual", "actual_rain", "observed", "obs")), None)
    if actual_col is not None:
        labels = [str(i + 1) for i in range(len(rows))]
        result["chart"] = viz.chartjs_comparison(
            labels, rows["nwp_rain"], rows["corrected_rain"],
            pd.to_numeric(df[actual_col], errors="coerce").fillna(0).head(500))
    return result


@app.post("/api/visualize")
def visualize(req: PredictRequest):
    r = pipe().predict_one(req.features())
    b, t = r["bias_correction"], r["threshold"]
    return {
        **r,
        "plots": {
            "regime": viz.to_base64(viz.plot_regime_probs(r["regime"]["probabilities"])),
            "bias_correction": viz.to_base64(
                viz.plot_bias_correction(b["raw_nwp"], b["corrected"], req.actual_rain)),
            "threshold": viz.to_base64(viz.plot_threshold_probs(t["probabilities"])),
        },
    }


@app.get("/api/metrics")
def metrics():
    if not METRICS_PATH.exists():
        raise HTTPException(404, "No metrics yet - call POST /api/train")
    return json.loads(METRICS_PATH.read_text())


@app.get("/api/plots/{name}.png")
def plots(name: str):
    if name == "metrics":
        if not METRICS_PATH.exists():
            raise HTTPException(404, "No metrics yet")
        png = viz.plot_metrics(json.loads(METRICS_PATH.read_text()))
    elif name == "scatter":
        if not TEST_PRED_PATH.exists():
            raise HTTPException(404, "No test predictions yet")
        png = viz.plot_scatter(pd.read_csv(TEST_PRED_PATH))
    else:
        raise HTTPException(404, "Unknown plot. Use: scatter, metrics")
    return Response(png, media_type="image/png")


@app.post("/api/train")
def train():
    report = pipe().train() if state["pipe"] else RainfallPipeline().train()
    state["pipe"] = RainfallPipeline.load()
    return report
