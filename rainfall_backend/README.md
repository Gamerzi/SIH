# AeroCorrect AI - Python Backend

Separate backend for the AeroCorrect frontend. It does not touch the Vercel frontend;
you connect it yourself by calling these endpoints.

## Structure
```
app/
  config.py            features, regimes, thresholds, paths, CORS
  data_generator.py    load real CSV or make SYNTHETIC data
  regime_classifier.py Model 1 - weather regime (RandomForest)
  bias_correction.py   Model 2 - corrected rainfall (Gradient Boosting)
  threshold_model.py   Model 3 - P(rain >= 10/30/50/100 mm)
  metrics.py           RMSE, MAE, POD, FAR, CSI
  pipeline.py          trains + chains the 3 models
  visualization.py     matplotlib PNGs + Chart.js JSON
  schemas.py           request validation
  main.py              FastAPI endpoints
train.py               train from command line
```

## Run
```
pip install -r requirements.txt
python train.py                      # optional, auto-trains on first server start
uvicorn app.main:app --reload --port 8000
```
Docs: http://localhost:8000/docs

## IMPORTANT: real data
With no data file the models train on SYNTHETIC data, so the metrics are NOT real results.
For real results put `data/training_data.csv` with columns:
`nwp_rain, temp, humidity, wind, pressure, cape, terrain_height, lat, lon, month, actual_rain, regime`
(common aliases like NWP / Forecast / Observed are auto-renamed), then run `python train.py`.
The `regime` column needs labels: one of active_monsoon, break_monsoon, depression,
coastal_depression, orographic, western_disturbance.

## Endpoints
| Method | Path | Purpose |
|---|---|---|
| POST | /api/regime | Model 1 |
| POST | /api/bias-correct | Model 1 + 2 |
| POST | /api/threshold | Model 1 + 2 + 3 |
| POST | /api/predict | full pipeline |
| POST | /api/predict/batch | upload CSV/Excel |
| POST | /api/visualize | full pipeline + base64 PNG plots |
| GET  | /api/metrics | RMSE / MAE / POD / CSI / FAR |
| GET  | /api/plots/scatter.png, /api/plots/metrics.png | plots |
| POST | /api/train | retrain |

Example:
```
curl -X POST localhost:8000/api/predict -H "Content-Type: application/json" \
  -d '{"nwp_rain":21.8,"humidity":97,"wind":36,"month":9}'
```
Frontend usage:
```js
const r = await fetch("https://YOUR-BACKEND/api/predict", {method:"POST",
  headers:{"Content-Type":"application/json"}, body: JSON.stringify({nwp_rain: 21.8})});
const d = await r.json();  // d.regime.label, d.bias_correction.corrected, d.threshold.probabilities
```

## Deploy / CORS
Vercel can't host this (sklearn + long-running server). Use Render, Railway or Fly.io
(start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`).
Set `ALLOWED_ORIGINS=https://your-app.vercel.app` (default `*`).
