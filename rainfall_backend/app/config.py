"""Central configuration: paths, feature names, regimes, thresholds."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"

# Put your real IMD/NWP training data here (see README for columns).
TRAIN_CSV = DATA_DIR / "training_data.csv"

RANDOM_STATE = 42

# Input features (NWP output + atmospheric/terrain info)
FEATURES = [
    "nwp_rain", "temp", "humidity", "wind", "pressure",
    "cape", "terrain_height", "lat", "lon", "month",
]

# Sorted so column order is identical everywhere (sklearn sorts classes).
REGIMES = sorted([
    "active_monsoon", "break_monsoon", "coastal_depression",
    "orographic", "western_disturbance", "depression",
])
REGIME_LABELS = {
    "active_monsoon": "Active Monsoon",
    "break_monsoon": "Break Monsoon",
    "coastal_depression": "Coastal Depression",
    "orographic": "Orographic Uplift",
    "western_disturbance": "Western Disturbance",
    "depression": "Depression / Low Pressure",
}
REGIME_PROB_COLS = [f"p_{r}" for r in REGIMES]

# Rainfall thresholds (mm/day) the threshold model predicts
THRESHOLDS = [10, 30, 50, 100]
ALERT_THRESHOLD = 50          # heavy-rain warning level
DECISION_CUTOFF = 0.5         # prob >= cutoff -> "YES"

# Defaults used when a request omits optional features
FEATURE_DEFAULTS = {
    "temp": 26.0, "humidity": 85.0, "wind": 25.0, "pressure": 1000.0,
    "cape": 1200.0, "terrain_height": 100.0,
    "lat": 17.385, "lon": 78.486, "month": 8,
}

ALLOWED_ORIGINS = [
    o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",") if o.strip()
]
