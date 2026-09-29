"""
Data loading + SYNTHETIC data generator.

The synthetic generator exists only so the backend can train and run
before you have real data. Replace it by putting real historical
NWP-vs-observed data in data/training_data.csv.
"""
import numpy as np
import pandas as pd
from .config import FEATURES, REGIMES, RANDOM_STATE, TRAIN_CSV

# Column-name aliases so different IMD/NWP spreadsheets can be read.
ALIASES = {
    "nwp_rain": ["nwp", "predicted", "forecast", "nwp_rain", "nwp_rainfall", "nwp (mm)"],
    "temp": ["temp", "temperature", "t2m"],
    "humidity": ["humidity", "rh", "relative_humidity"],
    "wind": ["wind", "wind_speed", "ws"],
    "pressure": ["pressure", "mslp", "slp"],
    "cape": ["cape"],
    "terrain_height": ["terrain_height", "elevation", "height", "orography"],
    "lat": ["lat", "latitude"],
    "lon": ["lon", "longitude", "long"],
    "month": ["month"],
    "actual_rain": ["actual", "actual_rain", "observed", "obs", "rain_gauge", "imd_rain"],
    "regime": ["regime", "weather_regime"],
}


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns to canonical names using ALIASES (case-insensitive)."""
    lower = {c.lower().strip(): c for c in df.columns}
    rename = {}
    for canon, options in ALIASES.items():
        for opt in options:
            if opt in lower and lower[opt] not in rename:
                rename[lower[opt]] = canon
                break
    out = df.rename(columns=rename)
    if "humidity" in out and out["humidity"].dtype == object:   # "98%" -> 98
        out["humidity"] = out["humidity"].astype(str).str.replace("%", "").astype(float)
    return out


def load_training_data() -> pd.DataFrame:
    """Real data if present, otherwise synthetic."""
    if TRAIN_CSV.exists():
        df = normalize_columns(pd.read_csv(TRAIN_CSV))
        needed = FEATURES + ["actual_rain", "regime"]
        missing = [c for c in needed if c not in df.columns]
        if missing:
            raise ValueError(f"{TRAIN_CSV.name} is missing columns: {missing}")
        return df[needed].dropna().reset_index(drop=True)
    return generate_synthetic(6000)


_SPEC = {
    "active_monsoon":      dict(hum=(90, 99), wind=(25, 45), pres=(995, 1004), cape=(800, 1800), ter=(0, 300),   lat=(12, 25), mon=[6, 7, 8, 9], nwp=22, ratio=(2.5, 4.5)),
    "break_monsoon":       dict(hum=(50, 75), wind=(5, 18),  pres=(1003, 1012), cape=(200, 900), ter=(0, 400),   lat=(15, 30), mon=[7, 8, 9],    nwp=25, ratio=(0.1, 0.5)),
    "depression":          dict(hum=(90, 100), wind=(40, 70), pres=(975, 995), cape=(500, 1500), ter=(0, 200),   lat=(12, 22), mon=[8, 9, 10],   nwp=30, ratio=(2.0, 4.0)),
    "coastal_depression":  dict(hum=(92, 100), wind=(45, 75), pres=(970, 992), cape=(600, 1600), ter=(0, 30),    lat=(8, 20),  mon=[10, 11, 12], nwp=31, ratio=(2.5, 4.0)),
    "orographic":          dict(hum=(85, 97), wind=(15, 35), pres=(980, 1000), cape=(400, 1300), ter=(800, 2500), lat=(28, 34), mon=[6, 7, 8, 9], nwp=15, ratio=(3.0, 6.5)),
    "western_disturbance": dict(hum=(40, 70), wind=(8, 25),  pres=(1005, 1018), cape=(50, 500), ter=(100, 600),  lat=(26, 34), mon=[12, 1, 2, 3], nwp=20, ratio=(0.2, 0.9)),
}


def generate_synthetic(n: int = 6000, seed: int = RANDOM_STATE) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    per = n // len(REGIMES)
    for regime in REGIMES:
        s = _SPEC[regime]
        nwp = rng.lognormal(mean=np.log(s["nwp"]), sigma=0.55, size=per)
        ratio = rng.uniform(*s["ratio"], size=per)
        noise = rng.normal(0, 0.12, size=per)
        actual = np.clip(nwp * ratio * (1 + noise), 0, None)
        rows.append(pd.DataFrame({
            "nwp_rain": nwp,
            "temp": rng.normal(27 if regime != "western_disturbance" else 14, 3, per),
            "humidity": rng.uniform(*s["hum"], per),
            "wind": rng.uniform(*s["wind"], per),
            "pressure": rng.uniform(*s["pres"], per),
            "cape": rng.uniform(*s["cape"], per),
            "terrain_height": rng.uniform(*s["ter"], per),
            "lat": rng.uniform(*s["lat"], per),
            "lon": rng.uniform(70, 88, per),
            "month": rng.choice(s["mon"], per),
            "actual_rain": actual,
            "regime": regime,
        }))
    return pd.concat(rows, ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)
