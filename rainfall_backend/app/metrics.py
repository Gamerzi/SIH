"""Verification metrics: RMSE, MAE, POD, FAR, CSI."""
import numpy as np


def rmse(obs, pred) -> float:
    return float(np.sqrt(np.mean((np.asarray(obs) - np.asarray(pred)) ** 2)))


def mae(obs, pred) -> float:
    return float(np.mean(np.abs(np.asarray(obs) - np.asarray(pred))))


def categorical_scores(obs_event, pred_event) -> dict:
    """POD / FAR / CSI from boolean arrays (event = rain >= threshold)."""
    o = np.asarray(obs_event).astype(bool)
    p = np.asarray(pred_event).astype(bool)
    hits = int(np.sum(o & p))
    misses = int(np.sum(o & ~p))
    false_alarms = int(np.sum(~o & p))
    pod = hits / (hits + misses) if (hits + misses) else 0.0
    far = false_alarms / (hits + false_alarms) if (hits + false_alarms) else 0.0
    csi = hits / (hits + misses + false_alarms) if (hits + misses + false_alarms) else 0.0
    return {"POD": round(pod, 3), "FAR": round(far, 3), "CSI": round(csi, 3),
            "hits": hits, "misses": misses, "false_alarms": false_alarms}


def pct_change(before: float, after: float) -> float:
    return round((after - before) / before * 100, 1) if before else 0.0
