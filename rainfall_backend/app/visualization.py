"""
Visualization: matplotlib PNGs (returned as bytes / base64) and
Chart.js-ready JSON, styled to match the dark frontend.
"""
import base64
import io
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BG, FG, GRID = "#0d1424", "#f8fafc", "#334155"
CYAN, RED, GREEN, INDIGO, AMBER = "#00f2fe", "#ef4444", "#10b981", "#6366f1", "#f59e0b"


def _style(ax, title, xlabel="", ylabel=""):
    ax.set_facecolor(BG)
    ax.set_title(title, color=FG, fontsize=12, fontweight="bold")
    ax.set_xlabel(xlabel, color="#94a3b8")
    ax.set_ylabel(ylabel, color="#94a3b8")
    ax.tick_params(colors="#94a3b8")
    ax.grid(color=GRID, alpha=0.4)
    for s in ax.spines.values():
        s.set_color(GRID)


def _to_png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def to_base64(png: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png).decode()


# ------------------------------------------------------------------ PNG plots
def plot_comparison(labels, nwp, ai, actual) -> bytes:
    """Raw NWP vs AI-corrected vs actual over time (like the frontend chart)."""
    fig, ax = plt.subplots(figsize=(8, 4), facecolor=BG)
    ax.plot(labels, nwp, "--", color=RED, lw=2, label="Raw NWP (biased)")
    ax.plot(labels, ai, color=CYAN, lw=2.5, label="AI corrected")
    ax.plot(labels, actual, "o-", color=GREEN, lw=2, ms=6, label="Actual (rain gauge)")
    _style(ax, "Rainfall: NWP vs AI vs Actual", ylabel="Rainfall (mm)")
    ax.legend(facecolor=BG, edgecolor=GRID, labelcolor=FG)
    return _to_png(fig)


def plot_scatter(df: pd.DataFrame) -> bytes:
    """Predicted vs observed rainfall on the test set (NWP vs AI)."""
    fig, ax = plt.subplots(figsize=(6, 6), facecolor=BG)
    ax.scatter(df["actual_rain"], df["nwp_rain"], s=8, alpha=0.35, color=RED, label="Raw NWP")
    ax.scatter(df["actual_rain"], df["corrected_rain"], s=8, alpha=0.35, color=CYAN, label="AI corrected")
    m = float(max(df["actual_rain"].max(), df["nwp_rain"].max(), df["corrected_rain"].max()))
    ax.plot([0, m], [0, m], color=GREEN, lw=1.5, label="Perfect forecast")
    _style(ax, "Observed vs Predicted (test set)", "Observed (mm)", "Predicted (mm)")
    ax.legend(facecolor=BG, edgecolor=GRID, labelcolor=FG)
    return _to_png(fig)


def plot_regime_probs(probs: dict) -> bytes:
    """probs: {label: percent}"""
    items = sorted(probs.items(), key=lambda kv: kv[1])
    fig, ax = plt.subplots(figsize=(7, 3.6), facecolor=BG)
    ax.barh([k for k, _ in items], [v for _, v in items], color=INDIGO)
    ax.set_xlim(0, 100)
    _style(ax, "Model 1: Regime probabilities", "Probability (%)")
    return _to_png(fig)


def plot_threshold_probs(probs: dict) -> bytes:
    """probs: {'>50mm': percent}"""
    keys, vals = list(probs.keys()), list(probs.values())
    colors = [RED if v >= 50 else AMBER if v >= 25 else CYAN for v in vals]
    fig, ax = plt.subplots(figsize=(7, 3.6), facecolor=BG)
    bars = ax.bar(keys, vals, color=colors)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.1f}%", ha="center", color=FG)
    ax.set_ylim(0, 110)
    _style(ax, "Model 3: Threshold exceedance probability", ylabel="Probability (%)")
    return _to_png(fig)


def plot_bias_correction(raw: float, corrected: float, actual: float | None = None) -> bytes:
    names, vals, cols = ["Raw NWP", "AI corrected"], [raw, corrected], [RED, CYAN]
    if actual is not None:
        names.append("Actual"); vals.append(actual); cols.append(GREEN)
    fig, ax = plt.subplots(figsize=(5.5, 3.6), facecolor=BG)
    bars = ax.bar(names, vals, color=cols)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}", ha="center", va="bottom", color=FG)
    _style(ax, "Model 2: Bias correction", ylabel="Rainfall (mm)")
    return _to_png(fig)


def plot_metrics(report: dict) -> bytes:
    """Grouped bars: NWP vs AI for RMSE, MAE, POD, CSI, FAR."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.8), facecolor=BG,
                                 gridspec_kw={"width_ratios": [2, 3]})
    x = np.arange(2)
    for ax, keys, title in ((a1, ["RMSE", "MAE"], "Error (mm, lower better)"),
                            (a2, ["POD", "CSI", "FAR"], "Skill scores")):
        x = np.arange(len(keys))
        ax.bar(x - 0.2, [report[k]["nwp"] for k in keys], 0.4, color=RED, label="NWP")
        ax.bar(x + 0.2, [report[k]["ai"] for k in keys], 0.4, color=CYAN, label="AI")
        ax.set_xticks(x); ax.set_xticklabels(keys)
        _style(ax, title)
    a1.legend(facecolor=BG, edgecolor=GRID, labelcolor=FG)
    return _to_png(fig)


# ------------------------------------------------------------ Chart.js payload
def chartjs_comparison(labels, nwp, ai, actual) -> dict:
    """Drop-in data for the frontend's existing Chart.js line chart."""
    return {
        "labels": list(labels),
        "datasets": [
            {"label": "Raw NWP Forecast (Biased)", "data": [float(v) for v in nwp]},
            {"label": "AeroCorrect AI Post-Processed", "data": [float(v) for v in ai]},
            {"label": "Actual Ground Station Rain Gauge", "data": [float(v) for v in actual]},
        ],
    }
