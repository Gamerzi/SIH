"""Train all 3 models from the command line:  python train.py"""
import json
from app.pipeline import RainfallPipeline
from app import visualization as viz
from app.config import OUTPUT_DIR
import pandas as pd

if __name__ == "__main__":
    report = RainfallPipeline().train()
    print(json.dumps(report, indent=2))
    (OUTPUT_DIR / "metrics.png").write_bytes(viz.plot_metrics(report))
    df = pd.read_csv(OUTPUT_DIR / "test_predictions.csv")
    (OUTPUT_DIR / "scatter.png").write_bytes(viz.plot_scatter(df))
    print(f"\nModels saved in models/, plots saved in {OUTPUT_DIR}")
