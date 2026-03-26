from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd

from src.utils.io import load_json


def find_latest_run(outputs_dir: Path) -> Path:
    runs = sorted((outputs_dir / "runs").glob("*"))
    if not runs:
        raise FileNotFoundError("No runs found in outputs/runs")
    return runs[-1]


def evaluate(run_dir: Path) -> None:
    models = ["gatte", "mgatte", "bert", "spacy"]
    rows = []
    for m in models:
        mdir = run_dir / m
        metrics_path = mdir / "metrics.json"
        if not metrics_path.exists():
            continue
        metrics = load_json(metrics_path)
        row = {
            "model": m,
            "precision": metrics.get("precision"),
            "recall": metrics.get("recall"),
            "f1": metrics.get("f1"),
            "accuracy": metrics.get("accuracy"),
            "average_distance_error": metrics.get("average_distance_error"),
            "spatial_precision_at_161km": metrics.get("spatial_precision_at_161km"),
        }
        rows.append(row)

    if not rows:
        return

    df = pd.DataFrame(rows)
    out_dir = run_dir / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "performance_summary.csv", index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run_dir", default="")
    args = ap.parse_args()

    outputs_dir = Path("outputs")
    run_dir = Path(args.run_dir) if args.run_dir else find_latest_run(outputs_dir)
    evaluate(run_dir)


if __name__ == "__main__":
    main()
