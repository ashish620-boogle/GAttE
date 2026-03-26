from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from src.train.data_loader import load_meta, load_split
from src.utils.io import load_npz


def find_latest_run(outputs_dir: Path) -> Path:
    runs = sorted((outputs_dir / "runs").glob("*"))
    if not runs:
        raise FileNotFoundError("No runs found in outputs/runs")
    return runs[-1]


def plot_spatial_precision(run_dir: Path, k_values: np.ndarray, out_dir: Path) -> None:
    plt.figure(figsize=(8, 5))
    for model in ["gatte", "mgatte", "bert", "spacy"]:
        dist_path = run_dir / model / "distances.npz"
        if not dist_path.exists():
            continue
        dist = load_npz(dist_path)
        sp = dist.get("spatial_precision")
        if sp is None:
            continue
        plt.plot(k_values, sp, label=model)
    plt.xlabel("Displacement k (km)")
    plt.ylabel("Spatial Precision (%)")
    plt.legend()
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_dir / "spatial_precision.png", dpi=150)
    plt.close()


def plot_confusion(y_true: np.ndarray, y_pred: np.ndarray, out_dir: Path) -> None:
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, cmap="Blues", cbar=False)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_dir / "confusion_matrix.png", dpi=150)
    plt.close()


def plot_class_metrics(y_true: np.ndarray, y_pred_a: np.ndarray, y_pred_b: np.ndarray, out_dir: Path) -> None:
    labels = np.unique(y_true)
    prec_a, rec_a, f1_a, _ = precision_recall_fscore_support(y_true, y_pred_a, labels=labels, zero_division=0)
    prec_b, rec_b, f1_b, _ = precision_recall_fscore_support(y_true, y_pred_b, labels=labels, zero_division=0)

    x = np.arange(len(labels))
    width = 0.35

    for metric_name, a_vals, b_vals in [
        ("precision", prec_a, prec_b),
        ("recall", rec_a, rec_b),
        ("f1", f1_a, f1_b),
    ]:
        plt.figure(figsize=(12, 5))
        plt.bar(x - width / 2, a_vals, width, label="GAttE")
        plt.bar(x + width / 2, b_vals, width, label="BERT")
        plt.title(f"Class-level {metric_name}")
        plt.xlabel("Class index")
        plt.ylabel(metric_name)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"class_{metric_name}.png", dpi=150)
        plt.close()


def plot_geo_scatter(lat_true, lon_true, lat_pred, lon_pred, out_dir: Path) -> None:
    plt.figure(figsize=(8, 5))
    plt.scatter(lon_true, lat_true, c="red", s=10, label="True", alpha=0.6)
    plt.scatter(lon_pred, lat_pred, c="yellow", s=10, label="Pred", alpha=0.6)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "geo_scatter.png", dpi=150)
    plt.close()


def make_figures(run_dir: Path, processed_dir: Path) -> None:
    out_dir = run_dir / "figures"

    meta = load_meta(processed_dir)
    test_data = load_split(processed_dir, "test")

    k_values = np.array(list(range(0, 5001, 100)))
    plot_spatial_precision(run_dir, k_values, out_dir)

    gatte_preds = load_npz(run_dir / "gatte" / "preds.npz")
    plot_confusion(gatte_preds["y_true"], gatte_preds["y_pred"], out_dir)

    bert_preds = load_npz(run_dir / "bert" / "preds.npz")
    plot_class_metrics(gatte_preds["y_true"], gatte_preds["y_pred"], bert_preds["y_pred"], out_dir)

    label_list = meta["label_list"]
    centroid_map = {c["label"]: (c["latitude"], c["longitude"]) for c in meta["centroids"]}
    pred_labels = [label_list[i] for i in gatte_preds["y_pred"]]
    pred_coords = np.array([centroid_map[l] for l in pred_labels])
    plot_geo_scatter(test_data["latitude"], test_data["longitude"], pred_coords[:, 0], pred_coords[:, 1], out_dir)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run_dir", default="")
    args = ap.parse_args()

    outputs_dir = Path("outputs")
    run_dir = Path(args.run_dir) if args.run_dir else find_latest_run(outputs_dir)
    processed_dir = Path("data/processed")
    make_figures(run_dir, processed_dir)


if __name__ == "__main__":
    main()
