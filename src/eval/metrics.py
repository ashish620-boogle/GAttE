from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


def haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return 6371.0 * c


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    labels = np.unique(y_true)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average="macro",
        zero_division=0,
    )
    acc = accuracy_score(y_true, y_pred)
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(acc),
    }


def distance_metrics(
    lat_true: np.ndarray,
    lon_true: np.ndarray,
    lat_pred: np.ndarray,
    lon_pred: np.ndarray,
    k_values: List[int],
) -> Dict[str, np.ndarray]:
    dist = haversine_km(lat_true, lon_true, lat_pred, lon_pred)
    avg_err = float(np.mean(dist))
    spatial_precision = [float((dist <= k).mean() * 100.0) for k in k_values]
    spatial_precision_by_km = {int(k): float(v) for k, v in zip(k_values, spatial_precision)}
    return {
        "average_distance_error": avg_err,
        "spatial_precision": np.array(spatial_precision),
        "spatial_precision_by_km": spatial_precision_by_km,
        "spatial_precision_at_161km": float((dist <= 161).mean() * 100.0),
        "distances": dist,
    }


def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return confusion_matrix(y_true, y_pred)
