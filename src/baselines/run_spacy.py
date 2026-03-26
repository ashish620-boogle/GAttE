from __future__ import annotations

import argparse
from pathlib import Path
import os
from typing import Dict, List, Tuple

import numpy as np
import spacy

from src.data.geocode import GeoNamesClient, get_geonames_username
from src.eval.metrics import classification_metrics, distance_metrics
from src.train.data_loader import load_meta, load_split
from src.utils.config import load_config, save_config
from src.utils.io import save_json, save_npz
from src.utils.run import create_run_dir, save_env_info
from src.utils.seed import set_seed


def run_spacy(cfg: Dict, run_dir: Path | None = None) -> Path:
    set_seed(cfg["seed"])
    processed_dir = Path(cfg["paths"]["processed_dir"])
    meta = load_meta(processed_dir)
    test_data = load_split(processed_dir, "test")

    label_list = meta["label_list"]
    label_to_idx = {l.lower(): i for i, l in enumerate(label_list)}
    centroids = {c["label"].lower(): (c["latitude"], c["longitude"]) for c in meta["centroids"]}

    resume = cfg["train"].get("resume_if_exists", True)
    if run_dir is not None:
        out_dir = run_dir / "spacy"
        if resume and (out_dir / "metrics.json").exists():
            return run_dir

    username = get_geonames_username(cfg["geocode"]["username_env"])
    rate_limit = float(os.getenv(cfg["geocode"]["rate_limit_env"], cfg["geocode"]["rate_limit"]))
    client = GeoNamesClient(
        username=username,
        cache_path=Path(cfg["paths"]["cache_dir"]) / "geocode.sqlite",
        rate_limit=rate_limit,
        round_precision=cfg["geocode"]["round"],
        country_code=cfg["dataset"]["country_code"],
        verify_ssl=cfg["geocode"].get("verify_ssl", True),
        allow_insecure_fallback=cfg["geocode"].get("allow_insecure_fallback", False),
    )

    nlp = spacy.load(cfg["spacy"]["model"])
    allowed_labels = set(cfg["spacy"]["labels"])
    rng = np.random.default_rng(cfg["seed"])

    preds = []
    pred_coords = []
    default_coords = np.array(centroids[label_list[0].lower()], dtype=float)

    for text in test_data["text"]:
        doc = nlp(str(text))
        ents = [e.text for e in doc.ents if e.label_ in allowed_labels]
        loc_text = " ".join(ents).strip()
        lat_lon = None
        pred_label = None

        if loc_text:
            # Try direct label match
            if loc_text.lower() in label_to_idx:
                pred_label = label_to_idx[loc_text.lower()]
                lat_lon = centroids.get(loc_text.lower())
            else:
                # If there is no exact class match, assign a deterministic random
                # class so macro precision/recall remain easier to interpret.
                pred_label = int(rng.integers(len(label_list)))
                geo = client.forward_geocode(loc_text)
                if geo:
                    lat_lon = (geo.lat, geo.lon)

        if pred_label is None:
            pred_label = int(rng.integers(len(label_list)))

        if lat_lon is None:
            lat_lon = default_coords

        preds.append(pred_label)
        pred_coords.append(lat_lon)

    y_pred = np.array(preds, dtype=np.int32)
    y_true = test_data["labels"]
    pred_coords = np.array(pred_coords, dtype=float)

    cls_metrics = classification_metrics(y_true, y_pred)
    dist_metrics = distance_metrics(
        test_data["latitude"],
        test_data["longitude"],
        pred_coords[:, 0],
        pred_coords[:, 1],
        cfg["eval"]["k_values"],
    )

    if run_dir is None:
        run_dir = create_run_dir(cfg["paths"]["outputs_dir"], tag="spacy")
        save_env_info(run_dir)
        save_config(cfg, run_dir / "config.yaml")

    out_dir = run_dir / "spacy"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_npz(out_dir / "preds.npz", y_true=y_true, y_pred=y_pred, coords_pred=pred_coords)
    metrics = {**cls_metrics, **{"average_distance_error": dist_metrics["average_distance_error"]}}
    save_json(metrics, out_dir / "metrics.json")
    save_npz(out_dir / "distances.npz", distances=dist_metrics["distances"], spatial_precision=dist_metrics["spatial_precision"])

    return run_dir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    run_spacy(cfg)


if __name__ == "__main__":
    main()
