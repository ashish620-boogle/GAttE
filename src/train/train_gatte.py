from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import tensorflow as tf

from src.eval.metrics import classification_metrics, distance_metrics
from src.models.gatte import build_gatte
from src.train.data_loader import load_meta, load_split
from src.utils.config import load_config, save_config
from src.utils.io import save_json, save_npz
from src.utils.run import create_run_dir, save_env_info
from src.utils.seed import set_seed


def train_gatte(cfg: Dict, run_dir: Path | None = None) -> Path:
    set_seed(cfg["seed"])
    processed_dir = Path(cfg["paths"]["processed_dir"])
    meta = load_meta(processed_dir)

    train_data = load_split(processed_dir, "train")
    val_data = load_split(processed_dir, "val")
    test_data = load_split(processed_dir, "test")

    num_classes = len(meta["label_list"])
    word_vocab_size = len(meta["word_vocab"])
    char_vocab_size = len(meta["char_vocab"])
    max_char_len = int(meta.get("max_char_len", cfg["preprocess"]["max_words"] * cfg["preprocess"].get("max_chars_per_word", 32)))

    use_precomputed = cfg["train"].get("use_precomputed", False)
    resume = cfg["train"].get("resume_if_exists", True)
    model_dir = None
    if run_dir is not None:
        model_dir = run_dir / "gatte"
        if resume and (model_dir / "model.keras").exists():
            return run_dir

    model = build_gatte(
        cfg,
        num_classes,
        word_vocab_size,
        char_vocab_size,
        max_char_len=max_char_len,
        use_precomputed=use_precomputed,
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=cfg["train"]["lr"])
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=cfg["train"].get("from_logits_loss", False))
    model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])

    X_train_use = train_data.get("use_vec")
    X_val_use = val_data.get("use_vec")
    X_test_use = test_data.get("use_vec")
    X_train_text = train_data["text"].astype(str)
    X_val_text = val_data["text"].astype(str)
    X_test_text = test_data["text"].astype(str)
    X_train_input = X_train_use if use_precomputed and X_train_use is not None else X_train_text
    X_val_input = X_val_use if use_precomputed and X_val_use is not None else X_val_text
    X_test_input = X_test_use if use_precomputed and X_test_use is not None else X_test_text

    y_train = tf.keras.utils.to_categorical(train_data["labels"], num_classes)
    y_val = tf.keras.utils.to_categorical(val_data["labels"], num_classes)

    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.1, patience=cfg["train"]["lr_patience"], verbose=1
        )
    ]

    history = model.fit(
        [X_train_input, train_data["words"], train_data["chars"]],
        y_train,
        validation_data=([
            X_val_input, val_data["words"], val_data["chars"]
        ], y_val),
        batch_size=cfg["train"]["batch_size"],
        epochs=cfg["train"]["epochs"],
        verbose=2,
        callbacks=callbacks,
    )

    # Evaluate on test
    preds = model.predict([X_test_input, test_data["words"], test_data["chars"]], batch_size=cfg["train"]["batch_size"])
    y_pred = np.argmax(preds, axis=1)
    y_true = test_data["labels"]

    cls_metrics = classification_metrics(y_true, y_pred)

    # distance metrics using class centroids
    centroid_map = {c["label"]: (c["latitude"], c["longitude"]) for c in meta["centroids"]}
    label_list = meta["label_list"]
    pred_labels = [label_list[i] for i in y_pred]
    pred_coords = np.array([centroid_map[l] for l in pred_labels])
    dist_metrics = distance_metrics(
        test_data["latitude"],
        test_data["longitude"],
        pred_coords[:, 0],
        pred_coords[:, 1],
        cfg["eval"]["k_values"],
    )

    if run_dir is None:
        run_dir = create_run_dir(cfg["paths"]["outputs_dir"], tag="gatte")
        save_env_info(run_dir)
        save_config(cfg, run_dir / "config.yaml")

    model_dir = run_dir / "gatte"
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save(model_dir / "model.keras")

    save_json({"history": history.history}, model_dir / "history.json")
    save_npz(model_dir / "preds.npz", y_true=y_true, y_pred=y_pred, y_prob=preds)

    metrics = {
        **cls_metrics,
        "average_distance_error": dist_metrics["average_distance_error"],
        "spatial_precision_at_161km": dist_metrics["spatial_precision_at_161km"],
        "spatial_precision_by_km": dist_metrics["spatial_precision_by_km"],
    }
    save_json(metrics, model_dir / "metrics.json")
    save_npz(model_dir / "distances.npz", distances=dist_metrics["distances"], spatial_precision=dist_metrics["spatial_precision"])

    return run_dir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    train_gatte(cfg)


if __name__ == "__main__":
    main()
