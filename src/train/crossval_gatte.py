from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold

from src.eval.metrics import classification_metrics
from src.models.gatte import build_gatte
from src.train.data_loader import load_meta, load_split
from src.utils.config import load_config, save_config
from src.utils.io import save_json, load_json, save_npz
from src.utils.run import create_run_dir, save_env_info
from src.utils.seed import set_seed


def crossval_gatte(cfg: Dict, run_dir: Path | None = None) -> Path:
    set_seed(cfg["seed"])
    processed_dir = Path(cfg["paths"]["processed_dir"])
    meta = load_meta(processed_dir)
    train_data = load_split(processed_dir, "train")
    val_data = load_split(processed_dir, "val")

    num_classes = len(meta["label_list"])
    word_vocab_size = len(meta["word_vocab"])
    char_vocab_size = len(meta["char_vocab"])
    max_char_len = int(meta.get("max_char_len", cfg["preprocess"]["max_words"] * cfg["preprocess"].get("max_chars_per_word", 32)))

    use_precomputed = cfg["train"].get("use_precomputed", False)
    X_text = np.concatenate([train_data["text"].astype(str), val_data["text"].astype(str)])
    X_use = None
    if train_data.get("use_vec") is not None and val_data.get("use_vec") is not None:
        X_use = np.concatenate([train_data["use_vec"], val_data["use_vec"]])
    X_input = X_use if use_precomputed and X_use is not None else X_text
    X_words = np.concatenate([train_data["words"], val_data["words"]])
    X_chars = np.concatenate([train_data["chars"], val_data["chars"]])
    y = np.concatenate([train_data["labels"], val_data["labels"]])

    kfolds = cfg["train"]["kfolds"]
    kf = KFold(n_splits=kfolds, shuffle=True, random_state=cfg["seed"])

    fold_metrics: List[Dict] = []
    resume = cfg["train"].get("resume_if_exists", True)

    crossval_dir = None
    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_words), start=1):
        if run_dir is None:
            run_dir = create_run_dir(cfg["paths"]["outputs_dir"], tag="crossval")
            save_env_info(run_dir)
            save_config(cfg, run_dir / "config.yaml")

        crossval_dir = run_dir / "crossval"
        crossval_dir.mkdir(parents=True, exist_ok=True)
        fold_dir = crossval_dir / f"fold_{fold:02d}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        metrics_path = fold_dir / "metrics.json"
        if resume and metrics_path.exists():
            metrics = load_json(metrics_path)
            metrics["fold"] = fold
            fold_metrics.append(metrics)
            continue

        model = build_gatte(
            cfg,
            num_classes,
            word_vocab_size,
            char_vocab_size,
            max_char_len=max_char_len,
            use_precomputed=use_precomputed,
        )
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=cfg["train"]["lr"]),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

        y_tr = tf.keras.utils.to_categorical(y[tr_idx], num_classes)
        y_va = tf.keras.utils.to_categorical(y[va_idx], num_classes)

        history = model.fit(
            [X_input[tr_idx], X_words[tr_idx], X_chars[tr_idx]],
            y_tr,
            validation_data=([
                X_input[va_idx], X_words[va_idx], X_chars[va_idx]
            ], y_va),
            batch_size=cfg["train"]["batch_size"],
            epochs=cfg["train"]["epochs"],
            verbose=0,
        )

        preds = model.predict([X_input[va_idx], X_words[va_idx], X_chars[va_idx]], batch_size=cfg["train"]["batch_size"])
        y_pred = np.argmax(preds, axis=1)
        metrics = classification_metrics(y[va_idx], y_pred)
        metrics["fold"] = fold
        fold_metrics.append(metrics)

        model.save(fold_dir / "model.keras")
        save_json({"history": history.history}, fold_dir / "history.json")
        save_npz(fold_dir / "preds.npz", y_true=y[va_idx], y_pred=y_pred, y_prob=preds)
        save_json(metrics, metrics_path)

    avg_metrics = {
        "precision": float(np.mean([m["precision"] for m in fold_metrics])),
        "recall": float(np.mean([m["recall"] for m in fold_metrics])),
        "f1": float(np.mean([m["f1"] for m in fold_metrics])),
        "accuracy": float(np.mean([m["accuracy"] for m in fold_metrics])),
    }

    if run_dir is None:
        run_dir = create_run_dir(cfg["paths"]["outputs_dir"], tag="crossval")
        save_env_info(run_dir)
        save_config(cfg, run_dir / "config.yaml")

    out = {"folds": fold_metrics, "average": avg_metrics}
    crossval_dir = run_dir / "crossval"
    crossval_dir.mkdir(parents=True, exist_ok=True)
    save_json(out, crossval_dir / "crossval_metrics.json")
    return run_dir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    crossval_gatte(cfg)


if __name__ == "__main__":
    main()
