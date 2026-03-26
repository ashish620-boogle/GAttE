from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import tensorflow as tf
from transformers import AutoTokenizer

from src.eval.metrics import classification_metrics, distance_metrics
from src.train.data_loader import load_meta, load_split
from src.utils.config import load_config, save_config
from src.utils.io import save_json, save_npz
from src.utils.run import create_run_dir, save_env_info
from src.utils.seed import set_seed


def train_bert(cfg: Dict, run_dir: Path | None = None) -> Path:
    set_seed(cfg["seed"])
    processed_dir = Path(cfg["paths"]["processed_dir"])
    meta = load_meta(processed_dir)

    train_data = load_split(processed_dir, "train")
    val_data = load_split(processed_dir, "val")
    test_data = load_split(processed_dir, "test")

    num_classes = len(meta["label_list"])

    resume = cfg["train"].get("resume_if_exists", True)
    if run_dir is not None:
        out_dir = run_dir / "bert"
        if resume and (out_dir / "model.keras").exists():
            return run_dir

    tokenizer = AutoTokenizer.from_pretrained(cfg["bert"]["model_name"])
    # Build a small TF classifier using PyTorch BERT embeddings (avoids TF/Keras incompat)
    from transformers import AutoModel

    pt_model = AutoModel.from_pretrained(cfg["bert"]["model_name"])
    pt_model.eval()

    input_ids = tf.keras.Input(shape=(cfg["bert"]["max_len"],), dtype=tf.int32, name="input_ids")
    attention_mask = tf.keras.Input(shape=(cfg["bert"]["max_len"],), dtype=tf.int32, name="attention_mask")

    def pt_forward(ids, mask):
        import torch

        with torch.no_grad():
            outputs = pt_model(input_ids=torch.tensor(ids), attention_mask=torch.tensor(mask))
            pooled = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        return pooled

    bert_emb = tf.keras.layers.Lambda(
        lambda x: tf.numpy_function(pt_forward, x, tf.float32), name="pt_bert"
    )([input_ids, attention_mask])
    bert_emb = tf.keras.layers.Lambda(lambda x: tf.ensure_shape(x, (None, 768)), name="pt_bert_shape")(bert_emb)
    bert_emb = tf.keras.layers.Dense(768, activation="relu")(bert_emb)
    logits = tf.keras.layers.Dense(num_classes, activation=None)(bert_emb)
    model = tf.keras.Model(inputs=[input_ids, attention_mask], outputs=logits)

    def make_dataset(texts, labels, shuffle=False):
        enc = tokenizer(
            list(texts),
            truncation=True,
            padding="max_length",
            max_length=cfg["bert"]["max_len"],
            return_tensors="np",
        )
        ds = tf.data.Dataset.from_tensor_slices(((enc["input_ids"], enc["attention_mask"]), labels))
        if shuffle:
            ds = ds.shuffle(1000)
        ds = ds.batch(cfg["bert"]["batch_size"])
        return ds

    train_ds = make_dataset(train_data["text"], train_data["labels"], shuffle=True)
    val_ds = make_dataset(val_data["text"], val_data["labels"], shuffle=False)

    optimizer = tf.keras.optimizers.Adam(learning_rate=cfg["bert"]["lr"])
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])

    model.fit(train_ds, validation_data=val_ds, epochs=cfg["bert"]["epochs"], verbose=2)

    # Evaluate
    test_ds = make_dataset(test_data["text"], test_data["labels"], shuffle=False)
    logits = model.predict(test_ds)
    y_pred = np.argmax(logits, axis=1)
    y_true = test_data["labels"]

    cls_metrics = classification_metrics(y_true, y_pred)

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
        run_dir = create_run_dir(cfg["paths"]["outputs_dir"], tag="bert")
        save_env_info(run_dir)
        save_config(cfg, run_dir / "config.yaml")

    out_dir = run_dir / "bert"
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save(out_dir / "model.keras")
    tokenizer.save_pretrained(str(out_dir / "tokenizer"))

    save_npz(out_dir / "preds.npz", y_true=y_true, y_pred=y_pred, logits=logits)
    metrics = {**cls_metrics, **{"average_distance_error": dist_metrics["average_distance_error"]}}
    save_json(metrics, out_dir / "metrics.json")
    save_npz(out_dir / "distances.npz", distances=dist_metrics["distances"], spatial_precision=dist_metrics["spatial_precision"])

    return run_dir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    train_bert(cfg)


if __name__ == "__main__":
    main()
