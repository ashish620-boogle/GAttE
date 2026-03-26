from __future__ import annotations

import argparse
import csv
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from src.data.preprocess import preprocess
from src.train.data_loader import load_meta, load_split
from src.train.train_gatte import train_gatte
from src.utils.config import load_config
from src.utils.io import load_json, save_json


MODEL_ABLATIONS: Sequence[Tuple[str, str]] = (
    ("embeddings_full", "configs/ablations/embeddings_full.yaml"),
    ("embeddings_use_only", "configs/ablations/embeddings_use_only.yaml"),
    ("embeddings_word_only", "configs/ablations/embeddings_word_only.yaml"),
    ("embeddings_char_only", "configs/ablations/embeddings_char_only.yaml"),
    ("embeddings_use_char", "configs/ablations/embeddings_use_char.yaml"),
    ("deconv_off", "configs/ablations/deconv_off.yaml"),
    ("attention_off", "configs/ablations/attention_off.yaml"),
    ("attention_simple", "configs/ablations/attention_simple.yaml"),
    ("attention_self", "configs/ablations/attention_self.yaml"),
)

PREPROCESSING_ABLATIONS: Sequence[Tuple[str, str]] = (
    ("emoji_remove", "configs/ablations/emoji_remove.yaml"),
    ("emoji_keep", "configs/ablations/emoji_keep.yaml"),
)


def _select_experiments(
    experiments: Sequence[Tuple[str, str]],
    requested_names: set[str],
) -> List[Tuple[str, str]]:
    return [item for item in experiments if not requested_names or item[0] in requested_names]


def prepare_processed_data(cfg_path: str, processed_dir: Path) -> Dict[str, Any]:
    cfg = deepcopy(load_config(cfg_path))
    cfg["paths"]["processed_dir"] = str(processed_dir)
    preprocess(cfg)
    return cfg


def load_split_sizes(processed_dir: Path) -> Dict[str, int]:
    return {
        "train_samples": int(len(load_split(processed_dir, "train")["labels"])),
        "val_samples": int(len(load_split(processed_dir, "val")["labels"])),
        "test_samples": int(len(load_split(processed_dir, "test")["labels"])),
    }


def run_single(
    cfg_path: str,
    name: str,
    category: str,
    processed_dir: Path,
    output_dir: Path,
) -> Dict[str, Any]:
    cfg = deepcopy(load_config(cfg_path))
    cfg["paths"]["processed_dir"] = str(processed_dir)
    cfg["train"]["resume_if_exists"] = False

    run_dir = output_dir / name
    run_dir.mkdir(parents=True, exist_ok=True)
    train_gatte(cfg, run_dir=run_dir)

    metrics = load_json(run_dir / "gatte" / "metrics.json")
    meta = load_meta(processed_dir)
    split_sizes = load_split_sizes(processed_dir)

    return {
        "name": name,
        "category": category,
        "config_path": cfg_path,
        "processed_dir": str(processed_dir),
        "run_dir": str(run_dir),
        "num_classes": int(len(meta["label_list"])),
        "num_samples": int(meta.get("num_samples", sum(split_sizes.values()))),
        "train_samples": split_sizes["train_samples"],
        "val_samples": split_sizes["val_samples"],
        "test_samples": split_sizes["test_samples"],
        "word_vocab_size": int(len(meta["word_vocab"])),
        "char_vocab_size": int(len(meta["char_vocab"])),
        "max_char_len": int(meta["max_char_len"]),
        "precision": float(metrics["precision"]),
        "recall": float(metrics["recall"]),
        "f1": float(metrics["f1"]),
        "accuracy": float(metrics["accuracy"]),
        "average_distance_error": float(metrics["average_distance_error"]),
        "spatial_precision_at_161km": float(metrics["spatial_precision_at_161km"]),
    }


def write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return

    fieldnames = [
        "name",
        "category",
        "config_path",
        "processed_dir",
        "run_dir",
        "num_classes",
        "num_samples",
        "train_samples",
        "val_samples",
        "test_samples",
        "word_vocab_size",
        "char_vocab_size",
        "max_char_len",
        "precision",
        "recall",
        "f1",
        "accuracy",
        "average_distance_error",
        "spatial_precision_at_161km",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run ablation experiments on a shared canonical dataset.")
    ap.add_argument("--output_dir", default="outputs/ablations")
    ap.add_argument("--base_config", default="configs/paper.yaml")
    ap.add_argument("--names", nargs="*", default=[])
    args = ap.parse_args()

    root_dir = Path(args.output_dir)
    root_dir.mkdir(parents=True, exist_ok=True)

    processed_root = root_dir / "_processed"
    processed_root.mkdir(parents=True, exist_ok=True)

    requested_names = set(args.names)
    selected_model_ablations = _select_experiments(MODEL_ABLATIONS, requested_names)
    selected_preprocessing_ablations = _select_experiments(PREPROCESSING_ABLATIONS, requested_names)

    rows: List[Dict[str, Any]] = []

    canonical_processed = processed_root / "canonical"
    if selected_model_ablations:
        prepare_processed_data(args.base_config, canonical_processed)

    emoji_processed_dirs = {
        "emoji_remove": processed_root / "emoji_remove",
        "emoji_keep": processed_root / "emoji_keep",
    }
    for name, cfg_path in selected_preprocessing_ablations:
        prepare_processed_data(cfg_path, emoji_processed_dirs[name])

    for name, cfg_path in selected_model_ablations:
        rows.append(run_single(cfg_path, name, "model", canonical_processed, root_dir))

    for name, cfg_path in selected_preprocessing_ablations:
        rows.append(run_single(cfg_path, name, "preprocess", emoji_processed_dirs[name], root_dir))

    summary = {
        "base_config": args.base_config,
        "canonical_processed_dir": str(canonical_processed) if selected_model_ablations else None,
        "runs": rows,
    }
    save_json(summary, root_dir / "ablation_summary.json")
    write_csv(rows, root_dir / "ablation_summary.csv")


if __name__ == "__main__":
    main()
