from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

import yaml


def default_config() -> Dict[str, Any]:
    return {
        "seed": 42,
        "paths": {
            "data_dir": "data",
            "raw_dir": "data/raw",
            "processed_dir": "data/processed",
            "cache_dir": "data/cache",
            "outputs_dir": "outputs",
        },
        "dataset": {
            "dataverse_doi": "doi:10.7910/DVN/LOTEGM",
            "country_code": "US",
            "preprocessed_geo_file": "data/processed/geo_data_full.csv",
            "prefer_preprocessed_geo": True,
        },
        "geocode": {
            "username_env": "GEONAMES_USERNAME",
            "rate_limit_env": "GEONAMES_RATE_LIMIT",
            "rate_limit": 1.0,
            "round": 4,
            "disambiguate_by_admin": False,
            "verify_ssl": False,
            "allow_insecure_fallback": True,
            "fill_missing": False,
        },
        "preprocess": {
            "min_text_len": 3,
            "min_class_count": 80,
            "max_class_count": 500,
            "target_class_count": None,
            "class_selection": "most_frequent",
            "remove_non_alnum": True,
            "keep_emojis": False,
            "lowercase": False,
            "vectorizer_lowercase": True,
            "vocab_from": "train",
            "max_words": 32,
            "char_length_percentile": 98,
            "max_chars_per_word": 32,
        },
        "split": {
            "train_size": 0.6,
            "val_size": 0.2,
            "test_size": 0.2,
            "random_state": 42,
            "stratify": False,
        },
        "model": {
            "use_url": "https://tfhub.dev/google/universal-sentence-encoder/4",
            "use_sentence_embedding": True,
            "use_word_embedding": True,
            "use_char_embedding": True,
            "attention_mode": "multihead",
            "no_attention_mode": "concat_qv",
            "variant": "paper",
            "word_dim": 300,
            "char_dim": 50,
            "dense_units": 640,
            "timesteps": 32,
            "deconv_filters": [1024, 256],
            "q_filters": 256,
            "num_heads": 1024,
            "key_dim": 2,
            "value_dim": 2,
            "dropout_rates": [0.3, 0.6],
            "l1": 0.001,
            "l2": 0.000001,
            "use_deconv": True,
            "use_attention": True,
            "use_activity_regularizer": False,
            "activity_l1": 0.001,
            "activity_l2": 0.000001,
        },
        "train": {
            "batch_size": 256,
            "epochs": 5,
            "lr": 0.001,
            "lr_patience": 1,
            "kfolds": 20,
            "use_precomputed": True,
            "from_logits_loss": False,
            "resume_if_exists": True,
        },
        "mgatte": {
            "regression_loss_weight": 1.0,
        },
        "bert": {
            "model_name": "bert-base-uncased",
            "max_len": 128,
            "batch_size": 16,
            "epochs": 3,
            "lr": 2e-5,
        },
        "spacy": {
            "model": "en_core_web_trf",
            "labels": ["LOC", "FAC", "ORG", "GPE"],
        },
        "eval": {
            "k_values": [1, 5, 10, 15, 20, 25, 30, 35],
        },
        "viz": {
            "dpi": 150,
        },
    }


def deep_merge(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def _load_yaml_with_extends(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    extends = cfg.pop("extends", None)
    if not extends:
        return cfg

    base_path = Path(extends)
    if not base_path.is_absolute():
        base_path = (path.parent / base_path).resolve()

    base_cfg = _load_yaml_with_extends(base_path)
    return deep_merge(base_cfg, cfg)


def load_config(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    user_cfg = _load_yaml_with_extends(path.resolve())
    cfg = deep_merge(deepcopy(default_config()), user_cfg)
    cfg["config_path"] = str(path)
    return cfg


def save_config(cfg: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
