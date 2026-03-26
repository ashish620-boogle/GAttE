from __future__ import annotations

import argparse
import os
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.data.geocode import GeoNamesClient, get_geonames_username
from src.utils.config import load_config
from src.utils.io import save_json, save_npz


TEXT_CANDIDATES = ["text", "message", "tweet", "content", "post", "body"]
LAT_CANDIDATES = ["latitude", "lat", "latittude"]
LON_CANDIDATES = ["longitude", "lon", "lng", "long"]
PLACE_CANDIDATES = ["place_name", "placename", "place", "location", "city"]
ID_CANDIDATES = ["id", "tweet_id", "post_id"]
DATE_CANDIDATES = ["date", "created_at", "time", "timestamp"]
SOURCE_CANDIDATES = ["source", "platform"]
GEOM_CANDIDATES = ["geom", "geometry"]


def find_dataset_file(raw_dir: Path) -> Path:
    # Prefer datasetA if present (paper uses datasetA)
    for name in ["datasetA.csv", "datasetA.tsv", "datasetA.txt"]:
        candidate = raw_dir / name
        if candidate.exists():
            return candidate
    for ext in [".csv", ".tsv", ".txt"]:
        files = sorted(raw_dir.glob(f"*{ext}"))
        if files:
            return files[0]
    raise FileNotFoundError("No dataset file found in data/raw")


def load_dataset(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".tsv":
        return pd.read_csv(path, sep="\t")
    if path.suffix.lower() == ".txt":
        return pd.read_csv(path, sep="\t")
    return pd.read_csv(path)


def _normalize_col(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.strip().lower())


def _pick_col(df: pd.DataFrame, candidates: List[str]) -> str:
    cols = {_normalize_col(c): c for c in df.columns}
    for c in candidates:
        key = _normalize_col(c)
        if key in cols:
            return cols[key]
    return ""


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    text_col = _pick_col(df, TEXT_CANDIDATES)
    lat_col = _pick_col(df, LAT_CANDIDATES)
    lon_col = _pick_col(df, LON_CANDIDATES)
    if not text_col or not lat_col or not lon_col:
        raise ValueError(f"Missing required columns. Found text={text_col}, lat={lat_col}, lon={lon_col}")

    id_col = _pick_col(df, ID_CANDIDATES)
    date_col = _pick_col(df, DATE_CANDIDATES)
    source_col = _pick_col(df, SOURCE_CANDIDATES)
    geom_col = _pick_col(df, GEOM_CANDIDATES)
    place_col = _pick_col(df, PLACE_CANDIDATES)

    out = pd.DataFrame()
    out["id"] = df[id_col] if id_col else np.arange(len(df))
    out["date"] = df[date_col] if date_col else ""
    out["source"] = df[source_col] if source_col else ""
    out["geom"] = df[geom_col] if geom_col else ""
    out["longitude"] = df[lon_col]
    out["latitude"] = df[lat_col]
    out["text"] = df[text_col].astype(str)
    if place_col:
        out["place_name"] = df[place_col].astype(str)
    return out


def clean_text(text: str, remove_non_alnum: bool, keep_emojis: bool, lowercase: bool) -> str:
    if lowercase:
        text = text.lower()
    if remove_non_alnum:
        if keep_emojis:
            # keep non-alnum, but normalize spaces
            text = re.sub(r"\s+", " ", text)
        else:
            # The original study preserved mentions and hashtags while removing other symbols.
            text = re.sub(r"[^A-Za-z0-9@#\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_label(label: str | None) -> str | None:
    if label is None:
        return None
    cleaned = clean_text(str(label), remove_non_alnum=True, keep_emojis=False, lowercase=False)
    if not cleaned or cleaned.lower() in {"none", "nan"}:
        return None
    return cleaned


def parse_place_name_like_notebook(val: str | float | None) -> str | None:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    s = str(val)

    def _comma_idx(text: str) -> int | None:
        for idx, ch in enumerate(text):
            if ch == ",":
                return idx - 1
        return None

    def _first_alnum_idx(text: str) -> int:
        for idx, ch in enumerate(text):
            if ch.isalnum():
                return idx
        return 0

    start = _first_alnum_idx(s)
    end = _comma_idx(s)
    if end is None or end <= start:
        return normalize_label(s)
    label = s[start:end]
    label = clean_text(label, remove_non_alnum=True, keep_emojis=False, lowercase=False)
    if not label or label == "Non":
        return None
    return label


def build_vocab(tokens: List[List[str]], min_freq: int = 1) -> Dict[str, int]:
    counter = Counter()
    for seq in tokens:
        counter.update(seq)
    vocab = {"<PAD>": 0, "<OOV>": 1}
    for token, freq in counter.most_common():
        if freq < min_freq:
            continue
        vocab[token] = len(vocab)
    return vocab


def select_labels_by_policy(counts: pd.Series, cfg: Dict) -> List[str]:
    filtered = counts[(counts > cfg["preprocess"]["min_class_count"]) & (counts < cfg["preprocess"]["max_class_count"])]
    if filtered.empty:
        raise ValueError("No labels satisfy the configured class-count bounds.")

    target_count = cfg["preprocess"].get("target_class_count")
    if target_count is None:
        return filtered.index.tolist()

    target_count = int(target_count)
    if target_count <= 0:
        raise ValueError("target_class_count must be positive when provided.")
    if len(filtered) < target_count:
        raise ValueError(
            f"Requested target_class_count={target_count}, but only {len(filtered)} labels satisfy the class-count bounds."
        )

    selection = cfg["preprocess"].get("class_selection", "most_frequent")
    if selection != "most_frequent":
        raise ValueError(f"Unsupported class_selection policy: {selection}")

    selected = (
        filtered.rename("count")
        .rename_axis("label")
        .reset_index()
        .sort_values(["count", "label"], ascending=[False, True], kind="mergesort")
        .head(target_count)
    )
    return selected["label"].tolist()


def encode_words(text: str, vocab: Dict[str, int], max_words: int) -> List[int]:
    words = text.split()
    ids = [vocab.get(w, vocab["<OOV>"]) for w in words[:max_words]]
    if len(ids) < max_words:
        ids.extend([vocab["<PAD>"]] * (max_words - len(ids)))
    return ids


def encode_chars(text: str, vocab: Dict[str, int], max_chars: int) -> List[int]:
    chars = list(text.replace(" ", ""))
    ids = [vocab.get(c, vocab["<OOV>"]) for c in chars[:max_chars]]
    if len(ids) < max_chars:
        ids.extend([vocab["<PAD>"]] * (max_chars - len(ids)))
    return ids


def preprocess(cfg: Dict) -> None:
    raw_dir = Path(cfg["paths"]["raw_dir"])
    processed_dir = Path(cfg["paths"]["processed_dir"])
    cache_dir = Path(cfg["paths"]["cache_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    pre_geo = Path(cfg["dataset"].get("preprocessed_geo_file", ""))
    prefer_geo = cfg["dataset"].get("prefer_preprocessed_geo", False)
    if prefer_geo and pre_geo.exists():
        df = load_dataset(pre_geo)
    else:
        dataset_path = find_dataset_file(raw_dir)
        df = load_dataset(dataset_path)
    df = normalize_columns(df)

    # Match the notebook preprocessing order:
    # filter short texts first, then strip URLs, without deduplicating samples.
    df = df[df["text"].astype(str).str.len() > cfg["preprocess"]["min_text_len"]].reset_index(drop=True)
    df["text"] = df["text"].astype(str).apply(lambda t: re.sub(r"http\S+", "", t, flags=re.MULTILINE))

    # label from preprocessed file if present, otherwise reverse geocode
    if "place_name" in df.columns and df["place_name"].notna().any():
        df["label"] = df["place_name"].apply(parse_place_name_like_notebook)
        if cfg["geocode"].get("fill_missing", False):
            missing = df["label"].isna() | (df["label"].astype(str).str.len() == 0)
            if missing.any():
                username = get_geonames_username(cfg["geocode"]["username_env"])
                rate_limit = float(os.getenv(cfg["geocode"]["rate_limit_env"], cfg["geocode"]["rate_limit"]))
                round_precision = cfg["geocode"]["round"]
                client = GeoNamesClient(
                    username=username,
                    cache_path=cache_dir / "geocode.sqlite",
                    rate_limit=rate_limit,
                    round_precision=round_precision,
                    country_code=cfg["dataset"]["country_code"],
                    verify_ssl=cfg["geocode"].get("verify_ssl", True),
                    allow_insecure_fallback=cfg["geocode"].get("allow_insecure_fallback", False),
                )
                coords = df.loc[missing, ["latitude", "longitude"]].drop_duplicates().values
                coord_to_place: Dict[Tuple[float, float], Tuple[str, str]] = {}
                for lat, lon in tqdm(coords, desc="Reverse geocoding (missing labels)"):
                    res = client.reverse_geocode(lat, lon)
                    if res is None or not res.name:
                        continue
                    coord_to_place[(client._round(lat), client._round(lon))] = (res.name, res.admin1)
                fill_labels = []
                for lat, lon in df.loc[missing, ["latitude", "longitude"]].values:
                    key = (client._round(lat), client._round(lon))
                    if key in coord_to_place:
                        name, admin = coord_to_place[key]
                        if cfg["geocode"].get("disambiguate_by_admin", False) and admin:
                            fill_labels.append(f"{name}, {admin}")
                        else:
                            fill_labels.append(name)
                    else:
                        fill_labels.append(None)
                df.loc[missing, "label"] = fill_labels
        df["label"] = df["label"].apply(normalize_label)
        df = df.dropna(subset=["label"]).reset_index(drop=True)
    else:
        username = get_geonames_username(cfg["geocode"]["username_env"])
        rate_limit = float(os.getenv(cfg["geocode"]["rate_limit_env"], cfg["geocode"]["rate_limit"]))
        round_precision = cfg["geocode"]["round"]
        client = GeoNamesClient(
            username=username,
            cache_path=cache_dir / "geocode.sqlite",
            rate_limit=rate_limit,
            round_precision=round_precision,
            country_code=cfg["dataset"]["country_code"],
            verify_ssl=cfg["geocode"].get("verify_ssl", True),
            allow_insecure_fallback=cfg["geocode"].get("allow_insecure_fallback", False),
        )

        unique_coords = df[["latitude", "longitude"]].drop_duplicates().values
        coord_to_place: Dict[Tuple[float, float], Tuple[str, str]] = {}
        for lat, lon in tqdm(unique_coords, desc="Reverse geocoding"):
            res = client.reverse_geocode(lat, lon)
            if res is None or not res.name:
                continue
            coord_to_place[(client._round(lat), client._round(lon))] = (res.name, res.admin1)

        labels = []
        for lat, lon in df[["latitude", "longitude"]].values:
            key = (client._round(lat), client._round(lon))
            label = None
            if key in coord_to_place:
                name, admin = coord_to_place[key]
                if cfg["geocode"].get("disambiguate_by_admin", False) and admin:
                    label = f"{name}, {admin}"
                else:
                    label = name
            labels.append(label)
        df["label"] = [normalize_label(v) for v in labels]
        df = df.dropna(subset=["label"]).reset_index(drop=True)

    # filter classes by count
    counts = df["label"].value_counts()
    allowed = select_labels_by_policy(counts, cfg)
    df = df[df["label"].isin(allowed)].reset_index(drop=True)

    # clean text
    df["text"] = df["text"].apply(
        lambda t: clean_text(
            t,
            remove_non_alnum=cfg["preprocess"]["remove_non_alnum"],
            keep_emojis=cfg["preprocess"]["keep_emojis"],
            lowercase=cfg["preprocess"]["lowercase"],
        )
    )
    df = df[df["text"].str.len() > cfg["preprocess"]["min_text_len"]].reset_index(drop=True)
    vectorizer_lowercase = cfg["preprocess"].get("vectorizer_lowercase", True)
    df["vector_text"] = df["text"].str.lower() if vectorizer_lowercase else df["text"]

    # split
    train_size = cfg["split"]["train_size"]
    val_size = cfg["split"]["val_size"]
    test_size = cfg["split"]["test_size"]
    temp_size = val_size + test_size
    stratify_labels = df["label"] if cfg["split"].get("stratify", False) else None

    train_df, temp_df = train_test_split(
        df,
        test_size=temp_size,
        stratify=stratify_labels,
        random_state=cfg["split"]["random_state"],
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=test_size / temp_size,
        stratify=temp_df["label"] if cfg["split"].get("stratify", False) else None,
        random_state=cfg["split"]["random_state"],
    )

    # build vocabs
    vocab_source = train_df if cfg["preprocess"]["vocab_from"] == "train" else df
    word_tokens = [t.split() for t in vocab_source["vector_text"].tolist()]
    char_tokens = [[c for c in t.replace(" ", "")] for t in vocab_source["vector_text"].tolist()]
    word_vocab = build_vocab(word_tokens)
    char_vocab = build_vocab(char_tokens)
    max_char_len = int(
        np.percentile(
            train_df["vector_text"].astype(str).str.len(),
            cfg["preprocess"].get("char_length_percentile", 98),
        )
    )
    max_char_len = max(max_char_len, 1)

    label_list = df["label"].drop_duplicates().tolist()
    label_to_idx = {l: i for i, l in enumerate(label_list)}

    def encode_split(split_df: pd.DataFrame, split_name: str) -> None:
        max_words = cfg["preprocess"]["max_words"]
        texts = split_df["text"].tolist()
        vector_texts = split_df["vector_text"].tolist()
        words = np.array([encode_words(t, word_vocab, max_words) for t in vector_texts], dtype=np.int32)
        chars = np.array([encode_chars(t, char_vocab, max_char_len) for t in vector_texts], dtype=np.int32)
        labels = np.array([label_to_idx[l] for l in split_df["label"].tolist()], dtype=np.int32)
        lats = split_df["latitude"].astype(float).values
        lons = split_df["longitude"].astype(float).values
        save_npz(
            processed_dir / f"{split_name}.npz",
            text=np.array(texts, dtype=object),
            words=words,
            chars=chars,
            labels=labels,
            latitude=lats,
            longitude=lons,
        )

    encode_split(train_df, "train")
    encode_split(val_df, "val")
    encode_split(test_df, "test")

    # optional precompute USE embeddings
    if cfg["train"].get("use_precomputed", False):
        try:
            import tensorflow_hub as hub

            use_model = hub.load(cfg["model"]["use_url"])
            for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
                texts = split_df["text"].tolist()
                vecs = use_model(texts).numpy()
                np.save(processed_dir / f"use_{split_name}.npy", vecs)
        except Exception as exc:
            print(f"USE precompute failed: {exc}")

    # label centroids
    centroids = (
        train_df.groupby("label")[["latitude", "longitude"]].mean().reset_index().to_dict(orient="records")
    )

    meta = {
        "label_list": label_list,
        "label_to_idx": label_to_idx,
        "word_vocab": word_vocab,
        "char_vocab": char_vocab,
        "max_char_len": max_char_len,
        "centroids": centroids,
        "class_counts": {label: int(counts[label]) for label in label_list},
        "num_samples": int(len(df)),
        "preprocess_config": {
            "min_text_len": cfg["preprocess"]["min_text_len"],
            "min_class_count": cfg["preprocess"]["min_class_count"],
            "max_class_count": cfg["preprocess"]["max_class_count"],
            "target_class_count": cfg["preprocess"].get("target_class_count"),
            "class_selection": cfg["preprocess"].get("class_selection", "most_frequent"),
            "keep_emojis": cfg["preprocess"]["keep_emojis"],
            "remove_non_alnum": cfg["preprocess"]["remove_non_alnum"],
        },
    }
    save_json(meta, processed_dir / "meta.json")



def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    preprocess(cfg)


if __name__ == "__main__":
    main()
