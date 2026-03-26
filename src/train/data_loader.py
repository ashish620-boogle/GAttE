from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np

from src.utils.io import load_json, load_npz


def load_split(processed_dir: str | Path, split: str) -> Dict[str, np.ndarray]:
    path = Path(processed_dir) / f"{split}.npz"
    data = load_npz(path)
    use_path = Path(processed_dir) / f"use_{split}.npy"
    if use_path.exists():
        data["use_vec"] = np.load(use_path)
    return data


def load_meta(processed_dir: str | Path) -> Dict:
    return load_json(Path(processed_dir) / "meta.json")
