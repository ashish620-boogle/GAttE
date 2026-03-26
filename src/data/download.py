from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path
from typing import Dict, List

import requests
from tqdm import tqdm

from src.utils.config import load_config


def dataverse_api_url(doi: str) -> str:
    return f"https://dataverse.harvard.edu/api/datasets/:persistentId/?persistentId={doi}"


def download_dataset(doi: str, raw_dir: Path) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    meta_url = dataverse_api_url(doi)
    resp = requests.get(meta_url, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    files = data.get("data", {}).get("latestVersion", {}).get("files", [])
    if not files:
        raise RuntimeError("No files found in Dataverse response")

    for f in tqdm(files, desc="Downloading files"):
        data_file = f.get("dataFile", {})
        file_id = data_file.get("id")
        filename = f.get("label") or data_file.get("filename")
        if not file_id or not filename:
            continue
        out_path = raw_dir / filename
        if out_path.exists():
            continue
        url = f"https://dataverse.harvard.edu/api/access/datafile/{file_id}"
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(out_path, "wb") as fh:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        fh.write(chunk)

        if out_path.suffix.lower() == ".zip":
            with zipfile.ZipFile(out_path, "r") as zf:
                zf.extractall(raw_dir)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    doi = cfg["dataset"]["dataverse_doi"]
    raw_dir = Path(cfg["paths"]["raw_dir"])

    download_dataset(doi, raw_dir)


if __name__ == "__main__":
    main()