from __future__ import annotations

import argparse
from pathlib import Path

from src.baselines.run_spacy import run_spacy
from src.baselines.train_bert import train_bert
from src.data.download import download_dataset
from src.data.preprocess import preprocess
from src.eval.evaluate import evaluate
from src.train.crossval_gatte import crossval_gatte
from src.train.train_gatte import train_gatte
from src.train.train_mgatte import train_mgatte
from src.utils.config import load_config, save_config
from src.utils.run import create_run_dir, save_env_info
from src.viz.make_figures import make_figures


def run_all(cfg_path: str, run_dir: str | None = None) -> None:
    cfg = load_config(cfg_path)
    if run_dir:
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        run_dir = create_run_dir(cfg["paths"]["outputs_dir"], tag="full")
    save_env_info(run_dir)
    save_config(cfg, run_dir / "config.yaml")

    download_dataset(cfg["dataset"]["dataverse_doi"], Path(cfg["paths"]["raw_dir"]))
    preprocess(cfg)
    train_gatte(cfg, run_dir=run_dir)
    train_mgatte(cfg, run_dir=run_dir)
    train_bert(cfg, run_dir=run_dir)
    run_spacy(cfg, run_dir=run_dir)
    evaluate(run_dir)
    make_figures(run_dir, Path(cfg["paths"]["processed_dir"]))
    # Run k-fold at the end (requested)
    crossval_gatte(cfg, run_dir=run_dir)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run_dir", default="")
    args = ap.parse_args()
    run_all(args.config, run_dir=args.run_dir or None)


if __name__ == "__main__":
    main()
