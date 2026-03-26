# GAttE Reproduction Pipeline

This repository contains a reproducible training and evaluation pipeline for the GAttE and M-GAttE location prediction models, together with BERT, spaCy, cross-validation, plotting, and ablation scripts.

The codebase is organized so you can either:

- run the full pipeline end to end
- run each stage manually
- run individual ablations against a shared processed dataset

## 1. What This Repository Runs

Main entry points:

- `python -m src.pipeline.run_all --config <config>`
  Runs download, preprocessing, GAttE, M-GAttE, BERT, spaCy, evaluation tables, figures, and GAttE cross-validation.
- `python -m src.data.preprocess --config <config>`
  Builds processed train/val/test artifacts.
- `python -m src.train.train_gatte --config <config>`
  Trains only GAttE.
- `python -m src.train.train_mgatte --config <config>`
  Trains only M-GAttE.
- `python -m src.train.crossval_gatte --config <config>`
  Runs GAttE k-fold cross-validation.
- `python -m src.baselines.train_bert --config <config>`
  Trains the BERT baseline.
- `python -m src.baselines.run_spacy --config <config>`
  Runs the spaCy baseline.
- `python -m src.train.run_ablation_suite --base_config <config> --output_dir <dir>`
  Runs ablation experiments.

## 2. Project Layout

- `configs/`
  YAML experiment configs.
- `configs/ablations/`
  Ablation configs for embeddings, attention, deconvolution, emoji, and self-attention variants.
- `data/raw/`
  Downloaded source datasets from Dataverse.
- `data/processed/`
  Preprocessed train/val/test files and metadata.
- `outputs/runs/`
  Timestamped full runs.
- `outputs/ablations/`
  Ablation results.
- `src/data/`
  Dataset download, geocoding, preprocessing.
- `src/models/`
  GAttE and M-GAttE model definitions.
- `src/train/`
  Training and ablation runners.
- `src/baselines/`
  BERT and spaCy baselines.
- `src/eval/`
  Metrics and summary table generation.
- `src/viz/`
  Figure generation.

## 3. Requirements

Install Python 3.11 or 3.12, then install the project dependencies:

```powershell
pip install -r requirements.txt
```

Install the spaCy transformer model if you plan to run the spaCy baseline:

```powershell
python -m spacy download en_core_web_trf
```

## 4. Optional GeoNames Setup

GeoNames is only needed when the pipeline must reverse geocode coordinates instead of using an already prepared `geo_data_full.csv`.

Set your GeoNames username in PowerShell:

```powershell
$env:GEONAMES_USERNAME="your_geonames_username"
```

You may also set a custom rate limit:

```powershell
$env:GEONAMES_RATE_LIMIT="1.0"
```

Important:

- the current configs prefer `data/processed/geo_data_full.csv` when it exists
- if that file is present, the main paper workflow does not need live reverse geocoding
- if you disable `prefer_preprocessed_geo`, GeoNames becomes part of preprocessing

## 5. Config Files

The main configs are:

- `configs/paper.yaml`
  Main research configuration.
- `configs/quick.yaml`
  Lightweight sanity-check configuration.
- `configs/colab.yaml`
  Colab-style configuration.

Current important behavior in `configs/paper.yaml`:

- it prefers `data/processed/geo_data_full.csv` as the preprocessing source
- it filters classes using the configured frequency bounds
- it currently keeps `66` classes via `target_class_count: 66`
- it uses the paper-style train/val/test split and the current evaluation thresholds

## 6. Step-by-Step Execution

### Step 1: Download the raw dataset

From the repository root:

```powershell
python -m src.data.download --config configs/paper.yaml
```

This downloads the Dataverse dataset into `data/raw/`.

If `datasetA.csv` is present, preprocessing will prefer it over other raw files.

### Step 2: Prepare preprocessing input

You have two possible paths:

1. Use the prepared geographic file:
   Keep `dataset.prefer_preprocessed_geo: true` and make sure `data/processed/geo_data_full.csv` exists.
2. Use raw data and reverse geocoding:
   Set `dataset.prefer_preprocessed_geo: false` in the config and provide `GEONAMES_USERNAME`.

### Step 3: Run preprocessing

```powershell
python -m src.data.preprocess --config configs/paper.yaml
```

This creates:

- `data/processed/train.npz`
- `data/processed/val.npz`
- `data/processed/test.npz`
- `data/processed/meta.json`
- optionally `use_train.npy`, `use_val.npy`, `use_test.npy` when USE precomputation is enabled

Check the processed metadata:

```powershell
Get-Content data\\processed\\meta.json
```

### Step 4: Train GAttE only

```powershell
python -m src.train.train_gatte --config configs/paper.yaml
```

This writes a timestamped run under `outputs/runs/<timestamp>_gatte/` when no run directory is supplied.

Artifacts include:

- `config.yaml`
- `gatte/model.keras`
- `gatte/history.json`
- `gatte/metrics.json`
- `gatte/preds.npz`
- `gatte/distances.npz`

### Step 5: Train M-GAttE only

```powershell
python -m src.train.train_mgatte --config configs/paper.yaml
```

### Step 6: Run the BERT baseline

```powershell
python -m src.baselines.train_bert --config configs/paper.yaml
```

### Step 7: Run the spaCy baseline

```powershell
python -m src.baselines.run_spacy --config configs/paper.yaml
```

### Step 8: Run GAttE cross-validation

```powershell
python -m src.train.crossval_gatte --config configs/paper.yaml
```

Cross-validation results are written under:

- `outputs/runs/.../crossval/`
- or a dedicated `outputs/runs/..._crossval/` run if called standalone

### Step 9: Build summary tables from an existing run

```powershell
python -m src.eval.evaluate --config configs/paper.yaml --run_dir outputs/runs/<your_run_dir>
```

This writes:

- `tables/performance_summary.csv`

### Step 10: Generate figures for an existing run

If you already ran the full pipeline, figures are generated automatically. If you want to rerun plotting manually from Python code, use the logic in `src/viz/make_figures.py`.

## 7. Run the Full Pipeline in One Command

For the full end-to-end pipeline:

```powershell
python -m src.pipeline.run_all --config configs/paper.yaml
```

This performs the following sequence:

1. download dataset
2. preprocess data
3. train GAttE
4. train M-GAttE
5. train BERT
6. run spaCy
7. build evaluation table
8. generate figures
9. run GAttE cross-validation

Use the quick config for a lightweight test:

```powershell
python -m src.pipeline.run_all --config configs/quick.yaml
```

Use the Colab-style config:

```powershell
python -m src.pipeline.run_all --config configs/colab.yaml
```

## 8. Run Ablation Studies

### Option A: Run the whole ablation suite

```powershell
python -m src.train.run_ablation_suite --base_config configs/paper.yaml --output_dir outputs/ablations
```

### Option B: Run selected ablations only

```powershell
python -m src.train.run_ablation_suite --base_config configs/paper.yaml --output_dir outputs/ablations --names embeddings_full embeddings_use_only deconv_off attention_off attention_self
```

### Available model ablations

- `embeddings_full`
- `embeddings_use_only`
- `embeddings_word_only`
- `embeddings_char_only`
- `embeddings_use_char`
- `deconv_off`
- `attention_off`
- `attention_simple`
- `attention_self`

### Available preprocessing ablations

- `emoji_remove`
- `emoji_keep`

How the ablation runner works:

- it builds one canonical processed dataset from `--base_config`
- it reuses that dataset for model-only ablations
- it rebuilds preprocessing only for preprocessing-changing ablations such as emoji experiments
- it writes results to `outputs/ablations/<ablation_name>/`

Summary files are written to:

- `outputs/ablations/ablation_summary.json`
- `outputs/ablations/ablation_summary.csv`

## 9. Where to Look After a Run

### Processed data

- `data/processed/meta.json`
- `data/processed/train.npz`
- `data/processed/val.npz`
- `data/processed/test.npz`

### Full run outputs

- `outputs/runs/<timestamp>_full/`
- `outputs/runs/<timestamp>_gatte/`
- `outputs/runs/<timestamp>_mgatte/`
- `outputs/runs/<timestamp>_bert/`
- `outputs/runs/<timestamp>_spacy/`

### Per-model metrics

- `gatte/metrics.json`
- `mgatte/metrics.json`
- `bert/metrics.json`
- `spacy/metrics.json`

### Cross-validation

- `crossval/crossval_metrics.json`

### Tables and figures

- `tables/performance_summary.csv`
- files generated under the run's figure directory

## 10. Recommended Execution Order

If you want the safest manual workflow, use this order:

1. `pip install -r requirements.txt`
2. `python -m spacy download en_core_web_trf`
3. `python -m src.data.download --config configs/paper.yaml`
4. verify whether `data/processed/geo_data_full.csv` is present
5. `python -m src.data.preprocess --config configs/paper.yaml`
6. inspect `data/processed/meta.json`
7. `python -m src.train.train_gatte --config configs/paper.yaml`
8. `python -m src.train.train_mgatte --config configs/paper.yaml`
9. `python -m src.baselines.train_bert --config configs/paper.yaml`
10. `python -m src.baselines.run_spacy --config configs/paper.yaml`
11. `python -m src.train.crossval_gatte --config configs/paper.yaml`
12. `python -m src.eval.evaluate --config configs/paper.yaml --run_dir outputs/runs/<run_dir>`

If you want one command instead, use:

```powershell
python -m src.pipeline.run_all --config configs/paper.yaml
```

## 11. Notes and Practical Warnings

- Full runs are expensive. `run_all` includes BERT, spaCy, plotting, and cross-validation at the end.
- `quick.yaml` is the right starting point when you are only checking that the code runs.
- The spaCy baseline requires the transformer model to be installed separately.
- Dataverse download requires internet access.
- GeoNames requests are cached in `data/cache/geocode.sqlite`.
- If you rerun the same output directory through the ablation runner, files in that ablation folder will be overwritten because `resume_if_exists` is disabled inside the suite runner.
- Some configs intentionally use precomputed USE vectors during training.

## 12. Minimal Command Examples

Only preprocess:

```powershell
python -m src.data.preprocess --config configs/paper.yaml
```

Only GAttE:

```powershell
python -m src.train.train_gatte --config configs/paper.yaml
```

Only one ablation:

```powershell
python -m src.train.run_ablation_suite --base_config configs/paper.yaml --output_dir outputs/ablations --names attention_self
```

Full pipeline:

```powershell
python -m src.pipeline.run_all --config configs/paper.yaml
```
