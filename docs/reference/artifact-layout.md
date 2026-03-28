# Artifact Layout Reference

This document describes where pipeline outputs are written and how to identify key artifacts for debugging and reproducibility.

## Root folders

- Raw market data: `data/raw/`
- Features: `data/features/`
- Labels: `data/labels/`
- Training model outputs: `models/`
- Backtest outputs: `artifacts/backtests/`
- Dataset registry manifests: `data/registry/`
- Run manifests: `runs/`

## 1) Raw data layout

### Klines

`data/raw/binance/<SYMBOL>/klines/*.csv`

### Aggtrades

`data/raw/binance/<SYMBOL>/aggtrades/*.csv`

Naming normally includes UTC ranges in the filename:

`...__YYYYMMDD_HHMMSS__YYYYMMDD_HHMMSS.csv`

Legacy names may exist (for example `1m.csv`), and the pipeline still attempts to handle them.

## 2) Features artifacts

### Folder pattern

`data/features/<SYMBOL>/<FEATURE_DATASET_ID>/`

### Key files

- `features.csv` (main output)

### Naming notes

- Feature CSV filename is always `features.csv`.
- Uniqueness is encoded by the parent `<FEATURE_DATASET_ID>` directory (hash built from source + params), not by the CSV filename itself.
- A `latest` tag file can be written at: `data/features/<SYMBOL>/_tags/latest.txt`.

## 3) Labels artifacts

### Folder pattern

`data/labels/<SYMBOL>/<LABEL_DATASET_ID>/`

### Key files

- `labels.csv` (main output)

### Labels/features naming relationship

- Label CSV filename is always `labels.csv`.
- The label dataset references one feature dataset as its input source.
- As with features, identity/version is in `<LABEL_DATASET_ID>` directory names.
- A `latest` tag file can be written at: `data/labels/<SYMBOL>/_tags/latest.txt`.

## 4) Baseline training artifacts

### Folder pattern

`models/baseline/<SYMBOL>/<RUN_ID>/`

### Key files

- `model.pt` — model checkpoint used for inference/backtest.
- `report.json` — train/val/test metrics and confusion matrix.
- `history.json` — per-epoch training history.

### Optional tags and manifests

- `models/baseline/<SYMBOL>/_tags/latest.txt` (when run tag is used)
- `runs/<RUN_ID>.json` (global run manifest)

## 5) Backtest artifacts

### Folder pattern

`artifacts/backtests/<SYMBOL>/<RUN_ID>/`

### Key files

- `report.json` — backtest summary metrics and thresholded classification report.
- `trades.csv` — executed trades (one row per simulated trade).
- `predictions.csv` — per-row probabilities and threshold decisions.

### Optional tags and manifests

- `artifacts/backtests/<SYMBOL>/_tags/latest.txt` (when run tag is used)
- `runs/<RUN_ID>.json` (global run manifest)

## 6) Registry manifests

### Dataset manifests

- Features manifests: `data/registry/features/<FEATURE_DATASET_ID>.json`
- Labels manifests: `data/registry/labels/<LABEL_DATASET_ID>.json`

These manifests store artifact path, source ranges, schema summaries, and creation metadata.

### Run manifests

- `runs/<RUN_ID>.json`

These manifests store run type, git SHA, core metrics, hyperparameters, and output artifact paths.

## 7) Quick path examples

For `BTCUSDT`:

- Features: `data/features/BTCUSDT/<dataset_id>/features.csv`
- Labels: `data/labels/BTCUSDT/<dataset_id>/labels.csv`
- Checkpoint: `models/baseline/BTCUSDT/<run_id>/model.pt`
- Train report: `models/baseline/BTCUSDT/<run_id>/report.json`
- History: `models/baseline/BTCUSDT/<run_id>/history.json`
- Backtest report: `artifacts/backtests/BTCUSDT/<run_id>/report.json`
- Backtest trades: `artifacts/backtests/BTCUSDT/<run_id>/trades.csv`
- Backtest predictions: `artifacts/backtests/BTCUSDT/<run_id>/predictions.csv`
