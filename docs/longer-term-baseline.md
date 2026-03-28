# Longer-Term Baseline Training Guide

This guide walks a new contributor through a first baseline run with lower memory pressure and clearer validation gates.

## 1) Choose a practical first configuration

For most first runs:

- Symbol: `BTCUSDT`
- Data interval (klines): `1m`
- Feature timeframe: `5m`
- Label horizon: `12` bars (about 60 minutes at `5m`)
- TP / SL: `0.35% / 0.25%`

Why this works:

- Fewer rows than second-level datasets.
- Enough market structure for short intraday swings.
- Runtime and memory are usually manageable on a laptop.

## 2) Canonical end-to-end flow

Start menu:

```bash
uv run trader menu
```

Then run these options in order:

1. **Initialize project config**
   - Set timeframe to `5m`.
2. **Fetch klines**
   - Interval: `1m`.
3. **Fetch aggtrades**
   - Same date range.
4. **Build features**
   - Timeframe: `5m`.
   - Include 1m context: `y`.
5. **Build labels**
   - Horizon in bars: `12`.
   - TP: `0.35`.
   - SL: `0.25`.
6. **Train baseline**
   - Epochs: `20`.
   - Batch size: `512`.
   - LR: `0.001`.
   - Lookback window: `12`.
   - Hidden dim: `512`.
   - MLP depth: `4`.
   - Dropout: `0.15`.
7. **Backtest baseline**
   - Eval split: `test`.
   - Long/short thresholds: `0.80 / 0.80`.
   - Margin: `0.05`.

## 3) Parameter reference (baseline-relevant)

| Parameter | Stage | Meaning | Typical starting value | Notes |
|---|---|---|---|---|
| `timeframe` | Feature build | Output bar duration for feature rows. | `5m` | Larger timeframe reduces row count and noise, but can reduce signal frequency. |
| `horizon_steps` | Label build | Number of **future bars** scanned for TP/SL and horizon outcomes. | `12` | Effective horizon time = `horizon_steps * timeframe_s`. |
| `lookback_window` | Training/backtest | Number of past rows concatenated per sample. | `12` | Must fit data size: each split must have at least this many rows. |
| `hidden_dim` | Training/backtest | Width of each MLP block. | `512` | Larger values increase model capacity and memory/compute usage. |
| `depth` | Training/backtest | Number of stacked MLP blocks. | `4` | Must be >= 1; deeper models can overfit faster on small datasets. |
| `dropout` | Training/backtest | Dropout rate inside each block. | `0.15` | Range `[0, 1)`; raise slightly if overfitting, lower if underfitting. |

## 4) Troubleshooting matrix

| Symptom | Likely cause | Fix |
|---|---|---|
| `No aggtrades CSV files found ...` during feature build | Raw aggtrades missing for symbol/date range. | Re-run **Fetch aggtrades** with matching symbol/range. |
| Feature build outputs very small dataset | Narrow date range or too-large timeframe. | Increase date range and/or use `5m` instead of `15m`/`1h`. |
| Labels are mostly `NO_TRADE` | TP/SL too strict for current horizon/timeframe. | Lower TP, increase horizon steps, or both. |
| `Dataset has fewer rows than lookback_window` | Lookback too large for cleaned dataset/split. | Reduce `lookback_window` or build more data. |
| `Dataset is too small to split safely` | Not enough rows after preprocessing. | Expand date range and keep one symbol for first run. |
| Backtest error: missing checkpoint keys | Checkpoint is too old/incompatible. | Retrain baseline checkpoint in current codebase. |
| Backtest error: missing feature columns required by checkpoint | Label file generated from different feature schema than training run. | Rebuild labels from matching feature dataset and retrain if needed. |
| Backtest produces zero trades | Thresholds and/or margin too strict. | Lower long/short thresholds or reduce margin incrementally. |
| High trade count but poor returns | Thresholds too loose; noisy decisions. | Raise thresholds/margin and inspect class balance + validation macro-F1. |

## 5) Validation checklist for first successful baseline

Use this as a completion gate before doing any tuning sweep.

### Data and labeling

- [ ] Feature file exists under `data/features/<symbol>/<dataset_id>/features.csv`.
- [ ] Label file exists under `data/labels/<symbol>/<dataset_id>/labels.csv`.
- [ ] Label distribution includes non-zero LONG and SHORT counts.
- [ ] No obvious timestamp ordering issues (rows sorted by `timestamp_ms`).

### Training

- [ ] Checkpoint saved to `models/baseline/<symbol>/<run_id>/model.pt`.
- [ ] Training report saved (`report.json`) and history saved (`history.json`).
- [ ] Validation/test macro-F1 is non-zero and confusion matrix is not degenerate (single-class predictions only).

### Backtest

- [ ] Report/trades/predictions files created under `artifacts/backtests/<symbol>/<run_id>/`.
- [ ] Backtest trade count is plausible (not zero unless expected, not absurdly high).
- [ ] Core metrics in report are parseable and finite (`cumulative_return_pct`, `max_drawdown_pct`, `profit_factor`).

### Reproducibility and traceability

- [ ] `runs/<run_id>.json` exists for both train and backtest phases.
- [ ] `latest` tags are written for trained model and backtest when default tags are used.
- [ ] Document the exact symbol/date range/timeframe/hyperparameters for your baseline run.

## 6) Simple first tuning loop

Keep everything fixed and sweep one variable at a time:

1. **Timeframe**: `5m` → `15m` → `1h`
2. **Horizon steps**: `6`, `12`, `24`
3. **TP/SL pairs**:
   - `0.30 / 0.20`
   - `0.35 / 0.25`
   - `0.50 / 0.35`
4. **Backtest thresholds/margin**:
   - Raise thresholds/margin if overtrading.
   - Lower thresholds/margin if no trades.

Track:

- Label class balance.
- Validation macro-F1.
- Backtest trade count + return dispersion.

## 7) Baseline architecture (current)

The baseline model is a flattened lookback-window MLP:

- Input shape: `[batch, lookback_window, features]`
- Flatten: `lookback_window * features`
- Backbone: stacked MLP blocks (`Linear -> LayerNorm -> GELU -> Dropout -> Linear -> LayerNorm -> GELU`)
- Residual connection when block input/output dimensions match
- Head: `Linear(hidden_dim -> 128) -> GELU -> Linear(128 -> 3)`
