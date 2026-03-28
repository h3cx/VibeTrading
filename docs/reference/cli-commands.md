# CLI Commands Reference

This reference explains each pipeline action exposed by `trader` CLI, including prompts, meanings, and defaults.

## Start the CLI

```bash
uv run trader menu
```

Menu actions map to these commands:

1. `init`
2. `fetch-klines`
3. `fetch-aggtrades`
4. `build-features`
5. `build-labels`
6. `train-baseline`
7. `backtest-baseline`

---

## 1) `init`

Initializes `configs/project.toml`.

### Prompts and defaults

| Prompt | Type | Default | Meaning |
|---|---|---|---|
| `Exchange` | string | `binance` | Exchange identifier written to config. |
| `Symbols (comma separated)` | string | `BTCUSDT,SOLUSDT` | Tradable symbols list for convenience defaults in later menu steps. |
| `Decision timeframe` | string | `5m` | Baseline timeframe preference stored in config. |

### Output

- Writes `configs/project.toml`.

---

## 2) `fetch-klines`

Fetches kline candles for a symbol/date range.

### Prompts and defaults

| Prompt | Type | Default | Meaning |
|---|---|---|---|
| `Symbol` | string | first configured symbol | Symbol to download. |
| `Kline interval` | string | `1m` | Raw kline interval from exchange. |
| `Start date (YYYY-MM-DD)` | date string | none | Inclusive start date. |
| `End date (YYYY-MM-DD)` | date string | none | End date boundary supplied to fetch function. |
| `Klines source` | enum | `auto` | Data source strategy: `auto`, `rest`, or `archive`. |

### Notes

- Symbol validity is checked before fetch.
- Dates are parsed as UTC midnight with format `%Y-%m-%d`.

---

## 3) `fetch-aggtrades`

Fetches aggregate trade stream data for a symbol/date range.

### Prompts and defaults

| Prompt | Type | Default | Meaning |
|---|---|---|---|
| `Symbol` | string | first configured symbol | Symbol to download. |
| `Start date (YYYY-MM-DD)` | date string | none | Inclusive start date. |
| `End date (YYYY-MM-DD)` | date string | none | End date boundary supplied to fetch function. |
| `Aggtrades source` | enum | `auto` | Data source strategy: `auto`, `rest`, or `archive`. |
| `Max download workers` | int | `6` | Number of daily archive downloads/decompressions run in parallel. |
| `Max inflight days (bounded queue for backpressure)` | int | `12` | Producer queue bound; larger values can improve throughput but increase transient memory pressure. |
| `Max parse workers` | int | `4` | Number of archive parse/decompression workers. |
| `Max parsed batches (bounded parse->persist queue)` | int | `12` | Queue bound between parse and persist stages. |
| `Parse chunk size rows` | int | `250000` | Per-chunk row budget during archive CSV parsing. |
| `Request timeout seconds` | float | `30` | Per-request timeout for archive downloads. |
| `Max retries per day` | int | `3` | Retry count per failed day download before aborting the entire fetch. |
| `Retry backoff base seconds` | float | `1` | Exponential backoff base (`base * 2^attempt`) between retries. |
| `Sequential mode? (debug fallback; disables pipeline)` | enum (`y`/`n`) | `n` | Forces single-threaded download/parse/persist execution. |
| `Skip malformed day archives instead of failing the whole run?` | enum (`y`/`n`) | `n` | If `y`, parse/persist failures are logged and skipped; download failures still stop the run. |

### Notes

- Symbol validity is checked before fetch.
- Use same date range as klines for coherent downstream features.
- RAM tradeoff: each additional worker typically adds ~40-120MB peak RAM while a day archive is in-memory during decompress/write.
- Parser resilience: archive header rows and malformed lines are filtered during chunk sanitization before dtype casting.
- When skip mode is enabled, skipped day metadata is written in the fetch report (`skipped_days`).

---

## 4) `build-features`

Builds timeframe-aware feature bars from raw data.

### Prompts and defaults

| Prompt | Type | Default | Meaning |
|---|---|---|---|
| `Symbol` | string | first configured symbol | Symbol to process. |
| `Start date (YYYY-MM-DD)` | date string | none | Start boundary for feature build. |
| `End date (YYYY-MM-DD)` | date string | none | End boundary for feature build. |
| `Dataset name` | string | `default` | Human-readable dataset name included in dataset-id source metadata. |
| `Feature timeframe` | string (`<int>s|m|h`) | `5m` | Target aggregation timeframe for rows. |
| `Include 1m kline context?` | enum (`y`/`n`) | `y` | Whether to append kline-context distance features. |

### Output

- `data/features/<symbol>/<dataset_id>/features.csv`
- Dataset manifest in `data/registry/features/<dataset_id>.json`
- Optional tag file in `data/features/<symbol>/_tags/`

---

## 5) `build-labels`

Builds classification labels and forward-return diagnostics from features.

### Prompts and defaults

| Prompt | Type | Default | Meaning |
|---|---|---|---|
| `Symbol` | string | first configured symbol | Symbol to label. |
| `Feature CSV path (blank = latest for symbol)` | path/string | blank | Explicit feature CSV path; blank resolves latest dataset for symbol. |
| `Horizon in bars` | int | `12` | Number of future bars scanned (`horizon_steps`). |
| `Take-profit percent` | float | `0.35` | TP threshold in percent. |
| `Stop-loss percent` | float | `0.25` | SL threshold in percent. |
| `Round-trip fee percent` | float | `0.08` | Total fee assumption (entry+exit). |
| `Round-trip slippage percent` | float | `0.02` | Total slippage assumption (entry+exit). |
| `Label dataset name` | string | `default_labels` | Human-readable label dataset name in metadata. |

### Output

- `data/labels/<symbol>/<dataset_id>/labels.csv`
- Dataset manifest in `data/registry/labels/<dataset_id>.json`
- Optional tag file in `data/labels/<symbol>/_tags/`

---

## 6) `train-baseline`

Trains the lookback-window residual MLP baseline classifier.

### Prompts and defaults

| Prompt | Type | Default | Meaning |
|---|---|---|---|
| `Symbol` | string | first configured symbol | Symbol run namespace. |
| `Label CSV path (blank = latest for symbol)` | path/string | blank | Explicit labels CSV; blank resolves latest label dataset. |
| `Model name` | string | `baseline_mlp` | Saved metadata model name. |
| `Epochs` | int | `20` | Training epochs. |
| `Batch size` | int | `512` | Training batch size. |
| `Learning rate` | float | `0.001` | Optimizer learning rate. |
| `Lookback window (bars)` | int | `12` | Rows concatenated per sample. |
| `Hidden size` | int | `512` | `hidden_dim` for MLP blocks. |
| `MLP block depth` | int | `4` | Number of stacked MLP blocks. |
| `Dropout` | float | `0.15` | Dropout in each block. |
| `Train fraction` | float | `0.70` | Chronological split fraction for train. |
| `Validation fraction` | float | `0.15` | Chronological split fraction for validation. |
| `Random seed` | int | `42` | Global seed for Python/NumPy/PyTorch initialization. |
| `Run tag` | string | `latest` | Optional tag written under `models/baseline/<symbol>/_tags/`. |
| `Use class weights?` | enum (`y`/`n`) | `y` | Whether to weight CE loss by inverse class frequency. |

### Hard validations in code

- `epochs > 0`
- `batch_size > 0`
- `learning_rate > 0`
- `0 < train_frac < 1`
- `0 < val_frac < 1`
- `train_frac + val_frac < 1`
- `lookback_window > 0`
- `hidden_dim > 0`
- `depth >= 1`
- `0 <= dropout < 1`

### Output

- `models/baseline/<symbol>/<run_id>/model.pt`
- `models/baseline/<symbol>/<run_id>/report.json`
- `models/baseline/<symbol>/<run_id>/history.json`
- `runs/<run_id>.json`
- Optional tag file in `models/baseline/<symbol>/_tags/`

---

## 7) `backtest-baseline`

Runs inference + thresholded one-trade-at-a-time simulation.

### Prompts and defaults

| Prompt | Type | Default | Meaning |
|---|---|---|---|
| `Symbol` | string | first configured symbol | Symbol run namespace. |
| `Checkpoint path (blank = latest for symbol)` | path/string | blank | Explicit model checkpoint path; blank resolves latest baseline run checkpoint. |
| `Label CSV path (blank = latest for symbol)` | path/string | blank | Explicit labels CSV path; blank resolves latest labels dataset. |
| `Eval split` | enum | `test` | Which time split to evaluate: `train`, `val`, `test`, or `all`. |
| `Train fraction used during training` | float | `0.70` | Must match training split assumptions for fair comparison. |
| `Validation fraction used during training` | float | `0.15` | Must match training split assumptions for fair comparison. |
| `Inference batch size` | int | `2048` | Batch size for forward pass only. |
| `Long probability threshold` | float | `0.80` | Minimum long class probability to open LONG. |
| `Short probability threshold` | float | `0.80` | Minimum short class probability to open SHORT. |
| `Trade-vs-no-trade margin` | float | `0.05` | Required margin over `p_no_trade` to execute trade. |
| `Run tag` | string | `latest` | Optional tag written under `artifacts/backtests/<symbol>/_tags/`. |

### Decision logic summary

A LONG executes if all are true:

- `p_long >= long_threshold`
- `p_long > p_short`
- `p_long >= p_no_trade + margin`

A SHORT executes if all are true:

- `p_short >= short_threshold`
- `p_short > p_long`
- `p_short >= p_no_trade + margin`

### Output

- `artifacts/backtests/<symbol>/<run_id>/report.json`
- `artifacts/backtests/<symbol>/<run_id>/trades.csv`
- `artifacts/backtests/<symbol>/<run_id>/predictions.csv`
- `runs/<run_id>.json`
- Optional tag file in `artifacts/backtests/<symbol>/_tags/`

---

## Suggested baseline-first command sequence

For contributors who just want a reliable first run:

1. `init`
2. `fetch-klines`
3. `fetch-aggtrades`
4. `build-features`
5. `build-labels`
6. `train-baseline`
7. `backtest-baseline`

Keep one symbol and a short date range for your first pass, then scale.
