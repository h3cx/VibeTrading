# VibeTrading

VibeTrading is a deliberately **vibe-coded** crypto trading research project: practical, iterative, and optimized for fast experiments over perfect architecture.

The current baseline pipeline is tuned for **longer intraday horizons** (minutes to hours), which dramatically reduces RAM and dataset size versus second-level scalping.

## What this project does

The CLI pipeline lets you:

1. Fetch market data (klines + aggtrades).
2. Build timeframe-aware feature datasets (`5m`, `15m`, `1h`, etc.).
3. Build labels using a horizon in **bars** (`horizon_steps`, not raw seconds).
4. Train a baseline PyTorch classifier (`NO_TRADE`, `LONG_SETUP`, `SHORT_SETUP`).
5. Backtest model probabilities with threshold-based execution rules.

## Architecture summary (current baseline)

The baseline model is a **lookback-window residual MLP**:

- Input: `[batch, lookback_window, feature_count]`.
- Flattening: each sample is reshaped to `[batch, lookback_window * feature_count]`.
- Backbone: stacked MLP blocks with:
  - `Linear -> LayerNorm -> GELU -> Dropout -> Linear -> LayerNorm -> GELU`
  - Residual connection when input/output dims match.
- Head: `Linear(hidden_dim -> 128) -> GELU -> Linear(128 -> 3)`.

This architecture keeps training simple and fast while still modeling short temporal context via fixed lookback windows.

## Canonical baseline run sequence

Install and open the menu:

```bash
uv sync
uv run trader menu
```

Run these menu options in order for a first reliable baseline:

1. **Initialize project config**
   - Exchange: `binance`
   - Symbols: `BTCUSDT` (or a small symbol set)
   - Timeframe: `5m`
2. **Fetch klines**
   - Interval: `1m`
   - Start/end date: small first range (for example, 1-2 weeks)
3. **Fetch aggtrades**
   - Same symbol + date range as klines
   - Fetch reports are written to `artifacts/fetch_reports/<run_id>.json` with stage timing/throughput metrics
4. **Build features**
   - Dataset name: `default`
   - Feature timeframe: `5m`
   - Include 1m context: `y`
5. **Build labels**
   - Horizon in bars: `12`
   - TP/SL: `0.35 / 0.25`
   - Fee/slippage: `0.08 / 0.02`
6. **Train baseline**
   - Epochs: `20`
   - Batch size: `512`
   - Learning rate: `0.001`
   - Lookback window: `12`
   - Hidden dim: `512`
   - Depth: `4`
   - Dropout: `0.15`
7. **Backtest baseline**
   - Eval split: `test`
   - Long threshold: `0.80`
   - Short threshold: `0.80`
   - Margin: `0.05`

## Compatibility notes for old checkpoints

Current backtesting requires checkpoints with at least:

- `model_state_dict`
- `input_dim`
- `feature_columns`
- `standardizer_mean`
- `standardizer_std`
- `symbol`

Compatibility behavior:

- If `lookback_window` is missing in an older checkpoint, backtest falls back to `1`.
- If `hidden_dim`, `depth`, or `dropout` are missing, backtest falls back to `512`, `4`, and `0.15`.
- If the label CSV used for backtest does not contain all `feature_columns` from the checkpoint, backtest fails with a missing-feature error.

Practical guidance:

- Prefer training and backtesting from the same pipeline version.
- If you must reuse old checkpoints, run a quick smoke backtest on a short slice first.

## Detailed docs

- Baseline walkthrough and first-run checklist: [`docs/longer-term-baseline.md`](docs/longer-term-baseline.md)
- CLI reference and defaults: [`docs/reference/cli-commands.md`](docs/reference/cli-commands.md)
- Data-fetch performance tuning: [`docs/data-fetching.md`](docs/data-fetching.md)
- Artifact/output folder layout: [`docs/reference/artifact-layout.md`](docs/reference/artifact-layout.md)
- QA/codebase limitations report: [`docs/qa/codebase-audit-report.md`](docs/qa/codebase-audit-report.md)

## Project layout

- `src/trader/cli.py` — interactive commands.
- `src/trader/features/feature_builder.py` — streaming feature generation.
- `src/trader/labels/label_builder.py` — forward-looking label generation.
- `src/trader/baseline/train_baseline.py` — baseline model training.
- `src/trader/baseline/backtest_baseline.py` — baseline model backtesting.

## Notes

- This is research code and not financial advice.
- Start with one symbol and short date ranges to validate setup before scaling.

## Known limitations

A current QA sweep identified active correctness and consistency gaps in the pipeline (including feature-build runtime failure and train/backtest parity issues).

See: `docs/qa/codebase-audit-report.md`
