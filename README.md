# VibeTrading

VibeTrading is a deliberately **vibe-coded** crypto trading research project: practical, iterative, and optimized for fast experiments over perfect architecture.

The current baseline pipeline is tuned for **longer intraday horizons** (minutes to hours), which dramatically reduces RAM and dataset size versus second-level scalping.

## What this project does

The CLI pipeline lets you:

1. Fetch market data (klines + aggtrades).
2. Build timeframe-aware feature datasets (`5m`, `15m`, `1h`, etc.).
3. Build labels using a horizon in **bars** (not raw seconds).
4. Train a baseline PyTorch MLP classifier (`NO_TRADE`, `LONG_SETUP`, `SHORT_SETUP`).
5. Backtest model probabilities with threshold-based rules.

## Why the new default is longer-term

If you were previously using `1s` features for scalping, row count explodes quickly.  
Switching to `5m` cuts rows by ~300x (from 300 seconds per bar), making laptop training realistic.

## Quickstart

```bash
uv sync
uv run trader menu
```

Recommended baseline first run:

- Feature timeframe: `5m`
- Label horizon: `12` bars (about 1 hour at `5m`)
- TP / SL: `0.35% / 0.25%`
- Epochs: `20`
- Batch size: `512`

## Detailed usage and tuning guide

See: `docs/longer-term-baseline.md`

## Project layout

- `src/trader/cli.py` — interactive commands.
- `src/trader/features/feature_builder.py` — streaming feature generation.
- `src/trader/labels/label_builder.py` — forward-looking label generation.
- `src/trader/baseline/train_baseline.py` — baseline model training.
- `src/trader/baseline/backtest_baseline.py` — baseline model backtesting.

## Notes

- This is research code and not financial advice.
- Start with one symbol and short date ranges to validate setup before scaling.
