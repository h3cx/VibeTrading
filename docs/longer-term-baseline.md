# Longer-Term Baseline Training Guide

This guide walks you through running a first baseline experiment on a laptop with lower memory pressure.

## 1) Choose a practical configuration

For most first runs:

- Symbol: `BTCUSDT`
- Data interval (klines): `1m`
- Feature timeframe: `5m`
- Label horizon: `12` bars (≈ 60 minutes at `5m`)
- TP / SL: `0.35% / 0.25%`

Why this works:

- Fewer rows than 1-second scalping.
- Enough market structure for short intraday swings.
- Baseline labels are still reactive enough for frequent experiments.

## 2) End-to-end flow

Use the menu:

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
   - Hidden size: `512`.
   - MLP depth: `4`.
   - Dropout: `0.15`.
7. **Backtest baseline**
   - Start with defaults, then tune thresholds.

## 3) How horizon works now

Labels use **bar-count horizon** (`horizon_steps`), not fixed seconds.  
Effective horizon time is:

`horizon_seconds = horizon_steps * timeframe_s`

So with `timeframe=15m` and `horizon_steps=8`, your horizon is 2 hours.

## 4) First tuning loop (simple and fast)

Keep everything fixed and sweep one variable at a time:

1. **Timeframe**: `5m` → `15m` → `1h`
2. **Horizon steps**: `6`, `12`, `24`
3. **TP/SL pairs**:
   - `0.30 / 0.20`
   - `0.35 / 0.25`
   - `0.50 / 0.35`
4. **Backtest thresholds**:
   - Increase long/short threshold if overtrading.
   - Increase margin if both long/short probabilities are noisy.

Track:

- Class balance in labels.
- Validation macro-F1.
- Backtest trade count and average expectancy.

## 5) Common pitfalls

- **Too few LONG/SHORT labels**: reduce TP or increase horizon steps.
- **Model predicts only NO_TRADE**: enable class weights, relax TP/SL.
- **RAM pressure remains high**: shorten date range or move to `15m`.
- **Noisy backtests**: raise thresholds and margin.

## 6) Baseline goals (realistic)

For an initial baseline, prioritize:

- Stable training runs.
- Sensible class distributions.
- Backtest behavior that is directionally reasonable.

You can optimize performance later once the dataset and labeling assumptions are stable.

## 7) Baseline architecture (current)

The baseline model is now a pure feedforward time-window MLP:

- Input shape: `[batch, lookback_window, features]`
- Flatten: `lookback_window * features`
- Backbone: stacked MLP blocks (`Linear -> LayerNorm -> GELU -> Dropout -> Linear -> LayerNorm -> GELU`)
- Residual connection applied when block input/output dimensions match
- Head: `Linear(512 -> 128) -> GELU -> Linear(128 -> 3)`

This keeps training simple and fast while adding enough capacity for richer baseline checks.
