# Codebase Audit Report

Date: 2026-03-28  
Method: static review + targeted runtime probes.

## Finding QA-001 — `critical`

**Title:** `build_feature_frames(...)` crashes immediately due to undefined `agg_source_paths`.

**Affected files/functions:**
- `src/trader/features/feature_builder.py`
  - `build_feature_frames(...)`

**Reproduction steps:**
1. Run:
   ```bash
   uv run python - <<'PY'
   from trader.features.feature_builder import build_feature_frames
   try:
       build_feature_frames(symbol='BTCUSDT', start_ms=0, end_ms=1000, include_kline_context=False)
   except Exception as e:
       print(type(e).__name__, e)
   PY
   ```
2. Observe immediate failure before any data IO.

**Expected behavior:**
- Function should resolve/select aggtrade source files and proceed to streaming aggregation.

**Actual behavior:**
- Raises `NameError: name 'agg_source_paths' is not defined`.

**Impact:**
- Feature pipeline is non-functional; downstream labeling/training/backtesting cannot proceed from fresh raw data.

---

## Finding QA-002 — `high`

**Title:** `time_to_exit_s` is recorded in bars, not seconds.

**Affected files/functions:**
- `src/trader/labels/label_builder.py`
  - `_compute_label_for_current(...)`

**Reproduction steps:**
1. Run:
   ```bash
   uv run python - <<'PY'
   from trader.labels.label_builder import FeatureRow, _compute_label_for_current
   current = FeatureRow(raw={}, timestamp_ms=0, timestamp_s=0, high=100.0, low=100.0, close=100.0)
   future = [FeatureRow(raw={}, timestamp_ms=5000, timestamp_s=5, high=101.0, low=100.0, close=101.0)]
   res = _compute_label_for_current(
       current=current,
       future_rows=future,
       tp=0.005,
       sl=0.005,
       total_cost=0.0,
       horizon_steps=12,
       horizon_seconds=60,
       take_profit_pct=0.5,
       stop_loss_pct=0.5,
       fee_pct=0.0,
       slippage_pct=0.0,
   )
   print(res['label_name'], res['time_to_exit_s'], res['horizon_seconds'])
   PY
   ```
2. Inspect printed output.

**Expected behavior:**
- `time_to_exit_s` should be elapsed seconds (e.g., 5 for one 5-second bar or 300 for one 5-minute bar).

**Actual behavior:**
- Output is `1.0` for one-bar exit regardless of bar duration.

**Impact:**
- Backtest holding period gate (`next_allowed_timestamp_ms`) is mis-timed, enabling unrealistic trade cadence and distorted PnL/risk metrics.

---

## Finding QA-003 — `medium`

**Title:** Label manifest `date_range.end_ms` assumes 1-second bars.

**Affected files/functions:**
- `src/trader/labels/label_builder.py`
  - `build_labels(...)` manifest construction

**Reproduction steps:**
1. Run:
   ```bash
   uv run python - <<'PY'
   from pathlib import Path
   import csv, json
   from trader.labels.label_builder import build_labels
   from trader.data.registry import find_dataset_id_by_artifact

   feat = Path('tmp_features_qa.csv')
   with feat.open('w', newline='', encoding='utf-8') as f:
       w = csv.writer(f)
       w.writerow(['timestamp','timestamp_ms','timestamp_s','timeframe_s','open','high','low','close'])
       w.writerow(['2024-01-01 00:00:00+00:00',0,0,300,100,100,100,100])
       w.writerow(['2024-01-01 00:05:00+00:00',300000,300,300,101,101,101,101])
       w.writerow(['2024-01-01 00:10:00+00:00',600000,600,300,102,102,102,102])

   out = build_labels(symbol='BTCUSDT', horizon_steps=1, take_profit_pct=0.1, stop_loss_pct=0.1, fee_pct=0.0, slippage_pct=0.0, dataset_name='qa_tmp', input_path=str(feat))
   dataset_id = find_dataset_id_by_artifact(dataset_type='labels', artifact_path=out)
   manifest = Path('data/registry/labels') / f'{dataset_id}.json'
   obj = json.loads(manifest.read_text())
   print(obj['date_range']['start_ms'], obj['date_range']['end_ms'], obj['timeframe_s'])
   feat.unlink()
   PY
   ```
2. Observe `end_ms` relative to last row and `timeframe_s`.

**Expected behavior:**
- For `timeframe_s=300`, `end_ms` should be `last_ts_ms + 300000`.

**Actual behavior:**
- `end_ms` is `last_ts_ms + 1000`.

**Impact:**
- Registry metadata understates temporal coverage, causing discoverability and lineage issues when selecting datasets by date window.

---

## Finding QA-004 — `high`

**Title:** Train/backtest label loaders diverge on duplicate timestamp handling.

**Affected files/functions:**
- `src/trader/baseline/train_baseline.py`
  - `_load_labels(...)`
- `src/trader/baseline/backtest_baseline.py`
  - `_load_labels(...)`

**Reproduction steps:**
1. Run:
   ```bash
   uv run python - <<'PY'
   from pathlib import Path
   import pandas as pd
   from trader.baseline.train_baseline import _load_labels as train_load
   from trader.baseline.backtest_baseline import _load_labels as backtest_load

   p = Path('tmp_labels_dup.csv')
   pd.DataFrame([
       {'timestamp':'2024-01-01 00:00:00+00:00','timestamp_ms':1000,'timestamp_s':1,'label':0,'label_name':'NO_TRADE','close':100,'long_net_return_pct':0,'short_net_return_pct':0,'time_to_exit_s':1,'long_event':'HORIZON','short_event':'HORIZON','exit_reason':'HORIZON','f1':1.0},
       {'timestamp':'2024-01-01 00:00:00+00:00','timestamp_ms':1000,'timestamp_s':1,'label':1,'label_name':'LONG_SETUP','close':100,'long_net_return_pct':1,'short_net_return_pct':-1,'time_to_exit_s':1,'long_event':'TP','short_event':'SL','exit_reason':'TP','f1':2.0},
       {'timestamp':'2024-01-01 00:00:05+00:00','timestamp_ms':5000,'timestamp_s':5,'label':2,'label_name':'SHORT_SETUP','close':99,'long_net_return_pct':-1,'short_net_return_pct':1,'time_to_exit_s':1,'long_event':'SL','short_event':'TP','exit_reason':'TP','f1':3.0},
   ]).to_csv(p, index=False)

   train_df = train_load('BTCUSDT', input_path=str(p))
   backtest_df = backtest_load(p)
   print('train_rows', len(train_df), 'backtest_rows', len(backtest_df))
   p.unlink()
   PY
   ```
2. Compare row counts.

**Expected behavior:**
- Label preprocessing should be identical between training and backtest paths.

**Actual behavior:**
- Training keeps duplicates; backtest drops duplicates by `timestamp_ms`.

**Impact:**
- Split boundaries and effective sample distribution differ between model fit and evaluation, weakening baseline comparability and reproducibility.

---

## Finding QA-005 — `low`

**Title:** CLI does not expose some baseline reproducibility knobs.

**Affected files/functions:**
- `src/trader/cli.py`
  - `train_baseline_cmd(...)`
  - `backtest_baseline_cmd(...)`

**Reproduction steps:**
1. Inspect CLI prompts and function call signatures for `train_baseline(...)` and `backtest_baseline(...)`.
2. Compare exposed params to callable defaults in baseline modules.

**Expected behavior:**
- Important reproducibility metadata controls (e.g., `seed`, `run_tag`) are user-configurable via CLI.

**Actual behavior:**
- CLI hardcodes default behavior; no prompt/flag for `seed` or `run_tag`.

**Impact:**
- Lower experiment traceability from interactive runs (not a correctness break, but a QA/repro friction point).

---

## Overall Assessment

The pipeline has one blocking feature-generation failure and two high-severity consistency/time-semantics issues that can materially distort backtest validity. These should be fixed before relying on baseline metrics for strategy decisions.
