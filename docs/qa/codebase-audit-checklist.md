# Codebase QA Audit Checklist

Date: 2026-03-28  
Scope: full static + runtime sweep with emphasis on data pipeline correctness and train/backtest consistency.

## 1) Feature generation integrity (`src/trader/features/feature_builder.py`)

- [ ] Validate function-level wiring in `build_feature_frames(...)` (all local variables defined before use).
- [ ] Confirm timeframe semantics are consistent for aggregated bar fields (`ret_*`, rolling windows, timestamp alignment).
- [ ] Validate source file selection behavior for aggtrades and klines (`_select_overlapping_files`, `_iter_aggtrade_second_aggregates`, `_load_kline_context`).
- [ ] Run a runtime smoke check of `build_feature_frames(...)` with minimal arguments to catch pre-I/O control flow failures.

## 2) Label correctness and time semantics (`src/trader/labels/label_builder.py`)

- [ ] Verify label decision logic in `_compute_label_for_current(...)` (TP/SL/horizon ordering and tie handling).
- [ ] Verify `time_to_exit_s` units are seconds (not bars) for every label path.
- [ ] Verify label dataset date range semantics in `build_labels(...)` manifest output (`date_range.end_ms` should match timeframe width).
- [ ] Run a runtime unit-style check for a known TP-on-first-future-bar scenario.
- [ ] Run a runtime integration check that builds labels from a synthetic features CSV and inspects resulting registry metadata.

## 3) Train/backtest parity (`src/trader/baseline/train_baseline.py`, `src/trader/baseline/backtest_baseline.py`)

- [ ] Compare label loading/cleaning semantics (`_load_labels`) for row-count parity and duplicate handling.
- [ ] Validate split mechanics are equivalent (`_split_by_time`) and applied at the same pipeline stage.
- [ ] Validate feature column parity and standardizer usage (`feature_columns`, `standardizer_mean/std`, lookback sequencing).
- [ ] Run a runtime parity probe using a synthetic labels CSV with duplicate timestamps.

## 4) CLI defaults and parameter plumbing (`src/trader/cli.py`)

- [ ] Verify CLI defaults align with callable defaults for `build_labels`, `train_baseline`, and `backtest_baseline`.
- [ ] Verify every prompted parameter is passed through to downstream functions.
- [ ] Identify any missing exposure of important model/backtest knobs (e.g., `seed`, `run_tag`) and document impact.

## Execution Log

- [x] Static review completed for all listed files/functions.
- [x] Runtime checks executed for targeted failure/consistency cases.
- [x] Findings documented in `docs/qa/codebase-audit-report.md` with severity and reproduction details.
- [x] Fix tasks captured in `docs/qa/fix-backlog.md`.
