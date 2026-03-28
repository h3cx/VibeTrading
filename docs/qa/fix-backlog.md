# Fix Backlog (QA Pass 2)

## QA-001 — Resolve aggtrade source-path wiring

- **Severity:** critical
- **Affected paths:** `src/trader/features/feature_builder.py`
- **Acceptance criteria:**
  - `build_feature_frames(...)` resolves `agg_source_paths` before first use.
  - Source files are propagated to manifest metadata.
  - Feature build no longer fails with `NameError`.
- **Verification command/check:**
  - `uv run ruff check src/trader/features/feature_builder.py`
  - `uv run python -m unittest tests.test_qa_regressions.QARegressionTests.test_time_to_exit_is_seconds`

## QA-002 — Fix label `time_to_exit_s` units

- **Severity:** high
- **Affected paths:** `src/trader/labels/label_builder.py`
- **Acceptance criteria:**
  - `time_to_exit_s` stores elapsed seconds (timestamp delta), not bar count.
  - One 5-minute future bar yields `300` seconds when the trade exits on that bar.
- **Verification command/check:**
  - `uv run python -m unittest tests.test_qa_regressions.QARegressionTests.test_time_to_exit_is_seconds`

## QA-003 — Correct label manifest end-range semantics

- **Severity:** medium
- **Affected paths:** `src/trader/labels/label_builder.py`
- **Acceptance criteria:**
  - Manifest `date_range.end_ms` = `last_ts_ms + timeframe_s*1000`.
  - Manifest `date_range.end_at` is derived from corrected `end_ms`.
- **Verification command/check:**
  - Static review of `build_labels(...)` manifest assignment.
  - End-to-end label build smoke in future QA runbook.

## QA-004 — Enforce train/backtest preprocessing parity

- **Severity:** high
- **Affected paths:**
  - `src/trader/baseline/train_baseline.py`
  - `src/trader/baseline/backtest_baseline.py`
- **Acceptance criteria:**
  - Both paths drop duplicate `timestamp_ms` rows with `keep='last'` after sorting.
  - Same input CSV yields same cleaned timestamp series.
- **Verification command/check:**
  - `uv run python -m unittest tests.test_qa_regressions.QARegressionTests.test_train_backtest_label_preprocess_parity`

## QA-005 — Expose reproducibility knobs in CLI

- **Severity:** low
- **Affected paths:** `src/trader/cli.py`
- **Acceptance criteria:**
  - Training exposes prompt for `seed` and `run_tag`.
  - Backtest exposes prompt for `run_tag`.
  - Values are passed through to called functions.
- **Verification command/check:**
  - Static review of `train_baseline_cmd(...)` and `backtest_baseline_cmd(...)`.

## QA-006 — Checkpoint compatibility alias handling

- **Severity:** high
- **Affected paths:** `src/trader/baseline/backtest_baseline.py`
- **Acceptance criteria:**
  - Backtest loader accepts legacy key aliases (`state_dict`, `feature_cols`, `standardizer_mu`, `standardizer_sigma`).
  - Missing canonical fields are populated from aliases before strict validation.
- **Verification command/check:**
  - `uv run python -m unittest tests.test_qa_regressions.QARegressionTests.test_checkpoint_alias_compatibility`

## QA-007 — Data-path robustness for training/backtest inputs

- **Severity:** medium
- **Affected paths:**
  - `src/trader/baseline/train_baseline.py`
  - `src/trader/baseline/backtest_baseline.py`
- **Acceptance criteria:**
  - File existence is checked up-front for explicit input files.
  - Error messages identify the missing path clearly.
- **Verification command/check:**
  - `uv run python -m unittest tests.test_qa_regressions`

## QA-008 — Feature leakage audit follow-up (deferred)

- **Severity:** medium
- **Affected paths:** `src/trader/baseline/train_baseline.py`, `src/trader/baseline/backtest_baseline.py`
- **Acceptance criteria:**
  - Add automated schema guard ensuring no post-event fields can enter feature columns.
- **Verification command/check:**
  - Deferred to QA pass 3 due to required schema policy RFC.
