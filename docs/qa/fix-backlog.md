# Fix Backlog (from Codebase Audit)

## QA-001 — Fix undefined aggtrade source variable in feature builder

- **ID:** QA-001
- **Severity:** critical
- **Owner:** `@owner-tbd`
- **Scope:** `src/trader/features/feature_builder.py::build_feature_frames`

**Acceptance criteria**
- `build_feature_frames(...)` defines and populates aggtrade source paths before calling `_iter_aggtrade_second_aggregates(...)`.
- Runtime smoke invocation no longer raises `NameError`.
- Feature manifest includes resolved aggtrade source files.

**Test plan**
- Unit/smoke: invoke `build_feature_frames(symbol='BTCUSDT', start_ms=..., end_ms=..., include_kline_context=False)` and verify no pre-I/O variable error.
- Integration: run end-to-end fetch→features on a small date range and verify `features.csv` + registry manifest are created.

---

## QA-002 — Correct `time_to_exit_s` to real seconds

- **ID:** QA-002
- **Severity:** high
- **Owner:** `@owner-tbd`
- **Scope:** `src/trader/labels/label_builder.py::_compute_label_for_current`

**Acceptance criteria**
- All `time_to_exit_s` assignments represent elapsed **seconds**, not bars.
- For a one-bar TP on timeframe `T`, output equals `T` seconds.
- Backtest hold logic consumes correctly scaled durations.

**Test plan**
- Unit: synthetic one-bar TP/SL scenarios across `timeframe_s` values (1, 5, 300) and assert `time_to_exit_s`.
- Regression: compare backtest trade counts before/after on same checkpoint+labels and confirm expected cadence changes.

---

## QA-003 — Fix label manifest end timestamp semantics

- **ID:** QA-003
- **Severity:** medium
- **Owner:** `@owner-tbd`
- **Scope:** `src/trader/labels/label_builder.py::build_labels` (manifest generation)

**Acceptance criteria**
- `date_range.end_ms` equals `last_ts_ms + timeframe_s*1000`.
- `date_range.end_at` reflects corrected `end_ms`.
- Behavior validated for non-1s bars (e.g., 5m timeframe).

**Test plan**
- Integration: build labels from synthetic `timeframe_s=300` features and assert manifest end is `+300000ms`.
- Add a test case covering 1s and 5m equivalence logic.

---

## QA-004 — Enforce train/backtest preprocessing parity

- **ID:** QA-004
- **Severity:** high
- **Owner:** `@owner-tbd`
- **Scope:**
  - `src/trader/baseline/train_baseline.py::_load_labels`
  - `src/trader/baseline/backtest_baseline.py::_load_labels`

**Acceptance criteria**
- Duplicate timestamp handling is consistent between train and backtest (either both drop with same rule or both keep).
- Same input labels CSV yields matching cleaned row count and timestamp set in both paths.
- Parity behavior is documented in module docstrings/comments.

**Test plan**
- Unit: synthetic labels with duplicate timestamps and mixed ordering; assert identical cleaned outputs.
- Regression: training and backtest run metadata should report consistent source row accounting when using same labels input.

---

## QA-005 — Expose reproducibility knobs in CLI

- **ID:** QA-005
- **Severity:** low
- **Owner:** `@owner-tbd`
- **Scope:** `src/trader/cli.py` baseline commands

**Acceptance criteria**
- CLI prompts/flags include `seed` and `run_tag` for training and backtesting.
- Defaults remain backward compatible.
- Generated artifacts/runs visibly reflect chosen values.

**Test plan**
- CLI smoke: run train/backtest with non-default seed/tag and verify output run metadata.
- Manual: validate interactive menu still works with defaults and with explicit values.
