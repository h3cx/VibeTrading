# QA Pass 2 Summary

Date: 2026-03-28  
Scope: static QA + targeted runtime validation + same-cycle fixes.

## Fixed items

1. **QA-001 (critical)** — fixed undefined `agg_source_paths` in feature build path.
2. **QA-002 (high)** — corrected label `time_to_exit_s` to elapsed seconds (timestamp delta).
3. **QA-003 (medium)** — corrected label manifest `date_range.end_ms`/`end_at` to use `timeframe_s`.
4. **QA-004 (high)** — aligned train label preprocessing with backtest duplicate handling.
5. **QA-005 (low)** — exposed `seed`/`run_tag` prompts in CLI baseline flows.
6. **QA-006 (high)** — added checkpoint compatibility alias handling for legacy artifacts.
7. **QA-007 (medium)** — added file-existence guards for explicit train/backtest input paths.

## Regression coverage added

- Unit regression for `time_to_exit_s` second-based semantics.
- Unit regression for train/backtest duplicate-cleaning parity.
- Unit regression for train/backtest lookback sequence alignment.
- Unit regression for legacy checkpoint alias compatibility.

## Deferred items

1. **QA-008 (medium)** — feature leakage policy hardening.
   - **Rationale:** existing leakage exclusion list works for current schema, but stronger policy enforcement (schema contract + CI guardrail) needs a separate design decision to avoid blocking iterative feature work.

## Residual risks

- End-to-end fixture-backed integration tests for feature→label manifest generation are still limited; current pass emphasizes targeted unit coverage.
- Legacy checkpoint compatibility currently supports known aliases; unknown historical variants may still fail.
- Runtime backtest realism still depends on external assumptions (slippage/fee settings and labeling horizon quality).

## Next steps

1. Add small deterministic fixture datasets for full feature+label integration tests.
2. Add schema-policy test that enforces no leakage columns can enter model features.
3. Add a compatibility matrix doc for historical checkpoint formats.
