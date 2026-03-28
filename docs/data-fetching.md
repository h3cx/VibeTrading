# Data Fetching Throughput Tuning

Aggtrades archive fetch now emits per-run profiling data to:

- `artifacts/fetch_reports/<run_id>.json`

Each report contains per-day counters/timings (`download_seconds`, `parse_seconds`, `write_seconds`, `archive_bytes`, `parsed_rows`, `persisted_rows`, `retries`) and aggregate throughput/utilization metrics.

## What to look at first

After `fetch-aggtrades`, check the console summary and the JSON report.

Key aggregate fields:

- `effective_download_mb_s` — network stage throughput.
- `parse_rows_s` — parse/decompress throughput.
- `persist_rows_s` — disk persist throughput.
- `worker_utilization` — busy ratio by stage (`download`, `parse`, `persist`).
- `queue_wait_*` — backpressure and starvation indicators.

Interpretation guide:

- **Network bottleneck**: low `effective_download_mb_s`, high `download` worker utilization, low parse/persist utilization.
- **Parse bottleneck**: high parse utilization, growing parse-input queue wait, download faster than parse.
- **Disk bottleneck**: high persist utilization, low `persist_rows_s`, parse outpacing write.

## Tuning knobs

### `max_download_workers`

- Increase when network is underutilized and API allows more throughput.
- Decrease when retries rise or API throttling appears.
- If memory spikes, this is usually the first value to reduce.

### Inflight buffer size (`max_inflight_days` and `max_parsed_batches`)

- Increase slightly to smooth bursty stages.
- Keep bounded: oversized queues increase peak RAM without improving steady-state throughput.
- If queue wait is near zero and utilization is balanced, larger buffers likely do not help.

### Parse chunk size (`parse_chunksize_rows`)

- Larger chunks reduce Python/pandas overhead but raise per-batch memory.
- Smaller chunks can help memory stability but may reduce rows/s due to overhead.
- Tune until parse throughput is stable without large RSS swings.

### When to reduce concurrency

Reduce workers when:

- Disk is saturated (persist throughput plateaus while write utilization stays high).
- API throttling or retry counts climb.
- Memory grows sharply with little throughput gain.

## Recommended baseline profiling run

Run the same symbol twice with identical knobs:

1. **7-day profile run** (quick baseline)
2. **30-day profile run** (scaling check)

Success criteria:

- Throughput (`effective_download_mb_s`, `parse_rows_s`, `persist_rows_s`) stays in the same rough band.
- Total runtime for 30 days is approximately linear vs 7 days.
- Memory remains stable (no unbounded growth during the longer run).

If 30-day throughput degrades substantially vs 7-day, re-check stage utilization and queue waits to identify the constrained stage.


## Archive parser resilience policy

Aggtrades archive parsing now tolerates common day-file quality issues while keeping pipeline concurrency:

- Header rows inside archive CSV payloads are ignored safely.
- Malformed rows are coerced at chunk-level and dropped if required numeric/bool fields are invalid.
- If `skip_bad_days` is enabled from CLI, parse/persist failures on one day are recorded under `skipped_days` in the report and the rest of the range continues.
- Download-stage failures still abort the run because no archive payload exists for that day.
