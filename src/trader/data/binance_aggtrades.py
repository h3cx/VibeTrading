from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from io import BytesIO
import json
from pathlib import Path
from queue import Queue
from threading import Event, Lock, Thread
from typing import Iterator
import csv
import time
import zipfile

import httpx
import pandas as pd

from trader.data.storage import ARTIFACTS_ROOT, raw_data_dir


BINANCE_ARCHIVE_BASE_URL = "https://data.binance.vision"


ARCHIVE_COLUMNS = [
    "agg_trade_id",
    "price",
    "quantity",
    "first_trade_id",
    "last_trade_id",
    "timestamp",
    "buyer_is_maker",
]

_NUMERIC_COLUMNS = [
    "agg_trade_id",
    "price",
    "quantity",
    "first_trade_id",
    "last_trade_id",
    "timestamp",
]

_INT_COLUMNS = ["agg_trade_id", "first_trade_id", "last_trade_id", "timestamp"]
_FLOAT_COLUMNS = ["price", "quantity"]

_TRUE_VALUES = {"true", "1", "t", "y", "yes"}
_FALSE_VALUES = {"false", "0", "f", "n", "no"}


def _day_label(day_start_ms: int) -> str:
    return datetime.fromtimestamp(day_start_ms / 1000, tz=UTC).strftime("%Y-%m-%d")


def _stage_error(*, stage: str, symbol: str, day_start_ms: int, details: str) -> RuntimeError:
    day = _day_label(day_start_ms)
    return RuntimeError(f"{stage} stage failed for {symbol} on {day}: {details}")


def _format_exc_message(exc: Exception) -> str:
    message = str(exc).strip()
    if message:
        return message
    return exc.__class__.__name__


def _normalize_archive_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    if chunk.empty:
        return pd.DataFrame(columns=ARCHIVE_COLUMNS)

    working = chunk.iloc[:, : len(ARCHIVE_COLUMNS)].copy()
    working.columns = ARCHIVE_COLUMNS
    for col in ARCHIVE_COLUMNS:
        working[col] = working[col].astype("string").str.strip()

    working = working[working["agg_trade_id"].str.fullmatch(r"\d+", na=False)]
    if working.empty:
        return pd.DataFrame(columns=ARCHIVE_COLUMNS)

    for col in _NUMERIC_COLUMNS:
        working[col] = pd.to_numeric(working[col], errors="coerce")
    working = working.dropna(subset=_NUMERIC_COLUMNS)
    if working.empty:
        return pd.DataFrame(columns=ARCHIVE_COLUMNS)

    maker = working["buyer_is_maker"].str.lower()
    bool_map = {value: True for value in _TRUE_VALUES}
    bool_map.update({value: False for value in _FALSE_VALUES})
    working["buyer_is_maker"] = maker.map(bool_map)
    working = working.dropna(subset=["buyer_is_maker"])
    if working.empty:
        return pd.DataFrame(columns=ARCHIVE_COLUMNS)

    for col in _INT_COLUMNS:
        working[col] = working[col].astype("int64")
    for col in _FLOAT_COLUMNS:
        working[col] = working[col].astype("float64")
    working["buyer_is_maker"] = working["buyer_is_maker"].astype("bool")

    return working[ARCHIVE_COLUMNS]


@dataclass(frozen=True)
class _DayJob:
    day_start_ms: int


@dataclass(frozen=True)
class _DayResult:
    day_start_ms: int
    output_path: Path


@dataclass(frozen=True)
class _DownloadResult:
    archive_bytes: bytes
    retries: int
    elapsed_s: float


@dataclass(frozen=True)
class _DownloadedDay:
    day_start_ms: int
    archive_bytes: bytes


@dataclass(frozen=True)
class _PersistChunk:
    day_start_ms: int
    frame: pd.DataFrame


@dataclass(frozen=True)
class _PersistDayDone:
    day_start_ms: int


@dataclass
class _DayMetrics:
    download_seconds: float = 0.0
    parse_seconds: float = 0.0
    write_seconds: float = 0.0
    archive_bytes: int = 0
    parsed_rows: int = 0
    persisted_rows: int = 0
    retries: int = 0


@dataclass
class _StageStats:
    active_s: float = 0.0
    wait_s: float = 0.0
    gets: int = 0
    wall_s: float = 0.0


@dataclass(frozen=True)
class FetchAggtradesResult:
    csv_path: Path
    metrics_path: Path
    run_id: str
    summary: str


def _utc_midnight_from_ms(value_ms: int) -> datetime:
    dt = datetime.fromtimestamp(value_ms / 1000, tz=UTC)
    return dt.replace(hour=0, minute=0, second=0, microsecond=0)


def _iter_utc_days(start_ms: int, end_ms: int) -> list[int]:
    if end_ms <= start_ms:
        raise ValueError("end_ms must be greater than start_ms")

    day = _utc_midnight_from_ms(start_ms)
    end_bound = datetime.fromtimestamp((end_ms - 1) / 1000, tz=UTC)
    jobs: list[int] = []

    while day <= end_bound:
        jobs.append(int(day.timestamp() * 1000))
        day += timedelta(days=1)

    return jobs


def _build_archive_url(symbol: str, day_start_ms: int) -> str:
    day = datetime.fromtimestamp(day_start_ms / 1000, tz=UTC).strftime("%Y-%m-%d")
    return (
        f"{BINANCE_ARCHIVE_BASE_URL}/data/futures/um/daily/aggTrades/"
        f"{symbol}/{symbol}-aggTrades-{day}.zip"
    )


def _build_day_csv_path(symbol: str, day_start_ms: int) -> Path:
    day = datetime.fromtimestamp(day_start_ms / 1000, tz=UTC).strftime("%Y%m%d")
    out_dir = raw_data_dir(exchange="binance", symbol=symbol, datatype="aggtrades") / "daily"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"aggtrades_{day}.csv"


def download_day_archive(
    *,
    client: httpx.Client,
    symbol: str,
    day_start_ms: int,
    request_timeout_s: float,
    max_retries: int,
    retry_backoff_s: float,
) -> _DownloadResult:
    url = _build_archive_url(symbol=symbol, day_start_ms=day_start_ms)

    last_error: Exception | None = None
    start = time.perf_counter()
    for attempt in range(max_retries + 1):
        try:
            response = client.get(url, timeout=request_timeout_s)
            response.raise_for_status()
            elapsed_s = time.perf_counter() - start
            return _DownloadResult(
                archive_bytes=response.content,
                retries=attempt,
                elapsed_s=elapsed_s,
            )
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt >= max_retries:
                break
            sleep_s = retry_backoff_s * (2**attempt)
            time.sleep(sleep_s)

    raise RuntimeError(f"Failed to download archive for {symbol} @ {url}") from last_error


def parse_decompress_day(*, archive_bytes: bytes, chunksize_rows: int) -> Iterator[pd.DataFrame]:
    with zipfile.ZipFile(BytesIO(archive_bytes)) as zf:
        members = [name for name in zf.namelist() if name.lower().endswith(".csv")]
        if not members:
            raise RuntimeError("Zip archive has no CSV payload")

        with zf.open(members[0], "r") as zipped:
            frame_iter = pd.read_csv(
                zipped,
                header=None,
                names=ARCHIVE_COLUMNS,
                usecols=range(len(ARCHIVE_COLUMNS)),
                dtype={col: "string" for col in ARCHIVE_COLUMNS},
                low_memory=False,
                on_bad_lines="skip",
                chunksize=chunksize_rows,
            )
            for chunk in frame_iter:
                sanitized = _normalize_archive_chunk(chunk)
                if not sanitized.empty:
                    yield sanitized


def persist_day(
    *,
    frame: pd.DataFrame,
    output_path: Path,
    write_header: bool,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, mode="a", header=write_header, index=False)


def _concatenate_daily_files(*, symbol: str, start_ms: int, end_ms: int, ordered_paths: list[Path]) -> Path:
    out_dir = raw_data_dir(exchange="binance", symbol=symbol, datatype="aggtrades")
    out_dir.mkdir(parents=True, exist_ok=True)

    start_stamp = datetime.fromtimestamp(start_ms / 1000, tz=UTC).strftime("%Y%m%d_%H%M%S")
    end_stamp = datetime.fromtimestamp(end_ms / 1000, tz=UTC).strftime("%Y%m%d_%H%M%S")
    final_path = out_dir / f"aggtrades_{start_stamp}__{end_stamp}.csv"

    with final_path.open("w", newline="", encoding="utf-8") as dst:
        writer = None
        for path in ordered_paths:
            with path.open("r", newline="", encoding="utf-8") as src:
                reader = csv.reader(src)
                header = next(reader, None)
                if header is None:
                    continue

                if writer is None:
                    writer = csv.writer(dst)
                    writer.writerow(header)

                for row in reader:
                    writer.writerow(row)

        if writer is None:
            raise RuntimeError("No daily aggtrade rows were written to the final artifact")

    return final_path


def fetch_aggtrades_range(
    *,
    symbol: str,
    start_ms: int,
    end_ms: int,
    source: str = "auto",
    max_download_workers: int = 6,
    max_inflight_days: int = 12,
    max_parse_workers: int = 4,
    max_parsed_batches: int = 12,
    parse_chunksize_rows: int = 250_000,
    request_timeout_s: float = 30.0,
    max_retries: int = 3,
    retry_backoff_s: float = 1.0,
    sequential: bool = False,
    skip_bad_days: bool = False,
) -> FetchAggtradesResult:
    """Fetch aggtrades archives across a UTC day range and materialize one ordered CSV artifact.

    By default the function uses a 3-stage producer/consumer pipeline:
    download -> parse/decompress -> persist.
    Each stage is connected with a bounded queue for backpressure so that downloads continue
    while previous batches are parsed/written.

    Args:
        symbol: Binance futures symbol (example: ``BTCUSDT``).
        start_ms: Inclusive UNIX timestamp in milliseconds.
        end_ms: Exclusive UNIX timestamp in milliseconds.
        source: ``auto``/``archive``/``rest``. ``auto`` and ``archive`` use archive mode.
        max_download_workers: Parallel day workers (safe default: 6).
        max_inflight_days: Bounded queue size for day download jobs.
        max_parse_workers: Number of parse/decompression workers.
        max_parsed_batches: Bounded queue size between parse and persist stages.
        parse_chunksize_rows: Rows per parsed chunk written by persist stage.
        request_timeout_s: Per-request timeout in seconds.
        max_retries: Retry attempts after the first request failure.
        retry_backoff_s: Exponential backoff base in seconds.
        sequential: If true, run in sequential mode (debug/regression fallback).
        skip_bad_days: Continue if a single day fails in parse/persist stage.
    """
    if source not in {"auto", "archive", "rest"}:
        raise ValueError("source must be one of: auto, archive, rest")

    if source == "rest":
        raise NotImplementedError("REST aggtrade range fetch is not implemented; use source='archive' or 'auto'")

    if max_download_workers < 1:
        raise ValueError("max_download_workers must be >= 1")
    if max_inflight_days < 1:
        raise ValueError("max_inflight_days must be >= 1")
    if max_parse_workers < 1:
        raise ValueError("max_parse_workers must be >= 1")
    if max_parsed_batches < 1:
        raise ValueError("max_parsed_batches must be >= 1")
    if parse_chunksize_rows < 1:
        raise ValueError("parse_chunksize_rows must be >= 1")

    jobs = _iter_utc_days(start_ms=start_ms, end_ms=end_ms)
    download_job_queue: Queue[_DayJob | None] = Queue(maxsize=max_inflight_days)
    parse_input_queue: Queue[_DownloadedDay | None] = Queue(maxsize=max_inflight_days)
    persist_queue: Queue[_PersistChunk | _PersistDayDone | None] = Queue(maxsize=max_parsed_batches)
    cancel_event = Event()
    first_error_lock = Lock()
    first_error: list[Exception] = []
    skipped_days: dict[int, str] = {}
    day_metrics: dict[int, _DayMetrics] = {day_start_ms: _DayMetrics() for day_start_ms in jobs}
    day_metrics_lock = Lock()
    download_stage_stats = _StageStats()
    parse_stage_stats = _StageStats()
    persist_stage_stats = _StageStats()

    def set_first_error(exc: Exception) -> None:
        with first_error_lock:
            if not first_error:
                first_error.append(exc)
        cancel_event.set()

    def maybe_skip_day(*, day_start_ms: int, stage: str, exc: Exception) -> bool:
        if skip_bad_days and stage in {"parse", "persist"}:
            with first_error_lock:
                skipped_days[day_start_ms] = f"{stage}: {_format_exc_message(exc)}"
            return True
        return False

    def producer() -> None:
        try:
            for day_start_ms in jobs:
                if cancel_event.is_set():
                    break
                download_job_queue.put(_DayJob(day_start_ms=day_start_ms))
        except Exception as exc:  # noqa: BLE001
            set_first_error(exc)
        finally:
            for _ in range(max_download_workers):
                download_job_queue.put(None)

    results: list[_DayResult] = []
    results_lock = Lock()

    def download_worker() -> None:
        with httpx.Client() as client:
            while True:
                job = download_job_queue.get()
                try:
                    if job is None:
                        return

                    if cancel_event.is_set():
                        continue

                    payload = download_day_archive(
                        client=client,
                        symbol=symbol,
                        day_start_ms=job.day_start_ms,
                        request_timeout_s=request_timeout_s,
                        max_retries=max_retries,
                        retry_backoff_s=retry_backoff_s,
                    )
                    with day_metrics_lock:
                        metric = day_metrics[job.day_start_ms]
                        metric.download_seconds += payload.elapsed_s
                        metric.archive_bytes += len(payload.archive_bytes)
                        metric.retries += payload.retries
                    parse_input_queue.put(
                        _DownloadedDay(day_start_ms=job.day_start_ms, archive_bytes=payload.archive_bytes)
                    )
                except Exception as exc:  # noqa: BLE001
                    set_first_error(exc)
                finally:
                    download_job_queue.task_done()

    def parse_worker() -> None:
        while True:
            downloaded = parse_input_queue.get()
            try:
                if downloaded is None:
                    return
                if cancel_event.is_set():
                    continue

                parse_started = time.perf_counter()
                parsed_rows = 0
                for chunk in parse_decompress_day(
                    archive_bytes=downloaded.archive_bytes,
                    chunksize_rows=parse_chunksize_rows,
                ):
                    parsed_rows += int(len(chunk.index))
                    persist_queue.put(_PersistChunk(day_start_ms=downloaded.day_start_ms, frame=chunk))
                persist_queue.put(_PersistDayDone(day_start_ms=downloaded.day_start_ms))
                parse_elapsed = time.perf_counter() - parse_started
                with day_metrics_lock:
                    metric = day_metrics[downloaded.day_start_ms]
                    metric.parse_seconds += parse_elapsed
                    metric.parsed_rows += parsed_rows
            except Exception as exc:  # noqa: BLE001
                set_first_error(exc)
            finally:
                parse_input_queue.task_done()

    def persist_worker() -> None:
        day_output_paths: dict[int, Path] = {}
        day_temp_paths: dict[int, Path] = {}
        day_has_rows: dict[int, bool] = {}

        while True:
            item = persist_queue.get()
            try:
                if item is None:
                    return
                if cancel_event.is_set():
                    continue

                if isinstance(item, _PersistChunk):
                    output_path = day_output_paths.get(item.day_start_ms)
                    temp_path = day_temp_paths.get(item.day_start_ms)
                    if output_path is None or temp_path is None:
                        output_path = _build_day_csv_path(symbol=symbol, day_start_ms=item.day_start_ms)
                        temp_name = f"{output_path.name}.tmp"
                        temp_path = output_path.with_name(temp_name)
                        if temp_path.exists():
                            temp_path.unlink()
                        day_output_paths[item.day_start_ms] = output_path
                        day_temp_paths[item.day_start_ms] = temp_path
                        day_has_rows[item.day_start_ms] = False

                    persist_day(
                        frame=item.frame,
                        output_path=temp_path,
                        write_header=(not day_has_rows[item.day_start_ms]),
                    )
                    rows = int(len(item.frame.index))
                    with day_metrics_lock:
                        metric = day_metrics[item.day_start_ms]
                        metric.persisted_rows += rows
                    day_has_rows[item.day_start_ms] = True
                    continue

                if isinstance(item, _PersistDayDone):
                    day_start_ms = item.day_start_ms
                    temp_path = day_temp_paths.get(day_start_ms)
                    output_path = day_output_paths.get(day_start_ms)
                    if temp_path is None or output_path is None or not day_has_rows.get(day_start_ms, False):
                        raise RuntimeError(f"No rows parsed for day {day_start_ms}")
                    temp_path.replace(output_path)
                    with results_lock:
                        results.append(_DayResult(day_start_ms=day_start_ms, output_path=output_path))
                    continue
            except Exception as exc:  # noqa: BLE001
                set_first_error(exc)
            finally:
                persist_queue.task_done()

    def run_sequential() -> None:
        with httpx.Client() as client:
            for day_start_ms in jobs:
                try:
                    payload = download_day_archive(
                        client=client,
                        symbol=symbol,
                        day_start_ms=day_start_ms,
                        request_timeout_s=request_timeout_s,
                        max_retries=max_retries,
                        retry_backoff_s=retry_backoff_s,
                    )
                except Exception as exc:  # noqa: BLE001
                    raise _stage_error(
                        stage="download",
                        symbol=symbol,
                        day_start_ms=day_start_ms,
                        details=_format_exc_message(exc),
                    ) from exc
                with day_metrics_lock:
                    metric = day_metrics[day_start_ms]
                    metric.download_seconds += payload.elapsed_s
                    metric.archive_bytes += len(payload.archive_bytes)
                    metric.retries += payload.retries
                output_path = _build_day_csv_path(symbol=symbol, day_start_ms=day_start_ms)
                temp_path = output_path.with_name(f"{output_path.name}.tmp-sequential")
                if temp_path.exists():
                    temp_path.unlink()
                wrote_any = False
                parse_started = time.perf_counter()
                try:
                    for chunk in parse_decompress_day(
                        archive_bytes=payload.archive_bytes,
                        chunksize_rows=parse_chunksize_rows,
                    ):
                        persist_day(frame=chunk, output_path=temp_path, write_header=not wrote_any)
                        rows = int(len(chunk.index))
                        with day_metrics_lock:
                            metric = day_metrics[day_start_ms]
                            metric.persisted_rows += rows
                        wrote_any = True
                    if not wrote_any:
                        raise RuntimeError(f"No rows parsed for day {_day_label(day_start_ms)}")
                except Exception as exc:  # noqa: BLE001
                    wrapped = _stage_error(
                        stage="parse",
                        symbol=symbol,
                        day_start_ms=day_start_ms,
                        details=_format_exc_message(exc),
                    )
                    if temp_path.exists():
                        temp_path.unlink()
                    if maybe_skip_day(day_start_ms=day_start_ms, stage="parse", exc=wrapped):
                        continue
                    raise wrapped from exc
                parse_elapsed = time.perf_counter() - parse_started
                with day_metrics_lock:
                    metric = day_metrics[day_start_ms]
                    metric.parse_seconds += parse_elapsed
                    metric.write_seconds += parse_elapsed
                    metric.parsed_rows = metric.persisted_rows
                temp_path.replace(output_path)
                results.append(_DayResult(day_start_ms=day_start_ms, output_path=output_path))

    if sequential:
        run_sequential()
    else:
        write_elapsed_lock = Lock()
        producer_thread = Thread(target=producer, daemon=True)
        producer_thread.start()

        def timed_download_worker() -> None:
            worker_started = time.perf_counter()
            with httpx.Client() as client:
                while True:
                    wait_started = time.perf_counter()
                    job = download_job_queue.get()
                    waited = time.perf_counter() - wait_started
                    with first_error_lock:
                        download_stage_stats.wait_s += waited
                        download_stage_stats.gets += 1
                    if job is None:
                        download_job_queue.task_done()
                        break
                    active_started = time.perf_counter()
                    try:
                        if cancel_event.is_set():
                            continue
                        payload = download_day_archive(
                            client=client,
                            symbol=symbol,
                            day_start_ms=job.day_start_ms,
                            request_timeout_s=request_timeout_s,
                            max_retries=max_retries,
                            retry_backoff_s=retry_backoff_s,
                        )
                        with day_metrics_lock:
                            metric = day_metrics[job.day_start_ms]
                            metric.download_seconds += payload.elapsed_s
                            metric.archive_bytes += len(payload.archive_bytes)
                            metric.retries += payload.retries
                        parse_input_queue.put(
                            _DownloadedDay(day_start_ms=job.day_start_ms, archive_bytes=payload.archive_bytes)
                        )
                    except Exception as exc:  # noqa: BLE001
                        wrapped = _stage_error(
                            stage="download",
                            symbol=symbol,
                            day_start_ms=job.day_start_ms,
                            details=_format_exc_message(exc),
                        )
                        set_first_error(wrapped)
                    finally:
                        elapsed = time.perf_counter() - active_started
                        with first_error_lock:
                            download_stage_stats.active_s += elapsed
                        download_job_queue.task_done()
            with first_error_lock:
                download_stage_stats.wall_s += time.perf_counter() - worker_started

        download_threads = [Thread(target=timed_download_worker, daemon=True) for _ in range(max_download_workers)]
        for thread in download_threads:
            thread.start()

        def timed_parse_worker() -> None:
            worker_started = time.perf_counter()
            while True:
                wait_started = time.perf_counter()
                downloaded = parse_input_queue.get()
                waited = time.perf_counter() - wait_started
                with first_error_lock:
                    parse_stage_stats.wait_s += waited
                    parse_stage_stats.gets += 1
                if downloaded is None:
                    parse_input_queue.task_done()
                    break
                active_started = time.perf_counter()
                try:
                    if cancel_event.is_set():
                        continue
                    parse_started = time.perf_counter()
                    parsed_rows = 0
                    for chunk in parse_decompress_day(
                        archive_bytes=downloaded.archive_bytes,
                        chunksize_rows=parse_chunksize_rows,
                    ):
                        parsed_rows += int(len(chunk.index))
                        persist_queue.put(_PersistChunk(day_start_ms=downloaded.day_start_ms, frame=chunk))
                    if parsed_rows == 0:
                        raise RuntimeError(f"No rows parsed for day {_day_label(downloaded.day_start_ms)}")
                    persist_queue.put(_PersistDayDone(day_start_ms=downloaded.day_start_ms))
                    parse_elapsed = time.perf_counter() - parse_started
                    with day_metrics_lock:
                        metric = day_metrics[downloaded.day_start_ms]
                        metric.parse_seconds += parse_elapsed
                        metric.parsed_rows += parsed_rows
                except Exception as exc:  # noqa: BLE001
                    wrapped = _stage_error(
                        stage="parse",
                        symbol=symbol,
                        day_start_ms=downloaded.day_start_ms,
                        details=_format_exc_message(exc),
                    )
                    if maybe_skip_day(day_start_ms=downloaded.day_start_ms, stage="parse", exc=wrapped):
                        continue
                    set_first_error(wrapped)
                finally:
                    elapsed = time.perf_counter() - active_started
                    with first_error_lock:
                        parse_stage_stats.active_s += elapsed
                    parse_input_queue.task_done()
            with first_error_lock:
                parse_stage_stats.wall_s += time.perf_counter() - worker_started

        parse_threads = [Thread(target=timed_parse_worker, daemon=True) for _ in range(max_parse_workers)]
        for thread in parse_threads:
            thread.start()

        def timed_persist_worker() -> None:
            worker_started = time.perf_counter()
            day_output_paths: dict[int, Path] = {}
            day_temp_paths: dict[int, Path] = {}
            day_has_rows: dict[int, bool] = {}
            while True:
                wait_started = time.perf_counter()
                item = persist_queue.get()
                waited = time.perf_counter() - wait_started
                with write_elapsed_lock:
                    persist_stage_stats.wait_s += waited
                    persist_stage_stats.gets += 1
                if item is None:
                    persist_queue.task_done()
                    break
                active_started = time.perf_counter()
                try:
                    if cancel_event.is_set():
                        continue
                    if isinstance(item, _PersistChunk):
                        output_path = day_output_paths.get(item.day_start_ms)
                        temp_path = day_temp_paths.get(item.day_start_ms)
                        if output_path is None or temp_path is None:
                            output_path = _build_day_csv_path(symbol=symbol, day_start_ms=item.day_start_ms)
                            temp_name = f"{output_path.name}.tmp"
                            temp_path = output_path.with_name(temp_name)
                            if temp_path.exists():
                                temp_path.unlink()
                            day_output_paths[item.day_start_ms] = output_path
                            day_temp_paths[item.day_start_ms] = temp_path
                            day_has_rows[item.day_start_ms] = False
                        write_started = time.perf_counter()
                        persist_day(
                            frame=item.frame,
                            output_path=temp_path,
                            write_header=(not day_has_rows[item.day_start_ms]),
                        )
                        write_elapsed = time.perf_counter() - write_started
                        rows = int(len(item.frame.index))
                        with day_metrics_lock:
                            metric = day_metrics[item.day_start_ms]
                            metric.write_seconds += write_elapsed
                            metric.persisted_rows += rows
                        day_has_rows[item.day_start_ms] = True
                        continue
                    if isinstance(item, _PersistDayDone):
                        day_start_ms = item.day_start_ms
                        temp_path = day_temp_paths.get(day_start_ms)
                        output_path = day_output_paths.get(day_start_ms)
                        if temp_path is None or output_path is None or not day_has_rows.get(day_start_ms, False):
                            raise RuntimeError(f"No rows parsed for day {day_start_ms}")
                        temp_path.replace(output_path)
                        with results_lock:
                            results.append(_DayResult(day_start_ms=day_start_ms, output_path=output_path))
                except Exception as exc:  # noqa: BLE001
                    day_start_ms = item.day_start_ms if isinstance(item, (_PersistChunk, _PersistDayDone)) else -1
                    if day_start_ms >= 0:
                        wrapped = _stage_error(
                            stage="persist",
                            symbol=symbol,
                            day_start_ms=day_start_ms,
                            details=_format_exc_message(exc),
                        )
                        temp_path = day_temp_paths.get(day_start_ms)
                        if temp_path is not None and temp_path.exists():
                            temp_path.unlink()
                        day_has_rows.pop(day_start_ms, None)
                        day_output_paths.pop(day_start_ms, None)
                        day_temp_paths.pop(day_start_ms, None)
                        if maybe_skip_day(day_start_ms=day_start_ms, stage="persist", exc=wrapped):
                            continue
                        set_first_error(wrapped)
                    else:
                        set_first_error(exc)
                finally:
                    elapsed = time.perf_counter() - active_started
                    with write_elapsed_lock:
                        persist_stage_stats.active_s += elapsed
                    persist_queue.task_done()
            with write_elapsed_lock:
                persist_stage_stats.wall_s += time.perf_counter() - worker_started

        persist_thread = Thread(target=timed_persist_worker, daemon=True)
        persist_thread.start()

        download_job_queue.join()
        producer_thread.join()
        for thread in download_threads:
            thread.join()

        for _ in range(max_parse_workers):
            parse_input_queue.put(None)
        parse_input_queue.join()
        for thread in parse_threads:
            thread.join()

        persist_queue.put(None)
        persist_queue.join()
        persist_thread.join()

    if first_error:
        raise RuntimeError("Aggtrades archive fetch failed") from first_error[0]

    skipped_day_keys = set(skipped_days)
    ordered_paths = [
        result.output_path
        for result in sorted(results, key=lambda item: item.day_start_ms)
        if result.day_start_ms not in skipped_day_keys
    ]
    if not ordered_paths:
        if first_error:
            raise RuntimeError("Aggtrades archive fetch failed") from first_error[0]
        raise RuntimeError("Aggtrades archive fetch failed") from RuntimeError(
            "All requested days were skipped due to malformed archive data"
        )

    csv_path = _concatenate_daily_files(
        symbol=symbol,
        start_ms=start_ms,
        end_ms=end_ms,
        ordered_paths=ordered_paths,
    )
    run_id = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
    metrics_dir = ARTIFACTS_ROOT / "fetch_reports"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = metrics_dir / f"{run_id}.json"
    total_download_s = sum(item.download_seconds for item in day_metrics.values())
    total_parse_s = sum(item.parse_seconds for item in day_metrics.values())
    total_write_s = sum(item.write_seconds for item in day_metrics.values())
    total_bytes = sum(item.archive_bytes for item in day_metrics.values())
    total_rows_parsed = sum(item.parsed_rows for item in day_metrics.values())
    total_rows_persisted = sum(item.persisted_rows for item in day_metrics.values())
    total_retries = sum(item.retries for item in day_metrics.values())

    def _util(stats: _StageStats) -> float:
        return (stats.active_s / stats.wall_s) if stats.wall_s > 0 else 0.0

    report = {
        "run_id": run_id,
        "symbol": symbol,
        "start_ms": start_ms,
        "end_ms": end_ms,
        "sequential": sequential,
        "pipeline": {
            "max_download_workers": max_download_workers,
            "max_inflight_days": max_inflight_days,
            "max_parse_workers": max_parse_workers,
            "max_parsed_batches": max_parsed_batches,
            "parse_chunksize_rows": parse_chunksize_rows,
        },
        "aggregate": {
            "download_bytes": total_bytes,
            "download_seconds": total_download_s,
            "parse_seconds": total_parse_s,
            "write_seconds": total_write_s,
            "total_rows_parsed": total_rows_parsed,
            "total_rows_persisted": total_rows_persisted,
            "total_retries": total_retries,
            "effective_download_mb_s": (total_bytes / 1_000_000) / total_download_s if total_download_s > 0 else 0.0,
            "parse_rows_s": total_rows_parsed / total_parse_s if total_parse_s > 0 else 0.0,
            "persist_rows_s": total_rows_persisted / total_write_s if total_write_s > 0 else 0.0,
            "worker_utilization": {
                "download": _util(download_stage_stats),
                "parse": _util(parse_stage_stats),
                "persist": _util(persist_stage_stats),
            },
            "queue_wait_seconds": {
                "download_job_queue": download_stage_stats.wait_s,
                "parse_input_queue": parse_stage_stats.wait_s,
                "persist_queue": persist_stage_stats.wait_s,
            },
            "queue_wait_avg_ms": {
                "download_job_queue": (download_stage_stats.wait_s / download_stage_stats.gets) * 1000
                if download_stage_stats.gets > 0
                else 0.0,
                "parse_input_queue": (parse_stage_stats.wait_s / parse_stage_stats.gets) * 1000
                if parse_stage_stats.gets > 0
                else 0.0,
                "persist_queue": (persist_stage_stats.wait_s / persist_stage_stats.gets) * 1000
                if persist_stage_stats.gets > 0
                else 0.0,
            },
        },
        "days": [
            {
                "day_start_ms": day_start_ms,
                "download_seconds": metrics.download_seconds,
                "parse_seconds": metrics.parse_seconds,
                "write_seconds": metrics.write_seconds,
                "archive_bytes": metrics.archive_bytes,
                "parsed_rows": metrics.parsed_rows,
                "persisted_rows": metrics.persisted_rows,
                "retries": metrics.retries,
            }
            for day_start_ms, metrics in sorted(day_metrics.items())
        ],
        "skipped_days": [
            {
                "day_start_ms": day_start_ms,
                "day": _day_label(day_start_ms),
                "reason": reason,
            }
            for day_start_ms, reason in sorted(skipped_days.items())
        ],
        "output_csv_path": str(csv_path),
    }
    metrics_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    summary = (
        f"download={report['aggregate']['effective_download_mb_s']:.2f} MB/s, "
        f"parse={report['aggregate']['parse_rows_s']:.0f} rows/s, "
        f"persist={report['aggregate']['persist_rows_s']:.0f} rows/s, "
        f"retries={total_retries}, "
        f"skipped_days={len(skipped_days)}, "
        f"util(d/p/w)="
        f"{report['aggregate']['worker_utilization']['download']:.2f}/"
        f"{report['aggregate']['worker_utilization']['parse']:.2f}/"
        f"{report['aggregate']['worker_utilization']['persist']:.2f}"
    )
    return FetchAggtradesResult(
        csv_path=csv_path,
        metrics_path=metrics_path,
        run_id=run_id,
        summary=summary,
    )
