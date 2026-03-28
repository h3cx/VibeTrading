from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from io import BytesIO
from pathlib import Path
from queue import Queue
from threading import Event, Lock, Thread
from typing import Iterator
import csv
import time
import zipfile

import httpx
import pandas as pd

from trader.data.storage import raw_data_dir


BINANCE_ARCHIVE_BASE_URL = "https://data.binance.vision"


@dataclass(frozen=True)
class _DayJob:
    day_start_ms: int


@dataclass(frozen=True)
class _DayResult:
    day_start_ms: int
    output_path: Path


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
) -> bytes:
    url = _build_archive_url(symbol=symbol, day_start_ms=day_start_ms)

    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            response = client.get(url, timeout=request_timeout_s)
            response.raise_for_status()
            return response.content
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
                names=[
                    "agg_trade_id",
                    "price",
                    "quantity",
                    "first_trade_id",
                    "last_trade_id",
                    "timestamp",
                    "buyer_is_maker",
                ],
                dtype={
                    "agg_trade_id": "int64",
                    "price": "float64",
                    "quantity": "float64",
                    "first_trade_id": "int64",
                    "last_trade_id": "int64",
                    "timestamp": "int64",
                    "buyer_is_maker": "bool",
                },
                low_memory=False,
                chunksize=chunksize_rows,
            )
            for chunk in frame_iter:
                yield chunk


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
) -> Path:
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

    def set_first_error(exc: Exception) -> None:
        with first_error_lock:
            if not first_error:
                first_error.append(exc)
        cancel_event.set()

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
                    parse_input_queue.put(
                        _DownloadedDay(day_start_ms=job.day_start_ms, archive_bytes=payload)
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

                for chunk in parse_decompress_day(
                    archive_bytes=downloaded.archive_bytes,
                    chunksize_rows=parse_chunksize_rows,
                ):
                    persist_queue.put(_PersistChunk(day_start_ms=downloaded.day_start_ms, frame=chunk))
                persist_queue.put(_PersistDayDone(day_start_ms=downloaded.day_start_ms))
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
                payload = download_day_archive(
                    client=client,
                    symbol=symbol,
                    day_start_ms=day_start_ms,
                    request_timeout_s=request_timeout_s,
                    max_retries=max_retries,
                    retry_backoff_s=retry_backoff_s,
                )
                output_path = _build_day_csv_path(symbol=symbol, day_start_ms=day_start_ms)
                temp_path = output_path.with_name(f"{output_path.name}.tmp-sequential")
                if temp_path.exists():
                    temp_path.unlink()
                wrote_any = False
                for chunk in parse_decompress_day(archive_bytes=payload, chunksize_rows=parse_chunksize_rows):
                    persist_day(frame=chunk, output_path=temp_path, write_header=not wrote_any)
                    wrote_any = True
                if not wrote_any:
                    raise RuntimeError(f"No rows parsed for day {day_start_ms}")
                temp_path.replace(output_path)
                results.append(_DayResult(day_start_ms=day_start_ms, output_path=output_path))

    if sequential:
        run_sequential()
    else:
        producer_thread = Thread(target=producer, daemon=True)
        producer_thread.start()

        download_threads = [Thread(target=download_worker, daemon=True) for _ in range(max_download_workers)]
        for thread in download_threads:
            thread.start()

        parse_threads = [Thread(target=parse_worker, daemon=True) for _ in range(max_parse_workers)]
        for thread in parse_threads:
            thread.start()

        persist_thread = Thread(target=persist_worker, daemon=True)
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

    ordered_paths = [
        result.output_path
        for result in sorted(results, key=lambda item: item.day_start_ms)
    ]

    return _concatenate_daily_files(
        symbol=symbol,
        start_ms=start_ms,
        end_ms=end_ms,
        ordered_paths=ordered_paths,
    )
