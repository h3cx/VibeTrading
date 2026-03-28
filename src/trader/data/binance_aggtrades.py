from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from io import BytesIO
from pathlib import Path
from queue import Queue
from threading import Event, Lock, Thread
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


def _download_archive_day(
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


def _decompress_day_to_csv(*, archive_bytes: bytes, output_path: Path) -> None:
    with zipfile.ZipFile(BytesIO(archive_bytes)) as zf:
        members = [name for name in zf.namelist() if name.lower().endswith(".csv")]
        if not members:
            raise RuntimeError(f"Zip archive has no CSV payload: {output_path}")

        with zf.open(members[0], "r") as zipped:
            frame = pd.read_csv(
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
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)


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
    request_timeout_s: float = 30.0,
    max_retries: int = 3,
    retry_backoff_s: float = 1.0,
) -> Path:
    """Fetch aggtrades archives across a UTC day range and materialize one ordered CSV artifact.

    The function uses a producer/consumer pipeline where the producer enqueues one UTC day
    per job into a bounded queue, and a ThreadPoolExecutor worker pool downloads/decompresses
    jobs concurrently. Memory usage scales with worker count because each worker can hold one
    zip payload + decoded dataframe at a time before flushing to per-day CSV. In practice,
    expect roughly 40-120 MB extra peak RAM per additional worker (symbol/day dependent).

    Args:
        symbol: Binance futures symbol (example: ``BTCUSDT``).
        start_ms: Inclusive UNIX timestamp in milliseconds.
        end_ms: Exclusive UNIX timestamp in milliseconds.
        source: ``auto``/``archive``/``rest``. ``auto`` and ``archive`` use archive mode.
        max_download_workers: Parallel day workers (safe default: 6).
        max_inflight_days: Bounded producer queue size for backpressure.
        request_timeout_s: Per-request timeout in seconds.
        max_retries: Retry attempts after the first request failure.
        retry_backoff_s: Exponential backoff base in seconds.
    """
    if source not in {"auto", "archive", "rest"}:
        raise ValueError("source must be one of: auto, archive, rest")

    if source == "rest":
        raise NotImplementedError("REST aggtrade range fetch is not implemented; use source='archive' or 'auto'")

    if max_download_workers < 1:
        raise ValueError("max_download_workers must be >= 1")
    if max_inflight_days < 1:
        raise ValueError("max_inflight_days must be >= 1")

    jobs = _iter_utc_days(start_ms=start_ms, end_ms=end_ms)

    queue: Queue[_DayJob | None] = Queue(maxsize=max_inflight_days)
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
                queue.put(_DayJob(day_start_ms=day_start_ms))
        except Exception as exc:  # noqa: BLE001
            set_first_error(exc)
        finally:
            for _ in range(max_download_workers):
                queue.put(None)

    results: list[_DayResult] = []
    results_lock = Lock()

    def worker() -> None:
        with httpx.Client() as client:
            while True:
                job = queue.get()
                try:
                    if job is None:
                        return

                    if cancel_event.is_set():
                        continue

                    output_path = _build_day_csv_path(symbol=symbol, day_start_ms=job.day_start_ms)
                    payload = _download_archive_day(
                        client=client,
                        symbol=symbol,
                        day_start_ms=job.day_start_ms,
                        request_timeout_s=request_timeout_s,
                        max_retries=max_retries,
                        retry_backoff_s=retry_backoff_s,
                    )
                    _decompress_day_to_csv(archive_bytes=payload, output_path=output_path)

                    with results_lock:
                        results.append(_DayResult(day_start_ms=job.day_start_ms, output_path=output_path))
                except Exception as exc:  # noqa: BLE001
                    set_first_error(exc)
                finally:
                    queue.task_done()

    producer_thread = Thread(target=producer, daemon=True)
    producer_thread.start()

    with ThreadPoolExecutor(max_workers=max_download_workers) as pool:
        futures = [pool.submit(worker) for _ in range(max_download_workers)]

        queue.join()
        producer_thread.join()

        for fut in futures:
            fut.result()

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
