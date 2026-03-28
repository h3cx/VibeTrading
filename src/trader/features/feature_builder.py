from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator
import csv
import math
import re

import numpy as np
import pandas as pd
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from trader.data.registry import (
    build_dataset_id,
    build_source_entries,
    summarize_csv_artifact,
    timestamp_ms_to_iso,
    write_dataset_manifest,
)
from trader.data.storage import features_dataset_dir, write_tag

console = Console()

STAMP_RE = re.compile(r"(\d{8}_\d{6})__(\d{8}_\d{6})$")
AGG_CHUNK_ROWS = 500_000
KLINE_CHUNK_ROWS = 200_000
EPS = 1e-12


def _parse_timeframe_to_seconds(value: str) -> int:
    cleaned = value.strip().lower()
    if not cleaned:
        raise ValueError("timeframe cannot be empty")

    match = re.fullmatch(r"(\d+)([smh])", cleaned)
    if match is None:
        raise ValueError("timeframe must match '<int>s', '<int>m', or '<int>h' (example: 5m)")

    qty = int(match.group(1))
    unit = match.group(2)
    if qty <= 0:
        raise ValueError("timeframe quantity must be greater than 0")

    factor = {"s": 1, "m": 60, "h": 3600}[unit]
    return qty * factor


def _utc_stamp(ms: int) -> str:
    dt = datetime.fromtimestamp(ms / 1000, tz=timezone.utc)
    return dt.strftime("%Y%m%d_%H%M%S")


def _stamp_to_ms(value: str) -> int:
    dt = datetime.strptime(value, "%Y%m%d_%H%M%S").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def _extract_range_from_filename(path: Path) -> tuple[int, int] | None:
    match = STAMP_RE.search(path.stem)
    if match is None:
        return None

    start_ms = _stamp_to_ms(match.group(1))
    end_ms = _stamp_to_ms(match.group(2))
    return start_ms, end_ms


def _select_overlapping_files(directory: Path, start_ms: int, end_ms: int) -> list[Path]:
    if not directory.exists():
        return []

    selected: list[Path] = []

    for path in sorted(directory.glob("*.csv")):
        parsed = _extract_range_from_filename(path)

        # Legacy fallback such as "1m.csv"
        if parsed is None:
            selected.append(path)
            continue

        file_start_ms, file_end_ms = parsed
        overlaps = file_start_ms < end_ms and start_ms < file_end_ms
        if overlaps:
            selected.append(path)

    return selected


def _build_output_path(symbol: str, dataset_id: str) -> Path:
    out_dir = features_dataset_dir(symbol=symbol, dataset_id=dataset_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / "features.csv"


def _open_output_csv(
    symbol: str,
    dataset_id: str,
    include_kline_context: bool,
) -> tuple[Path, Path, Any, csv.writer]:
    final_path = _build_output_path(symbol=symbol, dataset_id=dataset_id)
    part_path = final_path.with_suffix(final_path.suffix + ".part")

    handle = part_path.open("w", newline="", encoding="utf-8")
    writer = csv.writer(handle)

    header = [
        "timestamp",
        "timestamp_ms",
        "timestamp_s",
        "timeframe_s",
        "open",
        "high",
        "low",
        "close",
        "vwap_1s",
        "trade_count_1s",
        "trade_count_5s",
        "volume_1s",
        "notional_volume_1s",
        "notional_volume_5s",
        "buy_volume_1s",
        "sell_volume_1s",
        "signed_volume_1s",
        "signed_volume_5s",
        "trade_imbalance_1s",
        "buy_sell_ratio_1s",
        "avg_trade_size_1s",
        "avg_trade_size_5s",
        "max_trade_size_1s",
        "price_range_1s",
        "ret_1s",
        "ret_3s",
        "ret_5s",
        "ret_10s",
        "volatility_10s",
        "volatility_30s",
        "volume_zscore_30s",
    ]

    if include_kline_context:
        header.extend(
            [
                "dist_from_1m_open",
                "dist_from_1m_high",
                "dist_from_1m_low",
                "dist_from_1m_close",
            ]
        )

    writer.writerow(header)
    return final_path, part_path, handle, writer


def _finalize_output_csv(final_path: Path, part_path: Path, handle: Any) -> Path:
    handle.flush()
    handle.close()
    part_path.replace(final_path)
    return final_path


def _cleanup_part_file(part_path: Path, handle: Any | None = None) -> None:
    if handle is not None and not handle.closed:
        handle.close()
    if part_path.exists():
        part_path.unlink()


def _bool_series_to_numpy(series: pd.Series) -> np.ndarray:
    mapped = (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map({"true": True, "false": False, "1": True, "0": False})
        .fillna(False)
    )
    return mapped.to_numpy(dtype=bool)


def _agg_row_from_grouped(grouped_row: Any) -> dict[str, float | int]:
    return {
        "second_s": int(grouped_row.second_s),
        "open": float(grouped_row.open),
        "high": float(grouped_row.high),
        "low": float(grouped_row.low),
        "close": float(grouped_row.close),
        "trade_count_1s": int(grouped_row.trade_count_1s),
        "volume_1s": float(grouped_row.volume_1s),
        "buy_volume_1s": float(grouped_row.buy_volume_1s),
        "sell_volume_1s": float(grouped_row.sell_volume_1s),
        "notional_volume_1s": float(grouped_row.notional_volume_1s),
        "buy_notional_1s": float(grouped_row.buy_notional_1s),
        "sell_notional_1s": float(grouped_row.sell_notional_1s),
        "max_trade_size_1s": float(grouped_row.max_trade_size_1s),
    }


def _merge_agg_rows(
    left: dict[str, float | int],
    right: dict[str, float | int],
) -> dict[str, float | int]:
    if int(left["second_s"]) != int(right["second_s"]):
        raise ValueError("Cannot merge aggregate rows from different seconds")

    return {
        "second_s": int(left["second_s"]),
        "open": float(left["open"]),
        "high": max(float(left["high"]), float(right["high"])),
        "low": min(float(left["low"]), float(right["low"])),
        "close": float(right["close"]),
        "trade_count_1s": int(left["trade_count_1s"]) + int(right["trade_count_1s"]),
        "volume_1s": float(left["volume_1s"]) + float(right["volume_1s"]),
        "buy_volume_1s": float(left["buy_volume_1s"]) + float(right["buy_volume_1s"]),
        "sell_volume_1s": float(left["sell_volume_1s"]) + float(right["sell_volume_1s"]),
        "notional_volume_1s": float(left["notional_volume_1s"]) + float(right["notional_volume_1s"]),
        "buy_notional_1s": float(left["buy_notional_1s"]) + float(right["buy_notional_1s"]),
        "sell_notional_1s": float(left["sell_notional_1s"]) + float(right["sell_notional_1s"]),
        "max_trade_size_1s": max(float(left["max_trade_size_1s"]), float(right["max_trade_size_1s"])),
    }


def _iter_aggtrade_second_aggregates(
    symbol: str,
    start_ms: int,
    end_ms: int,
    paths: list[Path] | None = None,
) -> Iterator[dict[str, float | int]]:
    agg_dir = Path("data/raw/binance") / symbol / "aggtrades"
    paths = paths if paths is not None else _select_overlapping_files(agg_dir, start_ms, end_ms)

    if not paths:
        raise FileNotFoundError(f"No aggtrades CSV files found for {symbol} in {agg_dir}")

    pending: dict[str, float | int] | None = None

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task_id = progress.add_task(f"Streaming aggtrades for {symbol}", total=len(paths))

        for path in paths:
            reader = pd.read_csv(
                path,
                usecols=[
                    "agg_trade_id",
                    "price",
                    "quantity",
                    "timestamp",
                    "buyer_is_maker",
                ],
                dtype={
                    "agg_trade_id": "int64",
                    "price": "float64",
                    "quantity": "float64",
                    "timestamp": "int64",
                    "buyer_is_maker": "string",
                },
                chunksize=AGG_CHUNK_ROWS,
                low_memory=False,
            )

            for chunk in reader:
                mask = (chunk["timestamp"] >= start_ms) & (chunk["timestamp"] < end_ms)
                if not bool(mask.any()):
                    continue

                filtered = chunk.loc[mask, ["agg_trade_id", "price", "quantity", "timestamp", "buyer_is_maker"]].copy()
                if filtered.empty:
                    continue

                filtered = filtered.sort_values(by=["timestamp", "agg_trade_id"])

                buyer_is_maker = _bool_series_to_numpy(filtered["buyer_is_maker"])
                quantity = filtered["quantity"].to_numpy(dtype=np.float64)
                price = filtered["price"].to_numpy(dtype=np.float64)
                notional = price * quantity

                filtered["second_s"] = filtered["timestamp"] // 1000
                filtered["buy_volume"] = np.where(~buyer_is_maker, quantity, 0.0)
                filtered["sell_volume"] = np.where(buyer_is_maker, quantity, 0.0)
                filtered["notional"] = notional
                filtered["buy_notional"] = np.where(~buyer_is_maker, notional, 0.0)
                filtered["sell_notional"] = np.where(buyer_is_maker, notional, 0.0)

                grouped = (
                    filtered.groupby("second_s", sort=True)
                    .agg(
                        open=("price", "first"),
                        high=("price", "max"),
                        low=("price", "min"),
                        close=("price", "last"),
                        trade_count_1s=("agg_trade_id", "count"),
                        volume_1s=("quantity", "sum"),
                        buy_volume_1s=("buy_volume", "sum"),
                        sell_volume_1s=("sell_volume", "sum"),
                        notional_volume_1s=("notional", "sum"),
                        buy_notional_1s=("buy_notional", "sum"),
                        sell_notional_1s=("sell_notional", "sum"),
                        max_trade_size_1s=("quantity", "max"),
                    )
                    .reset_index()
                )

                for row in grouped.itertuples(index=False):
                    current = _agg_row_from_grouped(row)

                    if pending is None:
                        pending = current
                    elif int(current["second_s"]) == int(pending["second_s"]):
                        pending = _merge_agg_rows(pending, current)
                    else:
                        yield pending
                        pending = current

            progress.advance(task_id, 1)

    if pending is not None:
        yield pending


def _load_kline_context(
    symbol: str,
    start_ms: int,
    end_ms: int,
    paths: list[Path] | None = None,
) -> tuple[dict[int, tuple[float, float, float, float]], tuple[float, float, float, float] | None]:
    kline_dir = Path("data/raw/binance") / symbol / "klines"
    paths = paths if paths is not None else _select_overlapping_files(kline_dir, start_ms, end_ms)

    if not paths:
        raise FileNotFoundError(f"No kline CSV files found for {symbol} in {kline_dir}")

    minute_start_ms = (start_ms // 60_000) * 60_000
    minute_end_ms = ((end_ms + 59_999) // 60_000) * 60_000

    context: dict[int, tuple[float, float, float, float]] = {}
    first_value: tuple[float, float, float, float] | None = None

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task_id = progress.add_task(f"Loading 1m kline context for {symbol}", total=len(paths))

        for path in paths:
            reader = pd.read_csv(
                path,
                usecols=["open_time", "open", "high", "low", "close"],
                dtype={
                    "open_time": "int64",
                    "open": "float64",
                    "high": "float64",
                    "low": "float64",
                    "close": "float64",
                },
                chunksize=KLINE_CHUNK_ROWS,
                low_memory=False,
            )

            for chunk in reader:
                mask = (chunk["open_time"] >= minute_start_ms) & (chunk["open_time"] < minute_end_ms)
                if not bool(mask.any()):
                    continue

                filtered = chunk.loc[mask, ["open_time", "open", "high", "low", "close"]].copy()
                if filtered.empty:
                    continue

                filtered = filtered.sort_values(by="open_time")
                filtered = filtered.drop_duplicates(subset=["open_time"], keep="last")

                for row in filtered.itertuples(index=False):
                    value = (float(row.open), float(row.high), float(row.low), float(row.close))
                    context[int(row.open_time)] = value
                    if first_value is None:
                        first_value = value

            progress.advance(task_id, 1)

    if not context:
        raise RuntimeError(f"No 1m kline rows found for {symbol} inside the requested range")

    return context, first_value


def _sample_std(values: deque[float]) -> float:
    if len(values) < 2:
        return 0.0
    arr = np.asarray(values, dtype=np.float64)
    return float(arr.std(ddof=1))


def _population_zscore(values: deque[float]) -> float:
    if len(values) < 5:
        return 0.0
    arr = np.asarray(values, dtype=np.float64)
    mean = float(arr.mean())
    std = float(arr.std(ddof=0))
    if std <= 0.0:
        return 0.0
    return float((arr[-1] - mean) / std)


def build_feature_frames(
    symbol: str,
    start_ms: int,
    end_ms: int,
    dataset_name: str = "default",
    timeframe: str = "1s",
    include_kline_context: bool = True,
) -> Path:
    timeframe_s = _parse_timeframe_to_seconds(timeframe)

    if end_ms <= start_ms:
        raise ValueError("end_ms must be greater than start_ms")

    start_second = start_ms // 1000
    end_second = (end_ms - 1) // 1000
    total_seconds = end_second - start_second + 1
    total_bars = math.ceil(total_seconds / timeframe_s)

    kline_context: dict[int, tuple[float, float, float, float]] = {}
    first_kline: tuple[float, float, float, float] | None = None
    kline_source_paths: list[Path] = []
    if include_kline_context:
        kline_source_paths = _select_overlapping_files(
            Path("data/raw/binance") / symbol / "klines",
            start_ms,
            end_ms,
        )
        kline_context, first_kline = _load_kline_context(
            symbol=symbol,
            start_ms=start_ms,
            end_ms=end_ms,
            paths=kline_source_paths,
        )

    agg_iter = _iter_aggtrade_second_aggregates(
        symbol=symbol,
        start_ms=start_ms,
        end_ms=end_ms,
        paths=agg_source_paths,
    )

    first_agg = next(agg_iter, None)
    if first_agg is None:
        raise RuntimeError(f"No aggtrades rows found for {symbol} inside the requested range")

    current_agg = first_agg
    initial_close = float(first_agg["close"])
    last_close = initial_close
    last_kline = first_kline

    close_history: deque[float] = deque(maxlen=10)
    trade_count_5_window: deque[float] = deque(maxlen=5)
    signed_volume_5_window: deque[float] = deque(maxlen=5)
    notional_volume_5_window: deque[float] = deque(maxlen=5)
    avg_trade_size_5_window: deque[float] = deque(maxlen=5)
    ret_10_window: deque[float] = deque(maxlen=10)
    ret_30_window: deque[float] = deque(maxlen=30)
    notional_volume_30_window: deque[float] = deque(maxlen=30)

    source_info = {
        "exchange": "binance",
        "symbol": symbol,
        "timeframe": timeframe,
        "timeframe_s": timeframe_s,
        "start_ms": start_ms,
        "end_ms": end_ms,
        "include_kline_context": include_kline_context,
        "aggtrade_files": [str(path) for path in agg_source_paths],
        "kline_files": [str(path) for path in kline_source_paths],
    }
    params_info = {
        "dataset_name": dataset_name,
        "include_kline_context": include_kline_context,
    }
    dataset_id = build_dataset_id(source=source_info, params=params_info)

    final_path, part_path, handle, writer = _open_output_csv(
        symbol=symbol,
        dataset_id=dataset_id,
        include_kline_context=include_kline_context,
    )

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TextColumn("rows={task.fields[rows]}"),
            console=console,
        ) as progress:
            task_id = progress.add_task(
                f"Building feature frames for {symbol}",
                total=total_bars,
                rows=0,
            )

            bars_written = 0
            for bar_start_second in range(start_second, end_second + 1, timeframe_s):
                bar_end_second = min(bar_start_second + timeframe_s - 1, end_second)

                bucket_rows: list[dict[str, float | int]] = []
                for second_s in range(bar_start_second, bar_end_second + 1):
                    if current_agg is not None and int(current_agg["second_s"]) == second_s:
                        second_row = dict(current_agg)
                        last_close = float(second_row["close"])
                        current_agg = next(agg_iter, None)
                    else:
                        second_row = {
                            "second_s": second_s,
                            "open": last_close,
                            "high": last_close,
                            "low": last_close,
                            "close": last_close,
                            "trade_count_1s": 0,
                            "volume_1s": 0.0,
                            "buy_volume_1s": 0.0,
                            "sell_volume_1s": 0.0,
                            "notional_volume_1s": 0.0,
                            "buy_notional_1s": 0.0,
                            "sell_notional_1s": 0.0,
                            "max_trade_size_1s": 0.0,
                        }
                    bucket_rows.append(second_row)

                if not bucket_rows:
                    continue

                first_row = bucket_rows[0]
                last_row = bucket_rows[-1]

                close_px = float(last_row["close"])
                open_px = float(first_row["open"])
                high_px = max(float(r["high"]) for r in bucket_rows)
                low_px = min(float(r["low"]) for r in bucket_rows)
                trade_count_1s = int(sum(int(r["trade_count_1s"]) for r in bucket_rows))
                volume_1s = float(sum(float(r["volume_1s"]) for r in bucket_rows))
                buy_volume_1s = float(sum(float(r["buy_volume_1s"]) for r in bucket_rows))
                sell_volume_1s = float(sum(float(r["sell_volume_1s"]) for r in bucket_rows))
                notional_volume_1s = float(sum(float(r["notional_volume_1s"]) for r in bucket_rows))
                max_trade_size_1s = float(max(float(r["max_trade_size_1s"]) for r in bucket_rows))

                avg_trade_size_1s = (volume_1s / trade_count_1s) if trade_count_1s > 0 else 0.0
                signed_volume_1s = buy_volume_1s - sell_volume_1s
                trade_imbalance_1s = signed_volume_1s / (buy_volume_1s + sell_volume_1s + EPS)
                buy_sell_ratio_1s = buy_volume_1s / (sell_volume_1s + EPS)
                vwap_1s = (notional_volume_1s / (volume_1s + EPS)) if volume_1s > 0 else close_px

                log_close = math.log(max(close_px, EPS))
                ret_1s = log_close - math.log(max(close_history[-1], EPS)) if len(close_history) >= 1 else 0.0
                ret_3s = log_close - math.log(max(close_history[-3], EPS)) if len(close_history) >= 3 else 0.0
                ret_5s = log_close - math.log(max(close_history[-5], EPS)) if len(close_history) >= 5 else 0.0
                ret_10s = log_close - math.log(max(close_history[-10], EPS)) if len(close_history) >= 10 else 0.0

                close_history.append(close_px)
                trade_count_5_window.append(float(trade_count_1s))
                signed_volume_5_window.append(float(signed_volume_1s))
                notional_volume_5_window.append(float(notional_volume_1s))
                avg_trade_size_5_window.append(float(avg_trade_size_1s))
                ret_10_window.append(float(ret_1s))
                ret_30_window.append(float(ret_1s))
                notional_volume_30_window.append(float(notional_volume_1s))

                trade_count_5s = float(sum(trade_count_5_window))
                signed_volume_5s = float(sum(signed_volume_5_window))
                notional_volume_5s = float(sum(notional_volume_5_window))
                avg_trade_size_5s = (
                    float(sum(avg_trade_size_5_window)) / len(avg_trade_size_5_window)
                    if avg_trade_size_5_window
                    else 0.0
                )
                volatility_10s = _sample_std(ret_10_window)
                volatility_30s = _sample_std(ret_30_window)
                volume_zscore_30s = _population_zscore(notional_volume_30_window)
                price_range_1s = (high_px - low_px) / max(close_px, EPS)

                timestamp_ms = bar_start_second * 1000
                timestamp = datetime.fromtimestamp(bar_start_second, tz=timezone.utc).isoformat(sep=" ")

                row = [
                    timestamp,
                    timestamp_ms,
                    bar_start_second,
                    timeframe_s,
                    open_px,
                    high_px,
                    low_px,
                    close_px,
                    vwap_1s,
                    float(trade_count_1s),
                    trade_count_5s,
                    volume_1s,
                    notional_volume_1s,
                    notional_volume_5s,
                    buy_volume_1s,
                    sell_volume_1s,
                    signed_volume_1s,
                    signed_volume_5s,
                    trade_imbalance_1s,
                    buy_sell_ratio_1s,
                    avg_trade_size_1s,
                    avg_trade_size_5s,
                    max_trade_size_1s,
                    price_range_1s,
                    ret_1s,
                    ret_3s,
                    ret_5s,
                    ret_10s,
                    volatility_10s,
                    volatility_30s,
                    volume_zscore_30s,
                ]

                if include_kline_context:
                    minute_bucket_ms = (bar_start_second // 60) * 60 * 1000
                    kline = kline_context.get(minute_bucket_ms, last_kline)

                    if kline is None:
                        kline = (close_px, close_px, close_px, close_px)

                    last_kline = kline
                    k_open, k_high, k_low, k_close = kline

                    dist_from_1m_open = (close_px - k_open) / max(k_open, EPS)
                    dist_from_1m_high = (close_px - k_high) / max(k_high, EPS)
                    dist_from_1m_low = (close_px - k_low) / max(k_low, EPS)
                    dist_from_1m_close = (close_px - k_close) / max(k_close, EPS)

                    row.extend(
                        [
                            dist_from_1m_open,
                            dist_from_1m_high,
                            dist_from_1m_low,
                            dist_from_1m_close,
                        ]
                    )

                writer.writerow(row)

                progress.advance(task_id, 1)
                bars_written += 1
                progress.update(task_id, rows=bars_written)

        out_path = _finalize_output_csv(
            final_path=final_path,
            part_path=part_path,
            handle=handle,
        )
        csv_summary = summarize_csv_artifact(out_path)

        manifest = {
            "artifact_path": str(out_path),
            "symbol": symbol,
            "symbols": [symbol],
            "timeframe": timeframe,
            "timeframe_s": timeframe_s,
            "date_range": {
                "start_ms": start_ms,
                "end_ms": end_ms,
                "start_at": timestamp_ms_to_iso(start_ms),
                "end_at": timestamp_ms_to_iso(end_ms),
            },
            "source_raw_files": build_source_entries(agg_source_paths + kline_source_paths),
            "labeling_params": {},
            "parent_dataset_id": None,
            **csv_summary,
        }
        write_dataset_manifest(
            dataset_type="features",
            dataset_id=dataset_id,
            manifest=manifest,
        )
        write_tag(base_dir=Path("data/features") / symbol, tag="latest", target_id=dataset_id)
        return out_path

    except Exception:
        _cleanup_part_file(part_path=part_path, handle=handle)
        raise
