from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator
import csv

from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

console = Console()

LABEL_NO_TRADE = 0
LABEL_LONG = 1
LABEL_SHORT = 2


@dataclass
class FeatureRow:
    raw: dict[str, str]
    timestamp_ms: int
    timestamp_s: int
    high: float
    low: float
    close: float


def _utc_stamp(ms: int) -> str:
    dt = datetime.fromtimestamp(ms / 1000, tz=timezone.utc)
    return dt.strftime("%Y%m%d_%H%M%S")


def _build_output_path(
    symbol: str,
    dataset_name: str,
    start_ms: int,
    end_ms: int,
) -> Path:
    out_dir = Path("data/labels") / symbol
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{dataset_name}_{_utc_stamp(start_ms)}__{_utc_stamp(end_ms)}.csv"
    return out_dir / filename


def _find_latest_feature_file(symbol: str) -> Path:
    feature_dir = Path("data/features") / symbol
    if not feature_dir.exists():
        raise FileNotFoundError(f"No feature directory found for {symbol}: {feature_dir}")

    candidates = sorted(
        feature_dir.glob("*.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    if not candidates:
        raise FileNotFoundError(f"No feature CSV files found for {symbol} in {feature_dir}")

    return candidates[0]


def _parse_feature_row(raw: dict[str, str]) -> FeatureRow:
    return FeatureRow(
        raw=raw,
        timestamp_ms=int(raw["timestamp_ms"]),
        timestamp_s=int(raw["timestamp_s"]),
        high=float(raw["high"]),
        low=float(raw["low"]),
        close=float(raw["close"]),
    )


def _iter_feature_rows(path: Path) -> tuple[list[str], Iterator[FeatureRow]]:
    handle = path.open("r", newline="", encoding="utf-8")
    reader = csv.DictReader(handle)

    if reader.fieldnames is None:
        handle.close()
        raise ValueError(f"Feature CSV has no header: {path}")

    fieldnames = list(reader.fieldnames)

    required = {
        "timestamp",
        "timestamp_ms",
        "timestamp_s",
        "open",
        "high",
        "low",
        "close",
    }
    missing = required - set(fieldnames)
    if missing:
        handle.close()
        raise ValueError(f"Feature dataset is missing columns: {sorted(missing)}")

    def generator() -> Iterator[FeatureRow]:
        try:
            for raw in reader:
                if not raw:
                    continue
                try:
                    yield _parse_feature_row(raw)
                except (KeyError, ValueError):
                    continue
        finally:
            handle.close()

    return fieldnames, generator()


def _open_output_writer(
    symbol: str,
    dataset_name: str,
    fieldnames_in: list[str],
) -> tuple[Path, Path, Any, csv.DictWriter]:
    ordered_front = [
        "timestamp",
        "timestamp_ms",
        "timestamp_s",
        "close",
        "label",
        "label_name",
        "exit_reason",
        "time_to_exit_s",
        "long_event",
        "short_event",
        "long_net_return_pct",
        "short_net_return_pct",
        "long_horizon_return_pct",
        "short_horizon_return_pct",
        "future_max_upside_pct",
        "future_max_downside_pct",
        "horizon_s",
        "take_profit_pct",
        "stop_loss_pct",
        "fee_pct",
        "slippage_pct",
    ]

    remaining = [col for col in fieldnames_in if col not in ordered_front]
    fieldnames_out = ordered_front + remaining

    # Temporary placeholder. Real name is decided once we know start/end.
    out_dir = Path("data/labels") / symbol
    out_dir.mkdir(parents=True, exist_ok=True)

    part_path = out_dir / f"{dataset_name}.csv.part"
    handle = part_path.open("w", newline="", encoding="utf-8")
    writer = csv.DictWriter(handle, fieldnames=fieldnames_out)
    writer.writeheader()

    return out_dir, part_path, handle, writer


def _finalize_output(
    out_dir: Path,
    part_path: Path,
    handle: Any,
    symbol: str,
    dataset_name: str,
    start_ms: int,
    end_ms: int,
) -> Path:
    handle.flush()
    handle.close()

    final_path = _build_output_path(
        symbol=symbol,
        dataset_name=dataset_name,
        start_ms=start_ms,
        end_ms=end_ms,
    )
    if final_path.exists():
        final_path.unlink()
    part_path.replace(final_path)
    return final_path


def _cleanup_output(part_path: Path, handle: Any | None = None) -> None:
    if handle is not None and not handle.closed:
        handle.close()
    if part_path.exists():
        part_path.unlink()


def _compute_label_for_current(
    current: FeatureRow,
    future_rows: list[FeatureRow],
    tp: float,
    sl: float,
    total_cost: float,
    horizon_s: int,
    take_profit_pct: float,
    stop_loss_pct: float,
    fee_pct: float,
    slippage_pct: float,
) -> dict[str, Any]:
    entry_px = current.close

    if not future_rows:
        return {
            "future_max_upside_pct": "",
            "future_max_downside_pct": "",
            "long_horizon_return_pct": "",
            "short_horizon_return_pct": "",
            "long_event": "NONE",
            "short_event": "NONE",
            "long_net_return_pct": "",
            "short_net_return_pct": "",
            "time_to_exit_s": "",
            "exit_reason": "NONE",
            "label": LABEL_NO_TRADE,
            "label_name": "NO_TRADE",
            "horizon_s": horizon_s,
            "take_profit_pct": take_profit_pct,
            "stop_loss_pct": stop_loss_pct,
            "fee_pct": fee_pct,
            "slippage_pct": slippage_pct,
        }

    future_high = max(row.high for row in future_rows)
    future_low = min(row.low for row in future_rows)
    future_close = future_rows[-1].close

    future_max_upside_pct = ((future_high / entry_px) - 1.0) * 100.0
    future_max_downside_pct = ((future_low / entry_px) - 1.0) * 100.0

    long_horizon_return_pct = ((future_close / entry_px) - 1.0 - total_cost) * 100.0
    short_horizon_return_pct = ((entry_px / future_close) - 1.0 - total_cost) * 100.0

    long_tp_px = entry_px * (1.0 + tp)
    long_sl_px = entry_px * (1.0 - sl)
    short_tp_px = entry_px * (1.0 - tp)
    short_sl_px = entry_px * (1.0 + sl)

    found_long = False
    found_short = False

    long_net = None
    short_net = None
    long_tte: float | None = None
    short_tte: float | None = None
    long_reason = "NONE"
    short_reason = "NONE"
    long_event = "NONE"
    short_event = "NONE"

    for offset, row in enumerate(future_rows, start=1):
        high_j = row.high
        low_j = row.low

        if not found_long:
            hit_long_tp = high_j >= long_tp_px
            hit_long_sl = low_j <= long_sl_px

            if hit_long_tp and hit_long_sl:
                found_long = True
                long_reason = "AMBIGUOUS"
                long_event = "AMBIGUOUS"
                long_net = None
                long_tte = float(offset)
            elif hit_long_tp:
                found_long = True
                long_reason = "TP"
                long_event = "TP"
                long_net = (tp - total_cost) * 100.0
                long_tte = float(offset)
            elif hit_long_sl:
                found_long = True
                long_reason = "SL"
                long_event = "SL"
                long_net = (-sl - total_cost) * 100.0
                long_tte = float(offset)

        if not found_short:
            hit_short_tp = low_j <= short_tp_px
            hit_short_sl = high_j >= short_sl_px

            if hit_short_tp and hit_short_sl:
                found_short = True
                short_reason = "AMBIGUOUS"
                short_event = "AMBIGUOUS"
                short_net = None
                short_tte = float(offset)
            elif hit_short_tp:
                found_short = True
                short_reason = "TP"
                short_event = "TP"
                short_net = (tp - total_cost) * 100.0
                short_tte = float(offset)
            elif hit_short_sl:
                found_short = True
                short_reason = "SL"
                short_event = "SL"
                short_net = (-sl - total_cost) * 100.0
                short_tte = float(offset)

        if found_long and found_short:
            break

    if not found_long:
        long_event = "HORIZON"
        long_reason = "HORIZON"
        long_net = long_horizon_return_pct
        long_tte = float(len(future_rows))

    if not found_short:
        short_event = "HORIZON"
        short_reason = "HORIZON"
        short_net = short_horizon_return_pct
        short_tte = float(len(future_rows))

    long_good = (long_net is not None) and (float(long_net) > 0.0) and (long_reason == "TP")
    short_good = (short_net is not None) and (float(short_net) > 0.0) and (short_reason == "TP")

    label = LABEL_NO_TRADE
    label_name = "NO_TRADE"
    exit_reason = "NONE"
    time_to_exit_s: float | str = ""

    if long_good and not short_good:
        label = LABEL_LONG
        label_name = "LONG_SETUP"
        exit_reason = long_reason
        time_to_exit_s = long_tte if long_tte is not None else ""
    elif short_good and not long_good:
        label = LABEL_SHORT
        label_name = "SHORT_SETUP"
        exit_reason = short_reason
        time_to_exit_s = short_tte if short_tte is not None else ""
    elif long_good and short_good:
        if float(long_net) >= float(short_net):
            label = LABEL_LONG
            label_name = "LONG_SETUP"
            exit_reason = long_reason
            time_to_exit_s = long_tte if long_tte is not None else ""
        else:
            label = LABEL_SHORT
            label_name = "SHORT_SETUP"
            exit_reason = short_reason
            time_to_exit_s = short_tte if short_tte is not None else ""
    else:
        if long_net is not None and short_net is not None:
            if float(long_net) >= float(short_net):
                exit_reason = long_reason
                time_to_exit_s = long_tte if long_tte is not None else ""
            else:
                exit_reason = short_reason
                time_to_exit_s = short_tte if short_tte is not None else ""
        elif long_net is not None:
            exit_reason = long_reason
            time_to_exit_s = long_tte if long_tte is not None else ""
        elif short_net is not None:
            exit_reason = short_reason
            time_to_exit_s = short_tte if short_tte is not None else ""

    return {
        "future_max_upside_pct": future_max_upside_pct,
        "future_max_downside_pct": future_max_downside_pct,
        "long_horizon_return_pct": long_horizon_return_pct,
        "short_horizon_return_pct": short_horizon_return_pct,
        "long_event": long_event,
        "short_event": short_event,
        "long_net_return_pct": "" if long_net is None else long_net,
        "short_net_return_pct": "" if short_net is None else short_net,
        "time_to_exit_s": time_to_exit_s,
        "exit_reason": exit_reason,
        "label": label,
        "label_name": label_name,
        "horizon_s": horizon_s,
        "take_profit_pct": take_profit_pct,
        "stop_loss_pct": stop_loss_pct,
        "fee_pct": fee_pct,
        "slippage_pct": slippage_pct,
    }


def build_labels(
    symbol: str,
    horizon_s: int,
    take_profit_pct: float,
    stop_loss_pct: float,
    fee_pct: float,
    slippage_pct: float,
    dataset_name: str = "default_labels",
    input_path: str | None = None,
) -> Path:
    if horizon_s <= 0:
        raise ValueError("horizon_s must be greater than 0")
    if take_profit_pct <= 0:
        raise ValueError("take_profit_pct must be greater than 0")
    if stop_loss_pct <= 0:
        raise ValueError("stop_loss_pct must be greater than 0")

    feature_path = Path(input_path) if input_path else _find_latest_feature_file(symbol)
    console.print(f"[cyan]Streaming features from[/cyan] {feature_path}")

    fieldnames_in, row_iter = _iter_feature_rows(feature_path)
    out_dir, part_path, handle, writer = _open_output_writer(
        symbol=symbol,
        dataset_name=dataset_name,
        fieldnames_in=fieldnames_in,
    )

    tp = take_profit_pct / 100.0
    sl = stop_loss_pct / 100.0
    total_cost = (fee_pct + slippage_pct) / 100.0

    buffer: deque[FeatureRow] = deque()
    first_ts_ms: int | None = None
    last_ts_ms: int | None = None

    long_count = 0
    short_count = 0
    no_trade_count = 0
    rows_written = 0

    try:
        for _ in range(horizon_s + 1):
            try:
                buffer.append(next(row_iter))
            except StopIteration:
                break

        if not buffer:
            raise RuntimeError("Feature dataset is empty after cleaning")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            TextColumn("rows={task.fields[rows]}"),
            console=console,
        ) as progress:
            task_id = progress.add_task("Building labels", rows=0)

            while buffer:
                current = buffer[0]
                future_rows = list(buffer)[1:]

                derived = _compute_label_for_current(
                    current=current,
                    future_rows=future_rows,
                    tp=tp,
                    sl=sl,
                    total_cost=total_cost,
                    horizon_s=horizon_s,
                    take_profit_pct=take_profit_pct,
                    stop_loss_pct=stop_loss_pct,
                    fee_pct=fee_pct,
                    slippage_pct=slippage_pct,
                )

                out_row = dict(current.raw)
                out_row.update(derived)
                writer.writerow(out_row)

                if first_ts_ms is None:
                    first_ts_ms = current.timestamp_ms
                last_ts_ms = current.timestamp_ms

                if int(derived["label"]) == LABEL_LONG:
                    long_count += 1
                elif int(derived["label"]) == LABEL_SHORT:
                    short_count += 1
                else:
                    no_trade_count += 1

                rows_written += 1
                progress.update(task_id, rows=rows_written)

                buffer.popleft()

                try:
                    buffer.append(next(row_iter))
                except StopIteration:
                    pass

        if first_ts_ms is None or last_ts_ms is None:
            raise RuntimeError("No labels were written")

        out_path = _finalize_output(
            out_dir=out_dir,
            part_path=part_path,
            handle=handle,
            symbol=symbol,
            dataset_name=dataset_name,
            start_ms=first_ts_ms,
            end_ms=last_ts_ms + 1000,
        )

        console.print(
            f"[green]Labels built:[/green] "
            f"LONG={long_count} SHORT={short_count} NO_TRADE={no_trade_count}"
        )

        return out_path

    except Exception:
        _cleanup_output(part_path=part_path, handle=handle)
        raise
