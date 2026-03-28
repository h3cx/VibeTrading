
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast
import json
import math

import numpy as np
import pandas as pd
import torch
from rich.console import Console
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from trader.baseline.train_baseline import BaselineMLP, LABEL_NAMES
from trader.data.registry import (
    build_dataset_id,
    current_git_commit_sha,
    find_dataset_id_by_artifact,
    write_run_manifest,
)
from trader.data.storage import (
    resolve_backtest_run_dir,
    resolve_baseline_run_dir,
    resolve_labels_dataset_dir,
    write_tag,
)

console = Console()


@dataclass
class SplitData:
    df: pd.DataFrame
    x: np.ndarray
    y: np.ndarray
    timestamps_ms: np.ndarray


def _find_latest_checkpoint(symbol: str) -> Path:
    return resolve_baseline_run_dir(symbol=symbol, latest=True) / "model.pt"


def _find_latest_label_file(symbol: str) -> Path:
    return resolve_labels_dataset_dir(symbol=symbol, latest=True) / "labels.csv"


def _load_checkpoint(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint file does not exist: {path}")

    checkpoint = torch.load(path, map_location="cpu")
    # Legacy compatibility aliases for older baseline artifacts.
    if "state_dict" in checkpoint and "model_state_dict" not in checkpoint:
        checkpoint["model_state_dict"] = checkpoint["state_dict"]
    if "feature_cols" in checkpoint and "feature_columns" not in checkpoint:
        checkpoint["feature_columns"] = checkpoint["feature_cols"]
    if "standardizer_mu" in checkpoint and "standardizer_mean" not in checkpoint:
        checkpoint["standardizer_mean"] = checkpoint["standardizer_mu"]
    if "standardizer_sigma" in checkpoint and "standardizer_std" not in checkpoint:
        checkpoint["standardizer_std"] = checkpoint["standardizer_sigma"]

    required = {
        "model_state_dict",
        "input_dim",
        "feature_columns",
        "standardizer_mean",
        "standardizer_std",
        "symbol",
    }
    missing = required - set(checkpoint.keys())
    if missing:
        raise ValueError(f"Checkpoint is missing keys: {sorted(missing)}")
    return checkpoint


def _load_labels(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Label file does not exist: {path}")
    console.print(f"[cyan]Loading labels from[/cyan] {path}")
    df = cast(pd.DataFrame, pd.read_csv(path))

    required = {
        "timestamp",
        "timestamp_ms",
        "timestamp_s",
        "label",
        "label_name",
        "close",
        "long_net_return_pct",
        "short_net_return_pct",
        "time_to_exit_s",
        "long_event",
        "short_event",
        "exit_reason",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Label dataset is missing columns: {sorted(missing)}")

    df = df.copy()

    numeric_cols = [
        "timestamp_ms",
        "timestamp_s",
        "label",
        "close",
        "long_net_return_pct",
        "short_net_return_pct",
        "time_to_exit_s",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = cast(pd.DataFrame, df.dropna(subset=["timestamp_ms", "label", "close"]))
    df = cast(pd.DataFrame, df.sort_values(by="timestamp_ms"))
    df = cast(pd.DataFrame, df.drop_duplicates(subset=["timestamp_ms"], keep="last"))
    df = cast(pd.DataFrame, df.reset_index(drop=True))

    if df.empty:
        raise RuntimeError("Label dataset is empty after cleaning")

    return df


def _split_by_time(
    df: pd.DataFrame,
    x: np.ndarray,
    y: np.ndarray,
    timestamps_ms: np.ndarray,
    train_frac: float,
    val_frac: float,
) -> tuple[SplitData, SplitData, SplitData]:
    n = len(x)
    if n < 100:
        raise RuntimeError("Dataset is too small to split safely")

    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    train_end = max(train_end, 1)
    val_end = max(val_end, train_end + 1)
    val_end = min(val_end, n - 1)

    train = SplitData(
        df=cast(pd.DataFrame, df.iloc[:train_end].reset_index(drop=True)),
        x=x[:train_end],
        y=y[:train_end],
        timestamps_ms=timestamps_ms[:train_end],
    )
    val = SplitData(
        df=cast(pd.DataFrame, df.iloc[train_end:val_end].reset_index(drop=True)),
        x=x[train_end:val_end],
        y=y[train_end:val_end],
        timestamps_ms=timestamps_ms[train_end:val_end],
    )
    test = SplitData(
        df=cast(pd.DataFrame, df.iloc[val_end:].reset_index(drop=True)),
        x=x[val_end:],
        y=y[val_end:],
        timestamps_ms=timestamps_ms[val_end:],
    )

    if len(train.x) == 0 or len(val.x) == 0 or len(test.x) == 0:
        raise RuntimeError("One of the time splits is empty")

    return train, val, test


def _apply_standardizer(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    x_2d = x.reshape(x.shape[0], -1)
    x_scaled = ((x_2d - mean) / std).astype(np.float32)
    return x_scaled.reshape(x.shape)


def _build_lookback_sequences_for_backtest(
    df: pd.DataFrame,
    x: np.ndarray,
    y: np.ndarray,
    timestamps_ms: np.ndarray,
    lookback_window: int,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    if lookback_window <= 0:
        raise ValueError("lookback_window must be greater than 0")
    if len(x) < lookback_window:
        raise RuntimeError("Dataset has fewer rows than lookback_window")

    sample_count = x.shape[0] - lookback_window + 1
    feature_count = x.shape[1]
    x_seq = np.empty((sample_count, lookback_window, feature_count), dtype=np.float32)
    for i in range(sample_count):
        x_seq[i] = x[i : i + lookback_window]

    y_seq = y[lookback_window - 1 :]
    ts_seq = timestamps_ms[lookback_window - 1 :]
    df_seq = cast(pd.DataFrame, df.iloc[lookback_window - 1 :].reset_index(drop=True))
    return df_seq, x_seq, y_seq, ts_seq


def _predict_probabilities(
    model: nn.Module,
    x: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    x_tensor = torch.tensor(x, dtype=torch.float32)
    loader = DataLoader(
        TensorDataset(x_tensor),
        batch_size=batch_size,
        shuffle=False,
        pin_memory=(device.type == "cuda"),
    )

    model.eval()
    probs: list[np.ndarray] = []

    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(device, non_blocking=True)
            logits = model(xb)
            prob = torch.softmax(logits, dim=1)
            probs.append(prob.cpu().numpy())

    return np.concatenate(probs, axis=0)


def _threshold_decisions(
    probs: np.ndarray,
    long_threshold: float,
    short_threshold: float,
    margin: float,
) -> np.ndarray:
    decisions = np.zeros(len(probs), dtype=np.int64)

    p_no = probs[:, 0]
    p_long = probs[:, 1]
    p_short = probs[:, 2]

    for i in range(len(probs)):
        long_ok = (
            p_long[i] >= long_threshold
            and p_long[i] > p_short[i]
            and p_long[i] >= p_no[i] + margin
        )
        short_ok = (
            p_short[i] >= short_threshold
            and p_short[i] > p_long[i]
            and p_short[i] >= p_no[i] + margin
        )

        if long_ok and short_ok:
            decisions[i] = 1 if p_long[i] >= p_short[i] else 2
        elif long_ok:
            decisions[i] = 1
        elif short_ok:
            decisions[i] = 2
        else:
            decisions[i] = 0

    return decisions


def _max_drawdown_pct(equity_curve: list[float]) -> float:
    if not equity_curve:
        return 0.0

    peak = equity_curve[0]
    max_dd = 0.0

    for value in equity_curve:
        if value > peak:
            peak = value
        dd = (value / peak) - 1.0
        if dd < max_dd:
            max_dd = dd

    return abs(max_dd) * 100.0


def _sharpe_ratio_from_returns_pct(returns_pct: np.ndarray) -> float:
    if returns_pct.size < 2:
        return 0.0
    std = float(returns_pct.std(ddof=1))
    if std <= 1e-12:
        return 0.0
    mean = float(returns_pct.mean())
    return float((mean / std) * math.sqrt(float(returns_pct.size)))


def backtest_baseline(
    symbol: str,
    checkpoint_path: str | None = None,
    label_csv_path: str | None = None,
    eval_split: str = "test",
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    batch_size: int = 2048,
    long_threshold: float = 0.80,
    short_threshold: float = 0.80,
    margin: float = 0.05,
    run_tag: str | None = "latest",
) -> Path:
    if eval_split not in {"train", "val", "test", "all"}:
        raise ValueError("eval_split must be one of: train, val, test, all")

    checkpoint_file = Path(checkpoint_path) if checkpoint_path else _find_latest_checkpoint(symbol)
    checkpoint = _load_checkpoint(checkpoint_file)

    label_file = Path(label_csv_path) if label_csv_path else _find_latest_label_file(symbol)
    df = _load_labels(label_file)

    feature_cols = list(checkpoint["feature_columns"])
    missing_features = [col for col in feature_cols if col not in df.columns]
    if missing_features:
        raise ValueError(f"Label dataset is missing feature columns required by checkpoint: {missing_features}")

    x_all = cast(pd.DataFrame, df[feature_cols]).to_numpy(dtype=np.float32)
    y_all = cast(pd.Series, df["label"]).to_numpy(dtype=np.int64)
    timestamps_ms = cast(pd.Series, df["timestamp_ms"]).to_numpy(dtype=np.int64)

    finite_mask = np.isfinite(x_all).all(axis=1)
    df = cast(pd.DataFrame, df.loc[finite_mask].reset_index(drop=True))
    x_all = x_all[finite_mask]
    y_all = y_all[finite_mask]
    timestamps_ms = timestamps_ms[finite_mask]

    lookback_window = int(checkpoint.get("lookback_window", 1))
    df, x_all, y_all, timestamps_ms = _build_lookback_sequences_for_backtest(
        df=df,
        x=x_all,
        y=y_all,
        timestamps_ms=timestamps_ms,
        lookback_window=lookback_window,
    )

    train_split_data, val_split_data, test_split_data = _split_by_time(
        df=df,
        x=x_all,
        y=y_all,
        timestamps_ms=timestamps_ms,
        train_frac=train_frac,
        val_frac=val_frac,
    )

    if eval_split == "train":
        split = train_split_data
    elif eval_split == "val":
        split = val_split_data
    elif eval_split == "test":
        split = test_split_data
    else:
        split = SplitData(
            df=df,
            x=x_all,
            y=y_all,
            timestamps_ms=timestamps_ms,
        )

    mean = np.asarray(checkpoint["standardizer_mean"], dtype=np.float32)
    std = np.asarray(checkpoint["standardizer_std"], dtype=np.float32)
    x_eval = _apply_standardizer(split.x, mean, std)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.print(f"[cyan]Backtest device:[/cyan] {device}")

    model = BaselineMLP(
        input_dim=int(checkpoint["input_dim"]),
        hidden_dim=int(checkpoint.get("hidden_dim", 512)),
        depth=int(checkpoint.get("depth", 4)),
        dropout=float(checkpoint.get("dropout", 0.15)),
        output_dim=3,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    with console.status("Running model inference..."):
        probs = _predict_probabilities(
            model=model,
            x=x_eval,
            device=device,
            batch_size=batch_size,
        )

    threshold_preds = _threshold_decisions(
        probs=probs,
        long_threshold=long_threshold,
        short_threshold=short_threshold,
        margin=margin,
    )

    cm = confusion_matrix(split.y, threshold_preds, labels=[0, 1, 2])
    threshold_report = classification_report(
        split.y,
        threshold_preds,
        labels=[0, 1, 2],
        target_names=[LABEL_NAMES[i] for i in [0, 1, 2]],
        output_dict=True,
        zero_division=0,
    )

    eval_df = split.df.copy()
    eval_df["p_no_trade"] = probs[:, 0]
    eval_df["p_long"] = probs[:, 1]
    eval_df["p_short"] = probs[:, 2]
    eval_df["threshold_pred"] = threshold_preds
    eval_df["threshold_pred_name"] = [LABEL_NAMES[int(x)] for x in threshold_preds]

    trades: list[dict[str, Any]] = []
    equity = 1.0
    equity_curve: list[float] = [equity]
    next_allowed_timestamp_ms = -1

    with console.status("Simulating one-trade-at-a-time backtest..."):
        for row in eval_df.itertuples(index=False):
            ts_ms = int(row.timestamp_ms)

            if ts_ms < next_allowed_timestamp_ms:
                equity_curve.append(equity)
                continue

            decision = int(row.threshold_pred)
            if decision == 0:
                equity_curve.append(equity)
                continue

            if decision == 1:
                realized_return_pct = row.long_net_return_pct
                side = "LONG"
                event = row.long_event
            else:
                realized_return_pct = row.short_net_return_pct
                side = "SHORT"
                event = row.short_event

            tte = row.time_to_exit_s

            if pd.isna(realized_return_pct) or pd.isna(tte):
                equity_curve.append(equity)
                continue

            realized_return_pct = float(realized_return_pct)
            hold_s = max(int(math.ceil(float(tte))), 1)

            equity *= (1.0 + realized_return_pct / 100.0)
            equity_curve.append(equity)

            next_allowed_timestamp_ms = ts_ms + hold_s * 1000

            trades.append(
                {
                    "timestamp": row.timestamp,
                    "timestamp_ms": ts_ms,
                    "side": side,
                    "close": float(row.close),
                    "p_no_trade": float(row.p_no_trade),
                    "p_long": float(row.p_long),
                    "p_short": float(row.p_short),
                    "event": str(event),
                    "exit_reason": str(row.exit_reason),
                    "hold_s": hold_s,
                    "return_pct": realized_return_pct,
                    "equity_after": equity,
                }
            )

    trade_df = cast(pd.DataFrame, pd.DataFrame(trades))

    trade_count = int(len(trade_df))
    long_count = int((trade_df["side"] == "LONG").sum()) if trade_count else 0
    short_count = int((trade_df["side"] == "SHORT").sum()) if trade_count else 0

    if trade_count:
        wins = cast(pd.Series, trade_df["return_pct"]) > 0.0
        win_rate = float(wins.mean()) * 100.0
        avg_return_pct = float(cast(pd.Series, trade_df["return_pct"]).mean())
        cumulative_return_pct = (equity - 1.0) * 100.0
        gross_profit = float(cast(pd.Series, trade_df.loc[trade_df["return_pct"] > 0.0, "return_pct"]).sum())
        gross_loss = float(abs(cast(pd.Series, trade_df.loc[trade_df["return_pct"] < 0.0, "return_pct"]).sum()))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
        avg_hold_s = float(cast(pd.Series, trade_df["hold_s"]).mean())
    else:
        win_rate = 0.0
        avg_return_pct = 0.0
        cumulative_return_pct = 0.0
        gross_profit = 0.0
        gross_loss = 0.0
        profit_factor = 0.0
        avg_hold_s = 0.0

    max_drawdown_pct = _max_drawdown_pct(equity_curve)
    sharpe_ratio = _sharpe_ratio_from_returns_pct(
        cast(pd.Series, trade_df["return_pct"]).to_numpy(dtype=np.float64)
        if trade_count
        else np.array([], dtype=np.float64)
    )
    threshold_accuracy = float(threshold_report["accuracy"])
    threshold_macro_f1 = float(threshold_report["macro avg"]["f1-score"])
    label_dataset_id = find_dataset_id_by_artifact(dataset_type="labels", artifact_path=label_file)

    run_source = {
        "symbol": symbol,
        "checkpoint_path": str(checkpoint_file),
        "label_csv_path": str(label_file),
        "eval_split": eval_split,
    }
    run_params = {
        "train_frac": train_frac,
        "val_frac": val_frac,
        "batch_size": batch_size,
        "long_threshold": long_threshold,
        "short_threshold": short_threshold,
        "margin": margin,
    }
    run_id = build_dataset_id(source=run_source, params=run_params)

    out_dir = resolve_backtest_run_dir(symbol=symbol, run_id=run_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    report_path = out_dir / "report.json"
    trades_path = out_dir / "trades.csv"
    predictions_path = out_dir / "predictions.csv"

    trade_df.to_csv(trades_path, index=False)
    eval_df.to_csv(predictions_path, index=False)

    report = {
        "symbol": symbol,
        "checkpoint_path": str(checkpoint_file),
        "label_csv_path": str(label_file),
        "eval_split": eval_split,
        "device": str(device),
        "rows_evaluated": int(len(eval_df)),
        "long_threshold": long_threshold,
        "short_threshold": short_threshold,
        "margin": margin,
        "trade_count": trade_count,
        "long_count": long_count,
        "short_count": short_count,
        "win_rate_pct": win_rate,
        "avg_return_pct": avg_return_pct,
        "cumulative_return_pct": cumulative_return_pct,
        "max_drawdown_pct": max_drawdown_pct,
        "profit_factor": profit_factor,
        "gross_profit_pct": gross_profit,
        "gross_loss_pct": gross_loss,
        "avg_hold_s": avg_hold_s,
        "threshold_confusion_matrix": cm.tolist(),
        "true_label_counts": {
            "NO_TRADE": int((split.y == 0).sum()),
            "LONG_SETUP": int((split.y == 1).sum()),
            "SHORT_SETUP": int((split.y == 2).sum()),
        },
        "threshold_pred_counts": {
            "NO_TRADE": int((threshold_preds == 0).sum()),
            "LONG_SETUP": int((threshold_preds == 1).sum()),
            "SHORT_SETUP": int((threshold_preds == 2).sum()),
        },
        "run_id": run_id,
        "threshold_report": threshold_report,
    }

    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    console.print(f"[green]Saved backtest report:[/green] {report_path}")
    console.print(f"[green]Saved trades:[/green] {trades_path}")
    console.print(f"[green]Saved predictions:[/green] {predictions_path}")

    console.print("[bold]Thresholded confusion matrix[/bold]")
    console.print(str(cm))

    console.print(
        "[green]Backtest stats:[/green] "
        f"trades={trade_count} "
        f"win_rate={win_rate:.2f}% "
        f"avg_return={avg_return_pct:.4f}% "
        f"cum_return={cumulative_return_pct:.2f}% "
        f"max_dd={max_drawdown_pct:.2f}% "
        f"profit_factor={profit_factor:.4f}"
    )

    if run_tag:
        write_tag(base_dir=Path("artifacts") / "backtests" / symbol, tag=run_tag, target_id=run_id)

    run_manifest = {
        "run_type": "baseline_backtest",
        "git_commit_sha": current_git_commit_sha(),
        "input_dataset_ids": {
            "labels": label_dataset_id,
            "checkpoint_run_id": checkpoint.get("run_id"),
        },
        "hyperparameters": {
            "batch_size": batch_size,
            "long_threshold": long_threshold,
            "short_threshold": short_threshold,
            "margin": margin,
            "eval_split": eval_split,
        },
        "split_config": {
            "method": "time_ordered",
            "train_frac": train_frac,
            "val_frac": val_frac,
            "test_frac": 1.0 - train_frac - val_frac,
            "rows_evaluated": int(len(eval_df)),
        },
        "output_artifact_paths": {
            "checkpoint": str(checkpoint_file),
            "report": str(report_path),
            "backtest": str(trades_path),
        },
        "core_metrics": {
            "accuracy": threshold_accuracy,
            "macro_f1": threshold_macro_f1,
            "sharpe_ratio": sharpe_ratio,
            "profit_factor": profit_factor,
            "cumulative_return_pct": cumulative_return_pct,
        },
        "notes_tags": [run_tag] if run_tag else [],
    }
    manifest_path = write_run_manifest(run_id=run_id, manifest=run_manifest)
    console.print(f"[green]Saved run manifest:[/green] {manifest_path}")

    return report_path
