from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast
import json
import math
import random

import numpy as np
import pandas as pd
import torch
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from sklearn.metrics import classification_report, confusion_matrix

from trader.data.registry import (
    build_dataset_id,
    current_git_commit_sha,
    find_dataset_id_by_artifact,
    write_run_manifest,
)
from trader.data.storage import baseline_run_dir, resolve_labels_dataset_dir, write_tag
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

console = Console()

LABEL_NAMES = {
    0: "NO_TRADE",
    1: "LONG_SETUP",
    2: "SHORT_SETUP",
}


@dataclass
class SplitData:
    x: np.ndarray
    y: np.ndarray
    timestamps_ms: np.ndarray


class MLPBlock(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.act1 = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.act2 = nn.GELU()
        self.use_residual = input_dim == hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.fc1(x)
        x = self.norm1(x)
        x = self.act1(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.norm2(x)
        x = self.act2(x)
        if self.use_residual:
            x = x + residual
        return x


class BaselineMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        depth: int = 4,
        dropout: float = 0.15,
        output_dim: int = 3,
    ) -> None:
        if depth < 1:
            raise ValueError("depth must be at least 1")
        super().__init__()
        blocks: list[nn.Module] = []
        current_dim = input_dim
        for _ in range(depth):
            block = MLPBlock(input_dim=current_dim, hidden_dim=hidden_dim, dropout=dropout)
            blocks.append(block)
            current_dim = hidden_dim

        self.backbone = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Linear(128, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() > 2:
            x = x.reshape(x.shape[0], -1)
        x = self.backbone(x)
        return self.head(x)


def _find_latest_label_file(symbol: str) -> Path:
    return resolve_labels_dataset_dir(symbol=symbol, latest=True) / "labels.csv"


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_labels(symbol: str, input_path: str | None = None) -> pd.DataFrame:
    path = Path(input_path) if input_path else _find_latest_label_file(symbol)
    console.print(f"[cyan]Loading labels from[/cyan] {path}")

    df = cast(pd.DataFrame, pd.read_csv(path))

    required = {"timestamp_ms", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Label dataset is missing columns: {sorted(missing)}")

    df = df.copy()
    df["timestamp_ms"] = pd.to_numeric(df["timestamp_ms"], errors="coerce")
    df["label"] = pd.to_numeric(df["label"], errors="coerce")

    df = cast(pd.DataFrame, df.dropna(subset=["timestamp_ms", "label"]))
    df = cast(pd.DataFrame, df.sort_values(by="timestamp_ms"))
    df = cast(pd.DataFrame, df.reset_index(drop=True))

    if df.empty:
        raise RuntimeError("Label dataset is empty after cleaning")

    return df


def _feature_columns(df: pd.DataFrame) -> list[str]:
    leakage_and_non_features = {
        "timestamp",
        "timestamp_ms",
        "timestamp_s",
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
        "horizon_steps",
        "horizon_seconds",
        "horizon_s",
        "take_profit_pct",
        "stop_loss_pct",
        "fee_pct",
        "slippage_pct",
    }

    numeric_cols: list[str] = []
    for col in df.columns:
        if col in leakage_and_non_features:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)

    if not numeric_cols:
        raise RuntimeError("No numeric feature columns found after leakage filtering")

    return numeric_cols


def _split_by_time(
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

    train = SplitData(x=x[:train_end], y=y[:train_end], timestamps_ms=timestamps_ms[:train_end])
    val = SplitData(x=x[train_end:val_end], y=y[train_end:val_end], timestamps_ms=timestamps_ms[train_end:val_end])
    test = SplitData(x=x[val_end:], y=y[val_end:], timestamps_ms=timestamps_ms[val_end:])

    if len(train.x) == 0 or len(val.x) == 0 or len(test.x) == 0:
        raise RuntimeError("One of the time splits is empty")

    return train, val, test


def _fit_standardizer(x_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x_train_2d = x_train.reshape(x_train.shape[0], -1)
    mean = x_train_2d.mean(axis=0)
    std = x_train_2d.std(axis=0)
    std = np.where(std < 1e-12, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def _apply_standardizer(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    x_2d = x.reshape(x.shape[0], -1)
    x_scaled = ((x_2d - mean) / std).astype(np.float32)
    return x_scaled.reshape(x.shape)


def _build_lookback_sequences(
    x: np.ndarray,
    y: np.ndarray,
    timestamps_ms: np.ndarray,
    lookback_window: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if lookback_window <= 0:
        raise ValueError("lookback_window must be greater than 0")
    if x.shape[0] < lookback_window:
        raise RuntimeError("Dataset has fewer rows than lookback_window")

    sample_count = x.shape[0] - lookback_window + 1
    feature_count = x.shape[1]
    x_seq = np.empty((sample_count, lookback_window, feature_count), dtype=np.float32)
    for i in range(sample_count):
        x_seq[i] = x[i : i + lookback_window]

    y_seq = y[lookback_window - 1 :]
    ts_seq = timestamps_ms[lookback_window - 1 :]
    return x_seq, y_seq, ts_seq


def _compute_class_weights(y_train: np.ndarray) -> np.ndarray:
    counts = np.bincount(y_train.astype(np.int64), minlength=3).astype(np.float64)
    counts = np.where(counts == 0, 1.0, counts)
    total = counts.sum()
    weights = total / (len(counts) * counts)
    return weights.astype(np.float32)


def _make_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    x_tensor = torch.tensor(x, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    dataset = TensorDataset(x_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()

    total_loss = 0.0
    total_samples = 0
    all_true: list[np.ndarray] = []
    all_pred: list[np.ndarray] = []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            logits = model(xb)
            loss = criterion(logits, yb)

            batch_size = int(yb.shape[0])
            total_loss += float(loss.item()) * batch_size
            total_samples += batch_size

            preds = torch.argmax(logits, dim=1)

            all_true.append(yb.cpu().numpy())
            all_pred.append(preds.cpu().numpy())

    if total_samples == 0:
        raise RuntimeError("Evaluation loader produced zero samples")

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    avg_loss = total_loss / total_samples

    return avg_loss, y_true, y_pred


def train_baseline(
    symbol: str,
    model_name: str = "baseline_mlp",
    input_path: str | None = None,
    epochs: int = 20,
    batch_size: int = 512,
    learning_rate: float = 1e-3,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    use_class_weights: bool = True,
    lookback_window: int = 12,
    hidden_dim: int = 512,
    depth: int = 4,
    dropout: float = 0.15,
    seed: int = 42,
    run_tag: str | None = "latest",
) -> Path:
    if epochs <= 0:
        raise ValueError("epochs must be greater than 0")
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than 0")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be greater than 0")
    if not (0.0 < train_frac < 1.0):
        raise ValueError("train_frac must be between 0 and 1")
    if not (0.0 < val_frac < 1.0):
        raise ValueError("val_frac must be between 0 and 1")
    if train_frac + val_frac >= 1.0:
        raise ValueError("train_frac + val_frac must be less than 1")
    if lookback_window <= 0:
        raise ValueError("lookback_window must be greater than 0")
    if hidden_dim <= 0:
        raise ValueError("hidden_dim must be greater than 0")
    if depth < 1:
        raise ValueError("depth must be at least 1")
    if not (0.0 <= dropout < 1.0):
        raise ValueError("dropout must be in [0, 1)")

    _set_seed(seed)

    df = _load_labels(symbol=symbol, input_path=input_path)
    feature_cols = _feature_columns(df)

    console.print(f"[cyan]Using {len(feature_cols)} feature columns[/cyan]")

    x_all = cast(pd.DataFrame, df[feature_cols]).to_numpy(dtype=np.float32)
    y_all = cast(pd.Series, df["label"]).to_numpy(dtype=np.int64)
    timestamps_ms = cast(pd.Series, df["timestamp_ms"]).to_numpy(dtype=np.int64)

    finite_mask = np.isfinite(x_all).all(axis=1)
    x_all = x_all[finite_mask]
    y_all = y_all[finite_mask]
    timestamps_ms = timestamps_ms[finite_mask]

    x_seq, y_seq, timestamps_seq = _build_lookback_sequences(
        x=x_all,
        y=y_all,
        timestamps_ms=timestamps_ms,
        lookback_window=lookback_window,
    )

    train_split, val_split, test_split = _split_by_time(
        x=x_seq,
        y=y_seq,
        timestamps_ms=timestamps_seq,
        train_frac=train_frac,
        val_frac=val_frac,
    )

    mean, std = _fit_standardizer(train_split.x)

    x_train = _apply_standardizer(train_split.x, mean, std)
    x_val = _apply_standardizer(val_split.x, mean, std)
    x_test = _apply_standardizer(test_split.x, mean, std)

    train_loader = _make_loader(x_train, train_split.y, batch_size=batch_size, shuffle=True)
    val_loader = _make_loader(x_val, val_split.y, batch_size=batch_size, shuffle=False)
    test_loader = _make_loader(x_test, test_split.y, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.print(f"[cyan]Training device:[/cyan] {device}")

    input_dim = int(x_train.shape[1] * x_train.shape[2])
    model = BaselineMLP(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        depth=depth,
        dropout=dropout,
        output_dim=3,
    ).to(device)

    if use_class_weights:
        class_weights_np = _compute_class_weights(train_split.y)
        class_weights = torch.tensor(class_weights_np, dtype=torch.float32, device=device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        class_weights_np = np.ones(3, dtype=np.float32)
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_val_loss = math.inf
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float]] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TextColumn("train_loss={task.fields[train_loss]}"),
        TextColumn("val_loss={task.fields[val_loss]}"),
        console=console,
    ) as progress:
        task_id = progress.add_task(
            "Training baseline",
            total=epochs,
            train_loss="-",
            val_loss="-",
        )

        for epoch in range(1, epochs + 1):
            model.train()

            running_loss = 0.0
            running_samples = 0

            for xb, yb in train_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)

                optimizer.zero_grad()
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

                batch_n = int(yb.shape[0])
                running_loss += float(loss.item()) * batch_n
                running_samples += batch_n

            train_loss = running_loss / max(running_samples, 1)
            val_loss, _, _ = _evaluate(model, val_loader, device, criterion)

            history.append(
                {
                    "epoch": float(epoch),
                    "train_loss": float(train_loss),
                    "val_loss": float(val_loss),
                }
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in model.state_dict().items()
                }

            progress.advance(task_id, 1)
            progress.update(
                task_id,
                train_loss=f"{train_loss:.5f}",
                val_loss=f"{val_loss:.5f}",
            )

    if best_state is None:
        raise RuntimeError("Training did not produce a best model state")

    model.load_state_dict(best_state)

    train_eval_loss, y_train_true, y_train_pred = _evaluate(model, train_loader, device, criterion)
    val_eval_loss, y_val_true, y_val_pred = _evaluate(model, val_loader, device, criterion)
    test_eval_loss, y_test_true, y_test_pred = _evaluate(model, test_loader, device, criterion)

    labels = [0, 1, 2]
    target_names = [LABEL_NAMES[i] for i in labels]

    train_report = classification_report(
        y_train_true,
        y_train_pred,
        labels=labels,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )
    val_report = classification_report(
        y_val_true,
        y_val_pred,
        labels=labels,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )
    test_report = classification_report(
        y_test_true,
        y_test_pred,
        labels=labels,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )

    test_cm = confusion_matrix(y_test_true, y_test_pred, labels=labels)

    label_dataset_id = find_dataset_id_by_artifact(
        dataset_type="labels",
        artifact_path=Path(input_path) if input_path else _find_latest_label_file(symbol),
    )
    run_source = {
        "symbol": symbol,
        "label_dataset_id": label_dataset_id,
        "model_name": model_name,
        "train_frac": train_frac,
        "val_frac": val_frac,
    }
    run_params = {
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "use_class_weights": use_class_weights,
        "seed": seed,
    }
    run_id = build_dataset_id(source=run_source, params=run_params)

    out_dir = baseline_run_dir(symbol=symbol, run_id=run_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = out_dir / "model.pt"
    report_path = out_dir / "report.json"
    history_path = out_dir / "history.json"

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "input_dim": input_dim,
        "flattened_input_dim": input_dim,
        "feature_columns": feature_cols,
        "standardizer_mean": mean.tolist(),
        "standardizer_std": std.tolist(),
        "class_weights": class_weights_np.tolist(),
        "label_mapping": LABEL_NAMES,
        "model_name": model_name,
        "symbol": symbol,
        "lookback_window": lookback_window,
        "hidden_dim": hidden_dim,
        "depth": depth,
        "dropout": dropout,
    }
    torch.save(checkpoint, checkpoint_path)

    report = {
        "symbol": symbol,
        "model_name": model_name,
        "device": str(device),
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "use_class_weights": use_class_weights,
        "lookback_window": lookback_window,
        "hidden_dim": hidden_dim,
        "depth": depth,
        "dropout": dropout,
        "feature_count": len(feature_cols),
        "feature_columns": feature_cols,
        "train_rows": int(len(train_split.x)),
        "val_rows": int(len(val_split.x)),
        "test_rows": int(len(test_split.x)),
        "train_loss": float(train_eval_loss),
        "val_loss": float(val_eval_loss),
        "test_loss": float(test_eval_loss),
        "train_report": train_report,
        "val_report": val_report,
        "test_report": test_report,
        "test_confusion_matrix": test_cm.tolist(),
        "run_id": run_id,
        "label_dataset_id": label_dataset_id,
    }

    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    with history_path.open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)

    console.print(f"[green]Saved checkpoint:[/green] {checkpoint_path}")
    console.print(f"[green]Saved report:[/green] {report_path}")

    console.print("[bold]Test confusion matrix[/bold]")
    console.print(str(test_cm))

    macro_f1 = float(test_report["macro avg"]["f1-score"])
    weighted_f1 = float(test_report["weighted avg"]["f1-score"])
    accuracy = float(test_report["accuracy"])

    console.print(
        f"[green]Test metrics:[/green] "
        f"accuracy={accuracy:.4f} "
        f"macro_f1={macro_f1:.4f} "
        f"weighted_f1={weighted_f1:.4f}"
    )

    if run_tag:
        write_tag(base_dir=Path("models") / "baseline" / symbol, tag=run_tag, target_id=run_id)

    run_manifest = {
        "run_type": "baseline_train",
        "git_commit_sha": current_git_commit_sha(),
        "input_dataset_ids": {
            "labels": label_dataset_id,
        },
        "hyperparameters": {
            "model_name": model_name,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "use_class_weights": use_class_weights,
            "seed": seed,
        },
        "split_config": {
            "method": "time_ordered",
            "train_frac": train_frac,
            "val_frac": val_frac,
            "test_frac": 1.0 - train_frac - val_frac,
            "train_rows": int(len(train_split.x)),
            "val_rows": int(len(val_split.x)),
            "test_rows": int(len(test_split.x)),
        },
        "output_artifact_paths": {
            "checkpoint": str(checkpoint_path),
            "report": str(report_path),
            "backtest": None,
        },
        "core_metrics": {
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "sharpe_ratio": None,
            "profit_factor": None,
            "cumulative_return_pct": None,
        },
        "notes_tags": [run_tag] if run_tag else [],
    }
    manifest_path = write_run_manifest(run_id=run_id, manifest=run_manifest)
    console.print(f"[green]Saved run manifest:[/green] {manifest_path}")

    return checkpoint_path
