from datetime import datetime, timezone
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

from trader.data.config import ProjectConfig
from trader.data.binance_futures import fetch_klines_range, symbol_exists
from trader.data.binance_aggtrades import fetch_aggtrades_range
from trader.features.feature_builder import build_feature_frames
from trader.labels.label_builder import build_labels
from trader.baseline.train_baseline import train_baseline
from trader.baseline.backtest_baseline import backtest_baseline

app = typer.Typer(help="Trading model pipeline CLI")
console = Console()


def parse_date_to_ms(value: str) -> int:
    dt = datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


@app.command()
def menu() -> None:
    while True:
        console.print(
            Panel.fit(
                "\n".join(
                    [
                        "[1] Initialize project config",
                        "[2] Fetch klines",
                        "[3] Fetch aggtrades",
                        "[4] Build features",
                        "[5] Build labels",
                        "[6] Train baseline",
                        "[7] Backtest baseline",
                        "[8] Exit",
                    ]
                ),
                title="Trader",
            )
        )

        choice = Prompt.ask("Select an option", choices=["1", "2", "3", "4", "5", "6", "7", "8"])

        if choice == "1":
            init()
        elif choice == "2":
            fetch_klines_cmd()
        elif choice == "3":
            fetch_aggtrades_cmd()
        elif choice == "4":
            build_features_cmd()
        elif choice == "5":
            build_labels_cmd()
        elif choice == "6":
            train_baseline_cmd()
        elif choice == "7":
            backtest_baseline_cmd()
        elif choice == "8":
            console.print("[green]Goodbye.[/green]")
            break


@app.command()
def init() -> None:
    console.print("[bold]Project initialization[/bold]")

    exchange = Prompt.ask("Exchange", default="binance")
    symbols_raw = Prompt.ask("Symbols (comma separated)", default="BTCUSDT,SOLUSDT")
    timeframe = Prompt.ask("Decision timeframe", default="5m")

    symbols = [s.strip().upper() for s in symbols_raw.split(",") if s.strip()]

    config_dir = Path("configs")
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "project.toml"

    config_text = f"""exchange = "{exchange}"
timeframe = "{timeframe}"
symbols = [{", ".join(f'"{s}"' for s in symbols)}]
"""

    config_path.write_text(config_text, encoding="utf-8")
    console.print(f"[green]Wrote config to {config_path}[/green]")


@app.command("fetch-klines")
def fetch_klines_cmd() -> None:
    cfg = ProjectConfig.load()

    symbol = Prompt.ask("Symbol", default=cfg.symbols[0]).upper()
    interval = Prompt.ask("Kline interval", default="1m")
    start = Prompt.ask("Start date (YYYY-MM-DD)")
    end = Prompt.ask("End date (YYYY-MM-DD)")
    source = Prompt.ask(
        "Klines source",
        choices=["auto", "rest", "archive"],
        default="auto",
    )

    if not symbol_exists(symbol):
        console.print(f"[red]Symbol not tradable or not found: {symbol}[/red]")
        raise typer.Exit(code=1)

    out = fetch_klines_range(
        symbol=symbol,
        interval=interval,
        start_ms=parse_date_to_ms(start),
        end_ms=parse_date_to_ms(end),
        source=source,
    )
    console.print(f"[green]Saved klines to {out}[/green]")


@app.command("fetch-aggtrades")
def fetch_aggtrades_cmd() -> None:
    cfg = ProjectConfig.load()

    symbol = Prompt.ask("Symbol", default=cfg.symbols[0]).upper()
    start = Prompt.ask("Start date (YYYY-MM-DD)")
    end = Prompt.ask("End date (YYYY-MM-DD)")
    source = Prompt.ask(
        "Aggtrades source",
        choices=["auto", "rest", "archive"],
        default="auto",
    )
    console.print(
        "[dim]RAM tradeoff: each extra worker can add roughly 40-120MB peak RAM while a day is being "
        "downloaded/decompressed.[/dim]"
    )
    max_download_workers = int(
        Prompt.ask(
            "Max download workers",
            default="6",
        )
    )
    max_inflight_days = int(
        Prompt.ask(
            "Max inflight days (bounded queue for backpressure)",
            default="12",
        )
    )
    max_parse_workers = int(
        Prompt.ask(
            "Max parse workers",
            default="4",
        )
    )
    max_parsed_batches = int(
        Prompt.ask(
            "Max parsed batches (bounded parse->persist queue)",
            default="12",
        )
    )
    parse_chunksize_rows = int(
        Prompt.ask(
            "Parse chunk size rows",
            default="250000",
        )
    )
    request_timeout_s = float(Prompt.ask("Request timeout seconds", default="30"))
    max_retries = int(Prompt.ask("Max retries per day", default="3"))
    retry_backoff_s = float(Prompt.ask("Retry backoff base seconds", default="1"))
    sequential = Prompt.ask(
        "Sequential mode? (debug fallback; disables pipeline)",
        choices=["y", "n"],
        default="n",
    )

    if not symbol_exists(symbol):
        console.print(f"[red]Symbol not tradable or not found: {symbol}[/red]")
        raise typer.Exit(code=1)

    out = fetch_aggtrades_range(
        symbol=symbol,
        start_ms=parse_date_to_ms(start),
        end_ms=parse_date_to_ms(end),
        source=source,
        max_download_workers=max_download_workers,
        max_inflight_days=max_inflight_days,
        max_parse_workers=max_parse_workers,
        max_parsed_batches=max_parsed_batches,
        parse_chunksize_rows=parse_chunksize_rows,
        request_timeout_s=request_timeout_s,
        max_retries=max_retries,
        retry_backoff_s=retry_backoff_s,
        sequential=(sequential == "y"),
    )
    console.print(f"[green]Saved aggtrades to {out.csv_path}[/green]")
    console.print(f"[green]Saved fetch profiling report to {out.metrics_path}[/green]")
    console.print(f"[cyan]Fetch summary:[/cyan] {out.summary}")


@app.command("build-features")
def build_features_cmd() -> None:
    cfg = ProjectConfig.load()

    symbol = Prompt.ask("Symbol", default=cfg.symbols[0]).upper()
    start = Prompt.ask("Start date (YYYY-MM-DD)")
    end = Prompt.ask("End date (YYYY-MM-DD)")
    dataset_name = Prompt.ask("Dataset name", default="default")
    timeframe = Prompt.ask("Feature timeframe", default="5m")
    include_kline_context = Prompt.ask(
        "Include 1m kline context?",
        choices=["y", "n"],
        default="y",
    )

    out = build_feature_frames(
        symbol=symbol,
        start_ms=parse_date_to_ms(start),
        end_ms=parse_date_to_ms(end),
        dataset_name=dataset_name,
        timeframe=timeframe,
        include_kline_context=(include_kline_context == "y"),
    )
    console.print(f"[green]Saved feature dataset to {out}[/green]")


@app.command("build-labels")
def build_labels_cmd() -> None:
    cfg = ProjectConfig.load()

    symbol = Prompt.ask("Symbol", default=cfg.symbols[0]).upper()
    input_path = Prompt.ask("Feature CSV path (blank = latest for symbol)", default="")
    horizon_steps = int(Prompt.ask("Horizon in bars", default="12"))
    take_profit_pct = float(Prompt.ask("Take-profit percent", default="0.35"))
    stop_loss_pct = float(Prompt.ask("Stop-loss percent", default="0.25"))
    fee_pct = float(Prompt.ask("Round-trip fee percent", default="0.08"))
    slippage_pct = float(Prompt.ask("Round-trip slippage percent", default="0.02"))
    dataset_name = Prompt.ask("Label dataset name", default="default_labels")

    out = build_labels(
        symbol=symbol,
        horizon_steps=horizon_steps,
        take_profit_pct=take_profit_pct,
        stop_loss_pct=stop_loss_pct,
        fee_pct=fee_pct,
        slippage_pct=slippage_pct,
        dataset_name=dataset_name,
        input_path=input_path or None,
    )
    console.print(f"[green]Saved labels to {out}[/green]")


@app.command("train-baseline")
def train_baseline_cmd() -> None:
    cfg = ProjectConfig.load()

    symbol = Prompt.ask("Symbol", default=cfg.symbols[0]).upper()
    input_path = Prompt.ask("Label CSV path (blank = latest for symbol)", default="")
    model_name = Prompt.ask("Model name", default="baseline_mlp")
    epochs = int(Prompt.ask("Epochs", default="20"))
    batch_size = int(Prompt.ask("Batch size", default="512"))
    learning_rate = float(Prompt.ask("Learning rate", default="0.001"))
    lookback_window = int(Prompt.ask("Lookback window (bars)", default="12"))
    hidden_dim = int(Prompt.ask("Hidden size", default="512"))
    depth = int(Prompt.ask("MLP block depth", default="4"))
    dropout = float(Prompt.ask("Dropout", default="0.15"))
    train_frac = float(Prompt.ask("Train fraction", default="0.70"))
    val_frac = float(Prompt.ask("Validation fraction", default="0.15"))
    use_class_weights = Prompt.ask(
        "Use class weights?",
        choices=["y", "n"],
        default="y",
    )

    out = train_baseline(
        symbol=symbol,
        model_name=model_name,
        input_path=input_path or None,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        lookback_window=lookback_window,
        hidden_dim=hidden_dim,
        depth=depth,
        dropout=dropout,
        train_frac=train_frac,
        val_frac=val_frac,
        use_class_weights=(use_class_weights == "y"),
    )
    console.print(f"[green]Saved baseline model to {out}[/green]")


@app.command("backtest-baseline")
def backtest_baseline_cmd() -> None:
    cfg = ProjectConfig.load()

    symbol = Prompt.ask("Symbol", default=cfg.symbols[0]).upper()
    checkpoint_path = Prompt.ask("Checkpoint path (blank = latest for symbol)", default="")
    label_csv_path = Prompt.ask("Label CSV path (blank = latest for symbol)", default="")
    eval_split = Prompt.ask(
        "Eval split",
        choices=["train", "val", "test", "all"],
        default="test",
    )
    train_frac = float(Prompt.ask("Train fraction used during training", default="0.70"))
    val_frac = float(Prompt.ask("Validation fraction used during training", default="0.15"))
    batch_size = int(Prompt.ask("Inference batch size", default="2048"))
    long_threshold = float(Prompt.ask("Long probability threshold", default="0.80"))
    short_threshold = float(Prompt.ask("Short probability threshold", default="0.80"))
    margin = float(Prompt.ask("Trade-vs-no-trade margin", default="0.05"))

    out = backtest_baseline(
        symbol=symbol,
        checkpoint_path=checkpoint_path or None,
        label_csv_path=label_csv_path or None,
        eval_split=eval_split,
        train_frac=train_frac,
        val_frac=val_frac,
        batch_size=batch_size,
        long_threshold=long_threshold,
        short_threshold=short_threshold,
        margin=margin,
    )
    console.print(f"[green]Saved backtest report to {out}[/green]")


if __name__ == "__main__":
    app()
