from __future__ import annotations

from pathlib import Path


RAW_ROOT = Path("data/raw")
FEATURES_ROOT = Path("data/features")
LABELS_ROOT = Path("data/labels")
MODELS_ROOT = Path("models")
ARTIFACTS_ROOT = Path("artifacts")


def raw_data_dir(*, exchange: str, symbol: str, datatype: str) -> Path:
    return RAW_ROOT / exchange / symbol / datatype


def features_dataset_dir(*, symbol: str, dataset_id: str) -> Path:
    return FEATURES_ROOT / symbol / dataset_id


def labels_dataset_dir(*, symbol: str, dataset_id: str) -> Path:
    return LABELS_ROOT / symbol / dataset_id


def baseline_run_dir(*, symbol: str, run_id: str) -> Path:
    return MODELS_ROOT / "baseline" / symbol / run_id


def rl_run_dir(*, symbol: str, run_id: str) -> Path:
    return MODELS_ROOT / "rl" / symbol / run_id


def backtest_run_dir(*, symbol: str, run_id: str) -> Path:
    return ARTIFACTS_ROOT / "backtests" / symbol / run_id


def _tag_dir(base_dir: Path) -> Path:
    return base_dir / "_tags"


def write_tag(*, base_dir: Path, tag: str, target_id: str) -> Path:
    tags = _tag_dir(base_dir)
    tags.mkdir(parents=True, exist_ok=True)
    out = tags / f"{tag}.txt"
    out.write_text(target_id + "\n", encoding="utf-8")
    return out


def resolve_by_tag(*, base_dir: Path, tag: str) -> str:
    path = _tag_dir(base_dir) / f"{tag}.txt"
    if not path.exists():
        raise FileNotFoundError(f"No tag '{tag}' found under {base_dir}")
    value = path.read_text(encoding="utf-8").strip()
    if not value:
        raise ValueError(f"Tag '{tag}' under {base_dir} is empty")
    return value


def resolve_latest_id(*, base_dir: Path) -> str:
    if not base_dir.exists():
        raise FileNotFoundError(f"Directory does not exist: {base_dir}")

    candidates = [
        path
        for path in base_dir.iterdir()
        if path.is_dir() and path.name != "_tags"
    ]
    if not candidates:
        raise FileNotFoundError(f"No id directories found in {base_dir}")

    latest = max(candidates, key=lambda path: path.stat().st_mtime)
    return latest.name


def resolve_id(*, base_dir: Path, item_id: str | None = None, tag: str | None = None, latest: bool = False) -> str:
    selectors = int(item_id is not None) + int(tag is not None) + int(latest)
    if selectors != 1:
        raise ValueError("Provide exactly one selector: item_id, tag, or latest=True")

    if item_id is not None:
        return item_id
    if tag is not None:
        return resolve_by_tag(base_dir=base_dir, tag=tag)
    return resolve_latest_id(base_dir=base_dir)


def resolve_features_dataset_dir(
    *,
    symbol: str,
    dataset_id: str | None = None,
    tag: str | None = None,
    latest: bool = False,
) -> Path:
    root = FEATURES_ROOT / symbol
    resolved = resolve_id(base_dir=root, item_id=dataset_id, tag=tag, latest=latest)
    return features_dataset_dir(symbol=symbol, dataset_id=resolved)


def resolve_labels_dataset_dir(
    *,
    symbol: str,
    dataset_id: str | None = None,
    tag: str | None = None,
    latest: bool = False,
) -> Path:
    root = LABELS_ROOT / symbol
    resolved = resolve_id(base_dir=root, item_id=dataset_id, tag=tag, latest=latest)
    return labels_dataset_dir(symbol=symbol, dataset_id=resolved)


def resolve_baseline_run_dir(
    *,
    symbol: str,
    run_id: str | None = None,
    tag: str | None = None,
    latest: bool = False,
) -> Path:
    root = MODELS_ROOT / "baseline" / symbol
    resolved = resolve_id(base_dir=root, item_id=run_id, tag=tag, latest=latest)
    return baseline_run_dir(symbol=symbol, run_id=resolved)


def resolve_backtest_run_dir(
    *,
    symbol: str,
    run_id: str | None = None,
    tag: str | None = None,
    latest: bool = False,
) -> Path:
    root = ARTIFACTS_ROOT / "backtests" / symbol
    resolved = resolve_id(base_dir=root, item_id=run_id, tag=tag, latest=latest)
    return backtest_run_dir(symbol=symbol, run_id=resolved)
