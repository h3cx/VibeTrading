from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import json
import re

import pandas as pd

STAMP_RE = re.compile(r"(\d{8}_\d{6})__(\d{8}_\d{6})$")


def _project_root() -> Path:
    return Path.cwd()


def _to_utc_iso(ms: int) -> str:
    dt = datetime.fromtimestamp(ms / 1000, tz=timezone.utc)
    return dt.isoformat().replace("+00:00", "Z")


def parse_date_range_from_filename(path: Path) -> tuple[str, str] | None:
    match = STAMP_RE.search(path.stem)
    if match is None:
        return None

    start = datetime.strptime(match.group(1), "%Y%m%d_%H%M%S").replace(tzinfo=timezone.utc)
    end = datetime.strptime(match.group(2), "%Y%m%d_%H%M%S").replace(tzinfo=timezone.utc)
    return (
        start.isoformat().replace("+00:00", "Z"),
        end.isoformat().replace("+00:00", "Z"),
    )


def _artifact_relpath(path: Path) -> str:
    root = _project_root().resolve()
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(root))
    except ValueError:
        return str(resolved)


def build_dataset_id(*, source: dict, params: dict) -> str:
    payload = {
        "params": params,
        "source": source,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256(encoded).hexdigest()


def summarize_csv_artifact(path: Path, *, class_column: str | None = None) -> dict:
    frame = pd.read_csv(path, low_memory=False)

    row_count = int(len(frame))
    column_schema = [{"name": str(col), "dtype": str(frame[col].dtype)} for col in frame.columns]

    missingness_stats = {}
    for col in frame.columns:
        missing_count = int(frame[col].isna().sum())
        missing_pct = float((missing_count / row_count) * 100.0) if row_count else 0.0
        missingness_stats[str(col)] = {
            "missing_count": missing_count,
            "missing_pct": missing_pct,
        }

    summary = {
        "row_count": row_count,
        "column_schema": column_schema,
        "missingness_stats": missingness_stats,
    }

    if class_column is not None and class_column in frame.columns:
        class_counts = frame[class_column].value_counts(dropna=False)
        class_distribution = {}
        for cls, count in class_counts.items():
            key = "NULL" if pd.isna(cls) else str(cls)
            count_i = int(count)
            pct = float((count_i / row_count) * 100.0) if row_count else 0.0
            class_distribution[key] = {
                "count": count_i,
                "pct": pct,
            }
        summary["class_distribution"] = class_distribution

    return summary


def find_dataset_id_by_artifact(*, dataset_type: str, artifact_path: Path) -> str | None:
    registry_dir = _project_root() / "data" / "registry" / dataset_type
    if not registry_dir.exists():
        return None

    relpath = _artifact_relpath(artifact_path)

    for manifest_path in sorted(registry_dir.glob("*.json")):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue

        if manifest.get("artifact_path") == relpath:
            return str(manifest.get("dataset_id"))

    return None


def write_dataset_manifest(*, dataset_type: str, dataset_id: str, manifest: dict) -> Path:
    if dataset_type not in {"features", "labels"}:
        raise ValueError("dataset_type must be either 'features' or 'labels'")

    out_dir = _project_root() / "data" / "registry" / dataset_type
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"{dataset_id}.json"

    payload = {
        **manifest,
        "dataset_id": dataset_id,
        "dataset_type": dataset_type,
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }

    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path


def build_source_entries(paths: list[Path]) -> list[dict[str, str | None]]:
    entries: list[dict[str, str | None]] = []
    for path in paths:
        parsed_range = parse_date_range_from_filename(path)
        start_iso, end_iso = (parsed_range if parsed_range is not None else (None, None))
        entries.append(
            {
                "path": _artifact_relpath(path),
                "start_at": start_iso,
                "end_at": end_iso,
            }
        )
    return entries


def timestamp_ms_to_iso(ms: int) -> str:
    return _to_utc_iso(ms)
