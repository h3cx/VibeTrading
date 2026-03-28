from __future__ import annotations

from io import BytesIO
from pathlib import Path
import zipfile

import pandas as pd

from trader.data.binance_aggtrades import ARCHIVE_COLUMNS, parse_decompress_day


FIXTURE_DIR = Path(__file__).parent / "data"


def _zip_fixture(csv_name: str) -> bytes:
    content = (FIXTURE_DIR / csv_name).read_bytes()
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(csv_name, content)
    return buf.getvalue()


def _parse_fixture(csv_name: str) -> pd.DataFrame:
    chunks = list(parse_decompress_day(archive_bytes=_zip_fixture(csv_name), chunksize_rows=1))
    assert chunks, f"expected parsed chunks for {csv_name}"
    return pd.concat(chunks, ignore_index=True)


def test_parse_archive_handles_header_and_malformed_rows() -> None:
    no_header = _parse_fixture("aggtrades_no_header.csv")
    with_header = _parse_fixture("aggtrades_with_header.csv")
    malformed = _parse_fixture("aggtrades_malformed.csv")

    for frame in (no_header, with_header, malformed):
        assert list(frame.columns) == ARCHIVE_COLUMNS
        assert frame["agg_trade_id"].dtype == "int64"
        assert frame["first_trade_id"].dtype == "int64"
        assert frame["last_trade_id"].dtype == "int64"
        assert frame["timestamp"].dtype == "int64"
        assert frame["price"].dtype == "float64"
        assert frame["quantity"].dtype == "float64"
        assert frame["buyer_is_maker"].dtype == "bool"

    assert no_header["agg_trade_id"].tolist() == [1001, 1002]
    assert with_header["agg_trade_id"].tolist() == [1003, 1004]
    assert malformed["agg_trade_id"].tolist() == [1005, 1006]
