from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from trader.baseline.backtest_baseline import _build_lookback_sequences_for_backtest, _load_checkpoint
from trader.baseline.backtest_baseline import _load_labels as backtest_load_labels
from trader.baseline.train_baseline import _build_lookback_sequences, _load_labels as train_load_labels
from trader.labels.label_builder import FeatureRow, _compute_label_for_current


class QARegressionTests(unittest.TestCase):
    def test_time_to_exit_is_seconds(self) -> None:
        current = FeatureRow(raw={}, timestamp_ms=0, timestamp_s=0, high=100.0, low=100.0, close=100.0)
        future = [
            FeatureRow(raw={}, timestamp_ms=300000, timestamp_s=300, high=101.0, low=99.9, close=101.0),
        ]

        result = _compute_label_for_current(
            current=current,
            future_rows=future,
            tp=0.005,
            sl=0.005,
            total_cost=0.0,
            horizon_steps=12,
            horizon_seconds=3600,
            take_profit_pct=0.5,
            stop_loss_pct=0.5,
            fee_pct=0.0,
            slippage_pct=0.0,
        )

        self.assertEqual(result["label_name"], "LONG_SETUP")
        self.assertEqual(result["time_to_exit_s"], 300.0)

    def test_train_backtest_label_preprocess_parity(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "labels.csv"
            pd.DataFrame(
                [
                    {
                        "timestamp": "2024-01-01 00:00:00+00:00",
                        "timestamp_ms": 1000,
                        "timestamp_s": 1,
                        "label": 0,
                        "label_name": "NO_TRADE",
                        "close": 100,
                        "long_net_return_pct": 0,
                        "short_net_return_pct": 0,
                        "time_to_exit_s": 1,
                        "long_event": "HORIZON",
                        "short_event": "HORIZON",
                        "exit_reason": "HORIZON",
                        "f1": 1.0,
                    },
                    {
                        "timestamp": "2024-01-01 00:00:00+00:00",
                        "timestamp_ms": 1000,
                        "timestamp_s": 1,
                        "label": 1,
                        "label_name": "LONG_SETUP",
                        "close": 100,
                        "long_net_return_pct": 1,
                        "short_net_return_pct": -1,
                        "time_to_exit_s": 1,
                        "long_event": "TP",
                        "short_event": "SL",
                        "exit_reason": "TP",
                        "f1": 2.0,
                    },
                    {
                        "timestamp": "2024-01-01 00:00:05+00:00",
                        "timestamp_ms": 5000,
                        "timestamp_s": 5,
                        "label": 2,
                        "label_name": "SHORT_SETUP",
                        "close": 99,
                        "long_net_return_pct": -1,
                        "short_net_return_pct": 1,
                        "time_to_exit_s": 1,
                        "long_event": "SL",
                        "short_event": "TP",
                        "exit_reason": "TP",
                        "f1": 3.0,
                    },
                ]
            ).to_csv(csv_path, index=False)

            train_df = train_load_labels("BTCUSDT", input_path=str(csv_path))
            backtest_df = backtest_load_labels(csv_path)

            self.assertListEqual(train_df["timestamp_ms"].tolist(), backtest_df["timestamp_ms"].tolist())
            self.assertEqual(len(train_df), len(backtest_df))

    def test_lookback_sequence_alignment_train_and_backtest(self) -> None:
        x = np.arange(30, dtype=np.float32).reshape(10, 3)
        y = np.arange(10, dtype=np.int64)
        timestamps_ms = np.arange(10, dtype=np.int64) * 1000
        lookback = 4

        train_x, train_y, train_ts = _build_lookback_sequences(
            x=x,
            y=y,
            timestamps_ms=timestamps_ms,
            lookback_window=lookback,
        )

        df = pd.DataFrame({"timestamp_ms": timestamps_ms, "label": y})
        backtest_df, backtest_x, backtest_y, backtest_ts = _build_lookback_sequences_for_backtest(
            df=df,
            x=x,
            y=y,
            timestamps_ms=timestamps_ms,
            lookback_window=lookback,
        )

        self.assertTrue(np.array_equal(train_x, backtest_x))
        self.assertTrue(np.array_equal(train_y, backtest_y))
        self.assertTrue(np.array_equal(train_ts, backtest_ts))
        self.assertEqual(len(backtest_df), len(backtest_ts))

    def test_checkpoint_alias_compatibility(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "model.pt"
            torch.save(
                {
                    "state_dict": {"dummy": torch.tensor([1.0])},
                    "input_dim": 12,
                    "feature_cols": ["f1", "f2"],
                    "standardizer_mu": [0.0, 0.0],
                    "standardizer_sigma": [1.0, 1.0],
                    "symbol": "BTCUSDT",
                },
                ckpt_path,
            )

            loaded = _load_checkpoint(ckpt_path)
            self.assertIn("model_state_dict", loaded)
            self.assertIn("feature_columns", loaded)
            self.assertIn("standardizer_mean", loaded)
            self.assertIn("standardizer_std", loaded)


if __name__ == "__main__":
    unittest.main()
