from __future__ import annotations

from pathlib import Path

import pandas as pd


def test_schema_columns(run_pipeline) -> None:
    root = Path(__file__).resolve().parents[1]
    panel = pd.read_parquet(root / "data/derived/panel.parquet")
    clustered = pd.read_parquet(root / "data/derived/clustered.parquet")

    required_panel = {"pid", "wave", "age", "female", "education", "hrgpay", "frailty", "prev_frailty", "health_status_change"}
    required_clustered = required_panel | {"health_type"}

    assert required_panel.issubset(set(panel.columns))
    assert required_clustered.issubset(set(clustered.columns))
