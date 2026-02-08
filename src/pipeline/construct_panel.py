from __future__ import annotations

import pandas as pd

from src.pipeline.make_sample_data import HEALTH_COND_COLS, DISDIF_COLS
from src.utils.io import read_parquet, write_parquet
from src.utils.validation import validate_dataframe


EXPECTED_COLS = ["pid", "wave", "age", "female", "education", "hrgpay", *HEALTH_COND_COLS, *DISDIF_COLS]


def run(config: dict) -> tuple[pd.DataFrame, dict]:
    paths = config["paths"]

    df = read_parquet(paths["clean_path"])

    validate_dataframe(
        df,
        name="clean",
        expected_cols=EXPECTED_COLS,
        unique_keys=["pid", "wave"],
        min_rows=1,
        max_missing_frac=0.0,
    )

    df = df.sort_values(["pid", "wave"]).copy()

    health_sum = df[HEALTH_COND_COLS + DISDIF_COLS].sum(axis=1)
    df["frailty"] = health_sum / (len(HEALTH_COND_COLS) + len(DISDIF_COLS))

    df["prev_frailty"] = df.groupby("pid")["frailty"].shift(1)
    df["health_status_change"] = df["frailty"] - df["prev_frailty"]

    validate_dataframe(
        df,
        name="panel",
        expected_cols=EXPECTED_COLS + ["frailty", "prev_frailty", "health_status_change"],
        unique_keys=["pid", "wave"],
        min_rows=1,
    )

    write_parquet(df, paths["panel_path"])
    metrics = {
        "rows": int(len(df)),
    }
    return df, metrics
