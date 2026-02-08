from __future__ import annotations

import pandas as pd

from src.pipeline.make_sample_data import HEALTH_COND_COLS, DISDIF_COLS
from src.utils.io import read_parquet, write_parquet
from src.utils.validation import validate_dataframe


EXPECTED_COLS = ["pid", "wave", "age", "female", "education", "hrgpay", *HEALTH_COND_COLS, *DISDIF_COLS]


def run(config: dict) -> tuple[pd.DataFrame, dict]:
    paths = config["paths"]

    df = read_parquet(paths["raw_sample_path"])

    validate_dataframe(
        df,
        name="raw",
        expected_cols=EXPECTED_COLS,
        unique_keys=["pid", "wave"],
        min_rows=1,
    )

    df = df.drop_duplicates(["pid", "wave"]).copy()

    # Basic cleaning: fill missing numeric values
    df["age"] = df["age"].fillna(df["age"].median())
    df["hrgpay"] = df["hrgpay"].fillna(df["hrgpay"].median())

    for col in HEALTH_COND_COLS + DISDIF_COLS:
        df[col] = df[col].fillna(0)

    df["female"] = df["female"].fillna(0).astype(int)
    df["education"] = df["education"].fillna(df["education"].median()).astype(int)

    validate_dataframe(
        df,
        name="clean",
        expected_cols=EXPECTED_COLS,
        unique_keys=["pid", "wave"],
        min_rows=1,
        max_missing_frac=0.0,
    )

    write_parquet(df, paths["clean_path"])
    metrics = {
        "rows": int(len(df)),
    }
    return df, metrics
