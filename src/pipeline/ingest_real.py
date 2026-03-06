from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.utils.io import write_parquet
from src.utils.validation import validate_dataframe

HEALTHCOND_COLS = [f"healthcond{i}" for i in range(1, 17)]
DISDIF_COLS = [f"disdif{i}" for i in range(1, 12)]
WAVE_ORDER = {letter: i + 1 for i, letter in enumerate("abcdefghijklm")}


def run(config: dict) -> tuple[pd.DataFrame, dict]:
    paths = config["paths"]
    params = config["params"]

    id_col = params["id_col"]
    age_col = params["age_col"]
    wave_col = params["wave_col"]

    data_path = paths.get("real_data_path", paths.get("real_csv_path"))
    logging.info("Reading real data from %s", data_path)
    if str(data_path).endswith(".parquet"):
        df = pd.read_parquet(data_path)
    else:
        df = pd.read_csv(data_path)

    # Keep only the columns the pipeline needs
    keep_cols = [id_col, age_col, wave_col, "death", "frailty"] + HEALTHCOND_COLS + DISDIF_COLS
    missing = [c for c in keep_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Real data missing required columns: {missing}")
    df = df[keep_cols].copy()

    # Rename to pipeline-standard names
    df = df.rename(columns={id_col: "pid", age_col: "age"})

    # Convert letter waves to numeric
    df["wave"] = df[wave_col].map(WAVE_ORDER) if wave_col != "wave" else df["wave"].map(WAVE_ORDER)

    # Drop rows with no frailty and no health data at all
    df = df.dropna(subset=["frailty"])

    # Sort
    df = df.sort_values(["pid", "wave"]).reset_index(drop=True)

    # Fill missing health indicators with 0 (consistent with notebook approach)
    for col in HEALTHCOND_COLS + DISDIF_COLS:
        df[col] = df[col].fillna(0)

    # Compute lagged frailty and health status change (same as construct_panel)
    df["prev_frailty"] = df.groupby("pid")["frailty"].shift(1)
    df["health_status_change"] = df["frailty"] - df["prev_frailty"]

    validate_dataframe(
        df,
        name="real_panel",
        expected_cols=["pid", "wave", "age", "frailty", "death"] + HEALTHCOND_COLS + DISDIF_COLS,
        unique_keys=["pid", "wave"],
        min_rows=100,
    )

    write_parquet(df, paths["panel_path"])

    n_individuals = df["pid"].nunique()
    n_waves = df["wave"].nunique()
    logging.info("Ingested %d rows, %d individuals, %d waves", len(df), n_individuals, n_waves)

    metrics = {
        "rows": int(len(df)),
        "n_individuals": int(n_individuals),
        "n_waves": int(n_waves),
        "frailty_mean": float(df["frailty"].mean()),
        "frailty_median": float(df["frailty"].median()),
        "death_count": int(df["death"].sum()),
    }
    return df, metrics
