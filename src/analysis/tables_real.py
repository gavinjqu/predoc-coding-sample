from __future__ import annotations

import logging
import pathlib

import pandas as pd

from src.utils.io import read_parquet, write_csv


def _ensure_tables_dir(config: dict) -> pathlib.Path:
    tables_dir = pathlib.Path(config["paths"]["tables_dir"])
    tables_dir.mkdir(parents=True, exist_ok=True)
    return tables_dir


def tab01_summary_stats(config: dict) -> pd.DataFrame | None:
    tables_dir = _ensure_tables_dir(config)
    df = read_parquet(config["paths"]["clustered_path"])

    required = ["pid", "health_type", "age", "frailty", "death"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        logging.warning("tab01_summary_stats: missing columns %s; skipping", missing)
        return None

    summary = (
        df.groupby("health_type")
        .agg(
            n_obs=("pid", "count"),
            n_individuals=("pid", "nunique"),
            mean_age=("age", "mean"),
            mean_frailty=("frailty", "mean"),
            std_frailty=("frailty", "std"),
            median_frailty=("frailty", "median"),
            death_rate=("death", "mean"),
        )
        .reset_index()
    )

    path = tables_dir / "tab01_summary_stats.csv"
    write_csv(summary, path)
    logging.info("Wrote %s", path)
    return summary


def tab02_frailty_by_wave(config: dict) -> pd.DataFrame | None:
    tables_dir = _ensure_tables_dir(config)
    df = read_parquet(config["paths"]["clustered_path"])

    required = ["wave", "health_type", "frailty"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        logging.warning("tab02_frailty_by_wave: missing columns %s; skipping", missing)
        return None

    summary = (
        df.groupby(["wave", "health_type"])
        .agg(
            n_obs=("frailty", "count"),
            mean_frailty=("frailty", "mean"),
            std_frailty=("frailty", "std"),
        )
        .reset_index()
        .sort_values(["wave", "health_type"])
    )

    path = tables_dir / "tab02_frailty_by_wave.csv"
    write_csv(summary, path)
    logging.info("Wrote %s", path)
    return summary


TABLE_REGISTRY = {
    "tab01_summary_stats": tab01_summary_stats,
    "tab02_frailty_by_wave": tab02_frailty_by_wave,
}
