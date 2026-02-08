from __future__ import annotations

import pathlib
import pandas as pd


def ensure_parent(path: str | pathlib.Path) -> None:
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)


def read_parquet(path: str | pathlib.Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def write_parquet(df: pd.DataFrame, path: str | pathlib.Path) -> None:
    ensure_parent(path)
    df.to_parquet(path, index=False)


def write_csv(df: pd.DataFrame, path: str | pathlib.Path) -> None:
    ensure_parent(path)
    df.to_csv(path, index=False)
