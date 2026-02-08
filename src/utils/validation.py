from __future__ import annotations

from typing import Iterable
import pandas as pd


def validate_dataframe(
    df: pd.DataFrame,
    *,
    name: str,
    expected_cols: Iterable[str] | None = None,
    unique_keys: Iterable[str] | None = None,
    min_rows: int | None = None,
    max_missing_frac: float | None = None,
) -> None:
    """Validate basic integrity checks for a dataframe.

    Raises ValueError if checks fail.
    """
    if expected_cols is not None:
        missing = [c for c in expected_cols if c not in df.columns]
        if missing:
            raise ValueError(f"{name}: missing columns: {missing}")

    if min_rows is not None and len(df) < min_rows:
        raise ValueError(f"{name}: expected at least {min_rows} rows, got {len(df)}")

    if unique_keys is not None:
        if df[list(unique_keys)].isna().any().any():
            raise ValueError(f"{name}: nulls found in unique keys {list(unique_keys)}")
        dupes = df.duplicated(list(unique_keys)).sum()
        if dupes:
            raise ValueError(f"{name}: found {dupes} duplicate rows on keys {list(unique_keys)}")

    if max_missing_frac is not None:
        missing_frac = df.isna().mean()
        too_missing = missing_frac[missing_frac > max_missing_frac]
        if not too_missing.empty:
            cols = {k: float(v) for k, v in too_missing.items()}
            raise ValueError(f"{name}: columns exceed missingness threshold: {cols}")
