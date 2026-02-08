from __future__ import annotations

import logging
import pathlib
import pandas as pd
import statsmodels.formula.api as smf

from src.utils.io import read_parquet, write_csv
from src.utils.validation import validate_dataframe


def _ensure_tables_dir(config: dict) -> pathlib.Path:
    tables_dir = pathlib.Path(config["paths"]["tables_dir"])
    tables_dir.mkdir(parents=True, exist_ok=True)
    return tables_dir


def _safe_require(df: pd.DataFrame, cols: list[str], name: str) -> bool:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        logging.warning("%s: missing required columns %s; skipping", name, missing)
        return False
    return True


def _results_to_table(model_name: str, results) -> pd.DataFrame:
    params = results.params
    se = results.bse
    pvals = results.pvalues
    table = pd.DataFrame({
        "model": model_name,
        "term": params.index,
        "coef": params.values,
        "std_err": se.values,
        "p_value": pvals.values,
        "nobs": int(results.nobs),
    })
    return table


def tab01_summary_stats(config: dict) -> pd.DataFrame | None:
    tables_dir = _ensure_tables_dir(config)
    df = read_parquet(config["paths"]["clustered_path"])

    required = ["pid", "health_type", "age", "hrgpay", "frailty"]
    if not _safe_require(df, required, "tab01_summary_stats"):
        return None

    summary = (
        df.groupby("health_type")
        .agg(
            n_obs=("pid", "count"),
            mean_age=("age", "mean"),
            mean_hrgpay=("hrgpay", "mean"),
            mean_frailty=("frailty", "mean"),
        )
        .reset_index()
    )

    path = tables_dir / "tab01_summary_stats.csv"
    write_csv(summary, path)
    return summary


def tab02_main_regression(config: dict) -> pd.DataFrame | None:
    tables_dir = _ensure_tables_dir(config)
    reg_path = pathlib.Path(config["paths"]["regression_path"])
    if not reg_path.exists():
        logging.warning("tab02_main_regression: missing regression results at %s; skipping", reg_path)
        return None

    df = read_parquet(reg_path)
    path = tables_dir / "tab02_main_regression.csv"
    write_csv(df, path)
    return df


def tab03_robustness(config: dict) -> pd.DataFrame | None:
    tables_dir = _ensure_tables_dir(config)
    df = read_parquet(config["paths"]["clustered_path"])

    params = config["params"]["regression"]
    outcome = params["outcome"]
    controls = list(params["controls"])

    required = [outcome, *controls, "frailty", "wave"]
    if not _safe_require(df, required, "tab03_robustness"):
        return None

    validate_dataframe(df, name="robustness_data", expected_cols=required, min_rows=10)

    formula = f"{outcome} ~ frailty + " + " + ".join(controls) + " + C(wave)"
    results = smf.ols(formula, data=df).fit(cov_type="HC3")
    table = _results_to_table("robustness", results)

    path = tables_dir / "tab03_robustness.csv"
    write_csv(table, path)
    return table


TABLE_REGISTRY = {
    "tab01_summary_stats": tab01_summary_stats,
    "tab02_main_regression": tab02_main_regression,
    "tab03_robustness": tab03_robustness,
}
