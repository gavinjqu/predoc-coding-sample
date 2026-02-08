from __future__ import annotations

import pandas as pd
import statsmodels.formula.api as smf

from src.utils.io import read_parquet, write_parquet
from src.utils.validation import validate_dataframe


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


def run(config: dict) -> tuple[pd.DataFrame, dict]:
    params = config["params"]
    paths = config["paths"]

    df = read_parquet(paths["clustered_path"])

    outcome = params["regression"]["outcome"]
    controls = list(params["regression"]["controls"])
    include_cluster = bool(params["regression"]["include_cluster"])

    validate_dataframe(
        df,
        name="clustered_for_estimate",
        expected_cols=[outcome, *controls],
        min_rows=10,
    )

    basic_formula = f"{outcome} ~ " + " + ".join(controls)
    results_basic = smf.ols(basic_formula, data=df).fit(cov_type="HC3")
    table = _results_to_table("basic", results_basic)
    metrics = {
        "basic": {
            "r2": float(results_basic.rsquared),
            "adj_r2": float(results_basic.rsquared_adj),
            "nobs": int(results_basic.nobs),
        }
    }

    if include_cluster:
        type_cols = [c for c in df.columns if c.startswith("type_")]
        type_cols = sorted(type_cols)
        if len(type_cols) >= 2:
            full_formula = basic_formula + " + " + " + ".join(type_cols[:-1])
            results_full = smf.ols(full_formula, data=df).fit(cov_type="HC3")
            table = pd.concat([table, _results_to_table("full", results_full)], ignore_index=True)
            metrics["full"] = {
                "r2": float(results_full.rsquared),
                "adj_r2": float(results_full.rsquared_adj),
                "nobs": int(results_full.nobs),
            }

    write_parquet(table, paths["regression_path"])
    return table, metrics
