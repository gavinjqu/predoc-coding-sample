from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.io import write_parquet
from src.utils.seed import set_seed
from src.utils.validation import validate_dataframe


HEALTH_COND_COLS = [f"healthcond{i}" for i in range(1, 17)]
DISDIF_COLS = [f"disdif{i}" for i in range(1, 12)]


def run(config: dict) -> tuple[pd.DataFrame, dict]:
    params = config["params"]
    paths = config["paths"]

    seed = int(params["seed"])
    n_individuals = int(params["n_individuals"])
    n_waves = int(params["n_waves"])
    missing_rate = float(params["missing_rate"])

    set_seed(seed)
    rng = np.random.default_rng(seed)

    pid = np.arange(1, n_individuals + 1)
    base_age = rng.integers(25, 56, size=n_individuals)
    female = rng.integers(0, 2, size=n_individuals)
    education = rng.integers(0, 3, size=n_individuals)

    rows = []
    for wave in range(1, n_waves + 1):
        age = base_age + (wave - 1)
        health_prob = np.clip(0.05 + 0.01 * (age - 25), 0.05, 0.4)
        dis_prob = np.clip(0.03 + 0.008 * (age - 25), 0.03, 0.3)

        health = rng.binomial(1, health_prob[:, None], size=(n_individuals, len(HEALTH_COND_COLS)))
        dis = rng.binomial(1, dis_prob[:, None], size=(n_individuals, len(DISDIF_COLS)))

        frailty = (health.sum(axis=1) + dis.sum(axis=1)) / (len(HEALTH_COND_COLS) + len(DISDIF_COLS))
        base_pay = rng.normal(25, 5, size=n_individuals)
        hrgpay = base_pay + 3 * education - 10 * frailty + rng.normal(0, 2, size=n_individuals)
        hrgpay = np.maximum(hrgpay, 5)

        df_wave = pd.DataFrame({
            "pid": pid,
            "wave": wave,
            "age": age,
            "female": female,
            "education": education,
            "hrgpay": hrgpay,
        })
        df_wave[HEALTH_COND_COLS] = health
        df_wave[DISDIF_COLS] = dis
        rows.append(df_wave)

    df = pd.concat(rows, ignore_index=True)

    for col in ["hrgpay", *HEALTH_COND_COLS, *DISDIF_COLS]:
        mask = rng.random(len(df)) < missing_rate
        df.loc[mask, col] = np.nan

    validate_dataframe(
        df,
        name="sample_raw",
        expected_cols=["pid", "wave", "age", "female", "education", "hrgpay", *HEALTH_COND_COLS, *DISDIF_COLS],
        unique_keys=["pid", "wave"],
        min_rows=n_individuals * n_waves,
    )

    write_parquet(df, paths["raw_sample_path"])
    metrics = {
        "rows": int(len(df)),
        "n_individuals": n_individuals,
        "n_waves": n_waves,
    }
    return df, metrics
