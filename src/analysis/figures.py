from __future__ import annotations

import logging
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from src.utils.io import read_parquet
from src.utils.validation import validate_dataframe


def _ensure_figures_dir(config: dict) -> pathlib.Path:
    figures_dir = pathlib.Path(config["paths"]["figures_dir"])
    figures_dir.mkdir(parents=True, exist_ok=True)
    return figures_dir


def _safe_require(df: pd.DataFrame, cols: list[str], name: str) -> bool:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        logging.warning("%s: missing required columns %s; skipping", name, missing)
        return False
    return True


def fig01_frailty_trajectories(config: dict) -> None:
    figures_dir = _ensure_figures_dir(config)
    df = read_parquet(config["paths"]["clustered_path"])

    required = ["wave", "frailty", "health_type"]
    if not _safe_require(df, required, "fig01_frailty_trajectories"):
        return

    plot_df = (
        df.groupby(["wave", "health_type"])["frailty"]
        .mean()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(7, 4))
    for ht, grp in plot_df.groupby("health_type"):
        ax.plot(grp["wave"], grp["frailty"], marker="o", label=f"Type {int(ht)}")

    ax.set_xlabel("Wave")
    ax.set_ylabel("Mean Frailty")
    ax.set_title("Frailty Trajectories by Health Type")
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(figures_dir / "fig01_frailty_trajectories.png", dpi=150)
    plt.close(fig)


def fig02_outcomes_by_type(config: dict) -> None:
    figures_dir = _ensure_figures_dir(config)
    df = read_parquet(config["paths"]["clustered_path"])

    required = ["health_type", "hrgpay"]
    if not _safe_require(df, required, "fig02_outcomes_by_type"):
        return

    plot_df = df.groupby("health_type")["hrgpay"].mean().reset_index()

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(plot_df["health_type"].astype(int), plot_df["hrgpay"], color="#4C78A8")
    ax.set_xlabel("Health Type")
    ax.set_ylabel("Mean Hourly Pay")
    ax.set_title("Earnings by Health Type")
    ax.set_xticks(plot_df["health_type"].astype(int))
    fig.tight_layout()
    fig.savefig(figures_dir / "fig02_outcomes_by_type.png", dpi=150)
    plt.close(fig)


def fig03_cluster_diagnostics(config: dict) -> None:
    figures_dir = _ensure_figures_dir(config)
    params = config["params"]

    df = read_parquet(config["paths"]["panel_path"])
    features = list(params["clustering"]["features"])
    if not _safe_require(df, features, "fig03_cluster_diagnostics"):
        return

    validate_dataframe(df, name="cluster_diagnostics", expected_cols=features, min_rows=10)

    k_min, k_max = params["report"]["diagnostics_k_range"]
    k_min = int(k_min)
    k_max = int(k_max)

    max_rows = int(params.get("diagnostics_max_rows", 2000))
    if len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=int(params["seed"]))

    k_max = min(k_max, max(2, len(df) - 1))
    if k_min > k_max:
        k_min = 2
    if k_max < 2:
        logging.warning("fig03_cluster_diagnostics: not enough rows for diagnostics; skipping")
        return

    X = df[features].to_numpy()
    X = StandardScaler().fit_transform(X)

    silhouettes = []
    inertias = []
    ks = list(range(k_min, k_max + 1))

    for k in ks:
        model = KMeans(n_clusters=k, n_init=int(params["clustering"]["n_init"]), random_state=int(params["seed"]))
        labels = model.fit_predict(X)
        inertias.append(float(model.inertia_))
        if k > 1 and len(df) > k:
            silhouettes.append(float(silhouette_score(X, labels)))
        else:
            silhouettes.append(np.nan)

    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(ks, inertias, marker="o", color="#4C78A8", label="Inertia")
    ax1.set_xlabel("k")
    ax1.set_ylabel("Inertia", color="#4C78A8")
    ax1.tick_params(axis="y", labelcolor="#4C78A8")

    ax2 = ax1.twinx()
    ax2.plot(ks, silhouettes, marker="s", color="#F58518", label="Silhouette")
    ax2.set_ylabel("Silhouette", color="#F58518")
    ax2.tick_params(axis="y", labelcolor="#F58518")

    ax1.set_title("Cluster Diagnostics")
    fig.tight_layout()
    fig.savefig(figures_dir / "fig03_cluster_diagnostics.png", dpi=150)
    plt.close(fig)


FIGURE_REGISTRY = {
    "fig01_frailty_trajectories": fig01_frailty_trajectories,
    "fig02_outcomes_by_type": fig02_outcomes_by_type,
    "fig03_cluster_diagnostics": fig03_cluster_diagnostics,
}
