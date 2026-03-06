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


def fig01_frailty_trajectories(config: dict) -> None:
    figures_dir = _ensure_figures_dir(config)
    df = read_parquet(config["paths"]["clustered_path"])

    required = ["wave", "frailty", "health_type"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        logging.warning("fig01: missing columns %s; skipping", missing)
        return

    plot_df = (
        df.groupby(["wave", "health_type"])["frailty"]
        .mean()
        .reset_index()
        .sort_values(["health_type", "wave"])
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    for ht, grp in plot_df.groupby("health_type"):
        ax.plot(grp["wave"], grp["frailty"], marker="o", label=f"Type {int(ht)}")

    ax.set_xlabel("Wave")
    ax.set_ylabel("Mean Frailty Index")
    ax.set_title("Frailty Trajectories by Health Type (Real Data)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figures_dir / "fig01_frailty_trajectories.png", dpi=150)
    plt.close(fig)
    logging.info("Saved fig01_frailty_trajectories.png")


def fig02_frailty_distribution(config: dict) -> None:
    figures_dir = _ensure_figures_dir(config)
    df = read_parquet(config["paths"]["clustered_path"])

    required = ["frailty", "health_type"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        logging.warning("fig02: missing columns %s; skipping", missing)
        return

    k = int(config["params"]["clustering"]["k"])

    fig, axes = plt.subplots(1, k, figsize=(5 * k, 4), sharey=True)
    if k == 1:
        axes = [axes]

    colors = ["#4C78A8", "#F58518", "#E45756", "#72B7B2", "#54A24B", "#EECA3B"]
    for i, ax in enumerate(axes, start=1):
        subset = df[df["health_type"] == i]["frailty"]
        ax.hist(subset, bins=40, color=colors[(i - 1) % len(colors)], edgecolor="white", alpha=0.85)
        ax.set_title(f"Type {i} (n={len(subset):,})")
        ax.set_xlabel("Frailty Index")
        if i == 1:
            ax.set_ylabel("Count")
        ax.axvline(subset.mean(), color="black", linestyle="--", linewidth=1, label=f"mean={subset.mean():.3f}")
        ax.legend(fontsize=8)

    fig.suptitle("Frailty Distribution by Health Type (Real Data)", fontsize=13)
    fig.tight_layout()
    fig.savefig(figures_dir / "fig02_frailty_distribution.png", dpi=150)
    plt.close(fig)
    logging.info("Saved fig02_frailty_distribution.png")


def fig03_cluster_diagnostics(config: dict) -> None:
    figures_dir = _ensure_figures_dir(config)
    params = config["params"]

    df = read_parquet(config["paths"]["panel_path"])
    features = list(params["clustering"]["features"])
    missing = [c for c in features if c not in df.columns]
    if missing:
        logging.warning("fig03: missing columns %s; skipping", missing)
        return

    validate_dataframe(df, name="cluster_diagnostics", expected_cols=features, min_rows=10)

    k_min, k_max = params["report"]["diagnostics_k_range"]
    k_min, k_max = int(k_min), int(k_max)

    # Subsample for speed on large datasets
    max_rows = int(params.get("diagnostics_max_rows", 5000))
    if len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=int(params["seed"]))

    k_max = min(k_max, max(2, len(df) - 1))
    if k_min > k_max:
        k_min = 2

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

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(ks, inertias, marker="o", color="#4C78A8", label="Inertia")
    ax1.set_xlabel("k (Number of Clusters)")
    ax1.set_ylabel("Inertia", color="#4C78A8")
    ax1.tick_params(axis="y", labelcolor="#4C78A8")

    ax2 = ax1.twinx()
    ax2.plot(ks, silhouettes, marker="s", color="#F58518", label="Silhouette")
    ax2.set_ylabel("Silhouette Score", color="#F58518")
    ax2.tick_params(axis="y", labelcolor="#F58518")

    ax1.set_title("Cluster Diagnostics — Elbow & Silhouette (Real Data)")
    ax1.set_xticks(ks)
    fig.tight_layout()
    fig.savefig(figures_dir / "fig03_cluster_diagnostics.png", dpi=150)
    plt.close(fig)
    logging.info("Saved fig03_cluster_diagnostics.png")


FIGURE_REGISTRY = {
    "fig01_frailty_trajectories": fig01_frailty_trajectories,
    "fig02_frailty_distribution": fig02_frailty_distribution,
    "fig03_cluster_diagnostics": fig03_cluster_diagnostics,
}
