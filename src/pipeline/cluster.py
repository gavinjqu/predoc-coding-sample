from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from src.utils.io import read_parquet, write_parquet
from src.utils.seed import set_seed
from src.utils.validation import validate_dataframe


def run(config: dict) -> tuple[pd.DataFrame, dict]:
    params = config["params"]
    paths = config["paths"]

    df = read_parquet(paths["panel_path"])

    if not params["clustering"]["enabled"]:
        write_parquet(df, paths["clustered_path"])
        return df, {"rows": int(len(df))}

    set_seed(int(params["seed"]))
    k = int(params["clustering"]["k"])
    n_init = int(params["clustering"]["n_init"])
    features = list(params["clustering"]["features"])
    cluster_by = params["clustering"].get("cluster_by")  # e.g. "pid" to aggregate per individual

    validate_dataframe(
        df,
        name="panel_for_cluster",
        expected_cols=features,
        min_rows=k,
    )

    if cluster_by and cluster_by in df.columns:
        # Aggregate features per individual, cluster on means, then map back
        logging.info("Clustering by %s-level means (%d groups)", cluster_by, df[cluster_by].nunique())
        agg_df = df.groupby(cluster_by)[features].mean().reset_index()
        X = agg_df[features].to_numpy()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = KMeans(n_clusters=k, n_init=n_init, random_state=int(params["seed"]))
        agg_df["health_type"] = model.fit_predict(X_scaled) + 1

        # Map labels back to full panel
        label_map = agg_df.set_index(cluster_by)["health_type"]
        df = df.copy()
        df["health_type"] = df[cluster_by].map(label_map)

        # Silhouette on aggregated data
        silhouette = None
        if k > 1 and len(agg_df) > k:
            silhouette = float(silhouette_score(X_scaled, agg_df["health_type"].values - 1))
    else:
        # Cluster on every row (original behavior, fine for small data)
        X = df[features].to_numpy()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = KMeans(n_clusters=k, n_init=n_init, random_state=int(params["seed"]))
        labels = model.fit_predict(X_scaled)
        df = df.copy()
        df["health_type"] = labels + 1

        silhouette = None
        if k > 1 and len(df) > k:
            silhouette = float(silhouette_score(X_scaled, labels))

    for i in range(1, k + 1):
        df[f"type_{i}"] = (df["health_type"] == i).astype(int)

    counts = df["health_type"].value_counts().to_dict()
    if len(counts) < k:
        raise ValueError(f"cluster: expected {k} clusters, got {len(counts)}")

    write_parquet(df, paths["clustered_path"])

    metrics = {
        "rows": int(len(df)),
        "k": k,
        "n_init": n_init,
        "features": features,
        "inertia": float(model.inertia_),
        "silhouette": silhouette,
        "cluster_counts": {str(k): int(v) for k, v in counts.items()},
    }
    return df, metrics
