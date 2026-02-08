from __future__ import annotations

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

    validate_dataframe(
        df,
        name="panel_for_cluster",
        expected_cols=features,
        min_rows=k,
    )

    X = df[features].to_numpy()
    X = StandardScaler().fit_transform(X)

    model = KMeans(n_clusters=k, n_init=n_init, random_state=int(params["seed"]))
    labels = model.fit_predict(X)
    df = df.copy()
    df["health_type"] = labels + 1

    for i in range(1, k + 1):
        df[f"type_{i}"] = (df["health_type"] == i).astype(int)

    counts = df["health_type"].value_counts().to_dict()
    if len(counts) < k:
        raise ValueError(f"cluster: expected {k} clusters, got {len(counts)}")

    write_parquet(df, paths["clustered_path"])
    silhouette = None
    if k > 1 and len(df) > k:
        silhouette = float(silhouette_score(X, labels))

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
