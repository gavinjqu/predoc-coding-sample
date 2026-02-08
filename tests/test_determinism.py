from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _run_pipeline_fast() -> None:
    env = os.environ.copy()
    env["FAST"] = "1"
    subprocess.run(
        [sys.executable, "-m", "src.cli", "--config", "configs/config.yaml", "--fast"],
        cwd=ROOT,
        check=True,
        env=env,
    )


def _load_last_metrics() -> dict:
    path = ROOT / "output/metrics/metrics.json"
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data[-1]


def _strip(entry: dict) -> dict:
    return {
        "datasets": entry["datasets"],
        "clustering": {
            "k": entry["clustering"]["k"],
            "n_init": entry["clustering"]["n_init"],
            "features": entry["clustering"]["features"],
            "inertia": entry["clustering"]["inertia"],
            "silhouette": entry["clustering"]["silhouette"],
            "cluster_counts": entry["clustering"]["cluster_counts"],
        },
        "regression": entry["regression"],
    }


def test_determinism_fast() -> None:
    metrics_dir = ROOT / "output/metrics"
    if metrics_dir.exists():
        shutil.rmtree(metrics_dir)

    _run_pipeline_fast()
    first = _strip(_load_last_metrics())

    _run_pipeline_fast()
    second = _strip(_load_last_metrics())

    assert first["datasets"] == second["datasets"]
    assert first["clustering"]["k"] == second["clustering"]["k"]
    assert first["clustering"]["n_init"] == second["clustering"]["n_init"]
    assert first["clustering"]["features"] == second["clustering"]["features"]
    assert first["clustering"]["cluster_counts"] == second["clustering"]["cluster_counts"]
    assert first["clustering"]["inertia"] == pytest.approx(second["clustering"]["inertia"], rel=1e-6)
    assert first["clustering"]["silhouette"] == pytest.approx(second["clustering"]["silhouette"], rel=1e-6)

    for model in first["regression"]:
        assert first["regression"][model]["nobs"] == second["regression"][model]["nobs"]
        assert first["regression"][model]["r2"] == pytest.approx(second["regression"][model]["r2"], rel=1e-6)
        assert first["regression"][model]["adj_r2"] == pytest.approx(second["regression"][model]["adj_r2"], rel=1e-6)
