"""Tests for the real-data pipeline using a small subsample.

Run with:
    .venv/bin/python3 -m pytest tests/test_real_pipeline.py -v
"""
from __future__ import annotations

import json
import pathlib
import tempfile

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

HEALTHCOND_COLS = [f"healthcond{i}" for i in range(1, 17)]
DISDIF_COLS = [f"disdif{i}" for i in range(1, 12)]

REAL_PARQUET = pathlib.Path("data/raw/frailty_long_panel.parquet")
REAL_CSV = pathlib.Path("data/raw/frailty_long_panel.csv")
SUBSAMPLE_N_PIDS = 200  # small enough to be fast


def _find_real_data() -> pathlib.Path | None:
    if REAL_PARQUET.exists():
        return REAL_PARQUET
    if REAL_CSV.exists():
        return REAL_CSV
    return None


def _make_subsample_config(tmp: pathlib.Path, csv_path: pathlib.Path) -> dict:
    return {
        "paths": {
            "data_raw": str(tmp / "raw"),
            "data_derived": str(tmp / "derived"),
            "output": str(tmp / "output"),
            "real_csv_path": str(csv_path),
            "panel_path": str(tmp / "derived" / "panel.parquet"),
            "clustered_path": str(tmp / "derived" / "clustered.parquet"),
            "metrics_path": str(tmp / "output" / "metrics" / "metrics.json"),
            "tables_dir": str(tmp / "output" / "tables"),
            "figures_dir": str(tmp / "output" / "figures"),
            "logs_dir": str(tmp / "output" / "logs"),
        },
        "params": {
            "seed": 42,
            "mode": "real",
            "id_col": "pidp",
            "age_col": "age_dv",
            "wave_col": "wave",
            "clustering": {
                "enabled": True,
                "k": 3,
                "n_init": 5,
                "cluster_by": "pid",
                "features": ["frailty"],
            },
            "report": {
                "tables": ["tab01_summary_stats", "tab02_frailty_by_wave"],
                "figures": [
                    "fig01_frailty_trajectories",
                    "fig02_frailty_distribution",
                    "fig03_cluster_diagnostics",
                ],
                "diagnostics_k_range": [2, 5],
            },
        },
    }


@pytest.fixture(scope="module")
def subsample_csv(tmp_path_factory) -> pathlib.Path:
    """Create a small CSV subsample from the real data."""
    real_path = _find_real_data()
    if real_path is None:
        pytest.skip("Real data not available at data/raw/frailty_long_panel.{parquet,csv}")

    if str(real_path).endswith(".parquet"):
        df = pd.read_parquet(real_path)
    else:
        df = pd.read_csv(real_path)
    rng = np.random.default_rng(42)
    pids = rng.choice(df["pidp"].unique(), size=min(SUBSAMPLE_N_PIDS, df["pidp"].nunique()), replace=False)
    sub = df[df["pidp"].isin(pids)].copy()

    tmp = tmp_path_factory.mktemp("data")
    out = tmp / "frailty_subsample.csv"
    sub.to_csv(out, index=False)
    return out


@pytest.fixture(scope="module")
def pipeline_output(subsample_csv, tmp_path_factory):
    """Run the full pipeline on the subsample and return (config, tmp_dir)."""
    tmp = tmp_path_factory.mktemp("pipeline")
    cfg = _make_subsample_config(tmp, subsample_csv)

    # Ensure dirs
    for key, path in cfg["paths"].items():
        p = pathlib.Path(path)
        if key.endswith("_dir") or key in {"data_raw", "data_derived", "output"}:
            p.mkdir(parents=True, exist_ok=True)
        else:
            p.parent.mkdir(parents=True, exist_ok=True)

    from src.pipeline import ingest_real, cluster
    from src.pipeline.report_real import run as report_run

    ingest_real.run(cfg)
    cluster.run(cfg)
    report_run(cfg)

    return cfg, tmp


# ---------------------------------------------------------------------------
# Ingest tests
# ---------------------------------------------------------------------------

class TestIngest:
    def test_panel_parquet_exists(self, pipeline_output):
        cfg, _ = pipeline_output
        assert pathlib.Path(cfg["paths"]["panel_path"]).exists()

    def test_panel_has_required_columns(self, pipeline_output):
        cfg, _ = pipeline_output
        df = pd.read_parquet(cfg["paths"]["panel_path"])
        for col in ["pid", "wave", "age", "frailty", "death"] + HEALTHCOND_COLS + DISDIF_COLS:
            assert col in df.columns, f"Missing column: {col}"

    def test_no_duplicate_pid_wave(self, pipeline_output):
        cfg, _ = pipeline_output
        df = pd.read_parquet(cfg["paths"]["panel_path"])
        dupes = df.duplicated(["pid", "wave"]).sum()
        assert dupes == 0, f"Found {dupes} duplicate pid-wave pairs"

    def test_frailty_range(self, pipeline_output):
        cfg, _ = pipeline_output
        df = pd.read_parquet(cfg["paths"]["panel_path"])
        assert df["frailty"].min() >= 0.0
        assert df["frailty"].max() <= 1.0

    def test_wave_values_numeric(self, pipeline_output):
        cfg, _ = pipeline_output
        df = pd.read_parquet(cfg["paths"]["panel_path"])
        assert df["wave"].min() >= 1
        assert df["wave"].max() <= 13


# ---------------------------------------------------------------------------
# Cluster tests
# ---------------------------------------------------------------------------

class TestCluster:
    def test_clustered_parquet_exists(self, pipeline_output):
        cfg, _ = pipeline_output
        assert pathlib.Path(cfg["paths"]["clustered_path"]).exists()

    def test_health_type_column(self, pipeline_output):
        cfg, _ = pipeline_output
        df = pd.read_parquet(cfg["paths"]["clustered_path"])
        assert "health_type" in df.columns
        k = cfg["params"]["clustering"]["k"]
        assert set(df["health_type"].unique()) == set(range(1, k + 1))

    def test_type_dummies(self, pipeline_output):
        cfg, _ = pipeline_output
        df = pd.read_parquet(cfg["paths"]["clustered_path"])
        k = cfg["params"]["clustering"]["k"]
        for i in range(1, k + 1):
            col = f"type_{i}"
            assert col in df.columns
            assert set(df[col].unique()) <= {0, 1}

    def test_all_pids_assigned(self, pipeline_output):
        cfg, _ = pipeline_output
        df = pd.read_parquet(cfg["paths"]["clustered_path"])
        assert df["health_type"].isna().sum() == 0

    def test_consistent_type_per_pid(self, pipeline_output):
        """Each individual should have the same health_type across all waves."""
        cfg, _ = pipeline_output
        df = pd.read_parquet(cfg["paths"]["clustered_path"])
        types_per_pid = df.groupby("pid")["health_type"].nunique()
        assert (types_per_pid == 1).all(), "Some individuals have inconsistent health types across waves"


# ---------------------------------------------------------------------------
# Report tests
# ---------------------------------------------------------------------------

class TestReport:
    def test_tab01_exists(self, pipeline_output):
        cfg, _ = pipeline_output
        path = pathlib.Path(cfg["paths"]["tables_dir"]) / "tab01_summary_stats.csv"
        assert path.exists()
        df = pd.read_csv(path)
        assert len(df) == cfg["params"]["clustering"]["k"]
        assert "mean_frailty" in df.columns

    def test_tab02_exists(self, pipeline_output):
        cfg, _ = pipeline_output
        path = pathlib.Path(cfg["paths"]["tables_dir"]) / "tab02_frailty_by_wave.csv"
        assert path.exists()
        df = pd.read_csv(path)
        assert len(df) > 0

    def test_fig01_exists(self, pipeline_output):
        cfg, _ = pipeline_output
        path = pathlib.Path(cfg["paths"]["figures_dir"]) / "fig01_frailty_trajectories.png"
        assert path.exists()
        assert path.stat().st_size > 1000  # not empty

    def test_fig02_exists(self, pipeline_output):
        cfg, _ = pipeline_output
        path = pathlib.Path(cfg["paths"]["figures_dir"]) / "fig02_frailty_distribution.png"
        assert path.exists()
        assert path.stat().st_size > 1000

    def test_fig03_exists(self, pipeline_output):
        cfg, _ = pipeline_output
        path = pathlib.Path(cfg["paths"]["figures_dir"]) / "fig03_cluster_diagnostics.png"
        assert path.exists()
        assert path.stat().st_size > 1000
