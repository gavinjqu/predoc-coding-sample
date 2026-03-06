from __future__ import annotations

import argparse
import logging
import os
import pathlib
import uuid

from src.pipeline import ingest_real, cluster
from src.pipeline.report_real import run as report_run
from src.utils.config import load_config
from src.utils.metrics import log_metrics, utc_now_iso


def _ensure_dirs(cfg: dict) -> None:
    for key, path in cfg["paths"].items():
        p = pathlib.Path(path)
        if key.endswith("_dir") or key in {"data_raw", "data_derived", "output"}:
            p.mkdir(parents=True, exist_ok=True)
        else:
            p.parent.mkdir(parents=True, exist_ok=True)


def _get_git_commit() -> str | None:
    import subprocess
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except Exception:
        return None


def run_pipeline(cfg: dict) -> None:
    logging.info("Step 1: ingest real data")
    df_panel, m_ingest = ingest_real.run(cfg)

    logging.info("Step 2: cluster")
    df_clustered, m_cluster = cluster.run(cfg)

    logging.info("Step 3: report")
    report_run(cfg)

    metrics_entry = {
        "run_id": str(uuid.uuid4()),
        "timestamp": utc_now_iso(),
        "git_commit": _get_git_commit(),
        "mode": "real",
        "seed": int(cfg["params"]["seed"]),
        "datasets": {
            "panel_rows": m_ingest.get("rows"),
            "n_individuals": m_ingest.get("n_individuals"),
            "n_waves": m_ingest.get("n_waves"),
            "clustered_rows": m_cluster.get("rows"),
        },
        "frailty": {
            "mean": m_ingest.get("frailty_mean"),
            "median": m_ingest.get("frailty_median"),
            "death_count": m_ingest.get("death_count"),
        },
        "clustering": {
            "enabled": bool(cfg["params"]["clustering"]["enabled"]),
            "k": int(cfg["params"]["clustering"]["k"]),
            "silhouette": m_cluster.get("silhouette"),
            "inertia": m_cluster.get("inertia"),
            "cluster_counts": m_cluster.get("cluster_counts"),
        },
    }

    log_metrics(metrics_entry, cfg["paths"]["metrics_path"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the real-data pipeline")
    parser.add_argument("--config", default="configs/config_real.yaml")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    cfg = load_config(args.config)

    _ensure_dirs(cfg)
    log_path = pathlib.Path(cfg["paths"]["logs_dir"]) / "pipeline_real.log"
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logging.getLogger().addHandler(file_handler)

    run_pipeline(cfg)
    logging.info("Pipeline complete. Outputs in %s", cfg["paths"]["output"])


if __name__ == "__main__":
    main()
