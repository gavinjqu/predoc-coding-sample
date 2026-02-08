from __future__ import annotations

import argparse
import logging
import os
import pathlib
import subprocess
import uuid

from src.pipeline import make_sample_data, clean, construct_panel, cluster, estimate, report
from src.utils.config import load_config
from src.utils.metrics import log_metrics, utc_now_iso


def _ensure_dirs(cfg: dict) -> None:
    for key, path in cfg["paths"].items():
        p = pathlib.Path(path)
        if key.endswith("_dir") or key in {"data_sample", "data_derived", "output"}:
            p.mkdir(parents=True, exist_ok=True)
        else:
            p.parent.mkdir(parents=True, exist_ok=True)


def _get_git_commit() -> str | None:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except Exception:
        return None


def _apply_fast_config(cfg: dict) -> None:
    params = cfg["params"]
    params["n_individuals"] = min(int(params["n_individuals"]), 120)
    params["n_waves"] = min(int(params["n_waves"]), 3)
    params["clustering"]["n_init"] = min(int(params["clustering"]["n_init"]), 5)
    params["report"]["diagnostics_k_range"] = [2, 4]


def run_pipeline(cfg: dict, *, fast: bool) -> None:
    logging.info("Step 1: make_sample_data")
    df_raw, m_raw = make_sample_data.run(cfg)

    logging.info("Step 2: clean")
    df_clean, m_clean = clean.run(cfg)

    logging.info("Step 3: construct_panel")
    df_panel, m_panel = construct_panel.run(cfg)

    logging.info("Step 4: cluster")
    df_clustered, m_cluster = cluster.run(cfg)

    logging.info("Step 5: estimate")
    _, m_reg = estimate.run(cfg)

    logging.info("Step 6: report")
    report.run(cfg)

    metrics_entry = {
        "run_id": str(uuid.uuid4()),
        "timestamp": utc_now_iso(),
        "git_commit": _get_git_commit(),
        "fast_mode": fast,
        "seed": int(cfg["params"]["seed"]),
        "datasets": {
            "raw_rows": m_raw.get("rows", len(df_raw)),
            "clean_rows": m_clean.get("rows", len(df_clean)),
            "panel_rows": m_panel.get("rows", len(df_panel)),
            "clustered_rows": m_cluster.get("rows", len(df_clustered)),
        },
        "sample": {
            "n_individuals": m_raw.get("n_individuals"),
            "n_waves": m_raw.get("n_waves"),
        },
        "clustering": {
            "enabled": bool(cfg["params"]["clustering"]["enabled"]),
            "k": int(cfg["params"]["clustering"]["k"]),
            "n_init": int(cfg["params"]["clustering"]["n_init"]),
            "features": list(cfg["params"]["clustering"]["features"]),
            "silhouette": m_cluster.get("silhouette"),
            "inertia": m_cluster.get("inertia"),
            "cluster_counts": m_cluster.get("cluster_counts"),
        },
        "regression": m_reg,
    }

    log_metrics(metrics_entry, cfg["paths"]["metrics_path"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the research pipeline")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--fast", action="store_true", help="Run in fast mode")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    cfg = load_config(args.config)
    fast = args.fast or os.getenv("FAST") == "1"
    if fast:
        _apply_fast_config(cfg)

    _ensure_dirs(cfg)
    log_path = pathlib.Path(cfg["paths"]["logs_dir"]) / "pipeline.log"
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logging.getLogger().addHandler(file_handler)

    run_pipeline(cfg, fast=fast)


if __name__ == "__main__":
    main()
