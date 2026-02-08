from __future__ import annotations

from pathlib import Path


def test_smoke_outputs(run_pipeline) -> None:
    root = Path(__file__).resolve().parents[1]
    expected = [
        "output/tables/tab01_summary_stats.csv",
        "output/tables/tab02_main_regression.csv",
        "output/tables/tab03_robustness.csv",
        "output/figures/fig01_frailty_trajectories.png",
        "output/figures/fig02_outcomes_by_type.png",
        "output/figures/fig03_cluster_diagnostics.png",
        "output/metrics/metrics.json",
        "output/logs/pipeline.log",
    ]

    for rel in expected:
        assert (root / rel).exists(), f"Missing {rel}"
