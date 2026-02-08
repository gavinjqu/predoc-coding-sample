# predoc-coding-sample

Clean, reproducible research pipeline for an applied micro / health & labor project. The repo runs end‑to‑end on **synthetic** data and produces one table and one figure.

**Pipeline steps**
1. `make_sample_data`
2. `clean`
3. `construct_panel`
4. `cluster` (optional)
5. `estimate`
6. `report`

## Setup
- Python 3.11+
- Install deps (from `pyproject.toml`)

## Run
```bash
./run.sh
# or
python3 -m src.cli --config configs/config.yaml
```

## Outputs
Tables (default):
- `output/tables/tab01_summary_stats.csv`
- `output/tables/tab02_main_regression.csv`
- `output/tables/tab03_robustness.csv`

Figures (default):
- `output/figures/fig01_frailty_trajectories.png`
- `output/figures/fig02_outcomes_by_type.png`
- `output/figures/fig03_cluster_diagnostics.png`

Metrics:
- `output/metrics/metrics.json`
- `output/logs/pipeline.log`

## Fast mode
Run a smaller, faster version for tests or quick checks:
```bash
FAST=1 ./run.sh
# or
python3 -m src.cli --config configs/config.yaml --fast
```

## Notes
- All data in `data/sample/` are synthetic; no dissertation data are included.
- Notebooks are archived in `notebooks/archive/` for provenance only; the pipeline does not depend on them.
