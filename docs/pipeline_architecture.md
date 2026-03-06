# Real-Data Pipeline Architecture

## Command
```bash
./run_real.sh
# which runs: .venv/bin/python3 -m src.cli_real --config configs/config_real.yaml
```

## System-Level Flow

### 1. Entry Point — `src/cli_real.py:main()`
- Parses `--config` argument
- Loads `configs/config_real.yaml` into a dict via `yaml.safe_load`
- Creates all output directories (`data/derived/`, `output/tables/`, etc.)
- Sets up logging to both console and `output/logs/pipeline_real.log`
- Calls `run_pipeline(cfg)`

### 2. Step 1 — Ingest (`src/pipeline/ingest_real.py`)
- Reads `data/raw/frailty_long_panel.parquet` into a pandas DataFrame (~529k rows)
- Keeps only needed columns: `pidp`, `age_dv`, `wave`, `death`, `frailty`, `healthcond1-16`, `disdif1-11`
- Renames `pidp` → `pid`, `age_dv` → `age`
- Converts letter waves (`a`–`m`) to numbers (1–13)
- Drops rows with null frailty, fills missing health indicators with 0
- Computes `prev_frailty` and `health_status_change` (lagged per individual)
- Validates uniqueness on `(pid, wave)`
- Writes to `data/derived/panel.parquet`

### 3. Step 2 — Cluster (`src/pipeline/cluster.py`)
- Reads `data/derived/panel.parquet`
- Because `cluster_by: pid` is set, **aggregates frailty per individual** (mean across waves) → ~84k rows instead of 529k
- Standardizes with `StandardScaler`, runs `KMeans(k=3, n_init=10)` on the aggregated data
- Maps cluster labels (1, 2, 3) back to every row in the full panel via the `pid` key
- Adds `health_type` and dummy columns (`type_1`, `type_2`, `type_3`)
- Computes silhouette score
- Writes to `data/derived/clustered.parquet`

### 4. Step 3 — Report (`src/pipeline/report_real.py`)
Iterates through the configured tables/figures and calls each registered function:

**Tables** (`src/analysis/tables_real.py`):
- **tab01_summary_stats** — reads clustered parquet, groups by `health_type`, computes mean age/frailty/death rate → `output/tables/tab01_summary_stats.csv`
- **tab02_frailty_by_wave** — groups by `(wave, health_type)`, computes frailty stats → `output/tables/tab02_frailty_by_wave.csv`

**Figures** (`src/analysis/figures_real.py`):
- **fig01_frailty_trajectories** — plots mean frailty over waves, one line per health type → `output/figures/fig01_frailty_trajectories.png`
- **fig02_frailty_distribution** — histogram of frailty distribution per cluster → `output/figures/fig02_frailty_distribution.png`
- **fig03_cluster_diagnostics** — re-runs KMeans for k=2 through k=8 on a 5k subsample, plots inertia (elbow) and silhouette scores → `output/figures/fig03_cluster_diagnostics.png`

### 5. Metrics Collection
Back in `run_pipeline()`, collects all metrics (run ID, git commit, row counts, silhouette, cluster sizes) and appends them to `output/metrics/metrics.json`.

## Data Flow Diagram
```
data/raw/frailty_long_panel.parquet
    │
    ▼
[ingest_real] ──► data/derived/panel.parquet
    │
    ▼
[cluster]     ──► data/derived/clustered.parquet
    │
    ▼
[report]      ──► output/tables/tab01_summary_stats.csv
              ──► output/tables/tab02_frailty_by_wave.csv
              ──► output/figures/fig01_frailty_trajectories.png
              ──► output/figures/fig02_frailty_distribution.png
              ──► output/figures/fig03_cluster_diagnostics.png
              ──► output/metrics/metrics.json
```

## Configuration
All paths and parameters are controlled by `configs/config_real.yaml`. Key parameters:
- `clustering.k` — number of health types (default: 3)
- `clustering.n_init` — KMeans restarts (default: 10)
- `clustering.cluster_by` — aggregate level for clustering (default: `pid` for per-individual)
- `report.diagnostics_k_range` — range of k values for the elbow plot (default: 2–8)
