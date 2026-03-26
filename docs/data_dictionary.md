# Data Dictionary

This document catalogs every variable used across the raw UKHLS data, the constructed panel, and the pipeline outputs. Variables are grouped by stage of the data pipeline.

---

## Source: UK Household Longitudinal Study (UKHLS)

The raw data comes from waves a--m (13 waves) of the UKHLS, the successor to the British Household Panel Survey (BHPS, 1991--2008). Raw Stata files are read per-wave and stacked into a single long panel.

---

## 1. Identifiers and Demographics

| Variable | Type | Source | Description |
|----------|------|--------|-------------|
| `pidp` | int | UKHLS | Cross-wave person identifier (stable across all waves) |
| `pid` | int | Pipeline alias | Renamed from `pidp` during ingestion |
| `wave` | str &rarr; int | UKHLS | Survey wave. Raw: letter `a`--`m`. Pipeline converts to numeric `1`--`13` |
| `age_dv` | int | UKHLS (derived) | Age at interview date |
| `age` | int | Pipeline alias | Renamed from `age_dv` during ingestion |
| `female` | int (0/1) | Synthetic only | Female indicator. Synthetic: `Binomial(p=0.5)` |
| `education` | int (0--2) | Synthetic only | Education level (0=low, 1=mid, 2=high). Synthetic: `Uniform(0,3)` |
| `sex` | int (1/2) | UKHLS | 1=Male, 2=Female. Used in notebook analysis |

## 2. Health Condition Indicators (`healthcond1`--`healthcond16`)

Binary (0/1) indicators for chronic health conditions. Constructed from multiple raw UKHLS variables (`hcond`, `hcondn`, `hcondever`, `hcondnew`) that vary by wave -- see Section 5 of the project documentation for details on the harmonization logic.

| Variable | Condition |
|----------|-----------|
| `healthcond1` | Asthma |
| `healthcond2` | Arthritis |
| `healthcond3` | Congestive heart failure |
| `healthcond4` | Coronary heart disease |
| `healthcond5` | Angina |
| `healthcond6` | Heart attack or myocardial infarction |
| `healthcond7` | Stroke |
| `healthcond8` | Emphysema |
| `healthcond9` | Hyperthyroidism or an over-active thyroid |
| `healthcond10` | Hypothyroidism or an under-active thyroid |
| `healthcond11` | Chronic bronchitis |
| `healthcond12` | Any kind of liver condition |
| `healthcond13` | Cancer or malignancy |
| `healthcond14` | Diabetes |
| `healthcond15` | Epilepsy |
| `healthcond16` | High blood pressure |

**Note:** `hcondever9` (Hyperthyroidism) does not exist in the UKHLS data for waves that use the `hcondever` prefix. `healthcond9` is constructed from available waves only.

## 3. Disability/Difficulty Indicators (`disdif1`--`disdif11`)

Binary (0/1) indicators for functional limitations.

| Variable | Difficulty |
|----------|-----------|
| `disdif1` | Mobility (moving around at home and walking) |
| `disdif2` | Lifting, carrying or moving objects |
| `disdif3` | Manual dexterity (using hands for everyday tasks) |
| `disdif4` | Continence (bladder and bowel control) |
| `disdif5` | Hearing (apart from using a standard hearing aid) |
| `disdif6` | Sight (apart from wearing standard glasses) |
| `disdif7` | Communication or speech problems |
| `disdif8` | Memory or ability to concentrate, learn or understand |
| `disdif9` | Recognising when in physical danger |
| `disdif10` | Physical co-ordination (balance) |
| `disdif11` | Difficulties with own personal care |

## 4. Mortality

| Variable | Type | Source | Description |
|----------|------|--------|-------------|
| `death` | int (0/1) | UKHLS death panel | Mortality indicator. Set to 1 from the wave of death onward (forward-filled within person). 4,444 individuals in the dataset have `death=1` |

## 5. Derived Variables (Pipeline)

| Variable | Type | Calculation | Description |
|----------|------|-------------|-------------|
| `frailty` | float [0, 1] | `sum(healthcond1..16, disdif1..11) / 27` | Frailty index. 0 = no deficits, 1 = all deficits (or death). NaN &rarr; 0 before summation. Overridden to 1.0 if `death=1` |
| `prev_frailty` | float | `groupby(pid).frailty.shift(1)` | Lagged frailty (same person, previous wave) |
| `health_status_change` | float | `frailty - prev_frailty` | Wave-over-wave change in frailty |
| `health_type` | int (1--k) | K-Means cluster label + 1 | Health trajectory cluster assignment. Default k=3 |
| `type_1`, `type_2`, ..., `type_k` | int (0/1) | `(health_type == i)` | One-hot indicators for each cluster |

## 6. Outcome Variables (Synthetic Pipeline Only)

| Variable | Type | Source | Description |
|----------|------|--------|-------------|
| `hrgpay` | float | Synthetic | Hourly gross pay. Generated as `base_pay + 3*education - 10*frailty + noise`, clamped to min 5. Missing values imputed with median during cleaning |

## 7. Marital Status and Income (Notebook Analysis Only)

These variables appear in `clean_data_long_panel.ipynb` and are not part of the production pipeline.

| Variable | Type | Description |
|----------|------|-------------|
| `MaritalStatus` | str | Mapped from UKHLS codes: single_never_married, married, registered_partnership, separated, divorced, widowed, etc. |
| `MaritalStatus_*` | int (0/1) | One-hot dummies for each marital status |
| `Income` | int (0--14) | Income band code from UKHLS |
| `Income_Label` | str | Human-readable income bracket (e.g., "1300_to_2099") |
| `Income_Change` | float | `groupby(pidp).Income.diff()` -- change in income band across waves |

## 8. Regression Output Schema

The regression results table (`regression_results.parquet`) has a different structure from the panel data:

| Column | Type | Description |
|--------|------|-------------|
| `model` | str | Model name: `"basic"` or `"full"` (with cluster dummies) |
| `term` | str | Coefficient name (e.g., `age`, `female`, `type_1`) |
| `coef` | float | Point estimate |
| `std_err` | float | HC3 robust standard error |
| `p_value` | float | Two-sided p-value |
| `nobs` | int | Number of observations |

## 9. Pipeline Output Files

| File | Stage | Key Columns | Rows (Real Data) |
|------|-------|-------------|-----------------|
| `data/raw/frailty_long_panel.parquet` | Raw | pidp, wave, age_dv, death, frailty, healthcond1--16, disdif1--11 | ~529K |
| `data/derived/panel.parquet` | After ingest | pid, wave, age, frailty, prev_frailty, health_status_change, death, healthcond1--16, disdif1--11 | ~529K (after dropping NaN frailty) |
| `data/derived/clustered.parquet` | After cluster | All panel columns + health_type, type_1..type_k | Same as panel |
| `data/derived/regression_results.parquet` | After estimate | model, term, coef, std_err, p_value, nobs | Synthetic only |
| `output/tables/tab01_summary_stats.csv` | Report | health_type, n_obs, n_individuals, mean_age, mean_frailty, death_rate | k rows |
| `output/tables/tab02_frailty_by_wave.csv` | Report | wave, health_type, n_obs, mean_frailty, std_frailty | k * 13 rows |
| `output/metrics/metrics.json` | Report | Run metadata: run_id, timestamp, git_commit, cluster stats, frailty stats | Appended per run |
