# Project Documentation

This document explains the code architecture, data wrangling challenges, research questions, and future research directions for the project "From Medical Charts to Market Forces: How Health Types Shape Labour Market Outcomes."

---

## 1. Research Questions

The project investigates three related questions using 13 waves of the UK Household Longitudinal Study (UKHLS):

1. **Are individuals characterized by distinct health types?** Using unsupervised K-Means clustering on longitudinal frailty trajectories (ages 50--60), the project identifies three primary health types with distinct health outcome trajectories.

2. **What is the association between health types and labour market outcomes?** The project examines how employment status, earnings, and human capital accumulation differ across the three health types.

3. **Can health factors predict earning potential?** The project tests whether adding health type indicators to an earnings regression meaningfully improves model fit (R-squared), assessing the predictive power of health trajectories for economic outcomes.

---

## 2. Code Architecture

The repository implements two parallel pipelines that share core logic:

### Real Data Pipeline (`run_real.sh`)

```
frailty_long_panel.parquet → [ingest] → panel.parquet → [cluster] → clustered.parquet → [report] → tables + figures
```

| Step | Module | What It Does |
|------|--------|-------------|
| Ingest | `src/pipeline/ingest_real.py` | Reads raw UKHLS parquet, renames columns, converts wave letters to numbers, computes lagged frailty and health change |
| Cluster | `src/pipeline/cluster.py` | Aggregates per-person mean frailty, runs StandardScaler + KMeans(k=3), maps labels back to full panel |
| Report | `src/pipeline/report_real.py` | Generates summary tables and diagnostic figures via `src/analysis/tables_real.py` and `src/analysis/figures_real.py` |

### Synthetic Data Pipeline (`run.sh`)

```
[generate] → raw_panel.parquet → [clean] → [construct_panel] → [cluster] → [estimate] → [report]
```

Adds synthetic data generation (`src/pipeline/make_sample_data.py`), cleaning/imputation (`src/pipeline/clean.py`), frailty construction (`src/pipeline/construct_panel.py`), and OLS regression (`src/pipeline/estimate.py`). This pipeline exists to demonstrate the full analysis workflow without requiring access to restricted UKHLS microdata.

### Archived Notebooks (`notebooks/archive/`)

Three Jupyter notebooks document the original exploratory analysis that preceded the pipeline:

- **`clean_data_long_panel.ipynb`** -- Early exploration of marital status transitions and income changes across waves.
- **`frailty_main.ipynb`** -- Core notebook that constructs the frailty index from raw UKHLS Stata files, integrates the death panel, and produces the `frailty_long_panel.parquet` that feeds the pipeline.
- **`k_means.ipynb`** -- Clustering analysis, optimal-k selection (elbow + silhouette), and descriptive analysis of employment, earnings, and demographics by cluster.

### Configuration

All paths and parameters are centralized in YAML config files (`configs/config.yaml` for synthetic, `configs/config_real.yaml` for real data). Key parameters include clustering k, random seed, feature lists, and report specifications.

### Testing

The `tests/` directory includes schema validation tests, determinism tests (same seed produces same output), smoke tests, and real-pipeline integration tests.

---

## 3. Data Wrangling Challenges

### 3.1 Harmonizing Health Condition Variables Across Waves

**The problem.** UKHLS changed how it asked about chronic health conditions across waves. There is no single consistent variable -- instead, four different variable families exist:

| Variable Family | Meaning | Waves Available |
|----------------|---------|-----------------|
| `hcond1`--`16` | "Do you currently have this condition?" | Wave a, and intermittently in later waves |
| `hcondn1`--`16` | "Do you have any new conditions since last interview?" | Wave b onward |
| `hcondever1`--`16` | "Have you ever had this condition?" | Wave j |
| `hcondnew1`--`16` | "Any new conditions?" (revised wording) | Waves k--m |

The harmonization logic in `frailty_main.ipynb` maps each wave to the appropriate source variables:

- **Wave a**: Use `hcond` only (baseline)
- **Wave b**: Use `hcondn` only (new conditions since wave a)
- **Waves c--i**: Use `hcond` OR `hcondn` (either confirms presence)
- **Wave j**: Use `hcond` OR `hcondever` (the "ever" question captures cumulative history)
- **Waves k--m**: Use `hcond` OR `hcondnew`

Additionally, `hcondever9` (Hyperthyroidism) does not exist in the UKHLS data at all, requiring special handling for `healthcond9`.

### 3.2 Forward-Filling Chronic Illness Indicators

**The problem.** Chronic conditions are by definition permanent -- once an individual reports having arthritis or diabetes, they have it for life. But UKHLS may record NaN for that condition in subsequent waves if the respondent was not asked or did not respond. A naive treatment of NaN as "no condition" would create false recoveries in the panel.

**The solution.** After constructing each `healthcond` variable per wave, the pipeline forward-fills 1s within each individual: `groupby('pidp')[condition_col].ffill()`. Once a 1 is recorded, it persists for all future waves of that person. This was implemented per-condition (16 times), and an additional forced-imputation pass was needed because the initial forward-fill did not fully propagate in all edge cases (noted in the notebook: "I noticed that healthcond is not filled correctly, so the following code forces a 1 onto following waves once a 1 has been recorded").

### 3.3 Integrating the Death Panel

**The problem.** The UKHLS main survey stops recording observations for a person once they die. Death information comes from a separate administrative linkage (the death panel), not from survey responses. This means:

- A person who dies in wave c has no rows in the main survey for waves c onward
- The death panel records the fact of death but must be merged back into the health panel
- Once merged, `death=1` must be forward-filled so that subsequent waves (if any exist due to data structure) correctly reflect mortality

**The solution.** The death dataset was reshaped to long format, merged with the health panel on `(pidp, wave)`, and `death` was forward-filled within each person. Individuals with `death=1` are assigned `frailty=1.0` for that wave and all subsequent waves.

### 3.4 Missing Age Data and Extrapolation

**The problem.** Some person-wave observations have missing age values, particularly in waves where the respondent was not directly interviewed.

**The solution.** Age was interpolated and extrapolated within each individual. Since UKHLS waves are approximately annual, a person's age increments predictably. Missing ages are filled by interpolating between known ages or extrapolating from the nearest known age.

### 3.5 Constructing the Frailty Index

**The design decision.** The frailty index is a deficit-accumulation measure: `frailty = (count of deficits present) / 27`. The 27 deficits are the 16 chronic conditions plus 11 disability indicators.

Key choices and their implications:

- **NaN treated as 0 (no deficit) before summation.** This biases frailty downward for individuals with missing health data, since unreported conditions are counted as absent. The alternative -- dropping observations with any missing health data -- would have severely reduced the sample, especially in early waves.
- **Individuals with ALL health indicators NaN across ALL waves receive frailty=NaN**, rather than frailty=0. This distinguishes genuinely healthy individuals from those with no health data at all.
- **Ages 105+ assigned frailty=1.** A pragmatic mortality-plateau assumption based on demographic literature.

### 3.6 Wide-to-Long Panel Reshaping

**The problem.** UKHLS stores data in wide format with wave-prefixed variable names (e.g., `a_hcond1`, `b_hcond1`, ..., `m_hcond1`). Each wave's file has its own naming convention. Combining 13 waves into a single long panel required:

- Reading each wave's Stata file separately
- Stripping the wave prefix from variable names
- Adding an explicit `wave` column
- Concatenating all waves and verifying that `(pidp, wave)` forms a unique key

---

## 4. Analysis Methodology

### K-Means Clustering

The project clusters individuals by their frailty trajectories over ages 50--60 (sampled at 2-year intervals). The clustering unit is the individual, not the observation -- per-person mean frailty is computed, standardized, and passed to K-Means.

**Choosing k=3:**
- **Elbow method**: Within-cluster sum of squares (WCSS) plotted against k=2--8. The elbow (diminishing marginal reduction in WCSS) occurs at k=3.
- **Silhouette analysis**: Silhouette scores (measuring intra-cluster cohesion vs. inter-cluster separation) are highest at low k, with k=3 providing a good balance of parsimony and cluster quality.

### OLS Regression (Synthetic Pipeline)

Two models estimated with HC3 heteroskedasticity-robust standard errors:

- **Basic**: `hrgpay ~ age + female + education`
- **Full**: `hrgpay ~ age + female + education + type_1 + type_2` (cluster dummies, omitting the last category)

The comparison of R-squared between models quantifies the incremental predictive power of health types for earnings.

---

## 5. Future Research Directions

This section outlines potential extensions using the UKHLS and linked administrative datasets. It is organized by audience: predoc RA interviews (demonstrating familiarity with causal inference), PhD admissions (demonstrating research maturity), and industry data science (demonstrating practical impact).

### 5.1 For Predoc RA Interviews: Causal Inference and Identification Strategies

These proposals emphasize credible identification -- the core concern of applied microeconomics.

#### A. Health Shocks and Labour Market Outcomes (Difference-in-Differences)

**Question:** What is the causal effect of an acute health shock (e.g., heart attack, stroke, cancer diagnosis) on earnings, employment, and hours worked?

**Identification:** Use the panel structure to implement a staggered difference-in-differences design. The "treatment" is a new diagnosis of a severe condition (identifiable via the transition from `healthcond=0` to `healthcond=1` for conditions 3--7, 13). The control group is individuals who have not yet experienced a shock. Pre-treatment parallel trends in earnings can be tested directly in the data.

**Why UKHLS is well-suited:** 13 waves of panel data allow long pre- and post-treatment windows. The richness of health conditions allows studying heterogeneous effects by condition severity.

**Methodological considerations:** Staggered adoption requires modern DiD estimators (Callaway and Sant'Anna 2021; Sun and Abraham 2021) to avoid negative weighting bias from two-way fixed effects.

#### B. Spousal Health Spillovers (Instrumental Variables)

**Question:** Does a spouse's health deterioration causally affect the other partner's labour supply?

**Identification:** UKHLS interviews all adults in a household, providing matched spousal data. A spouse's acute health shock (e.g., stroke) can serve as an instrument for changes in caregiving burden. The exclusion restriction is that the spouse's stroke affects the respondent's labor supply only through its effect on caregiving demands, not through correlated health behaviors (testable by conditioning on the respondent's own health trajectory).

**Why UKHLS:** Household-level sampling means both partners are observed longitudinally. Spousal `pidp` linkage is available.

#### C. Disability Onset and Benefit Take-Up (Regression Discontinuity)

**Question:** How does crossing a health threshold affect disability benefit claiming?

**Identification:** If UK disability assessments use explicit health criteria with sharp cutoffs (e.g., a minimum number of functional limitations), a regression discontinuity design around the threshold can estimate the causal effect of benefit eligibility on labour supply. The `disdif1`--`11` variables map closely to the descriptors used in the Work Capability Assessment.

**Methodological considerations:** Requires verifying that the running variable (count of functional limitations) is not manipulable by respondents -- testable via density tests (McCrary 2008).

#### D. Education and Health Trajectories (Control Function / Matching)

**Question:** Does education causally slow health deterioration in later life?

**Identification:** Exploit the 1972 raising of the school leaving age (RoSLA) in the UK as exogenous variation in education. Individuals born just after the policy cutoff were required to stay in school longer. UKHLS contains birth cohorts on both sides of the cutoff. A fuzzy RD or IV design using the RoSLA policy as an instrument for years of education can estimate its causal effect on frailty trajectories.

**Why UKHLS:** The age range of respondents spans cohorts affected by the 1972 reform.

### 5.2 For PhD Admissions: Novel Research Contributions

These proposals demonstrate the ability to formulate research questions with clear contributions to the literature.

#### A. Dynamic Health Types and Labour Market Sorting

**Contribution:** Existing literature treats health as a static control variable. This project's health-type framework opens the door to studying dynamic sorting: do individuals in deteriorating health types sort into different occupations, industries, or employment arrangements (e.g., self-employment, part-time work) over time?

**Approach:** Extend the K-Means clustering to a Hidden Markov Model (HMM) that allows individuals to transition between health types over time. Estimate transition probabilities and correlate them with occupational mobility using UKHLS employment history data.

**Contribution to literature:** Bridges the health economics literature on frailty indices with the labour economics literature on occupational sorting and job mobility.

#### B. Intergenerational Transmission of Health Disadvantage

**Contribution:** UKHLS tracks parents and children within the same household. This enables studying whether parental health types (as identified by the clustering methodology) predict children's educational attainment, early labour market entry, and health behaviors.

**Approach:** Link parent health types to child outcomes using the household structure. Control for socioeconomic confounders. Test whether the health-type channel operates above and beyond income and education transmission.

**Why novel:** Most intergenerational health studies focus on specific conditions. The health-type framework captures the multidimensional, cumulative nature of health disadvantage.

#### C. Health Types and Macroeconomic Shocks

**Contribution:** Do health-vulnerable individuals bear a disproportionate share of labour market adjustment during recessions?

**Approach:** Interact health types with macroeconomic indicators (unemployment rate, GDP growth) across UKHLS waves that span the 2008 financial crisis and the COVID-19 pandemic. Test whether the earnings and employment gaps across health types widen during downturns.

**Why UKHLS:** Waves a--m span 2009--2022, covering both the Great Recession recovery and the COVID-19 shock.

### 5.3 For Industry Data Science: Prediction and Policy

These proposals emphasize practical applications and scalable methodology.

#### A. Predicting Health Deterioration for Early Intervention

**Application:** Build a predictive model that identifies individuals at high risk of transitioning from a low-frailty to a high-frailty health type within the next 2--4 waves.

**Approach:** Use gradient-boosted trees or penalized regression with the full set of health indicators, demographics, and socioeconomic variables as features. Evaluate with time-series cross-validation (train on waves 1--t, predict wave t+1). Benchmark against the simple frailty index.

**Policy relevance:** Early identification of at-risk individuals can inform targeted public health interventions and reduce downstream healthcare costs.

#### B. Causal Machine Learning for Heterogeneous Treatment Effects

**Application:** Use causal forests (Wager and Athey 2018) to estimate heterogeneous effects of health shocks on employment, discovering which subpopulations are most affected.

**Approach:** Define treatment as onset of a new chronic condition. Use the UKHLS panel to construct pre-treatment covariates and post-treatment outcomes. Causal forests estimate individual-level treatment effects and reveal which observable characteristics (age, education, occupation, region, prior health) moderate the impact.

**Policy relevance:** Identifies groups that would benefit most from employment retention programs following health events.

### 5.4 Extensions Using Linked Administrative Data

UKHLS supports linkage to several UK administrative datasets (subject to access approval), which would substantially expand the research possibilities:

| Linked Dataset | What It Adds | Research Possibilities |
|---------------|-------------|----------------------|
| **NHS Hospital Episode Statistics (HES)** | Inpatient admissions, diagnoses (ICD-10), procedures, A&E visits | Precise health shock timing, healthcare utilization costs, condition severity beyond self-report |
| **HMRC tax records** | Exact earnings, employment spells, employer identifiers | Continuous earnings (replacing self-reported income bands), job-to-job transitions, firm-level analysis |
| **DWP benefits data** | Benefit claims (Universal Credit, ESA, PIP), sanction events | Welfare take-up responses to health shocks, interaction of health and benefit system design |
| **National Pupil Database (NPD)** | Children's school records, test scores, absences | Intergenerational effects of parental health on child human capital |
| **ONS mortality records** | Exact date and cause of death | Survival analysis, life expectancy gradients by health type |

**Example linked-data project:** Combine UKHLS health types with HMRC earnings records and HES hospital admissions to estimate the full fiscal cost of health deterioration -- capturing both the earnings loss (tax revenue reduction) and the healthcare utilization increase. This would require a causal framework (e.g., health shock DiD with precise HES timing) and would speak directly to the cost-benefit analysis of preventive health interventions.
