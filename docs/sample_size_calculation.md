# Sample Size Calculation  Worked Method

This document explains, step by step, how the sample size figures in `PROTOCOL.md` (Section 4.4) were derived, and how to reproduce them with `scripts/00_sample_size_calculation.py`.

## Method

We use the three-criterion approach of Riley et al. (2020, *BMJ*, 368:m441) for developing a binary-outcome clinical prediction model  the same method cited in Nuryunarsih et al. (2025) for the hypertension model.

**Criterion (i)  limit overfitting/shrinkage** (binding constraint in our scenarios):

```
n ≥ p / [(S − 1) × ln(1 − R²/S)]
```

**Criterion (ii)  small (≤0.05) difference between apparent and optimism-adjusted R²:**

```
n ≥ (R² × p) / δ
```

**Criterion (iii)  precise estimation of overall outcome prevalence:**

```
n ≥ (z/δ)² × φ × (1 − φ)
```

Where `p` = number of candidate predictor parameters, `S` = target shrinkage factor, `R²` = anticipated Cox-Snell R², `δ` = target precision (0.05), `φ` = anticipated outcome prevalence, `z` = 1.96 (95% CI).

The required sample size is the **maximum** of n from criteria (i), (ii), and (iii).

## Converting a target C-statistic to R²

Criteria (i) and (ii) require an anticipated Cox-Snell R², which is not something investigators intuitively specify  it's easier to reason about anticipated discrimination (C-statistic / AUC). We derive R² from a target C-statistic via Monte Carlo simulation (the same logic underlying the `pmsampsize` package):

1. Generate a large synthetic population with a linear predictor `LP ~ N(0, σ²)`.
2. Calibrate the intercept `β0` so the simulated population hits the target prevalence `φ` exactly (via root-finding).
3. Search `σ` (again via root-finding) so the resulting model achieves the target C-statistic when outcomes are simulated from `p = expit(β0 + LP)`.
4. Once calibrated, compute the Cox-Snell R² directly from the true (data-generating) model's log-likelihood vs. the null model's log-likelihood.

This is implemented in `scripts/00_sample_size_calculation.py` (function `find_sigma_for_c`).

## Worked example (base case: 30 predictors)

With prevalence φ = 0.20 and target C-statistic = 0.80, simulation gives **R² ≈ 0.181**.

Plugging into criterion (i) with S = 0.90:

```
R²/S        = 0.181 / 0.90       = 0.2011
1 − R²/S    = 1 − 0.2011         = 0.7989
ln(0.7989)                       = −0.2246
(S − 1) × ln(...) = (−0.10) × (−0.2246) = 0.02246

n ≈ p / 0.02246 ≈ p × 44.5
```

So **each additional predictor parameter costs ≈44–45 participants** at these assumptions. This "cost per predictor" is the single most useful number for explaining the calculation to a non-statistical audience (e.g., a funding panel).

## Results table

| Candidate predictors (p) | Criterion (i) | Criterion (ii) | Criterion (iii) | **Required n** | EPP |
|---|---|---|---|---|---|
| 20 | 891 | 72 | 246 | **892** | 8.9 |
| 25 | 1,108 | 91 | 246 | **1,109** | 8.9 |
| 30 | 1,330 | 109 | 246 | **1,330** | 8.9 |
| 40 | 1,763 | 146 | 246 | **1,763** | 8.8 |
| 50 | 2,211 | 182 | 246 | **2,211** | 8.8 |

**Important:** `n` is the **total number of participants**, not the number of "cases" (people with cognitive impairment). At φ=0.20: n=1,330 → ≈266 cases and ≈1,064 non-cases. EPP (events-per-parameter) = cases ÷ predictors = 266 ÷ 30 ≈ 8.9, which is compared to the traditional EPP≥10 rule of thumb.

## Sensitivity to target C-statistic (p = 30, prevalence = 0.20)

| Target C-statistic | Anticipated R² | Cost per predictor | Required n |
|---|---|---|---|
| 0.70 (fair) | 0.079 | ≈109/predictor | 3,267 |
| 0.75 | 0.126 | ≈66/predictor | 1,988 |
| 0.80 (good) | 0.181 | ≈44/predictor | 1,333 |
| 0.85 | 0.248 | ≈31/predictor | 931 |
| 0.90 (excellent) | 0.329 | ≈22/predictor | 659 |

**Interpretation:** the weaker the anticipated true model performance, the more data each predictor "costs"  because with a weak true signal, more data is needed per predictor to distinguish real effects from noise without overfitting.

## Sensitivity to prevalence (p = 30, C-statistic = 0.80)

| Prevalence | Required n |
|---|---|
| 0.15 | 1,646 |
| 0.20 | 1,332 |
| 0.25 | 1,148 |

**Interpretation:** lower prevalence increases the required *total* sample (not just the number of cases needed)  get a realistic local prevalence estimate (e.g., from prior audit/chart-review data at the two hospitals) before finalising recruitment targets, since this single input moves the total requirement more than almost any other assumption.

## Per-site allocation

| Scenario | Total n | Per hospital (50/50 split) |
|---|---|---|
| Base case (p=30, C=0.80) | 1,330 | 665 |
| Consolidated predictors (p≈20) | 900 | 450 |
| Consolidated + 15% dropout buffer | ≈1,035 | ≈518 |

An even split assumes comparable eligible-patient volume at both sites  check expected relative patient volume at each hospital before finalising.

## Reproducing this calculation

```bash
pip install -r ../requirements.txt
python ../scripts/00_sample_size_calculation.py
```

Edit the `scenarios`, `prev`, and `target_c` values at the bottom of the script to explore other assumptions. For final grant/protocol submission, cross-check against the R `pmsampsize` package, which implements the same Riley et al. (2020) method with additional diagnostics.
