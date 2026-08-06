"""
Sample size calculation for developing a binary-outcome clinical prediction model,
using the three-criterion approach of Riley et al. (2020, BMJ, 368:m441).

This reproduces the calculation reported in docs/sample_size_calculation.md and
PROTOCOL.md (Section 4.4). No patient data is required -- this is a Monte Carlo
simulation driven only by assumed prevalence, target C-statistic, and candidate
predictor count.

Method summary
--------------
1. Convert a target C-statistic to an anticipated Cox-Snell R^2 via Monte Carlo
   calibration (equivalent in spirit to the `pmsampsize` R package's approach):
   simulate a linear predictor LP ~ N(0, sigma^2), calibrate the intercept to hit
   the target prevalence, and search sigma until the simulated model achieves the
   target C-statistic.
2. Compute R^2_max (Cox-Snell) analytically from prevalence.
3. Apply Riley's three sample-size criteria and take the maximum.

Usage
-----
    pip install -r requirements.txt
    python scripts/00_sample_size_calculation.py

Edit the `scenarios`, `prev`, and `target_c` values in the __main__ block to
explore other assumptions.
"""

import numpy as np
from scipy.optimize import brentq
from sklearn.metrics import roc_auc_score

RNG = np.random.default_rng(42)
N_SIM = 2_000_000  # Monte Carlo sample size for stable estimates


def _prevalence(beta0, sigma, lp):
    p = 1 / (1 + np.exp(-(beta0 + lp)))
    return p.mean()


def _calibrate_intercept(sigma, target_prev, lp):
    """Find beta0 such that the simulated population hits target_prev exactly."""
    f = lambda b0: _prevalence(b0, sigma, lp) - target_prev
    return brentq(f, -15, 15)


def _achieved_c_stat(beta0, sigma, lp, rng):
    p = 1 / (1 + np.exp(-(beta0 + lp)))
    y = rng.binomial(1, p)
    if y.sum() == 0 or y.sum() == len(y):
        return np.nan, p, y
    return roc_auc_score(y, p), p, y


def _find_sigma_for_c(target_c, target_prev, rng, n=N_SIM):
    """Search sigma (SD of linear predictor) so the simulated model achieves target_c."""
    lp_base = rng.normal(0, 1, n)

    def f(sigma):
        lp = lp_base * sigma
        b0 = _calibrate_intercept(sigma, target_prev, lp)
        c, _, _ = _achieved_c_stat(b0, sigma, lp, rng)
        return c - target_c

    sigma_sol = brentq(f, 0.01, 8, xtol=1e-4)
    lp = lp_base * sigma_sol
    b0 = _calibrate_intercept(sigma_sol, target_prev, lp)
    c, p, y = _achieved_c_stat(b0, sigma_sol, lp, rng)
    return sigma_sol, b0, c, p, y


def _cox_snell_r2(p, y, prev):
    n = len(y)
    ll_full = np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))
    ll_null = n * (prev * np.log(prev) + (1 - prev) * np.log(1 - prev))
    return 1 - np.exp((2 / n) * (ll_null - ll_full))


def _r2_cs_max(prev):
    """Analytic max Cox-Snell R^2 for a given prevalence (independent of n)."""
    return 1 - (prev**prev * (1 - prev) ** (1 - prev)) ** 2


def riley_sample_size(p_params, prev, target_c, shrinkage=0.90, delta=0.05, rng=RNG):
    """
    Compute Riley et al. (2020) three-criterion sample size for a binary
    outcome prediction model.

    Parameters
    ----------
    p_params : int
        Number of candidate predictor parameters in the model.
    prev : float
        Anticipated outcome prevalence (0-1).
    target_c : float
        Anticipated/target C-statistic (AUC) for the model.
    shrinkage : float
        Target global shrinkage factor S (default 0.90, i.e. <=10% overfitting).
    delta : float
        Target precision for criteria (ii) and (iii) (default 0.05).

    Returns
    -------
    dict with the calibration diagnostics and required sample sizes.
    """
    sigma, b0, c_achieved, p, y = _find_sigma_for_c(target_c, prev, rng)
    r2cs = _cox_snell_r2(p, y, prev)
    r2cs_max = _r2_cs_max(prev)
    r2_nagelkerke = r2cs / r2cs_max

    # Criterion (i): shrinkage
    n1 = p_params / ((shrinkage - 1) * np.log(1 - r2cs / shrinkage))

    # Criterion (ii): small difference between apparent and adjusted R^2
    n2 = (r2cs * p_params) / delta

    # Criterion (iii): precise estimation of overall risk (prevalence)
    z = 1.96
    n3 = (z / delta) ** 2 * prev * (1 - prev)

    n_final = max(n1, n2, n3)

    return dict(
        sigma=sigma,
        beta0=b0,
        c_achieved=c_achieved,
        R2cs=r2cs,
        R2cs_max=r2cs_max,
        R2nagelkerke=r2_nagelkerke,
        n1_shrinkage=n1,
        n2_overfit=n2,
        n3_precision=n3,
        n_final=n_final,
    )


if __name__ == "__main__":
    print("=" * 78)
    print("Base case: prevalence=0.20, target C-statistic=0.80, sweeping predictor count")
    print("=" * 78)
    print(f"{'p (params)':>10} | {'n1 (shrink)':>12} | {'n2 (overfit)':>13} | "
          f"{'n3 (precision)':>15} | {'n required':>11} | EPP")
    for p_params in [20, 25, 30, 40, 50]:
        res = riley_sample_size(p_params, prev=0.20, target_c=0.80)
        n_req = int(np.ceil(res["n_final"]))
        epp = (n_req * 0.20) / p_params
        print(f"{p_params:>10} | {res['n1_shrinkage']:>12.1f} | {res['n2_overfit']:>13.1f} | "
              f"{res['n3_precision']:>15.1f} | {n_req:>11} | {epp:>4.1f}")

    print()
    print("=" * 78)
    print("Sensitivity to target C-statistic (p=30, prevalence=0.20)")
    print("=" * 78)
    print(f"{'C-stat':>7} | {'R2cs':>6} | {'n1':>8} | {'n2':>8} | {'n3':>8} | {'n_final':>8}")
    for c_target in [0.70, 0.75, 0.80, 0.85, 0.90]:
        res = riley_sample_size(30, prev=0.20, target_c=c_target)
        print(f"{c_target:>7.2f} | {res['R2cs']:>6.3f} | {res['n1_shrinkage']:>8.1f} | "
              f"{res['n2_overfit']:>8.1f} | {res['n3_precision']:>8.1f} | {res['n_final']:>8.1f}")

    print()
    print("=" * 78)
    print("Sensitivity to prevalence (p=30, C-stat=0.80)")
    print("=" * 78)
    print(f"{'prev':>6} | {'R2cs':>6} | {'n1':>8} | {'n2':>8} | {'n3':>8} | {'n_final':>8}")
    for prev in [0.15, 0.20, 0.25]:
        res = riley_sample_size(30, prev=prev, target_c=0.80)
        print(f"{prev:>6.2f} | {res['R2cs']:>6.3f} | {res['n1_shrinkage']:>8.1f} | "
              f"{res['n2_overfit']:>8.1f} | {res['n3_precision']:>8.1f} | {res['n_final']:>8.1f}")
