r"""
intervals.py -- confidence intervals and their *actual* coverage.

A 95% interval promises that, over repeated samples, it contains the true
parameter 95% of the time. This module builds standard intervals and measures
whether that promise holds.

Binomial proportion (k successes in n trials), three intervals:
    Wald:            p_hat +/- z sqrt(p_hat (1-p_hat) / n)
    Wilson (score):  centre (p_hat + z^2/2n) / (1 + z^2/n), half-width
                     z/(1+z^2/n) sqrt(p_hat(1-p_hat)/n + z^2/4n^2)
    Clopper-Pearson: exact, inverts the binomial test via Beta quantiles.

Because k takes only n+1 values, the exact coverage at a true p is the finite sum
    coverage(p) = sum_k  Binom(k; n, p) * 1[ p in CI(k) ]
so no simulation is needed for the binomial case.

Mean of a sample: t-interval and percentile bootstrap, evaluated by simulation.
"""

import numpy as np
from scipy import stats


def wald_interval(k, n, alpha=0.05):
    z = stats.norm.ppf(1 - alpha / 2)
    p = k / n
    h = z * np.sqrt(p * (1 - p) / n)
    return np.clip(p - h, 0, 1), np.clip(p + h, 0, 1)


def wilson_interval(k, n, alpha=0.05):
    z = stats.norm.ppf(1 - alpha / 2)
    p = k / n
    denom = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / denom
    half = z / denom * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2))
    return centre - half, centre + half


def clopper_pearson_interval(k, n, alpha=0.05):
    k = np.asarray(k)
    lo = np.where(k == 0, 0.0, stats.beta.ppf(alpha / 2, np.maximum(k, 1), n - k + 1))
    hi = np.where(k == n, 1.0, stats.beta.ppf(1 - alpha / 2, k + 1, np.maximum(n - k, 1)))
    return lo, hi


BINOMIAL_INTERVALS = {"Wald": wald_interval, "Wilson": wilson_interval, "Clopper-Pearson": clopper_pearson_interval}


def exact_binomial_coverage(method, p, n, alpha=0.05):
    """Exact coverage probability at true proportion p (sum over all n+1 outcomes)."""
    k = np.arange(n + 1)
    lo, hi = BINOMIAL_INTERVALS[method](k, n, alpha)
    covered = (lo <= p + 1e-12) & (p <= hi + 1e-12)
    return float(np.sum(stats.binom.pmf(k, n, p) * covered))


def t_interval(x, alpha=0.05):
    """x: (..., n) array; interval for the mean along the last axis."""
    x = np.asarray(x, dtype=float)
    n = x.shape[-1]
    m = x.mean(axis=-1)
    se = x.std(axis=-1, ddof=1) / np.sqrt(n)
    t = stats.t.ppf(1 - alpha / 2, n - 1)
    return m - t * se, m + t * se


def bootstrap_percentile_interval(x, n_boot, alpha, rng):
    """Percentile bootstrap for the mean of a single 1-D sample."""
    x = np.asarray(x, dtype=float)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    means = x[idx].mean(axis=1)
    return tuple(np.quantile(means, [alpha / 2, 1 - alpha / 2]))


def mean_coverage(sampler, true_mean, n, n_reps, method, rng, alpha=0.05, n_boot=400):
    """Fraction of n_reps samples whose interval for the mean contains true_mean."""
    hits = 0
    if method == "t":
        lo, hi = t_interval(sampler(rng, (n_reps, n)), alpha)
        return float(np.mean((lo <= true_mean) & (true_mean <= hi)))
    for _ in range(n_reps):
        lo, hi = bootstrap_percentile_interval(sampler(rng, n), n_boot, alpha, rng)
        hits += lo <= true_mean <= hi
    return hits / n_reps
