r"""
testing.py -- permutation tests and multiple-testing corrections, with simulation
of what each correction actually controls.

Permutation test for a difference in means: under H0 (both samples from one
distribution) every relabelling is equally likely, so the p-value is the share
of relabellings with a statistic at least as extreme. With B random relabellings
the valid estimate is (1 + #{extreme}) / (1 + B) (the observed labelling counts).

With m hypotheses and p-values p_(1) <= ... <= p_(m):
    Bonferroni:  reject p_i <= alpha/m                 controls FWER
    Holm:        reject p_(i) while p_(i) <= alpha/(m-i+1)   controls FWER, uniformly more powerful
    BH:          reject the largest i with p_(i) <= alpha i/m, and everything below it
                 controls FDR = E[false discoveries / discoveries] (independent or PRDS p-values)
FWER = P(at least one false rejection); FDR is weaker but far more powerful.
"""

import numpy as np
from scipy import stats


def permutation_pvalue(x, y, n_perm, rng):
    """Two-sided permutation p-value for the difference in means (random relabellings)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    pooled = np.concatenate([x, y])
    nx = len(x)
    obs = abs(x.mean() - y.mean())
    idx = np.argsort(rng.random((n_perm, len(pooled))), axis=1)
    perm = pooled[idx]
    stat = np.abs(perm[:, :nx].mean(axis=1) - perm[:, nx:].mean(axis=1))
    return float((1 + np.sum(stat >= obs - 1e-12)) / (1 + n_perm))


def exact_permutation_pvalue(x, y):
    """Enumerate every split (small samples only) -- used to check the Monte Carlo version."""
    from itertools import combinations
    pooled = np.concatenate([x, y])
    n, nx = len(pooled), len(x)
    obs = abs(np.mean(x) - np.mean(y))
    total = pooled.sum()
    count = tot = 0
    for c in combinations(range(n), nx):
        s = pooled[list(c)].sum()
        stat = abs(s / nx - (total - s) / (n - nx))
        count += stat >= obs - 1e-12
        tot += 1
    return count / tot


def bonferroni(p, alpha=0.05):
    p = np.asarray(p)
    return p <= alpha / len(p)


def holm(p, alpha=0.05):
    p = np.asarray(p)
    m = len(p)
    order = np.argsort(p)
    reject = np.zeros(m, dtype=bool)
    for rank, i in enumerate(order):
        if p[i] <= alpha / (m - rank):
            reject[i] = True
        else:
            break
    return reject


def benjamini_hochberg(p, alpha=0.05):
    p = np.asarray(p)
    m = len(p)
    order = np.argsort(p)
    passed = p[order] <= alpha * (np.arange(1, m + 1) / m)
    reject = np.zeros(m, dtype=bool)
    if passed.any():
        cutoff = np.max(np.nonzero(passed)[0])
        reject[order[:cutoff + 1]] = True
    return reject


PROCEDURES = {
    "Uncorrected": lambda p, a: np.asarray(p) <= a,
    "Bonferroni": bonferroni,
    "Holm": holm,
    "Benjamini-Hochberg": benjamini_hochberg,
}


def simulate_multiple_testing(m, m1, effect, n_sims, rng, alpha=0.05):
    """
    m z-tests, the first m1 have true mean `effect` (in sd units), the rest are null.
    Returns {procedure: {"FWER", "FDR", "power"}} averaged over n_sims.
    """
    out = {k: {"fwer": 0.0, "fdr": 0.0, "power": 0.0} for k in PROCEDURES}
    for _ in range(n_sims):
        z = rng.standard_normal(m)
        z[:m1] += effect
        p = 2 * stats.norm.sf(np.abs(z))
        for name, proc in PROCEDURES.items():
            rej = proc(p, alpha)
            false = rej[m1:].sum()
            out[name]["fwer"] += false > 0
            out[name]["fdr"] += false / max(rej.sum(), 1)
            out[name]["power"] += rej[:m1].mean() if m1 else 0.0
    return {k: {kk: vv / n_sims for kk, vv in v.items()} for k, v in out.items()}
