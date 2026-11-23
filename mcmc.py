r"""
mcmc.py -- random-walk Metropolis sampler and convergence diagnostics.

Propose x' = x + step * N(0,1); accept with probability min(1, pi(x')/pi(x)). The
chain has pi as its stationary distribution (detailed balance), but samples are
correlated, so n draws carry fewer than n independent samples:
    ESS = n / (1 + 2 sum_{t>=1} rho_t),
with rho_t the autocorrelation, summed here until it first turns negative.
Gelman-Rubin R-hat compares between-chain and within-chain variance across
independently started chains; values near 1 are necessary, NOT sufficient, for
convergence (chains that all miss the same mode agree with each other).
"""

import numpy as np


def metropolis(log_target, x0, n_samples, step, rng):
    """1-D random-walk Metropolis. Returns (chain, acceptance_rate)."""
    x, lp = float(x0), log_target(float(x0))
    chain = np.empty(n_samples)
    accepted = 0
    noise = rng.standard_normal(n_samples)
    logu = np.log(rng.random(n_samples))
    for i in range(n_samples):
        prop = x + step * noise[i]
        lp_prop = log_target(prop)
        if logu[i] < lp_prop - lp:
            x, lp = prop, lp_prop
            accepted += 1
        chain[i] = x
    return chain, accepted / n_samples


def autocorrelation(x, max_lag):
    x = np.asarray(x, float) - np.mean(x)
    n = len(x)
    f = np.fft.rfft(x, 2 * n)
    acov = np.fft.irfft(f * np.conj(f))[:n] / n
    return acov[:max_lag + 1] / acov[0]


def effective_sample_size(x, max_lag=None):
    n = len(x)
    max_lag = min(n - 1, 1000) if max_lag is None else max_lag
    rho = autocorrelation(x, max_lag)
    total = 0.0
    for t in range(1, len(rho)):
        if rho[t] < 0:
            break
        total += rho[t]
    return float(n / (1 + 2 * total))


def gelman_rubin(chains):
    """chains: (m, n) array of m independent chains of length n."""
    chains = np.asarray(chains, float)
    m, n = chains.shape
    within = chains.var(axis=1, ddof=1).mean()
    between = n * chains.mean(axis=1).var(ddof=1)
    var_plus = (n - 1) / n * within + between / n
    return float(np.sqrt(var_plus / within))
