r"""
em.py -- EM algorithm for a one-dimensional Gaussian mixture.

Model: p(x) = sum_k w_k N(x; mu_k, s_k^2). Latent component z_i.
E-step: responsibilities  r_ik = w_k N(x_i; mu_k, s_k^2) / sum_j w_j N(x_i; mu_j, s_j^2).
M-step: with N_k = sum_i r_ik,
        w_k = N_k / n,   mu_k = sum_i r_ik x_i / N_k,   s_k^2 = sum_i r_ik (x_i - mu_k)^2 / N_k.
Each iteration cannot decrease the observed-data log-likelihood (the EM lower-bound
argument), but it only finds a LOCAL maximum. The likelihood is unbounded if a component
collapses on one point (s_k -> 0), so a variance floor is applied.
"""

import numpy as np
from scipy.special import logsumexp


def _log_components(x, w, mu, var):
    return np.log(w)[None, :] - 0.5 * np.log(2 * np.pi * var)[None, :] - (x[:, None] - mu[None, :]) ** 2 / (2 * var[None, :])


def log_likelihood(x, w, mu, var):
    return float(np.sum(logsumexp(_log_components(x, w, mu, var), axis=1)))


def gmm_em(x, w0, mu0, var0, n_iter=200, tol=1e-10, var_floor=1e-6):
    """Returns (w, mu, var, log-likelihood history)."""
    x = np.asarray(x, dtype=float)
    w, mu, var = np.array(w0, float), np.array(mu0, float), np.array(var0, float)
    history = [log_likelihood(x, w, mu, var)]
    for _ in range(n_iter):
        logp = _log_components(x, w, mu, var)
        r = np.exp(logp - logsumexp(logp, axis=1, keepdims=True))
        nk = r.sum(axis=0)
        w = nk / len(x)
        mu = (r * x[:, None]).sum(axis=0) / nk
        var = np.maximum((r * (x[:, None] - mu[None, :]) ** 2).sum(axis=0) / nk, var_floor)
        history.append(log_likelihood(x, w, mu, var))
        if abs(history[-1] - history[-2]) < tol * max(1.0, abs(history[-1])):
            break
    order = np.argsort(mu)
    return w[order], mu[order], var[order], history


def kmeans_init(x, k, rng):
    """Spread initial means over data quantiles; unit-share weights; pooled variance."""
    mu = np.quantile(x, (np.arange(k) + 0.5) / k)
    return np.full(k, 1 / k), mu, np.full(k, np.var(x))
