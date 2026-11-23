r"""
mle.py -- maximum likelihood for the Gamma(shape a, scale theta) family, from scratch,
and a check of the asymptotic theory.

Log-likelihood of x_1..x_n:
    l(a, theta) = -n log Gamma(a) - n a log(theta) + (a-1) sum log x_i - sum x_i / theta.
Setting d/d theta = 0 gives theta_hat = mean(x) / a. Substituting back, the shape
solves the one-dimensional equation
    log(a) - digamma(a) = log(mean(x)) - mean(log x)  =: s   (s > 0 by Jensen),
and the left side is strictly decreasing in a, so Newton's method with a good start
converges.

Fisher information per observation, in (a, theta):
    I = [[ trigamma(a),  1/theta ],
         [ 1/theta,      a/theta^2 ]]
Asymptotically sqrt(n)(theta_hat - theta) -> N(0, I^{-1}); the Cramer-Rao bound says no
unbiased estimator has smaller variance. `main.py` compares n * Var(a_hat) with
[I^{-1}]_{aa} by simulation.
"""

import numpy as np
from scipy.special import digamma, polygamma


def gamma_mle(x, tol=1e-12, max_iter=100):
    """Returns (shape, scale). Requires positive data with non-zero variance."""
    x = np.asarray(x, dtype=float)
    s = np.log(x.mean()) - np.mean(np.log(x))
    # Minka's closed-form start, then Newton on g(a) = log a - digamma(a) - s
    a = (3 - s + np.sqrt((s - 3) ** 2 + 24 * s)) / (12 * s)
    for _ in range(max_iter):
        g = np.log(a) - digamma(a) - s
        gp = 1 / a - polygamma(1, a)
        a_new = a - g / gp
        if a_new <= 0:
            a_new = a / 2
        if abs(a_new - a) < tol * a:
            a = a_new
            break
        a = a_new
    return float(a), float(x.mean() / a)


def gamma_loglik(x, a, theta):
    from scipy.special import gammaln
    x = np.asarray(x, dtype=float)
    return float(-len(x) * gammaln(a) - len(x) * a * np.log(theta) + (a - 1) * np.sum(np.log(x)) - np.sum(x) / theta)


def gamma_fisher_information(a, theta):
    """Per-observation Fisher information matrix in (shape, scale)."""
    return np.array([[polygamma(1, a), 1 / theta], [1 / theta, a / theta ** 2]])


def asymptotic_variance_shape(a, theta):
    """[I^{-1}]_{aa}: the Cramer-Rao / asymptotic value of n * Var(a_hat)."""
    return float(np.linalg.inv(gamma_fisher_information(a, theta))[0, 0])
