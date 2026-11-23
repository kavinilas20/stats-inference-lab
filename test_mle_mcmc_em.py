import numpy as np
import pytest
from scipy import stats
from scipy.special import digamma
from sklearn.mixture import GaussianMixture

from src import em, mcmc, mle


def test_gamma_mle_matches_scipy_and_score_equation():
    x = np.random.default_rng(0).gamma(2.5, 1.7, size=400)
    a, th = mle.gamma_mle(x)
    a_ref, _, th_ref = stats.gamma.fit(x, floc=0)
    assert a == pytest.approx(a_ref, rel=1e-4) and th == pytest.approx(th_ref, rel=1e-4)
    assert np.log(a) - digamma(a) == pytest.approx(np.log(x.mean()) - np.mean(np.log(x)), abs=1e-10)
    assert th * a == pytest.approx(x.mean())


def test_gamma_mle_is_a_local_maximum_of_the_likelihood():
    x = np.random.default_rng(1).gamma(3.0, 0.5, size=200)
    a, th = mle.gamma_mle(x)
    best = mle.gamma_loglik(x, a, th)
    for da, dt in [(0.05, 0), (-0.05, 0), (0, 0.02), (0, -0.02), (0.05, 0.02)]:
        assert mle.gamma_loglik(x, a + da, th + dt) < best


def test_gamma_fisher_information_by_monte_carlo():
    a, th = 2.0, 1.5
    x = np.random.default_rng(2).gamma(a, th, size=400000)
    score_a = np.log(x) - np.log(th) - digamma(a)         # d/da log f
    score_t = (x - a * th) / th ** 2                      # d/dtheta log f
    emp = np.cov(np.vstack([score_a, score_t]))
    np.testing.assert_allclose(emp, mle.gamma_fisher_information(a, th), rtol=0.03)


def test_asymptotic_variance_of_shape_is_approached():
    rng = np.random.default_rng(3)
    a, th, n = 3.0, 1.0, 400
    est = np.array([mle.gamma_mle(rng.gamma(a, th, n))[0] for _ in range(1500)])
    assert n * est.var() == pytest.approx(mle.asymptotic_variance_shape(a, th), rel=0.12)


def _normal_logpdf(x):
    return -0.5 * x * x


def test_metropolis_recovers_standard_normal():
    chain, acc = mcmc.metropolis(_normal_logpdf, 0.0, 60000, 2.4, np.random.default_rng(0))
    assert 0.2 < acc < 0.8
    assert chain[2000:].mean() == pytest.approx(0, abs=0.05)
    assert chain[2000:].std() == pytest.approx(1, abs=0.05)


def test_ess_of_iid_noise_is_close_to_n_and_ar1_is_smaller():
    rng = np.random.default_rng(1)
    iid = rng.standard_normal(20000)
    assert mcmc.effective_sample_size(iid) > 0.8 * 20000
    ar = np.zeros(20000)
    e = rng.standard_normal(20000)
    for t in range(1, 20000):
        ar[t] = 0.9 * ar[t - 1] + e[t]
    theory = 20000 * (1 - 0.9) / (1 + 0.9)           # n (1-phi)/(1+phi)
    assert mcmc.effective_sample_size(ar) == pytest.approx(theory, rel=0.35)


def test_rhat_near_one_for_mixed_chains_and_large_for_stuck_chains():
    rng = np.random.default_rng(2)
    good = rng.standard_normal((4, 2000))
    assert mcmc.gelman_rubin(good) == pytest.approx(1.0, abs=0.02)
    stuck = rng.standard_normal((4, 2000)) + np.array([[0], [5], [10], [15]])
    assert mcmc.gelman_rubin(stuck) > 2


def test_em_loglik_is_monotone_and_matches_sklearn_from_same_start():
    rng = np.random.default_rng(0)
    x = np.concatenate([rng.normal(-2, 1, 300), rng.normal(2.5, 0.7, 200)])
    w0, mu0, v0 = [0.5, 0.5], [-1.0, 1.0], [1.0, 1.0]
    w, mu, var, hist = em.gmm_em(x, w0, mu0, v0, n_iter=500, tol=1e-12)
    assert all(b >= a - 1e-9 for a, b in zip(hist, hist[1:]))
    gm = GaussianMixture(2, weights_init=w0, means_init=np.array(mu0)[:, None],
                         precisions_init=1 / np.array(v0)[:, None, None], tol=1e-12,
                         max_iter=500, reg_covar=1e-6).fit(x[:, None])
    assert hist[-1] == pytest.approx(gm.score(x[:, None]) * len(x), abs=1e-3)
    np.testing.assert_allclose(mu, np.sort(gm.means_.ravel()), atol=1e-3)


def test_em_recovers_well_separated_mixture():
    rng = np.random.default_rng(1)
    x = np.concatenate([rng.normal(0, 1, 1500), rng.normal(6, 1, 1000)])
    w, mu, var, _ = em.gmm_em(x, *em.kmeans_init(x, 2, rng))
    np.testing.assert_allclose(mu, [0, 6], atol=0.15)
    np.testing.assert_allclose(w, [0.6, 0.4], atol=0.03)


def test_em_variance_floor_prevents_collapse():
    x = np.concatenate([np.zeros(5), np.random.default_rng(3).standard_normal(100)])
    _, _, var, hist = em.gmm_em(x, [0.5, 0.5], [0.0, 1.0], [0.1, 1.0], n_iter=100)
    assert np.all(var >= 1e-6) and np.isfinite(hist[-1])


def test_metropolis_on_offcentre_narrow_target_matches_ks_and_moments():
    logp = lambda x: -0.5 * ((x - 3.0) / 0.5) ** 2
    chain, _ = mcmc.metropolis(logp, 0.0, 80000, 0.8, np.random.default_rng(5))
    c = chain[5000:]
    assert c.mean() == pytest.approx(3.0, abs=0.03) and c.std() == pytest.approx(0.5, abs=0.03)
    thinned = c[::25]
    assert stats.kstest(thinned, stats.norm(3.0, 0.5).cdf).pvalue > 0.001


def test_metropolis_matches_exact_beta_posterior():
    A, B = 19.0, 33.0
    lp = lambda phi: -A * np.logaddexp(0, -phi) - B * np.logaddexp(0, phi)
    chain, _ = mcmc.metropolis(lp, 0.0, 60000, 1.0, np.random.default_rng(6))
    theta = 1 / (1 + np.exp(-chain[3000:]))
    assert theta.mean() == pytest.approx(A / (A + B), abs=0.004)
    assert theta.std() == pytest.approx(stats.beta(A, B).std(), abs=0.004)


@pytest.mark.parametrize("shape", [0.05, 0.3, 1.0, 30.0, 300.0])
def test_gamma_mle_converges_for_extreme_shapes(shape):
    x = np.random.default_rng(7).gamma(shape, 2.0, size=5000)
    a, th = mle.gamma_mle(x)
    assert a == pytest.approx(shape, rel=0.15)
    assert np.log(a) - digamma(a) == pytest.approx(np.log(x.mean()) - np.mean(np.log(x)), abs=1e-9)
