import numpy as np
import pytest
from scipy import stats

from src import intervals as iv
from src import testing as tt


def test_binomial_intervals_match_scipy():
    for n in (10, 40):
        for k in (0, 1, 3, n // 2, n - 1, n):
            ref = stats.binomtest(k, n).proportion_ci(0.95, method="wilson")
            lo, hi = iv.wilson_interval(np.array([k]), n)
            assert lo[0] == pytest.approx(ref.low, abs=1e-10) and hi[0] == pytest.approx(ref.high, abs=1e-10)
            ref = stats.binomtest(k, n).proportion_ci(0.95, method="exact")
            lo, hi = iv.clopper_pearson_interval(np.array([k]), n)
            assert lo[0] == pytest.approx(ref.low, abs=1e-9) and hi[0] == pytest.approx(ref.high, abs=1e-9)


def test_clopper_pearson_coverage_never_below_nominal():
    for n in (10, 30, 100):
        for p in np.linspace(0.005, 0.995, 199):
            assert iv.exact_binomial_coverage("Clopper-Pearson", p, n) >= 0.95 - 1e-9


def test_wald_fails_near_boundary_wilson_does_not():
    assert iv.exact_binomial_coverage("Wald", 0.01, 30) < 0.30
    assert iv.exact_binomial_coverage("Wilson", 0.01, 30) > 0.85


def test_wald_interval_is_degenerate_at_zero_successes():
    lo, hi = iv.wald_interval(np.array([0]), 20)
    assert lo[0] == 0 and hi[0] == 0


def test_exact_coverage_is_a_probability_and_matches_simulation():
    rng = np.random.default_rng(0)
    p, n = 0.2, 25
    k = rng.binomial(n, p, size=200000)
    lo, hi = iv.wilson_interval(k, n)
    sim = np.mean((lo <= p) & (p <= hi))
    assert iv.exact_binomial_coverage("Wilson", p, n) == pytest.approx(sim, abs=0.004)


def test_t_interval_matches_scipy():
    x = np.random.default_rng(1).standard_normal(15)
    lo, hi = iv.t_interval(x)
    ref = stats.t.interval(0.95, 14, loc=x.mean(), scale=stats.sem(x))
    assert (lo, hi) == pytest.approx(ref)


def test_t_interval_covers_about_95_percent_for_normal_data():
    rng = np.random.default_rng(2)
    cov = iv.mean_coverage(lambda r, s: r.standard_normal(s), 0.0, 20, 20000, "t", rng)
    assert cov == pytest.approx(0.95, abs=0.01)


def test_permutation_pvalue_matches_exact_enumeration():
    rng = np.random.default_rng(3)
    x, y = rng.standard_normal(5) + 1, rng.standard_normal(5)
    exact = tt.exact_permutation_pvalue(x, y)
    assert tt.permutation_pvalue(x, y, 20000, rng) == pytest.approx(exact, abs=0.02)


def test_permutation_pvalue_never_zero():
    rng = np.random.default_rng(4)
    p = tt.permutation_pvalue(np.arange(20) + 100.0, np.arange(20), 99, rng)
    assert p >= 1 / 100


def _bh_reference(p, alpha):
    m = len(p)
    ps = np.sort(p)
    ok = [i for i in range(m) if ps[i] <= alpha * (i + 1) / m]
    if not ok:
        return np.zeros(m, bool)
    return p <= ps[max(ok)]


def test_bh_matches_definition_and_known_example():
    rng = np.random.default_rng(5)
    for _ in range(50):
        p = rng.random(30) ** 2
        assert np.array_equal(tt.benjamini_hochberg(p, 0.1), _bh_reference(p, 0.1))
    p = np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.216])
    assert tt.benjamini_hochberg(p, 0.05).sum() == 2


def test_rejection_sets_are_nested():
    rng = np.random.default_rng(6)
    for _ in range(100):
        p = rng.random(40) ** 3
        b, h, bh = tt.bonferroni(p), tt.holm(p), tt.benjamini_hochberg(p)
        assert np.all(h >= b) and np.all(bh >= h)


def test_holm_stops_at_first_failure():
    p = np.array([0.001, 0.04, 0.0105, 0.5])      # Holm thresholds: .0125 .0167 .025 .05
    assert list(tt.holm(p, 0.05)) == [True, False, True, False]


def test_fwer_and_fdr_control_under_simulation():
    rng = np.random.default_rng(7)
    null = tt.simulate_multiple_testing(50, 0, 0.0, 3000, rng)
    assert null["Uncorrected"]["fwer"] > 0.8
    assert null["Bonferroni"]["fwer"] < 0.065 and null["Holm"]["fwer"] < 0.065
    mixed = tt.simulate_multiple_testing(100, 20, 3.0, 1500, rng)
    assert mixed["Benjamini-Hochberg"]["fdr"] < 0.05 * 80 / 100 + 0.01
    assert mixed["Benjamini-Hochberg"]["power"] > mixed["Bonferroni"]["power"]


def test_holm_strictly_more_powerful_than_bonferroni_in_a_known_case():
    p = np.array([0.001, 0.02, 0.9])          # Bonferroni threshold .0167; Holm's second step is .025
    assert list(tt.bonferroni(p, 0.05)) == [True, False, False]
    assert list(tt.holm(p, 0.05)) == [True, True, False]
