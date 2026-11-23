"""
Runs every experiment; writes results/tables/*.csv, results/figures/*.png and
results/run_summary.json. Everything is seeded; two runs give an identical summary.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from src import em, intervals as iv, mcmc, mle, testing as tt
from src import visualization as viz

ROOT = Path(__file__).parent
FIG, TAB = ROOT / "results" / "figures", ROOT / "results" / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)
SUMMARY = {}


def f(x):
    return float(f"{x:.4g}")


# ------------------------------------------------------------ 1. binomial intervals (exact, no simulation)
def exp_binomial():
    n = 30
    ps = np.linspace(0.01, 0.99, 197)
    cov = {m: np.array([iv.exact_binomial_coverage(m, p, n) for p in ps]) for m in iv.BINOMIAL_INTERVALS}
    pd.DataFrame({"p": ps, **cov}).to_csv(TAB / "binomial_coverage_n30.csv", index=False)
    viz.plot_binomial_coverage(ps, cov, n, FIG / "01_binomial_coverage.png")
    SUMMARY["binomial_n30"] = {
        m: {"mean_coverage": f(c.mean()), "min_coverage": f(c.min()), "share_of_p_below_0.93": f(np.mean(c < 0.93))}
        for m, c in cov.items()}
    SUMMARY["binomial_n30"]["coverage_at_p_0.01"] = {m: f(iv.exact_binomial_coverage(m, 0.01, n)) for m in cov}


# ------------------------------------------------------------ 2. CI for a mean: skew and heavy tails
def exp_mean_ci():
    dists = {
        "normal": (lambda r, s: r.standard_normal(s), 0.0),
        "lognormal (sigma=1.5)": (lambda r, s: r.lognormal(0, 1.5, s), float(np.exp(1.125))),
        "Student t, 3 d.o.f.": (lambda r, s: r.standard_t(3, s), 0.0),
    }
    ns = [10, 30, 100, 300]
    results, rows = {}, []
    for name, (sampler, mean) in dists.items():
        rng = np.random.default_rng(11)
        t_cov = [iv.mean_coverage(sampler, mean, n, 4000, "t", rng) for n in ns]
        b_cov = [iv.mean_coverage(sampler, mean, n, 600, "bootstrap", rng) for n in ns]
        results[name] = {"t": t_cov, "bootstrap": b_cov}
        rows += [{"distribution": name, "n": n, "t_coverage": a, "bootstrap_coverage": b} for n, a, b in zip(ns, t_cov, b_cov)]
    pd.DataFrame(rows).to_csv(TAB / "mean_ci_coverage.csv", index=False)
    viz.plot_mean_coverage(ns, results, FIG / "02_mean_ci_coverage.png")
    SUMMARY["mean_ci"] = {"n_reps_t": 4000, "n_reps_bootstrap": 600, "bootstrap_resamples": 400,
                          "coverage": {k: {"n": ns, "t": [f(v) for v in r["t"]], "bootstrap": [f(v) for v in r["bootstrap"]]}
                                       for k, r in results.items()}}


# ------------------------------------------------------------ 3. type I error: t-test, Welch, permutation
def exp_type1():
    n_sims, n_perm = 2000, 400
    scen = {
        "normal,\nequal var,\nn=15/15": lambda r: (r.standard_normal(15), r.standard_normal(15)),
        "exponential,\nn=10/10": lambda r: (r.exponential(1, 10), r.exponential(1, 10)),
        "unequal var\nsmall group sd 3\nn=10/40": lambda r: (3 * r.standard_normal(10), r.standard_normal(40)),
        "unequal var\nbig group sd 3\nn=40/10": lambda r: (r.standard_normal(10) * 1, 3 * r.standard_normal(40)),
    }
    rates = {"pooled t-test": [], "Welch t-test": [], "permutation (diff in means)": []}
    rng = np.random.default_rng(21)
    for name, gen in scen.items():
        rej = np.zeros(3)
        for _ in range(n_sims):
            x, y = gen(rng)
            rej[0] += stats.ttest_ind(x, y, equal_var=True).pvalue < 0.05
            rej[1] += stats.ttest_ind(x, y, equal_var=False).pvalue < 0.05
            rej[2] += tt.permutation_pvalue(x, y, n_perm, rng) < 0.05
        for k, v in zip(rates, rej / n_sims):
            rates[k].append(v)
    labels = list(scen)
    pd.DataFrame({"scenario": [l.replace("\n", " ") for l in labels], **rates}).to_csv(TAB / "type1_error.csv", index=False)
    viz.plot_type1(labels, rates, FIG / "03_type1_error.png")
    SUMMARY["type1"] = {"n_sims": n_sims, "permutations": n_perm,
                        "rates": {l.replace("\n", " "): {k: f(rates[k][i]) for k in rates} for i, l in enumerate(labels)}}


# ------------------------------------------------------------ 4. multiple testing
def exp_multiple():
    out = {}
    rng = np.random.default_rng(31)
    for label, (m, m1, eff, sims) in {"null_m100": (100, 0, 0.0, 3000),
                                      "m100_10true_effect3": (100, 10, 3.0, 2000),
                                      "m1000_50true_effect3.5": (1000, 50, 3.5, 500)}.items():
        out[label] = tt.simulate_multiple_testing(m, m1, eff, sims, rng)
    rows = [{"scenario": s, "procedure": p, **v} for s, r in out.items() for p, v in r.items()]
    pd.DataFrame(rows).to_csv(TAB / "multiple_testing.csv", index=False)
    viz.plot_multiple_testing(out["m100_10true_effect3"], "100 tests, 10 true effects of size 3", FIG / "04_multiple_testing.png")
    SUMMARY["multiple_testing"] = {s: {p: {k: f(x) for k, x in v.items()} for p, v in r.items()} for s, r in out.items()}


# ------------------------------------------------------------ 5. MLE asymptotics
def exp_mle():
    a, th = 3.0, 1.0
    ns, sims = [15, 30, 100, 300, 1000], 2000
    cr = mle.asymptotic_variance_shape(a, th)
    ratio, bias, cover = [], [], []
    rng = np.random.default_rng(41)
    for n in ns:
        est = np.array([mle.gamma_mle(rng.gamma(a, th, n))[0] for _ in range(sims)])
        ratio.append(n * est.var(ddof=1) / cr)
        bias.append(est.mean() - a)
        se = np.sqrt(cr / n)
        cover.append(np.mean(np.abs(est - a) <= 1.96 * se))   # oracle-SE Wald band: tests the normal approximation only
    pd.DataFrame({"n": ns, "n_var_over_CR": ratio, "bias": bias, "oracle_wald_coverage": cover}).to_csv(TAB / "mle_gamma.csv", index=False)
    viz.plot_mle(ns, ratio, bias, FIG / "05_mle_asymptotics.png")
    SUMMARY["mle_gamma"] = {"true_shape": a, "true_scale": th, "sims": sims, "cramer_rao_n_var": f(cr),
                            "by_n": {str(n): {"n_var_over_CR": f(r), "bias": f(b), "oracle_wald_coverage": f(c)}
                                     for n, r, b, c in zip(ns, ratio, bias, cover)}}


# ------------------------------------------------------------ 6. MCMC
def _beta_logpost_logit(k, n, a0=1.0, b0=1.0):
    A, B = a0 + k, b0 + n - k

    def lp(phi):                      # density of phi = logit(theta) when theta ~ Beta(A, B)
        return -A * np.logaddexp(0, -phi) - B * np.logaddexp(0, phi)
    return lp, A, B


def exp_mcmc():
    k, n = 18, 50
    lp, A, B = _beta_logpost_logit(k, n)
    exact = stats.beta(A, B)
    steps = [0.05, 0.3, 1.0, 3.0, 10.0, 50.0]
    N, burn = 20000, 2000
    rows, keep = [], None
    for s in steps:
        chains, accs = [], []
        for c, x0 in enumerate([-3, -1, 1, 3]):
            ch, acc = mcmc.metropolis(lp, x0, N, s, np.random.default_rng(100 + c))
            chains.append(ch[burn:]); accs.append(acc)
        chains = np.array(chains)
        theta = 1 / (1 + np.exp(-chains))
        ess = np.mean([mcmc.effective_sample_size(c) for c in chains]) / chains.shape[1]
        rows.append({"step": s, "acceptance": np.mean(accs), "ess_over_n": ess, "rhat": mcmc.gelman_rubin(chains),
                     "post_mean": theta.mean(), "post_mean_error": theta.mean() - exact.mean(),
                     "post_sd": theta.std(), "post_sd_error": theta.std() - exact.std()})
        if s == 1.0:
            keep = theta.ravel()
    df = pd.DataFrame(rows)
    df.to_csv(TAB / "mcmc_step_sweep.csv", index=False)
    grid = np.linspace(0.05, 0.8, 300)
    viz.plot_mcmc(steps, df["acceptance"], df["ess_over_n"], grid, exact.pdf(grid), keep, FIG / "06_mcmc.png")
    # bimodal target: R-hat only detects what the chains explore
    bim = lambda x: np.logaddexp(-0.5 * (x + 4) ** 2, -0.5 * (x - 4) ** 2)
    bimodal = {}
    for label, starts, step in [("step 0.5, chains from both modes", [-4, -4, 4, 4], 0.5),
                                ("step 0.5, all chains from the right mode", [4, 4, 4, 4], 0.5),
                                ("step 6, all chains from the right mode", [4, 4, 4, 4], 6.0)]:
        ch = np.array([mcmc.metropolis(bim, x0, 20000, step, np.random.default_rng(200 + i))[0][2000:] for i, x0 in enumerate(starts)])
        bimodal[label] = {"rhat": f(mcmc.gelman_rubin(ch)), "pooled_mean(true 0)": f(ch.mean()), "ess_over_n": f(np.mean([mcmc.effective_sample_size(c) for c in ch]) / ch.shape[1])}
    SUMMARY["mcmc"] = {
        "data": {"successes": k, "trials": n, "prior": "Beta(1,1)"},
        "exact_posterior_mean": f(exact.mean()), "exact_posterior_sd": f(exact.std()),
        "step_sweep": {str(r["step"]): {k_: f(v) for k_, v in r.items() if k_ != "step"} for r in rows},
        "bimodal_target": bimodal}


# ------------------------------------------------------------ 7. EM
def exp_em():
    seps, reps, n = [0.5, 1.0, 2.0, 3.0, 4.0], 200, 1000
    err, fail, mono, iters = [], [], True, []
    rng = np.random.default_rng(51)
    for d in seps:
        errs = []
        for _ in range(reps):
            x = np.concatenate([rng.normal(0, 1, n // 2), rng.normal(d, 1, n // 2)])
            w, mu, var, hist = em.gmm_em(x, *em.kmeans_init(x, 2, rng), n_iter=300)
            mono &= all(b >= a - 1e-9 for a, b in zip(hist, hist[1:]))
            iters.append(len(hist) - 1)
            errs.append(np.max(np.abs(mu - np.array([0, d]))))
        err.append(np.median(errs)); fail.append(np.mean(np.array(errs) > 1.0))
    pd.DataFrame({"separation": seps, "median_max_mean_error": err, "share_failed": fail}).to_csv(TAB / "em_separation.csv", index=False)
    rng = np.random.default_rng(52)
    x = np.concatenate([rng.normal(0, 1, 500), rng.normal(3, 1, 500)])
    w, mu, var, _ = em.gmm_em(x, *em.kmeans_init(x, 2, rng))
    grid = np.linspace(-4, 7, 400)
    comps = [wi * stats.norm.pdf(grid, m, np.sqrt(v)) for wi, m, v in zip(w, mu, var)]
    viz.plot_em(seps, err, fail, x, grid, sum(comps), comps, FIG / "07_em_mixture.png")
    # a poor start on one dataset
    x4 = np.concatenate([rng.normal(0, 1, 500), rng.normal(5, 1, 500)])
    good = em.gmm_em(x4, *em.kmeans_init(x4, 2, rng), n_iter=500)
    bad = em.gmm_em(x4, [0.5, 0.5], [5.0, 5.5], [1.0, 1.0], n_iter=500)
    SUMMARY["em"] = {
        "reps": reps, "n": n, "loglik_monotone_in_all_runs": bool(mono), "mean_iterations": f(np.mean(iters)),
        "by_separation": {str(d): {"median_max_mean_error": f(e), "share_error_gt_1": f(fl)} for d, e, fl in zip(seps, err, fail)},
        "poor_start_demo": {"quantile_init_loglik": f(good[3][-1]), "bad_init_loglik": f(bad[3][-1]),
                            "bad_init_means": [f(m) for m in bad[1]], "quantile_init_means": [f(m) for m in good[1]]}}


if __name__ == "__main__":
    for fn in (exp_binomial, exp_mean_ci, exp_type1, exp_multiple, exp_mle, exp_mcmc, exp_em):
        fn()
        print("done", fn.__name__, flush=True)
    with open(ROOT / "results" / "run_summary.json", "w") as fh:
        json.dump(SUMMARY, fh, indent=2, sort_keys=True)
