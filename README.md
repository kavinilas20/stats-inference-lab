# Statistical Inference Lab

Probability and statistics methods implemented from scratch, then **tested against their own guarantees**: does a "95% interval" cover 95% of the time, does a "5% test" reject 5% of true nulls, does the MLE reach the Cramér–Rao variance, does an MCMC chain reproduce the exact posterior? Where a closed form exists the check is exact; otherwise it is a seeded simulation. Where a method fails, the failure is reported.

| # | Topic | Question answered |
|---|-------|-------------------|
| 1 | Binomial intervals (Wald, Wilson, Clopper–Pearson) | Exact coverage at every true p (no simulation) |
| 2 | CI for a mean (t, percentile bootstrap) | Coverage under normal, lognormal, and heavy-tailed data |
| 3 | Two-sample tests (pooled t, Welch, permutation) | Type I error when assumptions break |
| 4 | Multiple testing (Bonferroni, Holm, Benjamini–Hochberg) | FWER, FDR and power by simulation |
| 5 | Gamma MLE (Newton, Fisher information) | Does n·Var(â) reach the Cramér–Rao value? |
| 6 | Metropolis MCMC, ESS, R-hat | Exact Beta posterior vs chain; effect of step size; a bimodal target |
| 7 | EM for a Gaussian mixture | Monotone likelihood, match with scikit-learn, recovery vs separation |

## Run
```bash
pip install -r requirements.txt
python -m pytest -q      # 31 tests
python main.py           # ~3 min, writes results/ (7 figures, 7 tables, run_summary.json)
```
Everything is seeded; two full runs gave a byte-identical `results/run_summary.json`. No external data: all experiments are simulations. GitHub Actions runs the test suite automatically on pushes and pull requests (Python 3.11 and 3.12).

## Results (from `results/run_summary.json`)

### 1. Binomial intervals, n = 30 (exact coverage over 197 values of p)
| | mean coverage | worst coverage | share of p with coverage < 0.93 |
|---|---|---|---|
| Wald | 0.887 | **0.260** | 54% |
| Wilson | 0.953 | 0.924 | 4% |
| Clopper–Pearson | 0.974 | 0.954 | 0% |

At p = 0.01 the "95%" Wald interval covers 26% of the time (when zero successes occur it collapses to a single point). Clopper–Pearson never drops below 95% but is conservative (average 97.4%). Wilson is the best compromise here.

![](results/figures/01_binomial_coverage.png)

### 2. Interval for a mean (4000 reps for t, 600 reps for bootstrap, 400 resamples each)
| distribution | n | t-interval | bootstrap |
|---|---|---|---|
| normal | 10 / 300 | 0.946 / 0.955 | 0.910 / 0.942 |
| Student t, 3 d.o.f. | 10 / 300 | 0.963 / 0.956 | 0.910 / 0.935 |
| lognormal (σ = 1.5) | 10 | **0.719** | **0.682** |
| | 30 | 0.782 | 0.792 |
| | 100 | 0.850 | 0.850 |
| | 300 | 0.889 | 0.907 |

Findings, including the unflattering ones: the t-interval is fine for normal data and surprisingly fine for t₃; for strongly skewed data **both** methods undercover badly, and the bootstrap is *not* a cure (it is slightly worse at small n and only about as good at large n). Bootstrap coverages carry a simulation standard error of about 0.01 (600 reps), so differences of 0.01 are noise.

![](results/figures/02_mean_ci_coverage.png)

### 3. Type I error at nominal 5% (2000 simulations per scenario; SE ≈ 0.005)
| scenario | pooled t | Welch | permutation |
|---|---|---|---|
| normal, equal variance, 15/15 | 0.039 | 0.039 | 0.039 |
| exponential, 10/10 | 0.051 | 0.048 | 0.057 |
| unequal variance, **small group** has sd 3 (10/40) | **0.244** | 0.051 | **0.243** |
| unequal variance, big group has sd 3 (40/10) | 0.003 | 0.049 | 0.003 |

A permutation test is exact only when the groups are exchangeable under H₀; with unequal variances and unequal sizes it behaves like the pooled t-test (rejecting 24% of true nulls in one case and almost never in the other). Welch stays near 5%. The three methods run on the same simulated data, so their errors are correlated; the 0.039 in the first row is about 2 SEs below 0.05 and I treat it as simulation noise, not as a property of the tests.

![](results/figures/03_type1_error.png)

### 4. Multiple testing
100 tests, 10 true effects of size 3 sd (2000 sims):
| | FWER | FDR | power |
|---|---|---|---|
| Uncorrected | 0.992 | 0.329 | 0.851 |
| Bonferroni | 0.041 | 0.011 | 0.312 |
| Holm | 0.043 | 0.012 | 0.315 |
| Benjamini–Hochberg | 0.229 | 0.043 | 0.497 |

All 100 true nulls: uncorrected FWER 0.994; Bonferroni/Holm/BH ≈ 0.048–0.049. In the 1000-test scenario BH finds 64% of effects against 29% for Bonferroni at FDR 0.048. BH controls the *share* of false discoveries, not the chance of any (FWER 0.23–0.78 above): choosing a procedure means choosing which error you care about. This is the same issue as selecting the best of many backtests (see the backtest-overfitting lab).

![](results/figures/04_multiple_testing.png)

### 5. Gamma MLE (shape 3, scale 1; 2000 sims per n)
| n | 15 | 30 | 100 | 300 | 1000 |
|---|---|---|---|---|---|
| n·Var(â) / Cramér–Rao | 2.21 | 1.42 | 1.12 | 1.07 | 1.01 |
| bias of â | +0.69 | +0.27 | +0.083 | +0.033 | +0.009 |
| coverage of â ± 1.96·√(CR/n) | 0.848 | 0.906 | 0.925 | 0.940 | 0.950 |

The asymptotic theory is right in the limit and badly optimistic at small n: at n = 15 the variance is 2.2x the bound and the MLE of the shape is biased upward by 23%. (The coverage column uses the *true* Fisher information, so it tests only the normal approximation.)

![](results/figures/05_mle_asymptotics.png)

### 6. MCMC
Posterior for 18 successes in 50 trials with a uniform prior: exact Beta(19, 33), mean 0.3654, sd 0.0661. Four chains of 20 000 draws (2000 burn-in), sampling the logit of θ:

| proposal step | 0.05 | 0.3 | 1 | 3 | 10 | 50 |
|---|---|---|---|---|---|---|
| acceptance | 0.94 | 0.70 | 0.34 | 0.12 | 0.04 | 0.008 |
| ESS / n | 0.007 | 0.12 | **0.205** | 0.085 | 0.028 | 0.005 |

Every setting reproduces the exact posterior mean and sd to within about 0.005 (the large-step runs are just very inefficient), and efficiency peaks at acceptance ~ 0.34, in line with the usual 0.2–0.45 rule of thumb.

![](results/figures/06_mcmc.png)

**Bimodal target** 0.5·N(−4,1) + 0.5·N(4,1) (true mean 0):
| setup | R-hat | pooled mean | ESS/n |
|---|---|---|---|
| step 0.5, chains started in both modes | 1.72 | −1.58 | 0.023 |
| step 0.5, all chains started in the right mode | 1.42 | 1.00 | 0.011 |
| step 6, all chains in the right mode | 1.00 | −0.10 | 0.086 |

R-hat flagged the small-step runs in both cases. I expected the second row to show R-hat ≈ 1 with a wrong mean (the known blind spot of R-hat when *all* chains miss a mode), but one chain happened to cross the valley, which raised R-hat. So this run does **not** demonstrate that blind spot; it is a theoretical caveat only. (A different seed or step could make it appear; I did not search for one.)

### 7. EM for a two-component Gaussian mixture (n = 1000, 200 datasets per separation)
- Observed-data log-likelihood was non-decreasing in every run, and the fit matches `sklearn.mixture.GaussianMixture` from the same start (final log-likelihood to 1e-3, means to 1e-3).
- Median of the larger mean error: 0.21 (separation 0.5), 0.17 (1), 0.18 (2), 0.087 (3), 0.059 (4). No run had a mean error above 1, **including at separation 0.5**, so that "failure" metric did not detect weak identifiability and tells you little; the median error is the informative column. Mean iterations to converge: about 201 (cap 300, tolerance 1e-10; I did not break this down by separation).
- Poor-start check on one dataset (separation 5): starting both means on the same side (5.0, 5.5) still reached the same optimum (log-likelihood −2104.0, means 0.07 and 5.03) as the quantile initialisation. EM can stall in local maxima in general; this experiment did not show it.

![](results/figures/07_em_mixture.png)

## Limitations
- All data are simulated from known distributions, so every claim is about method behaviour under stated assumptions, not about any real dataset.
- Bootstrap results use 600 replications (SE ≈ 0.01); the 1000-test multiple-testing scenario uses 500 sims. Only the exact binomial coverage and the MLE/EM tables are free of that kind of noise to the displayed precision.
- Only the percentile bootstrap is implemented (no BCa or studentised bootstrap, which do better on skewed data). The conclusion "the bootstrap does not cure skewness" applies to the percentile version.
- MCMC is 1-D random-walk Metropolis; no Hamiltonian or Gibbs sampling, and R-hat is the classic (non-split) version.
- EM uses one deterministic quantile start plus one extra start in one check; no multi-start strategy or model selection (BIC).
- Tests: 31, including checks against SciPy/scikit-learn and exact enumeration. 9 of 10 deliberately injected bugs were caught; the survivor (starting Newton at a = 1) is harmless because Newton still converges to the same answer, and I added a stress test on shapes from 0.05 to 300.

## Layout
```
src/intervals.py testing.py mle.py mcmc.py em.py visualization.py
tests/   main.py   results/{figures,tables,run_summary.json}
PROJECT_NOTES.md   LEARNING_ROADMAP.md
```

## References
Brown, Cai & DasGupta (2001), *Interval estimation for a binomial proportion*; Efron & Tibshirani, *An Introduction to the Bootstrap*; Benjamini & Hochberg (1995); Holm (1979); Casella & Berger, *Statistical Inference*; Roberts, Gelman & Gilks (1997) on optimal Metropolis scaling; Gelman & Rubin (1992); Dempster, Laird & Rubin (1977).
