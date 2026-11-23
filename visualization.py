"""Figures for the statistics lab (same palette and rules as the other labs)."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8985"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
CYCLE = [BLUE, ORANGE, AQUA, GREY]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 1.4, "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "legend.frameon": False,
})


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_binomial_coverage(ps, cov, n, path):
    fig, ax = plt.subplots(figsize=(7, 3.8))
    for (name, c), col in zip(cov.items(), [ORANGE, BLUE, AQUA]):
        ax.plot(ps, c, color=col, label=name, lw=1.1)
    ax.axhline(0.95, color=INK2, ls=":", lw=1)
    ax.set(title=f"Exact coverage of nominal 95% intervals, n={n}", xlabel="true proportion $p$",
           ylabel="coverage probability", ylim=(0.0, 1.02))
    ax.legend(loc="lower center", ncol=3)
    _save(fig, path)


def plot_mean_coverage(ns, results, path):
    """results: {dist: {"t": [...], "bootstrap": [...]}}"""
    fig, axes = plt.subplots(1, len(results), figsize=(3.6 * len(results), 3.6), sharey=True)
    for ax, (dist, r) in zip(np.atleast_1d(axes), results.items()):
        ax.semilogx(ns, r["t"], "o-", ms=4, color=BLUE, label="t-interval")
        ax.semilogx(ns, r["bootstrap"], "o-", ms=4, color=ORANGE, label="percentile bootstrap")
        ax.axhline(0.95, color=INK2, ls=":", lw=1)
        ax.set(title=dist, xlabel="sample size $n$")
        ax.set_xticks(ns)
        ax.set_xticklabels([str(n) for n in ns])
    np.atleast_1d(axes)[0].set_ylabel("coverage of the mean")
    np.atleast_1d(axes)[0].legend(loc="lower right")
    _save(fig, path)


def plot_type1(scenarios, rates, path):
    names = list(rates)
    x = np.arange(len(scenarios))
    w = 0.8 / len(names)
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    for i, (nm, col) in enumerate(zip(names, CYCLE)):
        ax.bar(x + (i - (len(names) - 1) / 2) * w, rates[nm], w * 0.92, color=col, label=nm)
    ax.axhline(0.05, color=INK2, ls=":", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, fontsize=8)
    ax.set(title="Type I error under the null (nominal 5%)", ylabel="rejection rate")
    ax.legend()
    _save(fig, path)


def plot_multiple_testing(res, title, path):
    procs = list(res)
    x = np.arange(len(procs))
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
    ax[0].bar(x - 0.2, [res[p]["fwer"] for p in procs], 0.38, color=ORANGE, label="FWER")
    ax[0].bar(x + 0.2, [res[p]["fdr"] for p in procs], 0.38, color=BLUE, label="FDR")
    ax[0].axhline(0.05, color=INK2, ls=":", lw=1)
    ax[0].set(title="Error rates", ylabel="rate")
    ax[1].bar(x, [res[p]["power"] for p in procs], 0.55, color=AQUA)
    ax[1].set(title=f"Power ({title})", ylabel="share of true effects found")
    for a in ax:
        a.set_xticks(x)
        a.set_xticklabels([p.replace("Benjamini-Hochberg", "BH") for p in procs], fontsize=8)
    ax[0].legend()
    _save(fig, path)


def plot_mle(ns, ratio, bias, path):
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
    ax[0].semilogx(ns, ratio, "o-", ms=4, color=BLUE)
    ax[0].axhline(1.0, color=INK2, ls=":", lw=1)
    ax[0].set(title="$n\\,\\mathrm{Var}(\\hat a)$ / Cramér–Rao value", xlabel="sample size $n$", ylabel="ratio")
    ax[1].semilogx(ns, bias, "o-", ms=4, color=ORANGE)
    ax[1].axhline(0, color=INK2, ls=":", lw=1)
    ax[1].set(title="Bias of the shape MLE", xlabel="sample size $n$", ylabel="mean$(\\hat a)-a$")
    for a in ax:
        a.set_xticks(ns)
        a.set_xticklabels([str(n) for n in ns])
    _save(fig, path)


def plot_mcmc(steps, acc, ess, grid, post_pdf, samples, path):
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))
    ax[0].semilogx(steps, acc, "o-", ms=4, color=BLUE)
    ax[0].set(title="Acceptance rate", xlabel="proposal step", ylabel="acceptance")
    ax[1].semilogx(steps, ess, "o-", ms=4, color=ORANGE)
    ax[1].set(title="Effective sample size / n", xlabel="proposal step", ylabel="ESS / n")
    ax[2].hist(samples, bins=50, density=True, color=BLUE, alpha=0.6, label="Metropolis samples")
    ax[2].plot(grid, post_pdf, color=ORANGE, label="exact Beta posterior")
    ax[2].set(title="Posterior of a proportion", xlabel="$\\theta$", ylabel="density")
    ax[2].legend()
    _save(fig, path)


def plot_em(seps, err, fail, x, grid, dens, comps, path):
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.7))
    ax[0].plot(seps, err, "o-", ms=4, color=BLUE, label="median |mean error|")
    ax[0].plot(seps, fail, "o-", ms=4, color=ORANGE, label="share of runs with mean error > 1")
    ax[0].set(title="EM recovery vs component separation", xlabel="distance between means (sd units)", ylabel="value")
    ax[0].legend()
    ax[1].hist(x, bins=40, density=True, color=GREY, alpha=0.5, label="data")
    ax[1].plot(grid, dens, color=BLUE, label="fitted mixture")
    for c in comps:
        ax[1].plot(grid, c, color=ORANGE, lw=0.9, ls="--")
    ax[1].set(title="Fitted mixture (separation 3)", xlabel="$x$", ylabel="density")
    ax[1].legend()
    _save(fig, path)
