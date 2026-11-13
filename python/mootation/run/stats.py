# SPDX-License-Identifier: Apache-2.0
"""Statistics for campaign results: is a difference real, and how large is it?

A mean rank says where an algorithm stands; it does not say whether it stands
there by more than the spread between seeds. This module adds the four
answers a comparison of optimizers is expected to give:

  * wilcoxon_signed_rank — across PROBLEMS, one algorithm against a reference:
    the paired differences of their per-problem medians. With Holm's
    correction when several algorithms are tested against the same reference.
  * rank_sum — on ONE problem, one algorithm's seeds against the reference's.
  * a12 — the Vargha-Delaney effect size on one problem: the probability that
    a run of A beats a run of B, ties counting half. 0.5 is no effect.
  * bootstrap_mean_ranks — a percentile interval for each mean rank, by
    resampling the problems.

Both of Wilcoxon's tests (Wilcoxon, "Individual comparisons by ranking
methods", Biometrics Bulletin 1(6):80-83, 1945) are computed EXACTLY, ties
included: the null distribution of the statistic is built by dynamic
programming over doubled ranks (average ranks of ties are half-integers), which
is fast for any number of problems a campaign has — no normal approximation,
whose tie correction is exactly where approximations go wrong on the heavily
tied medians of converged runs. Zero differences are dropped, as in
Wilcoxon's paper.

Orientation: every function takes `lower_better`; "A better" always means
better for the metric, whichever way it runs.

A caution that no test fixes: the signed-rank test ranks the SIZE of the
differences across problems, so on a raw-unit metric (igd, igdp, eps) a
problem whose values are a thousand times larger decides the result. Use it on
the scale-free ones — hv, hv_h, igdp_norm, eps_norm — and on the last two, by
the logarithm (report.reference_report does): dimensionless is not
homogeneous, igdp_norm runs over orders of magnitude between problems, and
the problems where an algorithm fails would decide the ranks of the raw
differences. sign_test and hodges_lehmann go beside it: the sign test asks
on how many problems, the signed-rank test how much, and they can disagree.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence


# ── ranks ────────────────────────────────────────────────────────────────────
def average_ranks(values: Sequence[float]) -> list:
    """Ranks 1..n of `values`, ties sharing the average of their ranks."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        r = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = r
        i = j + 1
    return ranks


def _subset_sum_distribution(weights: Sequence[int]):
    """P(sum of a uniformly random subset of `weights` = s), s = 0..sum(w).

    Each item in or out with probability 1/2 — the signed-rank null.
    """
    import numpy as np
    total = int(sum(weights))
    p = np.zeros(total + 1)
    p[0] = 1.0
    for w in weights:
        q = p.copy()
        q[w:] += p[:total + 1 - w]
        p = 0.5 * q
    return p


def _k_subset_sum_distribution(weights: Sequence[int], k: int):
    """P(sum of a uniformly random k-subset of `weights` = s) — the rank-sum null."""
    import numpy as np
    total = int(sum(weights))
    # ways[j][s]: number of j-subsets of the items seen so far with sum s
    ways = np.zeros((k + 1, total + 1))
    ways[0, 0] = 1.0
    for w in weights:
        for j in range(min(k, len(weights)), 0, -1):
            ways[j, w:] += ways[j - 1, :total + 1 - w]
    count = ways[k].sum()
    return ways[k] / count


def _two_sided(p_dist, observed: int) -> float:
    """Two-sided p-value of an integer statistic against its null distribution:
    twice the smaller tail, the tails measured from the observed value."""
    lower = float(p_dist[:observed + 1].sum())
    upper = float(p_dist[observed:].sum())
    return min(1.0, 2.0 * min(lower, upper))


# ── Wilcoxon signed-rank, across problems ───────────────────────────────────
def wilcoxon_signed_rank(a: Sequence[float], b: Sequence[float], *,
                         lower_better: bool = True) -> dict:
    """Paired test of a against b (one value per problem, same order).

    Returns {"n": problems used (zeros dropped), "w_plus": sum of the ranks
    where A is better, "w_minus", "p": exact two-sided p-value, "better":
    "A" | "B" | None (which way the ranks lean)}.
    """
    if len(a) != len(b):
        raise ValueError("wilcoxon_signed_rank needs paired samples of equal length")
    d = []
    for x, y in zip(a, b):
        if x is None or y is None or math.isnan(x) or math.isnan(y) or x == y:
            continue
        # an infinite value is an empty run's (report.values): the largest
        # difference there is, in its direction
        diff = (y - x) if lower_better else (x - y)        # > 0: A better
        if diff != 0.0:
            d.append(diff)
    n = len(d)
    if n == 0:
        return {"n": 0, "w_plus": 0.0, "w_minus": 0.0, "p": 1.0, "better": None}
    r = average_ranks([abs(v) for v in d])
    w_plus = sum(rk for rk, v in zip(r, d) if v > 0)
    w_minus = sum(rk for rk, v in zip(r, d) if v < 0)
    doubled = [int(round(2 * rk)) for rk in r]
    p = _two_sided(_subset_sum_distribution(doubled), int(round(2 * w_plus)))
    better = "A" if w_plus > w_minus else ("B" if w_minus > w_plus else None)
    return {"n": n, "w_plus": w_plus, "w_minus": w_minus, "p": p, "better": better}


# ── Wilcoxon rank-sum, on one problem ───────────────────────────────────────
def rank_sum(a: Sequence[float], b: Sequence[float], *, lower_better: bool = True) -> dict:
    """Unpaired test of a's runs against b's on one problem (exact, ties included).

    Returns {"n_a", "n_b", "r_a": rank sum of A (ranks by quality, 1 = best),
    "p", "better"}. With 5 seeds a side the smallest attainable p is 1/126 ≈
    0.008, so 5-seed campaigns cannot resolve much below 0.01.
    """
    a = [float(v) for v in a if v is not None and not math.isnan(v)]   # infinity: an
    b = [float(v) for v in b if v is not None and not math.isnan(v)]   # empty run, worst
    if not a or not b:
        return {"n_a": len(a), "n_b": len(b), "r_a": None, "p": 1.0, "better": None}
    pooled = a + b
    key = pooled if lower_better else [-v for v in pooled]
    r = average_ranks(key)                                 # 1 = best
    r_a = sum(r[:len(a)])
    doubled = [int(round(2 * rk)) for rk in r]
    dist = _k_subset_sum_distribution(doubled, len(a))
    p = _two_sided(dist, int(round(2 * r_a)))
    expected = len(a) * (len(pooled) + 1) / 2.0
    better = "A" if r_a < expected else ("B" if r_a > expected else None)
    return {"n_a": len(a), "n_b": len(b), "r_a": r_a, "p": p, "better": better}


# ── effect size ──────────────────────────────────────────────────────────────
def a12(a: Sequence[float], b: Sequence[float], *, lower_better: bool = True) -> float | None:
    """Vargha-Delaney A12: P(a run of A is better than a run of B), ties count half.

    0.5 is no effect; the conventional thresholds are 0.56 / 0.64 / 0.71
    (small / medium / large) on either side of 0.5.
    """
    a = [float(v) for v in a if v is not None and not math.isnan(v)]
    b = [float(v) for v in b if v is not None and not math.isnan(v)]
    if not a or not b:
        return None
    wins = 0.0
    for x in a:
        for y in b:
            if x == y:
                wins += 0.5
            elif (x < y) == lower_better:
                wins += 1.0
    return wins / (len(a) * len(b))


def a12_magnitude(v: float | None) -> str:
    if v is None:
        return "-"
    e = abs(v - 0.5)
    return "large" if e >= 0.21 else "medium" if e >= 0.14 else "small" if e >= 0.06 else "negligible"


# ── multiple comparisons ─────────────────────────────────────────────────────
def holm(pvalues: Sequence[float]) -> list:
    """Holm's step-down adjusted p-values, in the input order."""
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvalues[i]))
        adjusted[i] = running
    return adjusted


def benjamini_hochberg(pvalues: Sequence[float]) -> list:
    """Benjamini & Hochberg's step-up adjusted p-values (J. R. Statist. Soc. B
    57(1), 1995), in the input order: they hold the false discovery rate where
    Holm's hold the chance of any false mark."""
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i], reverse=True)
    adjusted = [0.0] * m
    running = 1.0
    for k, i in enumerate(order):                 # the largest p first: rank m
        running = min(running, pvalues[i] * m / (m - k))
        adjusted[i] = min(1.0, running)
    return adjusted


def smallest_rank_sum_p(n_a: int, n_b: int) -> float:
    """The smallest two-sided p the exact rank-sum test can give with n_a and
    n_b runs, the two samples wholly apart: 2 / C(n_a + n_b, n_a)."""
    return min(1.0, 2.0 / math.comb(n_a + n_b, n_a))


# ── across problems, besides the signed-rank test ───────────────────────────
def sign_test(a: Sequence[float], b: Sequence[float], *, lower_better: bool = True) -> dict:
    """Paired sign test of a against b: the pairs A wins and loses (equal
    pairs dropped) and the exact two-sided binomial p."""
    wins = losses = 0
    for x, y in zip(a, b):
        if x is None or y is None or math.isnan(x) or math.isnan(y) or x == y:
            continue
        if (x < y) == lower_better:
            wins += 1
        else:
            losses += 1
    n = wins + losses
    if n == 0:
        return {"n": 0, "wins": 0, "losses": 0, "p": 1.0}
    tail = sum(math.comb(n, i) for i in range(min(wins, losses) + 1))
    return {"n": n, "wins": wins, "losses": losses, "p": min(1.0, 2.0 * tail / 2.0 ** n)}


def hodges_lehmann(d: Sequence[float]) -> float | None:
    """The Hodges-Lehmann estimate of where the differences d centre: the
    median of their Walsh averages (d_i + d_j) / 2, i <= j (Hodges & Lehmann,
    Ann. Math. Statist. 34(2), 1963) — the estimate that goes with the
    signed-rank test."""
    d = [float(v) for v in d if v is not None and math.isfinite(v)]
    if not d:
        return None
    w = sorted((d[i] + d[j]) / 2.0 for i in range(len(d)) for j in range(i, len(d)))
    n = len(w)
    return w[n // 2] if n % 2 else 0.5 * (w[n // 2 - 1] + w[n // 2])


# ── bootstrap of mean ranks ──────────────────────────────────────────────────
def bootstrap_mean_ranks(ranks_by_problem: dict, *, n_boot: int = 2000,
                         level: float = 0.95, seed: int = 20260922) -> dict:
    """Percentile intervals for each algorithm's mean rank, resampling problems.

    `ranks_by_problem` is {problem: {algorithm: rank}}; an algorithm missing on
    a resampled problem simply has one term fewer there. Returns
    {algorithm: (low, high)}.
    """
    import numpy as np
    probs = sorted(ranks_by_problem)
    algs = sorted({a for r in ranks_by_problem.values() for a in r})
    R = np.full((len(probs), len(algs)), np.nan)
    for i, p in enumerate(probs):
        for j, a in enumerate(algs):
            if a in ranks_by_problem[p]:
                R[i, j] = ranks_by_problem[p][a]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(probs), size=(n_boot, len(probs)))
    means = np.nanmean(R[idx], axis=1)                      # (n_boot, algs)
    lo_q, hi_q = (1 - level) / 2 * 100, (1 + level) / 2 * 100
    lo = np.nanpercentile(means, lo_q, axis=0)
    hi = np.nanpercentile(means, hi_q, axis=0)
    return {a: (float(lo[j]), float(hi[j])) for j, a in enumerate(algs)}


def friedman(blocks: Sequence[Sequence[float]], *, lower_better: bool = True) -> dict:
    """Friedman's test of k treatments over n blocks (M. Friedman, "The use of
    ranks to avoid the assumption of normality implicit in the analysis of
    variance", JASA 32(200):675-701, 1937): each block's values ranked 1..k
    (ties sharing the average rank), the statistic with the correction for ties
    T = (k - 1) sum_j (R_j - n (k + 1) / 2)^2 / (sum_ij r_ij^2 - n k (k + 1)^2 / 4),
    R_j the treatments' rank sums, against chi^2 with k - 1 degrees of freedom
    (W.J. Conover, Practical Nonparametric Statistics, 3rd ed., §5.8). Every
    block has a value of every treatment. Returns {"n", "k", "statistic",
    "p", "mean_ranks"}; with no spread at all (every block tied
    throughout) the statistic is 0 and p is 1."""
    from .report import _chi2_sf
    n = len(blocks)
    k = len(blocks[0]) if n else 0
    if n == 0 or k < 2:
        return {"n": n, "k": k, "statistic": 0.0, "p": 1.0, "mean_ranks": [1.0] * k}
    ranks = [average_ranks([v if lower_better else -v for v in b]) for b in blocks]
    R = [sum(r[j] for r in ranks) for j in range(k)]
    a = sum(x * x for r in ranks for x in r)
    c = n * k * (k + 1) ** 2 / 4.0
    if a - c <= 0.0:
        return {"n": n, "k": k, "statistic": 0.0, "p": 1.0, "mean_ranks": [rj / n for rj in R]}
    t = (k - 1) * sum((rj - n * (k + 1) / 2.0) ** 2 for rj in R) / (a - c)
    return {"n": n, "k": k, "statistic": t, "p": _chi2_sf(t, k - 1), "mean_ranks": [rj / n for rj in R]}


# ── after Friedman: Iman-Davenport, Nemenyi, Bonferroni-Dunn (task 26, Г5) ────
# J. Demšar, "Statistical comparisons of classifiers over multiple data sets",
# JMLR 7:1-30, 2006 (source statistical-comparisons-classifiers_demsar2006),
# §3.2.2 and §3.2.4:
#   Iman and Davenport's F_F = (N - 1) chi2_F / (N (k - 1) - chi2_F) against
#     F with k - 1 and (k - 1)(N - 1) degrees of freedom;
#   Nemenyi (all against all): two algorithms differ when their mean ranks
#     differ by at least CD = q_alpha sqrt(k (k + 1) / (6 N));
#   Bonferroni-Dunn (all against a control): the same CD with the critical
#     value of alpha / (k - 1);
#   the CD diagram: the mean ranks on an axis, the groups of algorithms no
#     two of which differ by CD connected (Figure 1(a)), or the interval of
#     one CD around the control (Figure 1(b)).
# READINGS:
#   ST-1. The critical values are computed, not looked up, so any k works:
#     Nemenyi's q_alpha is the studentized range's 1 - alpha quantile with
#     infinite degrees of freedom over sqrt 2 (Demšar's Table 5(a) at k = 2 is
#     z_0.975 = 1.960 = 2.772 / sqrt 2), P(range of k standard normals <= w) =
#     k int phi(z) (Phi(z + w) - Phi(z))^(k - 1) dz by Simpson's rule on
#     [-9, 9]; Bonferroni-Dunn's is z_(1 - alpha / (2 (k - 1))) (two-tailed).
#     Both agree with Table 5 within 1.5e-3 for k = 2 ... 10 (tests): the
#     table rounds from the studentized range's three-digit values (k = 7:
#     4.17 / sqrt 2 = 2.949, the quantile 4.1696 gives 2.9483), and its
#     Bonferroni-Dunn entry 2.724 at k = 9 is a misprint of z_(1 - 0.05/16) =
#     2.7344, the value this function gives (the test takes it).
#   ST-2. F_F takes the chi2_F it is given: `friedman` returns Conover's
#     statistic corrected for ties, Demšar's formula has no correction; they
#     agree without ties. The example of Demšar's §3.2.2 (chi2_F = 9.28, N =
#     14, k = 4, F_F = 3.69) checks the formula.
#   ST-3. The groups of the diagram are the maximal runs of algorithms,
#     sorted by mean rank, whose first and last differ by less than CD (a
#     difference of exactly CD is significant, "at least the critical
#     difference"); a run inside a longer one is not listed.
def f_sf(x: float, d1: float, d2: float) -> float:
    """P(F_{d1,d2} >= x) = I_{d2/(d2 + d1 x)}(d2/2, d1/2), the regularized
    incomplete beta function by its continued fraction (DLMF 8.17.22, Lentz's
    method), to about 1e-13."""
    if x <= 0.0:
        return 1.0
    return _betainc(d2 / 2.0, d1 / 2.0, d2 / (d2 + d1 * x))


def _betainc(a: float, b: float, x: float) -> float:
    """The regularized incomplete beta function I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    if x > (a + 1.0) / (a + b + 2.0):              # the fraction converges fast below
        return 1.0 - _betainc(b, a, 1.0 - x)
    front = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                     + a * math.log(x) + b * math.log1p(-x)) / a
    tiny = 1e-300
    f, c, d = 1.0, 1.0, 0.0
    for i in range(0, 2000):                   # DLMF 8.17.22: 1 / (1 + d1 / (1 + d2 / ...))
        m = i // 2
        if i == 0:
            num = 1.0
        elif i % 2 == 0:
            num = m * (b - m) * x / ((a + 2 * m - 1) * (a + 2 * m))
        else:
            num = -(a + m) * (a + b + m) * x / ((a + 2 * m) * (a + 2 * m + 1))
        d = 1.0 + num * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + num / c
        c = c if abs(c) > tiny else tiny
        f *= c * d
        if abs(c * d - 1.0) < 1e-15:
            break
    return front * (f - 1.0)


def iman_davenport(chi2: float, n: int, k: int) -> dict:
    """Iman and Davenport's F_F from Friedman's chi2_F over n blocks and k
    treatments, and its p against F(k - 1, (k - 1)(n - 1)) (Demšar §3.2.2;
    ST-2). Where chi2_F reaches n (k - 1), its largest value (every block
    ranks alike), F_F is infinite and p is 0."""
    d1, d2 = k - 1, (k - 1) * (n - 1)
    den = n * (k - 1) - chi2
    if d1 < 1 or d2 < 1:
        return {"statistic": 0.0, "df": (d1, d2), "p": 1.0}
    if den <= 0.0:
        return {"statistic": math.inf, "df": (d1, d2), "p": 0.0}
    ff = (n - 1) * chi2 / den
    return {"statistic": ff, "df": (d1, d2), "p": f_sf(ff, d1, d2)}


def _range_cdf(w: float, k: int) -> float:
    """P(max - min of k standard normals <= w) (ST-1)."""
    from statistics import NormalDist
    nd = NormalDist()
    steps, lo, hi = 1800, -9.0, 9.0
    h = (hi - lo) / steps
    total = 0.0
    for i in range(steps + 1):
        z = lo + i * h
        g = nd.pdf(z) * (nd.cdf(z + w) - nd.cdf(z)) ** (k - 1)
        total += g * (1 if i in (0, steps) else 4 if i % 2 else 2)
    return k * total * h / 3.0


def nemenyi_q(k: int, alpha: float = 0.05) -> float:
    """Nemenyi's two-tailed critical value q_alpha for k algorithms (ST-1):
    the studentized range's quantile at infinite degrees of freedom over
    sqrt 2, by bisection to 1e-10."""
    if k < 2:
        raise ValueError("nemenyi_q: at least two algorithms")
    lo, hi = 0.0, 20.0
    while hi - lo > 1e-10:
        mid = 0.5 * (lo + hi)
        if _range_cdf(mid, k) < 1.0 - alpha:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi) / math.sqrt(2.0)


def bonferroni_dunn_q(k: int, alpha: float = 0.05) -> float:
    """Bonferroni-Dunn's two-tailed critical value for k algorithms, the
    control among them: z at 1 - alpha / (2 (k - 1)) (Demšar §3.2.2; ST-1)."""
    from statistics import NormalDist
    if k < 2:
        raise ValueError("bonferroni_dunn_q: at least two algorithms")
    return NormalDist().inv_cdf(1.0 - alpha / (2.0 * (k - 1)))


def critical_difference(k: int, n: int, alpha: float = 0.05, *, test: str = "nemenyi") -> float:
    """CD = q_alpha sqrt(k (k + 1) / (6 n)) for k algorithms over n blocks;
    `test` "nemenyi" (all against all) or "bonferroni_dunn" (against a
    control)."""
    if test == "nemenyi":
        q = nemenyi_q(k, alpha)
    elif test == "bonferroni_dunn":
        q = bonferroni_dunn_q(k, alpha)
    else:
        raise ValueError(f"critical_difference: test = {test!r}: nemenyi or bonferroni_dunn")
    return q * math.sqrt(k * (k + 1) / (6.0 * n))


def cd_groups(mean_ranks: dict, cd: float) -> list:
    """The groups a CD diagram connects (Demšar Figure 1(a); ST-3): from
    {algorithm: mean rank}, the maximal runs, by mean rank, whose first and
    last differ by less than `cd`, each as a list of algorithms from the best
    rank; runs of one algorithm are left out."""
    algs = sorted(mean_ranks, key=lambda a: (mean_ranks[a], a))
    runs, last_end = [], -1
    for i in range(len(algs)):
        j = i
        while j + 1 < len(algs) and mean_ranks[algs[j + 1]] - mean_ranks[algs[i]] < cd:
            j += 1
        if j > last_end and j > i:
            runs.append(algs[i:j + 1])
        last_end = max(last_end, j)
    return runs


def compare_ranks(blocks: Sequence[Sequence[float]], names: Sequence[str], *,
                  lower_better: bool = True, alpha: float = 0.05,
                  control: str | None = None) -> dict:
    """Friedman over the blocks (each a value of every algorithm, in the order
    of `names`), Iman and Davenport's F_F, and the post-hoc test: with no
    `control`, Nemenyi's CD and the groups of the CD diagram; with one,
    Bonferroni-Dunn's CD and the algorithms whose mean rank differs from the
    control's by at least CD (Demšar §3.2.2, §3.2.4)."""
    fr = friedman(blocks, lower_better=lower_better)
    n, k = fr["n"], fr["k"]
    ranks = dict(zip(names, fr["mean_ranks"]))
    out = {"n": n, "k": k, "mean_ranks": ranks, "friedman": fr,
           "iman_davenport": iman_davenport(fr["statistic"], n, k)}
    if n == 0 or k < 2:
        return out
    if control is None:
        cd = critical_difference(k, n, alpha, test="nemenyi")
        out.update(test="nemenyi", cd=cd, groups=cd_groups(ranks, cd))
    else:
        if control not in ranks:
            raise ValueError(f"compare_ranks: control {control!r} is not among the algorithms")
        cd = critical_difference(k, n, alpha, test="bonferroni_dunn")
        out.update(test="bonferroni_dunn", cd=cd, control=control,
                   differ=sorted(a for a in ranks if a != control
                                 and abs(ranks[a] - ranks[control]) >= cd))
    return out


def kendall_tau(a: Sequence[float], b: Sequence[float]) -> float | None:
    """Kendall's tau-b between two rankings of the same items (ties allowed):
    (concordant - discordant) / sqrt((n0 - n1)(n0 - n2)), n0 the pairs, n1 and
    n2 those tied in a and in b (M.G. Kendall, "The treatment of ties in
    ranking problems", Biometrika 33(3):239-251, 1945). None when either
    ranking is constant. For the ranks of the algorithms at two budgets
    (Г5: how stable the order is along the ladder of budgets)."""
    n = len(a)
    if n != len(b):
        raise ValueError("kendall_tau: two rankings of the same length")
    conc = disc = ta = tb = 0
    for i in range(n):
        for j in range(i + 1, n):
            da, db = a[i] - a[j], b[i] - b[j]
            if da == 0 and db == 0:
                ta += 1
                tb += 1
            elif da == 0:
                ta += 1
            elif db == 0:
                tb += 1
            elif (da > 0) == (db > 0):
                conc += 1
            else:
                disc += 1
    n0 = n * (n - 1) // 2
    den = math.sqrt((n0 - ta) * (n0 - tb))
    return None if den == 0 else (conc - disc) / den


def mark(p: float | None, better: str | None, alpha: float = 0.05) -> str:
    """The usual table marker: '+' A significantly better, '-' worse, '=' neither."""
    if p is None or better is None or p >= alpha:
        return "="
    return "+" if better == "A" else "-"


def summarize(values: Iterable[float]) -> tuple:
    """(median, q1, q3) of finite values; Nones and NaNs are ignored."""
    v = sorted(float(x) for x in values if x is not None and math.isfinite(x))
    if not v:
        return (float("nan"),) * 3
    def q(f):
        k = f * (len(v) - 1)
        lo, hi = math.floor(k), math.ceil(k)
        return v[lo] + (v[hi] - v[lo]) * (k - lo)
    return (q(0.5), q(0.25), q(0.75))
