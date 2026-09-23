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
the scale-free ones — hv, hv_h, igdp_norm, eps_norm.
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
        if x is None or y is None or not (math.isfinite(x) and math.isfinite(y)):
            continue
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
    a = [float(v) for v in a if v is not None and math.isfinite(v)]
    b = [float(v) for v in b if v is not None and math.isfinite(v)]
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
    a = [float(v) for v in a if v is not None and math.isfinite(v)]
    b = [float(v) for v in b if v is not None and math.isfinite(v)]
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
