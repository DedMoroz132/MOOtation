# SPDX-License-Identifier: Apache-2.0
"""Choosing a model off a front of "error against identifiable parameters"
(task 26, item 24.7.5): the report's rules, applied to models already fitted.

A model is a dict with
  "name"    any label;
  "k"       its number of identifiable parameters (the rank of the Jacobian
            at the fit, lm's `rank`), the complexity every rule reads;
  "err"     its error on the selection sample, a sample the fit did not see
            (the rules choose on it, never on the training error);
  "rss", "n"  (for BIC) its residual sum of squares and the number of
            residuals it was fitted to;
  "test"    (optional) its error on a third sample, reported beside the
            choice and never used to make it.

eps_economy  the simplest model whose selection error is below the threshold
             C_th = max(floor, (1 + eps) C_min), C_min the smallest selection
             error (MS-1): M. Flaschel, S. Kumar, L. De Lorenzis, "Automated
             discovery of generalized standard material models with EUCLID"
             (source euclid-gsm_flaschel2022_arxiv): C_th = max(1e-5, 1.1
             C_min), "all solutions with C_j >= C_th are considered
             irrelevant", the simplest of the rest chosen; the same authors'
             "Discovering plasticity models without stress data" (source
             euclid-plasticity_flaschel2022_arxiv): C_th = 1.01 C_min; P.
             Thakolkaran et al., "NN-EUCLID: deep-learning hyperelasticity
             without stress data" (source nn-euclid_thakolkaran2022_arxiv):
             models "within 20% of the lowest loss" accepted (an acceptance
             rule there, not a choice of the simplest). eps 0.01, 0.1, 0.2
             are those three.
knee         the knee of the front by the utility-based measure of J.
             Branke, K. Deb, H. Dierolf, M. Osswald, "Finding knees in
             multi-objective optimization", PPSN VIII, LNCS 3242, 2004, §3.2
             (source finding-knees-multi_branke2004): the expected marginal
             utility of each point over linear utilities U = sum lambda_i f_i,
             sum lambda_i = 1, every weighting equally likely; the point of the
             largest is the knee.
bic          G. Schwarz's criterion ("Estimating the dimension of a model",
             Ann. Statist. 6(2):461-464, 1978) for Gaussian errors, n ln(RSS /
             n) + k ln n, with k the model's "k": the identifiable parameters,
             the rank, rather than all it declares.

READINGS:
  MS-1. EUCLID's threshold is strict (C < C_th); a model at C_min itself
    always qualifies, so the rule has an answer where C_th = C_min (eps = 0
    and C_min >= floor, or C_min = 0). Among the qualifying models: the
    fewest parameters, then the smallest selection error, then the first
    given.
  MS-2. The knee is taken in objectives scaled to [0, 1] over the front
    (Branke et al. do not scale; an error and a count of parameters have no
    common unit, and a linear utility over raw units would pick by the unit).
    An objective constant over the front scales to 0.
  MS-3. The expected marginal utility is estimated as the paper's §3.2 does
    beyond two objectives, by systematic sampling: for two objectives the
    weights lambda = (j + 1/2) / P, j = 0 ... P - 1, P = `precision` (the
    paper recommends at least the number of points; default 1000); for more,
    the simplex lattice of Das and Dennis with `precision` divisions. A
    point's marginal utility under one weighting is how much worse the
    second best is when it is the best (0 otherwise; on a tie for the best,
    0 for both).
  MS-4. The knee is chosen among the non-dominated points of what is given
    (duplicates count once, the first of them kept).
"""

from __future__ import annotations

import math
from typing import Sequence


def eps_economy(models: Sequence[dict], eps: float, *, floor: float = 0.0) -> dict | None:
    """The model eps_economy chooses (MS-1), or None without models."""
    if not models:
        return None
    if eps < 0.0:
        raise ValueError("eps_economy: eps >= 0")
    c_min = min(m["err"] for m in models)
    bound = max(floor, (1.0 + eps) * c_min)
    ok = [(m["k"], m["err"], i) for i, m in enumerate(models)
          if m["err"] < bound or m["err"] == c_min]
    return models[min(ok)[2]]


def bic(rss: float, n: int, k: int) -> float:
    """n ln(RSS / n) + k ln n (Gaussian errors); -inf at RSS = 0."""
    if n < 1:
        raise ValueError("bic: n >= 1")
    if rss <= 0.0:
        return -math.inf
    return n * math.log(rss / n) + k * math.log(n)


def _nondominated(points: Sequence[Sequence[float]]) -> list:
    keep = []
    for i, p in enumerate(points):
        if any(tuple(points[j]) == tuple(p) for j in keep):
            continue
        dominated = any(all(q[t] <= p[t] for t in range(len(p))) and
                        any(q[t] < p[t] for t in range(len(p)))
                        for j, q in enumerate(points) if j != i)
        if not dominated:
            keep.append(i)
    return keep


def _weights(m: int, precision: int) -> list:
    if m == 2:
        return [((j + 0.5) / precision, 1.0 - (j + 0.5) / precision) for j in range(precision)]
    out = []

    def rec(prefix, left, slots):
        if slots == 1:
            out.append(tuple(prefix + [left / precision]))
            return
        for v in range(left + 1):
            rec(prefix + [v / precision], left - v, slots - 1)
    rec([], precision, m)
    return out


def marginal_utilities(front: Sequence[Sequence[float]], *, precision: int = 1000) -> list:
    """Each point's expected marginal utility (MS-2, MS-3), all objectives
    minimized; a point dominated by another, or a repeat, gets 0 (MS-4)."""
    n = len(front)
    if n == 0:
        return []
    m = len(front[0])
    nd = _nondominated(front)
    lo = [min(front[i][t] for i in nd) for t in range(m)]
    hi = [max(front[i][t] for i in nd) for t in range(m)]
    scaled = {i: [((front[i][t] - lo[t]) / (hi[t] - lo[t]) if hi[t] > lo[t] else 0.0)
                  for t in range(m)] for i in nd}
    total = [0.0] * n
    ws = _weights(m, precision)
    if len(nd) < 2:
        return total
    for w in ws:
        u = sorted((sum(wt * s for wt, s in zip(w, scaled[i])), i) for i in nd)
        if u[1][0] > u[0][0]:
            total[u[0][1]] += u[1][0] - u[0][0]
    return [t / len(ws) for t in total]


def knee(front: Sequence[Sequence[float]], *, precision: int = 1000) -> int | None:
    """The index of the knee of `front` (the largest expected marginal
    utility; on a tie the first), or None for an empty front."""
    mu = marginal_utilities(front, precision=precision)
    if not mu:
        return None
    return max(range(len(mu)), key=lambda i: (mu[i], -i))


def choose(models: Sequence[dict], *, eps=(0.01, 0.1, 0.2), floor: float = 0.0,
           precision: int = 1000) -> dict:
    """Every rule's choice: {"eps_<eps>": model, "knee": model of the knee of
    (err, k), "bic": model of the smallest BIC where every model has rss and
    n}; each with its "test" error where the models carry one."""
    out = {}
    for e in eps:
        out[f"eps_{e:g}"] = eps_economy(models, e, floor=floor)
    if models:
        out["knee"] = models[knee([(m["err"], m["k"]) for m in models], precision=precision)]
    if models and all("rss" in m and "n" in m for m in models):
        out["bic"] = min(models, key=lambda m: bic(m["rss"], m["n"], m["k"]))
    return out
