# SPDX-License-Identifier: Apache-2.0
"""DAS-CMOP1-9: difficulty-adjustable constrained problems (task 7 of 2026-10-10, §8).

Z. Fan, W. Li, X. Cai, H. Li, C. Wei, Q. Zhang, K. Deb, E. Goodman,
"Difficulty Adjustable and Scalable Constrained Multi-objective Test Problem
Toolkit", Evolutionary Computation 28(3):339-378, 2020, doi:10.1162/evco_a_00259
(source das-cmop_fan2019, the arXiv version: Table 2, Eq. 7 and Table 3). The
formulas are the paper's Table 2, read off the rendered page. The objectives
agree with pymoo 0.6.2 (problems/multi/dascmop.py) at random points to 1e-12 at
every triplet, and so do the constraints but for the two readings below.

  DAS-CMOP1-3  f1 = x1 + g, f2 = 1 - x1^2 + g (1); 1 - sqrt(x1) + g (2);
               1 - sqrt(x1) + 0.5 |sin(5 pi x1)| + g (3);
               g = sum_{j=2..n} (x_j - sin(0.5 pi x1))^2
  DAS-CMOP4-6  the same f, g = (n - 1) + sum_{j=2..n} ((x_j - 0.5)^2
               - cos(20 pi (x_j - 0.5)))
  DAS-CMOP7    f1 = x1 x2 + g, f2 = x2 (1 - x1) + g, f3 = 1 - x2 + g,
               g = (n - 2) + sum_{j=3..n} ((x_j - 0.5)^2 - cos(20 pi (x_j - 0.5)))
  DAS-CMOP8    f = (cos(.5 pi x1) cos(.5 pi x2), cos(.5 pi x1) sin(.5 pi x2),
               sin(.5 pi x1)) + g, g as DAS-CMOP7
  DAS-CMOP9    the same f, g = sum_{j=3..n} (x_j - cos(0.25 j pi (x1 + x2) / n))^2
  n = 30, x in [0, 1]^n.

THE CONSTRAINTS (c >= 0 satisfied; the library's sign, <= 0 satisfied, is -c):
  two objectives  c1 = sin(a pi x1) - b                       Type I (diversity)
                  c2 = (e - g)(g - d)                         Type II (feasibility)
                  c_{k+2} = ((f1 - p_k) cos t - (f2 - q_k) sin t)^2 / a_k^2
                          + ((f1 - p_k) sin t + (f2 - q_k) cos t)^2 / b_k^2 - r,
                  k = 1..9, p = [0,1,0,1,2,0,1,2,3],
                  q = [1.5,0.5,2.5,1.5,0.5,3.5,2.5,1.5,0.5], a_k^2 = 0.3,
                  b_k^2 = 1.2, t = -0.25 pi                   Type III (convergence)
  three           c1 = sin(a pi x1) - b, c2 = cos(a pi x2) - b, c3 = (e - g)(g - d),
                  c_{k+3} = sum_{j != k} f_j^2 + (f_k - 1)^2 - r^2 (k = 1, 2, 3),
                  c7 = sum_j (f_j - 1/sqrt(3))^2 - r^2
  a = 20, d = 0.5, and the difficulty triplet (eta, zeta, gamma) in [0, 1]^3
  sets the rest (Eq. 7): b = 2 eta - 1, e = d - ln(zeta), r = gamma / 2; at
  zeta = 0 there is no upper bound e and d = 0 (Eq. 7, "if zeta == 0, d = 0"):
  c2 = g - d, which always holds (pymoo takes e = 1e30, (e - g)(g - d): the
  same sign everywhere, values 1e30 times larger). At zeta = 1, e = d: the
  second constraint is the equality g = d (the paper's triplets 13-16), taken
  as the paper's experiments take an equality (§7.3, Eq. 13, after Ullah et
  al. 2012): c2 = EQ_EPS - |g - d| >= 0, EQ_EPS = 1e-6 (pymoo: 1e-4).

THE TRIPLETS: the paper's sixteen (Table 3) are registered, DAS-CMOP<i>_t01 ..
_t16; any other through problem(i, eta, zeta, gamma), the knobs of an
ablation. Figure 6 shows (0, 0.5, 0.5) for DAS-CMOP1-3 and (0.5, 0.5, 0.5) for
4-9 — triplet 8 for 4-9; (0, 0.5, 0.5) is not in Table 3 and is registered as
DAS-CMOP<i>_f6 for 1-3.

THE FRONTS: the feasible non-dominated points of a dense sample of the
attainable set — g takes every value from its least (0) up (the distance
variables are free), so the sample is x1 (and x2) on a fine grid times g on a
grid of [d, e] (or [0, 1] above d when zeta = 0), each point kept if every
constraint holds; then the least feasible g of every non-dominated point by
bisection below its grid value; built on first use and cached. A construction of this
library, not the paper's (it plots them, Figure 6).
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import List

import numpy as np

N_VARS = 30
A, D = 20.0, 0.5
EQ_EPS = 1e-6                                  # Eq. 13's epsilon, zeta = 1
P_K = np.array([0, 1, 0, 1, 2, 0, 1, 2, 3], float)
Q_K = np.array([1.5, 0.5, 2.5, 1.5, 0.5, 3.5, 2.5, 1.5, 0.5])
AK2, BK2, THETA = 0.3, 1.2, -0.25 * math.pi
TRIPLETS = ((0.25, 0.0, 0.0), (0.0, 0.25, 0.0), (0.0, 0.0, 0.25), (0.25, 0.25, 0.25),
            (0.5, 0.0, 0.0), (0.0, 0.5, 0.0), (0.0, 0.0, 0.5), (0.5, 0.5, 0.5),
            (0.75, 0.0, 0.0), (0.0, 0.75, 0.0), (0.0, 0.0, 0.75), (0.75, 0.75, 0.75),
            (0.0, 1.0, 0.0), (0.5, 1.0, 0.0), (0.0, 1.0, 0.5), (0.5, 1.0, 0.5))
FIGURE6 = (0.0, 0.5, 0.5)                      # DAS-CMOP1-3's in Figure 6


def n_obj(i: int) -> int:
    return 2 if i <= 6 else 3


def params(eta: float, zeta: float, gamma: float):
    """(b, d, e, r) of a difficulty triplet (Eq. 7); e = inf where zeta = 0."""
    b = 2.0 * eta - 1.0
    d = 0.0 if zeta == 0.0 else D
    e = math.inf if zeta == 0.0 else D - math.log(zeta)
    return b, d, e, gamma / 2.0


def g_value(i: int, x) -> float:
    x = np.asarray(x, float)
    n = len(x)
    if i <= 3:
        return float(np.sum((x[1:] - math.sin(0.5 * math.pi * x[0])) ** 2))
    if i <= 6:
        z = x[1:] - 0.5
        return float((n - 1) + np.sum(z * z - np.cos(20.0 * math.pi * z)))
    if i <= 8:
        z = x[2:] - 0.5
        return float((n - 2) + np.sum(z * z - np.cos(20.0 * math.pi * z)))
    j = np.arange(3, n + 1)
    return float(np.sum((x[2:] - np.cos(0.25 * j / n * math.pi * (x[0] + x[1]))) ** 2))


def shape(i: int, x1: np.ndarray, x2=None) -> np.ndarray:
    """The objectives without g, rows of x1 (and x2)."""
    if i in (1, 4):
        return np.stack([x1, 1.0 - x1 ** 2], axis=-1)
    if i in (2, 5):
        return np.stack([x1, 1.0 - np.sqrt(x1)], axis=-1)
    if i in (3, 6):
        return np.stack([x1, 1.0 - np.sqrt(x1) + 0.5 * np.abs(np.sin(5.0 * math.pi * x1))],
                        axis=-1)
    if i == 7:
        return np.stack([x1 * x2, x2 * (1.0 - x1), 1.0 - x2], axis=-1)
    c1, s1 = np.cos(0.5 * math.pi * x1), np.sin(0.5 * math.pi * x1)
    return np.stack([c1 * np.cos(0.5 * math.pi * x2), c1 * np.sin(0.5 * math.pi * x2), s1],
                    axis=-1)


def constraints_c(i: int, F: np.ndarray, x1, x2, g, triplet) -> np.ndarray:
    """c >= 0 satisfied, one column per constraint, rows of F."""
    b, d, e, r = params(*triplet)
    F = np.atleast_2d(F)
    x1, x2, g = np.atleast_1d(x1), np.atleast_1d(x2), np.atleast_1d(g)
    # zeta = 0: no upper bound, and g >= d = 0 holds everywhere (c2 = g - d);
    # zeta = 1: the equality g = d within Eq. 13's epsilon
    if math.isinf(e):
        c_g = g - d
    elif e == d:
        c_g = EQ_EPS - np.abs(g - d)
    else:
        c_g = (e - g) * (g - d)
    if n_obj(i) == 2:
        f1 = F[:, 0:1] - P_K
        f2 = F[:, 1:2] - Q_K
        ell = ((f1 * math.cos(THETA) - f2 * math.sin(THETA)) ** 2 / AK2
               + (f1 * math.sin(THETA) + f2 * math.cos(THETA)) ** 2 / BK2 - r)
        return np.column_stack([np.sin(A * math.pi * x1) - b, c_g, ell])
    sq = F ** 2
    total = sq.sum(axis=1, keepdims=True)
    spheres = total - sq + (F - 1.0) ** 2 - r * r            # k = 1, 2, 3
    centre = np.sum((F - 1.0 / math.sqrt(3.0)) ** 2, axis=1, keepdims=True) - r * r
    return np.column_stack([np.sin(A * math.pi * x1) - b, np.cos(A * math.pi * x2) - b,
                            c_g, spheres, centre])


def evaluate(i: int, x, triplet) -> tuple:
    """(objectives, the library's constraints: <= 0 satisfied)."""
    x = np.asarray(x, float)
    g = g_value(i, x)
    F = shape(i, np.array([x[0]]), np.array([x[1]])) + g
    c = constraints_c(i, F, x[0], x[1], g, triplet)[0]
    return F[0].tolist(), [float(-v) for v in c]


def name(i: int, tag: str) -> str:
    return f"DAS-CMOP{i}_{tag}"


@lru_cache(maxsize=None)
def front(i: int, triplet: tuple, n_points: int = 1000) -> np.ndarray:
    """The constrained front from a dense sample (THE FRONTS above)."""
    from ..run.archive import dss_order
    b, d, e, r = params(*triplet)
    top = d + 1.0 if math.isinf(e) else e
    gs = np.linspace(d, top, 401)
    if n_obj(i) == 2:
        x1 = np.linspace(0.0, 1.0, 20001)
        X1, G = np.meshgrid(x1, gs, indexing="ij")
        X1, G = X1.ravel(), G.ravel()
        X2 = np.zeros_like(X1)
    else:
        x = np.linspace(0.0, 1.0, 301)
        g3 = np.linspace(d, top, 41)
        X1, X2, G = (a.ravel() for a in np.meshgrid(x, x, g3, indexing="ij"))
    F = shape(i, X1, X2) + G[:, None]
    ok = np.all(constraints_c(i, F, X1, X2, G, triplet) >= -1e-12, axis=1)
    F, X1, X2, G = F[ok], X1[ok], X2[ok], G[ok]
    if not len(F):
        return np.empty((0, n_obj(i)))
    k = _nd_index(F)
    F, X1, X2, G = F[k], X1[k], X2[k], G[k]
    # the least feasible g of every kept point, by bisection between the grid
    # step below (infeasible, or below d) and the grid value
    step = gs[1] - gs[0] if n_obj(i) == 2 else (top - d) / 40.0
    # (zeta = 1: g = d within EQ_EPS, so the least feasible g is d - EQ_EPS)
    lo = np.full_like(G, d - EQ_EPS) if e == d else np.maximum(G - step, d)
    hi = G.copy()
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        Fm = shape(i, X1, X2) + mid[:, None]
        good = np.all(constraints_c(i, Fm, X1, X2, mid, triplet) >= -1e-12, axis=1)
        hi = np.where(good, mid, hi)
        lo = np.where(good, lo, mid)
    F = shape(i, X1, X2) + hi[:, None]
    F = F[_nd_index(F)]
    idx = dss_order(F, n_points, ideal=F.min(axis=0), nadir=F.max(axis=0))
    return F[np.asarray(idx, int)]


def _nd_index(F: np.ndarray) -> np.ndarray:
    """The non-dominated rows' indices: re_fronts' sweeps, two or three objectives."""
    from .re_fronts import _nd_2d, _nd_3d
    return (_nd_2d if F.shape[1] == 2 else _nd_3d)(F)


def frame(i: int, triplet: tuple):
    Z = front(i, triplet)
    return tuple(Z.min(axis=0)), tuple(Z.max(axis=0))


def problem(i: int, eta: float, zeta: float, gamma: float):
    """(evaluate, constraints) of DAS-CMOP<i> at any triplet: the knobs."""
    t = (float(eta), float(zeta), float(gamma))
    return (lambda x: evaluate(i, x, t)[0]), (lambda x: evaluate(i, x, t)[1])
