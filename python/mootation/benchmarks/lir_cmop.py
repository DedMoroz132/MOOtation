# SPDX-License-Identifier: Apache-2.0
"""LIR-CMOP1-14: constrained problems with large infeasible regions (task 7 of
2026-10-10, §8).

Z. Fan, W. Li, X. Cai, H. Huang, Y. Fang, Y. You, J. Mo, C. Wei, E. Goodman,
"An improved epsilon constraint-handling method in MOEA/D for CMOPs with large
infeasible regions", Soft Computing 23(23):12491-12510, 2019,
doi:10.1007/s00500-019-03794-x (source lir-cmop+moead-iepsilon_fan2019_arxiv,
arXiv:1707.08767v1: Table 6, read off the rendered page; the published version
was not at hand). n = 30, x in [0, 1]^30; J1 = {3, 5, ..., 29}, J2 = {2, 4,
..., 30} (1-based); constraints c >= 0 satisfied, the library's sign is -c.

  1-4    f1 = x1 + g1, f2 = 1 - x1^2 + g2 (1, 3) or 1 - sqrt(x1) + g2 (2, 4),
         g1 = sum_J1 (x_j - sin(0.5 pi x1))^2, g2 = sum_J2 (x_j - cos(0.5 pi x1))^2;
         c1 = (a - g1)(g1 - b), c2 = (a - g2)(g2 - b), a = 0.51, b = 0.5: each g
         in [0.5, 0.51], a narrow feasible band; 3 and 4 add
         c3 = sin(20 pi x1) - 0.5.
  5-8    f1 = x1 + 10 g1 + 0.7057, f2 = 1 - sqrt(x1) (5, 7) or 1 - x1^2 (6, 8)
         + 10 g2 + 0.7057, g1 = sum_J1 (x_i - sin(0.5 i/30 pi x1))^2,
         g2 = sum_J2 (x_j - cos(0.5 j/30 pi x1))^2; ellipses
         c_k = ((f1 - p_k) cos t - (f2 - q_k) sin t)^2 / a_k^2
             + ((f1 - p_k) sin t + (f2 - q_k) cos t)^2 / b_k^2 - r, t = -0.25 pi,
         r = 0.1: 5 p = q = (1.6, 2.5), a = (2, 2), b = (4, 8); 6 p = q =
         (1.8, 2.8), a = (2, 2), b = (8, 8); 7 and 8 p = q = (1.2, 2.25, 3.5),
         a = (2, 2.5, 2.5), b = (6, 12, 10).
  9-12   f1 = 1.7057 x1 (10 g1 + 1), f2 = 1.7057 (1 - x1^2) (9, 12) or
         (1 - sqrt(x1)) (10, 11) (10 g2 + 1), g as 5-8; one ellipse c1 (p, q,
         a, b = 1.4, 1.4, 1.5, 6.0; 1.1, 1.2, 2.0, 4.0; 1.2, 1.2, 1.5, 5.0;
         1.6, 1.6, 1.5, 6.0) and c2 = f1 sin al + f2 cos al
         - sin(4 pi (f1 cos al - f2 sin al)) - e, al = 0.25 pi, e = 2, 1, 2.1,
         2.5.
  13-14  f = (1.7057 + g1) (cos(.5 pi x1) cos(.5 pi x2), cos(.5 pi x1)
         sin(.5 pi x2), sin(.5 pi x1)), g1 = sum_{i=3..30} 10 (x_i - 0.5)^2;
         g = f1^2 + f2^2 + f3^2, c1 = (g - 9)(g - 4), c2 = (g - 3.61)(g - 3.24),
         14 adds c3 = (g - 3.0625)(g - 2.56).

The table prints the second problem's name as "LIR-d2", and no ">= 0" after
LIR-CMOP13's and 14's constraints; read as for the others (the text: 13's
front is the unconstrained one, 14's lies on its constraints, which holds
only so).

THE FRONTS, from the problems' geometry (as mw.py's). A g is any value >= 0
(its variables are free), so with u = x1 the attainable objectives are, for
1-8, f1 >= u + s1 and f2 >= h(u) + s2 (s the shift: 0, or 0.7057), h the
shape (1 - u^2 or 1 - sqrt(u)), and for 9-12 the same scaled by 1.7057 with
factors (10 g + 1) >= 1. Then:
  1-4    g1, g2 in [0.5, 0.51]: the front is u + 0.5, h(u) + 0.5 (3, 4: on
         the windows where sin(20 pi u) >= 0.5), its least g in both;
  5-12   at every f1 of a grid (6001 values, 40 times finer on the steep
         parts) the least attainable f2 that satisfies every constraint, by
         a scan in f2 and bisection; then the non-dominated part. 9, 11 and
         12 keep (0, f2) and 9-12 (f1, 0): f1 = 0 only at x1 = 0 and f2 = 0
         only at x1 = 1, so the least feasible point on that axis is
         Pareto-optimal however far it lies from the rest (9: (0, 2.175...));
         the frame read off the front takes it;
  13-14  the sphere of radius 1.7057 (13), 1.75 (14: on c3's boundary,
         g = 3.0625), in the positive octant, the unconstrained front's
         shape.
A construction of this library, checked against the paper's Fig. 1-2: every
point of 5-12's fronts lies on the unconstrained front or on a constraint's
boundary; 5 and 6 on the unconstrained one, 7 and 8 on the boundary, 9 and 10
on pieces of the unconstrained one, 11 and 12 on the boundary, as its text
says.
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

from ._rows import math_cos, math_sin

N_VARS = 30
J1 = np.arange(3, 30, 2) - 1                  # 0-based
J2 = np.arange(2, 31, 2) - 1
SHIFT = 0.7057
A, B, C = 0.51, 0.5, 20.0
R, THETA, ALPHA = 0.1, -0.25 * math.pi, 0.25 * math.pi
ELLIPSES = {5: ((1.6, 2.5), (1.6, 2.5), (2.0, 2.0), (4.0, 8.0)),
            6: ((1.8, 2.8), (1.8, 2.8), (2.0, 2.0), (8.0, 8.0)),
            7: ((1.2, 2.25, 3.5), (1.2, 2.25, 3.5), (2.0, 2.5, 2.5), (6.0, 12.0, 10.0)),
            8: ((1.2, 2.25, 3.5), (1.2, 2.25, 3.5), (2.0, 2.5, 2.5), (6.0, 12.0, 10.0)),
            9: ((1.4,), (1.4,), (1.5,), (6.0,)),
            10: ((1.1,), (1.2,), (2.0,), (4.0,)),
            11: ((1.2,), (1.2,), (1.5,), (5.0,)),
            12: ((1.6,), (1.6,), (1.5,), (6.0,))}
SINE_E = {9: 2.0, 10: 1.0, 11: 2.1, 12: 2.5}
SPHERES = {13: ((9.0, 4.0), (3.61, 3.24)), 14: ((9.0, 4.0), (3.61, 3.24), (3.0625, 2.56))}


def n_obj(i: int) -> int:
    return 3 if i >= 13 else 2


def n_cons(i: int) -> int:
    return {1: 2, 2: 2, 3: 3, 4: 3, 13: 2, 14: 3}.get(i, 2 if i >= 9 else len(ELLIPSES[i][0]))


def _sqrt_shape(i: int) -> bool:
    return i in (2, 4, 5, 7, 10, 11)


def shape(i: int, u):
    u = np.asarray(u, float)
    return 1.0 - np.sqrt(u) if _sqrt_shape(i) else 1.0 - u * u


def g_values(i: int, X):
    """(g1, g2) of every row of X (n x 30). The odd and even variables are
    taken as C-ordered copies: picked by index they lie by columns, and NumPy
    would add a row's terms one by one instead of pairwise."""
    X = np.asarray(X, float).reshape(-1, N_VARS)
    X1, X2, u = np.ascontiguousarray(X[:, J1]), np.ascontiguousarray(X[:, J2]), X[:, :1]
    if i <= 4:
        g1 = np.sum((X1 - math_sin(0.5 * math.pi * u)) ** 2, axis=1)
        g2 = np.sum((X2 - math_cos(0.5 * math.pi * u)) ** 2, axis=1)
    else:
        g1 = np.sum((X1 - np.sin(0.5 * (J1 + 1) / 30.0 * math.pi * u)) ** 2, axis=1)
        g2 = np.sum((X2 - np.cos(0.5 * (J2 + 1) / 30.0 * math.pi * u)) ** 2, axis=1)
    return g1, g2


def objectives(i: int, X) -> np.ndarray:
    """The objectives of every row of X: n x M."""
    X = np.asarray(X, float).reshape(-1, N_VARS)
    if i >= 13:
        g1 = np.sum(10.0 * (X[:, 2:] - 0.5) ** 2, axis=1)
        c1, s1 = math_cos(0.5 * math.pi * X[:, 0]), math_sin(0.5 * math.pi * X[:, 0])
        rad = SHIFT + 1.0 + g1                 # 1.7057 + g1
        return np.column_stack([rad * c1 * math_cos(0.5 * math.pi * X[:, 1]),
                                rad * c1 * math_sin(0.5 * math.pi * X[:, 1]), rad * s1])
    g1, g2 = g_values(i, X)
    h = shape(i, X[:, 0])
    if i <= 4:
        return np.column_stack([X[:, 0] + g1, h + g2])
    if i <= 8:
        return np.column_stack([X[:, 0] + 10.0 * g1 + SHIFT, h + 10.0 * g2 + SHIFT])
    return np.column_stack([(1.0 + SHIFT) * X[:, 0] * (10.0 * g1 + 1.0),
                            (1.0 + SHIFT) * h * (10.0 * g2 + 1.0)])


def _ellipse(F1, F2, i):
    p, q, a, b = ELLIPSES[i]
    ct, st = math.cos(THETA), math.sin(THETA)
    out = []
    for pk, qk, ak, bk in zip(p, q, a, b):
        d1, d2 = F1 - pk, F2 - qk
        out.append((d1 * ct - d2 * st) ** 2 / ak ** 2 + (d1 * st + d2 * ct) ** 2 / bk ** 2 - R)
    return out


def constraints_c(i: int, F, x1=None, g=None):
    """c >= 0 satisfied, rows of F (and of x1 and the pair g = (g1, g2) for 1-4)."""
    F = np.atleast_2d(np.asarray(F, float))
    if i <= 4:
        g1, g2 = (np.atleast_1d(v) for v in g)
        cols = [(A - g1) * (g1 - B), (A - g2) * (g2 - B)]
        if i >= 3:
            cols.append(np.sin(C * math.pi * np.atleast_1d(x1)) - 0.5)
        return np.column_stack(cols)
    if i >= 13:
        gg = np.sum(F * F, axis=1)
        return np.column_stack([(gg - u) * (gg - v) for u, v in SPHERES[i]])
    cols = _ellipse(F[:, 0], F[:, 1], i)
    if i >= 9:
        ca, sa = math.cos(ALPHA), math.sin(ALPHA)
        cols.append(F[:, 0] * sa + F[:, 1] * ca
                    - np.sin(4.0 * math.pi * (F[:, 0] * ca - F[:, 1] * sa)) - SINE_E[i])
    return np.column_stack(cols)


def evaluate(i: int, x) -> tuple:
    """(objectives, the library's constraints: <= 0 satisfied)."""
    F, G = evaluate_batch(i, [x])
    return F[0].tolist(), G[0].tolist()


def evaluate_batch(i: int, X) -> tuple:
    """(objectives, constraints) of every row of X, one computation for both."""
    X = np.asarray(X, float).reshape(-1, N_VARS)
    F = objectives(i, X)
    return F, -constraints_c(i, F, X[:, 0], g_values(i, X) if i <= 4 else None)


def name(i: int) -> str:
    return f"LIR-CMOP{i}"


def _nd_2d(F):
    order = np.lexsort((F[:, 1], F[:, 0]))
    keep, best = [], math.inf
    for k in order:
        if F[k, 1] < best:
            keep.append(k)
            best = F[k, 1]
    return np.asarray(keep, int)


@lru_cache(maxsize=None)
def front(i: int, n_points: int = 1000) -> np.ndarray:
    """The constrained front (THE FRONTS above), thinned to about n_points."""
    from ..run.archive import dss_order
    if i >= 13:
        rad = SHIFT + 1.0 if i == 13 else math.sqrt(3.0625)
        k = int(math.ceil(math.sqrt(2.0 * n_points)))
        a, b = np.meshgrid(np.linspace(0.0, 1.0, k), np.linspace(0.0, 1.0, k), indexing="ij")
        a, b = a.ravel(), b.ravel()
        F = rad * np.column_stack([np.cos(0.5 * math.pi * a) * np.cos(0.5 * math.pi * b),
                                   np.cos(0.5 * math.pi * a) * np.sin(0.5 * math.pi * b),
                                   np.sin(0.5 * math.pi * a)])
        F = np.unique(np.round(F, 12), axis=0)
    elif i <= 4:
        u = np.linspace(0.0, 1.0, 20001)
        if i >= 3:
            u = u[np.sin(C * math.pi * u) - 0.5 >= 0.0]
        F = np.column_stack([u + B, shape(i, u) + B])
        F = F[_nd_2d(F)]
    else:
        F = _scan_front(i)
        F = F[_nd_2d(F)]
    if len(F) <= n_points:
        return F
    return F[dss_order(F, n_points, ideal=F.min(axis=0), nadir=F.max(axis=0))]


def _least_f2(i: int, f1: np.ndarray) -> np.ndarray:
    """The least attainable f2 at f1 (no constraint): h at the largest u the f1
    allows, h decreasing."""
    if i <= 8:
        u = np.clip(f1 - SHIFT, 0.0, 1.0)
        return shape(i, u) + SHIFT
    u = np.clip(f1 / (1.0 + SHIFT), 0.0, 1.0)
    return (1.0 + SHIFT) * shape(i, u)


def _scan_front(i: int) -> np.ndarray:
    """5-12: the least feasible attainable f2 (_least_feasible) at 6001 values
    of f1, and on a grid 40 times finer between two neighbours of the front
    one step of f1 apart and more than 3 median steps apart along it (a steep
    part: 1 - sqrt(u) near u = 0)."""
    lo1 = SHIFT if i <= 8 else 0.0
    f1 = np.linspace(lo1, lo1 + 3.0, 6001)
    coarse = _least_feasible(i, f1)
    F = coarse[_nd_2d(coarse)]
    along = np.linalg.norm(np.diff(F, axis=0), axis=1)
    steep = (np.diff(F[:, 0]) < 1.5 * (f1[1] - f1[0])) & (along > 3.0 * np.median(along))
    if not steep.any():
        return coarse
    fine = np.concatenate([np.linspace(a, b, 41)[1:-1]
                           for a, b in zip(F[:-1, 0][steep], F[1:, 0][steep])])
    return np.vstack([coarse, _least_feasible(i, fine)])


def _least_feasible(i: int, f1: np.ndarray) -> np.ndarray:
    """Rows (f1, the least feasible attainable f2 there), by a scan up from
    the least attainable one and a bisection below the first feasible step;
    an f1 with nothing feasible up to 4 above the floor is left out."""
    base = _least_f2(i, f1)
    step = 1e-3
    keep, lo, hi = [], [], []
    for a, b0 in zip(f1, base):
        f2 = b0 + step * np.arange(4001)                 # up to 4 above the floor
        ok = np.all(constraints_c(i, np.column_stack([np.full_like(f2, a), f2])) >= 0.0, axis=1)
        if not ok.any():
            continue
        k = int(np.argmax(ok))
        keep.append(a)
        hi.append(f2[k])
        lo.append(f2[k - 1] if k > 0 else f2[k])         # the floor itself: nothing below
    f1, lo, hi = (np.asarray(v, float) for v in (keep, lo, hi))
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        good = np.all(constraints_c(i, np.column_stack([f1, mid])) >= 0.0, axis=1)
        hi = np.where(good, mid, hi)
        lo = np.where(good, lo, mid)
    return np.column_stack([f1, hi])
