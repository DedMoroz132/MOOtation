# SPDX-License-Identifier: Apache-2.0
"""MW1-14: the constrained problems of Ma and Wang.

Z. Ma, Y. Wang, "Evolutionary constrained multiobjective optimization: test
suite construction and performance comparisons", IEEE TEVC 23(6):972-986,
2019, doi:10.1109/TEVC.2019.2896967 (task 6 of 2026-10-09, item 5), Eqs.
12-28, read on the PDF's pages 6-8. Values checked against pymoo 0.6.2's
mw.py (Apache-2.0, "based on the C++ implementation by Ma and Wang") at 85
points, reference in tests/data/pymoo-0.6.2_mw.json. n = 15 variables, as in
the paper's experiments (Section IV-C); MW4, MW8 and MW14 at m = 3 and 5
objectives with n = m + 12 (Section IV-H: n = m - 1 + 13). Boxes: [0, 1]^n, MW6 [0, 1.1]^n, MW11 [0, sqrt 2]^n, MW13 and
MW14 [0, 1.5]^n (Table II).

The distance functions, i = m..n counted from 1 (Eqs. 12-14):
  g1 = 1 + sum (1 - exp(-10 (z_i - 0.5 - (i-1)/(2n))^2)), z_i = x_i^(n-m)
  g2 = 1 + sum (1.5 + 0.1/n z_i^2 - 1.5 cos(2 pi z_i)), z_i = 1 - exp(-10 (x_i - (i-1)/n)^2)
  g3 = 1 + sum 2 (x_i + (x_{i-1} - 0.5)^2 - 1)^2
"sin(a)^8" in the paper is (sin a)^8 (pymoo's LA1-LA3).

THE SIGN. The paper writes each constraint as c >= 0 or c <= 0; the
library's convention is a value <= 0 when satisfied: -c for the first kind,
c for the second, in the paper's order.

READINGS:
  MW-1  MW11's distance function is g3 in Table II and in the authors' code
        (pymoo), g2 in the text under Eq. 25: g3.
  MW-2  MW4's constraint, (1 + 0.4 sin(2.5 pi l)^8) - f_1 - ... - f_m, has the
        power inside the bracket on the PDF's page 7 (the markdown copy put it
        outside), as in pymoo.
  MW-3  The square roots of MW5, MW6, MW7 and MW11 are taken of max(0, .):
        at x1 on its upper bound, (f1/g)^2 can exceed the constant by one
        rounding (MW11: (g sqrt 2 / g)^2 = 2.0000000000000004), and the root
        would be NaN there — where a clipping repair puts points.

THE FRONTS. The objective vectors a problem can reach are those of x1 (the
positions) and of every g from 1 up: MW1-3's front moved up by g - 1 (f2 =
g - 0.85 f1, g - f1), MW4-8's scaled by g and MW14's last objective, MW9-13's f =
(g t, g s(t)), t = x1 (MW10: x1^n) — the region on and beyond the
unconstrained front, the front at g = 1. The constraints act in objective
space. The front, the non-dominated part of the feasible reachable set, is
built here from that geometry (no file):
  MW2, MW4, MW14   Type I: all of the unconstrained front — the segment, the
                   simplex, and MW14's non-dominated part of f_m = mean
                   h(f_i), h(t) = 6 - e^t - 1.5 sin(1.1 pi t^2): the product
                   of h's strict left records, as DTLZ7's ([0, 0.67...] and
                   [1.40..., 1.5])
  MW6, MW8         the unconstrained front where feasible: along every ray the
                   constraint bounds the radius from above (MW6: an ellipse
                   whose axes depend on the angle; MW8: 1.25 - 0.5 sin(6 l)^2,
                   l the elevation), so a direction whose front point is
                   infeasible has no feasible reachable point
  MW5, MW7         along every ray the constraints bound the radius from below
                   and above: the curve of least feasible radii r*(theta) (MW5:
                   1 + max(0.5 s, -0.45 s), s = sin(6 l2^3), whose zeros give
                   the paper's "discrete" points, put in exactly; MW7: max(1,
                   1.15 - 0.2 sin(4 theta)^8)), its non-dominated part
  MW1, MW3, MW9-13 for every f1 the least f2 that is reachable and feasible:
                   from the floor (MW1, MW3, MW9-11: the shape at g = 1, 0 past
                   x1's bound; MW12, MW13, whose g s(f1/g) need not grow with g:
                   the least over g, a grid and golden-section search) and the
                   roots of the constraints' factors, linear in f2 (MW9-11,
                   MW13) or decreasing (MW12's, by bisection), exactly; MW1 and
                   MW3 by a scan of step 2e-4 and bisection. Then the
                   non-dominated points; MW11's isolated point (1, 1) put in.
The two-objective fronts are computed once per process (dense_front: about a
second, MW1 and MW3 2-4 s) at 20 001 values of f1 or the angle (MW1, MW3:
5 001), refined by halving where neighbours lie apart (steep parts, jumps),
where a piece ends and where the front leaves the floor — those to 1e-12 —
and spread along their length to the size asked for, every piece's ends
kept. MW1's front is all at g = 1 (Type II, as pymoo's): past every point
where the unconstrained front turns infeasible the least feasible f2 lies
above the last feasible one, dominated.
Pareto sets where the front lies at g = 1 (MW1, MW2, MW4, MW6, MW8, MW14):
the positions of the front's points and the distance variables at g's
optimum (g1: x_i = (0.5 + (i-1)/(2n))^(1/(n-m)); g2: x_i = (i-1)/n; g3: x_i =
1 - (x_{i-1} - 0.5)^2, a chain from the last position variable — variable
linkage). Elsewhere the front needs g > 1: no sampler.
"""

from __future__ import annotations

import functools
import math
from typing import List

import numpy as np

N_VARS = 15
SCALABLE = ("MW4", "MW8", "MW14")
SIZES = (3, 5)
UPPER = {"MW6": 1.1, "MW11": math.sqrt(2.0), "MW13": 1.5, "MW14": 1.5}
DISTANCE = {"MW1": "g1", "MW2": "g2", "MW3": "g3", "MW4": "g1", "MW5": "g1", "MW6": "g2",
            "MW7": "g3", "MW8": "g2", "MW9": "g1", "MW10": "g2", "MW11": "g3", "MW12": "g1",
            "MW13": "g2", "MW14": "g3"}
ON_G1 = ("MW1", "MW2", "MW4", "MW6", "MW8", "MW14")        # fronts at g = 1: a Pareto set
NAMES = tuple(f"MW{i}" for i in range(1, 15))


def n_vars(name: str, m: int) -> int:
    return m + 12 if name in SCALABLE else N_VARS


def upper(name: str) -> float:
    return UPPER.get(name, 1.0)


# ── distance functions (Eqs. 12-14) ──────────────────────────────────────────
def g1(x, m: int) -> float:
    x = np.asarray(x, float)
    n = len(x)
    i = np.arange(m, n + 1)
    z = x[m - 1:] ** (n - m)
    return float(1.0 + np.sum(1.0 - np.exp(-10.0 * (z - 0.5 - (i - 1) / (2.0 * n)) ** 2)))


def g2(x, m: int) -> float:
    x = np.asarray(x, float)
    n = len(x)
    i = np.arange(m, n + 1)
    z = 1.0 - np.exp(-10.0 * (x[m - 1:] - (i - 1) / n) ** 2)
    return float(1.0 + np.sum(1.5 + (0.1 / n) * z * z - 1.5 * np.cos(2.0 * np.pi * z)))


def g3(x, m: int) -> float:
    x = np.asarray(x, float)
    return float(1.0 + np.sum(2.0 * (x[m - 1:] + (x[m - 2:-1] - 0.5) ** 2 - 1.0) ** 2))


G = {"g1": g1, "g2": g2, "g3": g3}


def optimal_distance(name: str, positions) -> np.ndarray:
    """The distance variables at g's optimum after these positions (Pareto sets above)."""
    p = np.asarray(positions, float)
    m = len(p) + 1
    n = n_vars(name, m)
    i = np.arange(m, n + 1)
    kind = DISTANCE[name]
    if kind == "g1":
        return (0.5 + (i - 1) / (2.0 * n)) ** (1.0 / (n - m))
    if kind == "g2":
        return (i - 1) / float(n)
    out, last = [], p[-1]
    for _ in i:                                   # g3: x_i = 1 - (x_{i-1} - 0.5)^2
        last = 1.0 - (last - 0.5) ** 2
        out.append(last)
    return np.array(out)


# ── objectives ───────────────────────────────────────────────────────────────
def _root(v: float) -> float:
    return math.sqrt(max(0.0, v))                 # MW-3


def objectives(name: str, x, m: int = 2) -> List[float]:
    x = np.asarray(x, float)
    g = G[DISTANCE[name]](x, m)
    x1 = x[0]
    if name == "MW1":
        return [x1, g * (1.0 - 0.85 * x1 / g)]
    if name in ("MW2", "MW3"):
        return [x1, g * (1.0 - x1 / g)]
    if name == "MW4":
        f = np.full(m, g)
        f[1:] *= x[m - 2::-1]
        f[:-1] *= np.cumprod(1.0 - x[:m - 1])[::-1]
        return f.tolist()
    if name == "MW8":
        f = np.full(m, g)
        f[1:] *= np.sin(0.5 * np.pi * x[m - 2::-1])
        f[:-1] *= np.cumprod(np.cos(0.5 * np.pi * x[:m - 1]))[::-1]
        return f.tolist()
    if name == "MW14":
        f = list(x[:m - 1])
        h = 6.0 - np.exp(x[:m - 1]) - 1.5 * np.sin(1.1 * np.pi * x[:m - 1] ** 2)
        return f + [g / (m - 1) * float(np.sum(h))]
    f1 = g * x1 ** N_VARS if name == "MW10" else g * x1
    t = f1 / g
    if name in ("MW5", "MW7"):
        return [f1, g * _root(1.0 - t ** 2)]
    if name == "MW6":
        return [f1, g * _root(1.1 * 1.1 - t ** 2)]
    if name == "MW9":
        return [f1, g * (1.0 - t ** 0.6)]
    if name == "MW10":
        return [f1, g * (1.0 - t ** 2)]
    if name == "MW11":
        return [f1, g * _root(2.0 - t ** 2)]
    if name == "MW12":
        return [f1, g * (0.85 - 0.8 * t - 0.08 * abs(math.sin(3.2 * math.pi * t)))]
    if name == "MW13":
        return [f1, g * (5.0 - math.exp(t) - 0.5 * abs(math.sin(3.0 * math.pi * t)))]
    raise KeyError(name)


# ── constraints, in objective space, the library's sign ──────────────────────
def _atan(f2, f1):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.arctan(np.asarray(f2, float) / np.asarray(f1, float))


def constraints_f(name: str, F) -> np.ndarray:
    """(rows, constraints) for objective vectors F: <= 0 satisfied (THE SIGN)."""
    F = np.atleast_2d(np.asarray(F, float))
    f1, f2 = F[:, 0], F[:, 1]
    sq = np.sum(F * F, axis=1)
    s2 = np.sqrt(2.0)
    if name in ("MW1", "MW2"):
        l = s2 * f2 - s2 * f1
        b = 2.0 if name == "MW1" else 3.0
        cols = [-(1.0 - f1 - f2 + 0.5 * np.sin(b * np.pi * l) ** 8)]
    elif name == "MW3":
        l = s2 * f2 - s2 * f1
        cols = [-(1.05 - f1 - f2 + 0.45 * np.sin(0.75 * np.pi * l) ** 6),
                0.85 - f1 - f2 + 0.3 * np.sin(0.75 * np.pi * l) ** 2]
    elif name == "MW4":
        l = F[:, -1] - np.sum(F[:, :-1], axis=1)
        cols = [-((1.0 + 0.4 * np.sin(2.5 * np.pi * l) ** 8) - np.sum(F, axis=1))]
    elif name == "MW5":
        l1 = _atan(f2, f1)
        l2 = 0.5 * np.pi - 2.0 * np.abs(l1 - 0.25 * np.pi)
        cols = [-((1.7 - 0.2 * np.sin(2.0 * l1)) ** 2 - sq),
                (1.0 + 0.5 * np.sin(6.0 * l2 ** 3)) ** 2 - sq,
                (1.0 - 0.45 * np.sin(6.0 * l2 ** 3)) ** 2 - sq]
    elif name == "MW6":
        l = np.cos(6.0 * _atan(f2, f1) ** 4) ** 10
        cols = [-(1.0 - (f1 / (1.0 + 0.15 * l)) ** 2 - (f2 / (1.0 + 0.75 * l)) ** 2)]
    elif name == "MW7":
        l = _atan(f2, f1)
        cols = [-((1.2 + 0.4 * np.sin(4.0 * l) ** 16) ** 2 - sq),
                (1.15 - 0.2 * np.sin(4.0 * l) ** 8) ** 2 - sq]
    elif name == "MW8":
        l = np.arcsin(F[:, -1] / np.sqrt(sq))
        cols = [-((1.25 - 0.5 * np.sin(6.0 * l) ** 2) ** 2 - sq)]
    elif name == "MW9":
        t1 = (1.0 - 0.64 * f1 * f1 - f2) * (1.0 - 0.36 * f1 * f1 - f2)
        t2 = 1.35 ** 2 - (f1 + 0.35) ** 2 - f2
        t3 = 1.15 ** 2 - (f1 + 0.15) ** 2 - f2
        cols = [np.minimum(t1, t2 * t3)]
    elif name == "MW10":
        cols = [-((2.0 - 4.0 * f1 ** 2 - f2) * (2.0 - 8.0 * f1 ** 2 - f2)),
                (2.0 - 2.0 * f1 ** 2 - f2) * (2.0 - 16.0 * f1 ** 2 - f2),
                (1.0 - f1 ** 2 - f2) * (1.2 - 1.2 * f1 ** 2 - f2)]
    elif name == "MW11":
        cols = [-((3.0 - f1 ** 2 - f2) * (3.0 - 2.0 * f1 ** 2 - f2)),
                (3.0 - 0.625 * f1 ** 2 - f2) * (3.0 - 7.0 * f1 ** 2 - f2),
                -((1.62 - 0.18 * f1 ** 2 - f2) * (1.125 - 0.125 * f1 ** 2 - f2)),
                (2.07 - 0.23 * f1 ** 2 - f2) * (0.63 - 0.07 * f1 ** 2 - f2)]
    elif name == "MW12":
        t1 = 1.0 - 0.8 * f1 - f2 + 0.08 * np.sin(2.0 * np.pi * (f2 - f1 / 1.5))
        t2 = 1.0 - 0.625 * f1 - f2 + 0.08 * np.sin(2.0 * np.pi * (f2 - f1 / 1.6))
        t3 = 1.4 - 0.875 * f1 - f2 + 0.08 * np.sin(2.0 * np.pi * (f2 / 1.4 - f1 / 1.6))
        t4 = 1.8 - 1.125 * f1 - f2 + 0.08 * np.sin(2.0 * np.pi * (f2 / 1.8 - f1 / 1.6))
        cols = [t1 * t4, -(t2 * t3)]
    elif name == "MW13":
        w = 0.5 * np.sin(3.0 * np.pi * f1)
        t1 = 5.0 - np.exp(f1) - w - f2
        t2 = 5.0 - (1.0 + f1 + 0.5 * f1 * f1) - w - f2
        t3 = 5.0 - (1.0 + 0.7 * f1) - w - f2
        t4 = 5.0 - (1.0 + 0.4 * f1) - w - f2
        cols = [t1 * t4, -(t2 * t3)]
    elif name == "MW14":
        P = F[:, :-1]
        alpha = 1.0 + P + 0.5 * P * P + 1.5 * np.sin(1.1 * np.pi * P * P)
        cols = [-(np.mean(6.1 - alpha, axis=1) - F[:, -1])]
    else:
        raise KeyError(name)
    return np.column_stack(cols)


def constraints(name: str, x, m: int = 2) -> List[float]:
    return constraints_f(name, objectives(name, x, m))[0].tolist()


# ── the fronts ───────────────────────────────────────────────────────────────
RESOLUTION = 20_001        # values of f1 (or of the angle) of a two-objective front
SCAN_RESOLUTION = 5_001   # MW1 and MW3, scanned in f2 (_scan); refined like the rest
# the f1 the grid spans: x1's range (MW1, MW3), elsewhere past the front's end,
# with f1 = 1 and 1.5 on the grid
F1_RANGE = {"MW1": 1.0, "MW3": 1.0, "MW9": 2.0, "MW10": 2.0, "MW11": 2.5, "MW12": 2.0,
            "MW13": 2.5}
RADIAL = ("MW5", "MW6", "MW7")
SCANNED = ("MW1", "MW3")


def _nondominated(F: np.ndarray) -> np.ndarray:
    """The non-dominated rows of a two-column F, sorted by f1."""
    F = F[np.lexsort((F[:, 1], F[:, 0]))]
    keep, best = [], np.inf
    for k, f2 in enumerate(F[:, 1]):
        if f2 < best:
            keep.append(k)
            best = f2
    return F[keep]


def _feasible(name: str, F, tol: float = 0.0) -> np.ndarray:
    return np.all(constraints_f(name, F) <= tol, axis=1)


def _shape(name: str, t):
    """The shape of the two-objective problems of THE FRONTS' last group:
    MW9-13's f2 = g s(t), t = f1 / g; MW1's and MW3's front at g = 1."""
    t = np.asarray(t, float)
    if name == "MW1":
        return 1.0 - 0.85 * t
    if name == "MW3":
        return 1.0 - t
    if name == "MW9":
        return 1.0 - t ** 0.6
    if name == "MW10":
        return 1.0 - t ** 2
    if name == "MW11":
        return np.sqrt(np.maximum(0.0, 2.0 - t ** 2))
    if name == "MW12":
        return 0.85 - 0.8 * t - 0.08 * np.abs(np.sin(3.2 * np.pi * t))
    return 5.0 - np.exp(t) - 0.5 * np.abs(np.sin(3.0 * np.pi * t))          # MW13


def _least_over_g(name: str, f1: np.ndarray) -> np.ndarray:
    """min of g s(f1/g) over g >= max(1, f1/top), top x1's bound: on a grid of g
    (fine up to three times the least, coarse to 70, beyond every g's largest
    value), then 60 golden-section steps round the grid's best."""
    top = upper(name)
    grid = np.concatenate([np.linspace(1.0, 3.0, 2001), np.geomspace(3.0, 70.0, 200)[1:]])
    out = np.empty(len(f1))
    golden = (math.sqrt(5.0) - 1.0) / 2.0
    for a in range(0, len(f1), 500):
        F1 = f1[a:a + 500]
        G = np.maximum(1.0, F1 / top)[:, None] * grid[None, :]
        V = G * _shape(name, F1[:, None] / G)
        k = np.argmin(V, axis=1)
        rows = np.arange(len(F1))
        lo = G[rows, np.maximum(k - 1, 0)]
        hi = G[rows, np.minimum(k + 1, len(grid) - 1)]
        for _ in range(60):
            c, d = hi - golden * (hi - lo), lo + golden * (hi - lo)
            left = c * _shape(name, F1 / c) < d * _shape(name, F1 / d)
            hi, lo = np.where(left, d, hi), np.where(left, lo, c)
        mid = 0.5 * (lo + hi)
        out[a:a + 500] = np.minimum(V[rows, k], mid * _shape(name, F1 / mid))
    return out


def floor(name: str, f1) -> np.ndarray:
    """The least reachable f2 at f1 (THE FRONTS)."""
    f1 = np.asarray(f1, float)
    if name in SCANNED:                            # f1 = x1, f2 = g s(x1): g = 1
        return _shape(name, f1)
    if name in ("MW9", "MW10", "MW11"):            # g s(f1/g) grows with g; s(top) = 0
        top = upper(name)
        return np.where(f1 <= top, _shape(name, np.minimum(f1, top)), 0.0)
    return _least_over_g(name, f1)                 # MW12, MW13


def _decreasing_root(fn, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """The root of fn, decreasing in its argument, in [lo, hi] (60 bisections)."""
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        pos = fn(mid) > 0.0
        lo, hi = np.where(pos, mid, lo), np.where(pos, hi, mid)
    return 0.5 * (lo + hi)


def _roots(name: str, f1: np.ndarray) -> list:
    """Where each factor of the constraints is zero, as f2 at f1: they are
    linear in f2, but for MW12's, whose sine moves f2 by 0.08 at most and
    keeps them decreasing (slope -1 +- 0.08 2 pi / 1.4 at worst)."""
    q = f1 * f1
    if name == "MW9":
        return [1 - 0.64 * q, 1 - 0.36 * q, 1.35 ** 2 - (f1 + 0.35) ** 2,
                1.15 ** 2 - (f1 + 0.15) ** 2]
    if name == "MW10":
        return [2 - 4 * q, 2 - 8 * q, 2 - 2 * q, 2 - 16 * q, 1 - q, 1.2 - 1.2 * q]
    if name == "MW11":
        return [3 - q, 3 - 2 * q, 3 - 0.625 * q, 3 - 7 * q, 1.62 - 0.18 * q, 1.125 - 0.125 * q,
                2.07 - 0.23 * q, 0.63 - 0.07 * q]
    if name == "MW13":
        w = 0.5 * np.sin(3.0 * np.pi * f1)
        return [5 - np.exp(f1) - w, 5 - (1 + f1 + 0.5 * q) - w, 5 - (1 + 0.7 * f1) - w,
                5 - (1 + 0.4 * f1) - w]
    out = []                                       # MW12
    for a, b, c, d in ((1.0, 0.8, 1.0, 1.5), (1.0, 0.625, 1.0, 1.6), (1.4, 0.875, 1.4, 1.6),
                       (1.8, 1.125, 1.8, 1.6)):
        line = a - b * f1
        out.append(_decreasing_root(
            lambda f2, _l=line: _l - f2 + 0.08 * np.sin(2.0 * np.pi * (f2 / c - f1 / d)),
            line - 0.1, line + 0.1))
    return out


def _least_feasible(name: str, f1: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    """Per row the least candidate f2 at or above the floor that is feasible
    (to 1e-12); NaN where none is."""
    F1 = np.repeat(f1, candidates.shape[1])
    with np.errstate(invalid="ignore", over="ignore"):
        ok = _feasible(name, np.column_stack([F1, candidates.ravel()]), 1e-12)
    C = np.where(ok.reshape(candidates.shape), candidates, np.inf)
    best = C.min(axis=1)
    return np.where(np.isfinite(best), best, np.nan)


def _scan(name: str, f1: np.ndarray, step: float = 2e-4) -> np.ndarray:
    """MW1 and MW3: the least feasible f2 from the floor up to 1.5 - f1, where
    their first constraint ends every feasible point (f1 + f2 <= 1.5), by a
    scan of `step` and 60 bisections from the first feasible step down."""
    lo = floor(name, f1)
    k = np.arange(int(0.5 / step) + 2)
    out = np.full(len(f1), np.nan)
    for a in range(0, len(f1), 200):
        F1, L = f1[a:a + 200], lo[a:a + 200]
        f2 = L[:, None] + step * k[None, :]
        ok = _feasible(name, np.column_stack([np.repeat(F1, len(k)), f2.ravel()]))
        ok = ok.reshape(f2.shape) & (F1[:, None] + f2 <= 1.5 + step)
        first = np.argmax(ok, axis=1)
        rows = np.flatnonzero(ok[np.arange(len(F1)), first])
        j = first[rows]
        at_floor = j == 0
        b = np.where(at_floor, L[rows], f2[rows, np.maximum(j - 1, 0)])    # infeasible
        c = f2[rows, j]                                                    # feasible
        for _ in range(60):
            mid = 0.5 * (b + c)
            good = _feasible(name, np.column_stack([F1[rows], mid]))
            c, b = np.where(good, mid, c), np.where(good, b, mid)
        out[a + rows] = np.where(at_floor, L[rows], c)
    return out


def _radius(name: str, th: np.ndarray) -> np.ndarray:
    """MW5-7: the least feasible radius along the ray at angle th, reachable
    radii being g (MW5, MW7) or 1.1 g (MW6), g >= 1."""
    if name == "MW5":
        l2 = 0.5 * np.pi - 2.0 * np.abs(th - 0.25 * np.pi)
        s = np.sin(6.0 * l2 ** 3)
        s[np.abs(s) < 1e-12] = 0.0                 # its zeros: r = 1 exactly
        return 1.0 + np.maximum(0.5 * s, -0.45 * s)
    if name == "MW7":
        return np.maximum(1.0, 1.15 - 0.2 * np.sin(4.0 * th) ** 8)
    return np.full(len(th), 1.1)                   # MW6: then the ellipse decides


def _curve(name: str, u: np.ndarray) -> tuple:
    """The least feasible reachable point at parameter u — the angle (MW5-7)
    or f1 — as rows (f1, f2), f2 NaN where there is none; and whether it is
    the least reachable one (on the floor, or at the least radius g = 1)."""
    if name in RADIAL:
        r = _radius(name, u)
        F = np.column_stack([r * np.cos(u), r * np.sin(u)])
        F[np.abs(F) < 1e-15] = 0.0
        F[~_feasible(name, F, 1e-12), 1] = np.nan
        return F, r == (1.1 if name == "MW6" else 1.0)
    lo = floor(name, u)
    if name in SCANNED:
        f2 = _scan(name, u)
    else:
        C = np.column_stack([lo] + _roots(name, u))
        f2 = _least_feasible(name, u, np.where(C >= lo[:, None], C, np.inf))
    return np.column_stack([u, f2]), f2 == lo


def _refined(name: str, u: np.ndarray, rounds: int = 45) -> np.ndarray:
    """_curve at u and at midpoints added where neighbours lie farther apart
    than twice the grid's median step (a steep part: points added until it
    is as fine as the rest; a jump: down to 1e-12 in u), where one has a
    point and the other none (an end of a piece), and where one is the least
    reachable point and the other not (where the front leaves the
    unconstrained one or the floor: MW1's exits, which make every point
    beyond them dominated); both found to 1e-12."""
    F, low = _curve(name, u)
    ok = np.isfinite(F[:, 1])
    step = np.linalg.norm(np.diff(F, axis=0), axis=1)
    target = 2.0 * np.median(step[ok[:-1] & ok[1:]])
    for _ in range(rounds):
        ok = np.isfinite(F[:, 1])
        step = np.linalg.norm(np.diff(F, axis=0), axis=1)
        both = ok[:-1] & ok[1:]
        wide = ((both & (step > target)) | (ok[:-1] ^ ok[1:]) | (both & (low[:-1] ^ low[1:]))) \
            & (np.diff(u) > 1e-12)
        if not wide.any():
            break
        mid = 0.5 * (u[:-1][wide] + u[1:][wide])
        Fm, lm = _curve(name, mid)
        u, F, low = np.concatenate([u, mid]), np.vstack([F, Fm]), np.concatenate([low, lm])
        order = np.argsort(u, kind="stable")
        u, F, low = u[order], F[order], low[order]
    return F[np.isfinite(F[:, 1])]


@functools.lru_cache(maxsize=None)
def dense_front(name: str) -> np.ndarray:
    """The two-objective front, sorted by f1 (THE FRONTS): the least feasible
    reachable points at RESOLUTION values of f1 or the angle, refined
    (_refined), the dominated ones dropped; once per process."""
    if name == "MW2":
        t = np.linspace(0.0, 1.0, RESOLUTION)
        return np.column_stack([t, 1.0 - t])
    if name in RADIAL:
        u = np.linspace(0.0, 0.5 * np.pi, RESOLUTION)
        if name == "MW5":                          # the zeros of s: the discrete points
            l2 = (np.arange(8) * np.pi / 6.0) ** (1.0 / 3.0)
            u = np.unique(np.concatenate([u, 0.25 * np.pi + (0.5 * np.pi - l2) / 2.0,
                                          0.25 * np.pi - (0.5 * np.pi - l2) / 2.0]))
    else:
        u = np.linspace(0.0, F1_RANGE[name], SCAN_RESOLUTION if name in SCANNED else RESOLUTION)
    F = _refined(name, u)
    if name == "MW11":
        F = np.vstack([F, [[1.0, 1.0]]])           # the isolated point, exactly
    return _nondominated(F)


def _spread(F: np.ndarray, n: int) -> np.ndarray:
    """About n rows of the sorted two-objective front F, evenly by length
    along it; the ends of every piece kept (a step of more than 20 times the
    median separates two pieces, and counts nothing)."""
    if len(F) <= n:
        return F.copy()
    step = np.linalg.norm(np.diff(F, axis=0), axis=1)
    gap = step > 20.0 * np.median(step)
    s = np.concatenate([[0.0], np.cumsum(np.where(gap, 0.0, step))])
    ends = np.flatnonzero(np.concatenate([[True], gap]) | np.concatenate([gap, [True]]))
    pick = np.searchsorted(s, np.linspace(0.0, s[-1], max(n - len(ends), 2)))
    return F[np.unique(np.concatenate([ends, np.minimum(pick, len(F) - 1)]))]


def _mw14_records(grid: int = 200_001) -> np.ndarray:
    """The strict left records of h(t) = 6 - e^t - 1.5 sin(1.1 pi t^2) on
    [0, 1.5], on a grid: the values MW14's position objectives take on its
    front, [0, 0.67...] and [1.40..., 1.5]."""
    t = np.linspace(0.0, 1.5, grid)
    h = 6.0 - np.exp(t) - 1.5 * np.sin(1.1 * np.pi * t * t)
    prior = np.minimum.accumulate(np.concatenate([[np.inf], h[:-1]]))
    return t[h < prior]


def _mw14_front(m: int, n: int) -> np.ndarray:
    rec = _mw14_records()
    rng = np.random.default_rng(20261010 + m)
    corners = np.array(np.meshgrid(*[[rec[0], rec[-1]]] * (m - 1))).reshape(m - 1, -1).T
    P = np.vstack([corners, rec[rng.integers(0, len(rec), (n, m - 1))]])
    h = 6.0 - np.exp(P) - 1.5 * np.sin(1.1 * np.pi * P * P)
    return np.column_stack([P, h.mean(axis=1)])


def front(name: str, m: int, n: int, unit_simplex, unit_sphere) -> np.ndarray:
    """About n points of the front (THE FRONTS); `unit_simplex(M, k)` and
    `unit_sphere(M, k)` sample the simplex sum f = 1 and the sphere's octant."""
    if name == "MW4":
        return unit_simplex(m, n)
    if name == "MW8":
        F = unit_sphere(m, 2 * n)
        return F[_feasible(name, F, 1e-12)]
    if name == "MW14":
        return _mw14_front(m, n)
    return _spread(dense_front(name), n)


def pareto_set(name: str, m: int, n: int, unit_simplex=None,
               unit_sphere=None) -> np.ndarray | None:
    """About n points of the Pareto set where the front lies at g = 1 (ON_G1):
    the front's positions and the distance variables at g's optimum."""
    if name not in ON_G1:
        return None
    if name in ("MW1", "MW2", "MW6"):
        F = front(name, m, n, unit_simplex, unit_sphere)
        P = F[:, :1]                               # x1 = f1 at g = 1
    elif name == "MW14":
        P = _mw14_front(m, n)[:, :m - 1]
    elif name == "MW4":                            # f_m = x1, f_k = x_{m-k+1} prod (1 - x_i)
        F = unit_simplex(m, n)
        P = np.empty((len(F), m - 1))
        rest = np.ones(len(F))
        for j in range(m - 1):                     # x_{j+1} from f_{m-j}
            with np.errstate(divide="ignore", invalid="ignore"):
                P[:, j] = np.where(rest > 0, F[:, m - 1 - j] / rest, 0.0)
            rest = rest * (1.0 - P[:, j])
    else:                                          # MW8: the sphere's angles
        F = front(name, m, n, unit_simplex, unit_sphere)
        P = np.empty((len(F), m - 1))
        rest = np.ones(len(F))
        for j in range(m - 1):
            with np.errstate(divide="ignore", invalid="ignore"):
                v = np.clip(F[:, m - 1 - j] / rest, 0.0, 1.0)
            P[:, j] = np.where(rest > 0, np.arcsin(v) * 2.0 / np.pi, 0.0)
            rest = rest * np.cos(0.5 * np.pi * P[:, j])
    P = np.clip(P, 0.0, upper(name))
    return np.array([np.concatenate([p, optimal_distance(name, p)]) for p in P])


# The fronts' ideal and nadir points: the ends of the pieces, in closed form
# where they are vertices of the feasible region (MW10: where 2 - 16 f1^2 and
# 1.2 - 1.2 f1^2 meet; MW11: 2.07 - 0.23 f1^2 and 3 - 7 f1^2, 0.63 - 0.07 f1^2
# and 3 - 0.625 f1^2; MW13: the shape at x1 = 1.5), computed (dense_front, to
# 1e-12) where the front ends on a curved boundary (MW6, MW12). A test holds
# them to dense_front's to 1e-9.
FRAMES = {
    "MW1": ((0.0, 1.0 - 0.85), (1.0, 1.0)),
    "MW2": ((0.0, 0.0), (1.0, 1.0)),
    "MW3": ((0.0, 0.0), (1.0, 1.0)),
    "MW5": ((0.0, 0.0), (1.0, 1.0)),
    "MW6": ((0.016277631295290976, 0.0), (1.1, 1.0998795564603492)),
    "MW7": ((0.0, 0.0), (1.15, 1.15)),
    "MW9": ((0.0, 0.0), (1.0, 1.0)),
    "MW10": ((math.sqrt(0.8 / 14.8), 0.0), (1.0, 1.2 - 1.2 * 0.8 / 14.8)),
    "MW11": ((math.sqrt(0.93 / 6.77), 0.63 - 0.07 * 2.37 / 0.555),
             (math.sqrt(2.37 / 0.555), 2.07 - 0.23 * 0.93 / 6.77)),
    "MW12": ((0.0, 0.003919148899180422), (1.3163964357562365, 1.0)),
    "MW13": ((0.0, 5.0 - math.exp(1.5) - 0.5), (1.5, 4.0)),
}


def frame(name: str, m: int = 2) -> tuple:
    """(ideal, nadir) of the front: FRAMES; MW4 and MW8 the unit cube's corners
    (the simplex's and the sphere's, all feasible); MW14 0 and 1.5 in the
    positions, h(1.5) and h(0) = 5 in the last objective."""
    if name in ("MW4", "MW8"):
        return (0.0,) * m, (1.0,) * m
    if name == "MW14":
        h = 6.0 - math.exp(1.5) - 1.5 * math.sin(1.1 * math.pi * 2.25)
        return (0.0,) * (m - 1) + (h,), (1.5,) * (m - 1) + (5.0,)
    lo, hi = FRAMES[name]
    return tuple(lo), tuple(hi)


def distance(name: str, m: int):
    """x -> g(x) - 1, the distance to the Pareto set where the front lies at
    g = 1 (ON_G1; distances.py), None elsewhere."""
    if name not in ON_G1:
        return None
    g = G[DISTANCE[name]]
    return lambda x: g(x, m) - 1.0
