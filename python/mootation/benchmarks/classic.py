# SPDX-License-Identifier: Apache-2.0
"""The verification set (task 17, supplement 2, §3): the classical problems
the papers of the fifteen algorithms checked against them use — not atlas
problems, apart from the families' summaries (families.VERIFICATION).

They are the C++ suite's (include/mootation/suite/classic.hpp, where the
sources and the formulas are written out); without the extension the
papers' formulas in NumPy (rows), which the tests compare with them.

  SCH, FON, POL, KUR   Deb et al., IEEE TEVC 6(2), 2002, Table I
  SPH_2D, SPH_3D, QV   Zitzler, Laumanns & Thiele, TIK-Report 103, 2001,
                       Table 1 (SPH-m, m = 2, 3), n = 100
  KURZ                 ibid., the errata's KUR, n = 100
  ZDT6_D100            ZDT6 at n = 100, as SPEA2's paper runs it (the
                       registry's ZDT6 has n = 10; no size knob)

Their fronts: SCH, FON, SPH-m, ZDT6 in closed form; POL and KUR (Deb's,
n = 2 and 3) from a grid over the box — every point evaluated, the
non-dominated kept (front_exact False); QV from its one-variable front:
both objectives are the mean of one function of a variable, so the
attainable means are the convex combinations of that front's points, whose
lower-left hull, then ^(1/4), is QV's front as n grows (front_exact False);
KURZ none (its paper: "the exact location of the true Pareto set cannot be
determined").
"""

from __future__ import annotations

import functools
import math

import numpy as np

# name: (n variables, objectives, bounds per variable)
SIZES = {
    "SCH": (1, 2, (-1e3, 1e3)),
    "FON": (3, 2, (-4.0, 4.0)),
    "POL": (2, 2, (-math.pi, math.pi)),
    "KUR": (3, 2, (-5.0, 5.0)),
    "SPH_2D": (100, 2, (-1e3, 1e3)),
    "SPH_3D": (100, 3, (-1e3, 1e3)),
    "QV": (100, 2, (-5.0, 5.0)),
    "KURZ": (100, 2, (-1e3, 1e3)),
    "ZDT6_D100": (100, 2, (0.0, 1.0)),
}


def rows(name: str, X) -> np.ndarray:
    """The objectives of the rows of X by the papers' formulas in NumPy: what
    runs without the extension (with it the C++ suite's, the same to a few
    ulps; the tests compare the two)."""
    X = np.atleast_2d(np.asarray(X, float))
    if name == "SCH":
        return np.column_stack([X[:, 0] ** 2, (X[:, 0] - 2.0) ** 2])
    if name == "FON":
        c = 1.0 / math.sqrt(3.0)
        return np.column_stack([1.0 - np.exp(-((X - c) ** 2).sum(axis=1)),
                                1.0 - np.exp(-((X + c) ** 2).sum(axis=1))])
    if name == "POL":
        a1 = 0.5 * math.sin(1.0) - 2.0 * math.cos(1.0) + math.sin(2.0) - 1.5 * math.cos(2.0)
        a2 = 1.5 * math.sin(1.0) - math.cos(1.0) + 2.0 * math.sin(2.0) - 0.5 * math.cos(2.0)
        s1, c1, s2, c2 = np.sin(X[:, 0]), np.cos(X[:, 0]), np.sin(X[:, 1]), np.cos(X[:, 1])
        b1 = 0.5 * s1 - 2.0 * c1 + s2 - 1.5 * c2
        b2 = 1.5 * s1 - c1 + 2.0 * s2 - 0.5 * c2
        return np.column_stack([1.0 + (a1 - b1) ** 2 + (a2 - b2) ** 2,
                                (X[:, 0] + 3.0) ** 2 + (X[:, 1] + 1.0) ** 2])
    if name == "KUR":
        return np.column_stack([
            (-10.0 * np.exp(-0.2 * np.sqrt(X[:, :-1] ** 2 + X[:, 1:] ** 2))).sum(axis=1),
            (np.abs(X) ** 0.8 + 5.0 * np.sin(X ** 3)).sum(axis=1)])
    if name in ("SPH_2D", "SPH_3D"):
        sq = (X ** 2).sum(axis=1)
        return np.column_stack([sq - 2.0 * X[:, j] + 1.0 for j in range(SIZES[name][1])])
    if name == "QV":
        r = lambda Y: (Y ** 2 - 10.0 * np.cos(2.0 * np.pi * Y) + 10.0).mean(axis=1) ** 0.25  # noqa: E731
        return np.column_stack([r(X), r(X - 1.5)])
    if name == "KURZ":
        return np.column_stack([
            (np.abs(X) ** 0.8 + 5.0 * np.sin(X) ** 3 + 3.5828).sum(axis=1),
            (1.0 - np.exp(-0.2 * np.sqrt(X[:, :-1] ** 2 + X[:, 1:] ** 2))).sum(axis=1)])
    if name == "ZDT6_D100":
        f1 = 1.0 - np.exp(-4.0 * X[:, 0]) * np.sin(6.0 * np.pi * X[:, 0]) ** 6
        g = 1.0 + 9.0 * (X[:, 1:].sum(axis=1) / (X.shape[1] - 1)) ** 0.25
        return np.column_stack([f1, g * (1.0 - (f1 / g) ** 2)])
    raise KeyError(name)


def _nondominated_2d(F: np.ndarray, X: np.ndarray):
    """The rows of F no other row dominates (two objectives), sorted by f1."""
    order = np.lexsort((F[:, 1], F[:, 0]))
    F, X = F[order], X[order]
    before = np.concatenate([[np.inf], np.minimum.accumulate(F[:-1, 1])])
    keep = F[:, 1] < before                          # below every f2 to its left
    return F[keep], X[keep]


def _even(F: np.ndarray, n: int) -> np.ndarray:
    """n of the rows of F (sorted along the front), evenly by index."""
    if len(F) <= n:
        return F
    return F[np.round(np.linspace(0, len(F) - 1, n)).astype(int)]


@functools.lru_cache(maxsize=None)
def _grid_front(name: str) -> np.ndarray:
    """POL's and KUR's non-dominated points over a grid of the box (the C++
    suite's values), then once more over a finer grid of the cell around
    every one of them."""
    from .registry import PROBLEMS
    p = PROBLEMS[name]
    n_var = p.n_vars
    lo, hi = p.bounds[0]
    k = 1501 if n_var == 2 else 161
    axis = np.linspace(lo, hi, k)
    X = np.stack(np.meshgrid(*([axis] * n_var), indexing="ij"), axis=-1).reshape(-1, n_var)
    F, X = _nondominated_2d(np.asarray(p.evaluate_batch(X), float), X)
    step = (hi - lo) / (k - 1)
    sub = np.linspace(-step, step, 21 if n_var == 2 else 11)
    D = np.stack(np.meshgrid(*([sub] * n_var), indexing="ij"), axis=-1).reshape(-1, n_var)
    Y = np.clip((X[:, None, :] + D[None, :, :]).reshape(-1, n_var), lo, hi)
    F2, _ = _nondominated_2d(np.vstack([F, np.asarray(p.evaluate_batch(Y), float)]),
                             np.vstack([X, Y]))
    return F2


@functools.lru_cache(maxsize=None)
def _qv_front() -> np.ndarray:
    """QV's front as n grows: the lower-left convex hull of the one-variable
    front of (r(x), r(x - 1.5)), r(y) = y^2 - 10 cos(2 pi y) + 10, then
    ^(1/4); the hull's segments sampled densely."""
    x = np.linspace(0.0, 1.5, 300_001)
    r = lambda y: y * y - 10.0 * np.cos(2.0 * np.pi * y) + 10.0      # noqa: E731
    F, _ = _nondominated_2d(np.column_stack([r(x), r(x - 1.5)]), x[:, None])
    hull = [F[0]]
    for p in F[1:]:                                   # the lower convex hull, left to right
        while len(hull) >= 2:
            a, b = hull[-2], hull[-1]
            if (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) <= 0.0:
                hull.pop()
            else:
                break
        hull.append(p)
    H = np.array(hull)
    pts = [np.linspace(H[i], H[i + 1], 200, endpoint=False) for i in range(len(H) - 1)]
    return np.vstack(pts + [H[-1:]]) ** 0.25


def pareto_front(name: str, n: int) -> np.ndarray | None:
    """n points of the front (None for KURZ)."""
    t = np.linspace(0.0, 1.0, n)
    if name == "SCH":
        x = 2.0 * t
        return np.column_stack([x * x, (x - 2.0) ** 2])
    if name == "FON":
        c = 1.0 / math.sqrt(3.0)
        x = -c + 2.0 * c * t
        return np.column_stack([1.0 - np.exp(-3.0 * (x - c) ** 2), 1.0 - np.exp(-3.0 * (x + c) ** 2)])
    if name in ("SPH_2D", "SPH_3D"):
        W = _simplex(SIZES[name][1], n)
        sq = (W * W).sum(axis=1, keepdims=True)
        return sq - 2.0 * W + 1.0                     # f_j = |x|^2 - 2 x_j + 1, x on the simplex
    if name == "ZDT6_D100":
        from .registry import _pf_zdt6
        return _pf_zdt6(n)
    if name in ("POL", "KUR"):
        return _even(_grid_front(name), n)
    if name == "QV":
        return _even(_qv_front(), n)
    return None


def _simplex(m: int, n: int) -> np.ndarray:
    """About n points of the unit simplex in m coordinates (the Das-Dennis
    lattice of the largest H with at most n points)."""
    from ..run.metrics import das_dennis, lattice_h
    return das_dennis(m, lattice_h(m, n))


def pareto_set(name: str, n: int) -> np.ndarray | None:
    """n points of the Pareto set where it is known in closed form."""
    t = np.linspace(0.0, 1.0, n)
    if name == "SCH":
        return (2.0 * t)[:, None]
    if name == "FON":
        c = 1.0 / math.sqrt(3.0)
        return np.repeat((-c + 2.0 * c * t)[:, None], 3, axis=1)
    if name in ("SPH_2D", "SPH_3D"):
        m = SIZES[name][1]
        W = _simplex(m, n)
        return np.hstack([W, np.zeros((len(W), 100 - m))])
    return None


def frame(name: str):
    """(ideal, nadir) where they are known; None where the front gives them."""
    if name == "SCH":
        return (0.0, 0.0), (4.0, 4.0)
    if name == "FON":
        v = 1.0 - math.exp(-4.0)
        return (0.0, 0.0), (v, v)
    if name in ("SPH_2D", "SPH_3D"):
        m = SIZES[name][1]
        return (0.0,) * m, (2.0,) * m
    return None


def register(PROBLEMS: dict, BenchProblem, budget, ref_norm, divisor) -> list:
    """Every problem of the set into PROBLEMS, its evaluation the C++ suite's
    once native.attach runs; the names whose frame is exact."""
    exact = []
    for name, (n_var, m, (lo, hi)) in SIZES.items():
        fr = frame(name)
        ideal, nadir = fr if fr is not None else ((0.0,) * m, (1.0,) * m)
        if fr is not None or name == "ZDT6_D100":
            exact.append(name)
        if name == "ZDT6_D100":
            ideal, nadir = PROBLEMS["ZDT6"].ideal, PROBLEMS["ZDT6"].nadir
        pop, ng = budget(m)
        PROBLEMS[name] = BenchProblem(
            name=name, n_vars=n_var, bounds=[(lo, hi)] * n_var, n_obj=m,
            evaluate=lambda x, _n=name: rows(_n, [list(x)])[0].tolist(),
            evaluate_batch=lambda X, _n=name: rows(_n, X), constraints=lambda x: [],
            hv_ref_raw=tuple(v + 0.1 * (v - u) for u, v in zip(ideal, nadir)),
            hv_ref_norm=ref_norm(m), hv_norm_divisor=divisor(m),
            ideal=tuple(ideal), nadir=tuple(nadir),
            pop_size=pop, n_gen=ng, K_runs=21, has_cons=False,
            pareto_front=(None if name == "KURZ" else (lambda k, _n=name: pareto_front(_n, k))),
            pareto_set=(None if pareto_set(name, 2) is None else (lambda k, _n=name: pareto_set(_n, k))),
            front_exact=name not in ("POL", "KUR", "QV"))
    return exact
