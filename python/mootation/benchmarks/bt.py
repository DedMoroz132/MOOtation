# SPDX-License-Identifier: Apache-2.0
# ============================================================================
# BT1-9, the biased multi-objective test suite.
# H. Li, Q. Zhang, J. Deng — "Biased Multiobjective Optimization and
# Decomposition Algorithm", IEEE Transactions on Cybernetics 47(1), 2017,
# Appendix A.
#
# n = 30 variables over [0, 1]^30 (BT7 uses x_2..x_n in [-1, 1]).
# BT1-8 are bi-objective; BT9 has three objectives.
#
# The difficulty is a distance/position bias (the D1/D2/S1/S2 transforms
# below). On the Pareto front every distance contribution is zero, so the
# front is fixed by the position functions alpha:
#   BT1-4, 6, 7, 8:  f2 = 1 - sqrt(f1)
#   BT5:             disconnected
#   BT9:             an octant of the sphere
# ============================================================================
from __future__ import annotations
from typing import List
import numpy as np

from ._rows import spow

N = 30


# ---- bias transforms ------------------------------------------------
def D1(g, theta):
    return g * g + (1.0 - np.exp(-g * g / theta)) / 5.0


def D2(g, theta):
    return g * g + np.abs(g) ** theta / 5.0


def S1(x, gamma):
    # x_1 at every row; its power as of one float (spow), as at one point
    return spow(np.abs(x), gamma)


def S2(x, gamma):
    # np.where evaluates BOTH branches, so the bases are guarded against
    # negatives; the unselected branch cannot affect the result. The powers
    # as of one float (spow), as at one point.
    x = np.asarray(x, float)
    p = lambda base: spow(np.maximum(base, 0.0), gamma)          # noqa: E731
    return np.where(x < 0.25, (1 - p(1 - 4 * x)) / 4.0,
           np.where(x < 0.50, (1 + p(4 * x - 1)) / 4.0,
           np.where(x < 0.75, (3 - p(3 - 4 * x)) / 4.0,
                              (3 + p(4 * x - 3)) / 4.0)))


def Q(z):
    return 4.0 * z * z - np.cos(8.0 * np.pi * z) + 1.0


# ---- index sets I_k (1-indexed j = m..n) ---------------------------
def _idx_sets(n: int, m: int):
    sets = [[] for _ in range(m)]
    for j in range(m, n + 1):
        sets[j % m].append(j)        # mod(j,m) = k-1
    return sets


_I2 = _idx_sets(N, 2)
_I3 = _idx_sets(N, 3)


def _sin_off(j):  return np.sin(j * np.pi / (2 * N))


# ---- eval, of a batch (task 10 of 2026-10-18) -----------------------
# X holds one point per row; bt_rows(name, X) is the (n, M) array of their
# objectives, and btN(x) its batch of one row, by the expressions of the point
# code before, so that a row's values are bit for bit what it gave the point:
# the y_j = x_j - offset(j) as C-ordered rows (a row's sum then runs as the
# point's did), a scalar's ** at every element as pow gives it (spow).
def _ycols(X, idx, off):
    """y_j = x_j - off_j for j in idx (1-indexed), every row; off the offsets
    of the columns (one per j), or a function of j giving a column."""
    Y = np.ascontiguousarray(X[:, [j - 1 for j in idx]])
    if callable(off):
        return Y - np.column_stack([off(j) for j in idx])
    return Y - np.array([off[j] for j in idx])


_SIN_OFF = {j: _sin_off(j) for j in range(1, N + 1)}       # the point's offsets, as computed


def bt_rows(name: str, X) -> np.ndarray:
    X = np.ascontiguousarray(X, dtype=float).reshape(-1, N)
    x1 = X[:, 0]
    if name == "BT9":
        x2 = X[:, 1]
        s = [10.0 * D1(_ycols(X, _I3[k], _SIN_OFF), 1e-9).sum(axis=1) for k in range(3)]
        c1, c2 = np.cos(0.5 * np.pi * x1), np.cos(0.5 * np.pi * x2)
        s1, s2 = np.sin(0.5 * np.pi * x1), np.sin(0.5 * np.pi * x2)
        return np.column_stack([c1 * c2 + s[0], c1 * s2 + s[1], s1 + s[2]])
    theta = {"BT1": 1e-10, "BT3": 1e-8, "BT4": 1e-8, "BT5": 1e-10, "BT6": 1e-4,
             "BT7": 1e-3, "BT8": 1e-3}.get(name)
    if name in ("BT6", "BT8"):
        off = lambda j: spow(x1, 0.5 + 1.5 * (j - 1) / (N - 1))   # noqa: E731
    elif name == "BT7":
        s7 = np.sin(6 * np.pi * x1)
        off = lambda j: s7                                         # noqa: E731
    else:
        off = _SIN_OFF
    if name == "BT2":
        a, b = (D2(_ycols(X, _I2[k], off), 0.2).sum(axis=1) for k in range(2))
    elif name == "BT8":
        a, b = (Q(D1(_ycols(X, _I2[k], off), theta)).sum(axis=1) for k in range(2))
    else:
        a, b = (D1(_ycols(X, _I2[k], off), theta).sum(axis=1) for k in range(2))
    if name == "BT3":
        s = S1(x1, 0.02)
        return np.column_stack([s + a, 1 - np.sqrt(s) + b])
    if name == "BT4":
        s = S2(x1, 0.06)
        return np.column_stack([s + a, 1 - np.sqrt(s) + b])
    if name == "BT5":
        return np.column_stack([x1 + a, (1 - x1) * (1 - x1 * np.sin(8.5 * np.pi * x1)) + b])
    return np.column_stack([x1 + a, 1 - np.sqrt(x1) + b])


def _one(name: str):
    def one(x: List[float]) -> List[float]:
        return bt_rows(name, [list(x)])[0].tolist()
    one.__name__ = name.lower()
    return one


bt1, bt2, bt3, bt4, bt5, bt6, bt7, bt8, bt9 = (_one(f"BT{i}") for i in range(1, 10))


# ---- reference Pareto fronts ---------------------------------------
def _nd(F):
    """Nondominated subset (minimisation). FIX 2026-09-05: the previous
    version computed the rows that DOMINATE F[i] and then dropped THEM, so it
    returned the dominated part of the BT5 curve as the reference front."""
    keep = np.ones(len(F), bool)
    for i in range(len(F)):
        if not keep[i]:
            continue
        dominated_by = np.all(F <= F[i], 1) & np.any(F < F[i], 1)
        if np.any(dominated_by):
            keep[i] = False
    return F[keep]


def pf_concave(n=500):           # f2 = 1 - √f1  (BT1-4,6,7,8)
    f1 = np.linspace(0, 1, n)
    return np.column_stack([f1, 1 - np.sqrt(f1)])


def pf_bt5(n=2000):
    u = np.linspace(0, 1, n)
    F = np.column_stack([u, (1 - u) * (1 - u * np.sin(8.5 * np.pi * u))])
    return _nd(F)


def pf_bt9(n=1000):
    p = max(2, int(np.sqrt(n)))
    a, b = np.meshgrid(np.linspace(0, 1, p), np.linspace(0, 1, p))
    a, b = a.ravel(), b.ravel()
    c1, c2 = np.cos(0.5 * np.pi * a), np.cos(0.5 * np.pi * b)
    s1, s2 = np.sin(0.5 * np.pi * a), np.sin(0.5 * np.pi * b)
    return np.column_stack([c1 * c2, c1 * s2, s1])


# ---- registry: name -> (eval, M, pf, bounds_fn) --------------------
def _b01(): return [(0.0, 1.0)] * N
def _b_bt7():
    return [(0.0, 1.0)] + [(-1.0, 1.0)] * (N - 1)

BT_SPECS = {
    "BT1": (bt1, 2, pf_concave, _b01),
    "BT2": (bt2, 2, pf_concave, _b01),
    "BT3": (bt3, 2, pf_concave, _b01),
    "BT4": (bt4, 2, pf_concave, _b01),
    "BT5": (bt5, 2, pf_bt5,     _b01),
    "BT6": (bt6, 2, pf_concave, _b01),
    "BT7": (bt7, 2, pf_concave, _b_bt7),
    "BT8": (bt8, 2, pf_concave, _b01),
    "BT9": (bt9, 3, pf_bt9,     _b01),
}
