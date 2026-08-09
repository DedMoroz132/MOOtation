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

N = 30


# ---- bias transforms ------------------------------------------------
def D1(g, theta):
    return g * g + (1.0 - np.exp(-g * g / theta)) / 5.0


def D2(g, theta):
    return g * g + np.abs(g) ** theta / 5.0


def S1(x, gamma):
    return np.abs(x) ** gamma


def S2(x, gamma):
    # np.where evaluates BOTH branches, so the bases are guarded against
    # negatives; the unselected branch cannot affect the result.
    x = np.asarray(x, float)
    p = lambda base: np.maximum(base, 0.0) ** gamma
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


def _yvec(x, idx, offset):
    """y_j = x_j - offset(j) for j in idx (1-indexed)."""
    return np.array([x[j - 1] - offset(j) for j in idx])


# ---- eval (M=2) ----------------------------------------------------
def _sin_off(j):  return np.sin(j * np.pi / (2 * N))


def bt1(x):
    x = np.asarray(x, float)
    a = D1(_yvec(x, _I2[0], _sin_off), 1e-10).sum()
    b = D1(_yvec(x, _I2[1], _sin_off), 1e-10).sum()
    return [x[0] + a, 1 - np.sqrt(x[0]) + b]


def bt2(x):
    x = np.asarray(x, float)
    a = D2(_yvec(x, _I2[0], _sin_off), 0.2).sum()
    b = D2(_yvec(x, _I2[1], _sin_off), 0.2).sum()
    return [x[0] + a, 1 - np.sqrt(x[0]) + b]


def bt3(x):
    x = np.asarray(x, float); s = S1(x[0], 0.02)
    a = D1(_yvec(x, _I2[0], _sin_off), 1e-8).sum()
    b = D1(_yvec(x, _I2[1], _sin_off), 1e-8).sum()
    return [s + a, 1 - np.sqrt(s) + b]


def bt4(x):
    x = np.asarray(x, float); s = float(S2(x[0], 0.06))
    a = D1(_yvec(x, _I2[0], _sin_off), 1e-8).sum()
    b = D1(_yvec(x, _I2[1], _sin_off), 1e-8).sum()
    return [s + a, 1 - np.sqrt(s) + b]


def bt5(x):
    x = np.asarray(x, float); x1 = x[0]
    a = D1(_yvec(x, _I2[0], _sin_off), 1e-10).sum()
    b = D1(_yvec(x, _I2[1], _sin_off), 1e-10).sum()
    return [x1 + a, (1 - x1) * (1 - x1 * np.sin(8.5 * np.pi * x1)) + b]


def _pow_off(j, x1):  # BT6/BT8: x_j = x1^(0.5+1.5(j-1)/(n-1))
    return x1 ** (0.5 + 1.5 * (j - 1) / (N - 1))


def bt6(x):
    x = np.asarray(x, float); x1 = x[0]
    off = lambda j: _pow_off(j, x1)
    a = D1(_yvec(x, _I2[0], off), 1e-4).sum()
    b = D1(_yvec(x, _I2[1], off), 1e-4).sum()
    return [x1 + a, 1 - np.sqrt(x1) + b]


def bt7(x):
    x = np.asarray(x, float); x1 = x[0]
    off = lambda j: np.sin(6 * np.pi * x1)
    a = D1(_yvec(x, _I2[0], off), 1e-3).sum()
    b = D1(_yvec(x, _I2[1], off), 1e-3).sum()
    return [x1 + a, 1 - np.sqrt(x1) + b]


def bt8(x):
    x = np.asarray(x, float); x1 = x[0]
    off = lambda j: _pow_off(j, x1)
    a = Q(D1(_yvec(x, _I2[0], off), 1e-3)).sum()
    b = Q(D1(_yvec(x, _I2[1], off), 1e-3)).sum()
    return [x1 + a, 1 - np.sqrt(x1) + b]


def bt9(x):
    x = np.asarray(x, float); x1, x2 = x[0], x[1]
    off = _sin_off
    s = [10.0 * D1(_yvec(x, _I3[k], off), 1e-9).sum() for k in range(3)]
    c1, c2 = np.cos(0.5 * np.pi * x1), np.cos(0.5 * np.pi * x2)
    s1, s2 = np.sin(0.5 * np.pi * x1), np.sin(0.5 * np.pi * x2)
    return [c1 * c2 + s[0], c1 * s2 + s[1], s1 + s[2]]


# ---- reference Pareto fronts ---------------------------------------
def _nd(F):
    keep = np.ones(len(F), bool)
    for i in range(len(F)):
        if keep[i]:
            dom = np.all(F <= F[i], 1) & np.any(F < F[i], 1); dom[i] = False
            keep[dom] = False
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
