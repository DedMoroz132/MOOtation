# SPDX-License-Identifier: Apache-2.0
"""UF1-UF10, the unconstrained problems of the CEC 2009 competition.

Q. Zhang, A. Zhou, S. Zhao, P. N. Suganthan, W. Liu, S. Tiwari,
"Multiobjective optimization Test Instances for the CEC 2009 Special Session
and Competition", Technical Report CES-487, University of Essex, version
"April 20, 2009 DRAFT" (source cec2009-uf-cf_zhang2008), §II; task 20 of
2026-11-02, §20.1.1. The C++ suite (include/mootation/suite/uf.hpp) is the
reference, this module its cross-check.

n = 30; UF1-UF7 two objectives, UF8-UF10 three; UF5 N = 10, epsilon = 0.1,
UF6 N = 2, epsilon = 0.1, UF9 epsilon = 0.1 (the contest's values). The
boxes: [0, 1] x [-1, 1]^(n-1) (UF1, UF2, UF5-UF7), [0, 1]^n (UF3), [0, 1] x
[-2, 2]^(n-1) (UF4), [0, 1]^2 x [-2, 2]^(n-2) (UF8-UF10). UF1, UF2 and UF3
are LZ09's F2, F5 and F8 at n = 30 (lz09.py; the report's "F2 in [9]" ...).

The fronts (the report's "Its PF"): f2 = 1 - sqrt(f1) (UF1-UF3), 1 - f1^2
(UF4), the 2N + 1 points (i / 2N, 1 - i / 2N) (UF5), the point (0, 1) and
f2 = 1 - f1 over f1 in the N intervals [(2i - 1) / 2N, 2i / 2N] (UF6),
f2 = 1 - f1 (UF7), the octant of the unit sphere (UF8, UF10), two parts of
the plane f1 + f2 + f3 = 1 (UF9). The Pareto sets: the report's "Its PS is"
(optimum below) over x_1 (and x_2); UF5's x_1 = i / 2N, UF6's x_1 = 0 and
the intervals, UF9's x_1 in [0, 1/4] or [3/4, 1].

READINGS (the report's text):
  UF-1. UF10's sums: f2's and f3's run over J2 and J3, not over J1 as the
     report prints (reading P1 of the task): pagmo2, PlatEMO and jMetal sum
     so, and the report's own factors 2/|J2| and 2/|J3| say so.
  UF-2. Harmless misprints, read as meant: UF3's "J1 and J2 are the same as
     those of F1" is UF1's; UF4's "y_i = x_j - ..." is y_j; UF4's "Its PF is
     f2 = 1 - f2^2" is 1 - f1^2; UF10's "f1^2 + f2^2 + f3^3 = 1" is f3^2.
  UF11-UF13 are not here: they need the competition's data files (a rotation
  and bounds).

The order of every operation is the one uf.hpp repeats (common.hpp, NUMPY'S
ORDER): a set's terms summed with np.sum (pairwise, the set's columns copied
row by row first), the cosines multiplied with np.prod (left to right), y^2
as y * y.
"""

from __future__ import annotations

import numpy as np

N_VARS = 30
N_PARTS = {5: 10, 6: 2}                 # the N of UF5 and UF6
EPS = 0.1                               # UF5, UF6, UF9


def n_obj(i: int) -> int:
    return 3 if i >= 8 else 2


def bounds(i: int) -> list:
    n = N_VARS
    if i == 3:
        return [(0.0, 1.0)] * n
    if i >= 8:
        return [(0.0, 1.0)] * 2 + [(-2.0, 2.0)] * (n - 2)
    return [(0.0, 1.0)] + [(-2.0, 2.0) if i == 4 else (-1.0, 1.0)] * (n - 1)


def index_sets(i: int) -> list:
    """J1, J2 (and J3 for UF8-UF10): the 1-indexed variables of each objective."""
    n = N_VARS
    if i >= 8:
        return [np.array([j for j in range(3, n + 1) if (j - k) % 3 == 0]) for k in (1, 2, 0)]
    return [np.arange(3, n + 1, 2), np.arange(2, n + 1, 2)]


def optimum(i: int, X: np.ndarray) -> np.ndarray:
    """The optimal x_j, j = 2 .. n (UF8-UF10: 3 .. n), given x_1 (and x_2)."""
    X = np.asarray(X, float)
    n = N_VARS
    x1 = X[:, 0][:, None]
    if i >= 8:
        J = np.arange(3, n + 1)
        return 2.0 * X[:, 1][:, None] * np.sin(2.0 * np.pi * x1 + J * np.pi / n)
    J = np.arange(2, n + 1)
    if i == 3:
        return x1 ** (0.5 * (1.0 + 3.0 * (J - 2) / (n - 2)))
    if i == 2:
        a = 0.3 * x1 * x1 * np.cos(24.0 * np.pi * x1 + 4.0 * J * np.pi / n) + 0.6 * x1
        return np.where(J % 2 == 1, a * np.cos(6.0 * np.pi * x1 + J * np.pi / n),
                        a * np.sin(6.0 * np.pi * x1 + J * np.pi / n))
    return np.sin(6.0 * np.pi * x1 + J * np.pi / n)


def _sums(i: int, Y: np.ndarray, sets) -> list:
    """Each set's distance term: (2 / |J|) times its sum (UF3, UF6: 4 sum y^2
    - 2 prod cos(20 y pi / sqrt(j)) + 2)."""
    first = 3 if i >= 8 else 2
    out = []
    for Js in sets:
        Yc = np.ascontiguousarray(Y[:, Js - first])     # rows in memory: pairwise sums (_rows.py)
        if i in (3, 6):
            sq = np.sum(Yc * Yc, axis=1)
            pr = np.prod(np.cos(20.0 * Yc * np.pi / np.sqrt(Js)), axis=1)
            s = 4.0 * sq - 2.0 * pr + 2.0
        elif i == 4:
            A = np.abs(Yc)
            s = np.sum(A / (1.0 + np.exp(2.0 * A)), axis=1)
        elif i == 5:
            s = np.sum(2.0 * Yc * Yc - np.cos(4.0 * np.pi * Yc) + 1.0, axis=1)
        elif i == 10:
            s = np.sum(4.0 * Yc * Yc - np.cos(8.0 * np.pi * Yc) + 1.0, axis=1)
        else:
            s = np.sum(Yc * Yc, axis=1)
        out.append(2.0 / len(Js) * s)
    return out


def uf_rows(i: int, X) -> np.ndarray:
    """The objectives of the rows of X for UF<i>."""
    n = N_VARS
    X = np.ascontiguousarray(X, dtype=float).reshape(-1, n)
    first = 3 if i >= 8 else 2
    Y = X[:, first - 1:] - optimum(i, X)
    S = _sums(i, Y, index_sets(i))
    x1 = X[:, 0]
    if i >= 8:
        x2 = X[:, 1]
        if i == 9:
            m = np.maximum(0.0, (1.0 + EPS) * (1.0 - 4.0 * (2.0 * x1 - 1.0) * (2.0 * x1 - 1.0)))
            return np.column_stack([0.5 * (m + 2.0 * x1) * x2 + S[0],
                                    0.5 * (m - 2.0 * x1 + 2.0) * x2 + S[1],
                                    1.0 - x2 + S[2]])
        c1, c2 = np.cos(0.5 * x1 * np.pi), np.cos(0.5 * x2 * np.pi)
        s1, s2 = np.sin(0.5 * x1 * np.pi), np.sin(0.5 * x2 * np.pi)
        return np.column_stack([c1 * c2 + S[0], c1 * s2 + S[1], s1 + S[2]])
    if i == 4:
        return np.column_stack([x1 + S[0], 1.0 - x1 * x1 + S[1]])
    if i in (5, 6):
        N = N_PARTS[i]
        if i == 5:
            e = (1.0 / (2.0 * N) + EPS) * np.abs(np.sin(2.0 * N * np.pi * x1))
        else:
            e = np.maximum(0.0, 2.0 * (1.0 / (2.0 * N) + EPS) * np.sin(2.0 * N * np.pi * x1))
        return np.column_stack([x1 + e + S[0], 1.0 - x1 + e + S[1]])
    if i == 7:
        r = x1 ** 0.2
        return np.column_stack([r + S[0], 1.0 - r + S[1]])
    return np.column_stack([x1 + S[0], 1.0 - np.sqrt(x1) + S[1]])


def _grid(p: int, x1_values: np.ndarray) -> np.ndarray:
    """(x_1, x_2) over x1_values x linspace(0, 1, p), x_1 the faster."""
    a, b = np.meshgrid(x1_values, np.linspace(0.0, 1.0, p))
    return np.column_stack([a.ravel(), b.ravel()])


def _uf6_x1(n: int) -> np.ndarray:
    """0, then max(2, (n - 1) // N) points evenly over each of UF6's N intervals."""
    N = N_PARTS[6]
    m = max(2, (n - 1) // N)
    return np.concatenate([[0.0]] + [np.linspace((2 * k - 1) / (2.0 * N), 2 * k / (2.0 * N), m)
                                     for k in range(1, N + 1)])


def _uf9_x1(p: int) -> np.ndarray:
    """p points evenly over [0, 1/4] and p over [3/4, 1]."""
    return np.concatenate([np.linspace(0.0, 0.25, p), np.linspace(0.75, 1.0, p)])


def positions(i: int, n: int) -> np.ndarray:
    """The Pareto set's x_1 (and x_2) behind about n points."""
    if i >= 8:
        p = max(2, int(round(np.sqrt(n))))
        if i == 9:
            q = max(2, int(round(np.sqrt(n / 2.0))))
            return _grid(q, _uf9_x1(q))
        return _grid(p, np.linspace(0.0, 1.0, p))
    if i == 5:
        N = N_PARTS[5]
        return (np.arange(2 * N + 1) / (2.0 * N))[:, None]
    if i == 6:
        return _uf6_x1(n)[:, None]
    return np.linspace(0.0, 1.0, n)[:, None]


def pareto_set(i: int, n: int) -> np.ndarray:
    P = positions(i, n)
    return np.column_stack([P, optimum(i, P)])


def front(i: int, n: int) -> np.ndarray:
    """The reference front of UF5, UF6, UF7 and UF9 (the others: MOP's
    samplers, the same curves): the image of the Pareto set's positions."""
    P = positions(i, n)
    x1 = P[:, 0]
    if i == 9:
        x2 = P[:, 1]
        return np.column_stack([x1 * x2, (1.0 - x1) * x2, 1.0 - x2])
    if i == 7:
        x1 = np.linspace(0.0, 1.0, n)
    return np.column_stack([x1, 1.0 - x1])


def name(i: int) -> str:
    return f"UF{i}"
