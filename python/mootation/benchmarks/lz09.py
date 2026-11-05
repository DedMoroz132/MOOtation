# SPDX-License-Identifier: Apache-2.0
"""LZ09 F1-F9, the test problems with complicated Pareto sets.

H. Li and Q. Zhang, "Multiobjective Optimization Problems With Complicated
Pareto Sets, MOEA/D and NSGA-II", IEEE TEVC 13(2): 284-302, 2009 (source
moead-de_li2009), Table I, read off the PDF (task 20 of 2026-11-02,
§20.1.3); the C++ suite (include/mootation/suite/lz09.hpp) is the reference,
this module its cross-check.

Every instance is the paper's (2)-(3): f_i = alpha_i(x_I) + beta_i(x_II -
g(x_I)), so the Pareto set is x_II = g(x_I) (Theorem 1) — optimum() below —
over x_1 in [0, 1] (F6: x_1, x_2 in [0, 1]^2), and the front is alpha's:

  F1-F5, F7, F8   f2 = 1 - sqrt(f1), 0 <= f1 <= 1     (two objectives)
  F9              f2 = 1 - f1^2
  F6              f1^2 + f2^2 + f3^2 = 1, f >= 0       (three objectives)

  n = 30 for F1-F5 and F9, 10 for F6-F8 (§V-A, 2)); the box [0, 1]^n for F1,
  F7, F8, [0, 1] x [-1, 1]^(n-1) for F2-F5 and F9, [0, 1]^2 x [-2, 2]^(n-2)
  for F6.

READINGS (the paper's text, reading P4 of the task):
  LZ09-1. J1 = {j odd, 2 <= j <= n} goes into f1, J2 = {j even} into f2 (F6:
     j - 1, j - 2 and j multiples of 3, 3 <= j <= n, into f1, f2, f3), as
     printed; j counts the variables from 1.
  LZ09-2. F8's cosines are cos(20 y_j pi / sqrt(j)) with j the variable's
     number, as printed.
The variant of jMetal's code (reading P5: variables in [0, 1] stretched to
[-1, 1], J1 and J2 swapped, F8's sqrt of the index within the set) is not
here yet.

The order of every operation is the one lz09.hpp repeats (common.hpp, NUMPY'S
ORDER): the set's terms summed with np.sum (pairwise: the columns a set
picks copied row by row first, _rows.py), F8's cosines multiplied with
np.prod (left to right), y^2 as y * y.
"""

from __future__ import annotations

import numpy as np

N_VARS = {1: 30, 2: 30, 3: 30, 4: 30, 5: 30, 6: 10, 7: 10, 8: 10, 9: 30}


def n_obj(i: int) -> int:
    return 3 if i == 6 else 2


def bounds(i: int) -> list:
    n = N_VARS[i]
    if i in (1, 7, 8):
        return [(0.0, 1.0)] * n
    if i == 6:
        return [(0.0, 1.0)] * 2 + [(-2.0, 2.0)] * (n - 2)
    return [(0.0, 1.0)] + [(-1.0, 1.0)] * (n - 1)


def index_sets(i: int) -> list:
    """J1, J2 (and J3 for F6): the 1-indexed variables of each objective."""
    n = N_VARS[i]
    if i == 6:
        return [np.array([j for j in range(3, n + 1) if (j - k) % 3 == 0]) for k in (1, 2, 0)]
    return [np.arange(3, n + 1, 2), np.arange(2, n + 1, 2)]


def optimum(i: int, X: np.ndarray) -> np.ndarray:
    """g(x_I) of the rows of X: the optimal x_j, j = 2 .. n (F6: 3 .. n),
    given x_1 (and x_2) — the Pareto set's (Table I, "Its PS is")."""
    X = np.asarray(X, float)
    n = N_VARS[i]
    x1 = X[:, 0][:, None]
    if i == 6:
        J = np.arange(3, n + 1)
        return 2.0 * X[:, 1][:, None] * np.sin(2.0 * np.pi * x1 + J * np.pi / n)
    J = np.arange(2, n + 1)
    if i in (1, 7, 8):
        return x1 ** (0.5 * (1.0 + 3.0 * (J - 2) / (n - 2)))
    if i in (2, 9):
        return np.sin(6.0 * np.pi * x1 + J * np.pi / n)
    odd = (J % 2 == 1)                                 # J1 (cos) and J2 (sin)
    if i == 3:
        return np.where(odd, 0.8 * x1 * np.cos(6.0 * np.pi * x1 + J * np.pi / n),
                        0.8 * x1 * np.sin(6.0 * np.pi * x1 + J * np.pi / n))
    if i == 4:
        return np.where(odd, 0.8 * x1 * np.cos((6.0 * np.pi * x1 + J * np.pi / n) / 3.0),
                        0.8 * x1 * np.sin(6.0 * np.pi * x1 + J * np.pi / n))
    a = 0.3 * x1 * x1 * np.cos(24.0 * np.pi * x1 + 4.0 * J * np.pi / n) + 0.6 * x1        # F5
    return np.where(odd, a * np.cos(6.0 * np.pi * x1 + J * np.pi / n),
                    a * np.sin(6.0 * np.pi * x1 + J * np.pi / n))


def lz09_rows(i: int, X) -> np.ndarray:
    """The objectives of the rows of X for F<i>."""
    n = N_VARS[i]
    X = np.ascontiguousarray(X, dtype=float).reshape(-1, n)
    first = 3 if i == 6 else 2
    Y = X[:, first - 1:] - optimum(i, X)               # y_j, j = first .. n
    sets = index_sets(i)
    cols = [Js - first for Js in sets]
    x1 = X[:, 0]
    if i == 6:
        x2 = X[:, 1]
        S = [np.sum(Yc * Yc, axis=1) for Yc in (np.ascontiguousarray(Y[:, c]) for c in cols)]
        c1, c2 = np.cos(0.5 * x1 * np.pi), np.cos(0.5 * x2 * np.pi)
        s1, s2 = np.sin(0.5 * x1 * np.pi), np.sin(0.5 * x2 * np.pi)
        return np.column_stack([c1 * c2 + 2.0 / len(sets[0]) * S[0],
                                c1 * s2 + 2.0 / len(sets[1]) * S[1],
                                s1 + 2.0 / len(sets[2]) * S[2]])
    if i == 7:
        H = 4.0 * Y * Y - np.cos(8.0 * Y * np.pi) + 1.0
        S = [np.sum(np.ascontiguousarray(H[:, c]), axis=1) for c in cols]
    elif i == 8:
        S = []
        for Js, c in zip(sets, cols):
            Yc = np.ascontiguousarray(Y[:, c])
            sq = np.sum(Yc * Yc, axis=1)
            pr = np.prod(np.cos(20.0 * Yc * np.pi / np.sqrt(Js)), axis=1)
            S.append(4.0 * sq - 2.0 * pr + 2.0)
    else:
        S = [np.sum(Yc * Yc, axis=1) for Yc in (np.ascontiguousarray(Y[:, c]) for c in cols)]
    a2 = 1.0 - x1 * x1 if i == 9 else 1.0 - np.sqrt(x1)
    return np.column_stack([x1 + 2.0 / len(sets[0]) * S[0], a2 + 2.0 / len(sets[1]) * S[1]])


def pareto_set(i: int, n: int) -> np.ndarray:
    """n points of the Pareto set: x_1 = linspace(0, 1, n) (F6: x_1, x_2 on
    a p x p grid, p = round(sqrt(n)), x_1 the faster), the rest optimum()."""
    if i == 6:
        p = max(2, int(round(np.sqrt(n))))
        a, b = np.meshgrid(np.linspace(0.0, 1.0, p), np.linspace(0.0, 1.0, p))
        P = np.column_stack([a.ravel(), b.ravel()])
    else:
        P = np.linspace(0.0, 1.0, n)[:, None]
    return np.column_stack([P, optimum(i, P)])


def name(i: int) -> str:
    return f"LZ09_F{i}"
