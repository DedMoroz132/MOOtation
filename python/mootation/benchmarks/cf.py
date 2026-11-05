# SPDX-License-Identifier: Apache-2.0
"""CF1-CF10, the constrained problems of the CEC 2009 competition.

Q. Zhang, A. Zhou, S. Zhao, P. N. Suganthan, W. Liu, S. Tiwari, CES-487,
version "April 20, 2009 DRAFT" (source cec2009-uf-cf_zhang2008), §III; task
20 of 2026-11-02, §20.1.2. The C++ suite (include/mootation/suite/cf.hpp) is
the reference, this module its cross-check. The registry's names are
CF1-CEC2009 .. CF10-CEC2009: jMetal's package cf/CF1 .. CF16 is another set
(Zhou, Xiang, He 2021).

n = 10; CF1-CF7 two objectives, CF8-CF10 three. The constants (the contest's
values): CF1 N = 10, a = 1; CF2 and CF3 N = 2, a = 1; CF8 N = 2, a = 4; CF9 N
= 2, a = 3; CF10 N = 2, a = 1. The boxes: [0, 1]^n (CF1), [0, 1] x [-1,
1]^(n-1) (CF2), [0, 1] x [-2, 2]^(n-1) (CF3-CF7), [0, 1]^2 x [-4, 4]^(n-2)
(CF8), [0, 1]^2 x [-2, 2]^(n-2) (CF9, CF10). The report writes each
constraint c(x) >= 0; the library's constraint is g = -c (g <= 0 feasible).

The Pareto sets (optimum and pareto_set below), from the report's fronts:
every distance term at its optimum but those the constraints hold away from
it — CF4's and CF5's y_2 at 0 (x_1 <= 1/2), 0.5 x_1 - 0.25 (to 3/4) or 1
(above), CF6's and CF7's y_2 = sqrt(max(0, a)) and y_4 = sqrt(max(0, b)),
a = 0.5 (1 - x_1) - (1 - x_1)^2, b = 0.25 sqrt(1 - x_1) - 0.5 (1 - x_1);
x_1 over the parts the constraint leaves (CF1: i / 2N; CF2: 0 and f_1 = x_1
in [((2i - 1) / 2N)^2, (2i / 2N)^2]; CF3: 0 and [sqrt((2i - 1) / 2N),
sqrt(2i / 2N)]); CF8-CF10 x_2 by r = cos^2(pi x_2 / 2), which is f_1^2 / (1 -
f_3^2): CF8 r = i / 2N, CF9 and CF10 r = 0 and r in [(2i - 1) / 2N, 2i / 2N].
The front is the problem's own image of that sample (front below).

READINGS:
  CF-1. No clipping of x into the box inside the problem (reading P3 of the
     task): PlatEMO clips in CF8 and CF9; the box is the algorithm's.
  CF-2. The competition's tolerance (§IV: a constraint counts as met from
     -1e-6) belongs to its assessment, not to the problem: g = -c as printed.
     CF1's and CF8's Pareto points sit on the constraint's boundary, where
     the computed g is a rounding error either side of 0.
  CF-3. CF8-CF10 divide by 1 - f3^2, which is 0 at f3 = 1 (x_1 = 1 with the J3
     terms at their optimum): the constraint is then 0/0 or x/0 and comes
     out NaN, as the printed formula and PlatEMO's code give it; the point
     is the front's pole (0, 0, 1).
  CF-4. CF8-CF10 follow the version of 20 April 2009 (reading P2): the final
     report of the competition says earlier versions had misprints there,
     and results published in 2008-2009 on CF8-CF10 are no reference.
  CF-5. |sin| in CF1 and CF8, sin without the modulus in CF2, CF3, CF9 and
     CF10, as printed; the threshold of CF4's and CF5's h_2 is 3/2 (1 -
     sqrt(2)/2), as printed.
"""

from __future__ import annotations

import numpy as np

N_VARS = 10
PARAMS = {1: (10, 1.0), 2: (2, 1.0), 3: (2, 1.0), 8: (2, 4.0), 9: (2, 3.0), 10: (2, 1.0)}   # (N, a)
THRESHOLD = 1.5 * (1.0 - np.sqrt(2.0) / 2.0)                                            # CF4, CF5's h_2


def n_obj(i: int) -> int:
    return 3 if i >= 8 else 2


def n_con(i: int) -> int:
    return 2 if i in (6, 7) else 1


def bounds(i: int) -> list:
    n = N_VARS
    if i == 1:
        return [(0.0, 1.0)] * n
    if i >= 8:
        w = 4.0 if i == 8 else 2.0
        return [(0.0, 1.0)] * 2 + [(-w, w)] * (n - 2)
    w = 1.0 if i == 2 else 2.0
    return [(0.0, 1.0)] + [(-w, w)] * (n - 1)


def index_sets(i: int) -> list:
    n = N_VARS
    if i >= 8:
        return [np.array([j for j in range(3, n + 1) if (j - k) % 3 == 0]) for k in (1, 2, 0)]
    return [np.arange(3, n + 1, 2), np.arange(2, n + 1, 2)]


def optimum(i: int, X: np.ndarray) -> np.ndarray:
    """The x_j, j = 2 .. n (CF8-CF10: 3 .. n), at which y_j = 0, given x_1
    (and x_2)."""
    X = np.asarray(X, float)
    n = N_VARS
    x1 = X[:, 0][:, None]
    if i >= 8:
        J = np.arange(3, n + 1)
        return 2.0 * X[:, 1][:, None] * np.sin(2.0 * np.pi * x1 + J * np.pi / n)
    J = np.arange(2, n + 1)
    odd = J % 2 == 1
    if i == 1:
        return x1 ** (0.5 * (1.0 + 3.0 * (J - 2) / (n - 2)))
    if i in (3, 4):
        return np.sin(6.0 * np.pi * x1 + J * np.pi / n)
    if i == 2 or i == 7:
        return np.where(odd if i == 2 else ~odd, np.sin(6.0 * np.pi * x1 + J * np.pi / n),
                        np.cos(6.0 * np.pi * x1 + J * np.pi / n))
    return np.where(odd, 0.8 * x1 * np.cos(6.0 * np.pi * x1 + J * np.pi / n),          # CF5, CF6
                    0.8 * x1 * np.sin(6.0 * np.pi * x1 + J * np.pi / n))


def _h2(t):
    """CF4's and CF5's h_2."""
    return np.where(t < THRESHOLD, np.abs(t), 0.125 + (t - 1.0) * (t - 1.0))


def cf_rows(i: int, X):
    """(F, G) of the rows of X for CF<i>: the objectives and g = -c."""
    n = N_VARS
    X = np.ascontiguousarray(X, dtype=float).reshape(-1, n)
    first = 3 if i >= 8 else 2
    Y = X[:, first - 1:] - optimum(i, X)
    sets = index_sets(i)
    x1 = X[:, 0]
    if i >= 8:
        S = []
        for Js in sets:
            Yc = Y[:, Js - first]
            T = 4.0 * Yc * Yc - np.cos(8.0 * np.pi * Yc) + 1.0 if i == 10 else Yc * Yc
            S.append(2.0 / len(Js) * np.sum(T, axis=1))
        x2 = X[:, 1]
        c1, c2 = np.cos(0.5 * x1 * np.pi), np.cos(0.5 * x2 * np.pi)
        s1, s2 = np.sin(0.5 * x1 * np.pi), np.sin(0.5 * x2 * np.pi)
        f1, f2, f3 = c1 * c2 + S[0], c1 * s2 + S[1], s1 + S[2]
        N, a = PARAMS[i]
        with np.errstate(divide="ignore", invalid="ignore"):
            r = 1.0 - f3 * f3
            s = np.sin(N * np.pi * ((f1 * f1 - f2 * f2) / r + 1.0))
            c = (f1 * f1 + f2 * f2) / r - a * (np.abs(s) if i == 8 else s) - 1.0
        return np.column_stack([f1, f2, f3]), -c[:, None]
    if i in (1, 2, 3):
        S = []
        for Js in sets:
            Yc = Y[:, Js - first]
            if i == 3:
                sq = np.sum(Yc * Yc, axis=1)
                pr = np.prod(np.cos(20.0 * Yc * np.pi / np.sqrt(Js)), axis=1)
                S.append(2.0 / len(Js) * (4.0 * sq - 2.0 * pr + 2.0))
            else:
                S.append(2.0 / len(Js) * np.sum(Yc * Yc, axis=1))
        N, a = PARAMS[i]
        if i == 1:
            f1, f2 = x1 + S[0], 1.0 - x1 + S[1]
            c = f1 + f2 - a * np.abs(np.sin(N * np.pi * (f1 - f2 + 1.0))) - 1.0
        elif i == 2:
            f1, f2 = x1 + S[0], 1.0 - np.sqrt(x1) + S[1]
            q = np.sqrt(f1)
            t = f2 + q - a * np.sin(N * np.pi * (q - f2 + 1.0)) - 1.0
            c = t / (1.0 + np.exp(4.0 * np.abs(t)))
        else:
            f1, f2 = x1 + S[0], 1.0 - x1 * x1 + S[1]
            c = f2 + f1 * f1 - a * np.sin(N * np.pi * (f1 * f1 - f2 + 1.0)) - 1.0
        return np.column_stack([f1, f2]), -c[:, None]
    # CF4-CF7: f1 = x1 + sum over J1 of h_j(y_j), f2 = (1 - x1) or (1 - x1)^2 plus J2's
    H = np.empty_like(Y)
    J = np.arange(2, n + 1)
    for col, j in enumerate(J):
        y = Y[:, col]
        if i in (4, 5) and j == 2:
            H[:, col] = _h2(y)
        elif i == 4 or i == 6 or (i == 7 and j in (2, 4)):
            H[:, col] = y * y
        else:
            H[:, col] = 2.0 * y * y - np.cos(4.0 * np.pi * y) + 1.0
    s1, s2 = np.sum(H[:, sets[0] - 2], axis=1), np.sum(H[:, sets[1] - 2], axis=1)
    if i in (4, 5):
        F = np.column_stack([x1 + s1, 1.0 - x1 + s2])
        base = np.sin(6.0 * np.pi * x1 + 2.0 * np.pi / n)
        c = X[:, 1] - (base if i == 4 else 0.8 * x1 * base) - 0.5 * x1 + 0.25
        if i == 4:
            c = c / (1.0 + np.exp(4.0 * np.abs(c)))
        return F, -c[:, None]
    F = np.column_stack([x1 + s1, (1.0 - x1) * (1.0 - x1) + s2])
    u = 1.0 - x1
    a = 0.5 * u - u * u
    b = 0.25 * np.sqrt(u) - 0.5 * u
    k = 1.0 if i == 7 else 0.8 * x1
    c1 = X[:, 1] - k * np.sin(6.0 * np.pi * x1 + 2.0 * np.pi / n) - np.sign(a) * np.sqrt(np.abs(a))
    c2 = X[:, 3] - k * np.sin(6.0 * np.pi * x1 + 4.0 * np.pi / n) - np.sign(b) * np.sqrt(np.abs(b))
    return F, -np.column_stack([c1, c2])


def _parts(lo_hi, m: int) -> np.ndarray:
    return np.concatenate([np.linspace(lo, hi, m) for lo, hi in lo_hi])


def positions(i: int, n: int) -> np.ndarray:
    """x_1 (and x_2) of about n Pareto-optimal points (module docstring)."""
    if i in (1, 2, 3):
        N = PARAMS[i][0]
        if i == 1:
            return (np.arange(2 * N + 1) / (2.0 * N))[:, None]
        m = max(2, (n - 1) // N)
        k = np.arange(1, N + 1)
        lo, hi = (2 * k - 1) / (2.0 * N), 2 * k / (2.0 * N)
        lo, hi = (lo * lo, hi * hi) if i == 2 else (np.sqrt(lo), np.sqrt(hi))
        return np.concatenate([[0.0], _parts(zip(lo, hi), m)])[:, None]
    if i < 8:
        return np.linspace(0.0, 1.0, n)[:, None]
    N = PARAMS[i][0]
    q = max(2, int(round(np.sqrt(n))))
    if i == 8:
        r = np.arange(2 * N + 1) / (2.0 * N)
        q = max(2, int(round(n / (2.0 * N + 1.0))))
    else:
        m = max(2, (q - 1) // N)
        k = np.arange(1, N + 1)
        r = np.concatenate([[0.0], _parts(zip((2 * k - 1) / (2.0 * N), 2 * k / (2.0 * N)), m)])
    x2 = 2.0 / np.pi * np.arccos(np.sqrt(r))
    a, b = np.meshgrid(np.linspace(0.0, 1.0, q), x2)
    return np.column_stack([a.ravel(), b.ravel()])


def pareto_set(i: int, n: int) -> np.ndarray:
    P = positions(i, n)
    X = np.column_stack([P, optimum(i, P)])
    if i in (4, 5, 6, 7):
        x1 = P[:, 0]
        if i in (4, 5):
            X[:, 1] += np.where(x1 <= 0.5, 0.0, np.where(x1 <= 0.75, 0.5 * x1 - 0.25, 1.0))
        else:
            u = 1.0 - x1
            X[:, 1] += np.sqrt(np.maximum(0.0, 0.5 * u - u * u))
            X[:, 3] += np.sqrt(np.maximum(0.0, 0.25 * np.sqrt(u) - 0.5 * u))
    return X


def front(i: int, n: int) -> np.ndarray:
    """The objectives of pareto_set(i, n): the front, by the problem's own code."""
    return cf_rows(i, pareto_set(i, n))[0]


def name(i: int) -> str:
    return f"CF{i}-CEC2009"
