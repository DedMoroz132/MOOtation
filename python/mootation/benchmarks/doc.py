# SPDX-License-Identifier: Apache-2.0
"""DOC1-9: constrained problems with constraints in both the decision and the
objective space (task 7 of 2026-10-10, §8).

Z.-Z. Liu and Y. Wang, "Handling constrained multiobjective optimization
problems with constraints in both the decision and objective spaces", IEEE
Transactions on Evolutionary Computation 23(5):870-884, 2019,
doi:10.1109/TEVC.2019.2894743 (source doc_liu2019: the instances in its
supplementary file, S-I, read off the rendered pages). Each takes a problem of
the CEC 2006 suite as its decision part — g04, g19, g23, g09, g21, g07, g14,
g10, g18 — whose objective, shifted so that its least feasible value is 1,
becomes the distance function g of a ZDT- or DTLZ-like pair or triple, and
adds constraints on the objectives. The paper's names are DOC-1 .. DOC-9.

  1  f = (x1, g - sqrt(x1)); f1^2 + f2^2 >= 1; g04's six constraints.
  2  f = (x1, g - cbrt(x1)); sqrt(f1) + f2 >= 1 and f inside one of three
     circles of radius 0.15 centred on f2 = 1 - sqrt(f1); g19's five.
  3  f = (x1, g - x1); f1^2 + f2^2 >= 1 and |f1 - f2 - c| >= 0.1 for c = 0.5,
     0, -0.5; g23's two inequalities and four equalities.
  4  f = (x1, g - sqrt(x1)); f1 + f2 >= 1 and
     f1 + f2 - 1 - |sin(10 pi (f1 - f2 + 1))| >= 0; g09's four.
  5  as 4, and (f1 - 0.8)(f2 - 0.6) <= 0; g21's inequality and five
     equalities (log is the natural one: x6 in [6.3, 6.7] = ln[600, 800]).
  6  as 4 with the sine constraint times (f1 - 0.5); g07's eight.
  7  as 6, and |f1 - f2| >= 0.1; g14's three equalities.
  8  f = (x1 x2 g, x1 (1 - x2) g, (1 - x1) g); (f3 - 0.4)(f3 - 0.6) >= 0;
     g10's six.
  9  f = (cos a cos b g, cos a sin b g, sin a g), a = pi x1 / 2,
     b = pi x2 / 2; f1^2 + f2^2 >= 1; g18's thirteen.
f2 = g (1 - r(f1) / g) is written g - r(f1), the same value, defined where
g = 0 (g goes below 0 at infeasible points: g04's objective has no positive
floor on the box).

Two readings. The supplement prints DOC-7's sine constraint as
(f1 - 0.5)(f1 + f2 - |sin(10 pi (f1 - f2 + 1))| + 1) >= 0, whose second
factor is at least 1 wherever f1 + f2 >= 1, so that the constraint would be
f1 >= 0.5 alone; the front it states, f1 in [0, 0.45] and the points
f1 = i/20, i = 11..20, is that of DOC-6's form, "- 1" for "+ 1", which is
taken. And it prints DOC-9's f3 as sin(pi x2 / 2) g, under which f3 = 0
forces f2 = 0; the front it states, the quarter circle f2 = sqrt(1 - f1^2) at
f3 = 0, is that of the DTLZ2 triple, f3 = sin(pi x1 / 2) g, which is taken.

The equalities h are 4 of DOC-3, 5 of DOC-5 and 3 of DOC-7; constraints()
returns them as |h| - 1e-4 <= 0, the paper's Eq. 3 (eta = 1e-4), which is,
bit for bit, minimize's eq_constraints at eq_tol = 1e-4; equalities() gives
h itself, for a run with a moving tolerance (eq_tol_schedule). DOC-7's
x_i ln(x_i / sum x) is 0 at x_i = 0, its limit.

THE FRONTS are the supplement's (S-I), each the least value of g, 1, on the
part of the unconstrained front's neighbourhood the objective constraints
leave: 1 the arc f2 = sqrt(1 - f1^2); 2 the curve f2 = 1 - sqrt(f1) inside
the circles, its ends solved here, [0.0500, 0.2203], [0.3803, 0.6247],
[0.7439, 1] (the paper prints 0.2202, 0.3830 and 0.7440; f1 = 0.3830 lies
inside the second circle); 3 the arc of 1 outside the bands; 4 the 21
points f1 = i/20 of f1 + f2 = 1; 5 the 14 of them with f1 <= 0.4 or >= 0.8;
6 the segment f1 in [0, 0.5] and the points f1 = i/20, i = 11..20; 7 the
segment f1 in [0, 0.45] and the same points; 8 the simplex f1 + f2 + f3 = 1
without 0.4 < f3 < 0.6; 9 the quarter circle at f3 = 0.

THE C++ SUITE (task 17, supplement 3, stage 3a): with the extension built the
problems are computed by include/mootation/suite/doc.hpp (benchmarks/native.py); this
module stays the reference (BenchProblem.reference) and what runs without it.
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

EQ_TOL = 1e-4                                  # the paper's eta (Eq. 3)

BOUNDS = {
    1: [(0.0, 1.0), (78.0, 102.0), (33.0, 45.0)] + [(27.0, 45.0)] * 3,
    2: [(0.0, 1.0)] + [(0.0, 10.0)] * 15,
    3: [(0.0, 1.0), (0.0, 1.0), (0.0, 300.0), (0.0, 100.0), (0.0, 200.0), (0.0, 100.0),
        (0.0, 1.0), (0.0, 100.0), (0.0, 200.0), (0.0, 0.03)],
    4: [(0.0, 1.0)] + [(-10.0, 10.0)] * 7,
    5: [(0.0, 1.0), (0.0, 1000.0), (0.0, 40.0), (0.0, 40.0), (100.0, 300.0), (6.3, 6.7),
        (5.9, 6.4), (4.5, 6.25)],
    6: [(0.0, 1.0)] + [(-10.0, 10.0)] * 10,
    7: [(0.0, 1.0)] + [(0.0, 10.0)] * 10,
    8: [(0.0, 1.0), (0.0, 1.0), (500.0, 1000.0), (1000.0, 2000.0), (5000.0, 6000.0)]
       + [(100.0, 500.0)] * 5,
    9: [(0.0, 1.0), (0.0, 1.0)] + [(-1.0, 10.0)] * 9,
}
N_EQ = {3: 4, 5: 5, 7: 3}

# DOC-2's data, Table S-I (g19 of CEC 2006): A[i][j] = a_{i+1, j+1}, i < 10, j < 5
G19_A = np.array([[-16.0, 2.0, 0.0, 1.0, 0.0],
                  [0.0, -2.0, 0.0, 0.4, 2.0],
                  [-3.5, 0.0, 2.0, 0.0, 0.0],
                  [0.0, -2.0, 0.0, -4.0, -1.0],
                  [0.0, -9.0, -2.0, 1.0, -2.8],
                  [2.0, 0.0, -4.0, 0.0, 0.0],
                  [-1.0, -1.0, -1.0, -1.0, -1.0],
                  [-1.0, -2.0, -3.0, -2.0, -1.0],
                  [1.0, 2.0, 3.0, 4.0, 5.0],
                  [1.0, 1.0, 1.0, 1.0, 1.0]])
G19_C = np.array([[30.0, -20.0, -10.0, 32.0, -10.0],
                  [-20.0, 39.0, -6.0, -31.0, 32.0],
                  [-10.0, -6.0, 10.0, -6.0, -10.0],
                  [32.0, -31.0, -6.0, 39.0, -20.0],
                  [-10.0, 32.0, -10.0, -20.0, 30.0]])
G19_D = np.array([4.0, 8.0, 10.0, 6.0, 2.0])
G19_E = np.array([-15.0, -27.0, -36.0, -18.0, -12.0])
G19_B = np.array([-40.0, -2.0, -0.25, -4.0, -4.0, -1.0, -40.0, -60.0, 5.0, 1.0])
# DOC-7's c_1..c_10 (g14)
G14_C = np.array([-6.089, -17.164, -34.054, -5.914, -24.721, -14.986, -24.1, -10.708,
                  -26.662, -22.179])
# DOC-2's circles: centres on f2 = 1 - sqrt(f1), radius 0.15
CIRCLES = ((1.0 / 8.0, 1.0 - math.sqrt(2.0) / 4.0), (0.5, 1.0 - math.sqrt(2.0) / 2.0),
           (7.0 / 8.0, 1.0 - math.sqrt(14.0) / 4.0))


def n_obj(i: int) -> int:
    return 3 if i >= 8 else 2


def n_vars(i: int) -> int:
    return len(BOUNDS[i])


def name(i: int) -> str:
    return f"DOC{i}"


def _dot(a, b) -> float:
    """sum a_i b_i term by term from 0 (DOC2's products with g19's matrices)."""
    s = 0.0
    for u, v in zip(a, b):
        s = s + u * v
    return s


def _sine(f1: float, f2: float) -> float:
    """f1 + f2 - 1 - |sin(10 pi (f1 - f2 + 1))|, DOC-4 to 7's constraint."""
    return f1 + f2 - 1.0 - abs(math.sin(10.0 * math.pi * (f1 - f2 + 1.0)))


def objective_constraints(i: int, f) -> list:
    """The constraints on the objectives, <= 0 satisfied."""
    f1, f2 = f[0], f[1]
    if i in (1, 9):
        return [1.0 - (f1 ** 2 + f2 ** 2)]
    if i == 2:
        return [1.0 - math.sqrt(f1) - f2,
                min((f1 - p) ** 2 + (f2 - q) ** 2 - 0.0225 for p, q in CIRCLES)]
    if i == 3:
        d = f1 - f2
        return [1.0 - (f1 ** 2 + f2 ** 2), 0.1 - abs(d - 0.5), 0.1 - abs(d), 0.1 - abs(d + 0.5)]
    if i == 4:
        return [1.0 - f1 - f2, -_sine(f1, f2)]
    if i == 5:
        return [1.0 - f1 - f2, -_sine(f1, f2), (f1 - 0.8) * (f2 - 0.6)]
    if i == 6:
        return [1.0 - f1 - f2, -(f1 - 0.5) * _sine(f1, f2)]
    if i == 7:
        return [1.0 - f1 - f2, -(f1 - 0.5) * _sine(f1, f2), 0.1 - abs(f1 - f2)]
    if i == 8:
        return [-(f[2] - 0.4) * (f[2] - 0.6)]
    raise ValueError(f"DOC{i}: 1 to 9")


def evaluate(i: int, x) -> tuple:
    """(objectives, inequalities g <= 0, equalities h = 0) at x, the paper's
    constraints in the library's sign."""
    x = [float(v) for v in x]
    X = [0.0] + x                                # X[k] is the paper's x_k
    if i == 1:
        g = (5.3578547 * X[4] ** 2 + 0.8356891 * X[2] * X[6] + 37.293239 * X[2]
             - 10125.6023282166)
        f = [X[1], g - math.sqrt(X[1])]
        a = 0.0056858 * X[3] * X[6] + 0.0006262 * X[2] * X[5] - 0.0022053 * X[4] * X[6]
        b = 0.0071317 * X[3] * X[6] + 0.0029955 * X[2] * X[3] + 0.0021813 * X[4] ** 2
        c = 0.0047026 * X[4] * X[6] + 0.0012547 * X[2] * X[4] + 0.0019085 * X[4] * X[5]
        ineq = objective_constraints(1, f) + [
                85.334407 + a - 92.0, -85.334407 - a,
                80.51249 + b - 110.0, -80.51249 - b + 90.0,
                9.300961 + c - 25.0, -9.300961 - c + 20.0]
        return f, ineq, []
    if i == 2:
        y = X[12:17]                             # x12..x16
        z = X[2:12]                              # x2..x11
        # y C y, the sums term by term from 0 (_dot), as include/mootation/
        # suite/doc.hpp writes them (BLAS's @ adds in an order of its own)
        yC = [_dot(y, G19_C[:, k]) for k in range(5)]
        g = float(_dot(yC, y) + 2.0 * _dot(G19_D, [v ** 3 for v in y]) - _dot(G19_B, z)
                  - 31.6555929502)
        f = [X[1], g - X[1] ** (1.0 / 3.0)]
        ineq = objective_constraints(2, f)
        ineq += [float(-2.0 * _dot(G19_C[:, k], y) - 3.0 * G19_D[k] * (y[k] * y[k]) - G19_E[k]
                       + _dot(z, G19_A[:, k])) for k in range(5)]
        return f, ineq, []
    if i == 3:
        g = (-9.0 * X[6] - 15.0 * X[9] + 6.0 * X[2] + 16.0 * X[3] + 10.0 * (X[7] + X[8])
             + 401.0551)
        f = [X[1], g - X[1]]
        ineq = objective_constraints(3, f) + [
                X[10] * X[4] + 0.02 * X[7] - 0.025 * X[6],
                X[10] * X[5] + 0.02 * X[8] - 0.015 * X[9]]
        eq = [X[2] + X[3] - X[4] - X[5],
              0.03 * X[2] + 0.01 * X[3] - X[10] * (X[4] + X[5]),
              X[4] + X[7] - X[6],
              X[5] + X[8] - X[9]]
        return f, ineq, eq
    if i == 4:
        g = ((X[2] - 10.0) ** 2 + 5.0 * (X[3] - 12.0) ** 2 + X[4] ** 4 + 3.0 * (X[5] - 11.0) ** 2
             + 10.0 * X[6] ** 6 + 7.0 * X[7] ** 2 + X[8] ** 4 - 4.0 * X[7] * X[8]
             - 10.0 * X[7] - 8.0 * X[8] - 679.6300573745)
        f = [X[1], g - math.sqrt(X[1])]
        ineq = objective_constraints(4, f) + [
                -127.0 + 2.0 * X[2] ** 2 + 3.0 * X[3] ** 4 + X[4] + 4.0 * X[5] ** 2 + 5.0 * X[6],
                -282.0 + 7.0 * X[2] + 3.0 * X[3] + 10.0 * X[4] ** 2 + X[5] - X[6],
                -196.0 + 23.0 * X[2] + X[3] ** 2 + 6.0 * X[7] ** 2 - 8.0 * X[8],
                4.0 * X[2] ** 2 + X[3] ** 2 - 3.0 * X[2] * X[3] + 2.0 * X[4] ** 2
                + 5.0 * X[7] - 11.0 * X[8]]
        return f, ineq, []
    if i == 5:
        g = X[2] - 192.724510070035
        f = [X[1], g - math.sqrt(X[1])]
        ineq = objective_constraints(5, f) + [
                -X[2] + 35.0 * X[3] ** 0.6 + 35.0 * X[4] ** 0.6]
        eq = [-300.0 * X[4] + 7500.0 * X[6] - 7500.0 * X[7] - 25.0 * X[5] * X[6]
              + 25.0 * X[5] * X[7] + X[4] * X[5],
              100.0 * X[3] + 155.365 * X[5] + 2500.0 * X[8] - X[3] * X[5] - 25.0 * X[5] * X[8]
              - 15536.5,
              -X[6] + math.log(-X[5] + 900.0),
              -X[7] + math.log(X[5] + 300.0),
              -X[8] + math.log(-2.0 * X[5] + 700.0)]
        return f, ineq, eq
    if i == 6:
        g = (X[2] ** 2 + X[3] ** 2 + X[2] * X[3] - 14.0 * X[2] - 16.0 * X[3]
             + (X[4] - 10.0) ** 2 + 4.0 * (X[5] - 5.0) ** 2 + (X[6] - 3.0) ** 2
             + 2.0 * (X[7] - 1.0) ** 2 + 5.0 * X[8] ** 2 + 7.0 * (X[9] - 11.0) ** 2
             + 2.0 * (X[10] - 10.0) ** 2 + (X[11] - 7.0) ** 2 + 21.693790931900001)
        f = [X[1], g - math.sqrt(X[1])]
        ineq = objective_constraints(6, f) + [
                -105.0 + 4.0 * X[2] + 5.0 * X[3] - 3.0 * X[8] + 9.0 * X[9],
                10.0 * X[2] - 8.0 * X[3] - 17.0 * X[8] + 2.0 * X[9],
                -8.0 * X[2] + 2.0 * X[3] + 5.0 * X[10] - 2.0 * X[11] - 12.0,
                3.0 * (X[2] - 2.0) ** 2 + 4.0 * (X[3] - 3.0) ** 2 + 2.0 * X[4] ** 2
                - 7.0 * X[5] - 120.0,
                5.0 * X[2] ** 2 + 8.0 * X[3] + (X[4] - 6.0) ** 2 - 2.0 * X[5] - 40.0,
                X[2] ** 2 + 2.0 * (X[3] - 2.0) ** 2 - 2.0 * X[2] * X[3] + 14.0 * X[6]
                - 6.0 * X[7],
                0.5 * (X[2] - 8.0) ** 2 + 2.0 * (X[3] - 4.0) ** 2 + 3.0 * X[6] ** 2 - X[7] - 30.0,
                -3.0 * X[2] + 6.0 * X[3] + 12.0 * (X[10] - 8.0) ** 2 - 7.0 * X[11]]
        return f, ineq, []
    if i == 7:
        y = X[2:12]                              # x2..x11
        # term by term from 0: Python's sum of floats is compensated from 3.12
        s = 0.0
        for v in y:
            s = s + v
        g = 0.0
        for v, c in zip(y, G14_C):
            if v > 0.0:
                g = g + v * (float(c) + math.log(v / s))
        g = g + 48.7648884
        f = [X[1], g - math.sqrt(X[1])]
        ineq = objective_constraints(7, f)
        eq = [X[2] + 2.0 * X[3] + 2.0 * X[4] + X[7] + X[11] - 2.0,
              X[5] + 2.0 * X[6] + X[7] + X[8] - 1.0,
              X[4] + X[8] + X[9] + 2.0 * X[10] + X[11] - 1.0]
        return f, ineq, eq
    if i == 8:
        g = X[3] + X[4] + X[5] - 7048.2480205286
        f = [X[1] * X[2] * g, X[1] * (1.0 - X[2]) * g, (1.0 - X[1]) * g]
        ineq = objective_constraints(8, f) + [
                -1.0 + 0.0025 * (X[6] + X[8]),
                -1.0 + 0.0025 * (X[7] + X[9] - X[6]),
                -1.0 + 0.01 * (X[10] - X[7]),
                -X[3] * X[8] + 833.33252 * X[6] + 100.0 * X[3] - 83333.333,
                -X[4] * X[9] + 1250.0 * X[7] + X[4] * X[6] - 1250.0 * X[6],
                -X[5] * X[10] + 1250000.0 + X[5] * X[7] - 2500.0 * X[7]]
        return f, ineq, []
    if i == 9:
        g = (-0.5 * (X[3] * X[6] - X[4] * X[5] + X[5] * X[11] - X[7] * X[11] + X[7] * X[10]
                     - X[8] * X[9]) + 1.8660254038)
        a, b = 0.5 * math.pi * X[1], 0.5 * math.pi * X[2]
        f = [math.cos(a) * math.cos(b) * g, math.cos(a) * math.sin(b) * g, math.sin(a) * g]
        ineq = objective_constraints(9, f) + [
                X[5] ** 2 + X[6] ** 2 - 1.0,
                X[11] ** 2 - 1.0,
                X[7] ** 2 + X[8] ** 2 - 1.0,
                X[3] ** 2 + (X[4] - X[11]) ** 2 - 1.0,
                (X[3] - X[7]) ** 2 + (X[4] - X[8]) ** 2 - 1.0,
                (X[3] - X[9]) ** 2 + (X[4] - X[10]) ** 2 - 1.0,
                (X[5] - X[7]) ** 2 + (X[6] - X[8]) ** 2 - 1.0,
                (X[5] - X[9]) ** 2 + (X[6] - X[10]) ** 2 - 1.0,
                X[9] ** 2 + (X[10] - X[11]) ** 2 - 1.0,
                X[4] * X[5] - X[3] * X[6],
                -X[5] * X[11],
                X[7] * X[11],
                X[8] * X[9] - X[7] * X[10]]
        return f, ineq, []
    raise ValueError(f"DOC{i}: 1 to 9")


def constraints(i: int, x) -> list:
    """Every constraint as a value <= 0 satisfied: an equality as |h| - 1e-4
    (the paper's Eq. 3), the inequalities first."""
    _, ineq, eq = evaluate(i, x)
    return ineq + [abs(h) - EQ_TOL for h in eq]


def equalities(i: int, x) -> list:
    """The equalities' h, for minimize's eq_constraints."""
    return evaluate(i, x)[2]


def inequalities(i: int, x) -> list:
    return evaluate(i, x)[1]


def _circle_ends(p: float) -> tuple:
    """Where f2 = 1 - sqrt(f1) leaves DOC-2's circle centred on it at f1 = p:
    (f1 - p)^2 + (sqrt(f1) - sqrt(p))^2 = 0.0225, or the box's end, f1 in
    [0, 1]; by bisection to the last bit."""
    def inside(u):
        return (u - p) ** 2 + (math.sqrt(u) - math.sqrt(p)) ** 2 <= 0.0225
    ends = []
    for far in (0.0, 1.0):
        if inside(far):
            ends.append(far)
            continue
        lo, hi = p, far                          # lo inside, hi outside
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if mid in (lo, hi):
                break
            if inside(mid):
                lo = mid
            else:
                hi = mid
        ends.append(lo)
    return ends[0], ends[1]


def _bands_kept(f1, f2) -> np.ndarray:
    d = np.asarray(f1) - np.asarray(f2)
    return (np.abs(d - 0.5) >= 0.1) & (np.abs(d) >= 0.1) & (np.abs(d + 0.5) >= 0.1)


def _line_points(ks) -> np.ndarray:
    u = np.asarray(ks, float) / 20.0
    return np.column_stack([u, 1.0 - u])


@lru_cache(maxsize=None)
def front(i: int, n_points: int = 1000) -> np.ndarray:
    """The front (THE FRONTS above), thinned to about n_points; the discrete
    fronts of DOC4 and DOC5 whole."""
    from ..run.archive import dss_order
    if i == 4:
        return _line_points(range(21))
    if i == 5:
        return _line_points(list(range(9)) + list(range(16, 21)))
    if i == 1:
        u = np.linspace(0.0, 1.0, 20001)
        F = np.column_stack([u, np.sqrt(1.0 - u * u)])
    elif i == 2:
        parts = []
        for p, _ in CIRCLES:
            a, b = _circle_ends(p)
            u = np.linspace(a, b, 8001)
            parts.append(np.column_stack([u, 1.0 - np.sqrt(u)]))
        F = np.vstack(parts)
    elif i == 3:
        # the arc's pieces between the bands' edges, where f1 - f2 = d, that is
        # cos t - sin t = d: t = acos(d / sqrt 2) - pi / 4
        edges = [c + s for c in (-0.5, 0.0, 0.5) for s in (-0.1, 0.1)]
        cuts = sorted([0.0, 0.5 * math.pi]
                      + [math.acos(d / math.sqrt(2.0)) - 0.25 * math.pi for d in edges])
        pieces = [np.linspace(a, b, 5001) for a, b in zip(cuts, cuts[1:])
                  if _bands_kept(math.cos(0.5 * (a + b)), math.sin(0.5 * (a + b)))]
        t = np.concatenate(pieces)
        F = np.column_stack([np.cos(t), np.sin(t)])
    elif i in (6, 7):
        end = 0.5 if i == 6 else 0.45
        u = np.linspace(0.0, end, 20001)
        F = np.vstack([np.column_stack([u, 1.0 - u]), _line_points(range(11, 21))])
    elif i == 8:
        H = 60                                   # f3 = 0.4 and 0.6 on the grid
        rows = [(a / H, b / H, (H - a - b) / H) for a in range(H + 1) for b in range(H + 1 - a)]
        F = np.asarray(rows, float)
        F = F[(F[:, 2] <= 0.4 + 1e-12) | (F[:, 2] >= 0.6 - 1e-12)]
    elif i == 9:
        t = np.linspace(0.0, 0.5 * math.pi, 20001)
        F = np.column_stack([np.cos(t), np.sin(t), np.zeros_like(t)])
    else:
        raise ValueError(f"DOC{i}: 1 to 9")
    if len(F) <= n_points:
        return F
    return F[dss_order(F, n_points, ideal=F.min(axis=0), nadir=F.max(axis=0))]
