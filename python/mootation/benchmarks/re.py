# SPDX-License-Identifier: Apache-2.0
"""The RE and CRE real-world problems of Tanabe and Ishibuchi.

R. Tanabe, H. Ishibuchi, "An easy-to-use real-world multi-objective
optimization problem suite", Applied Soft Computing 89:106078, 2020,
doi:10.1016/j.asoc.2020.106078 (task 6 of 2026-10-09, item 6). Ported from the
authors' Python version, github.com/ryojitanabe/reproblems (MIT License,
Copyright (c) 2020 Ryoji Tanabe; the license is in _re/LICENSE.txt),
reproblem_python_ver/reproblem.py at commit 7876b4e (2026), formula for
formula: the expressions, constants and the order of the operations are the
file's, so that the values agree with it (the tests hold five points of every
problem to 1e-12). Names are the repository's: RE21 is the paper's RE2-4-1.

  RE21-25, RE31-37, RE41, RE42, RE61, RE91   16 problems without constraints:
      where the engineering problem has constraints, the last objective is
      the sum of their violations (the paper's reformulation)
  CRE21-25, CRE31, CRE32, CRE51              8 constrained problems, the
      original versions of RE31, RE32, RE33, RE35, RE36, RE41, RE42, RE61:
      the same objectives without the violation, and the constraints

Integer and discrete variables are rounded inside the function, as in the
original (RE22's x1 and RE25's x3 snapped to the nearest value of a list,
RE23's x1, x2 to multiples of 0.0625, RE25's x1, RE35's x3, RE36's four
variables rounded); the variables are declared continuous.

THE SIGN. reproblems returns a CRE's constraints as violations, max(-g, 0) of
its g >= 0: zero when satisfied and positive by how much it is not — the
library's convention (<= 0 satisfied) as it stands, so they pass through.

THE FRONTS are approximations, not the true fronts (front_exact = False):
  RE    the repository's approximated_Pareto_fronts/reference_points_RE*.dat,
        500 M points selected by the authors from every non-dominated point
        their optimizers found (28 for RE36, all there were), shipped in _re/
        as they are, and used whole whatever size is asked for; the frame is
        the repository's ideal_nadir_points (the issue #13 fix of 2026-08
        included: RE21's two files had been swapped)
  CRE   the repository has none. Taken here from the RE counterpart's file:
        its points with no violation (last objective 0), without that
        objective, the non-dominated ones — a CRE Pareto point is feasible
        and nothing infeasible can dominate it in the RE form, so these are
        CRE Pareto points as far as the file goes. A construction of this
        library, and thin: 2 points for CRE21, 17 to 1 258 for the others.
        Its frame is the counterpart's ideal and nadir without the last
        objective (they span the infeasible trade-offs too).
IGD+ and eps on these problems measure the distance to an approximation.

KNOWN DEFECTS, kept so that the values match the published fronts (the
repository's change log and issues):
  * RE22: the list of x1's discrete values reads "3.08, 3,10, 3.16" in all
    four language versions — a comma for a point, so the list holds 3 and 10
    where 3.10 was meant. Not corrected: a corrected RE22 would need a new
    reference front.
  * RE33 (and CRE23): issue #9 (2023, open) — the range of x4 is given as
    [11, 20] and should be [2, 11], the author agrees. Kept as [11, 20].
  * RE22: the violation sign (issue #5) — max(-g, 0) where some write
    max(g, 0); at x2 = 0, RE22's first constraint is -inf. Kept.

THE CORRECTED COPIES (task 7 of 2026-10-10, §9), beside the originals, which
stay as they are for the published fronts:
  RE22c  RE22 with x1's list as meant: 3.10 where the original has "3,10"
         (76 values, not 77). The list is RE22's own; its source, Amir and
         Hasegawa's reinforced concrete beam (J. Struct. Eng. 115(3), 1989),
         is not openly available and was not checked.
  RE33c  RE33 with x4 in [2, 11] instead of [11, 20], after reproblems'
         issue #9 (opened 2023-02-25, still open; the author confirmed the
         error there on 2023-04-11, and the code still has [11, 20]). The
         disc brake's source (Ray and Liew, Eng. Optim. 34(2), 2002) is not
         openly available and was not checked.
  Their reference fronts are built here (re_fronts.py: an exhaustive grid,
  then local refinement, thinned by DSS to 500 M points), shipped in
  _re/corrected/ with the fronts' own ideal and nadir points; front_exact is
  False, as for every RE problem.
  * Fixed upstream before this port, so already in the code taken here: RE21
    and RE25's E = 2 * 10e5 and G = 11.5 * 10e+6 (issue #1: now 2e5 and
    11.5e6 — consistent with the original definitions but not with the
    numbers of the paper); RE21's (2.0 / x2) for (2.0 / x1) (issues #2, #3);
    RE21's lower bounds in the Matlab code (#4); RE24's upper bound of x2 in
    the C code; RE42's and CRE32's 50000 - DWT for 500000 - DWT (#8, after
    Parsons and Scott); RE21's swapped ideal and nadir files (#13).
  * RE91 is stochastic: four of its variables are normal draws at every
    evaluation (np.random.normal in the original, the global generator). The
    formulas are the file's; the draws come from the run's seed and the
    evaluation's number (make_evaluator, as the noisy problems'), and without
    a seed from x itself — so that a campaign repeats a run exactly.
"""

from __future__ import annotations

import functools
import math
import struct
from pathlib import Path
from typing import List

import numpy as np

from .uninformative import _MASK, _splitmix64, _unit

DATA = Path(__file__).with_name("_re")
_DOMAIN = 5 << 60                        # RE91's draws, apart from the other streams


# ── bounds ───────────────────────────────────────────────────────────────────
def _re21_bounds():
    F, sigma = 10.0, 10.0
    tmp_val = F / sigma
    ub = np.full(4, 3 * tmp_val)
    lb = np.array([tmp_val, np.sqrt(2.0) * tmp_val, np.sqrt(2.0) * tmp_val, tmp_val])
    return lb, ub


_SHIP = ([150.0, 20.0, 13.0, 10.0, 14.0, 0.63], [274.32, 32.31, 25.0, 11.71, 18.0, 0.75])
_CAR = ([0.5, 0.45, 0.5, 0.5, 0.875, 0.4, 0.4], [1.5, 1.35, 1.5, 1.5, 2.625, 1.2, 1.2])
_TRUSS = ([0.00001, 0.00001, 1.0], [100.0, 100.0, 3.0])
_WELD = ([0.125, 0.1, 0.1, 0.125], [5.0, 10.0, 10.0, 5.0])
_BRAKE = ([55, 75, 1000, 11], [80, 110, 3000, 20])
RE33C_BOUNDS = ([55, 75, 1000, 2], [80, 110, 3000, 11])     # RE33c: issue #9
_REDUCER = ([2.6, 0.7, 17, 7.3, 7.3, 2.9, 5.0], [3.6, 0.8, 28, 8.3, 8.3, 3.9, 5.5])
_GEAR = ([12] * 4, [60] * 4)
_WATER = ([0.01, 0.01, 0.01], [0.45, 0.10, 0.10])

RE22_VALUES = np.array([0.20, 0.31, 0.40, 0.44, 0.60, 0.62, 0.79, 0.80, 0.88, 0.93, 1.0, 1.20,
                        1.24, 1.32, 1.40, 1.55, 1.58, 1.60, 1.76, 1.80, 1.86, 2.0, 2.17, 2.20,
                        2.37, 2.40, 2.48, 2.60, 2.64, 2.79, 2.80, 3.0, 3.08, 3,10, 3.16, 3.41,  # noqa: E231 (the original's "3,10")
                        3.52, 3.60, 3.72, 3.95, 3.96, 4.0, 4.03, 4.20, 4.34, 4.40, 4.65, 4.74,
                        4.80, 4.84, 5.0, 5.28, 5.40, 5.53, 5.72, 6.0, 6.16, 6.32, 6.60, 7.11,
                        7.20, 7.80, 7.90, 8.0, 8.40, 8.69, 9.0, 9.48, 10.27, 11.0, 11.06, 11.85,
                        12.0, 13.0, 14.0, 15.0])
# RE22c: the list as meant, 3.10 for the original's "3,10"
RE22C_VALUES = np.array([v for v in RE22_VALUES[:33]] + [3.10] + [v for v in RE22_VALUES[35:]])
RE25_VALUES = np.array([0.009, 0.0095, 0.0104, 0.0118, 0.0128, 0.0132, 0.014, 0.015, 0.0162,
                        0.0173, 0.018, 0.02, 0.023, 0.025, 0.028, 0.032, 0.035, 0.041, 0.047,
                        0.054, 0.063, 0.072, 0.08, 0.092, 0.105, 0.12, 0.135, 0.148, 0.162,
                        0.177, 0.192, 0.207, 0.225, 0.244, 0.263, 0.283, 0.307, 0.331, 0.362,
                        0.394, 0.4375, 0.5])


def _violations(g):
    return np.where(g < 0, -g, 0)


# ── the shared engineering models ────────────────────────────────────────────
def _truss(x):
    """RE31 / CRE21: two bar truss — the two objectives and g."""
    x1, x2, x3 = x[0], x[1], x[2]
    f = np.zeros(2)
    g = np.zeros(3)
    f[0] = x1 * np.sqrt(16.0 + (x3 * x3)) + x2 * np.sqrt(1.0 + x3 * x3)
    f[1] = (20.0 * np.sqrt(16.0 + (x3 * x3))) / (x1 * x3)
    g[0] = 0.1 - f[0]
    g[1] = 100000.0 - f[1]
    g[2] = 100000 - ((80.0 * np.sqrt(1.0 + x3 * x3)) / (x3 * x2))
    return f, _violations(g)


def _weld(x):
    """RE32 / CRE22: welded beam."""
    x1, x2, x3, x4 = x[0], x[1], x[2], x[3]
    f = np.zeros(2)
    g = np.zeros(4)
    P = 6000
    L = 14
    E = 30 * 1e6
    G = 12 * 1e6
    tauMax = 13600
    sigmaMax = 30000
    f[0] = (1.10471 * x1 * x1 * x2) + (0.04811 * x3 * x4) * (14.0 + x2)
    f[1] = (4 * P * L * L * L) / (E * x4 * x3 * x3 * x3)
    M = P * (L + (x2 / 2))
    tmpVar = ((x2 * x2) / 4.0) + np.power((x1 + x3) / 2.0, 2)
    R = np.sqrt(tmpVar)
    tmpVar = ((x2 * x2) / 12.0) + np.power((x1 + x3) / 2.0, 2)
    J = 2 * np.sqrt(2) * x1 * x2 * tmpVar
    tauDashDash = (M * R) / J
    tauDash = P / (np.sqrt(2) * x1 * x2)
    tmpVar = tauDash * tauDash + ((2 * tauDash * tauDashDash * x2) / (2 * R)) + (tauDashDash * tauDashDash)
    tau = np.sqrt(tmpVar)
    sigma = (6 * P * L) / (x4 * x3 * x3)
    tmpVar = 4.013 * E * np.sqrt((x3 * x3 * x4 * x4 * x4 * x4 * x4 * x4) / 36.0) / (L * L)
    tmpVar2 = (x3 / (2 * L)) * np.sqrt(E / (4 * G))
    PC = tmpVar * (1 - tmpVar2)
    g[0] = tauMax - tau
    g[1] = sigmaMax - sigma
    g[2] = x4 - x1
    g[3] = PC - P
    return f, _violations(g)


def _brake(x):
    """RE33 / CRE23: disc brake."""
    x1, x2, x3, x4 = x[0], x[1], x[2], x[3]
    f = np.zeros(2)
    g = np.zeros(4)
    f[0] = 4.9 * 1e-5 * (x2 * x2 - x1 * x1) * (x4 - 1.0)
    f[1] = ((9.82 * 1e6) * (x2 * x2 - x1 * x1)) / (x3 * x4 * (x2 * x2 * x2 - x1 * x1 * x1))
    g[0] = (x2 - x1) - 20.0
    g[1] = 0.4 - (x3 / (3.14 * (x2 * x2 - x1 * x1)))
    g[2] = 1.0 - (2.22 * 1e-3 * x3 * (x2 * x2 * x2 - x1 * x1 * x1)) / np.power((x2 * x2 - x1 * x1), 2)
    g[3] = (2.66 * 1e-2 * x3 * x4 * (x2 * x2 * x2 - x1 * x1 * x1)) / (x2 * x2 - x1 * x1) - 900.0
    return f, _violations(g)


def _reducer(x):
    """RE35 / CRE24: speed reducer (x3 an integer)."""
    x1 = x[0]
    x2 = x[1]
    x3 = np.round(x[2])
    x4 = x[3]
    x5 = x[4]
    x6 = x[5]
    x7 = x[6]
    f = np.zeros(2)
    g = np.zeros(11)
    f[0] = 0.7854 * x1 * (x2 * x2) * (((10.0 * x3 * x3) / 3.0) + (14.933 * x3) - 43.0934) - 1.508 * x1 * (x6 * x6 + x7 * x7) + 7.477 * (x6 * x6 * x6 + x7 * x7 * x7) + 0.7854 * (x4 * x6 * x6 + x5 * x7 * x7)
    tmpVar = np.power((745.0 * x4) / (x2 * x3), 2.0) + 1.69 * 1e7
    f[1] = np.sqrt(tmpVar) / (0.1 * x6 * x6 * x6)
    g[0] = -(1.0 / (x1 * x2 * x2 * x3)) + 1.0 / 27.0
    g[1] = -(1.0 / (x1 * x2 * x2 * x3 * x3)) + 1.0 / 397.5
    g[2] = -(x4 * x4 * x4) / (x2 * x3 * x6 * x6 * x6 * x6) + 1.0 / 1.93
    g[3] = -(x5 * x5 * x5) / (x2 * x3 * x7 * x7 * x7 * x7) + 1.0 / 1.93
    g[4] = -(x2 * x3) + 40.0
    g[5] = -(x1 / x2) + 12.0
    g[6] = -5.0 + (x1 / x2)
    g[7] = -1.9 + x4 - 1.5 * x6
    g[8] = -1.9 + x5 - 1.1 * x7
    g[9] = -f[1] + 1300.0
    tmpVar = np.power((745.0 * x5) / (x2 * x3), 2.0) + 1.575 * 1e8
    g[10] = -np.sqrt(tmpVar) / (0.1 * x7 * x7 * x7) + 1100.0
    return f, _violations(g)


def _gear(x):
    """RE36 / CRE25: gear train (four integers)."""
    x1 = np.round(x[0])
    x2 = np.round(x[1])
    x3 = np.round(x[2])
    x4 = np.round(x[3])
    f = np.zeros(2)
    g = np.zeros(1)
    f[0] = np.abs(6.931 - ((x3 / x1) * (x4 / x2)))
    f[1] = max([x1, x2, x3, x4])
    g[0] = 0.5 - (f[0] / 6.931)
    return f, _violations(g)


def _car(x):
    """RE41 / CRE31: car side impact."""
    x1, x2, x3, x4, x5, x6, x7 = x[0], x[1], x[2], x[3], x[4], x[5], x[6]
    f = np.zeros(3)
    g = np.zeros(10)
    f[0] = 1.98 + 4.9 * x1 + 6.67 * x2 + 6.98 * x3 + 4.01 * x4 + 1.78 * x5 + 0.00001 * x6 + 2.73 * x7
    f[1] = 4.72 - 0.5 * x4 - 0.19 * x2 * x3
    Vmbp = 10.58 - 0.674 * x1 * x2 - 0.67275 * x2
    Vfd = 16.45 - 0.489 * x3 * x7 - 0.843 * x5 * x6
    f[2] = 0.5 * (Vmbp + Vfd)
    g[0] = 1 -(1.16 - 0.3717 * x2 * x4 - 0.0092928 * x3)
    g[1] = 0.32 -(0.261 - 0.0159 * x1 * x2 - 0.06486 * x1 -  0.019 * x2 * x7 + 0.0144 * x3 * x5 + 0.0154464 * x6)
    g[2] = 0.32 -(0.214 + 0.00817 * x5 - 0.045195 * x1 - 0.0135168 * x1 + 0.03099 * x2 * x6 - 0.018 * x2 * x7  + 0.007176 * x3 + 0.023232 * x3 - 0.00364 * x5 * x6 - 0.018 * x2 * x2)
    g[3] = 0.32 -(0.74 - 0.61 * x2 - 0.031296 * x3 - 0.031872 * x7 + 0.227 * x2 * x2)
    g[4] = 32 -(28.98 + 3.818 * x3 - 4.2 * x1 * x2 + 1.27296 * x6 - 2.68065 * x7)
    g[5] = 32 -(33.86 + 2.95 * x3 - 5.057 * x1 * x2 - 3.795 * x2 - 3.4431 * x7 + 1.45728)
    g[6] = 32 -(46.36 - 9.9 * x2 - 4.4505 * x1)
    g[7] = 4 - f[1]
    g[8] = 9.9 - Vmbp
    g[9] = 15.7 - Vfd
    return f, _violations(g)


def _ship(x):
    """RE42 / CRE32: conceptual marine design."""
    x_L, x_B, x_D, x_T, x_Vk, x_CB = x[0], x[1], x[2], x[3], x[4], x[5]
    f = np.zeros(3)
    constraintFuncs = np.zeros(9)
    displacement = 1.025 * x_L * x_B * x_T * x_CB
    V = 0.5144 * x_Vk
    g = 9.8065
    Fn = V / np.power(g * x_L, 0.5)
    a = (4977.06 * x_CB * x_CB) - (8105.61 * x_CB) + 4456.51
    b = (-10847.2 * x_CB * x_CB) + (12817.0 * x_CB) - 6960.32
    power = (np.power(displacement, 2.0/3.0) * np.power(x_Vk, 3.0)) / (a + (b * Fn))
    outfit_weight = 1.0 * np.power(x_L , 0.8) * np.power(x_B , 0.6) * np.power(x_D, 0.3) * np.power(x_CB, 0.1)
    steel_weight = 0.034 * np.power(x_L ,1.7) * np.power(x_B ,0.7) * np.power(x_D ,0.4) * np.power(x_CB ,0.5)
    machinery_weight = 0.17 * np.power(power, 0.9)
    light_ship_weight = steel_weight + outfit_weight + machinery_weight
    ship_cost = 1.3 * ((2000.0 * np.power(steel_weight, 0.85))  + (3500.0 * outfit_weight) + (2400.0 * np.power(power, 0.8)))
    capital_costs = 0.2 * ship_cost
    DWT = displacement - light_ship_weight
    running_costs = 40000.0 * np.power(DWT, 0.3)
    round_trip_miles = 5000.0
    sea_days = (round_trip_miles / 24.0) * x_Vk
    handling_rate = 8000.0
    daily_consumption = ((0.19 * power * 24.0) / 1000.0) + 0.2
    fuel_price = 100.0
    fuel_cost = 1.05 * daily_consumption * sea_days * fuel_price
    port_cost = 6.3 * np.power(DWT, 0.8)
    fuel_carried = daily_consumption * (sea_days + 5.0)
    miscellaneous_DWT = 2.0 * np.power(DWT, 0.5)
    cargo_DWT = DWT - fuel_carried - miscellaneous_DWT
    port_days = 2.0 * ((cargo_DWT / handling_rate) + 0.5)
    RTPA = 350.0 / (sea_days + port_days)
    voyage_costs = (fuel_cost + port_cost) * RTPA
    annual_costs = capital_costs + running_costs + voyage_costs
    annual_cargo = cargo_DWT * RTPA
    f[0] = annual_costs / annual_cargo
    f[1] = light_ship_weight
    f[2] = -annual_cargo                       # f_2 is dealt as a minimization problem
    constraintFuncs[0] = (x_L / x_B) - 6.0
    constraintFuncs[1] = -(x_L / x_D) + 15.0
    constraintFuncs[2] = -(x_L / x_T) + 19.0
    constraintFuncs[3] = 0.45 * np.power(DWT, 0.31) - x_T
    constraintFuncs[4] = 0.7 * x_D + 0.7 - x_T
    constraintFuncs[5] = 500000.0 - DWT
    constraintFuncs[6] = DWT - 3000.0
    constraintFuncs[7] = 0.32 - Fn
    KB = 0.53 * x_T
    BMT = ((0.085 * x_CB - 0.002) * x_B * x_B) / (x_T * x_CB)
    KG = 1.0 + 0.52 * x_D
    constraintFuncs[8] = (KB + BMT - KG) - (0.07 * x_B)
    return f, _violations(constraintFuncs)


def _water(x):
    """RE61 / CRE51: water resource planning."""
    f = np.zeros(5)
    g = np.zeros(7)
    f[0] = 106780.37 * (x[1] + x[2]) + 61704.67
    f[1] = 3000 * x[0]
    f[2] = 305700 * 2289 * x[1] / np.power(0.06*2289, 0.65)
    f[3] = 250 * 2289 * np.exp(-39.75*x[1]+9.9*x[2]+2.74)
    f[4] = 25 * (1.39 /(x[0]*x[1]) + 4940*x[2] -80)
    g[0] = 1 - (0.00139/(x[0]*x[1])+4.94*x[2]-0.08)
    g[1] = 1 - (0.000306/(x[0]*x[1])+1.082*x[2]-0.0986)
    g[2] = 50000 - (12.307/(x[0]*x[1]) + 49408.24*x[2]+4051.02)
    g[3] = 16000 - (2.098/(x[0]*x[1])+8046.33*x[2]-696.71)
    g[4] = 10000 - (2.138/(x[0]*x[1])+7883.39*x[2]-705.04)
    g[5] = 2000 - (0.417*x[0]*x[1] + 1721.26*x[2]-136.54)
    g[6] = 550 - (0.164/(x[0]*x[1])+631.13*x[2]-54.48)
    return f, _violations(g)


def _reformulated(model):
    """An RE problem from a CRE model: the objectives, then the sum of the
    violations as the last one, summed in the original's order."""
    def evaluate(x):
        f, g = model(x)
        s = g[0]
        for v in g[1:]:
            s = s + v
        return np.append(f, s)
    return evaluate


# ── the RE problems that are not reformulations of a CRE ─────────────────────
def re21(x):
    f = np.zeros(2)
    x1, x2, x3, x4 = x[0], x[1], x[2], x[3]
    F = 10.0
    E = 2.0 * 1e5
    L = 200.0
    f[0] = L * ((2 * x1) + np.sqrt(2.0) * x2 + np.sqrt(x3) + x4)
    f[1] = ((F * L) / E) * ((2.0 / x1) + (2.0 * np.sqrt(2.0) / x2) - (2.0 * np.sqrt(2.0) / x3) + (2.0 / x4))
    return f


def re22(x, values=RE22_VALUES):
    f = np.zeros(2)
    g = np.zeros(2)
    idx = np.abs(np.asarray(values) - x[0]).argmin()
    x1 = values[idx]
    x2 = x[1]
    x3 = x[2]
    f[0] = (29.4 * x1) + (0.6 * x2 * x3)
    g[0] = (x1 * x3) - 7.735 * ((x1 * x1) / x2) - 180.0
    g[1] = 4.0 - (x3 / x2)
    g = _violations(g)
    f[1] = g[0] + g[1]
    return f


def re23(x):
    f = np.zeros(2)
    g = np.zeros(3)
    x1 = 0.0625 * int(np.round(x[0]))
    x2 = 0.0625 * int(np.round(x[1]))
    x3 = x[2]
    x4 = x[3]
    f[0] = (0.6224 * x1 * x3* x4) + (1.7781 * x2 * x3 * x3) + (3.1661 * x1 * x1 * x4) + (19.84 * x1 * x1 * x3)
    g[0] = x1 - (0.0193 * x3)
    g[1] = x2 - (0.00954 * x3)
    g[2] = (np.pi * x3 * x3 * x4) + ((4.0/3.0) * (np.pi * x3 * x3 * x3)) - 1296000
    g = _violations(g)
    f[1] = g[0] + g[1] + g[2]
    return f


def re24(x):
    f = np.zeros(2)
    g = np.zeros(4)
    x1 = x[0]
    x2 = x[1]
    f[0] = x1 + (120 * x2)
    E = 700000
    sigma_b_max = 700
    tau_max = 450
    delta_max = 1.5
    sigma_k = (E * x1 * x1) / 100
    sigma_b = 4500 / (x1 * x2)
    tau = 1800 / x2
    delta = (56.2 * 10000) / (E * x1 * x2 * x2)
    g[0] = 1 - (sigma_b / sigma_b_max)
    g[1] = 1 - (tau / tau_max)
    g[2] = 1 - (delta / delta_max)
    g[3] = 1 - (sigma_b / sigma_k)
    g = _violations(g)
    f[1] = g[0] + g[1] + g[2] + g[3]
    return f


def re25(x):
    f = np.zeros(2)
    g = np.zeros(6)
    x1 = np.round(x[0])
    x2 = x[1]
    idx = np.abs(np.asarray(RE25_VALUES) - x[2]).argmin()
    x3 = RE25_VALUES[idx]
    f[0] = (np.pi * np.pi * x2 * x3 * x3 * (x1 + 2)) / 4.0
    Cf = ((4.0 * (x2 / x3) - 1) / (4.0 * (x2 / x3) - 4)) + (0.615 * x3 / x2)
    Fmax = 1000.0
    S = 189000.0
    G = 11.5 * 1e+6
    K = (G * x3 * x3 * x3 * x3) / (8 * x1 * x2 * x2 * x2)
    lmax = 14.0
    lf = (Fmax / K) + 1.05 * (x1 + 2) * x3
    Fp = 300.0
    sigmaP = Fp / K
    sigmaPM = 6
    sigmaW = 1.25
    g[0] = -((8 * Cf * Fmax * x2) / (np.pi * x3 * x3 * x3)) + S
    g[1] = -lf + lmax
    g[2] = -3 + (x2 / x3)
    g[3] = -sigmaP + sigmaPM
    g[4] = -sigmaP - ((Fmax - Fp) / K) - 1.05 * (x1 + 2) * x3 + lf
    g[5] = sigmaW - ((Fmax - Fp) / K)
    g = _violations(g)
    f[1] = g[0] + g[1] + g[2] + g[3] + g[4] + g[5]
    return f


def re34(x):
    f = np.zeros(3)
    x1, x2, x3, x4, x5 = x[0], x[1], x[2], x[3], x[4]
    f[0] = 1640.2823 + (2.3573285 * x1) + (2.3220035 * x2) + (4.5688768 * x3) + (7.7213633 * x4) + (4.4559504 * x5)
    f[1] = 6.5856 + (1.15 * x1) - (1.0427 * x2) + (0.9738 * x3) + (0.8364 * x4) - (0.3695 * x1 * x4) + (0.0861 * x1 * x5) + (0.3628 * x2 * x4)  - (0.1106 * x1 * x1)  - (0.3437 * x3 * x3) + (0.1764 * x4 * x4)
    f[2] = -0.0551 + (0.0181 * x1) + (0.1024 * x2) + (0.0421 * x3) - (0.0073 * x1 * x2) + (0.024 * x2 * x3) - (0.0118 * x2 * x4) - (0.0204 * x3 * x4) - (0.008 * x3 * x5) - (0.0241 * x2 * x2) + (0.0109 * x4 * x4)
    return f


def re37(x):
    f = np.zeros(3)
    xAlpha, xHA, xOA, xOPTT = x[0], x[1], x[2], x[3]
    f[0] = 0.692 + (0.477 * xAlpha) - (0.687 * xHA) - (0.080 * xOA) - (0.0650 * xOPTT) - (0.167 * xAlpha * xAlpha) - (0.0129 * xHA * xAlpha) + (0.0796 * xHA * xHA) - (0.0634 * xOA * xAlpha) - (0.0257 * xOA * xHA) + (0.0877 * xOA * xOA) - (0.0521 * xOPTT * xAlpha) + (0.00156 * xOPTT * xHA) + (0.00198 * xOPTT * xOA) + (0.0184 * xOPTT * xOPTT)
    f[1] = 0.153 - (0.322 * xAlpha) + (0.396 * xHA) + (0.424 * xOA) + (0.0226 * xOPTT) + (0.175 * xAlpha * xAlpha) + (0.0185 * xHA * xAlpha) - (0.0701 * xHA * xHA) - (0.251 * xOA * xAlpha) + (0.179 * xOA * xHA) + (0.0150 * xOA * xOA) + (0.0134 * xOPTT * xAlpha) + (0.0296 * xOPTT * xHA) + (0.0752 * xOPTT * xOA) + (0.0192 * xOPTT * xOPTT)
    f[2] = 0.370 - (0.205 * xAlpha) + (0.0307 * xHA) + (0.108 * xOA) + (1.019 * xOPTT) - (0.135 * xAlpha * xAlpha) + (0.0141 * xHA * xAlpha) + (0.0998 * xHA * xHA) + (0.208 * xOA * xAlpha) - (0.0301 * xOA * xHA) - (0.226 * xOA * xOA) + (0.353 * xOPTT * xAlpha) - (0.0497 * xOPTT * xOA) - (0.423 * xOPTT * xOPTT) + (0.202 * xHA * xAlpha * xAlpha) - (0.281 * xOA * xAlpha * xAlpha) - (0.342 * xHA * xHA * xAlpha) - (0.245 * xHA * xHA * xOA) + (0.281 * xOA * xOA * xHA) - (0.184 * xOPTT * xOPTT * xAlpha) - (0.281 * xHA * xAlpha * xOA)
    return f


def re91(x, z):
    """RE91 with its four normal draws z given (see KNOWN DEFECTS)."""
    f = np.zeros(9)
    x1, x2, x3, x4, x5, x6, x7 = x[0], x[1], x[2], x[3], x[4], x[5], x[6]
    x8 = 0.006 * (z[0]) + 0.345
    x9 = 0.006 * (z[1]) + 0.192
    x10 = 10 * (z[2]) + 0.0
    x11 = 10 * (z[3]) + 0.0
    f[0] = 1.98 + 4.9 * x1 + 6.67 * x2 + 6.98 * x3 +  4.01 * x4 +  1.75 * x5 +  0.00001 * x6  +  2.73 * x7
    f[1] = max(0.0, (1.16 - 0.3717* x2 * x4 - 0.00931 * x2 * x10 - 0.484 * x3 * x9 + 0.01343 * x6 * x10 )/1.0)
    f[2] = max(0.0, (0.261 - 0.0159 * x1 * x2 - 0.188 * x1 * x8 - 0.019 * x2 * x7 + 0.0144 * x3 * x5 + 0.87570001 * x5 * x10 + 0.08045 * x6 * x9 + 0.00139 * x8 * x11 + 0.00001575 * x10 * x11)/0.32)
    f[3] = max(0.0, (0.214 + 0.00817 * x5 - 0.131 * x1 * x8 - 0.0704 * x1 * x9 + 0.03099 * x2 * x6 - 0.018 * x2 * x7 + 0.0208 * x3 * x8 + 0.121 * x3 * x9 - 0.00364 * x5 * x6 + 0.0007715 * x5 * x10 - 0.0005354 * x6 * x10 + 0.00121 * x8 * x11 + 0.00184 * x9 * x10 - 0.018 * x2 * x2)/0.32)
    f[4] = max(0.0, (0.74 - 0.61* x2 - 0.163 * x3 * x8 + 0.001232 * x3 * x10 - 0.166 * x7 * x9 + 0.227 * x2 * x2)/0.32)
    tmp = (( 28.98 + 3.818 * x3 - 4.2 * x1 * x2 + 0.0207 * x5 * x10 + 6.63 * x6 * x9 - 7.77 * x7 * x8 + 0.32 * x9 * x10) + (33.86 + 2.95 * x3 + 0.1792 * x10 - 5.057 * x1 * x2 - 11 * x2 * x8 - 0.0215 * x5 * x10 - 9.98 * x7 * x8 + 22 * x8 * x9) + (46.36 - 9.9 * x2 - 12.9 * x1 * x8 + 0.1107 * x3 * x10) )/3
    f[5] = max(0.0, tmp/32)
    f[6] = max(0.0, (4.72 - 0.5 * x4 - 0.19 * x2 * x3 - 0.0122 * x4 * x10 + 0.009325 * x6 * x10 + 0.000191 * x11 * x11)/4.0)
    f[7] = max(0.0, (10.58 - 0.674 * x1 * x2 - 1.95  * x2 * x8  + 0.02054  * x3 * x10 - 0.0198  * x4 * x10  + 0.028  * x6 * x10)/9.9)
    f[8] = max(0.0, (16.45 - 0.489 * x3 * x7 - 0.843 * x5 * x6 + 0.0432 * x9 * x10 - 0.0556 * x9 * x11 - 0.000786 * x11 * x11)/15.7)
    return f


# ── RE91's draws ─────────────────────────────────────────────────────────────
def _normal(seed: int, k: int, i: int) -> float:
    """Draw i (0..3) of evaluation k of the run with this seed: N(0, 1)."""
    u1 = _unit(_splitmix64((seed & _MASK) ^ _splitmix64(_DOMAIN + 8 * k + 2 * i)))
    u2 = _unit(_splitmix64((seed & _MASK) ^ _splitmix64(_DOMAIN + 8 * k + 2 * i + 1)))
    return math.sqrt(-2.0 * math.log(1.0 - u1)) * math.cos(2.0 * math.pi * u2)


class RE91Evaluator:
    """One run's RE91: the k-th call draws the k-th four normals of the stream."""

    def __init__(self, seed: int):
        self.seed = int(seed)
        self.k = 0

    def __call__(self, x) -> List[float]:
        z = [_normal(self.seed, self.k, i) for i in range(4)]
        self.k += 1
        return re91(x, z).tolist()


def re91_by_x(x) -> List[float]:
    """Without a seed: the four normals drawn from x itself."""
    h = 0
    for v in x:
        h = _splitmix64(h ^ struct.unpack("<Q", struct.pack("<d", float(v)))[0])
    z = []
    for i in range(4):
        u1, u2 = _unit(_splitmix64(h + 2 * i + 1)), _unit(_splitmix64(h + 2 * i + 2))
        z.append(math.sqrt(-2.0 * math.log(1.0 - u1)) * math.cos(2.0 * math.pi * u2))
    return re91(x, z).tolist()


# ── the variables' types (task 7, §10) ─────────────────────────────────────
# What the functions above round inside, declared, so that the operators can
# round too (mootation.var_types): RE22's x1 and RE25's x3 from their lists,
# RE23's x1 and x2 integers (the multiples of 0.0625 are 0.0625 times them),
# RE25's x1, RE35's x3 and RE36's four integers. RE33c's x4 stays real, as
# RE33's: reproblems does not round it.
VAR_TYPES = {
    "RE22": [("discrete", sorted(set(float(v) for v in RE22_VALUES))), "real", "real"],
    "RE22c": [("discrete", sorted(float(v) for v in RE22C_VALUES)), "real", "real"],
    "RE23": ["int", "int", "real", "real"],
    "RE25": ["int", "real", ("discrete", sorted(float(v) for v in RE25_VALUES))],
    "RE33c": ["real"] * 4,
    "RE35": ["real", "real", "int", "real", "real", "real", "real"],
    "RE36": ["int"] * 4,
    "CRE24": ["real", "real", "int", "real", "real", "real", "real"],
    "CRE25": ["int"] * 4,
}


# ── the suite ────────────────────────────────────────────────────────────────
# name -> (objectives, (lower, upper), evaluate) for RE; for CRE evaluate
# returns (f, g) and the entry names its RE counterpart.
RE = {
    "RE21": (2, _re21_bounds(), re21),
    "RE22": (2, ([0.2, 0.0, 0.0], [15, 20, 40]), re22),
    "RE23": (2, ([1, 1, 10, 10], [100, 100, 200, 240]), re23),
    "RE24": (2, ([0.5, 0.5], [4, 50]), re24),
    "RE25": (2, ([1, 0.6, 0.09], [70, 3, 0.5]), re25),
    "RE31": (3, _TRUSS, _reformulated(_truss)),
    "RE32": (3, _WELD, _reformulated(_weld)),
    "RE33": (3, _BRAKE, _reformulated(_brake)),
    "RE34": (3, ([1] * 5, [3] * 5), re34),
    "RE35": (3, _REDUCER, _reformulated(_reducer)),
    "RE36": (3, _GEAR, _reformulated(_gear)),
    "RE37": (3, ([0] * 4, [1] * 4), re37),
    "RE41": (4, _CAR, _reformulated(_car)),
    "RE42": (4, _SHIP, _reformulated(_ship)),
    "RE61": (6, _WATER, _reformulated(_water)),
    "RE91": (9, _CAR, re91_by_x),
    # the corrected copies (above)
    "RE22c": (2, ([0.2, 0.0, 0.0], [15, 20, 40]), lambda x: re22(x, RE22C_VALUES)),
    "RE33c": (3, RE33C_BOUNDS, _reformulated(_brake)),
}
CRE = {
    "CRE21": ("RE31", _truss), "CRE22": ("RE32", _weld), "CRE23": ("RE33", _brake),
    "CRE24": ("RE35", _reducer), "CRE25": ("RE36", _gear), "CRE31": ("RE41", _car),
    "CRE32": ("RE42", _ship), "CRE51": ("RE61", _water),
}


def bounds(name: str):
    if name in CRE:
        name = CRE[name][0]
    lb, ub = RE[name][1]
    return [(float(a), float(b)) for a, b in zip(lb, ub)]


def _data(name: str) -> Path:
    """Where a problem's files are: reproblems' own, or ours for RE22c, RE33c."""
    return DATA / "corrected" if name.endswith("c") else DATA


@functools.lru_cache(maxsize=None)
def _reference(name: str) -> np.ndarray:
    return np.loadtxt(_data(name) / f"reference_points_{name}.dat", ndmin=2)


def reference(name: str) -> np.ndarray:
    """The repository's approximated front of an RE problem (a copy)."""
    return _reference(name).copy()


def frame(name: str):
    """(ideal, nadir) of an RE problem: the repository's files."""
    ideal = np.loadtxt(_data(name) / f"ideal_point_{name}.dat", ndmin=1)
    nadir = np.loadtxt(_data(name) / f"nadir_point_{name}.dat", ndmin=1)
    return tuple(float(v) for v in ideal), tuple(float(v) for v in nadir)


def cre_reference(name: str) -> np.ndarray:
    """A CRE problem's front, read off its RE counterpart's (FRONTS above)."""
    F = reference(CRE[name][0])
    F = F[F[:, -1] == 0.0][:, :-1]
    keep = np.array([not np.any(np.all(F <= f, axis=1) & np.any(F < f, axis=1)) for f in F], bool)
    return F[keep] if len(F) else F
