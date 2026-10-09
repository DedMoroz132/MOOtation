# SPDX-License-Identifier: Apache-2.0
# ============================================================================
# shiftZDT1-4, shiftZDT6, shiftDTLZ6 and shiftDTLZ7: ZDT1-4, ZDT6, DTLZ6 and
# DTLZ7 with the optimum of every distance variable moved off the bound (task
# 6 of 2026-10-09, item 2). A control written for this library, not a
# published suite.
#
# Why. ZDT1-3, ZDT6, DTLZ6 and DTLZ7 put the optimum of every distance
# variable exactly on the lower bound, x = 0 (Zitzler, Deb & Thiele 2000,
# Eq. 7-9 and 12: "The Pareto-optimal front is formed with g(x) = 1"; Deb,
# Thiele, Laumanns & Zitzler 2002 for DTLZ6 and DTLZ7), and ZDT4 at x = 0, the
# middle of [-5, 5] (Eq. 10). An operator that clips a variable to the box
# lands on such an optimum for free: Huband, Hingston, Barone & While (IEEE
# TEVC 10(5), 2006), Recommendation 1, "Extremal parameters are easily
# optimized 'by accident' when EAs employ mutation strategies that truncate
# invalid parameter values back to the edge of their domain".
#
# Not the cyclic shift of shiftDTLZ1-4 (dtlz_variants.py). There every g term
# takes the same value at y = 0 and at y = 1, so reading y = (x - c + 1/2)
# mod 1 keeps g continuous. The g of ZDT1-3, ZDT6 and DTLZ7 is linear in x and
# DTLZ6's increasing: across the wrap g would jump from its largest value to
# its smallest.
#
# THE FOLD. Each distance variable is read through
#     h(x) = |x - c| / max(c, 1 - c),
# 0 at x = c and 1 at the farther bound, so h covers [0, 1] as x does in the
# base problem and g keeps its range. The base problem is then evaluated at
# h(x) in place of x:
#   shiftZDT1-3  g = 1 + 9 sum h(x_i) / (n - 1)
#   shiftZDT6    g = 1 + 9 (sum h(x_i) / (n - 1))^0.25
#   shiftDTLZ7   g = 1 + (9 / k) sum h(x_i)
#   shiftDTLZ6   g = sum h(x_i)^0.1
# The fold has a kink at c, a property of the construction. DTLZ6's x^0.1 has
# an infinite derivative at its optimum, and h^0.1 keeps it there, from both
# sides of c now.
#
# shiftZDT4 keeps ZDT4's box [-5, 5] and moves the optimum by a plain shift,
# u_i = x_i - c_i, g = 1 + 10 (n - 1) + sum (u_i^2 - 10 cos 4 pi u_i). Every
# term is at least -10, with equality only at u = 0, so x = c is the only
# global optimum. The cosine's period is 1/2 and the local optima sit near
# u = m/2; were c a multiple of 1/2 they would fall on round values of x, out
# to the bounds. The golden-ratio offsets of shift_centres give no such
# multiple (the tests check it).
#
# CENTRES: dtlz_variants.shift_centres over the distance variables, as for
# shiftDTLZ1-4, so every optimum is at least 0.15 of the range from both bounds
# and from the middle; for shiftZDT4 c_i = 10 (shift_centres - 1/2), |c_i| in
# [1.5, 3.5]. The objectives depend on the distance variables only through g,
# whose attainable values do not change, so the position variables, the front,
# the ideal and the nadir are the base problem's — for DTLZ6 from four
# objectives too, where the front has a part with g > 0 (fronts_full.py). With
# c = 0, h(x) = x and every problem is its base; shiftZDT4 with c = 0 is ZDT4.
# ============================================================================
from __future__ import annotations

import numpy as np

from .dtlz_variants import shift_centres

ZDT_BASES = ("ZDT1", "ZDT2", "ZDT3", "ZDT4", "ZDT6")
DTLZ_BASES = ("DTLZ6", "DTLZ7")


def fold(x, c) -> np.ndarray:
    """h(x) = |x - c| / max(c, 1 - c): 0 at c, 1 at the farther bound of [0, 1]."""
    x = np.asarray(x, float)
    c = np.asarray(c, float)
    return np.abs(x - c) / np.maximum(c, 1.0 - c)


def zdt_centres(base: str, n_vars: int) -> np.ndarray:
    """The optimum of each of x_2..x_n: shift_centres, scaled to [-5, 5] for ZDT4."""
    c = shift_centres(n_vars - 1)
    return 10.0 * (c - 0.5) if base == "ZDT4" else c


def zdt_input(base: str, x, centres=None) -> np.ndarray:
    """The point the base ZDT is evaluated at: x_2..x_n folded (ZDT4: shifted)."""
    y = np.array(x, dtype=float)
    c = zdt_centres(base, len(y)) if centres is None else np.asarray(centres, float)
    y[1:] = y[1:] - c if base == "ZDT4" else fold(y[1:], c)
    return y


def dtlz_input(x, M: int, centres=None) -> np.ndarray:
    """The point DTLZ6 or DTLZ7 is evaluated at: the distance variables folded."""
    y = np.array(x, dtype=float)
    c = shift_centres(len(y) - M + 1) if centres is None else np.asarray(centres, float)
    y[M - 1:] = fold(y[M - 1:], c)
    return y
