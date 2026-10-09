# SPDX-License-Identifier: Apache-2.0
# ============================================================================
# g(x) - g*: how far a point is from the Pareto set, by the problem's own
# distance function (task 6 of 2026-10-09, item 8: the metrics g_min and
# g_median). Exact and free of any reference sample, where the problem is
# built as positions times a scalar g that is at its minimum g* exactly on the
# Pareto set:
#   ZDT1-4, ZDT6   g* = 1 ("The Pareto-optimal front is formed with g(x) = 1",
#                  Zitzler, Deb & Thiele 2000, Eq. 7-10 and 12);
#   DTLZ1-4        g* = 0 (Deb, Thiele, Laumanns & Zitzler 2002), and the
#                  IDTLZ, SDTLZ and wideDTLZ3 built on them;
#   DTLZ5, DTLZ6   g* = 0 at two and three objectives only: from four their
#                  fronts have a part with g > 0 (fronts_full.py), where g is no
#                  distance to the Pareto set;
#   DTLZ7          g* = 1 (g = 1 + 9/k sum x_i).
# The shifted variants read g after the transformation the problem applies:
# shiftDTLZ1-4 through the cyclic shift (dtlz_variants.shift_dtlz), shiftZDT
# and shiftDTLZ6/7 through the fold (shifted.py), so g - g* is 0 exactly at
# their moved optima.
# ============================================================================
from __future__ import annotations

import math

import numpy as np

from . import dtlz as _d

DTLZ_G = {"DTLZ1": _d._g_dtlz1, "DTLZ2": _d._g_dtlz2, "DTLZ3": _d._g_dtlz1,
          "DTLZ4": _d._g_dtlz2, "DTLZ5": _d._g_dtlz2, "DTLZ6": _d._g_dtlz6,
          "DTLZ7": _d._g_dtlz7}
DTLZ_G_STAR = {"DTLZ7": 1.0}                 # every other DTLZ: 0


def zdt_g(base: str, x) -> float:
    """ZDT's g(x_2 .. x_m) (Zitzler, Deb & Thiele 2000, Eq. 7-10 and 12)."""
    z = np.asarray(x, float)[1:]
    if base == "ZDT4":
        return float(1.0 + 10.0 * len(z) + np.sum(z * z - 10.0 * np.cos(4.0 * math.pi * z)))
    if base == "ZDT6":
        return float(1.0 + 9.0 * (np.sum(z) / len(z)) ** 0.25)
    return float(1.0 + 9.0 * np.sum(z) / len(z))           # ZDT1-3


def zdt(base: str, transform=None):
    """g(x) - 1 of a ZDT, read at transform(x) when given (shiftZDT)."""
    def distance(x):
        y = x if transform is None else transform(x)
        return zdt_g(base, y) - 1.0
    return distance


def dtlz(base: str, M: int, transform=None):
    """g(x) - g* of a DTLZ, its distance variables read at transform(x)."""
    g, star = DTLZ_G[base], DTLZ_G_STAR.get(base, 0.0)

    def distance(x):
        y = np.asarray(x if transform is None else transform(x), float)
        return float(g(y[M - 1:])) - star
    return distance
