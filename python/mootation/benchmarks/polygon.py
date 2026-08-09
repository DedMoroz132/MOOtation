# SPDX-License-Identifier: Apache-2.0
# ============================================================================
# The M-objective Polygon problem.
# H. Ishibuchi, N. Akedo, Y. Nojima — "A many-objective test problem for
# visually examining diversity maintenance behavior in evolutionary
# many-objective optimization", GECCO 2011.
#
# Idea:
#   * a 2D position (x_1, x_2) plus k distance variables x_3..x_n;
#   * f_m(x) = ||(x_1, x_2) - v_m||_2 + g(x_3..x_n), where
#     v_m = (cos(2*pi*(m-1)/M), sin(2*pi*(m-1)/M)) are the vertices of a
#     regular M-gon inscribed in the unit circle;
#   * g(z) = sum z_i^2, which drives the distance variables to zero so that
#     the front is exactly the polygon.
#
# Bounds: x_1, x_2 in [-1, 1] (position); x_3..x_n in [-1, 1] (distance).
#
# Pareto front: the interior of the M-gon (precisely, the convex hull of its
# vertices) — an (M-1)-dimensional manifold in objective space.
#
# Ideal / nadir: ideal_m = 0, reached at vertex m itself; nadir_m = 2, the
# diameter of the unit circle, an upper bound on max f_m over the front.
# ============================================================================
from __future__ import annotations

import math
from typing import List

import numpy as np


def polygon_eval(x: List[float], M: int) -> List[float]:
    x = np.asarray(x, dtype=float)
    # position
    p = x[:2]
    # distance vars → g
    g = float(np.sum(x[2:] ** 2)) if x.shape[0] > 2 else 0.0
    # vertices
    f = np.empty(M)
    for m in range(M):
        theta = 2.0 * math.pi * m / M
        vx = math.cos(theta)
        vy = math.sin(theta)
        f[m] = math.sqrt((p[0] - vx) ** 2 + (p[1] - vy) ** 2) + g
    return f.tolist()


def polygon_bounds(n_vars: int = 10) -> list:
    """Standard: 2 position + (n-2) distance, all in [-1, 1]."""
    return [(-1.0, 1.0)] * n_vars


def polygon_nadir(M: int) -> tuple:
    """Diameter of unit circle = 2 ≥ any PF objective."""
    return tuple([2.0] * M)


def polygon_ideal(M: int) -> tuple:
    return tuple([0.0] * M)
