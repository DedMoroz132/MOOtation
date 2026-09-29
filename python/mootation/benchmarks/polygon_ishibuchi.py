# SPDX-License-Identifier: Apache-2.0
# ============================================================================
# Multi-polygon distance minimization.
# H. Ishibuchi, N. Akedo, Y. Nojima — GECCO 2011, Section 2.
#
# A problem posed in a 2D DECISION space [0, 100]^2 containing m identical
# regular polygons of k vertices each. The objective count is k, and
# f_i(x) = the distance from x to the i-th vertex, minimized over the m
# polygons. The paper: "When all polygons are the same and they are not too
# close, all points inside the polygons (including the sides) are Pareto
# optimal" — m EQUIVALENT Pareto regions in decision space, which is the
# point: a test of decision-space diversity, where objective-space metrics
# alone cannot tell whether an algorithm found one region or all of them.
#
# THE INSTANCE (2026-09-29, reference version 2). The paper defines the family
# and illustrates it with m = 4 triangles (Fig. 1) and m = 4 rectangles
# (Fig. 2), whose coordinates it does not print. Here, as in its figures,
# m = 4 regular polygons, radius 8, centred at (25, 25), (75, 25), (25, 75)
# and (75, 75): every two centres at least 50 apart, more than 6 radii. That
# is far enough (our argument, not the paper's): for x in polygon P, its own
# vertex i is within 2r and every other polygon's beyond 50 - 2r > 4r, so f is
# P's own distances; and a point y with f_i(y) <= 2r for every i has all its
# nearest vertices in one polygon Q (two of them in different polygons would
# put their centres within 2r + r + 2r + r = 6r), so y dominating x would make
# y moved by c_P - c_Q dominate x in the one-polygon problem, where the
# polygon itself is the Pareto set. Checked the paper's way on a dense grid
# (TASK4 ipolygon_v2_check): the grid's non-dominated points are exactly
# those inside the four polygons, and pf(1000) has no dominated point.
#
# The instance before it (version 1, stage 3 ran it): m = 2 polygons of radius
# 20 centred at (30, 50) and (70, 50) — too close: the squares shared the vertex
# (50, 50), the triangles' lower vertices were 5.4 apart. At three objectives
# the Pareto set was a region between the triangles and 862 of pf(1000)'s
# points were dominated; at four only the squares' inner halves were
# Pareto-optimal (f2 = f4 there) and 523 of pf(1000)'s points were dominated.
#
# Provides: eval(x) -> [f...]; pf(n) -> objective vectors (one polygon, the
# same for every polygon); ps(n) -> points of all m polygons, for IGDX and PSP;
# frame(M) -> the exact ideal and nadir.
# ============================================================================
from __future__ import annotations
from typing import List
import numpy as np

BOX = 100.0
RADIUS = 8.0
N_POLY = 4            # m: the number of equivalent regions
# Centres of the m polygons, one in each quarter of [0,100]^2, 50 apart.
CENTERS = np.array([[25.0, 25.0], [75.0, 25.0], [25.0, 75.0], [75.0, 75.0]])


def _vertices(M: int):
    """m x M x 2: the vertices of each regular M-gon, identical in shape and size."""
    ang = 2.0 * np.pi * np.arange(M) / M + np.pi / 2.0
    base = RADIUS * np.column_stack([np.cos(ang), np.sin(ang)])   # M×2
    return np.array([base + CENTERS[p] for p in range(N_POLY)])    # m×M×2


def _eval_M(x, M, V):
    x = np.asarray(x, float)[:2]
    # For vertex i: the distance to the nearest of the m polygons.
    d = np.linalg.norm(V - x, axis=2)        # m×M
    return d.min(axis=0).tolist()            # length M


def make_eval(M):
    V = _vertices(M)
    return lambda x, _V=V, _M=M: _eval_M(x, _M, _V)


# ---- point-in-polygon (ray casting) --------------------------------
def _inside(pts: np.ndarray, poly: np.ndarray) -> np.ndarray:
    n = len(poly); inside = np.zeros(len(pts), bool)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]; xj, yj = poly[j]
        cond = ((yi > pts[:, 1]) != (yj > pts[:, 1])) & \
               (pts[:, 0] < (xj - xi) * (pts[:, 1] - yi) / (yj - yi + 1e-30) + xi)
        inside ^= cond
        j = i
    return inside


def _sample_interior(poly: np.ndarray, n: int) -> np.ndarray:
    """Uniform points inside the polygon, by rejection sampling."""
    lo = poly.min(0); hi = poly.max(0)
    out = []
    rng = np.random.default_rng(12345)
    while len(out) < n:
        c = rng.uniform(lo, hi, (4 * n, 2))
        c = c[_inside(c, poly)]
        out.extend(c.tolist())
    return np.asarray(out[:n], float)


def make_pf(M):
    V = _vertices(M)
    def pf(n=1000, _V=V, _M=M):
        pts = _sample_interior(_V[0], n)           # the interior of a single polygon
        return np.array([_eval_M(p, _M, _V) for p in pts])
    return pf


def make_ps(M):
    V = _vertices(M)
    def ps(n=1000, _V=V):
        per = max(1, n // N_POLY)
        return np.vstack([_sample_interior(_V[p], per) for p in range(N_POLY)])
    return ps


def bounds():
    return [(0.0, BOX), (0.0, BOX)]


def frame(M: int) -> tuple:
    """(ideal, nadir) of the front, exactly: f_i is 0 at vertex i itself, and
    its largest value over the polygon — a convex function's maximum, at a
    vertex — is the distance from vertex i to the farthest other vertex."""
    V = _vertices(M)[0]
    far = np.linalg.norm(V[:, None, :] - V[None, :, :], axis=2).max(axis=1)
    return tuple([0.0] * M), tuple(float(v) for v in far)


# name -> builders (M = vertex count = objective count)
def specs(Ms=(3, 4, 8)):
    out = {}
    for M in Ms:
        out[f"IPolygon_{M}D"] = dict(
            M=M, eval=make_eval(M), pf=make_pf(M), ps=make_ps(M), frame=frame(M))
    return out
