# SPDX-License-Identifier: Apache-2.0
# ============================================================================
# Multi-polygon distance minimization.
# H. Ishibuchi, N. Akedo, Y. Nojima — GECCO 2011.
#
# A problem posed in a 2D DECISION space [0, 100]^2 containing m identical
# regular polygons of k vertices each. The objective count is k, and
# f_i(x) = the distance from x to the i-th vertex, minimized over the m
# polygons. Every point inside a polygon is Pareto-optimal, so there are m
# EQUIVALENT Pareto regions in decision space — which is the point: this is a
# test of decision-space diversity, where objective-space metrics alone cannot
# tell whether an algorithm found one region or all of them.
#
# Provides: eval(x) -> [f...]; pf(n) -> objective vectors (one polygon);
# ps(n) -> points of the Pareto SET across all m regions, for IGDX and PSP.
# ============================================================================
from __future__ import annotations
from typing import List
import numpy as np

BOX = 100.0
RADIUS = 20.0
N_POLY = 2            # m: the number of equivalent regions
# Centres of the m polygons, spread across [0,100]^2.
CENTERS = np.array([[30.0, 50.0], [70.0, 50.0]])


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


# name -> builders (M = vertex count = objective count)
def specs(Ms=(3, 4, 8)):
    out = {}
    for M in Ms:
        out[f"IPolygon_{M}D"] = dict(
            M=M, eval=make_eval(M), pf=make_pf(M), ps=make_ps(M))
    return out
