# SPDX-License-Identifier: Apache-2.0
"""Samples of the Pareto SET — decision vectors — where it is known exactly.

IGDX and the cover rate CR (Tanabe & Ishibuchi, "A niching indicator-based
multi-modal many-objective optimizer", Swarm Evol. Comput. 2019, Eqs. 5, 7-8)
measure an answer set in the decision space, against "about 5000 Pareto
optimal solutions ... uniformly distributed in the solution space" (§4.2).
These samplers give that set for:

  * Polygon_{M}D — the regular M-gon on the unit circle (polygon.py): the
    Pareto set is its interior with the distance variables x_3.. at 0. A
    square grid clipped to the polygon, plus the M vertices, the extreme
    Pareto-optimal points, which a grid would miss.
  * DTLZ1-4 — every position vector is Pareto-optimal once the distance
    variables sit at 1/2 (g = 0): the positions are a Halton sample of
    [0, 1]^(M-1) plus the corners of that cube, deterministic and even.
  * shiftDTLZ1-4 — the same positions, the distance variables at their
    shifted optima c_i (dtlz_variants.shift_centres). Those variables are
    CYCLIC: the problem reads x_i through (x_i − c_i + 1/2) mod 1, so 0.98 is
    0.04 away from 0.02 there, and a distance that ignores the wrap reports
    solutions next to the optimum as far from it. `cyclic_vars` names them.
  * ZCAT1-20 — by the suite's framework (zcat.py): the Pareto set is
    y_II = g(y_I | m) over the position vectors y_I whose alpha image is
    nondominated (Theorem 1), mapped back through x_i = (y_i − 1/2) i. The
    position candidates are drawn the way zcat.pareto_front draws them
    (corners, grid, uniform), each with its own m where m depends on the
    region (ZCAT19, ZCAT20). Only strictly dominated candidates are dropped:
    distinct position vectors with EQUAL images — ZCAT17/18's degenerate
    region, where every F_j is y_1 — are all Pareto-optimal.

Every sampler returns an (n, n_vars) array and is deterministic.
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

_PRIMES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53)


def halton(n: int, d: int, skip: int = 1) -> np.ndarray:
    """The first n points of the d-dimensional Halton sequence (radical inverses)."""
    if d > len(_PRIMES):
        raise ValueError(f"halton: at most {len(_PRIMES)} dimensions")
    out = np.empty((n, d))
    idx = np.arange(skip, skip + n)
    for j in range(d):
        b = _PRIMES[j]
        i = idx.copy()
        f = np.ones(n)
        r = np.zeros(n)
        while np.any(i > 0):
            f /= b
            r += f * (i % b)
            i //= b
        out[:, j] = r
    return out


def _cube_positions(d: int, n: int) -> np.ndarray:
    """n points of [0, 1]^d: the corners (at most a quarter of n), then Halton."""
    if d == 0:
        return np.zeros((n, 0))
    if d == 1:
        return np.linspace(0.0, 1.0, n).reshape(-1, 1)
    corners = ((np.arange(1 << d)[:, None] >> np.arange(d)) & 1).astype(float) \
        if d <= 12 else np.zeros((0, d))
    corners = corners[: n // 4]
    return np.vstack([corners, halton(n - len(corners), d)])


def polygon(M: int, n_vars: int, n: int) -> np.ndarray:
    """The regular M-gon's interior (vertices on the unit circle), x_3.. = 0."""
    ang = 2.0 * np.pi * np.arange(M) / M
    V = np.column_stack([np.cos(ang), np.sin(ang)])
    area = 0.5 * M * math.sin(2.0 * math.pi / M)
    h = math.sqrt(area / max(1, n - M))
    g = np.arange(-1.0, 1.0 + h, h)
    P = np.array(np.meshgrid(g, g, indexing="ij")).reshape(2, -1).T
    P = P[_inside_convex(P, V)]
    P = np.vstack([V, P])
    X = np.zeros((len(P), n_vars))
    X[:, :2] = P
    return X


def _inside_convex(P: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Points inside or on the convex polygon V (counter-clockwise vertices)."""
    ok = np.ones(len(P), bool)
    for i in range(len(V)):
        a, b = V[i], V[(i + 1) % len(V)]
        cross = (b[0] - a[0]) * (P[:, 1] - a[1]) - (b[1] - a[1]) * (P[:, 0] - a[0])
        ok &= cross >= -1e-12
    return ok


def dtlz(M: int, n_vars: int, n: int, centres=None) -> np.ndarray:
    """DTLZ1-4: positions over [0, 1]^(M-1), distance variables at 1/2 (or `centres`)."""
    X = np.empty((n, n_vars))
    X[:, :M - 1] = _cube_positions(M - 1, n)
    X[:, M - 1:] = 0.5 if centres is None else np.asarray(centres, float)
    return X


def _strictly_dominated_by(C: np.ndarray, V: np.ndarray, block: int = 256) -> np.ndarray:
    """For each row of C: does some row of V strictly dominate it? (equal is not)"""
    out = np.zeros(len(C), bool)
    for a in range(0, len(C), block):
        c = C[a:a + block, None, :]
        out[a:a + block] = np.any(np.all(V[None] <= c, axis=2) & np.any(V[None] < c, axis=2),
                                  axis=1)
    return out


def _zcat_points(name: str, M: int, P: np.ndarray):
    """(y, alpha(y)) for position vectors P: y_II set to g(y_I | m), m per point."""
    from . import zcat as Z
    nv = Z.n_vars(M)
    d = P.shape[1]
    Y = np.zeros((len(P), nv))
    Y[:, :d] = P
    which = Z.G_OF[name]
    for r in range(len(Y)):
        m = Z.position_count(name, Y[r], M)
        Y[r, m:] = Z.topology(which, Y[r, :m], m, nv)
    scale = np.arange(1, M + 1, dtype=float) ** 2
    return Y, np.array([scale * Z.F_FUNCS[name](y, M) for y in Y])


def zcat(name: str, M: int, n: int, *, verify: int = 20_000) -> np.ndarray:
    """ZCAT's Pareto set, computed once per arguments and process (seconds
    each; a campaign asks on every run); the caller gets a copy."""
    return _zcat(name, int(M), int(n), int(verify)).copy()


@lru_cache(maxsize=None)
def _zcat(name: str, M: int, n: int, verify: int) -> np.ndarray:
    """ZCAT's Pareto set: y_II = g(y_I | m) where the alpha image is nondominated.

    A candidate is kept only if its image is strictly dominated by none of the
    candidates, of an independent sample of `verify` more position vectors, or
    of its own neighbours — each position coordinate moved alone, both ways,
    by 0.05 down to 0.001, and 32 Gaussian moves of all of them at once. The
    candidates alone let through points that are merely unbeaten among
    themselves: 92 of 1000 on ZCAT11_5D, next to the gaps of its disconnected
    front, were dominated by points of zcat.pareto_front's own 3000-point
    sample. With the independent sample 21 were; with the moves none, and of
    ZCAT11-13, 15, 16, 18-20 at five objectives one point in all (ZCAT13_5D,
    2026-09-22). What such a check cannot see — a dominator no sample and no
    move reaches — is not excluded; this is a reference for IGDX, not a proof.
    """
    from . import zcat as Z
    nv = Z.n_vars(M)
    rng = np.random.default_rng(20260922 + 100 * M)
    cand = max(4000, 4 * n)
    d = 1 if name in Z.DEGENERATE else M - 1
    if d == 1:
        P = np.vstack([np.linspace(0.0, 1.0, cand).reshape(-1, 1), rng.random((verify, 1))])
    else:
        corners = ((np.arange(1 << d)[:, None] >> np.arange(d)) & 1).astype(float) \
            if d <= 14 else np.zeros((0, d))
        if len(corners) > cand // 4:
            corners = corners[rng.choice(len(corners), cand // 4, replace=False)]
        P = np.vstack([corners, rng.random((cand - len(corners) + verify, d))])
    Y, F = _zcat_points(name, M, P)
    k = len(P) - verify                               # the candidates come first
    keep = np.flatnonzero(~_strictly_dominated_by(F[:k], F))
    # the moves cost d * 12 evaluations a point: spend them on a margin over n
    if len(keep) > n + n // 4:
        keep = np.sort(rng.choice(keep, n + n // 4, replace=False))
    Pk, Fk = P[keep], F[keep]
    bad = np.zeros(len(keep), bool)
    moves = []
    for step in (0.05, 0.02, 0.01, 0.005, 0.002, 0.001):
        for j in range(d):
            for sgn in (-1.0, 1.0):
                Q = Pk.copy()
                Q[:, j] = np.clip(Q[:, j] + sgn * step, 0.0, 1.0)
                moves.append(Q)
    for scale in (0.2, 0.1, 0.05, 0.02):              # several positions at once
        for _ in range(8):
            moves.append(np.clip(Pk + rng.normal(0.0, scale, Pk.shape), 0.0, 1.0))
    for Q in moves:
        _, FQ = _zcat_points(name, M, Q)
        bad |= np.all(FQ <= Fk, axis=1) & np.any(FQ < Fk, axis=1)
    Y = Y[keep[~bad]]
    if len(Y) > n:
        Y = Y[np.sort(rng.choice(len(Y), n, replace=False))]
    X = (Y - 0.5) * np.arange(1, nv + 1, dtype=float)
    X.setflags(write=False)                           # shared through the cache
    return X
