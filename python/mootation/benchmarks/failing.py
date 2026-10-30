# SPDX-License-Identifier: Apache-2.0
"""Problems whose evaluation fails in part of the box: a construction of this
library for experiments with failed evaluations (task 7 of 2026-10-10,
supplement 3, §2.4; mootation.failures, data_vault.hpp FAILED EVALUATIONS).

<base>_fail<share><kind> is the base problem, but a point of the failure region
F returns NaN in every objective, as a solver that crashed would; minimize and
a campaign count it as failed. share is 10 or 30 (per cent); the kind:

  r  random      FAIL_BALLS balls of one radius at centres drawn uniformly in
                 the box, the radius set so that their union covers share of
                 the box's volume (Monte Carlo, 2^14 points, bisection on the
                 radius) — all drawn by C++ (campaign/registry.hpp,
                 failing_balls: the library's generator, a seed of the
                 problem's name; task 17, addendum 7, section 2.3), which this
                 module takes and checks;
  n  near_front  the Pareto set fails in part: everything within TUBE of W,
                 the part of the base's Pareto set — a box in normalised units
                 (window) — whose first variable lies in [1/2 - share/2 +
                 TUBE, 1/2 + share/2 - TUBE]: the window narrowed by the tube,
                 so that share of the set (by the uniform measure on its
                 parameters) fails; a share under 2 TUBE is refused. Here
                 share is a share of the Pareto set, not of the volume: in
                 12-30 variables a neighbourhood of part of the set that
                 fills 10-30 % of the box takes in all of it;
  b  boundary    the band within t of the box's faces, (1 - 2t)^n = 1 - share
                 of the volume.

All distances in the box's normalised units, (x - lb) / (ub - lb).

THE FRONT. A sample of the base's Pareto set (pareto_set; for WFG4 the
positions uniform and the distance variables 0.35 * 2i) is tested against F.
Where F misses it, front and frame are the base's, front_exact as the base's.
Where F takes some of it, the copy's front is not known: what is left of the
base's front stays on it, but points at the edge of F, which the failed part
no longer dominates, join it. The reference is then the image of the sample
points outside F — a part of the true front — and front_exact is False. Where F
takes all of it (the boundary band of ZDT1, whose optimum lies on a face), the
reference is the base's front, which no point outside F reaches, and
front_exact is False.

THE BASES. ZDT1, DTLZ2_3D and WFG4_3D, each at both shares of each kind.
"""

from __future__ import annotations

import math

import numpy as np

BASES = ("ZDT1", "DTLZ2_3D", "WFG4_3D")
SHARES = (10, 30)
KINDS = {"r": "random", "n": "near_front", "b": "boundary"}
FAIL_BALLS = 4
TUBE = 0.05
SAMPLE = 2000


def name(base: str, share: int, kind: str) -> str:
    return f"{base}_fail{share}{kind}"


def wfg4_reference_sample(n: int, k: int, l: int, key: str) -> np.ndarray:
    """WFG4's Pareto set (Huband et al. 2006, the optimum of s_multi): the k
    positions uniform in [0, 2i], the l distance variables at 0.35 * 2i, drawn
    by the library's generator (library_rng; the C++ builder's, refs_build.hpp,
    task 17, section 2.1): the sample the reference front of a failing copy
    of WFG4 keeps the rows of."""
    from .library_rng import LibraryRng, crc32, seed
    rng = LibraryRng(seed(20261011, crc32(key)))
    z = np.empty((n, k + l))
    scale = 2.0 * np.arange(1, k + l + 1)
    for i in range(n):
        for j in range(k):
            z[i, j] = rng.uniform01() * scale[j]
    z[:, k:] = 0.35 * scale[k:]
    return z


def window(base: str, n_obj: int, n_vars: int, share: float) -> tuple:
    """(lo, hi) of W, normalised: the base's Pareto set — ZDT's x1 in [0, 1],
    the others 0; DTLZ's first M - 1 in [0, 1], the others 1/2; WFG's k = 2 (M
    - 1) positions in [0, 1], the distance variables 0.35 (of 2i) — its first
    variable in the window narrowed by the tube (as campaign/registry.hpp's
    failing_window, which is the reference)."""
    if share < 2.0 * TUBE:
        raise ValueError(f"{base}: a near-front share under 2 TUBE ({share})")
    if base.startswith("ZDT"):
        free, rest = 1, 0.0
    elif base.startswith("DTLZ"):
        free, rest = n_obj - 1, 0.5
    elif base.startswith("WFG"):
        free, rest = 2 * (n_obj - 1), 0.35
    else:
        raise ValueError(f"{base}: no near-front region")
    lo, hi = np.full(n_vars, rest), np.full(n_vars, rest)
    lo[:free], hi[:free] = 0.0, 1.0
    lo[0], hi[0] = 0.5 - share / 2.0 + TUBE, 0.5 + share / 2.0 - TUBE
    return lo, hi


class Region:
    """The failure region F of one problem, in normalised units."""

    def __init__(self, kind: str, share: float, n_vars: int, key: str, window=None):
        self.kind, self.share, self.n = kind, share, n_vars
        if kind == "b":
            self.t = (1.0 - (1.0 - share) ** (1.0 / n_vars)) / 2.0
        elif kind == "r":
            from .. import _core                      # the balls are C++'s (failing_balls)
            centres, radius = _core.failing_balls(key, n_vars, share)
            self.centres, self.radius = np.asarray(centres, float), float(radius)
        else:
            self.lo, self.hi = (np.asarray(v, float) for v in window)

    def inside(self, U) -> np.ndarray:
        """Rows of U (normalised points) that fail."""
        U = np.atleast_2d(np.asarray(U, float))
        if self.kind == "b":
            return np.any((U < self.t) | (U > 1.0 - self.t), axis=1)
        if self.kind == "n":                          # |u - clamp(u, W)| <= TUBE
            return ((U - np.clip(U, self.lo, self.hi)) ** 2).sum(axis=1) <= TUBE ** 2
        if not len(self.centres):
            return np.zeros(len(U), bool)
        # squared distances to every centre by one matrix product
        d2 = ((U * U).sum(axis=1)[:, None] + (self.centres * self.centres).sum(axis=1)[None, :]
              - 2.0 * U @ self.centres.T)
        return d2.min(axis=1) < self.radius ** 2


def make(base, share_pct: int, kind: str, ps_sample):
    """(evaluate, region, the sample's rows outside F) for one failing copy."""
    lo = np.array([b[0] for b in base.bounds], float)
    hi = np.array([b[1] for b in base.bounds], float)
    unit = lambda X: (np.asarray(X, float) - lo) / (hi - lo)      # noqa: E731
    S = np.asarray(ps_sample, float)
    share = share_pct / 100.0
    region = Region(kind, share, base.n_vars, name(base.name, share_pct, kind),
                    window(base.name, base.n_obj, base.n_vars, share) if kind == "n" else None)
    keep = ~region.inside(unit(S))

    def evaluate(x, _f=base.evaluate, _r=region, _m=base.n_obj):
        if _r.inside(unit(x))[0]:
            return [math.nan] * _m
        return _f(x)
    return evaluate, region, S[keep]
