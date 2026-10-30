# SPDX-License-Identifier: Apache-2.0
"""Problems whose evaluation fails in part of the box: a construction of this
library for experiments with failed evaluations (task 7 of 2026-10-10,
supplement 3, §2.4; mootation.failures, data_vault.hpp FAILED EVALUATIONS).

<base>_fail<share><kind> is the base problem, but a point of the failure region
F returns NaN in every objective, as a solver that crashed would; minimize and
a campaign count it as failed. share is 10 or 30 (per cent); the kind:

  r  random      FAIL_BALLS balls of one radius at centres drawn uniformly in
                 the box (a seed of the problem's name), the radius set so
                 that their union covers share of the box's volume (Monte
                 Carlo, 2^14 points, bisection on the radius);
  n  near_front  the Pareto set fails in part: everything within TUBE of the
                 points of a Pareto-set sample whose first variable lies in a
                 window about its median, the window's width set (bisection)
                 so that share of the sample fails. Here share is a share of
                 the Pareto set, not of the volume: in 12-30 variables a
                 neighbourhood of part of the set that fills 10-30 % of the
                 box takes in all of it;
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
front_exact is False. The random balls take much of DTLZ2's and WFG4's
central Pareto sets (at 30 %, all but a few dozen and a few points of 2 000).

THE BASES. ZDT1, DTLZ2_3D and WFG4_3D, each at both shares of each kind.
"""

from __future__ import annotations

import math
import zlib

import numpy as np

BASES = ("ZDT1", "DTLZ2_3D", "WFG4_3D")
SHARES = (10, 30)
KINDS = {"r": "random", "n": "near_front", "b": "boundary"}
FAIL_BALLS = 4
TUBE = 0.05
SAMPLE = 2000


def name(base: str, share: int, kind: str) -> str:
    return f"{base}_fail{share}{kind}"


def _rng(key: str) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([20261011, zlib.crc32(key.encode())]))


def wfg4_pareto_set(n: int, k: int, l: int, rng) -> np.ndarray:
    """WFG4's Pareto set: the k positions uniform in [0, 2i], the l distance
    variables at 0.35 * 2i (Huband et al. 2006, the optimum of s_multi)."""
    z = np.empty((n, k + l))
    scale = 2.0 * np.arange(1, k + l + 1)
    z[:, :k] = rng.random((n, k)) * scale[:k]
    z[:, k:] = 0.35 * scale[k:]
    return z


def wfg4_reference_sample(n: int, k: int, l: int, key: str) -> np.ndarray:
    """wfg4_pareto_set drawn by the library's generator (library_rng; the C++
    builder's, refs_build.hpp, task 17, section 2.1): the sample the reference
    front of a failing copy of WFG4 keeps the rows of. The failure region stays
    the one made from _rng's sample, part of the problem's definition."""
    from .library_rng import LibraryRng, crc32, seed
    rng = LibraryRng(seed(20261011, crc32(key)))
    z = np.empty((n, k + l))
    scale = 2.0 * np.arange(1, k + l + 1)
    for i in range(n):
        for j in range(k):
            z[i, j] = rng.uniform01() * scale[j]
    z[:, k:] = 0.35 * scale[k:]
    return z


class Region:
    """The failure region F of one problem, in normalised units."""

    def __init__(self, kind: str, share: float, n_vars: int, key: str, ps_unit=None):
        self.kind, self.share, self.n = kind, share, n_vars
        rng = _rng(key)
        if kind == "b":
            self.t = (1.0 - (1.0 - share) ** (1.0 / n_vars)) / 2.0
        elif kind == "r":
            self.centres = rng.random((FAIL_BALLS, n_vars))
            probe = rng.random((1 << 14, n_vars))
            d = np.min(np.linalg.norm(probe[:, None, :] - self.centres[None, :, :], axis=2), axis=1)
            self.radius = float(np.quantile(d, share))     # the share of the probe within it
        else:
            S = np.asarray(ps_unit, float)
            self.radius = TUBE
            a, b = 0.0, share                  # the window's share of the sample
            for _ in range(30):
                self.centres = self._window(S, (a + b) / 2.0)
                if self.inside(S).mean() > share:
                    b = (a + b) / 2.0
                else:
                    a = (a + b) / 2.0
            self.centres = self._window(S, a)

    @staticmethod
    def _window(S, q):
        lo, hi = np.quantile(S[:, 0], [0.5 - q / 2.0, 0.5 + q / 2.0])
        return S[(S[:, 0] >= lo) & (S[:, 0] <= hi)]

    def inside(self, U) -> np.ndarray:
        """Rows of U (normalised points) that fail."""
        U = np.atleast_2d(np.asarray(U, float))
        if self.kind == "b":
            return np.any((U < self.t) | (U > 1.0 - self.t), axis=1)
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
    region = Region(kind, share_pct / 100.0, base.n_vars, name(base.name, share_pct, kind),
                    unit(S))
    keep = ~region.inside(unit(S))

    def evaluate(x, _f=base.evaluate, _r=region, _m=base.n_obj):
        if _r.inside(unit(x))[0]:
            return [math.nan] * _m
        return _f(x)
    return evaluate, region, S[keep]
