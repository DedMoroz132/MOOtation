# SPDX-License-Identifier: Apache-2.0
"""The reference fronts of RE22c and RE33c, built here (task 7 of 2026-10-10, §9).

    python -m mootation.benchmarks.re_fronts            # writes _re/corrected/

RE22c and RE33c are RE22 and RE33 with their known defects corrected (re.py);
reproblems has no front for them, so this builds one the way its RE fronts
were made — every non-dominated point found, thinned to 500 M points (here by
DSS, dss.hpp's rule, as the library's references) — but from an exhaustive
search instead of optimizers' runs:

  RE22c  x1 takes each of the list's 77 values; x2 and x3 a grid of 1 001 x
         2 001 points over [0, 20] x [0, 40] (x2 = 0 left out: the first
         constraint is -inf there); the non-dominated points of each x1, then
         of all; then twice a local grid of 21 x 21 points around 4 000 of
         them spread evenly along f1, a tenth of the step each time, merged
         with the set; 1 000 points by DSS from 20 000 spread evenly.
  RE33c  a grid of 26 x 36 x 41 x 46 points over the box (x4 in [2, 11]);
         the non-dominated points (a sweep in f1 over a staircase in f2, f3);
         then six rounds of 20 Gaussian steps around 5 000 of them drawn at
         random (each objective's extremes always), the step 2 % of the
         range, halved each round, merged with the set, which is kept to
         30 000 points the same way; 1 500 points by DSS.

The ideal and nadir points written beside are the fronts' own minima and
maxima. Deterministic (a fixed seed): the files are this function's output.
"""

from __future__ import annotations

import bisect
from pathlib import Path

import numpy as np

OUT = Path(__file__).with_name("_re") / "corrected"


def _nd_2d(F: np.ndarray) -> np.ndarray:
    """Indices of the non-dominated rows of an n x 2 array (duplicates: one
    kept): in order of f1, then f2, a row whose f2 is below every earlier one."""
    order = np.lexsort((F[:, 1], F[:, 0]))
    f2 = F[order, 1]
    before = np.concatenate([[np.inf], np.minimum.accumulate(f2)[:-1]])
    return order[f2 < before]


def _nd_3d(F: np.ndarray) -> np.ndarray:
    """Indices of the non-dominated rows of an n x 3 array: a sweep in f1 over a
    staircase of (f2, f3) — f2 ascending, f3 descending (duplicates: one kept)."""
    order = np.lexsort((F[:, 2], F[:, 1], F[:, 0]))
    s2, s3, keep = [], [], []
    for i in order:
        a, b = F[i, 1], F[i, 2]
        k = bisect.bisect_right(s2, a)
        if k > 0 and s3[k - 1] <= b:
            continue                       # some earlier point is at least as good
        keep.append(i)
        # insert (a, b) and drop the staircase points it dominates in (f2, f3)
        j = k
        while j < len(s2) and s3[j] >= b:
            j += 1
        del s2[k:j], s3[k:j]
        s2.insert(k, a)
        s3.insert(k, b)
    return np.asarray(keep, int)


def _thin(F: np.ndarray, X: np.ndarray, n: int):
    """n points by DSS, in the set's own frame (archive.dss_order)."""
    from ..run.archive import dss_order
    idx = np.asarray(dss_order(F, n, ideal=F.min(axis=0), nadir=F.max(axis=0)), int)
    return F[idx], X[idx]


def _even(F: np.ndarray, X: np.ndarray, k: int):
    """k rows spread evenly in the order of f1 (a 2-D front), both ends kept."""
    if len(F) <= k:
        return F, X
    order = np.argsort(F[:, 0], kind="stable")
    idx = order[np.unique(np.linspace(0, len(F) - 1, k).round().astype(int))]
    return F[idx], X[idx]


def _sample(F: np.ndarray, X: np.ndarray, k: int, rng):
    """k rows at random, each objective's least and largest always."""
    if len(F) <= k:
        return F, X
    keep = sorted(set(np.argmin(F, axis=0).tolist()) | set(np.argmax(F, axis=0).tolist()))
    rest = np.setdiff1d(np.arange(len(F)), keep)
    idx = np.concatenate([keep, rng.choice(rest, k - len(keep), replace=False)]).astype(int)
    return F[idx], X[idx]


def _re22c(x1, X2, X3):
    """RE22c's objectives on arrays, as re.re22c evaluates one point."""
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = 29.4 * x1 + 0.6 * X2 * X3
        g1 = x1 * X3 - 7.735 * (x1 * x1 / X2) - 180.0
        g2 = 4.0 - X3 / X2
    v = np.where(g1 < 0, -g1, 0) + np.where(g2 < 0, -g2, 0)
    return np.stack([f1, v], axis=-1)


def front_re22c(n: int = 1000):
    from .re import RE22C_VALUES
    x2 = np.linspace(0.0, 20.0, 1001)[1:]
    x3 = np.linspace(0.0, 40.0, 2001)
    d2, d3 = x2[1] - x2[0], x3[1] - x3[0]
    F_all, X_all = [], []
    for x1 in RE22C_VALUES:
        X2, X3 = np.meshgrid(x2, x3, indexing="ij")
        F = _re22c(x1, X2, X3).reshape(-1, 2)
        ok = np.all(np.isfinite(F), axis=1)
        X = np.stack([np.full(X2.size, x1), X2.ravel(), X3.ravel()], axis=1)[ok]
        F = F[ok]
        k = _nd_2d(F)
        F_all.append(F[k])
        X_all.append(X[k])
    F, X = np.vstack(F_all), np.vstack(X_all)
    k = _nd_2d(F)
    F, X = F[k], X[k]
    for level in (1, 2):                   # local grids a tenth of the step each time
        s2, s3 = d2 / 10 ** (level - 1), d3 / 10 ** (level - 1)
        F_new, X_new = [F], [X]
        off = np.linspace(-1.0, 1.0, 21)
        for x in _even(F, X, 4000)[1]:
            G2, G3 = np.meshgrid(np.clip(x[1] + s2 * off, 1e-9, 20.0),
                                 np.clip(x[2] + s3 * off, 0.0, 40.0), indexing="ij")
            G = _re22c(x[0], G2, G3).reshape(-1, 2)
            ok = np.all(np.isfinite(G), axis=1)
            F_new.append(G[ok])
            X_new.append(np.stack([np.full(G2.size, x[0]), G2.ravel(), G3.ravel()], axis=1)[ok])
        F, X = np.vstack(F_new), np.vstack(X_new)
        k = _nd_2d(F)
        F, X = F[k], X[k]
    return _thin(*_even(F, X, 20000), n)


def front_re33c(n: int = 1500, seed: int = 20261010):
    from .re import RE, RE33C_BOUNDS
    evaluate = RE["RE33c"][2]
    lo, hi = (np.asarray(v, float) for v in RE33C_BOUNDS)

    def batch(X):
        return np.array([evaluate(x) for x in X], float)
    axes = [np.linspace(lo[i], hi[i], k) for i, k in enumerate((26, 36, 41, 46))]
    X = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 4)
    F = _vectorised_re33(X)
    ok = np.all(np.isfinite(F), axis=1)          # x1 = x2: 0 / 0 in f2
    F, X = F[ok], X[ok]
    k = _nd_3d(F)
    F, X = F[k], X[k]
    rng = np.random.default_rng(seed)
    F, X = _sample(F, X, 30000, rng)
    sigma = 0.02 * (hi - lo)
    for _ in range(6):
        base = _sample(F, X, 5000, rng)[1]
        Y = np.repeat(base, 20, axis=0) + sigma * rng.standard_normal((20 * len(base), 4))
        Y = np.clip(Y, lo, hi)
        G = _vectorised_re33(Y)
        ok = np.all(np.isfinite(G), axis=1)
        F, X = np.vstack([F, G[ok]]), np.vstack([X, Y[ok]])
        k = _nd_3d(F)
        F, X = _sample(F[k], X[k], 30000, rng)
        sigma = sigma / 2
    # the vectorised values against the function's own, on the kept points
    assert np.allclose(batch(X[:50]), F[:50], rtol=1e-12, atol=1e-12)
    return _thin(F, X, n)


def _vectorised_re33(X):
    """re._brake reformulated (RE33's three objectives) over rows of X; a row
    with x1 = x2 comes out NaN, as the function's own does."""
    x1, x2, x3, x4 = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = 4.9 * 1e-5 * (x2 * x2 - x1 * x1) * (x4 - 1.0)
        f2 = ((9.82 * 1e6) * (x2 * x2 - x1 * x1)) / (x3 * x4 * (x2 * x2 * x2 - x1 * x1 * x1))
        g = np.stack([(x2 - x1) - 20.0,
                      0.4 - (x3 / (3.14 * (x2 * x2 - x1 * x1))),
                      1.0 - (2.22 * 1e-3 * x3 * (x2 * x2 * x2 - x1 * x1 * x1))
                      / np.power((x2 * x2 - x1 * x1), 2),
                      (2.66 * 1e-2 * x3 * x4 * (x2 * x2 * x2 - x1 * x1 * x1))
                      / (x2 * x2 - x1 * x1) - 900.0], axis=1)
    v = np.where(g < 0, -g, 0)
    s = v[:, 0]
    for j in range(1, 4):
        s = s + v[:, j]
    return np.stack([f1, f2, s], axis=1)


def write(name: str, F: np.ndarray) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    np.savetxt(OUT / f"reference_points_{name}.dat", F, fmt="%.17g")
    np.savetxt(OUT / f"ideal_point_{name}.dat", F.min(axis=0)[None, :], fmt="%.17g")
    np.savetxt(OUT / f"nadir_point_{name}.dat", F.max(axis=0)[None, :], fmt="%.17g")


def main() -> int:
    for name, build in (("RE22c", front_re22c), ("RE33c", front_re33c)):
        F, _ = build()
        write(name, F)
        print(f"{name}: {len(F)} points, ideal {F.min(axis=0)}, nadir {F.max(axis=0)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
