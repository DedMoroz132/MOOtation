# SPDX-License-Identifier: Apache-2.0
"""Designs of experiments: where a run puts its first points.

    design(kind, n, bounds, seed)   n points of the box
    start_points(F, k)              which k points of an evaluated design an
                                    algorithm starts from

At budgets of 100-1 000 evaluations a GA makes a few generations, and how its
first points lie may decide as much as what it does after (task 5 of
2026-10-01, run B1). The kind:

  sobol    the first n points of a scrambled Sobol sequence
           (scipy.stats.qmc.Sobol, linear matrix scrambling with a digital
           shift, seeded), as sobol_search draws them

(The uniform start is every algorithm's own: `init = "uniform"` leaves it to
the algorithm.)

minimize(init=...) evaluates a design and starts the algorithm from it; with
init_share > 0 the design is a plan of experiments larger than the population,
and start_points picks the population from it: the nondominated points first
(DSS among them when there are more), the rest by DSS among the others.
"""

from __future__ import annotations

import warnings

KINDS = ("sobol",)


def design(kind: str, n: int, bounds, seed: int):
    """n points of the box `bounds` [(lo, hi), ...], as an (n, d) array."""
    import numpy as np
    if kind not in KINDS:
        raise ValueError(f"unknown design '{kind}'; known: {', '.join(KINDS)}")
    lo = np.array([b[0] for b in bounds], float)
    hi = np.array([b[1] for b in bounds], float)
    try:
        from scipy.stats import qmc
    except ImportError as e:                        # pragma: no cover - machine-dependent
        raise ImportError("a Sobol design needs SciPy: pip install scipy") from e
    with warnings.catch_warnings():                 # "balance properties ... power of 2"
        warnings.simplefilter("ignore")
        U = qmc.Sobol(len(lo), scramble=True, seed=seed).random(n)
    return lo + U * (hi - lo)


def nondominated(F):
    """Indices of the rows of F no other row dominates (minimization)."""
    import numpy as np
    F = np.asarray(F, float)
    keep = []
    for i in range(len(F)):
        worse = np.all(F <= F[i], axis=1) & np.any(F < F[i], axis=1)
        if not worse.any():
            keep.append(i)
    return np.asarray(keep, dtype=np.intp)


def start_points(F, k: int):
    """Indices of k rows of F to start from: the nondominated ones first, DSS
    choosing among them when there are more than k; the rest, when there are
    fewer, chosen by DSS among the other rows. Every row once at most."""
    import numpy as np
    from .run.archive import dss_order
    F = np.asarray(F, float)
    if k >= len(F):
        return np.arange(len(F))
    nd = nondominated(F)
    if len(nd) >= k:
        return nd[dss_order(F[nd], k)]
    rest = np.setdiff1d(np.arange(len(F)), nd)
    return np.concatenate([nd, rest[dss_order(F[rest], k - len(nd))]])
