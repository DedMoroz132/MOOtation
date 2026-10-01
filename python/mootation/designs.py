# SPDX-License-Identifier: Apache-2.0
"""Designs of experiments: where a run puts its first points.

    design(kind, n, bounds, seed)   n points of the box
    start_points(F, k)              which k points of an evaluated design an
                                    algorithm starts from

At budgets of 100-1 000 evaluations a GA makes a few generations, and how its
first points lie may decide as much as what it does after (task 5 of
2026-10-01, run B1). The kinds:

  sobol    the first n points of a scrambled Sobol sequence
           (scipy.stats.qmc.Sobol, linear matrix scrambling with a digital
           shift, seeded), as sobol_search draws them
  lhs      a maximin Latin hypercube (MAXIMIN LHS below)

(The uniform start is every algorithm's own: `init = "uniform"` leaves it to
the algorithm.)

minimize(init=...) evaluates a design and starts the algorithm from it; with
init_share > 0 the design is a plan of experiments larger than the population,
and start_points picks the population from it: the nondominated points first
(DSS among them when there are more), the rest by DSS among the others.

MAXIMIN LHS, after Morris & Mitchell, "Exploratory designs for computational
experiments", J. Statist. Plann. Inference 43 (1995) 381-402. A Latin
hypercube of n points in k variables gives each variable n levels, each used
once: every column of the n-by-k level matrix is a permutation of 0..n-1. Of
those, the search wants the maximin one: the largest smallest distance d1
between two points, then the fewest pairs J1 at d1, then the largest d2, and
so on (Section 2). It minimizes phi_p = (sum over the pairs of d^-p)^(1/p),
Eq. 2.1 (the sum over the distinct distances d_j of J_j d_j^-p is the same
sum), which ranks designs that way for p large enough, with Euclidean distance
taken squared as the paper does for it (Section 4), by the annealing of its
Fig. 1: a perturbation swaps two levels in a column; a better design is kept,
a worse one with probability exp(-(phi_try - phi)/t); after I_max
perturbations without a new best design the temperature drops by FAC_t = 0.95,
and the search ends at a temperature that accepts nothing. t0: on a design
whose C(n, 2) squared distances spread evenly over 50-150 % of the class's
mean, k n (n + 1) / 6 in levels, one of the smallest made one level smaller is
accepted with probability 0.99 (Section 4). The search runs once for each p of
Section 5 (1, 2, 5, 10, 20, 50, 100), and the best of the results by the
maximin order itself is the design. Two departures:
  * the paper's I_max, about 10 C(n, 2) k perturbations, and its open-ended
    temperature loop are out of reach beyond a few dozen points, and a
    perturbation costs O(n): I_max is that or 2 000, whichever is smaller, and
    a search stops after 20 000 perturbations, 2e7 / n above 1 000 points.
    That reaches the optima of the paper's catalog for small designs (Table
    2(A): n = 5, 8 and 9 at k = 2, the tests check two); the larger ones of
    Table 2(A) took 300 000 perturbations a search and the paper's I_max
    (n = 12 at k = 2, n = 6 at k = 4 in 3 seeds of 3, n = 10 at k = 3 in 1);
  * the paper's levels are i/(n - 1), 0 and 1 included and 1/2 at odd n, and its
    best designs at odd n hold the centre of the box itself (Table 3): exactly
    the optima of DTLZ1-4's and ZDT's distance variables, the artefact task 5
    found in DMS (run A1). Here level i is the cell [i/n, (i + 1)/n) of
    McKay, Conover & Beckman's Latin hypercube (1979) and the point is drawn
    uniformly in its cell: the maximin arrangement of the cells is the paper's,
    the points are on no grid.
"""

from __future__ import annotations

import math
import warnings

KINDS = ("sobol", "lhs")
P_VALUES = (1, 2, 5, 10, 20, 50, 100)      # Morris & Mitchell 1995, Section 5
FAC_T = 0.95                               # Section 4: "0.90 and 0.95"
ACCEPT = 0.99                              # Section 4: t0 accepts a step of one level so
I_MAX_CAP = 2000                           # the departures (MAXIMIN LHS): I_max at most,
BUDGET, BUDGET_WORK = 20000, 2 * 10**7     # perturbations per search: BUDGET, BUDGET_WORK / n above


def design(kind: str, n: int, bounds, seed: int):
    """n points of the box `bounds` [(lo, hi), ...], as an (n, d) array."""
    import numpy as np
    if kind not in KINDS:
        raise ValueError(f"unknown design '{kind}'; known: {', '.join(KINDS)}")
    lo = np.array([b[0] for b in bounds], float)
    hi = np.array([b[1] for b in bounds], float)
    if kind == "lhs":
        rng = np.random.default_rng(seed)
        L = maximin_lhs(n, len(lo), rng)
        U = (L + rng.random(L.shape)) / n          # a point drawn in each cell
        return lo + U * (hi - lo)
    try:
        from scipy.stats import qmc
    except ImportError as e:                        # pragma: no cover - machine-dependent
        raise ImportError("a Sobol design needs SciPy: pip install scipy") from e
    with warnings.catch_warnings():                 # "balance properties ... power of 2"
        warnings.simplefilter("ignore")
        U = qmc.Sobol(len(lo), scramble=True, seed=seed).random(n)
    return lo + U * (hi - lo)


# ── maximin Latin hypercubes ────────────────────────────────────────────────


def squared_distances(L):
    """The squared Euclidean distances between the rows of the level matrix L
    (MAXIMIN LHS: in levels), an (n, n) float array, infinite on the diagonal."""
    import numpy as np
    L = np.asarray(L, float)
    sq = (L * L).sum(axis=1)
    # |x - y|^2 = |x|^2 + |y|^2 - 2 x.y, one matrix product; exact on integer
    # levels (below 2^53 for any design here)
    D2 = sq[:, None] + sq[None, :] - 2.0 * (L @ L.T)
    np.fill_diagonal(D2, np.inf)
    return D2


def maximin_order(D2, depth: int = 8) -> tuple:
    """The key the maximin order sorts designs by, best first: (-d1, J1, -d2,
    J2, ...) over the `depth` smallest distinct distances (Section 2: the
    largest d1, then the fewest pairs at it, then the largest d2, ...)."""
    import numpy as np
    key, floor = [], -np.inf
    for _ in range(depth):
        d = float(D2.min(initial=np.inf, where=D2 > floor))
        if d == np.inf:
            break
        key += [-d, int((D2 == d).sum()) // 2]
        floor = d
    return tuple(key)


def maximin_lhs(n: int, k: int, rng):
    """The level matrix (n by k, every column a permutation of 0..n-1) of a
    maximin Latin hypercube by Morris & Mitchell's search (MAXIMIN LHS)."""
    import numpy as np
    best_L, best_key = None, None
    for p in P_VALUES:
        L = np.column_stack([rng.permutation(n) for _ in range(k)])
        if n > 2:
            L = _anneal(L, p, rng)
        key = maximin_order(squared_distances(L))
        if best_key is None or key < best_key:
            best_L, best_key = L, key
    return best_L


def _anneal(L, p: int, rng):
    """Fig. 1 of the paper on phi_p of the squared distances, from the random
    Latin hypercube L; returns the best design it met."""
    import numpy as np
    n, k = L.shape
    cols = L.T.astype(float)                   # the columns, each contiguous
    # The squared distances in units of the smallest at the start, so that the
    # terms d^-p stay within range up to p = 100; phi and t0 scale alike, so
    # neither the ranking nor the acceptance depends on the unit.
    D2 = squared_distances(L)
    scale = float(D2.min())
    D2 /= scale
    with np.errstate(over="ignore", divide="ignore"):
        T = D2 ** (-p)                         # the pairs' terms, 0 on the diagonal
    S = float(T.sum()) / 2
    phi = S ** (1.0 / p)
    t = _initial_temperature(n, k, p, scale)
    i_max = min(10 * k * n * (n - 1) // 2, I_MAX_CAP)
    budget = min(BUDGET, BUDGET_WORK // n)
    best_cols, best_phi, at_best = cols.copy(), phi, True
    spent = 0
    while spent < budget:
        accepted, tries = False, 1
        while tries < i_max and spent < budget:
            spent += 1
            c = int(rng.integers(k))
            a = int(rng.integers(n))
            b = int(rng.integers(n - 1))
            b += b >= a                         # a second row, not a
            col = cols[c]
            # a takes b's level and b a's: the other rows' distances to them
            # change; theirs to each other does not
            delta = ((col[b] - col) ** 2 - (col[a] - col) ** 2) / scale
            delta[a] = delta[b] = 0.0
            new_a, new_b = D2[a] + delta, D2[b] - delta
            with np.errstate(over="ignore", divide="ignore"):
                t_a, t_b = new_a ** (-p), new_b ** (-p)
            S_try = S + float(t_a.sum() - T[a].sum() + t_b.sum() - T[b].sum())
            phi_try = S_try ** (1.0 / p) if S_try > 0 else math.inf
            if phi_try < phi or rng.random() < math.exp(-(phi_try - phi) / t):
                if at_best and phi_try > best_phi:
                    best_cols, at_best = cols.copy(), False     # leaving the best design
                col[a], col[b] = col[b], col[a]
                D2[a], D2[b], D2[:, a], D2[:, b] = new_a, new_b, new_a, new_b
                T[a], T[b], T[:, a], T[:, b] = t_a, t_b, t_a, t_b
                S, phi, accepted = S_try, phi_try, True
            if phi_try < best_phi:
                best_phi, at_best, tries = phi_try, True, 1
            else:
                tries += 1
        S = float(T.sum()) / 2                  # again in full, against the drift
        phi = S ** (1.0 / p)
        if not accepted:
            break
        t *= FAC_T
    return (cols if at_best else best_cols).T.astype(np.int64)


def _initial_temperature(n: int, k: int, p: int, scale: float) -> float:
    """t0 of Section 4: on a hypothetical design whose C(n, 2) squared distances
    are spread evenly over 50-150 % of the class's mean, k n (n + 1) / 6 in
    levels, one of the smallest made one level smaller (the paper's 1/(n - 1)^2)
    is accepted with probability ACCEPT."""
    import numpy as np
    m = n * (n - 1) // 2
    mean = k * n * (n + 1) / 6
    lo, hi = 0.5 * mean / scale, 1.5 * mean / scale
    total = 0.0
    for start in range(0, m, 1_000_000):             # in pieces: m is 12.5 million at 5 000
        h = lo + (hi - lo) * np.arange(start, min(m, start + 1_000_000)) / max(m - 1, 1)
        with np.errstate(over="ignore"):
            total += float((h ** (-p)).sum())
    smallest = lo
    closer = max(lo - 1.0 / scale, 1e-12)
    total_try = total - smallest ** (-p) + closer ** (-p)
    d_phi = total_try ** (1.0 / p) - total ** (1.0 / p)
    return d_phi / -math.log(ACCEPT)


# ── the start of an algorithm from a design ────────────────────────────────


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
