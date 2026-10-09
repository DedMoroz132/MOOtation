# SPDX-License-Identifier: Apache-2.0
# ============================================================================
# C-DTLZ: constrained DTLZ problems of H. Jain and K. Deb, "An Evolutionary
# Many-Objective Optimization Algorithm Using Reference-point Based
# Non-dominated Sorting Approach, Part II: Handling Constraints and Extending
# to an Adaptive Approach", IEEE TEVC 18(4):602-622, 2014,
# doi:10.1109/TEVC.2013.2281534 (task 6 of 2026-10-09, item 5). Checked
# against pymoo 0.6.2's cdtlz.py (Apache-2.0) at fixed points.
#
#   C1-DTLZ1  DTLZ1, k = 5, c = 1 - f_M/0.6 - sum_{i<M} f_i/0.5 >= 0 (Eq. 4):
#             a feasible band next to the front, which is DTLZ1's
#   C1-DTLZ3  DTLZ3, k = 10, c = (sum f^2 - 16)(sum f^2 - r^2) >= 0, r = 9 at
#             M = 3 and 12.5 at M = 5 (Eq. 5): an infeasible shell between the
#             front, DTLZ3's, and the local fronts beyond it
#   C2-DTLZ2  DTLZ2, k = 10, feasible only inside one of M + 1 spheres of
#             radius r = 0.4 at M = 3, 0.5 otherwise: about the M corners
#             (1, 0, .., 0), ... and about (1, .., 1)/sqrt(M) (Section VI-C)
#   C2-ConvexDTLZ2
#             the convex DTLZ2 of Part I (Deb & Jain 2014, Section VI-D: f_i
#             <- f_i^4 for i < M, f_M <- f_M^2), k = 10, infeasible inside the
#             cylinder about (1, .., 1) of radius r = 0.225 at M = 3 and 5:
#             c = sum (f_i - lambda)^2 - r^2 >= 0, lambda the mean of f (Eq. 6)
#   C3-DTLZ1  DTLZ1, k = 5, M constraints c_j >= 0 (Eq. 7); the front lies on
#             them, DTLZ1's own is infeasible
#   C3-DTLZ4  DTLZ4, n = M + 4, M constraints c_j = f_j^2/4 + sum_{i != j}
#             f_i^2 - 1 >= 0 (Eq. 8); likewise
# at M = 3 and 5.
#
# THE SIGN. The paper writes a constraint as c(x) >= 0 when satisfied; the
# library's convention is a value <= 0 when satisfied, a positive one the size
# of the violation (python/examples/02_constraints.py). Each constraint here
# is -c(x) of the paper.
#
# READINGS (the accepted manuscript's letter, checked on its pages 7 and 9):
#   CDTLZ-1  C2-DTLZ2's constraint is printed as the MAXIMUM over the M + 1
#            spheres of (squared distance - r^2), without a relation; the text
#            makes feasible "the region of objective space that lies inside
#            each of the M + 1 hyper-spheres", and the maximum would ask for
#            all spheres at once, whose intersection is empty. Read as the
#            minimum, as pymoo does: feasible inside at least one sphere.
#   CDTLZ-2  Eq. 7 prints c_j = sum_{i != j} f_j + f_i/0.5 - 1, the indices
#            swapped; read as c_j = f_j/0.5 + sum_{i != j} f_i - 1, the
#            pattern of Eq. 8 (the j-th term scaled, the others summed) and
#            pymoo's. At two objectives both readings are the same set.
#
# FRONTS (analytic, from the constraints and DTLZ's attainable set):
#   C1-DTLZ1, C1-DTLZ3  DTLZ1's and DTLZ3's own: on them the constraint holds
#   C2-DTLZ2            the points of DTLZ2's sphere inside one of the spheres
#   C2-ConvexDTLZ2      the convex front outside the cylinder
#   C3-DTLZ1            f = w / (1 + min_j w_j), w on the unit simplex: the
#                       constraints read sum f + f_j >= 1 for every j, whose
#                       boundary every DTLZ1 objective vector can reach by its
#                       g, and every point of it is Pareto-optimal (lowering
#                       any objective breaks the tightest constraint)
#   C3-DTLZ4            f = u / sqrt(1 - 3/4 max_j u_j^2), u on the unit
#                       sphere's octant: c_j = |f|^2 - 3/4 f_j^2 - 1, as pymoo
# Pareto sets: C1 and C2-DTLZ2 as DTLZ's (distance variables at 1/2),
# C2-DTLZ2's restricted to the feasible positions. C3's front needs g > 0, and
# so does the part of C2-ConvexDTLZ2's behind the hole, which DTLZ's g reaches
# on many-dimensional level sets: no sampler.
# ============================================================================
from __future__ import annotations

from typing import List

import numpy as np

from . import dtlz as _d

NAMES = ("C1-DTLZ1", "C1-DTLZ3", "C2-DTLZ2", "C2-ConvexDTLZ2", "C3-DTLZ1", "C3-DTLZ4")
SIZES = (3, 5)
BASE = {"C1-DTLZ1": "DTLZ1", "C1-DTLZ3": "DTLZ3", "C2-DTLZ2": "DTLZ2",
        "C2-ConvexDTLZ2": "DTLZ2", "C3-DTLZ1": "DTLZ1", "C3-DTLZ4": "DTLZ4"}
K = {"C1-DTLZ1": 5, "C1-DTLZ3": 10, "C2-DTLZ2": 10, "C2-ConvexDTLZ2": 10, "C3-DTLZ1": 5,
     "C3-DTLZ4": 5}                                     # C3-DTLZ4: n = M + 4
C1_DTLZ3_R = {3: 9.0, 5: 12.5}
C2_DTLZ2_R = {3: 0.4, 5: 0.5}
C2_CONVEX_R = {3: 0.225, 5: 0.225}


def n_vars(name: str, M: int) -> int:
    return M + K[name] - 1


def convex(f) -> np.ndarray:
    """Part I's convex DTLZ2 from DTLZ2's objectives: f_i^4, f_M^2."""
    f = np.array(f, float)
    f[..., :-1] = f[..., :-1] ** 4
    f[..., -1] = f[..., -1] ** 2
    return f


def objectives(name: str, x, M: int) -> List[float]:
    f = _d.DTLZ_FUNCS[BASE[name]](list(x), M)
    return convex(f).tolist() if name == "C2-ConvexDTLZ2" else f


def constraints(name: str, f, M: int) -> List[float]:
    """The paper's constraints as the library reads them: <= 0 satisfied."""
    f = np.asarray(f, float)
    s2 = float(np.sum(f * f))
    if name == "C1-DTLZ1":
        return [-(1.0 - f[-1] / 0.6 - float(np.sum(f[:-1])) / 0.5)]
    if name == "C1-DTLZ3":
        return [-((s2 - 16.0) * (s2 - C1_DTLZ3_R[M] ** 2))]
    if name == "C2-DTLZ2":
        r2 = C2_DTLZ2_R[M] ** 2
        corners = [(f[i] - 1.0) ** 2 + (s2 - f[i] ** 2) - r2 for i in range(M)]
        middle = float(np.sum((f - 1.0 / np.sqrt(M)) ** 2)) - r2
        return [min(min(corners), middle)]                       # CDTLZ-1
    if name == "C2-ConvexDTLZ2":
        return [-(float(np.sum((f - f.mean()) ** 2)) - C2_CONVEX_R[M] ** 2)]
    if name == "C3-DTLZ1":
        s = float(np.sum(f))
        return [-(f[j] / 0.5 + (s - f[j]) - 1.0) for j in range(M)]   # CDTLZ-2
    if name == "C3-DTLZ4":
        return [-(f[j] ** 2 / 4.0 + (s2 - f[j] ** 2) - 1.0) for j in range(M)]
    raise KeyError(name)


def feasible(name: str, F, M: int) -> np.ndarray:
    """Which rows of F satisfy every constraint."""
    return np.array([max(constraints(name, f, M)) <= 0.0 for f in np.atleast_2d(F)])


def front(name: str, M: int, n: int, unit_simplex, unit_sphere) -> np.ndarray:
    """About n points of the Pareto front (FRONTS above). `unit_simplex(M, k)`
    and `unit_sphere(M, k)` are the registry's samplers of DTLZ1's (scaled to
    sum 1) and DTLZ2's fronts; the C2 fronts are drawn from a denser sample
    and filtered."""
    if name == "C1-DTLZ1":
        return 0.5 * unit_simplex(M, n)
    if name == "C1-DTLZ3":
        return unit_sphere(M, n)
    if name == "C3-DTLZ1":
        W = unit_simplex(M, n)
        return W / (1.0 + W.min(axis=1, keepdims=True))
    if name == "C3-DTLZ4":
        U = unit_sphere(M, n)
        return U / np.sqrt(1.0 - 0.75 * (U ** 2).max(axis=1, keepdims=True))
    if name == "C2-DTLZ2":
        F = unit_sphere(M, 20 * n)
        return F[feasible(name, F, M)]
    if name == "C2-ConvexDTLZ2":
        return convex_c2_front(M, n)
    raise KeyError(name)


def pareto_set(name: str, M: int, n: int, dtlz_set) -> np.ndarray | None:
    """About n points of the Pareto set where it is known (Pareto sets above),
    None elsewhere. `dtlz_set(M, n_vars, k)` is pareto_sets.dtlz; C2-DTLZ2
    keeps the feasible ones of 20 n, thinned evenly to n."""
    nv = n_vars(name, M)
    if name in ("C1-DTLZ1", "C1-DTLZ3"):
        return dtlz_set(M, nv, n)
    if name == "C2-DTLZ2":
        X = dtlz_set(M, nv, 20 * n)
        X = X[feasible(name, [objectives(name, x, M) for x in X], M)]
        return X[np.unique(np.linspace(0, len(X) - 1, min(n, len(X))).round().astype(int))]
    return None


def convex_c2_front(M: int, n: int, seed: int = 20261009) -> np.ndarray:
    """C2-ConvexDTLZ2's front, sampled (an approximation; FRONTS above).

    DTLZ2's objectives at g are (1 + g) s, s on the unit sphere's octant, and
    the convex ones convex((1 + g) s). A = the feasible points of g = 0 stay
    Pareto-optimal: a point of larger g dominating one would leave its own
    g = 0 point dominating it as well, which the unconstrained front rules out.
    But the cylinder cuts a hole in the front, and behind the hole the convex
    map turns the direction as g grows (f_i grows like (1 + g)^4, f_M like
    (1 + g)^2) until the point leaves the cylinder: B = for every direction in
    the hole the first such point (every point beyond it in that direction it
    dominates). The front is A and those of B that nothing in A or B
    dominates — found 2026-10-09 (task 6): Jain & Deb's and pymoo's front is
    A alone. Directions: the M corners, feasible, and 20 n draws
    |N(0, I)|/norm from a fixed seed; g on a grid of step 1e-3 up to DTLZ2's
    largest, 0.25 k = 2.5, then bisected."""
    rng = np.random.default_rng(seed + M)
    S = np.abs(rng.standard_normal((20 * n, M)))
    S = np.vstack([np.eye(M), S / np.linalg.norm(S, axis=1, keepdims=True)])
    A = convex(S)
    inside = ~feasible("C2-ConvexDTLZ2", A, M)
    hole = S[inside]
    A = A[~inside]
    rho = np.ones(len(hole))
    found = np.zeros(len(hole), bool)
    r2 = C2_CONVEX_R[M] ** 2
    for step in np.arange(1.001, 3.5005, 0.001):           # the first exit
        F = convex(step * hole[~found])
        out = np.sum((F - F.mean(axis=1, keepdims=True)) ** 2, axis=1) - r2 >= 0.0
        idx = np.flatnonzero(~found)[out]
        rho[idx], found[idx] = step, True
        if found.all():
            break
    lo, hi = rho[found] - 0.001, rho[found].copy()
    for _ in range(40):                                     # bisect inside (lo, hi]
        mid = 0.5 * (lo + hi)
        F = convex(mid[:, None] * hole[found])
        out = np.sum((F - F.mean(axis=1, keepdims=True)) ** 2, axis=1) - r2 >= 0.0
        hi, lo = np.where(out, mid, hi), np.where(out, lo, mid)
    B = convex(hi[:, None] * hole[found])
    B = B[~_dominated(B, A) & ~_dominated(B, B)]
    # A is a sample: a candidate it misses may still have a feasible g = 0
    # dominator between its points. Such a dominator s lies where convex(s)
    # <= P, i.e. s_i <= P_i^(1/4) (i < M) and s_M <= P_M^(1/2) on the sphere:
    # that region sampled densely for every candidate, and a candidate with a
    # feasible point there dropped.
    cap = np.column_stack([B[:, :-1] ** 0.25, B[:, -1:] ** 0.5])
    keep = np.ones(len(B), bool)
    for i in range(len(B)):
        U = rng.random((4000, M - 1)) * cap[i, :-1]
        last = 1.0 - np.sum(U * U, axis=1)
        ok = (last >= 0.0) & (last <= cap[i, -1] ** 2)
        if not ok.any():
            continue
        s = np.column_stack([U[ok], np.sqrt(last[ok])])
        keep[i] = not feasible("C2-ConvexDTLZ2", convex(s), M).any()
    return np.vstack([A, B[keep]])


def _dominated(P: np.ndarray, R: np.ndarray, block: int = 256) -> np.ndarray:
    """For each row of P: does some row of R dominate it?"""
    out = np.zeros(len(P), bool)
    for a in range(0, len(P), block):
        p = P[a:a + block, None, :]
        out[a:a + block] = np.any(np.all(R[None] <= p, axis=2) & np.any(R[None] < p, axis=2),
                                  axis=1)
    return out
