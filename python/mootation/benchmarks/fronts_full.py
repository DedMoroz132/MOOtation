# SPDX-License-Identifier: Apache-2.0
"""The FULL Pareto fronts of DTLZ5, DTLZ6, MaF6 and WFG3.

These four were designed to have a degenerate, curve-shaped front, and their
reference sets used to be exactly that curve. Ishibuchi, Masuda & Nojima,
"Pareto Fronts of Many-Objective Degenerate Test Problems" (IEEE TEVC 20(5):
807-813, 2016, doi:10.1109/TEVC.2015.2505784; source: ishibuchi2016) showed
that WFG3's true front also has a NON-degenerate part from three objectives
(Section IV; its shape at M = 3 is Eq. (25)) and recalled that DTLZ5's and
DTLZ6's have one from four (Section II-B, after Huband et al. and Saxena et
al.) — so the curve was a proper subset of the front there. The paper's
constraints (7)-(8) (DTLZ5, from Saxena et al.) and (27)-(28) (WFG3) would
cut the non-degenerate part off; the library keeps the problems as defined
and uses the full fronts. For DTLZ5 and DTLZ6 this was already
in Huband, Hingston, Barone & While (IEEE TEVC 10(5), 2006, Section VI-A,
the DTLZ suite), which is in the library: at M = 4 with g = 10 ("possible
when k >= 40"; with the usual k = 10, g <= 2.5), y_1 = 0, y_2 = 0.95 and
y_3 = 0, DTLZ5 gives f_2 = 11 cos(20 pi/44) sin(pi/44) = 0.11
with f_4 = 0, where the curve's point with f_4 = 0 has f_2 = 0.5 — "we have
found that this is untrue for instances with four or more objectives". MaF6,
DTLZ5(I = 2, M) with a factor (1 + 100 g), came later than both papers; it
leaves the curve from seven objectives (below). The mechanism is the same in
all four: the front contains points whose distance component is NOT at its
optimum.

  * DTLZ5, DTLZ6, MaF6: with g > 0, theta_i for i >= 2 moves away from pi/4,
    which can lower some objectives by more than the factor (1 + g) — or
    (1 + 100 g) for MaF6 — raises them.
  * WFG3: with t_M > 0, x_i = t_M (t_i − 1/2) + 1/2 moves away from 1/2.

HOW THE FRONTS ARE BUILT (build_front). Everything happens in the REDUCED
coordinates u — the M−1 positions plus the distance scalar (g, or t_M) — whose
dimension is M, not the number of variables:

  1. sample the box of u uniformly, and every face of it separately, of every
     dimension down to the vertices: x_1 or t_1 at 0 and at 1, g = 0 or
     t_M = 0, two, three or more coordinates at once, and so on (_sample).
     Without the faces false points survive — (0, 0, 0, 0, 1.03) is dominated
     by (0, 0, 0, 0, 1) with g = 0, which a uniform sample practically never
     hits — and without the deeper ones too: the far parts of the DTLZ5/6
     fronts lie on faces three and four coordinates deep. The edges and the
     two-dimensional faces get dense regular grids on top (_face_grids);
  2. keep the nondominated points (an exact sweep, see nd_mask);
  3. local descent: a point that a small perturbation of u dominates is
     replaced by it, at shrinking step sizes; then a (1+lambda)-ES looks for
     a dominating point from each (dominance_search), and a point off the
     curve that the curve dominates moves to its dominator; step 2 after each;
  4. VERIFY every remaining point — against an independent uniform sample of
     the box and its faces (at least 10^6 points) and denser grids on its
     edges and two-dimensional faces than the build used, against a fresh local
     search around it, and, for a point off the curve, exactly against the
     curve (curve_dominated), whose dominator can lie too far away in u for
     any local search; and last EXACTLY against the whole attainable set
     (radial_dominated, wfg3_dominated). For DTLZ5/6 and MaF6 that set is
     {base(g) s(theta)} over an angle box that widens with g, for WFG3 it is
     s + 2m h(x) over a box that widens with s = t_M, and for one g (or s)
     whether some point lies below a given one is decided greedily in M
     steps; g is scanned on a grid of 20 001, between whose values only a
     point beaten by very little can slip (the one found, checking a WFG3_4D
     build on another grid, by 9.7e-6 of the range). A point that anything
     dominates is removed and counted. Without this the reference is
     contaminated: in a trial at M = 5 three of the five points that set the
     nadir were dominated by a fresh sample, and after every sampling check
     a quarter to a third of the DTLZ5/6 fronts, and 1-2 % of WFG3's, was
     still dominated — by 2e-4 to 4.5e-2 of the range — which only the exact
     test showed;
  5. re-evaluate the survivors with the problem's OWN code — the registered
     evaluator on a decision vector built to have that g, or WFG3's own
     calc_x / shape / final — and require agreement to 1e-9 relative;
  6. store them in DSS order (mootation.run.archive), so that thinning to
     n_ref points is taking the first n_ref.

THE FRONT REACHES THE BOUND OF g — the finding that sets the nadir, and the
reason DTLZ5 and DTLZ6 do NOT share a front although they share the formula.
Take x_1 = 0 (so f_M = 0), x_2 = 1 and x_3 = 0, the largest theta_2 and the
smallest theta_3 a given g allows. At M = 4 the smallest f_2 attainable with
f_M = 0 is then (1 + g) sin^2(pi / (4 (1 + g))), which DECREASES strictly in g
(0.5 at g = 0, 0.174 at g = 2.48). So the point (0.759, 0.174, 3.392, 0) at
g = 2.48 is Pareto-optimal: anything with f_2 <= 0.174 needs g >= 2.48, and
then f_3 >= 3.39, with equality only at the point itself. The front therefore
extends to the largest g the DECISION SPACE allows, and its nadir with it:
g = sum (x_j − 1/2)^2 <= k/4 = 2.5 for DTLZ5 but g = sum x_j^0.1 <= k = 10 for
DTLZ6, k = 10 distance variables. In general the objective that multiplies
the M − 2 cosines theta_2 .. theta_M−1 has, with f_M = 0, the smallest value
(1 + g) sin^(M−2)(pi / (4 (1 + g))). For MaF6 the factor is (1 + 100 g), and
(1 + 100 g) sin^(M−2)(pi / (4 (1 + g))) rises with g first; it comes back
below its value at g = 0, (1/sqrt 2)^(M−2), only at large g and only once
M − 2 >= 5 (at the bound g = 2.5: 0.137 < 0.177 at M = 7, 0.61 > 0.25 at
M = 6). So by this argument MaF6 leaves the curve at seven objectives (the
registry has it at 3, 5, 8, 10 and 15), while the build at five, which looks
for every kind of off-curve point, finds none; where it does leave, it too
runs to the bound of g: at M = 8 the unique minimiser of f_1 on f_8 = 0 has
f_7 = 251 sin(6 pi / 14) = 244.7. The extreme points sit on vertices and
edges of the box of u, which is why every face is sampled on purpose, and
the edges and two-dimensional faces on dense grids besides (step 1).

NO OBJECTIVE f_M ABOVE 1. For DTLZ5, DTLZ6 and MaF6 every point with f_M > 1
is dominated by the curve: take the curve point whose cos(theta_1) is the
smallest of f_j / c_j over j < M (c_j the curve's cosine products, capped at
1); it is no worse in f_1 .. f_M−1 and has f_M = sin(theta_1) <= 1. At
x_1 = 1 this needs cos(pi / 2) = 0 exactly, which floating point gives as
6.1e-17: every point with x_1 = 1 and g > 0 then looks nondominated by a
hair, and the rounded cosine set the nadir of f_M to 1 + g_max — 251 for
MaF6 at eight objectives — before objectives() was made to use the exact
zero there. The problems' own code keeps the rounded value; step 5 accepts
the difference, which is below 1e-13.

Regenerate the shipped files with `python -m mootation.benchmarks.make_fronts`.
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

FRONT_DIR = Path(__file__).with_name("_fronts")

# The sizes with a non-degenerate part among those the registry has (MaF6 at
# 5 has none, see the header). Lower sizes keep their exact curve.
SIZES = {
    "DTLZ5": (4, 5, 6, 10, 15),
    "DTLZ6": (4, 5, 6, 10, 15),
    "MaF6":  (8, 10, 15),
    "WFG3":  (3, 4, 5, 6, 10),
}
SHARED: dict = {}      # none: DTLZ5 and DTLZ6 reach different g (see the header)

# The distance scalar's REACHABLE range, not a guess at where the front ends:
# g = sum (x_j − 1/2)^2 over the k = 10 distance variables reaches k/4 = 2.5
# (DTLZ5, MaF6); g = sum x_j^0.1 reaches k = 10 (DTLZ6). From four objectives
# the DTLZ5/6 fronts run all the way to it (the header's argument), so this
# bound is part of the problem's definition, not of the sampler.
G_MAX = {"DTLZ5": 2.5, "DTLZ6": 10.0, "MaF6": 2.5}

# A dominance margin above this (relative to the front's range) proves a point
# dominated; below it is rounding at v ~ u. The smallest real margin met in
# practice is a WFG3 point at t_M ~ 7e-6, beaten by its own projection by ~1e-6.
MARGIN_TOL = 1e-9


# ── reduced coordinates and objectives ──────────────────────────────────────
def box(name: str, M: int):
    lo, hi = np.zeros(M), np.ones(M)
    if name in G_MAX:
        hi[M - 1] = G_MAX[name]
    return lo, hi


def objectives(name: str, U: np.ndarray, M: int) -> np.ndarray:
    """Vectorized objectives at reduced coordinates U (rows). See _own for the check."""
    U = np.atleast_2d(np.asarray(U, float))
    if name in ("DTLZ5", "DTLZ6", "MaF6"):
        X, g = U[:, :M - 1], U[:, M - 1]
        th = np.empty_like(X)
        th[:, 0] = X[:, 0] * (math.pi / 2.0)
        if M > 2:
            th[:, 1:] = (math.pi / (4.0 * (1.0 + g)))[:, None] * (1.0 + 2.0 * g[:, None] * X[:, 1:])
        c, s = np.cos(th), np.sin(th)
        c[:, 0] = np.where(X[:, 0] == 1.0, 0.0, c[:, 0])      # cos(pi/2) = 0, see the header
        base = 1.0 + (100.0 if name == "MaF6" else 1.0) * g
        F = np.empty((len(U), M))
        F[:, 0] = base * np.prod(c, axis=1)
        for i in range(2, M):
            F[:, i - 1] = base * np.prod(c[:, :M - i], axis=1) * s[:, M - i]
        F[:, M - 1] = base * s[:, 0]
        return F
    if name == "WFG3":
        t, tM = U[:, :M - 1], U[:, M - 1]
        X = np.empty((len(U), M))
        X[:, 0] = t[:, 0]                                        # A_1 = 1
        if M > 2:
            X[:, 1:M - 1] = tM[:, None] * (t[:, 1:] - 0.5) + 0.5  # A_i = 0
        X[:, M - 1] = tM
        xp = X[:, :M - 1]
        H = np.empty((len(U), M))
        H[:, 0] = np.prod(xp, axis=1)
        for m in range(1, M - 1):
            H[:, m] = np.prod(xp[:, :M - m - 1], axis=1) * (1.0 - xp[:, M - m - 1])
        H[:, M - 1] = 1.0 - xp[:, 0]
        return X[:, [M - 1]] + 2.0 * np.arange(1, M + 1) * H
    raise KeyError(name)


def _own(name: str, U: np.ndarray, M: int) -> np.ndarray:
    """The same points through the problem's own code (step 5 of the header)."""
    if name == "WFG3":
        from . import wfg as W
        A = np.zeros(M - 1); A[0] = 1.0
        out = np.empty((len(U), M))
        for r, t in enumerate(np.asarray(U, float)):
            x = W._nb_calc_x(np.ascontiguousarray(t), A)
            out[r] = W._nb_final(x, W._nb_shape_linear(x[:-1]), M)
        return out
    from .registry import PROBLEMS
    p = PROBLEMS[f"{name}_{M}D"]
    k = p.n_vars - (M - 1)
    out = np.empty((len(U), M))
    for r, u in enumerate(np.asarray(U, float)):
        g = float(u[M - 1])
        if name == "DTLZ6":
            dist = [(g / k) ** 10] * k                   # g = sum x^0.1
        else:
            dist = [0.5 + math.sqrt(g / k)] * k          # g = sum (x - 1/2)^2
        out[r] = p.evaluate(list(u[:M - 1]) + dist)
    return out


# ── dominance ───────────────────────────────────────────────────────────────
def nd_mask(F: np.ndarray) -> np.ndarray:
    """Exact nondominated filter; of equal points one is kept.

    In lexicographic order every point comes after all that dominate it, so a
    single sweep that checks each point against the points kept so far is
    exact: a dominated dominator's own dominator was kept before it.

    Above 20 000 points a first pass drops what 512 well-spread points of a
    random subsample's nondominated set strictly dominate, which is most of a
    sample, and the sweep runs on the rest. The result is the same: a point
    any other point strictly dominates is not nondominated, equal points are
    never strictly dominated so the sweep still decides between them, and a
    stable sort keeps their order.
    """
    F = np.asarray(F, float)
    idx = np.arange(len(F))
    if len(F) > 20_000:
        from ..run.archive import dss_order
        sub = np.random.default_rng(0).choice(len(F), 20_000, replace=False)
        S = F[sub][nd_mask(F[sub])]
        S = S[dss_order(S, k=min(len(S), 512))]
        idx = np.flatnonzero(~_dominated_block(F, S))
    G = F[idx]
    order = np.lexsort(G.T[::-1])
    keep = np.zeros(len(F), bool)
    arch = np.empty_like(G)
    k = 0
    for i in order:
        f = G[i]
        if k and np.any(np.all(arch[:k] <= f, axis=1)):   # dominated, or a duplicate
            continue
        keep[idx[i]] = True
        arch[k] = f
        k += 1
    return keep


def _dominated_block(A: np.ndarray, B: np.ndarray, ablock: int = 2048,
                     bblock: int = 512) -> np.ndarray:
    """For each row of A: is it dominated by some row of B? (blocked, exact)"""
    out = np.zeros(len(A), bool)
    for b in range(0, len(B), bblock):
        s = B[b:b + bblock]
        for a in range(0, len(A), ablock):
            r = A[a:a + ablock]
            todo = ~out[a:a + ablock]
            if not todo.any():
                continue
            le = np.all(s[None, :, :] <= r[:, None, :], axis=2)
            lt = np.any(s[None, :, :] < r[:, None, :], axis=2)
            out[a:a + ablock] |= np.any(le & lt, axis=1)
    return out


def dominated_by_any(R: np.ndarray, S: np.ndarray) -> np.ndarray:
    """For each row of R — mutually nondominated — is it dominated by a row of S?

    Exact, and fast when S is large: a point of S that some r' of R dominates
    cannot dominate any r of R (r' would then dominate r), so S is first cut
    down to what R leaves undominated — against 512 well-spread points of R,
    which removes nearly all of a random sample for the price of a few
    thousand comparisons each, then against all of R — and only the remainder
    is compared with R.
    """
    from ..run.archive import dss_order
    R = np.asarray(R, float)
    S = np.asarray(S, float)
    if not len(R) or not len(S):
        return np.zeros(len(R), bool)
    sentinels = R[dss_order(R, k=min(len(R), 512))]
    S = S[~_dominated_block(S, sentinels)]
    S = S[~_dominated_block(S, R)]
    return _dominated_block(R, S) if len(S) else np.zeros(len(R), bool)


def _sample(lo, hi, n_int, n_face, rng):
    """The box and every face of it, of every dimension, down to the vertices.

    A face holds k coordinates at their bounds and samples the rest; each has
    zero probability under any sampling of the faces above it. The front's
    extreme points sit on vertices (see the header), and its far parts on
    low-dimensional faces: DTLZ6_5D points built from the faces and
    two-dimensional faces alone were beaten, by up to 0.96 in f_3, by points
    with x_1 = 0, x_3 = 1 and x_4 near 1 — a face three coordinates deep, which
    that sampling reached only by accident. Budget: n_face points on every
    face (k = 1); n_face over the C(d, 2) pairs for each of their four corner
    assignments (k = 2); 2 n_face per level spread over its C(d, k) 2^k faces
    for 3 <= k < d, at least one point each, up to d = 10 (from 11 the levels
    number in the thousands of faces, and only k <= 2 is sampled); every
    vertex once, up to d = 16.
    """
    import itertools
    d = len(lo)
    parts = [lo + rng.random((n_int, d)) * (hi - lo)]
    for j in range(d):
        for v in (lo[j], hi[j]):
            P = lo + rng.random((n_face, d)) * (hi - lo)
            P[:, j] = v
            parts.append(P)
    pairs = [(i, j) for i in range(d) for j in range(i + 1, d)]
    per = max(1, n_face // max(1, len(pairs)))       # the same total as one face
    for i, j in pairs:
        for vi in (lo[i], hi[i]):
            for vj in (lo[j], hi[j]):
                P = lo + rng.random((per, d)) * (hi - lo)
                P[:, i] = vi
                P[:, j] = vj
                parts.append(P)
    if d <= 10:
        for k in range(3, d):
            subsets = list(itertools.combinations(range(d), k))
            corners = (np.arange(1 << k)[:, None] >> np.arange(k)) & 1
            per = max(1, (2 * n_face) // (len(subsets) * len(corners)))
            for S in subsets:
                S = list(S)
                for bits in corners:
                    P = lo + rng.random((per, d)) * (hi - lo)
                    P[:, S] = np.where(bits == 1, hi[S], lo[S])
                    parts.append(P)
    if d <= 16:
        bits = (np.arange(1 << d)[:, None] >> np.arange(d)) & 1
        parts.append(np.where(bits == 1, hi, lo))
    return np.vstack(parts)


def _face_grids(lo, hi, edge_n: int, side: int) -> np.ndarray:
    """Regular grids on every edge of the box (one coordinate free, edge_n
    points) and, up to d = 6, on every two-dimensional face (side x side).

    The random face sampling of _sample reaches these faces too, but thinly,
    and a point of the front next to one is beaten only by a small stretch of
    it: DTLZ6_4D points of the face x_1 = 0 by the edge x_1 = 0, x_2 = 1,
    x_3 = 0 at g near 0.05, 14 of 2000 stored points by 850 000 random face
    points after a build that had sampled 527 000.
    """
    import itertools
    d = len(lo)
    parts = [np.zeros((0, d))]
    if d > 10:                                       # 15 * 2^14 edges: none at all
        return parts[0]
    for j in range(d):
        others = [i for i in range(d) if i != j]
        t = np.linspace(lo[j], hi[j], edge_n)
        for bits in itertools.product((0, 1), repeat=d - 1):
            P = np.empty((edge_n, d))
            P[:, others] = np.where(np.array(bits) == 1, hi[others], lo[others])
            P[:, j] = t
            parts.append(P)
    if d <= 6 and side > 1:
        for i, j in itertools.combinations(range(d), 2):
            others = [k for k in range(d) if k not in (i, j)]
            a, b = np.meshgrid(np.linspace(lo[i], hi[i], side), np.linspace(lo[j], hi[j], side),
                               indexing="ij")
            for bits in itertools.product((0, 1), repeat=d - 2):
                P = np.empty((side * side, d))
                P[:, others] = np.where(np.array(bits) == 1, hi[others], lo[others])
                P[:, i] = a.ravel()
                P[:, j] = b.ravel()
                parts.append(P)
    return np.vstack(parts)


def _grid_budget(d: int, verify: bool, scale: float = 1.0) -> tuple:
    """(edge_n, side) for _face_grids: denser for the check than for the build,
    and scaled down with the sample (scale = its share of the full budget:
    n_int / 10^5 for the build, n_verify / 10^6 for the check)."""
    if d <= 5:
        edge_n, side = (5000, 120) if verify else (1000, 60)
    elif d == 6:
        edge_n, side = (2000, 80) if verify else (500, 40)
    else:
        edge_n, side = (200, 0) if verify else (50, 0)
    scale = min(1.0, max(0.0, scale))
    return max(20, int(edge_n * scale)), (max(8, int(side * scale ** 0.5)) if side else 0)


def _moves(U, lo, hi, rng, scales, gauss_tries):
    """Perturbed copies of every row of U, for the descent and the local check.

    Deterministic moves first: the distance scalar (the last coordinate) sent
    to its lower bound and halved — a point off the degenerate curve is most
    often beaten by its own projection onto it, which a random step, landing
    in a thin cone of dominating directions, finds only by luck — and every
    coordinate pushed to each of its bounds; then every coordinate alone, both
    ways, at each scale; then full Gaussian steps for the combined directions.
    """
    d = U.shape[1]
    P = U.copy(); P[:, d - 1] = lo[d - 1]
    yield P
    P = U.copy(); P[:, d - 1] = lo[d - 1] + 0.5 * (U[:, d - 1] - lo[d - 1])
    yield P
    # every coordinate pushed to either bound: the front's far parts lie on
    # faces several coordinates deep, and a point at 0.99 is beaten by its
    # copy at 1 more often than by any step that stops short of it
    for j in range(d):
        for v in (lo[j], hi[j]):
            P = U.copy(); P[:, j] = v
            yield P
    for scale in scales:
        for j in range(d):
            for sgn in (-1.0, 1.0):
                P = U.copy()
                P[:, j] = np.clip(U[:, j] + sgn * scale * (hi[j] - lo[j]), lo[j], hi[j])
                yield P
        for _ in range(gauss_tries):
            yield np.clip(U + rng.normal(0.0, scale, U.shape) * (hi - lo), lo, hi)


def _curve_u(name: str, M: int, s: np.ndarray) -> np.ndarray:
    """Reduced coordinates of the degenerate curve at parameter s in [0, 1].

    The first position is s; the distance scalar is 0; the other positions do
    not matter there (theta_i = pi/4, x_i = 1/2), so they are set to 1/2.
    """
    U = np.full((len(s), M), 0.5)
    U[:, 0] = s
    U[:, M - 1] = 0.0
    return U


def curve_dominated(name: str, M: int, F: np.ndarray, *, iters: int = 60):
    """For each row of F: (is it dominated by a point of the curve, that point's s).

    Exact up to bisection precision (2^-60). Along the curve every objective is
    monotone in s — f_j = c_j cos(s pi/2) and f_M = sin(s pi/2) for DTLZ5/6 and
    MaF6, f_m = 2m a_m s and f_M = 2M (1 − s) for WFG3 — so the curve points no
    worse than a row in objective j form an interval of s, and the row is
    dominated by the curve exactly when the M intervals meet. The bounds are
    found by bisection on the problem's own objectives. This catches the
    dominator a local search cannot reach: a point off the curve beaten by a
    curve point far away in u, across a valley where every nearer point is worse
    in some objective (MaF6 at eight objectives: x_1 just below 1 and g at its
    bound, f_8 near 251, beaten by the curve near s = 0.9999).
    """
    F = np.atleast_2d(np.asarray(F, float))
    n = len(F)
    ends = objectives(name, _curve_u(name, M, np.array([0.0, 1.0])), M)
    lo_s, hi_s = np.zeros(n), np.ones(n)
    for j in range(M):
        increasing = ends[1, j] > ends[0, j]
        a, b = np.zeros(n), np.ones(n)                # bisection bracket per row
        for _ in range(iters):
            mid = 0.5 * (a + b)
            ok = objectives(name, _curve_u(name, M, mid), M)[:, j] <= F[:, j]
            if increasing:                            # feasible: s <= threshold
                a = np.where(ok, mid, a); b = np.where(ok, b, mid)
            else:                                     # feasible: s >= threshold
                b = np.where(ok, mid, b); a = np.where(ok, a, mid)
        f0 = objectives(name, _curve_u(name, M, np.zeros(n)), M)[:, j]
        f1 = objectives(name, _curve_u(name, M, np.ones(n)), M)[:, j]
        if increasing:
            thr = np.where(f1 <= F[:, j], 1.0, np.where(f0 <= F[:, j], a, -1.0))
            hi_s = np.minimum(hi_s, thr)
        else:
            thr = np.where(f0 <= F[:, j], 0.0, np.where(f1 <= F[:, j], b, 2.0))
            lo_s = np.maximum(lo_s, thr)
    hit = lo_s <= hi_s
    return hit, np.where(hit, 0.5 * (lo_s + hi_s), np.nan)


def radial_dominated(name: str, M: int, F: np.ndarray, *, span=None, tau: float = 1e-9,
                     n_g: int = 20_001, block: int = 32) -> np.ndarray:
    """For DTLZ5, DTLZ6 and MaF6: is each row beaten by ANY attainable point?

    Exact up to a grid on g. In the reduced coordinates the attainable set is
    {base(g) s(theta)}: s the unit-sphere map of the objectives, theta_1 in
    [0, pi/2] and theta_i in [a(g), pi/2 − a(g)] for i >= 2, a(g) = pi/(4(1+g)),
    base = 1 + g (1 + 100 g for MaF6), g in [0, g_max]. For one g, whether some
    theta puts base s(theta) <= P is decided greedily in M steps: f_M = base
    sin theta_1 bounds theta_1 from above, and every later objective carries
    cos theta_1, so the largest theta_1 allowed is the best choice; the same
    holds for theta_2 against f_M−1, and so on down to f_1, which only has to
    fit what is left. A row is reported when some grid g admits a point below
    it by tau * span in every positive objective (and equal where it is 0).
    This is the check sampling cannot make: points of the front's far parts
    were beaten by a small stretch of an edge that no sample hit.
    """
    F = np.atleast_2d(np.asarray(F, float))
    if span is None:
        span = F.max(axis=0) - F.min(axis=0)
    span = np.where(np.asarray(span, float) > 0, span, 1.0)
    g = np.linspace(0.0, G_MAX[name], n_g)
    base = 1.0 + (100.0 if name == "MaF6" else 1.0) * g
    a = math.pi / (4.0 * (1.0 + g))                     # theta_i in [a, pi/2 - a]
    P = np.where(F > 0.0, F - tau * span, 0.0)
    out = np.zeros(len(F), bool)
    for s0 in range(0, len(F), block):
        Q = P[s0:s0 + block, None, :] / base[None, :, None]       # (rows, n_g, M)
        ok = Q[..., M - 1] >= 0.0
        th = np.arcsin(np.clip(Q[..., M - 1], 0.0, 1.0))
        C = np.where(th >= math.pi / 2.0, 0.0, np.cos(th))
        for i in range(1, M - 1):                   # theta_{i+1} against f_{M-i}
            q = Q[..., M - 1 - i]
            live = C > 0.0
            r = np.where(live, q / np.where(live, C, 1.0), 1.0)
            t = np.minimum(math.pi / 2.0 - a[None, :], np.arcsin(np.clip(r, 0.0, 1.0)))
            ok &= ~live | ((r >= 0.0) & (t >= a[None, :]))
            C = np.where(live, C * np.cos(t), 0.0)
        ok &= C <= Q[..., 0]
        out[s0:s0 + block] = ok.any(axis=1)
    return out


def wfg3_dominated(M: int, F: np.ndarray, *, span=None, tau: float = 1e-9,
                   n_t: int = 20_001, block: int = 32) -> np.ndarray:
    """For WFG3: is each row beaten by ANY attainable point? Exact up to a grid
    on t_M.

    With t_M = s the attainable points are s + 2m h_m(x): x_1 in [0, 1], x_i in
    [(1 − s)/2, (1 + s)/2] for 2 <= i <= M−1, and the linear shape h_M = 1 − x_1,
    h_M−k = x_1 .. x_k (1 − x_k+1), h_1 = x_1 .. x_M−1. For one s, whether some
    x puts the point below P is decided greedily: h_M bounds x_1 from below,
    and x_1 multiplies every other h, so the smallest x_1 allowed is the best
    choice; then x_2 against h_M−1, and so on, h_1 last. Reported as in
    radial_dominated: below P by tau * span in every positive objective.
    """
    F = np.atleast_2d(np.asarray(F, float))
    if span is None:
        span = F.max(axis=0) - F.min(axis=0)
    span = np.where(np.asarray(span, float) > 0, span, 1.0)
    s = np.linspace(0.0, 1.0, n_t)
    S = 2.0 * np.arange(1, M + 1, dtype=float)
    lo_i, hi_i = 0.5 - s / 2.0, 0.5 + s / 2.0
    P = np.where(F > 0.0, F - tau * span, 0.0)
    out = np.zeros(len(F), bool)
    for r0 in range(0, len(F), block):
        Q = (P[r0:r0 + block, None, :] - s[None, :, None]) / S          # h_m <= Q_m
        ok = Q[..., M - 1] >= 0.0
        x = np.maximum(0.0, 1.0 - Q[..., M - 1])                       # x_1
        ok &= x <= 1.0
        prod = x
        for k in range(1, M - 1):                   # x_{k+1} against h_{M-k}
            q = Q[..., M - 1 - k]
            live = prod > 0.0
            need = np.where(live, 1.0 - q / np.where(live, prod, 1.0), -np.inf)
            xk = np.maximum(lo_i[None, :], need)
            ok &= ~live | (xk <= hi_i[None, :])
            ok &= q >= 0.0
            prod = prod * xk
        ok &= prod <= Q[..., 0]
        out[r0:r0 + block] = ok.any(axis=1)
    return out


def dominance_search(name, U, F, M, lo, hi, rng, *, iters: int = 40, lam: int = 16,
                     sigma0: float = 0.05):
    """A local search for a point that dominates each row: (1+λ)-ES per point.

    Maximizes the dominance margin s(v) = min_j (f_j(u) − f_j(v)) / range_j
    from v = u, where s(u) = 0; any v with s(v) > 0 dominates u in every
    objective, so a positive result PROVES u is not Pareto-optimal (a zero
    result proves nothing, which is why the independent sample stays). The
    step size grows on success and shrinks on failure, so the search reaches
    both a nearby dominator and one that needs several coordinates moved
    together — the case fixed random steps miss: a point off the curve with
    theta_1 near pi/2 is beaten by a curve point near (0, .., 0, 1), and only
    by moving x_1 and g at once.

    Returns (margin, argmax) per row.
    """
    n, d = U.shape
    span = F.max(axis=0) - F.min(axis=0)
    span = np.where(span > 0.0, span, 1.0)
    V = U.copy()
    best = np.zeros(n)
    sigma = np.full(n, sigma0)
    width = hi - lo
    for _ in range(iters):
        P = V[:, None, :] + rng.normal(size=(n, lam, d)) * (sigma[:, None, None] * width)
        P = np.clip(P, lo, hi).reshape(n * lam, d)
        G = objectives(name, P, M).reshape(n, lam, M)
        s = np.min((F[:, None, :] - G) / span, axis=2)            # margin of every candidate
        k = np.argmax(s, axis=1)
        top = s[np.arange(n), k]
        up = top > best
        best = np.where(up, top, best)
        V[up] = P.reshape(n, lam, d)[np.flatnonzero(up), k[up]]
        sigma = np.where(up, np.minimum(sigma * 1.5, 0.5), sigma * 0.7)
    return best, V


def _descend(name, U, F, M, lo, hi, rng, scales, tries):
    """Replace every point that a move dominates by that move, until none does."""
    moved = 0
    for _ in range(4):                          # repeat: a moved point may move again
        before = moved
        for P in _moves(U, lo, hi, rng, scales, tries):
            G = objectives(name, P, M)
            better = np.all(G <= F, axis=1) & np.any(G < F, axis=1)
            if better.any():
                U[better], F[better] = P[better], G[better]
                moved += int(better.sum())
        if moved == before:
            break
    return U, F, moved


def build_front(name: str, M: int, *, seed: int = 20260922, n_int: int = 200_000,
                n_face: int = 20_000, n_verify: int = 1_000_000, n_keep: int = 2000,
                log=print):
    """Build, verify and DSS-order the full front; returns (F, report)."""
    from ..run.archive import dss_order
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed + 1000 * M + (7 if name == "WFG3" else 0))
    lo, hi = box(name, M)
    d = len(lo)

    U = np.vstack([_sample(lo, hi, n_int, n_face, rng),
                   _face_grids(lo, hi, *_grid_budget(d, verify=False, scale=n_int / 100_000))])
    n_sampled = len(U)
    F = objectives(name, U, M)
    m = nd_mask(F)
    U, F = U[m], F[m]
    n_sampled_nd = len(U)
    U, F, moved = _descend(name, U, F, M, lo, hi, rng, np.geomspace(0.05, 1e-4, 5), 16)
    m = nd_mask(F)
    U, F = U[m], F[m]
    # and the stronger local search: a point it proves dominated moves to the
    # dominator it found, which lies nearer the front
    moved_es = 0
    for _ in range(4):
        margin, V = dominance_search(name, U, F, M, lo, hi, rng)
        hit = margin > MARGIN_TOL
        if not hit.any():
            break
        moved_es += int(hit.sum())
        U[hit] = V[hit]
        F[hit] = objectives(name, V[hit], M)
        m = nd_mask(F)
        U, F = U[m], F[m]
    # and a point off the curve that the curve dominates goes to its dominator
    off = np.flatnonzero(U[:, d - 1] > 0.0)
    moved_curve = 0
    if len(off):
        hit, s = curve_dominated(name, M, F[off])
        if hit.any():
            rows = off[hit]
            U[rows] = _curve_u(name, M, s[hit])
            F[rows] = objectives(name, U[rows], M)
            moved_curve = int(hit.sum())
            m = nd_mask(F)
            U, F = U[m], F[m]
    log(f"  {name}_{M}D: {n_sampled_nd} nondominated in the sample, {len(F)} after descent "
        f"({moved} step, {moved_es} search, {moved_curve} curve moves), "
        f"{time.perf_counter() - t0:.0f}s")

    # step 4a: an independent sample of the box and its faces
    vrng = np.random.default_rng(seed + 77 + 1000 * M)
    dropped_sample = 0
    done = 0
    while done < n_verify:
        chunk = min(250_000, n_verify - done)
        V = _sample(lo, hi, chunk, max(1, chunk // (4 * d)), vrng)
        bad = dominated_by_any(F, objectives(name, V, M))
        dropped_sample += int(bad.sum())
        U, F = U[~bad], F[~bad]
        done += len(V)
    # ... and dense grids on the edges and two-dimensional faces
    Gu = _face_grids(lo, hi, *_grid_budget(d, verify=True, scale=n_verify / 1_000_000))
    bad = dominated_by_any(F, objectives(name, Gu, M)) if len(Gu) else np.zeros(len(F), bool)
    dropped_grid = int(bad.sum())
    U, F = U[~bad], F[~bad]
    # step 4b: a fresh local search around every point — fixed moves, then the
    # dominance-margin search with its own random stream
    bad = np.zeros(len(U), bool)
    for P in _moves(U, lo, hi, vrng, (1e-2, 1e-3, 1e-4, 1e-5, 1e-6), 8):
        G = objectives(name, P, M)
        bad |= np.all(G <= F, axis=1) & np.any(G < F, axis=1)
    margin, _ = dominance_search(name, U, F, M, lo, hi, vrng, iters=60)
    bad |= margin > MARGIN_TOL
    dropped_local = int(bad.sum())
    U, F = U[~bad], F[~bad]
    # step 4c: exactly, whether the curve dominates any point off it
    off = np.flatnonzero(U[:, d - 1] > 0.0)
    dropped_curve = 0
    if len(off):
        hit, _ = curve_dominated(name, M, F[off])
        dropped_curve = int(hit.sum())
        keep = np.ones(len(U), bool)
        keep[off[hit]] = False
        U, F = U[keep], F[keep]
    # step 4d: exactly, against every attainable point — the check that
    # sampling cannot make (radial_dominated, wfg3_dominated)
    span = F.max(axis=0) - F.min(axis=0)
    exact = (radial_dominated(name, M, F, span=span) if name in G_MAX else
             wfg3_dominated(M, F, span=span))
    dropped_exact = int(exact.sum())
    U, F = U[~exact], F[~exact]

    # step 5: the problem's own code must agree
    own = _own(name, U, M)
    err = float(np.max(np.abs(own - F) / np.maximum(1.0, np.abs(F)))) if len(F) else 0.0
    if not err <= 1e-9:
        raise AssertionError(f"{name}_{M}D: own code disagrees by {err:.3g}")

    # step 6: DSS order, in the front's own frame
    order = dss_order(F, k=n_keep)
    F_out = F[order]
    dist = U[:, d - 1]
    report = {
        "problem": f"{name}_{M}D", "seed": seed, "sampled": int(n_sampled),
        "nondominated_in_sample": n_sampled_nd, "descent_moves": moved,
        "search_moves": moved_es, "curve_moves": moved_curve,
        "verify_sample": done, "dropped_by_sample": dropped_sample,
        "verify_grid": int(len(Gu)), "dropped_by_grid": dropped_grid,
        "dropped_by_local_search": dropped_local, "dropped_by_curve": dropped_curve,
        "dropped_by_exact_test": dropped_exact,
        "verified": int(len(F)),
        "stored": int(len(F_out)), "own_code_max_rel_error": err,
        "distance_scalar_box": float(hi[d - 1]),
        "distance_scalar_max_on_front": float(dist.max()) if len(dist) else 0.0,
        "share_off_the_curve": float(np.mean(dist > 1e-9)) if len(dist) else 0.0,
        "ideal": F.min(0).tolist(), "nadir": F.max(0).tolist(),
        "seconds": round(time.perf_counter() - t0, 1),
    }
    log(f"    verified {len(F)} (dropped {dropped_sample} by {done} samples, "
        f"{dropped_grid} by {len(Gu)} grid points, "
        f"{dropped_local} by local search, {dropped_curve} by the curve, "
        f"{dropped_exact} by the exact test); max distance scalar "
        f"{report['distance_scalar_max_on_front']:.4g} of {hi[d - 1]:.4g}; "
        f"off the curve {report['share_off_the_curve']:.1%}; {report['seconds']}s")
    return F_out, report


def _path(name: str, M: int) -> Path:
    return FRONT_DIR / f"{SHARED.get(name, name)}_{M}D.npz"


def has_front(name: str, M: int) -> bool:
    return _path(name, M).is_file()


_LOADED: dict = {}


def load(name: str, M: int):
    """(the stored front in DSS order, its build report), or None."""
    path = _path(name, M)
    if not path.is_file():
        return None
    if path not in _LOADED:
        with np.load(path) as z:
            _LOADED[path] = (np.array(z["F"], float), json.loads(str(z["report"])))
    return _LOADED[path]


def front(name: str, M: int, n: int) -> np.ndarray:
    """The first n points of the stored front — its DSS selection of size n."""
    F, _ = load(name, M)
    return F[:max(1, int(n))].copy()


def _update_frame_cache(reports: dict) -> None:
    """Write each front's ideal and nadir into the registry's frame cache.

    The frame is the WHOLE verified set's, which the stored DSS prefix need not
    contain — the nadir sits on the few far points of the front. The cache is
    rewritten in its own format (one line, sorted keys), after checking that
    the unchanged table reproduces the file, so no other entry can move.
    """
    from .registry import _REFRAME_CACHE_FILE as path
    raw = path.read_bytes()
    table = json.loads(raw.decode("utf-8"))
    if json.dumps(table, sort_keys=True).encode("utf-8") != raw:
        raise RuntimeError(f"{path} is not in the expected format; not rewriting it")
    for key, rep in reports.items():
        table[key] = [list(map(float, rep["ideal"])), list(map(float, rep["nadir"]))]
    path.write_bytes(json.dumps(table, sort_keys=True).encode("utf-8"))


def main(argv=None) -> int:
    """python -m mootation.benchmarks.make_fronts [KEY|FAMILY ...] [--no-cache | --cache-only]

    Builds the named fronts (all of them by default) and writes each one's
    frame into the registry's cache. Several builds may run as separate
    processes with --no-cache; one --cache-only run afterwards writes the
    frames of every stored front, so the shared cache file has one writer.
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    cache_only = "--cache-only" in argv
    no_cache = "--no-cache" in argv
    only = {a for a in argv if not a.startswith("--")}
    FRONT_DIR.mkdir(exist_ok=True)
    reports = {}
    for name, sizes in SIZES.items():
        if name in SHARED:
            continue
        for M in sizes:
            key = f"{name}_{M}D"
            if only and key not in only and name not in only:
                continue
            if cache_only:
                if has_front(name, M):
                    reports[key] = load(name, M)[1]
                continue
            # Half the sample from eight objectives: the nondominated filter is
            # quadratic in what survives it, and the share that does grows with
            # M — 22 % of a WFG3 sample at six, a third at ten (13 % of DTLZ5's,
            # 4 % of MaF6's). The exact test of step 4d, not the sample size,
            # is what makes a front clean; the sample decides how well covered.
            n_int, n_face = (100_000, 10_000) if M <= 6 else (50_000, 5_000)
            F, report = build_front(name, M, n_int=n_int, n_face=n_face)
            np.savez_compressed(FRONT_DIR / f"{key}.npz", F=F,
                                report=np.array(json.dumps(report)))
            reports[key] = report
    if reports and not no_cache:
        _update_frame_cache(reports)
        print(f"frames of {len(reports)} fronts written to the registry's cache")
    return 0


