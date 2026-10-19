# SPDX-License-Identifier: Apache-2.0
# ============================================================================
# MaF1, MaF3, MaF4, MaF5 — corrected definitions.
# R. Cheng et al. — "A benchmark test suite for evolutionary many-objective
# optimization", Complex & Intelligent Systems (2017),
# doi:10.1007/s40747-017-0039-7.
#
# This module supersedes the MaF1/3/4/5 entries built by _maf_register() in
# registry.py, which carried wrong objective formulae and/or wrong nadir
# points. MaF2 and MaF6-9 are left with the original registration: their
# fronts and nadirs are correct for hypervolume as they stand.
#
# Formulae (M objectives, D = M + K - 1, K = 10, x in [0,1]^D):
#   pr — the linear position products of DTLZ1
#   d  — the DTLZ2 directions, with sum d^2 = 1
# The reference front is built from uniform points on the simplex or sphere.
#
# MaF14 and MaF15, the two large-scale problems, are at the end of the module
# (task 12 of 2026-10-19, §2).
# ============================================================================
from __future__ import annotations
from typing import List
import itertools
import math
from functools import lru_cache as _lru_cache
import numpy as np

from ._rows import math_cos, math_sin

K = 10
A = 2.0       # scale factor (MaF4, MaF5)
ALPHA = 100.0 # bias (MaF5)


def _exps(M: int) -> np.ndarray:
    e = np.full(M, 4.0)
    e[M - 1] = 2.0
    return e


# ---- eval, of a batch (task 10 of 2026-10-18) -----------------------
# X holds one point per row; mafN_rows(X, M) is the (n, M) array of their
# objectives, and mafN(x, M) its batch of one row. The expressions are those
# of the point code before, elementwise and summed along the row, so that a
# row's values are bit for bit what that code gave the point.
def _rows(X, M: int) -> np.ndarray:
    return np.ascontiguousarray(X, dtype=float).reshape(-1, M + K - 1)


# g: the quadratic (MaF1, MaF5) and the Rastrigin-like (MaF3, MaF4) distance
# functions; pr: the linear DTLZ1 position products, summing to 1; d: the
# DTLZ2 directions, with sum d^2 = 1


def _g_quad_rows(XM: np.ndarray) -> np.ndarray:
    return np.sum((XM - 0.5) ** 2, axis=1)


def _g_rastrigin_rows(XM: np.ndarray) -> np.ndarray:
    return 100.0 * (XM.shape[1] + np.sum(
        (XM - 0.5) ** 2 - np.cos(20.0 * np.pi * (XM - 0.5)), axis=1))


def _dtlz1_products_rows(P: np.ndarray, M: int) -> np.ndarray:
    pr = np.empty((len(P), M))
    pr[:, 0] = np.prod(P, axis=1) if M > 1 else 1.0
    for i in range(1, M):
        pre = np.prod(P[:, :M - 1 - i], axis=1) if (M - 1 - i) > 0 else 1.0
        pr[:, i] = pre * (1.0 - P[:, M - 1 - i])
    return pr


def _dtlz2_dirs_rows(P: np.ndarray, M: int, cos=np.cos, sin=np.sin) -> np.ndarray:
    c = cos(0.5 * np.pi * P)
    s = sin(0.5 * np.pi * P)
    d = np.empty((len(P), M))
    d[:, 0] = np.prod(c, axis=1) if M > 1 else 1.0
    for i in range(1, M):
        pre = np.prod(c[:, :M - 1 - i], axis=1) if (M - 1 - i) > 0 else 1.0
        d[:, i] = pre * s[:, M - 1 - i]
    return d


def maf1_rows(X, M: int) -> np.ndarray:
    X = _rows(X, M)
    g = _g_quad_rows(X[:, M - 1:])
    return (1.0 - _dtlz1_products_rows(X[:, :M - 1], M)) * (1.0 + g)[:, None]


def maf3_rows(X, M: int) -> np.ndarray:
    X = _rows(X, M)
    g = _g_rastrigin_rows(X[:, M - 1:])
    return (_dtlz2_dirs_rows(X[:, :M - 1], M) * (1.0 + g)[:, None]) ** _exps(M)


def maf4_rows(X, M: int) -> np.ndarray:
    X = _rows(X, M)
    g = _g_rastrigin_rows(X[:, M - 1:])
    d = _dtlz2_dirs_rows(X[:, :M - 1], M)
    return A ** np.arange(1, M + 1) * (1.0 - d) * (1.0 + g)[:, None]


def maf5_rows(X, M: int) -> np.ndarray:
    X = _rows(X, M)
    g = _g_quad_rows(X[:, M - 1:])
    d = _dtlz2_dirs_rows(X[:, :M - 1] ** ALPHA, M)
    return A ** np.arange(M, 0, -1) * (d * (1.0 + g)[:, None]) ** 4


def _one(rows_fn):
    def one(x: List[float], M: int) -> List[float]:
        return rows_fn([x], M)[0].tolist()
    one.__name__ = rows_fn.__name__.replace("_rows", "")
    return one


maf1, maf3, maf4, maf5 = (_one(fn) for fn in (maf1_rows, maf3_rows, maf4_rows, maf5_rows))


# ---- simplex and sphere samplers -----------------------------------
@_lru_cache(maxsize=None)
def _simplex_cached(M: int, n: int) -> np.ndarray:
    """Memoized `_simplex`. The result is shared, so it is made read-only.

    Registration asks for the same (M, n) once per MaF problem — twenty calls
    for five distinct grids — and at M = 15 one call enumerates 3^14 tuples.
    Caching removes the repetition without touching the mathematics, which
    matters: these points ARE the reference Pareto front, so changing how they
    are generated would change every IGD number computed against them.
    """
    W = _simplex_impl(M, n)
    W.flags.writeable = False
    return W


def _simplex(M: int, n: int) -> np.ndarray:
    return _simplex_cached(M, n)


def _simplex_impl(M: int, n: int) -> np.ndarray:
    # FIX 2026-07-09: at high M the grid degenerated - about 55 points at
    # M=10, and between 15 and 120 at M=15, against the 1000 requested - so
    # the reference front for IGD+ was built from a handful of points. It is
    # now topped up to n with a uniform Dirichlet(1,...,1) draw, which IS the
    # uniform distribution on the simplex, from a fixed seed so the reference
    # stays reproducible. IGD+ values at M >= 10 are NOT comparable with
    # numbers produced before this fix.
    p = max(1, int(round(n ** (1.0 / (M - 1))))) if M > 1 else 1
    pts = [list(c) + [1.0 - sum(c)]
           for c in itertools.product(np.linspace(0, 1, p + 1), repeat=M - 1)
           if sum(c) <= 1.0 + 1e-9]
    W = np.asarray(pts, float)
    if len(W) < n:
        rng = np.random.default_rng(20260709 + 1000 * M + n)
        extra = rng.dirichlet(np.ones(M), size=n - len(W))
        W = np.vstack([W, extra])
    return W


def _sphere(M: int, n: int) -> np.ndarray:
    w = _simplex(M, n)
    nrm = np.linalg.norm(w, axis=1, keepdims=True)
    nrm[nrm == 0] = 1.0
    return w / nrm


# ---- reference fronts (g = 0) --------------------------------------
def pf_maf1(M: int, n: int = 1000) -> np.ndarray:
    return 1.0 - _simplex(M, n)                  # the inverted simplex


def pf_maf3(M: int, n: int = 1000) -> np.ndarray:
    return _sphere(M, n) ** _exps(M)             # convex, via the exponents


def pf_maf4(M: int, n: int = 1000) -> np.ndarray:
    return (A ** np.arange(1, M + 1)) * (1.0 - _sphere(M, n))


def pf_maf5(M: int, n: int = 1000) -> np.ndarray:
    return (A ** np.arange(M, 0, -1)) * (_sphere(M, n) ** 4)


# name -> (eval, pf)
MAF_FIX = {
    "MaF1": (maf1, pf_maf1),
    "MaF3": (maf3, pf_maf3),
    "MaF4": (maf4, pf_maf4),
    "MaF5": (maf5, pf_maf5),
}
# name -> the batch eval
MAF_FIX_ROWS = {"MaF1": maf1_rows, "MaF3": maf3_rows, "MaF4": maf4_rows, "MaF5": maf5_rows}


def maf_n_vars(M: int) -> int:
    return M + K - 1


# ---- MaF14 and MaF15 (task 12 of 2026-10-19, §2) ---------------------------
# MaF14 is LSMOP3 and MaF15 the inverted LSMOP8 (Cheng et al. 2017, Eq. 41-50).
# M objectives and D = 20 M variables (the paper's setting): x^f = x_1 ..
# x_{M-1}, the positions, in [0, 1]; x^s = x_M .. x_D, in [0, 10] for MaF14
# and [0, 1] for MaF15, split into M groups x^s_1 .. x^s_M of N_k = 2
# subcomponents each. Every variable of x^s is linked to x_1 first (Eq. 45, 50):
#     z_i = a_i (x^s_i - l_i) - x_1 (u_i - l_i),     i = 1 .. |x^s|,
#     a_i = 1 + i / |x^s| (MaF14),   1 + cos(0.5 pi i / |x^s|) (MaF15),
# and objective j reads a group through (Eq. 43, 48)
#     gbar_j(x^s_k) = 1/N_k  sum over the subcomponents s of x^s_k of  eta(z_s) / |s|,
# eta = eta_1 for an odd j and eta_2 for an even one: Rastrigin and Rosenbrock
# in MaF14 (Eq. 44), Griewank and the sphere in MaF15 (Eq. 49).
#   MaF14: f_j = h_j (1 + gbar_j(x^s_j)) (c = I, Eq. 42), h the position
#          products of DTLZ1 without its 1/2: the front is the simplex
#          sum f = 1 (Eq. 41).
#   MaF15: f_j = h_j (1 + (gbar_j(x^s_j) + gbar_j(x^s_{j+1}))), the second
#          term for j < M only (c_jk = 1 for k = j, j + 1, Eq. 47), h = 1 - the
#          DTLZ2 directions: the front is 1 minus the sphere's octant (Eq. 46).
# The terms of c that are 0 are left out of the sums.
#
# What the paper leaves open (the LSMOP paper, Cheng, Jin, Olhofer & Sendhoff,
# IEEE Trans. Cybern. 47(12), 2017, defines it), and the readings here:
#   * x^f and x^s: the first M - 1 variables and the rest, LSMOP's split;
#   * the bounds: Eq. 41 prints x in [0, 10]^|x| for MaF14, but positions
#     beyond 1 make h negative, and the front would not be the simplex of
#     Fig. 14 and Table 1: [0, 10] is read for x^s, [0, 1] for x^f. MaF15 as
#     printed (Eq. 46), every variable in [0, 1];
#   * the sizes of the groups and subcomponents: the LSMOP construction's
#     non-uniform groups, group k the share c_k / sum(c) of |x^s| by the
#     logistic map c_1 = 3.8 * 0.1 * (1 - 0.1), c_{k+1} = 3.8 c_k (1 - c_k),
#     rounded to whole variables by the largest remainders so that D stays
#     20 M, and cut into N_k subcomponents as equal as they come, the first
#     the larger (lsmop_parts);
#   * MaF15's second term reads group j + 1 through objective j's eta, as
#     printed (c_jk gbar_j(x^s_k)).
#
# CROSS-CHECK WITH LSMOP (task 13 of 2026-10-19, §4; the authors' version of
# Cheng et al. 2017, LSMOP_Final Sept 2016). The MaF paper rules where it
# speaks, LSMOP where it does not:
#   * groups: MaF fixes N_k = 2 and D = 20 M (after Eq. 45); the shares are
#     LSMOP's logistic map, Eq. 9-10 with alpha = 3.8, c_0 = 0.1. LSMOP Eq. 9
#     rounds each group up, |x^s_i| = ceil(r_i n_s), so its D may exceed the
#     nominal one; here the largest remainders keep D = 20 M exactly — a
#     declared difference from LSMOP;
#   * linkage: i / |x^s| over the whole x^s, as MaF Eq. 45 and 50 and LSMOP
#     L_1 (Eq. 22, MaF14) and L_2 (Eq. 23, MaF15);
#   * bounds: LSMOP uses l_i = 0, u_i = 10 for x^s (after Eq. 23), as MaF14
#     here; MaF15 keeps [0, 1] for every variable as MaF Eq. 46 prints it,
#     self-consistent with its formula;
#   * landscapes: LSMOP Table II, LSMOP3 = (L_1, H_1, g^I = eta_4 Rastrigin,
#     g^II = eta_3 Rosenbrock, C_1 = I of Eq. 19) is MaF14; LSMOP8 = (L_2,
#     H_2, eta_5 Griewank, eta_1 sphere, C_2 of Eq. 20, objective j reading
#     groups j and j + 1) is MaF15's g (MaF Eq. 47-49), MaF15's front the
#     inverted H_2;
#   * the normalisation by the subcomponent's length and N_k: LSMOP Eq. 17,
#     odd and even objectives by Eq. 18, as here;
#   * MaF14's front at M = 8, 10, 15: the simplex with front_exact False
#     (THE FRONT OF MaF14 below), the strip off it stays, the linkage being
#     the paper's.
#
# THE FRONT OF MaF14. The optimum of a Rosenbrock subcomponent (an even
# group) is z = 1, x_i = l_i + (1 + 10 x_1) / a_i, inside the box while
# x_1 <= a_i - 1/10. Group 2 starts after group 1, whose share falls below a
# tenth of |x^s| from eight objectives on: there positions with x_1 above
# lsmop_x1_max leave some of group 2 off its optimum, and the front (f_2 =
# h_2 (1 + gbar_2) with gbar_2 > 0) leaves the simplex in the strip f_M <
# 1 - lsmop_x1_max, its f_2 reaching beyond 1 at the corner. pareto_front
# gives the simplex, as the paper's Fig. 14 (no point reaches below it), with
# its frame, ideal 0 and nadir 1, front_exact False there (registry.py); and
# pareto_set positions with x_1 up to lsmop_x1_max.
NK = 2              # N_k, the subcomponents of a group


def lsmop_n_vars(M: int) -> int:
    return 20 * M


def lsmop_parts(M: int) -> list:
    """The sizes of the N_k subcomponents of each of the M groups of x^s,
    group by group: [[|x^s_11|, |x^s_12|], [|x^s_21|, |x^s_22|], ...]."""
    ns = lsmop_n_vars(M) - M + 1
    c = [3.8 * 0.1 * (1.0 - 0.1)]
    while len(c) < M:
        c.append(3.8 * c[-1] * (1.0 - c[-1]))
    share = [v / sum(c) * ns for v in c]
    size = [math.floor(v) for v in share]
    for k in sorted(range(M), key=lambda k: (size[k] - share[k], k))[:ns - sum(size)]:
        size[k] += 1
    parts = [[s // NK + (1 if j < s % NK else 0) for j in range(NK)] for s in size]
    assert sum(map(sum, parts)) == ns and min(map(min, parts)) >= 1, parts
    return parts


def lsmop_link(M: int, which: int) -> np.ndarray:
    """(a_i, l_i, u_i - l_i) of every variable of x^s, an (|x^s|, 3) array:
    MaF14's a_i = 1 + i / |x^s| over [0, 10], MaF15's 1 + cos(0.5 pi i / |x^s|)
    over [0, 1] (Eq. 45, 50)."""
    ns = lsmop_n_vars(M) - M + 1
    r = np.arange(1, ns + 1) / ns
    a = 1.0 + r if which == 14 else 1.0 + math_cos(0.5 * np.pi * r)
    span = 10.0 if which == 14 else 1.0
    return np.column_stack([a, np.zeros(ns), np.full(ns, span)])


def _lsmop_columns(M: int) -> list:
    """The columns of x^s each subcomponent takes, group by group: [[(start,
    stop), ...], ...]."""
    out, at = [], 0
    for sizes in lsmop_parts(M):
        group = []
        for s in sizes:
            group.append((at, at + s))
            at += s
        out.append(group)
    return out


def lsmop_x1_max(M: int) -> float:
    """MaF14: the largest x_1 at which every Rosenbrock subcomponent (of two
    variables or more) can reach its optimum z = 1 inside the box (THE FRONT
    OF MaF14 above); 1 where all can."""
    link = lsmop_link(M, 14)
    limit = 1.0
    for k, group in enumerate(_lsmop_columns(M)):
        if k % 2 == 1:                                  # groups 2, 4, ...: eta_2
            for s, t in group:
                if t - s >= 2:
                    a, w = link[s:t, 0], link[s:t, 2]
                    limit = min(limit, float(np.min(a - 1.0 / w)))
    return limit


def _eta_rastrigin(Z: np.ndarray) -> np.ndarray:
    return np.sum(Z ** 2 - 10.0 * math_cos(2 * np.pi * Z) + 10.0, axis=1)


def _eta_rosenbrock(Z: np.ndarray) -> np.ndarray:
    a, b = Z[:, :-1], Z[:, 1:]
    return np.sum(100.0 * (a ** 2 - b) ** 2 + (a - 1.0) ** 2, axis=1)


def _eta_griewank(Z: np.ndarray) -> np.ndarray:
    root = np.sqrt(np.arange(1, Z.shape[1] + 1))
    return (np.sum(Z ** 2 / 4000.0, axis=1)
            - np.prod(math_cos(Z / root), axis=1) + 1.0)


def _eta_sphere(Z: np.ndarray) -> np.ndarray:
    return np.sum(Z ** 2, axis=1)


def lsmop_rows(X, M: int, which: int) -> np.ndarray:
    """MaF14 (which = 14) or MaF15 (15) at the rows of X: an (n, M) array.
    cos and sin are the C library's at every element (_rows.math_cos)."""
    D = lsmop_n_vars(M)
    X = np.ascontiguousarray(X, dtype=float).reshape(-1, D)
    link = lsmop_link(M, which)
    Z = link[:, 0] * (X[:, M - 1:] - link[:, 1]) - X[:, :1] * link[:, 2]
    cols = _lsmop_columns(M)
    eta = (_eta_rastrigin, _eta_rosenbrock) if which == 14 else (_eta_griewank, _eta_sphere)

    def gbar(j, k):
        """gbar_(j+1) of group k + 1: eta_1 for objective 1, 3, ..."""
        acc = np.zeros(len(X))
        for s, t in cols[k]:
            acc = acc + eta[j % 2](np.ascontiguousarray(Z[:, s:t])) / (t - s)
        return acc / NK

    P = X[:, :M - 1]
    if which == 14:
        h = _dtlz1_products_rows(P, M)
    else:
        h = 1.0 - _dtlz2_dirs_rows(P, M, math_cos, math_sin)
    F = np.empty((len(X), M))
    for j in range(M):
        g = gbar(j, j) if which == 14 or j == M - 1 else gbar(j, j) + gbar(j, j + 1)
        F[:, j] = h[:, j] * (1.0 + g)
    return F


def pf_lsmop(M: int, which: int, n: int = 1000) -> np.ndarray:
    """MaF14: the simplex sum f = 1; MaF15: 1 minus the sphere's octant."""
    return _simplex(M, n) if which == 14 else 1.0 - _sphere(M, n)


def ps_lsmop(M: int, which: int, n: int = 1000) -> np.ndarray:
    """n points of the Pareto set: positions over [0, 1]^(M-1) (corners, then
    Halton; MaF14's x_1 scaled to [0, lsmop_x1_max]), every subcomponent at
    its eta's optimum, z = 0 (Rastrigin, Griewank, the sphere) or z = 1
    (Rosenbrock), x_i = l_i + (z + x_1 (u_i - l_i)) / a_i, kept in the box."""
    from .pareto_sets import _cube_positions
    D = lsmop_n_vars(M)
    P = _cube_positions(M - 1, n)
    if which == 14:
        P[:, 0] *= lsmop_x1_max(M)
    link = lsmop_link(M, which)
    z = np.zeros(D - M + 1)
    if which == 14:
        for k, group in enumerate(_lsmop_columns(M)):
            if k % 2 == 1:
                z[group[0][0]:group[-1][1]] = 1.0
    X = np.empty((len(P), D))
    X[:, :M - 1] = P
    X[:, M - 1:] = np.minimum(link[:, 1] + (z + P[:, :1] * link[:, 2]) / link[:, 0],
                              link[:, 1] + link[:, 2])
    return X
