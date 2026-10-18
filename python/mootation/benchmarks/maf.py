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
# ============================================================================
from __future__ import annotations
from typing import List
import itertools
from functools import lru_cache as _lru_cache
import numpy as np

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


def _dtlz2_dirs_rows(P: np.ndarray, M: int) -> np.ndarray:
    c = np.cos(0.5 * np.pi * P)
    s = np.sin(0.5 * np.pi * P)
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
