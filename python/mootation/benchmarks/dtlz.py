# SPDX-License-Identifier: Apache-2.0
# ============================================================================
# DTLZ1-7, implemented exactly as printed in
# K. Deb, L. Thiele, M. Laumanns, E. Zitzler — "Scalable Test Problems for
# Evolutionary Multiobjective Optimization", 2002.
#
# Conventions:
#   * the paper's formulae are 1-indexed; the code is 0-indexed;
#   * position variables are x[0..M-2]   (x_1..x_{M-1} in the paper),
#     distance variables are x[M-1..n-1] (x_M..x_n in the paper),
#     so k = n - M + 1 is the size of the distance block;
#   * standard k: DTLZ1 -> 5, DTLZ2..6 -> 10, DTLZ7 -> 20;
#   * every variable lies in [0, 1].
#
# API: f = dtlzN(x, M) -> list[float] of length M; F = dtlzN_rows(X, M)
# -> an (n, M) array of the rows of X (DTLZ_ROWS).
# ============================================================================
from __future__ import annotations

import math
from typing import List

import numpy as np


# One implementation, of a batch (task 10 of 2026-10-18, §4.1): X holds one
# point per row and dtlzN_rows(X, M) is the (n, M) array of their objectives;
# dtlzN(x, M) is its batch of one row. The expressions are those of the point
# code before, elementwise and summed along the row, so that a row's values
# are bit for bit what that code gave the point. The g functions take one
# point's distance block (and give a float) or a batch's (and give an array).

# =============================================================
#  g functions
# =============================================================
def _point_or_rows(v):
    return float(v) if np.ndim(v) == 0 else v


def _g_dtlz1(xm: np.ndarray):
    """g_1 (multimodal; DTLZ1, DTLZ3)."""
    k = xm.shape[-1]
    return _point_or_rows(100.0 * (k + np.sum(
        (xm - 0.5) ** 2 - np.cos(20.0 * math.pi * (xm - 0.5)), axis=-1)))


def _g_dtlz2(xm: np.ndarray):
    """g_2 (unimodal; DTLZ2, DTLZ4, DTLZ5)."""
    return _point_or_rows(np.sum((xm - 0.5) ** 2, axis=-1))


def _g_dtlz6(xm: np.ndarray):
    """g_6 (DTLZ6 — biased, x^0.1)."""
    return _point_or_rows(np.sum(xm ** 0.1, axis=-1))


def _g_dtlz7(xm: np.ndarray):
    """g_7 (DTLZ7)."""
    k = xm.shape[-1]
    return _point_or_rows(1.0 + 9.0 / k * np.sum(xm, axis=-1))


def _rows(x) -> np.ndarray:
    """X as a C-ordered (n, d) array (a row's sums then run as a point's do)."""
    X = np.ascontiguousarray(x, dtype=float)
    return X.reshape(1, -1) if X.ndim == 1 else X


def _one(rows_fn):
    """The point function of a batch function: its batch of one row, as a list."""
    def one(x: List[float], M: int) -> List[float]:
        return rows_fn(_rows(x), M)[0].tolist()
    one.__name__ = rows_fn.__name__.replace("_rows", "")
    return one


def _products(c: np.ndarray, last: np.ndarray, base: np.ndarray, M: int) -> np.ndarray:
    """f_1 = base prod(c_1..c_{M-1}), f_i = base prod(c_1..c_{M-i}) last_{M-i+1}
    (i = 2..M-1), f_M = base last_1: the shape every DTLZ but 7 shares."""
    F = np.empty((len(base), M))
    F[:, 0] = base * np.prod(c[:, :M - 1], axis=1) if M > 1 else base
    for i in range(2, M):
        F[:, i - 1] = base * np.prod(c[:, :M - i], axis=1) * last[:, M - i]
    if M > 1:
        F[:, M - 1] = base * last[:, 0]
    return F


# =============================================================
#  DTLZ1 — linear PF, multimodal
# =============================================================
def dtlz1_rows(X: np.ndarray, M: int) -> np.ndarray:
    X = _rows(X)
    base = 0.5 * (1.0 + _g_dtlz1(X[:, M - 1:]))
    # f_1 = base * x_1 * ... * x_{M-1}; f_i = base * (x_1 ... x_{M-i}) * (1 - x_{M-i+1})
    # for i = 2..M-1; f_M = base * (1 - x_1)
    return _products(X, 1.0 - X[:, :M - 1], base, M)


# =============================================================
#  DTLZ2 — concave (sphere) PF
# =============================================================
def _dtlz_concave(X: np.ndarray, g: np.ndarray, M: int, alpha: float = 1.0) -> np.ndarray:
    """Generic concave (DTLZ2; DTLZ4 at alpha = 100)."""
    XP = X[:, :M - 1] ** alpha if alpha != 1.0 else X[:, :M - 1]
    return _products(np.cos(XP * math.pi / 2.0), np.sin(XP * math.pi / 2.0), 1.0 + g, M)


def dtlz2_rows(X: np.ndarray, M: int) -> np.ndarray:
    X = _rows(X)
    return _dtlz_concave(X, _g_dtlz2(X[:, M - 1:]), M, alpha=1.0)


# =============================================================
#  DTLZ3 — concave PF + multimodal g
# =============================================================
def dtlz3_rows(X: np.ndarray, M: int) -> np.ndarray:
    X = _rows(X)
    return _dtlz_concave(X, _g_dtlz1(X[:, M - 1:]), M, alpha=1.0)


# =============================================================
#  DTLZ4 — concave PF, biased (x^100)
# =============================================================
def dtlz4_rows(X: np.ndarray, M: int) -> np.ndarray:
    X = _rows(X)
    return _dtlz_concave(X, _g_dtlz2(X[:, M - 1:]), M, alpha=100.0)


# =============================================================
#  DTLZ5 — degenerate (curve)
# =============================================================
def _dtlz_degenerate(X: np.ndarray, g: np.ndarray, M: int) -> np.ndarray:
    # Deb et al. 2002, Eq. 6.23 prints f = ... cos(theta_i pi/2) ... with
    # theta_i = pi/(4(1+g)) (1 + 2 g x_i), i.e. a pi inside theta AND a pi/2
    # outside it, which at g = 0 would give cos(pi^2/8) and no curve at all.
    # Every reference implementation (PlatEMO, jMetal, pymoo) reads it as
    # theta_1 = pi x_1 / 2, theta_i = pi/(4(1+g)) (1 + 2 g x_i), f = cos/sin
    # of theta directly, so that theta_i = pi/4 on the front; that is the
    # reading used here.
    theta = np.empty((len(X), M - 1))
    theta[:, 0] = X[:, 0] * math.pi / 2.0
    if M > 2:
        denom = 4.0 * (1.0 + g)
        for i in range(1, M - 1):
            theta[:, i] = (math.pi / denom) * (1.0 + 2.0 * g * X[:, i])
    return _products(np.cos(theta), np.sin(theta), 1.0 + g, M)


def dtlz5_rows(X: np.ndarray, M: int) -> np.ndarray:
    X = _rows(X)
    return _dtlz_degenerate(X, _g_dtlz2(X[:, M - 1:]), M)


# =============================================================
#  DTLZ6 — degenerate + biased g
# =============================================================
def dtlz6_rows(X: np.ndarray, M: int) -> np.ndarray:
    X = _rows(X)
    return _dtlz_degenerate(X, _g_dtlz6(X[:, M - 1:]), M)


# =============================================================
#  DTLZ7 — disconnected PF
# =============================================================
def dtlz7_rows(X: np.ndarray, M: int) -> np.ndarray:
    X = _rows(X)
    g = _g_dtlz7(X[:, M - 1:])
    F = np.empty((len(X), M))
    F[:, :M - 1] = X[:, :M - 1]
    # h = M - Σ_{i=1..M-1} (f_i/(1+g)) * (1 + sin(3π f_i))
    P = F[:, :M - 1]
    h = float(M) - np.sum((P / (1.0 + g)[:, None]) * (1.0 + np.sin(3.0 * math.pi * P)), axis=1)
    F[:, M - 1] = (1.0 + g) * h
    return F


dtlz1, dtlz2, dtlz3, dtlz4, dtlz5, dtlz6, dtlz7 = (
    _one(f) for f in (dtlz1_rows, dtlz2_rows, dtlz3_rows, dtlz4_rows, dtlz5_rows, dtlz6_rows,
                      dtlz7_rows))


# =============================================================
#  Registry + helpers
# =============================================================
DTLZ_FUNCS = {
    "DTLZ1": dtlz1, "DTLZ2": dtlz2, "DTLZ3": dtlz3,
    "DTLZ4": dtlz4, "DTLZ5": dtlz5, "DTLZ6": dtlz6,
    "DTLZ7": dtlz7,
}
# The same, of a batch: an (n, M) array of the rows of X.
DTLZ_ROWS = {
    "DTLZ1": dtlz1_rows, "DTLZ2": dtlz2_rows, "DTLZ3": dtlz3_rows,
    "DTLZ4": dtlz4_rows, "DTLZ5": dtlz5_rows, "DTLZ6": dtlz6_rows,
    "DTLZ7": dtlz7_rows,
}


# The standard k, per Deb 2002.
DTLZ_K = {
    "DTLZ1": 5,
    "DTLZ2": 10, "DTLZ3": 10, "DTLZ4": 10,
    "DTLZ5": 10, "DTLZ6": 10,
    "DTLZ7": 20,
}


def dtlz_n_vars(name: str, M: int) -> int:
    """n = M + k - 1 for the DTLZ problems."""
    return M + DTLZ_K[name] - 1


def dtlz_nadir(name: str, M: int) -> tuple:
    """Theoretical nadir per Deb 2002."""
    if name == "DTLZ1":
        return tuple([0.5] * M)
    if name == "DTLZ7":
        # f_i in [0,1] for i < M; f_M is bounded above by about 2M (Deb 2002).
        return tuple([1.0] * (M - 1) + [2.0 * M])
    # DTLZ2..6: f_i ∈ [0,1]
    return tuple([1.0] * M)


def dtlz_ideal(name: str, M: int) -> tuple:
    return tuple([0.0] * M)
