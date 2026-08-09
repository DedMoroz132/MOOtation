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
# API: f = dtlzN(x, M) -> list[float] of length M.
# ============================================================================
from __future__ import annotations

import math
from typing import List

import numpy as np


# =============================================================
#  g functions
# =============================================================
def _g_dtlz1(xm: np.ndarray) -> float:
    """g_1 (multimodal; DTLZ1, DTLZ3)."""
    k = len(xm)
    return 100.0 * (k + float(np.sum(
        (xm - 0.5) ** 2 - np.cos(20.0 * math.pi * (xm - 0.5))
    )))


def _g_dtlz2(xm: np.ndarray) -> float:
    """g_2 (unimodal; DTLZ2, DTLZ4, DTLZ5)."""
    return float(np.sum((xm - 0.5) ** 2))


def _g_dtlz6(xm: np.ndarray) -> float:
    """g_6 (DTLZ6 — biased, x^0.1)."""
    return float(np.sum(xm ** 0.1))


def _g_dtlz7(xm: np.ndarray) -> float:
    """g_7 (DTLZ7)."""
    k = len(xm)
    return 1.0 + 9.0 / k * float(np.sum(xm))


# =============================================================
#  DTLZ1 — linear PF, multimodal
# =============================================================
def dtlz1(x: List[float], M: int) -> List[float]:
    x = np.asarray(x, dtype=float)
    g = _g_dtlz1(x[M - 1:])
    f = np.empty(M)
    base = 0.5 * (1.0 + g)
    # f_1 = base * x_1 * x_2 * ... * x_{M-1}
    f[0] = base * float(np.prod(x[:M - 1])) if M > 1 else base
    # f_i = base * (x_1 ... x_{M-i}) * (1 - x_{M-i+1}) for i = 2..M-1
    for i in range(2, M):
        prefix = float(np.prod(x[:M - i])) if (M - i) > 0 else 1.0
        f[i - 1] = base * prefix * (1.0 - x[M - i])
    # f_M = base * (1 - x_1)
    if M > 1:
        f[M - 1] = base * (1.0 - x[0])
    return f.tolist()


# =============================================================
#  DTLZ2 — concave (sphere) PF
# =============================================================
def _dtlz_concave(x: np.ndarray, g: float, M: int,
                  alpha: float = 1.0) -> List[float]:
    """Generic concave (DTLZ2; DTLZ4 at alpha = 100)."""
    xp = x[:M - 1] ** alpha if alpha != 1.0 else x[:M - 1]
    cos = np.cos(xp * math.pi / 2.0)
    sin = np.sin(xp * math.pi / 2.0)
    f = np.empty(M)
    base = 1.0 + g
    f[0] = base * float(np.prod(cos)) if M > 1 else base
    for i in range(2, M):
        prefix = float(np.prod(cos[:M - i])) if (M - i) > 0 else 1.0
        f[i - 1] = base * prefix * sin[M - i]
    if M > 1:
        f[M - 1] = base * sin[0]
    return f.tolist()


def dtlz2(x: List[float], M: int) -> List[float]:
    x = np.asarray(x, dtype=float)
    g = _g_dtlz2(x[M - 1:])
    return _dtlz_concave(x, g, M, alpha=1.0)


# =============================================================
#  DTLZ3 — concave PF + multimodal g
# =============================================================
def dtlz3(x: List[float], M: int) -> List[float]:
    x = np.asarray(x, dtype=float)
    g = _g_dtlz1(x[M - 1:])
    return _dtlz_concave(x, g, M, alpha=1.0)


# =============================================================
#  DTLZ4 — concave PF, biased (x^100)
# =============================================================
def dtlz4(x: List[float], M: int) -> List[float]:
    x = np.asarray(x, dtype=float)
    g = _g_dtlz2(x[M - 1:])
    return _dtlz_concave(x, g, M, alpha=100.0)


# =============================================================
#  DTLZ5 — degenerate (curve)
# =============================================================
def _dtlz_degenerate(x: np.ndarray, g: float, M: int) -> List[float]:
    theta = np.empty(M - 1)
    theta[0] = x[0] * math.pi / 2.0
    if M > 2:
        denom = 4.0 * (1.0 + g)
        for i in range(1, M - 1):
            theta[i] = (math.pi / denom) * (1.0 + 2.0 * g * x[i])
    cos = np.cos(theta)
    sin = np.sin(theta)
    f = np.empty(M)
    base = 1.0 + g
    f[0] = base * float(np.prod(cos)) if M > 1 else base
    for i in range(2, M):
        prefix = float(np.prod(cos[:M - i])) if (M - i) > 0 else 1.0
        f[i - 1] = base * prefix * sin[M - i]
    if M > 1:
        f[M - 1] = base * sin[0]
    return f.tolist()


def dtlz5(x: List[float], M: int) -> List[float]:
    x = np.asarray(x, dtype=float)
    g = _g_dtlz2(x[M - 1:])
    return _dtlz_degenerate(x, g, M)


# =============================================================
#  DTLZ6 — degenerate + biased g
# =============================================================
def dtlz6(x: List[float], M: int) -> List[float]:
    x = np.asarray(x, dtype=float)
    g = _g_dtlz6(x[M - 1:])
    return _dtlz_degenerate(x, g, M)


# =============================================================
#  DTLZ7 — disconnected PF
# =============================================================
def dtlz7(x: List[float], M: int) -> List[float]:
    x = np.asarray(x, dtype=float)
    g = _g_dtlz7(x[M - 1:])
    f = np.empty(M)
    for i in range(M - 1):
        f[i] = x[i]
    # h = M - Σ_{i=1..M-1} (f_i/(1+g)) * (1 + sin(3π f_i))
    h = float(M) - float(np.sum(
        (f[:M - 1] / (1.0 + g)) * (1.0 + np.sin(3.0 * math.pi * f[:M - 1]))
    ))
    f[M - 1] = (1.0 + g) * h
    return f.tolist()


# =============================================================
#  Registry + helpers
# =============================================================
DTLZ_FUNCS = {
    "DTLZ1": dtlz1, "DTLZ2": dtlz2, "DTLZ3": dtlz3,
    "DTLZ4": dtlz4, "DTLZ5": dtlz5, "DTLZ6": dtlz6,
    "DTLZ7": dtlz7,
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
