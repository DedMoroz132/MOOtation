# SPDX-License-Identifier: Apache-2.0
# ============================================================================
# ZCAT1-20 — "Challenging test problems for multi- and many-objective
# optimization".
# S. Zapotecas-Martínez, C. A. Coello Coello, H. E. Aguirre, K. Tanaka —
# Swarm and Evolutionary Computation 81 (2023) 101350.
# doi:10.1016/j.swevo.2023.101350
# (source: zcat_zapotecas-martinez2023, and its supplementary material,
# zcat_zapotecas-martinez2023_supplement)
#
# Written from the paper. The authors' reference code is GPL-3.0 and is not
# read or ported here.
#
# THE FRAMEWORK (Eq. 1-10). Every problem is
#     f_i(y) = alpha_i(y_I) + beta_i(y_II - g(y_I|m)),   i = 1..M
# over Omega = prod_i [-i/2, i/2], normalised to y_i = (x_i - a_i)/(b_i - a_i).
#   * y_I = (y_1..y_m) are the position variables, y_II the distance ones;
#   * alpha_i = i^2 * F_i(y_I) fixes the Pareto front (Table 2, one F per
#     problem), so the front is {(alpha_1..alpha_M)} over the Pareto set of the
#     alpha-only problem — Theorem 1 — and the sampler below therefore keeps
#     the nondominated part of that image;
#   * g (Table 3, g0..g10) fixes the Pareto set: on it y_II = g(y_I|m);
#   * beta_i = i^2 * Z_Level(w|J_i) is the distance to the front, with the
#     modular index sets J_i = {j : (j - m - i) mod M = 0} (Eq. 8), the
#     difficulty levels Z_1..Z_6 of Table 4 (K = 5) and, with Bias on,
#     w_j = |z_j|^0.05 (Eq. 11). Imbalance replaces Level by Z_4 on even
#     objectives and Z_1 on odd ones (Eq. 9).
# Defaults are the paper's (Section 5.1): n = 10M, Level = 1,
# Complicated_PS = True, Bias = False, Imbalance = False. Section 4.1 gives the
# ideal point (0,..,0) and the nadir (1^2, 2^2, .., M^2); it derives that nadir
# from F in [0,1]^M, so in principle it is the corner of the box the front lies
# in rather than the front's own maximum. Measured, the two coincide: every
# objective of all 20 problems reaches its bound on the sampled front at M = 3
# and at M = 10, once pareto_front samples the cube's corners and keeps the
# per-objective extremes when it thins.
#
# READINGS (points the paper leaves open, resolved here and declared):
#   ZCAT-1. Table 4 prints Z_2 and Z_4 without absolute-value bars, which would
#     make them negative and their optimum wrong. Every Z must be >= 0 with its
#     only zero at w = 0 (Section 4.3.1), the worked example in Appendix A.3 of
#     the supplement writes max{|.|}, and Figure D.4 describes Z_2 as V-shaped.
#     |w_j| is used in both.
#   ZCAT-2. The constant A of ZCAT19's F_M is defined nowhere in the paper or
#     the supplement. A = 5 is used here: F_M = 1 - ybar + sin(2*A*pi*ybar)/
#     (2*A*pi) has zero derivative at ybar = k/A, so A = 5 puts its flat parts
#     exactly on 0, 0.2, ..., 1 — the boundaries of the regions in which ZCAT19
#     switches m (it is degenerate on [0, 0.2] and [0.4, 0.6]) — and an integer
#     A is what keeps F_M inside [0, 1] as Section 4.1 requires.
#   ZCAT-3. Table 2 writes the topology call as g(y_I|M-1) even for the
#     problems whose m is 1. The supplement's ZCAT20 example calls it with the
#     m of the region (g(y_1|1)), which is also the only reading under which
#     the Pareto set has n - m distance components, so g is called with the
#     actual m.
#   ZCAT-4. In ZCAT17 and ZCAT18 the F of the degenerate region is printed with
#     a free index ("y_i, for all y_i <= 0.5"). Section 4.1.4 says the region
#     maps different position vectors onto one point of the front, so inside it
#     every F_j is y_1.
#   ZCAT-5. ZCAT5's and ZCAT17's F_M print exp(mu)^8 over exp(1)^8. Read as
#     e^(8*mu): the alternative, exp(mu^8), does not reach the printed
#     denominator at mu = 1 and would leave F_M outside [0, 1].
#
# THE DECEPTIVE LEVELS, which look like a bug and are not. Z_5 and Z_6 carry a
# |w|^0.002 term, which is 0.93 already at |w| = 1e-17: beta falls to zero only
# at exactly w = 0 and sits on a plateau of about 10*i^2 everywhere else. A run
# in floating point therefore never lands exactly on the front at Level 5 or 6.
# That is the deception the "D" of Table 4 means, and it is why the tests check
# those two levels on the Z function itself rather than through the variables.
# ============================================================================
from __future__ import annotations

import math
from functools import lru_cache
from typing import List

import numpy as np

K = 5           # Table 4: the number of disconnections (Sections 4.1.2-4.1.3)
A = 5           # ZCAT19, see ZCAT-2 above
GAMMA = 0.05    # Eq. 11, the bias exponent
HALF_PI = math.pi / 2.0

DEGENERATE = ("ZCAT14", "ZCAT15", "ZCAT16")          # m = 1 (Section 4.1.3)
REGION_M = {"ZCAT19": ((0.0, 0.2), (0.4, 0.6)),      # m = 1 inside these
            "ZCAT20": ((0.1, 0.4), (0.6, 0.9))}


# ---- position functions F (Table 2) ---------------------------------
def _wavy(y1: float) -> float:
    """The disconnected F_M of ZCAT11-13, 15, 16."""
    return (math.cos((2 * K - 1) * y1 * math.pi) + 2 * y1 + 4 * K * (1 - y1) - 1) / (4 * K)


def _f_zcat1(y: np.ndarray, M: int) -> np.ndarray:
    s = np.sin(y * HALF_PI)
    c = np.cos(y * HALF_PI)
    F = np.empty(M)
    F[0] = float(np.prod(s[:M - 1])) if M > 1 else 1.0
    for j in range(2, M):
        F[j - 1] = float(np.prod(s[:M - j])) * c[M - j]
    F[M - 1] = 1.0 - s[0]
    return F


def _f_zcat2(y: np.ndarray, M: int) -> np.ndarray:
    a = 1.0 - np.cos(y * HALF_PI)
    b = 1.0 - np.sin(y * HALF_PI)
    F = np.empty(M)
    F[0] = float(np.prod(a[:M - 1])) if M > 1 else 1.0
    for j in range(2, M):
        F[j - 1] = float(np.prod(a[:M - j])) * b[M - j]
    F[M - 1] = 1.0 - math.sin(y[0] * HALF_PI)
    return F


def _f_zcat3(y: np.ndarray, M: int) -> np.ndarray:
    F = np.empty(M)
    F[0] = float(np.sum(y[:M - 1])) / (M - 1)
    for j in range(2, M):
        F[j - 1] = (float(np.sum(y[:M - j])) + (1.0 - y[M - j])) / (M - j + 1)
    F[M - 1] = 1.0 - y[0]
    return F


def _f_zcat4(y: np.ndarray, M: int) -> np.ndarray:
    F = np.empty(M)
    F[:M - 1] = y[:M - 1]
    F[M - 1] = 1.0 - float(np.sum(y[:M - 1])) / (M - 1)
    return F


def _f_zcat5(y: np.ndarray, M: int) -> np.ndarray:
    F = np.empty(M)
    F[:M - 1] = y[:M - 1]
    mu = float(np.sum(1.0 - y[:M - 1])) / (M - 1)
    F[M - 1] = (math.exp(8.0 * mu) - 1.0) / (math.exp(8.0) - 1.0)
    return F


def _f_zcat6(y: np.ndarray, M: int) -> np.ndarray:
    kappa, rho = 40.0, 0.05
    mu = float(np.sum(y[:M - 1])) / (M - 1)
    num = (1 + math.exp(2 * kappa * mu - kappa)) ** -1 - rho * mu \
        - (1 + math.exp(kappa)) ** -1 + rho
    den = (1 + math.exp(-kappa)) ** -1 - (1 + math.exp(kappa)) ** -1 + rho
    F = np.empty(M)
    F[:M - 1] = y[:M - 1]
    F[M - 1] = num / den
    return F


def _f_zcat7(y: np.ndarray, M: int) -> np.ndarray:
    F = np.empty(M)
    F[:M - 1] = y[:M - 1]
    F[M - 1] = float(np.sum((0.5 - y[:M - 1]) ** 5)) / (2 * (M - 1) * 0.5 ** 5) + 0.5
    return F


def _f_zcat8(y: np.ndarray, M: int) -> np.ndarray:
    a = 1.0 - np.sin(y * HALF_PI)
    b = 1.0 - np.cos(y * HALF_PI)
    F = np.empty(M)
    F[0] = 1.0 - (float(np.prod(a[:M - 1])) if M > 1 else 1.0)
    for j in range(2, M):
        F[j - 1] = 1.0 - float(np.prod(a[:M - j])) * b[M - j]
    F[M - 1] = math.cos(y[0] * HALF_PI)
    return F


def _f_zcat9(y: np.ndarray, M: int) -> np.ndarray:
    s = np.sin(y * HALF_PI)
    c = np.cos(y * HALF_PI)
    F = np.empty(M)
    F[0] = float(np.sum(s[:M - 1])) / (M - 1)
    for j in range(2, M):
        F[j - 1] = (float(np.sum(s[:M - j])) + c[M - j]) / (M - j + 1)
    F[M - 1] = math.cos(y[0] * HALF_PI)
    return F


def _f_zcat10(y: np.ndarray, M: int) -> np.ndarray:
    rho = 0.02
    mu = float(np.sum(1.0 - y[:M - 1])) / (M - 1)
    F = np.empty(M)
    F[:M - 1] = y[:M - 1]
    F[M - 1] = (rho ** -1 - (mu + rho) ** -1) / (rho ** -1 - (1 + rho) ** -1)
    return F


def _f_zcat11(y: np.ndarray, M: int) -> np.ndarray:
    F = _f_zcat3(y, M)
    F[M - 1] = _wavy(y[0])
    return F


def _f_zcat12(y: np.ndarray, M: int) -> np.ndarray:
    a = 1.0 - y
    F = np.empty(M)
    F[0] = 1.0 - (float(np.prod(a[:M - 1])) if M > 1 else 1.0)
    for j in range(2, M):
        F[j - 1] = 1.0 - float(np.prod(a[:M - j])) * y[M - j]
    F[M - 1] = _wavy(y[0])
    return F


def _f_zcat13(y: np.ndarray, M: int) -> np.ndarray:
    s = np.sin(y * HALF_PI)
    c = np.cos(y * HALF_PI)
    F = np.empty(M)
    F[0] = 1.0 - float(np.sum(s[:M - 1])) / (M - 1)
    for j in range(2, M):
        F[j - 1] = 1.0 - (float(np.sum(s[:M - j])) + c[M - j]) / (M - j + 1)
    F[M - 1] = 1.0 - _wavy(y[0])
    return F


def _f_zcat14(y: np.ndarray, M: int) -> np.ndarray:
    s1 = math.sin(y[0] * HALF_PI)
    F = np.empty(M)
    F[0] = s1 ** 2
    for j in range(2, M - 1):
        F[j - 1] = s1 ** (2.0 + (j - 1) / (M - 2))
    if M > 2:
        F[M - 2] = 0.5 * (1.0 + math.sin(6.0 * y[0] * HALF_PI - HALF_PI))
    F[M - 1] = math.cos(y[0] * HALF_PI)
    return F


def _f_zcat15(y: np.ndarray, M: int) -> np.ndarray:
    F = np.empty(M)
    for j in range(1, M):
        F[j - 1] = y[0] ** (1.0 + (j - 1) / (4.0 * M))
    F[M - 1] = _wavy(y[0])
    return F


def _f_zcat16(y: np.ndarray, M: int) -> np.ndarray:
    s1 = math.sin(y[0] * HALF_PI)
    F = np.empty(M)
    F[0] = s1
    for j in range(2, M - 1):
        F[j - 1] = s1 ** (1.0 + (j - 1) / (M - 2))
    if M > 2:
        F[M - 2] = 0.5 * (1.0 + math.sin(10.0 * y[0] * HALF_PI - HALF_PI))
    F[M - 1] = _wavy(y[0])
    return F


def _f_zcat17(y: np.ndarray, M: int) -> np.ndarray:
    F = np.empty(M)
    if np.all(y[:M - 1] <= 0.5):                       # ZCAT-4: the degenerate region
        F[:M - 1] = y[0]
        F[M - 1] = (math.exp(8.0 * (1.0 - y[0])) - 1.0) / (math.exp(8.0) - 1.0)
        return F
    F[:M - 1] = y[:M - 1]
    mu = float(np.sum(1.0 - y[:M - 1])) / (M - 1)
    F[M - 1] = (math.exp(8.0 * mu) - 1.0) / (math.exp(8.0) - 1.0)
    return F


def _quintic(v: float) -> float:
    return ((0.5 - v) ** 5 + 0.5 ** 5) / (2.0 * 0.5 ** 5)


def _f_zcat18(y: np.ndarray, M: int) -> np.ndarray:
    F = np.empty(M)
    if np.all(y[:M - 1] <= 0.4) or np.all(y[:M - 1] >= 0.6):
        F[:M - 1] = y[0]
        F[M - 1] = _quintic(y[0])
        return F
    F[:M - 1] = y[:M - 1]
    F[M - 1] = float(np.sum((0.5 - y[:M - 1]) ** 5)) / (2 * (M - 1) * 0.5 ** 5) + 0.5
    return F


def _f_zcat19(y: np.ndarray, M: int) -> np.ndarray:
    F = np.empty(M)
    if _region_m("ZCAT19", y, M) == 1:
        F[:M - 1] = y[0]
        ybar = y[0]
    else:
        F[:M - 1] = y[:M - 1]
        ybar = float(np.sum(y[:M - 1])) / (M - 1)
    F[M - 1] = 1.0 - ybar - math.cos(2 * A * math.pi * ybar + HALF_PI) / (2 * A * math.pi)
    return F


def _f_zcat20(y: np.ndarray, M: int) -> np.ndarray:
    F = np.empty(M)
    if _region_m("ZCAT20", y, M) == 1:
        F[:M - 1] = y[0]
        F[M - 1] = _quintic(y[0])
        return F
    F[:M - 1] = y[:M - 1]
    F[M - 1] = float(np.sum((0.5 - y[:M - 1]) ** 5)) / (2 * (M - 1) * 0.5 ** 5) + 0.5
    return F


F_FUNCS: dict = {
    "ZCAT1": _f_zcat1, "ZCAT2": _f_zcat2, "ZCAT3": _f_zcat3, "ZCAT4": _f_zcat4,
    "ZCAT5": _f_zcat5, "ZCAT6": _f_zcat6, "ZCAT7": _f_zcat7, "ZCAT8": _f_zcat8,
    "ZCAT9": _f_zcat9, "ZCAT10": _f_zcat10, "ZCAT11": _f_zcat11,
    "ZCAT12": _f_zcat12, "ZCAT13": _f_zcat13, "ZCAT14": _f_zcat14,
    "ZCAT15": _f_zcat15, "ZCAT16": _f_zcat16, "ZCAT17": _f_zcat17,
    "ZCAT18": _f_zcat18, "ZCAT19": _f_zcat19, "ZCAT20": _f_zcat20,
}

# Table 2: which Pareto set topology each problem uses when Complicated_PS.
G_OF = {
    "ZCAT1": 4, "ZCAT2": 5, "ZCAT3": 2, "ZCAT4": 7, "ZCAT5": 9, "ZCAT6": 4,
    "ZCAT7": 5, "ZCAT8": 2, "ZCAT9": 7, "ZCAT10": 9, "ZCAT11": 3, "ZCAT12": 10,
    "ZCAT13": 1, "ZCAT14": 6, "ZCAT15": 8, "ZCAT16": 10, "ZCAT17": 1,
    "ZCAT18": 8, "ZCAT19": 6, "ZCAT20": 3,
}


# ---- the number of position variables -------------------------------
def _region_m(name: str, y: np.ndarray, M: int) -> int:
    lo1, hi1 = REGION_M[name][0]
    lo2, hi2 = REGION_M[name][1]
    y1 = float(y[0])
    return 1 if (lo1 <= y1 <= hi1 or lo2 <= y1 <= hi2) else M - 1


def position_count(name: str, y: np.ndarray, M: int) -> int:
    """m for this problem, which two of them decide per region (Section 4.1.4)."""
    if name in DEGENERATE:
        return 1
    if name in REGION_M:
        return _region_m(name, y, M)
    return M - 1


# ---- Pareto set topologies g0..g10 (Table 3) ------------------------
# Each g_j sums over the m position variables, so the whole vector is one sum
# over the (n - m) x m matrix theta_j + c*y_i rather than a Python loop of tiny
# numpy calls: at M = 10 that loop ran 91 times per evaluation, and an
# evaluation is what a run spends its whole budget on.
@lru_cache(maxsize=64)
def _theta(m: int, n: int) -> np.ndarray:
    j = np.arange(m + 1, n + 1, dtype=float)          # 1-based j, as printed
    th = (2.0 * np.pi * (j - (m + 1)) / n).reshape(-1, 1)
    th.setflags(write=False)                          # it is shared, not copied
    return th


def topology(which: int, y_I: np.ndarray, m: int, n: int) -> np.ndarray:
    """g_j for j = m+1..n; `which` is 0..10 of Table 3."""
    t = _theta(m, n)                                  # (n-m, 1)
    y = y_I.reshape(1, -1)                            # (1, m)
    if which == 0:
        return np.full(n - m, 0.2210)
    if which == 1:
        return np.sin(1.5 * np.pi * y + t).sum(1) / (2 * m) + 0.5
    if which == 2:
        return (y ** 2 * np.sin(4.5 * np.pi * y + t)).sum(1) / (2 * m) + 0.5
    if which == 3:
        return (np.cos(np.pi * y + t) ** 2).sum(1) / m
    if which == 4:
        s = float(np.sum(y_I))
        return (y * np.cos(4.0 * np.pi * s / m + t)).sum(1) / (2 * m) + 0.5
    if which == 5:
        return (np.sin(2.0 * np.pi * y - 1.0 + t) ** 3).sum(1) / (2 * m) + 0.5
    if which == 6:
        rms = math.sqrt(float(np.sum(y_I ** 2)) / m)
        num0 = -10.0 * math.exp(-0.4 * rms) + 10.0 + math.e
        den = -10.0 * math.exp(-0.4) - math.exp(-1.0) + 10.0 + math.e
        return (num0 - np.exp((np.cos(11.0 * np.pi * y + t) ** 3).sum(1) / m)) / den
    if which == 7:
        mu = float(np.sum(y_I)) / m
        den = 1.0 + math.e - math.exp(-1.0)
        return (mu + np.exp(np.sin(7.0 * np.pi * mu - HALF_PI + t[:, 0]))
                - math.exp(-1.0)) / den
    if which == 8:
        return np.abs(np.sin(2.5 * np.pi * (y - 0.5) + t)).sum(1) / m
    if which == 9:
        s = float(np.sum(y_I)) / (2 * m)
        return (s - np.abs(np.sin(2.5 * np.pi * y - HALF_PI + t)).sum(1) / (2 * m)
                + 0.5)
    if which == 10:
        return np.sin((4.0 * y - 2.0) * np.pi + t).sum(1) ** 3 / (2.0 * m ** 3) + 0.5
    raise ValueError(f"unknown ZCAT topology g{which}")


# ---- difficulty levels Z_1..Z_6 (Table 4) ---------------------------
def level_value(level: int, w: np.ndarray) -> float:
    """Z_level over the w of one index set J_i; zero exactly at w = 0."""
    if len(w) == 0:
        return 0.0
    aw = np.abs(w)                                    # ZCAT-1
    if level == 1:
        return 10.0 * float(np.mean(w ** 2))
    if level == 2:
        return 10.0 * float(np.max(aw))
    if level == 3:
        return 10.0 * float(np.mean((w ** 2 - np.cos((2 * K - 1) * np.pi * w) + 1.0) / 3.0))
    if level == 4:
        left = math.exp(float(np.max(aw)) ** 0.5)
        right = math.exp(float(np.mean(0.5 * (np.cos((2 * K - 1) * np.pi * w) + 1.0))))
        return 10.0 / (2 * math.e - 2) * (left - right - 1.0 + math.e)
    if level == 5:
        return -0.7 * level_value(3, w) + 10.0 * float(np.mean(aw ** 0.002))
    if level == 6:
        return -0.7 * level_value(4, w) + 10.0 * float(np.mean(aw)) ** 0.002
    raise ValueError(f"unknown ZCAT level {level}")


@lru_cache(maxsize=64)
def _index_sets(M: int, m: int, n: int) -> tuple:
    """J_i of Eq. 8, as positions inside y_II. Fixed per (M, m, n), so cached."""
    j = np.arange(m + 1, n + 1)                       # 1-based j
    return tuple(np.nonzero((j - m - i) % M == 0)[0] for i in range(1, M + 1))


@lru_cache(maxsize=64)
def _scales(M: int, n: int) -> tuple:
    """(i^2 for i = 1..M, i for i = 1..n): the objective and variable scalings."""
    return (np.arange(1, M + 1, dtype=float) ** 2, np.arange(1, n + 1, dtype=float))


# ---- one evaluation -------------------------------------------------
def evaluate(name: str, x, M: int, *, level: int = 1, complicated_ps: bool = True,
             bias: bool = False, imbalance: bool = False) -> List[float]:
    x = np.asarray(x, dtype=float)
    n = len(x)
    scale, idx = _scales(M, n)
    y = np.clip(x / idx + 0.5, 0.0, 1.0)              # a_i = -i/2, b_i - a_i = i
    m = position_count(name, y, M)
    y_I, y_II = y[:m], y[m:]

    # Every F reads y_1..y_{M-1} only, and the two region problems decide their
    # own m from y_1, so the whole y goes in.
    alpha = scale * F_FUNCS[name](y, M)
    g = topology(G_OF[name] if complicated_ps else 0, y_I, m, n)   # ZCAT-3
    z = y_II - g
    w = np.abs(z) ** GAMMA if bias else z

    beta = np.empty(M)
    for i, js in enumerate(_index_sets(M, m, n), start=1):
        lv = (4 if i % 2 == 0 else 1) if imbalance else level      # Eq. 9
        beta[i - 1] = (i ** 2) * level_value(lv, w[js])
    return (alpha + beta).tolist()


def n_vars(M: int) -> int:
    """Section 5.1: n = 10M by default."""
    return 10 * M


def bounds(n: int) -> List[tuple]:
    return [(-(i + 1) / 2.0, (i + 1) / 2.0) for i in range(n)]


def ideal(M: int) -> tuple:
    return tuple([0.0] * M)


def nadir(M: int) -> tuple:
    """Section 4.1: F in [0,1]^M and alpha_i = i^2 F_i."""
    return tuple(float(i * i) for i in range(1, M + 1))


# ---- reference front ------------------------------------------------
def _nondominated_mask(F: np.ndarray) -> np.ndarray:
    keep = np.ones(len(F), bool)
    for i in range(len(F)):
        if not keep[i]:
            continue
        dom = np.all(F <= F[i], axis=1) & np.any(F < F[i], axis=1)
        if np.any(dom):
            keep[i] = False
    return keep


def _nondominated(F: np.ndarray) -> np.ndarray:
    return F[_nondominated_mask(F)]


def _images(name: str, M: int, P: np.ndarray) -> np.ndarray:
    scale = np.arange(1, M + 1, dtype=float) ** 2
    return np.array([scale * F_FUNCS[name](y, M) for y in P])


def _strictly_dominated_by(C: np.ndarray, V: np.ndarray, block: int = 256) -> np.ndarray:
    out = np.zeros(len(C), bool)
    for a in range(0, len(C), block):
        c = C[a:a + block, None, :]
        out[a:a + block] = np.any(np.all(V[None] <= c, axis=2) & np.any(V[None] < c, axis=2),
                                  axis=1)
    return out


def _dominated_elsewhere(name: str, M: int, Y: np.ndarray, F: np.ndarray,
                         samples: int = 60_000) -> np.ndarray:
    """Which rows of the sample's nondominated set something outside it beats.

    Nondominated WITHIN the sample is not Pareto-optimal: next to the gaps of
    a disconnected front, a candidate the sample happened not to beat is
    beaten by a position vector it did not contain. Measured 2026-09-22 at
    n_points = 1000, against 50 000 fresh points: 164 of the 1000 front points
    of ZCAT11_5D were dominated (by up to 1.48 in f_5, whose span is 25), 98 of
    ZCAT12_5D, 66 of ZCAT13_5D (up to 3.72), 24 of ZCAT11_3D. So every row is
    checked against an independent sample of position vectors and against its
    own neighbours — each position moved alone, both ways, by 0.05 down to
    0.001, and 64 Gaussian moves of all of them at once — and dropped when any
    of them strictly dominates it. What is left is not zero: a fresh sample of
    50 000 still beats 1 of ZCAT12_5D's 1000 points (by 0.37) and 2 of
    ZCAT13_5D's (by 0.27), next to the gaps, where a dominator sits across a
    gap no sample and no move reached. The draws come from their own generator,
    so a front in which nothing is dropped is bit-identical to the unchecked
    one; and only the points the thinning keeps are checked (88 moves each at
    three objectives), which is what keeps a front at seconds.
    """
    rng = np.random.default_rng(20260922 + 100 * M)
    d = 1 if name in DEGENERATE else M - 1            # the positions that vary
    width = Y.shape[1]
    P = np.zeros((samples, width))
    P[:, :d] = rng.random((samples, d))
    bad = _strictly_dominated_by(F, _images(name, M, P))
    moves = []
    for step in (0.05, 0.02, 0.01, 0.005, 0.002, 0.001):
        for j in range(d):
            for sgn in (-1.0, 1.0):
                Q = Y.copy()
                Q[:, j] = np.clip(Q[:, j] + sgn * step, 0.0, 1.0)
                moves.append(Q)
    for s in (0.2, 0.1, 0.05, 0.02):
        for _ in range(16):
            Q = Y.copy()
            Q[:, :d] = np.clip(Q[:, :d] + rng.normal(0.0, s, (len(Y), d)), 0.0, 1.0)
            moves.append(Q)
    for Q in moves:
        G = _images(name, M, Q)
        bad |= np.all(G <= F, axis=1) & np.any(G < F, axis=1)
    return bad


def pareto_front(name: str, M: int, n_points: int = 1000) -> np.ndarray:
    """The nondominated part of {(alpha_1..alpha_M)(y_I) : y_I in [0,1]^m}.

    Computed once per (problem, M, n_points) and process: a campaign asks for
    it on every run, and the verification below costs seconds. The caller gets
    a copy.
    """
    return _pareto_front(name, M, int(n_points)).copy()


@lru_cache(maxsize=None)
def _pareto_front(name: str, M: int, n_points: int) -> np.ndarray:
    """The front itself; see pareto_front.

    Theorem 1 puts the front over the Pareto set of the alpha-only problem, not
    over every y_I, so the image is filtered. The position vector is sampled on
    a grid in the degenerate (m = 1) problems and half grid, half uniform noise
    otherwise, from a fixed seed, so a reference front is the same on every
    machine.

    The candidate count is CAPPED rather than grown from the grid resolution.
    A grid fine enough to be interesting in m = 9 dimensions has 3^9 = 19 683
    points, and the nondominated filter is quadratic in that, which turns one
    reference front into minutes. The cap costs resolution along each axis and
    buys a front that can be computed at ten objectives at all; the random part
    is what keeps the sample from being an axis-aligned lattice.

    Reaching the front's EXTREMES takes two separate measures, and ZCAT2 at
    M = 10 needed both: its F_1 = prod(1 - cos(y_i pi/2)) equals 1 at
    y = (1,..,1) and almost nothing anywhere else, and the sampled front
    reached 0.007 of that on f_1 until each was in place.
      * the corners of [0,1]^m are sampled explicitly, because the chance of a
        uniform sample landing near a given corner of a nine-dimensional cube
        is nil, so the extreme point is not a candidate at all without them;
      * the thinning below keeps the per-objective extremes, because a corner
        that is one row in a front of thousands is dropped with near certainty
        by a uniform choice of n_points, however it got there.
    An IGD reference front that stops short of the real one measures the
    sampler rather than the algorithm, and flatters whatever fails to spread.

    And nondominated within the sample is not yet Pareto-optimal: every point
    the thinning keeps is checked against an independent sample and its own
    neighbours, and one that fails is replaced by a checked point of the rest
    (_dominated_elsewhere, which also gives the measurements).
    """
    m = 1 if name in DEGENERATE else M - 1
    rng = np.random.default_rng(20260916 + 100 * M)
    cand = max(4000, 8 * n_points)
    if m == 1:
        Y = np.linspace(0.0, 1.0, cand).reshape(-1, 1)
        if M > 2:                                     # keep the other positions fixed
            Y = np.hstack([Y, np.full((len(Y), M - 2), 0.0)])
    else:
        corners = (((np.arange(1 << m)[:, None] >> np.arange(m)) & 1)
                   .astype(float)) if m <= 14 else np.zeros((0, m))
        if len(corners) > cand // 4:
            corners = corners[rng.choice(len(corners), cand // 4, replace=False)]
        room = cand - len(corners)
        p = max(2, int(round((room // 2) ** (1.0 / m))))
        grid = np.linspace(0.0, 1.0, p)
        mesh = np.array(np.meshgrid(*([grid] * m), indexing="ij")).reshape(m, -1).T
        if len(mesh) > room // 2:
            mesh = mesh[rng.choice(len(mesh), room // 2, replace=False)]
        Y = np.vstack([corners, mesh, rng.random((room - len(mesh), m))])
        # At m = 9 the grid resolution falls to p = 2, which makes the mesh the
        # corner set again. Equal rows do not dominate each other, so without
        # this the duplicates would both survive the filter and spend the
        # caller's n_points on the same point twice.
        Y = np.unique(Y, axis=0)
    scale = np.arange(1, M + 1, dtype=float) ** 2
    F = np.array([scale * F_FUNCS[name](y, M) for y in Y])
    keep = _nondominated_mask(F)
    Y, F = Y[keep], F[keep]
    sel = np.arange(len(F))
    if len(F) > n_points:
        keep = np.unique(np.concatenate([F.argmin(0), F.argmax(0)]))
        if len(keep) >= n_points:
            sel = keep[:n_points]
        else:
            rest = np.setdiff1d(np.arange(len(F)), keep)
            sel = np.concatenate([keep, rng.choice(rest, n_points - len(keep),
                                                   replace=False)])
    # The selected points are checked, and a dropped one is replaced from the
    # rest, which is checked in turn: a front in which nothing is dropped comes
    # out exactly as the unchecked sampler made it.
    bad = _dominated_elsewhere(name, M, Y[sel], F[sel])
    if bad.any():
        pool = np.random.default_rng(20260923 + 100 * M).permutation(
            np.setdiff1d(np.arange(len(F)), sel))
        good = [sel[~bad]]
        need = int(bad.sum())
        for start in range(0, len(pool), 256):
            if need <= 0:
                break
            chunk = pool[start:start + 256]
            ok = chunk[~_dominated_elsewhere(name, M, Y[chunk], F[chunk])]
            good.append(ok[:need])
            need -= len(ok[:need])
        sel = np.concatenate(good)
    F = F[sel]
    F.setflags(write=False)                           # shared through the cache
    return F


NAMES = tuple(f"ZCAT{i}" for i in range(1, 21))


def spec(name: str, M: int) -> dict:
    """Everything the registry needs for one ZCAT problem at M objectives."""
    return {
        "n_vars": n_vars(M),
        "bounds": bounds(n_vars(M)),
        "evaluate": (lambda x, _n=name, _M=M: evaluate(_n, x, _M)),
        "pareto_front": (lambda k, _n=name, _M=M: pareto_front(_n, _M, k)),
        "ideal": ideal(M),
        "nadir": nadir(M),
    }
