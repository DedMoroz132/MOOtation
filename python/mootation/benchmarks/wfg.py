# SPDX-License-Identifier: Apache-2.0
# ============================================================================
# WFG1-9.
# S. Huband, P. Hingston, L. Barone, L. While — "A Review of Multiobjective
# Test Problems and a Scalable Test Problem Toolkit", IEEE Transactions on
# Evolutionary Computation 10(5), 2006, pp. 477-506.
#
# The NumPy reference of the C++ suite's WFG (include/mootation/suite/wfg.hpp;
# task 17, supplement 1, stage 1), which evaluates WFG1-9, MaF10-12 and rWFG
# with the extension built: every wfgN calls the scalar kernels below, which
# wfg.hpp repeats operation for operation. Until stage 1 numba compiled them
# where it was installed; it is gone (the interpreted kernels were already the
# same bits), and the project's environment never had it.
#
# FIX 2026-10-09: until then a plain-NumPy copy of each wfgN ran without
# numba, and its shape functions took one factor too few in the middle
# objectives — h_m = prod(s[:M-2-m]) * c[M-1-m] (concave; likewise convex and
# linear) where Huband et al.'s concave_m multiplies M - m sines (Table X).
# Without numba f_2 .. f_{M-1} of every WFG problem were wrong from three
# objectives on (f_1 and f_M were right, and nothing at two objectives).
# Every WFG value computed without numba before that date is wrong there.
#
# API: f = wfgN(z, M) -> list[float].
# ============================================================================
from __future__ import annotations

import math
from typing import List

import numpy as np


# =============================================================
#  Primitives — all scalar loops, work on 1-D arrays
# =============================================================

def _nb_normalize(z):
    n = len(z)
    out = np.empty(n)
    for i in range(n):
        out[i] = z[i] / (2.0 * (i + 1))
    return out

def _nb_s_linear(y, A):
    out = np.empty(len(y))
    for i in range(len(y)):
        out[i] = abs(y[i] - A) / abs(math.floor(A - y[i]) + A)
    return out

def _nb_s_decept(y, A, B, C):
    out = np.empty(len(y))
    for i in range(len(y)):
        yi = y[i]
        inner = abs(yi - A) - B
        t1 = math.floor(yi - A + B) * (1.0 - C + (A - B) / B) / (A - B)
        t2 = math.floor(A + B - yi) * (1.0 - C + (1.0 - A - B) / B) / (1.0 - A - B)
        out[i] = 1.0 + inner * (t1 + t2 + 1.0 / B)
    return out

def _nb_s_multi(y, A, B, C):
    out = np.empty(len(y))
    for i in range(len(y)):
        yi = y[i]
        denom = 2.0 * (math.floor(C - yi) + C)
        u = abs(yi - C) / denom
        num = 1.0 + math.cos((4.0*A + 2.0)*math.pi*(0.5 - u)) + 4.0*B*u*u
        out[i] = num / (B + 2.0)
    return out

def _nb_b_poly(y, alpha):
    out = np.empty(len(y))
    for i in range(len(y)):
        out[i] = y[i] ** alpha
    return out

def _nb_b_flat(y, A, B, C):
    out = np.empty(len(y))
    for i in range(len(y)):
        yi = y[i]
        t1 = min(0.0, math.floor(yi - B)) * (A * (B - yi) / B)
        t2 = min(0.0, math.floor(C - yi)) * ((1.0 - A) * (yi - C) / (1.0 - C))
        v = A + t1 - t2
        out[i] = max(0.0, min(1.0, v))
    return out

def _nb_b_param(y, y_prime, A, B, C):
    out = np.empty(len(y))
    for i in range(len(y)):
        v = A - (1.0 - 2.0*y_prime[i]) * abs(math.floor(0.5 - y_prime[i]) + A)
        out[i] = y[i] ** (B + (C - B) * v)
    return out

def _nb_r_nonsep(y, A):
    n = len(y)
    ceil_A_half = math.ceil(A / 2.0)
    total = 0.0
    for j in range(n):
        s = y[j]
        for kk in range(A - 1):
            s += abs(y[j] - y[(j + kk + 1) % n])
        total += s
    denom = (n / A) * ceil_A_half * (1.0 + 2.0*A - 2.0*ceil_A_half)
    return total / denom

def _nb_r_sum_chunks(t, k, M, w):
    """Weighted r_sum; w=None → uniform (pass zeros to signal uniform)."""
    chunk = k // (M - 1)
    out = np.empty(M)
    uniform = (w[0] < 0.0)  # sentinel: negative w[0] → uniform
    for m in range(M - 1):
        lo = m * chunk
        hi = lo + chunk
        if uniform:
            s = 0.0
            for i in range(lo, hi): s += t[i]
            out[m] = s / (hi - lo)
        else:
            sw = 0.0; s = 0.0
            for i in range(lo, hi): s += w[i]*t[i]; sw += w[i]
            out[m] = s / sw
    l = len(t) - k
    if uniform:
        s = 0.0
        for i in range(k, len(t)): s += t[i]
        out[M-1] = s / l
    else:
        sw = 0.0; s = 0.0
        for i in range(k, len(t)): s += w[i]*t[i]; sw += w[i]
        out[M-1] = s / sw
    return out

def _nb_r_nonsep_chunks(t, A, k, M, l):
    chunk = k // (M - 1)
    out = np.empty(M)
    for m in range(M - 1):
        lo = m * chunk
        hi = lo + chunk
        out[m] = _nb_r_nonsep(t[lo:hi], A)
    out[M-1] = _nb_r_nonsep(t[k:], l)
    return out

def _nb_shape_linear(x):
    # h[0] = prod(x[0..M-2])
    # h[m] = prod(x[0..M-m-2]) * (1-x[M-m-1])  for m=1..M-2
    # h[M-1] = 1 - x[0]
    M = len(x) + 1
    h = np.empty(M)
    cp = 1.0
    for i in range(len(x)): cp *= x[i]
    h[0] = cp
    for m in range(1, M-1):
        acc = 1.0
        for i in range(M-m-1): acc *= x[i]
        h[m] = acc * (1.0 - x[M-m-1])
    h[M-1] = 1.0 - x[0]
    return h

def _nb_shape_convex(x):
    # h[0] = prod(1-c[i])
    # h[m] = prod(1-c[0..M-m-2]) * (1-s[M-m-1])  for m=1..M-2
    # h[M-1] = 1 - s[0]
    M = len(x) + 1
    s = np.empty(len(x)); c = np.empty(len(x))
    for i in range(len(x)):
        s[i] = math.sin(math.pi*x[i]/2.0)
        c[i] = math.cos(math.pi*x[i]/2.0)
    omc = np.empty(len(x))
    for i in range(len(x)): omc[i] = 1.0 - c[i]
    h = np.empty(M)
    cp = 1.0
    for i in range(len(x)): cp *= omc[i]
    h[0] = cp
    for m in range(1, M-1):
        acc = 1.0
        for i in range(M-m-1): acc *= omc[i]
        h[m] = acc * (1.0 - s[M-m-1])
    h[M-1] = 1.0 - s[0]
    return h

def _nb_shape_concave(x):
    # h[0] = prod(s[i])
    # h[m] = prod(s[0..M-m-2]) * c[M-m-1]  for m=1..M-2
    # h[M-1] = c[0]
    M = len(x) + 1
    s = np.empty(len(x)); c = np.empty(len(x))
    for i in range(len(x)):
        s[i] = math.sin(math.pi*x[i]/2.0)
        c[i] = math.cos(math.pi*x[i]/2.0)
    h = np.empty(M)
    cp = 1.0
    for i in range(len(x)): cp *= s[i]
    h[0] = cp
    for m in range(1, M-1):
        acc = 1.0
        for i in range(M-m-1): acc *= s[i]
        h[m] = acc * c[M-m-1]
    h[M-1] = c[0]
    return h

def _nb_calc_x(t, A_vec):
    M = len(t)
    x = np.empty(M)
    tM = t[-1]
    for i in range(M-1):
        x[i] = max(tM, A_vec[i]) * (t[i] - 0.5) + 0.5
    x[-1] = tM
    return x

def _nb_final(x, h, M):
    out = np.empty(M)
    for m in range(M):
        out[m] = x[-1] + 2.0*(m+1)*h[m]
    return out

# Uniform weight sentinel
_UNIF = np.array([-1.0])  # signals uniform in _nb_r_sum_chunks

# =============================================================
#  WFG1-9 (Huband et al. 2006, Table XIV), on the kernels above
# =============================================================
def _make_w(n):
    return 2.0 * np.arange(1, n+1, dtype=float)

def wfg1(z: List[float], M: int, k: int | None = None) -> List[float]:
    z = np.asarray(z, dtype=float)
    n = len(z)
    if k is None: k = 2*(M-1)
    y  = _nb_normalize(z)
    t1 = y.copy(); t1[k:] = _nb_s_linear(y[k:], 0.35)
    t2 = t1.copy(); t2[k:] = _nb_b_flat(t1[k:], 0.8, 0.75, 0.85)
    t3 = _nb_b_poly(t2, 0.02)
    w  = _make_w(n)
    t4 = _nb_r_sum_chunks(t3, k, M, w)
    x  = _nb_calc_x(t4, np.ones(M-1))
    h  = _nb_shape_convex(x[:-1])
    x1 = x[0]
    h[-1] = (1.0 - x1 - math.cos(10.0*math.pi*x1 + math.pi/2.0)/(10.0*math.pi))**1.0
    return _nb_final(x, h, M).tolist()

def wfg2(z: List[float], M: int, k: int | None = None) -> List[float]:
    z = np.asarray(z, dtype=float)
    n = len(z)
    if k is None: k = 2*(M-1)
    l = n-k
    y  = _nb_normalize(z)
    t1 = y.copy(); t1[k:] = _nb_s_linear(y[k:], 0.35)
    t2 = np.empty(k + l//2); t2[:k] = t1[:k]
    for i in range(l//2): t2[k+i] = _nb_r_nonsep(t1[k+2*i:k+2*(i+1)], 2)
    t3 = _nb_r_sum_chunks(t2, k, M, _UNIF)
    x  = _nb_calc_x(t3, np.ones(M-1))
    h  = _nb_shape_convex(x[:-1])
    x1 = x[0]; h[-1] = 1.0 - x1*(math.cos(5*math.pi*x1)**2)
    return _nb_final(x, h, M).tolist()

def wfg3(z: List[float], M: int) -> List[float]:
    z = np.asarray(z, dtype=float)
    n, k = len(z), 2*(M-1); l = n-k
    y  = _nb_normalize(z)
    t1 = y.copy(); t1[k:] = _nb_s_linear(y[k:], 0.35)
    t2 = np.empty(k + l//2); t2[:k] = t1[:k]
    for i in range(l//2): t2[k+i] = _nb_r_nonsep(t1[k+2*i:k+2*(i+1)], 2)
    t3 = _nb_r_sum_chunks(t2, k, M, _UNIF)
    A_vec = np.zeros(M-1); A_vec[0] = 1.0
    x  = _nb_calc_x(t3, A_vec)
    h  = _nb_shape_linear(x[:-1])
    return _nb_final(x, h, M).tolist()

def wfg4(z: List[float], M: int) -> List[float]:
    z = np.asarray(z, dtype=float)
    k = 2*(M-1)
    y  = _nb_normalize(z)
    t1 = _nb_s_multi(y, 30, 10.0, 0.35)
    t2 = _nb_r_sum_chunks(t1, k, M, _UNIF)
    x  = _nb_calc_x(t2, np.ones(M-1))
    h  = _nb_shape_concave(x[:-1])
    return _nb_final(x, h, M).tolist()

def wfg5(z: List[float], M: int) -> List[float]:
    z = np.asarray(z, dtype=float)
    k = 2*(M-1)
    y  = _nb_normalize(z)
    t1 = _nb_s_decept(y, 0.35, 0.001, 0.05)
    t2 = _nb_r_sum_chunks(t1, k, M, _UNIF)
    x  = _nb_calc_x(t2, np.ones(M-1))
    h  = _nb_shape_concave(x[:-1])
    return _nb_final(x, h, M).tolist()

def wfg6(z: List[float], M: int) -> List[float]:
    z = np.asarray(z, dtype=float)
    n, k = len(z), 2*(M-1); l = n-k
    y  = _nb_normalize(z)
    t1 = y.copy(); t1[k:] = _nb_s_linear(y[k:], 0.35)
    t2 = _nb_r_nonsep_chunks(t1, k//(M-1), k, M, l)
    x  = _nb_calc_x(t2, np.ones(M-1))
    h  = _nb_shape_concave(x[:-1])
    return _nb_final(x, h, M).tolist()

def wfg7(z: List[float], M: int) -> List[float]:
    z = np.asarray(z, dtype=float)
    n, k = len(z), 2*(M-1)
    y  = _nb_normalize(z)
    cs = np.cumsum(y[::-1])[::-1]
    tail_mean = cs[1:k+1] / np.arange(n-1, n-k-1, -1, dtype=float)   # mean of y_{i+1..n}
    t1 = y.copy()
    t1[:k] = _nb_b_param(y[:k], tail_mean, 0.98/49.98, 0.02, 50.0)
    t1[k:] = _nb_s_linear(t1[k:], 0.35)
    t2 = _nb_r_sum_chunks(t1, k, M, _UNIF)
    x  = _nb_calc_x(t2, np.ones(M-1))
    h  = _nb_shape_concave(x[:-1])
    return _nb_final(x, h, M).tolist()

def wfg8(z: List[float], M: int) -> List[float]:
    z = np.asarray(z, dtype=float)
    n, k = len(z), 2*(M-1)
    y  = _nb_normalize(z)
    cs = np.cumsum(y)
    head_mean = cs[k-1:n-1] / np.arange(k, n, dtype=float)            # mean of y_{1..i-1}
    t1 = y.copy()
    t1[k:] = _nb_b_param(y[k:], head_mean, 0.98/49.98, 0.02, 50.0)
    t1[k:] = _nb_s_linear(t1[k:], 0.35)
    t2 = _nb_r_sum_chunks(t1, k, M, _UNIF)
    x  = _nb_calc_x(t2, np.ones(M-1))
    h  = _nb_shape_concave(x[:-1])
    return _nb_final(x, h, M).tolist()

def wfg9(z: List[float], M: int, k: int | None = None) -> List[float]:
    z = np.asarray(z, dtype=float)
    n = len(z)
    if k is None: k = 2*(M-1)
    l = n-k
    y  = _nb_normalize(z)
    cs = np.cumsum(y[::-1])[::-1]
    tail_mean = cs[1:] / np.arange(n-1, 0, -1, dtype=float)           # mean of y_{i+1..n}
    t1 = y.copy()
    t1[:n-1] = _nb_b_param(y[:n-1], tail_mean, 0.98/49.98, 0.02, 50.0)
    t2 = np.empty(n)
    t2[:k] = _nb_s_decept(t1[:k], 0.35, 0.001, 0.05)
    t2[k:] = _nb_s_multi(t1[k:], 30, 95.0, 0.35)
    t3 = _nb_r_nonsep_chunks(t2, k//(M-1), k, M, l)
    x  = _nb_calc_x(t3, np.ones(M-1))
    h  = _nb_shape_concave(x[:-1])
    return _nb_final(x, h, M).tolist()


# =============================================================
#  Registry + bounds helper
# =============================================================
WFG_FUNCS = {
    "WFG1": wfg1, "WFG2": wfg2, "WFG3": wfg3,
    "WFG4": wfg4, "WFG5": wfg5, "WFG6": wfg6,
    "WFG7": wfg7, "WFG8": wfg8, "WFG9": wfg9,
}

def wfg_bounds(M: int, k: int | None = None, l: int = 20):
    if k is None: k = 2*(M-1)
    return [(0.0, 2.0*(i+1)) for i in range(k+l)]

def wfg_nadir(M: int):
    return tuple(2.0*m for m in range(1, M+1))
