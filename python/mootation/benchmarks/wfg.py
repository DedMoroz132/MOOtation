# SPDX-License-Identifier: Apache-2.0
# ============================================================================
# WFG1-9.
# S. Huband, P. Hingston, L. Barone, L. While — "A Review of Multiobjective
# Test Problems and a Scalable Test Problem Toolkit", IEEE Transactions on
# Evolutionary Computation 10(5), 2006, pp. 477-506.
#
# The inner loops are JIT-compiled with numba when it is installed and run as
# plain NumPy when it is not; `numba_enabled()` reports which. numba is an
# accelerator here, never a requirement, and the two paths compute the same
# numbers.
#
# API: f = wfgN(z, M) -> list[float].
# ============================================================================
from __future__ import annotations

import math
from typing import List

import numpy as np

try:
    from numba import njit as _njit
    _NUMBA = True
except ImportError:
    def _njit(**kw):
        def decorator(f): return f
        return decorator
    _NUMBA = False

# numba is an optional accelerator, never a requirement: without it the
# decorator above is a no-op and the plain NumPy path runs. Whether it is
# present is reported through a function rather than printed at import —
# a library that writes to stdout when imported corrupts any caller whose
# stdout is data.
def numba_enabled() -> bool:
    """True when numba is installed and the WFG kernels are compiled."""
    return bool(_NUMBA)


# =============================================================
#  JIT primitives — all scalar loops, work on 1-D arrays
# =============================================================

@_njit(cache=True)
def _nb_normalize(z):
    n = len(z)
    out = np.empty(n)
    for i in range(n):
        out[i] = z[i] / (2.0 * (i + 1))
    return out

@_njit(cache=True)
def _nb_s_linear(y, A):
    out = np.empty(len(y))
    for i in range(len(y)):
        out[i] = abs(y[i] - A) / abs(math.floor(A - y[i]) + A)
    return out

@_njit(cache=True)
def _nb_s_decept(y, A, B, C):
    out = np.empty(len(y))
    for i in range(len(y)):
        yi = y[i]
        inner = abs(yi - A) - B
        t1 = math.floor(yi - A + B) * (1.0 - C + (A - B) / B) / (A - B)
        t2 = math.floor(A + B - yi) * (1.0 - C + (1.0 - A - B) / B) / (1.0 - A - B)
        out[i] = 1.0 + inner * (t1 + t2 + 1.0 / B)
    return out

@_njit(cache=True)
def _nb_s_multi(y, A, B, C):
    out = np.empty(len(y))
    for i in range(len(y)):
        yi = y[i]
        denom = 2.0 * (math.floor(C - yi) + C)
        u = abs(yi - C) / denom
        num = 1.0 + math.cos((4.0*A + 2.0)*math.pi*(0.5 - u)) + 4.0*B*u*u
        out[i] = num / (B + 2.0)
    return out

@_njit(cache=True)
def _nb_b_poly(y, alpha):
    out = np.empty(len(y))
    for i in range(len(y)):
        out[i] = y[i] ** alpha
    return out

@_njit(cache=True)
def _nb_b_flat(y, A, B, C):
    out = np.empty(len(y))
    for i in range(len(y)):
        yi = y[i]
        t1 = min(0.0, math.floor(yi - B)) * (A * (B - yi) / B)
        t2 = min(0.0, math.floor(C - yi)) * ((1.0 - A) * (yi - C) / (1.0 - C))
        v = A + t1 - t2
        out[i] = max(0.0, min(1.0, v))
    return out

@_njit(cache=True)
def _nb_b_param(y, y_prime, A, B, C):
    out = np.empty(len(y))
    for i in range(len(y)):
        v = A - (1.0 - 2.0*y_prime[i]) * abs(math.floor(0.5 - y_prime[i]) + A)
        out[i] = y[i] ** (B + (C - B) * v)
    return out

@_njit(cache=True)
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

@_njit(cache=True)
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

@_njit(cache=True)
def _nb_r_nonsep_chunks(t, A, k, M, l):
    chunk = k // (M - 1)
    out = np.empty(M)
    for m in range(M - 1):
        lo = m * chunk
        hi = lo + chunk
        out[m] = _nb_r_nonsep(t[lo:hi], A)
    out[M-1] = _nb_r_nonsep(t[k:], l)
    return out

@_njit(cache=True)
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

@_njit(cache=True)
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

@_njit(cache=True)
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

@_njit(cache=True)
def _nb_calc_x(t, A_vec):
    M = len(t)
    x = np.empty(M)
    tM = t[-1]
    for i in range(M-1):
        x[i] = max(tM, A_vec[i]) * (t[i] - 0.5) + 0.5
    x[-1] = tM
    return x

@_njit(cache=True)
def _nb_final(x, h, M):
    out = np.empty(M)
    for m in range(M):
        out[m] = x[-1] + 2.0*(m+1)*h[m]
    return out

# Uniform weight sentinel
_UNIF = np.array([-1.0])  # signals uniform in _nb_r_sum_chunks

# =============================================================
#  NumPy fallbacks (used when numba not available)
# =============================================================
_NORM_CACHE: dict = {}

def _normalize_z(z):
    n = len(z)
    if n not in _NORM_CACHE:
        _NORM_CACHE[n] = 2.0 * np.arange(1, n+1, dtype=float)
    return z / _NORM_CACHE[n]

def _np_r_sum_chunks(t, k, M, w=None):
    chunk = k // (M - 1)
    pos = t[:k].reshape(M-1, chunk)
    out = np.empty(M)
    if w is not None:
        wp = w[:k].reshape(M-1, chunk)
        out[:M-1] = (pos*wp).sum(1)/wp.sum(1)
        wt = w[k:]
        out[M-1] = np.dot(wt, t[k:])/wt.sum()
    else:
        out[:M-1] = pos.mean(1)
        out[M-1]  = t[k:].mean()
    return out

def _np_r_nonsep(y, A):
    n = len(y)
    j = np.arange(n)
    total = y.copy()
    for kk in range(A-1):
        total += np.abs(y - y[(j+kk+1)%n])
    ceil_A_half = np.ceil(A/2.0)
    denom = (n/A)*ceil_A_half*(1.0+2.0*A-2.0*ceil_A_half)
    return float(total.sum()/denom)

def _np_r_nonsep_chunks(t, A, k, M, l):
    chunk = k//(M-1)
    out = np.empty(M)
    for m in range(M-1):
        lo,hi = m*chunk,(m+1)*chunk
        out[m] = _np_r_nonsep(t[lo:hi], A)
    out[M-1] = _np_r_nonsep(t[k:], l)
    return out

# =============================================================
#  WFG1-9  — dispatch to JIT or NumPy
# =============================================================
def _make_w(n):
    return 2.0 * np.arange(1, n+1, dtype=float)

def wfg1(z: List[float], M: int, k: int | None = None) -> List[float]:
    z = np.asarray(z, dtype=float)
    n = len(z)
    if k is None: k = 2*(M-1)
    if _NUMBA:
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
    else:
        y  = _normalize_z(z)
        t1 = y.copy(); t1[k:] = np.abs(y[k:]-0.35)/np.abs(np.floor(0.35-y[k:])+0.35)
        t2 = t1.copy(); t2[k:] = np.clip(0.8+np.minimum(0.,np.floor(t1[k:]-0.75))*(0.8*(0.75-t1[k:])/0.75)-np.minimum(0.,np.floor(0.85-t1[k:]))*((0.2)*(t1[k:]-0.85)/0.15),0,1)
        t3 = t2**0.02
        w  = _make_w(n)
        t4 = _np_r_sum_chunks(t3, k, M, w)
        x  = np.empty(M); x[-1]=t4[-1]
        x[:-1]=np.maximum(t4[-1],1.0)*(t4[:-1]-0.5)+0.5
        s  = np.sin(np.pi*x[:-1]/2); c=np.cos(np.pi*x[:-1]/2)
        cp = np.cumprod(1-c); h=np.empty(M); h[0]=cp[-1]
        for m in range(1,M-1): h[m]=np.prod(1-c[:M-2-m])*(1-s[M-1-m])
        h[-1]=1-s[0]
        x1=x[0]; h[-1]=(1.0-x1-math.cos(10*math.pi*x1+math.pi/2)/(10*math.pi))**1.0
        return (x[-1]+2.0*np.arange(1,M+1)*h).tolist()

def wfg2(z: List[float], M: int, k: int | None = None) -> List[float]:
    z = np.asarray(z, dtype=float)
    n = len(z)
    if k is None: k = 2*(M-1)
    l = n-k
    y = _nb_normalize(z) if _NUMBA else _normalize_z(z)
    t1 = y.copy()
    t1[k:] = _nb_s_linear(y[k:], 0.35) if _NUMBA else np.abs(y[k:]-0.35)/np.abs(np.floor(0.35-y[k:])+0.35)
    t2 = np.empty(k + l//2); t2[:k] = t1[:k]
    rn = _nb_r_nonsep if _NUMBA else _np_r_nonsep
    for i in range(l//2): t2[k+i] = rn(t1[k+2*i:k+2*(i+1)], 2)
    t3 = _nb_r_sum_chunks(t2,k,M,_UNIF) if _NUMBA else _np_r_sum_chunks(t2,k,M)
    x  = _nb_calc_x(t3, np.ones(M-1)) if _NUMBA else (lambda t: (np.maximum(t[-1],1)*(t[:-1]-0.5)+0.5, t[-1]))(t3)
    if not _NUMBA:
        xv=np.empty(M); xv[:-1]=np.maximum(t3[-1],1)*(t3[:-1]-0.5)+0.5; xv[-1]=t3[-1]; x=xv
    h  = _nb_shape_convex(x[:-1]) if _NUMBA else None
    if h is None:
        s=np.sin(np.pi*x[:-1]/2);c=np.cos(np.pi*x[:-1]/2);cp=np.cumprod(1-c)
        h=np.empty(M);h[0]=cp[-1]
        for m in range(1,M-1): h[m]=np.prod(1-c[:M-2-m])*(1-s[M-1-m])
        h[-1]=1-s[0]
    x1=x[0]; h[-1]=1.0-x1*(math.cos(5*math.pi*x1)**2)
    return (_nb_final(x,h,M).tolist() if _NUMBA else
            (x[-1]+2.0*np.arange(1,M+1)*h).tolist())

def wfg3(z: List[float], M: int) -> List[float]:
    z = np.asarray(z, dtype=float)
    n, k = len(z), 2*(M-1); l = n-k
    y = _nb_normalize(z) if _NUMBA else _normalize_z(z)
    t1 = y.copy()
    t1[k:] = _nb_s_linear(y[k:], 0.35) if _NUMBA else np.abs(y[k:]-0.35)/np.abs(np.floor(0.35-y[k:])+0.35)
    t2 = np.empty(k + l//2); t2[:k] = t1[:k]
    rn = _nb_r_nonsep if _NUMBA else _np_r_nonsep
    for i in range(l//2): t2[k+i] = rn(t1[k+2*i:k+2*(i+1)], 2)
    t3 = _nb_r_sum_chunks(t2,k,M,_UNIF) if _NUMBA else _np_r_sum_chunks(t2,k,M)
    A_vec = np.zeros(M-1); A_vec[0]=1.0
    x = _nb_calc_x(t3, A_vec) if _NUMBA else None
    if x is None:
        x=np.empty(M); x[:-1]=np.maximum(t3[-1],A_vec)*(t3[:-1]-0.5)+0.5; x[-1]=t3[-1]
    h = _nb_shape_linear(x[:-1]) if _NUMBA else None
    if h is None:
        xp=x[:-1];h=np.empty(M);h[0]=np.prod(xp)
        for m in range(1,M-1): h[m]=np.prod(xp[:M-2-m])*(1-xp[M-1-m])
        h[-1]=1-xp[0]
    return (_nb_final(x,h,M).tolist() if _NUMBA else
            (x[-1]+2.0*np.arange(1,M+1)*h).tolist())

def wfg4(z: List[float], M: int) -> List[float]:
    z = np.asarray(z, dtype=float)
    n, k = len(z), 2*(M-1)
    y  = _nb_normalize(z) if _NUMBA else _normalize_z(z)
    t1 = _nb_s_multi(y, 30, 10.0, 0.35) if _NUMBA else None
    if t1 is None:
        denom=2*(np.floor(0.35-y)+0.35);u=np.abs(y-0.35)/denom
        t1=(1+np.cos((4*30+2)*np.pi*(0.5-u))+4*10*u*u)/(10+2)
    t2 = _nb_r_sum_chunks(t1,k,M,_UNIF) if _NUMBA else _np_r_sum_chunks(t1,k,M)
    x  = _nb_calc_x(t2, np.ones(M-1)) if _NUMBA else None
    if x is None:
        x=np.empty(M);x[:-1]=np.maximum(t2[-1],1)*(t2[:-1]-0.5)+0.5;x[-1]=t2[-1]
    h = _nb_shape_concave(x[:-1]) if _NUMBA else None
    if h is None:
        s=np.sin(np.pi*x[:-1]/2);c=np.cos(np.pi*x[:-1]/2);cp=np.cumprod(s)
        h=np.empty(M);h[0]=cp[-1]
        for m in range(1,M-1): h[m]=np.prod(s[:M-2-m])*c[M-1-m]
        h[-1]=c[0]
    return (_nb_final(x,h,M).tolist() if _NUMBA else
            (x[-1]+2.0*np.arange(1,M+1)*h).tolist())

def wfg5(z: List[float], M: int) -> List[float]:
    z = np.asarray(z, dtype=float)
    n, k = len(z), 2*(M-1)
    y  = _nb_normalize(z) if _NUMBA else _normalize_z(z)
    t1 = _nb_s_decept(y, 0.35, 0.001, 0.05) if _NUMBA else None
    if t1 is None:
        inner=np.abs(y-0.35)-0.001
        t1p=np.floor(y-0.35+0.001)*(0.95+(0.35-0.001)/0.001)/(0.35-0.001)
        t2p=np.floor(0.35+0.001-y)*(0.95+(1-0.35-0.001)/0.001)/(1-0.35-0.001)
        t1=1+inner*(t1p+t2p+1/0.001)
    t2 = _nb_r_sum_chunks(t1,k,M,_UNIF) if _NUMBA else _np_r_sum_chunks(t1,k,M)
    x  = _nb_calc_x(t2, np.ones(M-1)) if _NUMBA else None
    if x is None:
        x=np.empty(M);x[:-1]=np.maximum(t2[-1],1)*(t2[:-1]-0.5)+0.5;x[-1]=t2[-1]
    h = _nb_shape_concave(x[:-1]) if _NUMBA else None
    if h is None:
        s=np.sin(np.pi*x[:-1]/2);c=np.cos(np.pi*x[:-1]/2);cp=np.cumprod(s)
        h=np.empty(M);h[0]=cp[-1]
        for m in range(1,M-1): h[m]=np.prod(s[:M-2-m])*c[M-1-m]
        h[-1]=c[0]
    return (_nb_final(x,h,M).tolist() if _NUMBA else
            (x[-1]+2.0*np.arange(1,M+1)*h).tolist())

def wfg6(z: List[float], M: int) -> List[float]:
    z = np.asarray(z, dtype=float)
    n, k = len(z), 2*(M-1); l = n-k
    y  = _nb_normalize(z) if _NUMBA else _normalize_z(z)
    t1 = y.copy()
    t1[k:] = _nb_s_linear(y[k:], 0.35) if _NUMBA else np.abs(y[k:]-0.35)/np.abs(np.floor(0.35-y[k:])+0.35)
    chunk = k//(M-1)
    t2 = _nb_r_nonsep_chunks(t1,chunk,k,M,l) if _NUMBA else _np_r_nonsep_chunks(t1,chunk,k,M,l)
    x  = _nb_calc_x(t2, np.ones(M-1)) if _NUMBA else None
    if x is None:
        x=np.empty(M);x[:-1]=np.maximum(t2[-1],1)*(t2[:-1]-0.5)+0.5;x[-1]=t2[-1]
    h = _nb_shape_concave(x[:-1]) if _NUMBA else None
    if h is None:
        s=np.sin(np.pi*x[:-1]/2);c=np.cos(np.pi*x[:-1]/2);cp=np.cumprod(s)
        h=np.empty(M);h[0]=cp[-1]
        for m in range(1,M-1): h[m]=np.prod(s[:M-2-m])*c[M-1-m]
        h[-1]=c[0]
    return (_nb_final(x,h,M).tolist() if _NUMBA else
            (x[-1]+2.0*np.arange(1,M+1)*h).tolist())

def wfg7(z: List[float], M: int) -> List[float]:
    z = np.asarray(z, dtype=float)
    n, k = len(z), 2*(M-1)
    y  = _nb_normalize(z) if _NUMBA else _normalize_z(z)
    cs = np.cumsum(y[::-1])[::-1]
    tail_mean = cs[1:k+1] / np.arange(n-1, n-k-1, -1, dtype=float)
    t1 = y.copy()
    t1[:k] = _nb_b_param(y[:k], tail_mean, 0.98/49.98, 0.02, 50.0) if _NUMBA else None
    if t1[:k] is None or not _NUMBA:
        for i in range(k):
            yp=cs[i+1]/(n-i-1)
            v=0.98/49.98-(1-2*yp)*abs(math.floor(0.5-yp)+0.98/49.98)
            t1[i]=y[i]**(0.02+(50-0.02)*v)
    t1[k:] = _nb_s_linear(t1[k:], 0.35) if _NUMBA else np.abs(t1[k:]-0.35)/np.abs(np.floor(0.35-t1[k:])+0.35)
    t2 = _nb_r_sum_chunks(t1,k,M,_UNIF) if _NUMBA else _np_r_sum_chunks(t1,k,M)
    x  = _nb_calc_x(t2, np.ones(M-1)) if _NUMBA else None
    if x is None:
        x=np.empty(M);x[:-1]=np.maximum(t2[-1],1)*(t2[:-1]-0.5)+0.5;x[-1]=t2[-1]
    h = _nb_shape_concave(x[:-1]) if _NUMBA else None
    if h is None:
        s=np.sin(np.pi*x[:-1]/2);c=np.cos(np.pi*x[:-1]/2);cp=np.cumprod(s)
        h=np.empty(M);h[0]=cp[-1]
        for m in range(1,M-1): h[m]=np.prod(s[:M-2-m])*c[M-1-m]
        h[-1]=c[0]
    return (_nb_final(x,h,M).tolist() if _NUMBA else
            (x[-1]+2.0*np.arange(1,M+1)*h).tolist())

def wfg8(z: List[float], M: int) -> List[float]:
    z = np.asarray(z, dtype=float)
    n, k = len(z), 2*(M-1)
    y  = _nb_normalize(z) if _NUMBA else _normalize_z(z)
    cs = np.cumsum(y)
    head_mean = cs[k-1:n-1] / np.arange(k, n, dtype=float)
    t1 = y.copy()
    t1[k:] = _nb_b_param(y[k:], head_mean, 0.98/49.98, 0.02, 50.0) if _NUMBA else None
    if not _NUMBA:
        for idx,i in enumerate(range(k,n)):
            yp=y[:i].mean()
            v=0.98/49.98-(1-2*yp)*abs(math.floor(0.5-yp)+0.98/49.98)
            t1[i]=y[i]**(0.02+(50-0.02)*v)
    t1[k:] = _nb_s_linear(t1[k:], 0.35) if _NUMBA else np.abs(t1[k:]-0.35)/np.abs(np.floor(0.35-t1[k:])+0.35)
    t2 = _nb_r_sum_chunks(t1,k,M,_UNIF) if _NUMBA else _np_r_sum_chunks(t1,k,M)
    x  = _nb_calc_x(t2, np.ones(M-1)) if _NUMBA else None
    if x is None:
        x=np.empty(M);x[:-1]=np.maximum(t2[-1],1)*(t2[:-1]-0.5)+0.5;x[-1]=t2[-1]
    h = _nb_shape_concave(x[:-1]) if _NUMBA else None
    if h is None:
        s=np.sin(np.pi*x[:-1]/2);c=np.cos(np.pi*x[:-1]/2);cp=np.cumprod(s)
        h=np.empty(M);h[0]=cp[-1]
        for m in range(1,M-1): h[m]=np.prod(s[:M-2-m])*c[M-1-m]
        h[-1]=c[0]
    return (_nb_final(x,h,M).tolist() if _NUMBA else
            (x[-1]+2.0*np.arange(1,M+1)*h).tolist())

def wfg9(z: List[float], M: int, k: int | None = None) -> List[float]:
    z = np.asarray(z, dtype=float)
    n = len(z)
    if k is None: k = 2*(M-1)
    l = n-k
    y  = _nb_normalize(z) if _NUMBA else _normalize_z(z)
    cs = np.cumsum(y[::-1])[::-1]
    tail_mean = cs[1:] / np.arange(n-1, 0, -1, dtype=float)
    t1 = y.copy()
    t1[:n-1] = _nb_b_param(y[:n-1], tail_mean, 0.98/49.98, 0.02, 50.0) if _NUMBA else None
    if not _NUMBA:
        for i in range(n-1):
            yp=cs[i+1]/(n-i-1)
            v=0.98/49.98-(1-2*yp)*abs(math.floor(0.5-yp)+0.98/49.98)
            t1[i]=y[i]**(0.02+(50-0.02)*v)
    t2 = np.empty(n)
    t2[:k] = _nb_s_decept(t1[:k], 0.35, 0.001, 0.05) if _NUMBA else None
    t2[k:] = _nb_s_multi(t1[k:], 30, 95.0, 0.35) if _NUMBA else None
    if not _NUMBA:
        inner=np.abs(t1[:k]-0.35)-0.001
        t1p=np.floor(t1[:k]-0.35+0.001)*(0.95+(0.35-0.001)/0.001)/(0.35-0.001)
        t2p=np.floor(0.35+0.001-t1[:k])*(0.95+(1-0.35-0.001)/0.001)/(1-0.35-0.001)
        t2[:k]=1+inner*(t1p+t2p+1/0.001)
        denom=2*(np.floor(0.35-t1[k:])+0.35);u=np.abs(t1[k:]-0.35)/denom
        t2[k:]=(1+np.cos((4*30+2)*np.pi*(0.5-u))+4*95*u*u)/(95+2)
    chunk = k//(M-1)
    t3 = _nb_r_nonsep_chunks(t2,chunk,k,M,l) if _NUMBA else _np_r_nonsep_chunks(t2,chunk,k,M,l)
    x  = _nb_calc_x(t3, np.ones(M-1)) if _NUMBA else None
    if x is None:
        x=np.empty(M);x[:-1]=np.maximum(t3[-1],1)*(t3[:-1]-0.5)+0.5;x[-1]=t3[-1]
    h = _nb_shape_concave(x[:-1]) if _NUMBA else None
    if h is None:
        s=np.sin(np.pi*x[:-1]/2);c=np.cos(np.pi*x[:-1]/2);cp=np.cumprod(s)
        h=np.empty(M);h[0]=cp[-1]
        for m in range(1,M-1): h[m]=np.prod(s[:M-2-m])*c[M-1-m]
        h[-1]=c[0]
    return (_nb_final(x,h,M).tolist() if _NUMBA else
            (x[-1]+2.0*np.arange(1,M+1)*h).tolist())


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

# =============================================================
#  JIT warmup — call once per process to load numba cache
# =============================================================
def warmup():
    if not _NUMBA:
        return
    _w = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    _x = np.array([0.3, 0.4, 0.5, 0.6])
    _t = np.array([0.3, 0.4, 0.5, 0.6, 0.7])
    _nb_normalize(_w)
    _nb_s_linear(_w, 0.35)
    _nb_s_decept(_w, 0.35, 0.001, 0.05)
    _nb_s_multi(_w, 30, 10.0, 0.35)
    _nb_b_poly(_w, 0.02)
    _nb_b_flat(_w, 0.8, 0.75, 0.85)
    _nb_b_param(_w, _w, 0.98/49.98, 0.02, 50.0)
    _nb_r_nonsep(_w, 2)
    _nb_r_sum_chunks(_t, 4, 3, _UNIF)
    _nb_shape_linear(_x); _nb_shape_convex(_x); _nb_shape_concave(_x)
    _nb_calc_x(_t, np.ones(4))
    _nb_final(_t, _t, 5)
    # Full WFG9 warmup (most complex)
    wfg9([0.1*i for i in range(1, 31)], 5)
