# SPDX-License-Identifier: Apache-2.0
# ============================================================================
# bbob-largescale (task 20, package 24-27, §24.6.5, variant (b)): the NumPy
# cross-check of include/mootation/suite/bbob_largescale.hpp, which defines
# the suite (Varelas et al. 2020, Elhara et al. 2019) and gives its readings
# LS-1..8. The same draws of the library's generator (library_rng.py) in the
# same order; the same formulas. The registry evaluates the suite by C++
# (registry.py); these values agree with C++'s to a tolerance (NumPy's exp,
# log, pow and the Haar rotation's QR are not the library's).
# ============================================================================
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import bbob as _b
from . import library_rng as _lrng

BLOCK = 40
DIMS = (20, 40, 80, 160, 320, 640)


def truncated_swaps(rng: _lrng.LibraryRng, n: int) -> np.ndarray:
    """Varelas et al.'s Algorithm 1 (LS-7): n swaps within floor(n / 3)."""
    p = list(range(n))
    pi = rng.shuffle(list(range(n)))
    r = n // 3
    for i in pi:
        lb, ub = max(0, i - r), min(n - 1, i + r)
        if ub == lb:
            continue
        j = lb + rng.below(ub - lb)
        if j >= i:
            j += 1
        p[i], p[j] = p[j], p[i]
    return np.array(p)


@dataclass
class Rotation:
    """R = P_left B P_right, (P v)_i = v[p_i]; B's blocks of size min(n, 40)."""
    n: int
    blocks: list
    left: np.ndarray
    right: np.ndarray

    def block_product(self, v: np.ndarray) -> np.ndarray:
        out, o = np.empty(self.n), 0
        for B in self.blocks:
            k = B.shape[0]
            out[o:o + k] = B @ v[o:o + k]
            o += k
        return out

    def apply(self, v: np.ndarray) -> np.ndarray:
        return self.block_product(np.asarray(v, float)[self.right])[self.left]

    def apply_transposed(self, v: np.ndarray) -> np.ndarray:
        a = np.empty(self.n)
        a[self.left] = v
        b, o = np.empty(self.n), 0
        for B in self.blocks:
            k = B.shape[0]
            b[o:o + k] = B.T @ a[o:o + k]
            o += k
        out = np.empty(self.n)
        out[self.right] = b
        return out


def rotation(rng: _lrng.LibraryRng, n: int) -> Rotation:
    s = min(n, BLOCK)
    blocks = [_lrng.haar(rng, min(s, n - o)) for o in range(0, n, s)]
    left = truncated_swaps(rng, n)
    return Rotation(n, blocks, left, truncated_swaps(rng, n))


def rosenbrock_scale(n: int) -> float:
    return max(1.0, math.sqrt(min(n, BLOCK)) / 8.0)


@dataclass
class Instance:
    fid: int
    n: int
    x_opt: np.ndarray
    f_opt: float
    R: Rotation = field(default=None, repr=False)
    Q: Rotation = field(default=None, repr=False)
    l10: np.ndarray = field(default=None, repr=False)
    la: np.ndarray = field(default=None, repr=False)
    sign: np.ndarray = field(default=None, repr=False)
    peaks: dict = field(default=None, repr=False)

    def __call__(self, x) -> float:
        return float(_value(self, np.asarray(x, float)))


def instance(fid: int, iid: int, n: int) -> Instance:
    """f<fid> at instance iid in n variables, drawn as bbob_largescale.hpp draws it."""
    rng = _lrng.LibraryRng(_lrng.seed(20261113, _lrng.crc32("bbob_largescale"), fid, iid, n))
    x_opt = _b._uniform(rng, -4.0, 4.0, n)
    if fid in (8, 9):
        x_opt = _b._uniform(rng, -3.0, 3.0, n)
    inst = Instance(fid, n, x_opt, _b._draw_f_opt(rng))
    if fid in _b._USES_R:
        inst.R = rotation(rng, n)
    if fid in _b._USES_Q:
        inst.Q = rotation(rng, n)
    if fid in _b._USES_L10:
        inst.l10 = _b.lam(10.0, n)
    if fid in _b._ALPHA:
        inst.la = _b.lam(_b._ALPHA[fid], n)
    if fid in _b._SIGN_SCALE:
        inst.sign = _b._plus_minus(rng, n)
        inst.x_opt = _b._SIGN_SCALE[fid] * inst.sign
    if fid == 19:
        inst.x_opt = inst.R.apply_transposed(np.full(n, 0.5 / rosenbrock_scale(n)))
    if fid in (21, 22):
        inst.peaks = _b._gallagher(rng, n, fid)
        inst.x_opt = inst.peaks["y"][0].copy()
    return inst


def f_opt(fid: int, iid: int, n: int) -> float:
    """The instance's f_opt alone: the draws up to it (the registry's)."""
    rng = _lrng.LibraryRng(_lrng.seed(20261113, _lrng.crc32("bbob_largescale"), fid, iid, n))
    _b._uniform(rng, -4.0, 4.0, n)
    if fid in (8, 9):
        _b._uniform(rng, -3.0, 3.0, n)
    return _b._draw_f_opt(rng)


def _rastrigin(z):
    return 10.0 * (z.size - np.sum(np.cos(2.0 * math.pi * z)))


def _rosenbrock(z):
    return float(np.sum(100.0 * (z[:-1] ** 2 - z[1:]) ** 2 + (z[:-1] - 1.0) ** 2))


def _value(s: Instance, x: np.ndarray) -> float:
    n, fid = s.n, s.fid
    g = min(1.0, 40.0 / n)
    k = -(-n // BLOCK)                         # ceil(n / 40) axes
    t = np.arange(n) / (n - 1.0)
    d = x - s.x_opt
    pen = float(_b.f_pen(x))
    R = s.R.apply if s.R is not None else None
    Q = s.Q.apply if s.Q is not None else None
    c = rosenbrock_scale(n)
    if fid == 1:
        return g * float(d @ d) + s.f_opt
    if fid == 2:
        return g * float(np.sum(10.0 ** (6.0 * t) * _b.t_osz(d) ** 2)) + s.f_opt
    if fid == 3:
        z = s.l10 * _b.t_asy(_b.t_osz(d), 0.2)
        return g * (_rastrigin(z) + float(z @ z)) + s.f_opt
    if fid == 4:
        v = _b.t_osz(d)
        z = np.where((v > 0.0) & (np.arange(n) % 2 == 0), 10.0 * s.l10, s.l10) * v
        return g * (_rastrigin(z) + float(np.sum(z * z))) + 100.0 * pen + s.f_opt
    if fid == 5:
        sl = np.sign(s.x_opt) * 10.0 ** t
        z = np.where(s.x_opt * x < 25.0, x, s.x_opt)
        return g * float(np.sum(5.0 * np.abs(sl) - sl * z)) + s.f_opt
    if fid == 6:
        z = Q(s.l10 * R(d))
        v = np.where(z * s.x_opt > 0.0, 100.0, 1.0) * z
        return float(_b.t_osz(g * float(np.sum(v * v)))) ** 0.9 + s.f_opt
    if fid == 7:
        zh = s.l10 * R(d)
        zt = np.where(np.abs(zh) > 0.5, np.floor(0.5 + zh), np.floor(0.5 + 10.0 * zh) / 10.0)
        z = Q(zt)
        return g * 0.1 * max(abs(zh[0]) / 1e4, float(np.sum(10.0 ** (2.0 * t) * z * z))) + pen + s.f_opt
    if fid == 8:
        return g * _rosenbrock(c * d + 1.0) + s.f_opt
    if fid == 9:
        return g * _rosenbrock(c * R(d) + 1.0) + s.f_opt
    if fid == 10:
        return g * float(np.sum(10.0 ** (6.0 * t) * _b.t_osz(R(d)) ** 2)) + s.f_opt
    if fid == 11:
        z = _b.t_osz(R(d))
        return g * (1e6 * float(np.sum(z[:k] ** 2)) + float(np.sum(z[k:] ** 2))) + s.f_opt
    if fid == 12:
        z = R(_b.t_asy(R(d), 0.5))
        return g * (float(np.sum(z[:k] ** 2)) + 1e6 * float(np.sum(z[k:] ** 2))) + s.f_opt
    if fid == 13:
        z = Q(s.l10 * R(d))
        return g * (float(np.sum(z[:k] ** 2)) + 100.0 * math.sqrt(float(np.sum(z[k:] ** 2)))) + s.f_opt
    if fid == 14:                                  # LS-3: no square root
        return g * float(np.sum(np.abs(R(d)) ** (2.0 + 4.0 * t))) + s.f_opt
    if fid == 15:
        z = R(s.l10 * Q(_b.t_asy(_b.t_osz(R(d)), 0.2)))
        return g * (_rastrigin(z) + float(z @ z)) + s.f_opt
    if fid == 16:
        z = R(s.la * Q(_b.t_osz(R(d))))
        kk = np.arange(12)
        f0 = float(np.sum(0.5 ** kk * np.cos(2.0 * math.pi * 3.0 ** kk * 0.5)))
        acc = np.sum(0.5 ** kk[None, :] * np.cos(2.0 * math.pi * 3.0 ** kk[None, :] * (z[:, None] + 0.5)), axis=1)
        v = float(np.sum(acc)) / n - f0
        return 10.0 * v ** 3 + 10.0 / n * pen + s.f_opt
    if fid in (17, 18):
        z = (s.l10 if fid == 17 else s.la) * Q(_b.t_asy(R(d), 0.5))
        si = np.sqrt(z[:-1] ** 2 + z[1:] ** 2)
        inner = float(np.sum(np.sqrt(si) + np.sqrt(si) * np.sin(50.0 * si ** 0.2) ** 2)) / (n - 1.0)
        return inner * inner + 10.0 * pen + s.f_opt
    if fid == 19:
        z = c * R(x) + 0.5
        si = 100.0 * (z[:-1] ** 2 - z[1:]) ** 2 + (z[:-1] - 1.0) ** 2
        return 10.0 / (n - 1.0) * float(np.sum(si / 4000.0 - np.cos(si))) + 10.0 + s.f_opt
    if fid == 20:
        two_abs = 2.0 * np.abs(s.x_opt)
        xh = 2.0 * s.sign * x
        zh = xh.copy()
        zh[1:] = xh[1:] + 0.25 * (xh[:-1] - two_abs[:-1])
        z = 100.0 * (s.l10 * (zh - two_abs) + two_abs)
        term = -float(np.sum(z * np.sin(np.sqrt(np.abs(z))))) / (100.0 * n)
        return term + 4.189828872724339 + 100.0 * float(_b.f_pen(z / 100.0)) + s.f_opt
    if fid in (21, 22):                            # LS-6: B without the permutations
        P = s.peaks
        vals = [P["w"][i] * math.exp(-float(np.sum(P["C"][i] * s.R.block_product(x - P["y"][i]) ** 2)) / (2.0 * n))
                for i in range(len(P["w"]))]
        return float(_b.t_osz(10.0 - max(vals))) ** 2 + pen + s.f_opt
    if fid == 23:
        z = Q(s.la * R(d))
        two = 2.0 ** np.arange(1, 33)
        v = two[None, :] * z[:, None]
        ssum = np.sum(np.abs(v - np.rint(v)) / two[None, :], axis=1)
        kk = 10.0 / (n * n)
        return kk * float(np.prod((1.0 + np.arange(1, n + 1) * ssum) ** (10.0 / n ** 1.2))) - kk + pen + s.f_opt
    # f24
    ls = 1.0 - 1.0 / (2.0 * math.sqrt(n + 20.0) - 8.2)
    mu1 = -math.sqrt((2.5 * 2.5 - 1.0) / ls)
    xh = 2.0 * s.sign * x
    funnel = min(float(np.sum((xh - 2.5) ** 2)), 1.0 * n + ls * float(np.sum((xh - mu1) ** 2)))
    z = Q(s.la * R(xh - 2.5))
    return g * (funnel + _rastrigin(z)) + 1e4 * pen + s.f_opt
