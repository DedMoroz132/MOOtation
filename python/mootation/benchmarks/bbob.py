# SPDX-License-Identifier: Apache-2.0
# ============================================================================
# The ten single-objective bbob functions the bbob-biobj suite is built from.
# N. Hansen, S. Finck, R. Ros, A. Auger. "Real-Parameter Black-Box
# Optimization Benchmarking 2009: Noiseless Functions Definitions."
# INRIA RR-6829, with the 2019 errata.
#        (source: bbob-functions_hansen2009, bbob-functions-def_hansen2009)
#
# Written from the report. The ten f1, f2, f6, f8, f13, f14, f15, f17, f20,
# f21 that Brockhoff et al. select for bbob-biobj came first; the other
# fourteen came with task 18 (stage 2), when bbob itself, one objective, moved
# to C++ (include/mootation/suite/bbob.hpp, family "bbob"): this module is its
# cross-check, the same expressions in the same order.
#
# THE SHAPE OF EVERY FUNCTION. Each is f(x) = <something>(z) + f_opt, where z
# is x moved to a random optimum x_opt and put through some sequence of a
# rotation R, a conditioning Lambda^alpha, a second rotation Q, and the two
# non-linear symmetry breakers T_osz and T_asy (Section 0.2). x_opt and f_opt
# and the rotations are what an INSTANCE is: they are redrawn per instance, so
# an algorithm cannot learn the optimum's position (Section 0.1).
#
# READINGS (points the converted report leaves open, resolved and declared):
#   BBOB-1. The report marks its 2019 errata "with colored text", which the
#     markdown conversion lost, leaving them as "A --> B" inside the formulae.
#     Read as "A was replaced by B", the later text winning. That is what makes
#     f21's peaks come from [-5,5]^D rather than [-4.9,4.9]^D.
#   BBOB-2. f20 prints 2|x_hat^opt| where the correct term is 2|x^opt|. This
#     one is not a judgement call: x_hat^opt = 2 * s * x^opt = 4.2096874633
#     elementwise, so 2|x_hat^opt| would be 8.419..., and only 2|x^opt| =
#     4.2096874633 makes z = 420.96874633, which is where the Schwefel optimum
#     sits, and hence f20(x_opt) = f_opt. Every one of the ten satisfies that
#     identity, and the tests check it.
#   BBOB-3. f17's printed formula lost a parenthesis. Read as
#     (1/(D-1) * sum_i (sqrt(s_i) + sqrt(s_i) sin^2(50 s_i^0.2)))^2, and its
#     s_i, printed "for i = 1..D", runs to D-1 since it reads z_{i+1}.
#   BBOB-4. THE INSTANCES ARE OURS, NOT COCO'S. The report gives the
#     distributions each instance is drawn from — x_opt uniform on [-4,4]^D,
#     f_opt Cauchy about zero, rotations from Gram-Schmidt on normal entries —
#     but the generator that turns an instance NUMBER into those draws lives
#     only in COCO's source, and no paper states it. Instance 1 here is
#     therefore not instance 1 there: results are comparable between runs of
#     this library and NOT with the COCO data archives. From 2026-10-24 (task
#     17, supplement 5, §3; version 2 of every bbob-biobj problem) the draws
#     are the library's generator's, keyed (20260916, fid, iid, D): the C++
#     suite draws them (include/mootation/suite/bbob.hpp, instance), and this
#     module the same through library_rng.py, the cross-check's; before, NumPy's
#     default_rng drew other instances. f_opt is rounded to two decimals by
#     rint(100 c) / 100, which C++ repeats. The fourteen of task 18 draw in the
#     same order: x_opt, f_opt, R, Q, then the signs or the peaks.
#   BBOB-5. f9 and f19 draw no x_opt: z = max(1, sqrt(D)/8) R x + 1/2 with
#     z_opt = 1, so x_opt = R^T 1 / (2 max(1, sqrt(D)/8)), the sum over the
#     rows in order; f(x_opt) is f_opt to rounding, not exactly.
#   BBOB-6. f23's [.] (the nearest integer) rounds half to even.
#   BBOB-7. f22 as printed after the errata: alpha_1 = 1000^2, y_1 on
#     [-3.92, 3.92]^D, the other peaks on [-4.9, 4.9]^D.
#   BBOB-8. f16's inner sum runs over k = 0..11 in order, each term
#     2^-k cos(2 pi 3^k (z_i + 1/2)) with 2 pi 3^k taken first; f0 the same at
#     z_i = 0. f23's over j = 1..32 in order.
# ============================================================================
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple

import numpy as np

from . import library_rng as _lrng
from ._rows import pow_each, rows_dot, rows_matvec, spow

# The ten the bbob-biobj suite selects (Brockhoff et al., Section 5.1).
BASE_IDS: Tuple[int, ...] = (1, 2, 6, 8, 13, 14, 15, 17, 20, 21)

NAMES: Dict[int, str] = {
    1: "Sphere", 2: "Ellipsoid", 3: "RastriginSeparable", 4: "BucheRastrigin", 5: "LinearSlope",
    6: "AttractiveSector", 7: "StepEllipsoid", 8: "Rosenbrock", 9: "RosenbrockRotated",
    10: "EllipsoidRotated", 11: "Discus", 12: "BentCigar", 13: "SharpRidge", 14: "DifferentPowers",
    15: "Rastrigin", 16: "Weierstrass", 17: "SchaffersF7", 18: "SchaffersF7Cond",
    19: "GriewankRosenbrock", 20: "Schwefel", 21: "Gallagher101", 22: "Gallagher21",
    23: "Katsuura", 24: "LunacekBiRastrigin",
}

# what each function takes (bbob.hpp: uses_r, uses_q, uses_l10, alpha_of, uses_sign)
_USES_R = tuple(f for f in range(1, 25) if not (f <= 5 or f in (8, 20)))
_USES_Q = (6, 7, 13, 15, 16, 17, 18, 23, 24)
_USES_L10 = (3, 4, 6, 7, 13, 15, 17, 20)
_ALPHA = {16: 0.01, 18: 1000.0, 23: 100.0, 24: 100.0}
_SIGN_SCALE = {5: 5.0, 20: 0.5 * 4.2096874633, 24: 0.5 * 2.5}

_SCHWEFEL = 4.2096874633


# ---- the shared transformations (Section 0.2) -----------------------
def t_osz(x):
    """Oscillation, applied elementwise; also used on a scalar (n = 1)."""
    x = np.asarray(x, dtype=float)
    xh = np.where(x != 0.0, np.log(np.abs(np.where(x != 0.0, x, 1.0))), 0.0)
    c1 = np.where(x > 0.0, 10.0, 5.5)
    c2 = np.where(x > 0.0, 7.9, 3.1)
    return np.sign(x) * np.exp(xh + 0.049 * (np.sin(c1 * xh) + np.sin(c2 * xh)))


def t_asy(x: np.ndarray, beta: float) -> np.ndarray:
    """Asymmetry. The identity for negative entries, so 1/2^D of the box is
    untouched. x a point, or a batch of them (rows)."""
    D = x.shape[-1]
    if D < 2:
        return x.copy()
    e = 1.0 + beta * (np.arange(D) / (D - 1.0)) * np.sqrt(np.abs(x))
    return np.where(x > 0.0, np.abs(x) ** e, x)


def lam(alpha: float, D: int) -> np.ndarray:
    """The diagonal of Lambda^alpha: lambda_ii = alpha^(1/2 * (i-1)/(D-1))."""
    if D < 2:
        return np.ones(1)
    return alpha ** (0.5 * np.arange(D) / (D - 1.0))


def f_pen(x: np.ndarray) -> np.ndarray:
    """The penalty outside [-5, 5]^D, of every row of x."""
    return np.sum(np.maximum(0.0, np.abs(x) - 5.0) ** 2, axis=-1)


# ---- what an instance is (Section 0.1) ------------------------------
def _seeds(fid: int, iid: int, D: int) -> _lrng.LibraryRng:
    """Deterministic per (function, instance, dimension). See BBOB-4."""
    return _lrng.LibraryRng(_lrng.seed(20260916, fid, iid, D))


def _uniform(rng: _lrng.LibraryRng, lo: float, hi: float, D: int) -> np.ndarray:
    return np.array([rng.uniform(lo, hi) for _ in range(D)], float)


def _draw_f_opt(rng: _lrng.LibraryRng) -> float:
    """Cauchy about zero, about half the mass in [-100,100], 2 decimals, |.| <= 1000."""
    c = 0.0 + 100.0 * math.tan(math.pi * (rng.uniform01() - 0.5))
    return float(min(1000.0, max(-1000.0, float(np.rint(100.0 * c)) / 100.0)))


def _plus_minus(rng: _lrng.LibraryRng, D: int) -> np.ndarray:
    return np.array([-1.0 if rng.uniform01() < 0.5 else 1.0 for _ in range(D)])


@dataclass
class Instance:
    """One instantiation of one bbob function: everything drawn, precomputed."""
    fid: int
    iid: int
    D: int
    x_opt: np.ndarray
    f_opt: float
    R: np.ndarray = field(default=None, repr=False)
    Q: np.ndarray = field(default=None, repr=False)
    l10: np.ndarray = field(default=None, repr=False)
    la: np.ndarray = field(default=None, repr=False)     # Lambda^alpha of f16, f18, f23, f24
    peaks: dict = field(default=None, repr=False)

    def __call__(self, x) -> float:
        return float(self.batch(np.asarray(x, dtype=float)[None, :])[0])

    def batch(self, X) -> np.ndarray:
        """The values at the rows of X (n x D): the same, bit for bit, as at
        each row alone (_rows.py says what that takes)."""
        return _EVAL[self.fid](self, np.asarray(X, dtype=float).reshape(-1, self.D))


def instance(fid: int, iid: int, D: int) -> Instance:
    """Build function `fid` at instance `iid` in `D` variables."""
    if fid not in NAMES:
        raise ValueError(f"bbob function f{fid}: f1 .. f24")
    rng = _seeds(fid, iid, D)
    x_opt = _uniform(rng, -4.0, 4.0, D)
    if fid == 8:                                   # "Exceptionally, here x_opt in [-3,3]^D"
        x_opt = _uniform(rng, -3.0, 3.0, D)
    inst = Instance(fid=fid, iid=iid, D=D, x_opt=x_opt, f_opt=_draw_f_opt(rng))
    if fid in _USES_R:
        inst.R = _lrng.haar(rng, D)
    if fid in _USES_Q:
        inst.Q = _lrng.haar(rng, D)
    if fid in _USES_L10:
        inst.l10 = lam(10.0, D)
    if fid in _ALPHA:
        inst.la = lam(_ALPHA[fid], D)
    if fid in _SIGN_SCALE:                         # f5, f20, f24
        s = _plus_minus(rng, D)
        inst.x_opt = _SIGN_SCALE[fid] * s
        inst.peaks = {"sign": s}
    if fid in (9, 19):                             # BBOB-5
        two_c = 2.0 * max(1.0, math.sqrt(D) / 8.0)
        x = np.empty(D)
        for j in range(D):
            t = 0.0
            for i in range(D):
                t = t + float(inst.R[i, j])
            x[j] = t / two_c
        inst.x_opt = x
    if fid in (21, 22):
        inst.peaks = _gallagher(rng, D, fid)
        inst.x_opt = inst.peaks["y"][0]            # the global optimum is y_1
    return inst


def _gallagher(rng: _lrng.LibraryRng, D: int, fid: int = 21) -> dict:
    """f21's 101 peaks or f22's 21: positions y_i, weights w_i and the
    diagonals of C_i."""
    n = 101 if fid == 21 else 21
    q = n - 2.0                                    # 99, 19
    w = np.empty(n)
    w[0] = 10.0
    w[1:] = 1.1 + 8.0 * np.arange(n - 1) / q
    # alpha_1 = 1000 (f22: 1000^2); the rest a permutation of {1000^(2j/q)}, without replacement
    alpha = np.empty(n)
    alpha[0] = 1000.0 if fid == 21 else 1000.0 * 1000.0
    alpha[1:] = rng.shuffle(list(1000.0 ** (2.0 * np.arange(n - 1) / q)))
    C = np.empty((n, D))
    for i in range(n):
        C[i] = np.array(rng.shuffle(list(lam(alpha[i], D)))) / alpha[i] ** 0.25
    first, rest = (4.0, 5.0) if fid == 21 else (3.92, 4.9)     # BBOB-1, BBOB-7
    y = np.empty((n, D))
    y[0] = _uniform(rng, -first, first, D)
    for i in range(1, n):
        y[i] = _uniform(rng, -rest, rest, D)
    return {"w": w, "C": C, "y": y}


# ---- the ten functions ----------------------------------------------
# ---- the ten functions, of a batch X (one point per row) -----------
# Every row's value is computed by the expressions of the function at one
# point; where one point's computation was a matrix product, a dot product or
# a power of a scalar, the batch takes it row by row (_rows.py).
def _f1(s: Instance, X: np.ndarray) -> np.ndarray:
    Z = X - s.x_opt
    return rows_dot(Z) + s.f_opt


def _f2(s: Instance, X: np.ndarray) -> np.ndarray:
    Z = t_osz(X - s.x_opt)
    return np.sum(10.0 ** (6.0 * np.arange(s.D) / (s.D - 1.0)) * Z * Z, axis=1) + s.f_opt


def _f6(s: Instance, X: np.ndarray) -> np.ndarray:
    Z = rows_matvec(s.Q, s.l10 * rows_matvec(s.R, X - s.x_opt))
    si = np.where(Z * s.x_opt > 0.0, 100.0, 1.0)
    return spow(t_osz(np.sum((si * Z) ** 2, axis=1)), 0.9) + s.f_opt


def _f8(s: Instance, X: np.ndarray) -> np.ndarray:
    Z = max(1.0, math.sqrt(s.D) / 8.0) * (X - s.x_opt) + 1.0
    a = Z[:, :-1] ** 2 - Z[:, 1:]
    b = Z[:, :-1] - 1.0
    return np.sum(100.0 * a * a + b * b, axis=1) + s.f_opt


def _f13(s: Instance, X: np.ndarray) -> np.ndarray:
    Z = rows_matvec(s.Q, s.l10 * rows_matvec(s.R, X - s.x_opt))
    return pow_each(Z[:, 0], 2) + 100.0 * np.sqrt(rows_dot(Z[:, 1:])) + s.f_opt


def _f14(s: Instance, X: np.ndarray) -> np.ndarray:
    Z = rows_matvec(s.R, X - s.x_opt)
    e = 2.0 + 4.0 * np.arange(s.D) / (s.D - 1.0)
    return np.sqrt(np.sum(np.abs(Z) ** e, axis=1)) + s.f_opt


def _f15(s: Instance, X: np.ndarray) -> np.ndarray:
    Z = rows_matvec(s.R, s.l10 * rows_matvec(s.Q, t_asy(t_osz(rows_matvec(s.R, X - s.x_opt)),
                                                          0.2)))
    return (10.0 * (s.D - np.sum(np.cos(2.0 * math.pi * Z), axis=1)) + rows_dot(Z)) + s.f_opt


def _schaffer(s: Instance, X: np.ndarray, lam_: np.ndarray) -> np.ndarray:
    """f17 with Lambda^10, f18 with Lambda^1000."""
    Z = lam_ * rows_matvec(s.Q, t_asy(rows_matvec(s.R, X - s.x_opt), 0.5))
    si = np.sqrt(Z[:, :-1] ** 2 + Z[:, 1:] ** 2)                 # BBOB-3: i = 1..D-1
    rt = np.sqrt(si)
    inner = np.sum(rt + rt * np.sin(50.0 * si ** 0.2) ** 2, axis=1) / (s.D - 1.0)
    return inner * inner + 10.0 * f_pen(X) + s.f_opt


def _f17(s: Instance, X: np.ndarray) -> np.ndarray:
    return _schaffer(s, X, s.l10)


def _f18(s: Instance, X: np.ndarray) -> np.ndarray:
    return _schaffer(s, X, s.la)


def _f20(s: Instance, X: np.ndarray) -> np.ndarray:
    sign = s.peaks["sign"]
    xh = 2.0 * sign * X
    two_abs = 2.0 * np.abs(s.x_opt)                           # BBOB-2
    zh = np.empty_like(xh)
    zh[:, 0] = xh[:, 0]
    zh[:, 1:] = xh[:, 1:] + 0.25 * (xh[:, :-1] - two_abs[:-1])
    Z = 100.0 * (s.l10 * (zh - two_abs) + two_abs)
    term = -np.sum(Z * np.sin(np.sqrt(np.abs(Z))), axis=1) / (100.0 * s.D)
    return term + 4.189828872724339 + 100.0 * f_pen(Z / 100.0) + s.f_opt


def _f21(s: Instance, X: np.ndarray) -> np.ndarray:
    p = s.peaks
    best = np.empty(len(X))
    for i in range(len(X)):
        U = (X[i] - p["y"]) @ s.R.T                            # (peaks, D), one point's product
        quad = np.sum(p["C"] * U * U, axis=1)
        best[i] = np.max(p["w"] * np.exp(-quad / (2.0 * s.D)))
    return pow_each(t_osz(10.0 - best), 2) + f_pen(X) + s.f_opt


# ---- the fourteen of task 18 -----------------------------------------
def _rastrigin(Z: np.ndarray) -> np.ndarray:
    """10 (D - sum cos(2 pi z_i)) of every row."""
    return 10.0 * (Z.shape[1] - np.sum(np.cos(2.0 * math.pi * Z), axis=1))


def _rosenbrock(Z: np.ndarray) -> np.ndarray:
    a = Z[:, :-1] * Z[:, :-1] - Z[:, 1:]
    b = Z[:, :-1] - 1.0
    return np.sum(100.0 * a * a + b * b, axis=1)


def _f3(s: Instance, X: np.ndarray) -> np.ndarray:
    Z = s.l10 * t_asy(t_osz(X - s.x_opt), 0.2)
    return (_rastrigin(Z) + rows_dot(Z)) + s.f_opt


def _f4(s: Instance, X: np.ndarray) -> np.ndarray:
    T = t_osz(X - s.x_opt)
    odd = (np.arange(s.D) % 2) == 0                             # i = 1, 3, 5, ... (1-based)
    Z = np.where((T > 0.0) & odd, 10.0 * s.l10, s.l10) * T
    return _rastrigin(Z) + np.sum(Z * Z, axis=1) + 100.0 * f_pen(X) + s.f_opt


def _f5(s: Instance, X: np.ndarray) -> np.ndarray:
    si = np.sign(s.x_opt) * 10.0 ** (np.arange(s.D) / (s.D - 1.0))
    Z = np.where(s.x_opt * X < 25.0, X, s.x_opt)
    return np.sum(5.0 * np.abs(si) - si * Z, axis=1) + s.f_opt


def _f7(s: Instance, X: np.ndarray) -> np.ndarray:
    Zh = s.l10 * rows_matvec(s.R, X - s.x_opt)
    Zt = np.where(np.abs(Zh) > 0.5, np.floor(0.5 + Zh), np.floor(0.5 + 10.0 * Zh) / 10.0)
    Z = rows_matvec(s.Q, Zt)
    c = 10.0 ** (2.0 * np.arange(s.D) / (s.D - 1.0))
    big = np.maximum(np.abs(Zh[:, 0]) / 1e4, np.sum(c * Z * Z, axis=1))
    return 0.1 * big + f_pen(X) + s.f_opt


def _f9(s: Instance, X: np.ndarray) -> np.ndarray:
    Z = max(1.0, math.sqrt(s.D) / 8.0) * rows_matvec(s.R, X) + 0.5   # BBOB-5
    return _rosenbrock(Z) + s.f_opt


def _f10(s: Instance, X: np.ndarray) -> np.ndarray:
    Z = t_osz(rows_matvec(s.R, X - s.x_opt))
    return np.sum(10.0 ** (6.0 * np.arange(s.D) / (s.D - 1.0)) * Z * Z, axis=1) + s.f_opt


def _f11(s: Instance, X: np.ndarray) -> np.ndarray:
    Z = t_osz(rows_matvec(s.R, X - s.x_opt))
    return 1e6 * (Z[:, 0] * Z[:, 0]) + np.sum(Z[:, 1:] * Z[:, 1:], axis=1) + s.f_opt


def _f12(s: Instance, X: np.ndarray) -> np.ndarray:
    Z = rows_matvec(s.R, t_asy(rows_matvec(s.R, X - s.x_opt), 0.5))
    return Z[:, 0] * Z[:, 0] + 1e6 * np.sum(Z[:, 1:] * Z[:, 1:], axis=1) + s.f_opt


def _f16(s: Instance, X: np.ndarray) -> np.ndarray:
    Z = rows_matvec(s.R, s.la * rows_matvec(s.Q, t_osz(rows_matvec(s.R, X - s.x_opt))))
    acc = np.zeros_like(Z)
    f0 = 0.0
    three, half = 1.0, 1.0
    for _ in range(12):                                          # BBOB-8
        wave = 2.0 * math.pi * three
        acc = acc + half * np.cos(wave * (Z + 0.5))
        f0 = f0 + half * math.cos(wave * 0.5)
        three, half = 3.0 * three, 0.5 * half
    G = np.sum(acc, axis=1) / s.D - f0
    return 10.0 * (G * G * G) + (10.0 / s.D) * f_pen(X) + s.f_opt


def _f19(s: Instance, X: np.ndarray) -> np.ndarray:
    Z = max(1.0, math.sqrt(s.D) / 8.0) * rows_matvec(s.R, X) + 0.5   # BBOB-5
    a = Z[:, :-1] * Z[:, :-1] - Z[:, 1:]
    b = Z[:, :-1] - 1.0
    S = 100.0 * a * a + b * b
    return 10.0 / (s.D - 1.0) * np.sum(S / 4000.0 - np.cos(S), axis=1) + 10.0 + s.f_opt


def _f23(s: Instance, X: np.ndarray) -> np.ndarray:
    Z = rows_matvec(s.Q, s.la * rows_matvec(s.R, X - s.x_opt))
    S = np.zeros_like(Z)
    two = 1.0
    for _ in range(32):                                          # BBOB-6, BBOB-8
        two = 2.0 * two
        V = two * Z
        S = S + np.abs(V - np.rint(V)) / two
    P = spow(1.0 + np.arange(1.0, s.D + 1.0) * S, 10.0 / float(s.D) ** 1.2)
    k = 10.0 / (s.D * s.D)
    prod = P[:, 0].copy()
    for i in range(1, s.D):                                      # left to right, as np.prod
        prod = prod * P[:, i]
    return k * prod - k + f_pen(X) + s.f_opt


def _f24(s: Instance, X: np.ndarray) -> np.ndarray:
    D = s.D
    sv = 1.0 - 1.0 / (2.0 * math.sqrt(D + 20.0) - 8.2)
    mu1 = -math.sqrt((2.5 * 2.5 - 1.0) / sv)
    Xh = 2.0 * s.peaks["sign"] * X
    A = Xh - 2.5
    B = Xh - mu1
    funnel = np.minimum(np.sum(A * A, axis=1), 1.0 * D + sv * np.sum(B * B, axis=1))
    Z = rows_matvec(s.Q, s.la * rows_matvec(s.R, A))
    return funnel + _rastrigin(Z) + 1e4 * f_pen(X) + s.f_opt


_EVAL: Dict[int, Callable[[Instance, np.ndarray], np.ndarray]] = {
    1: _f1, 2: _f2, 3: _f3, 4: _f4, 5: _f5, 6: _f6, 7: _f7, 8: _f8, 9: _f9, 10: _f10,
    11: _f11, 12: _f12, 13: _f13, 14: _f14, 15: _f15, 16: _f16, 17: _f17, 18: _f18,
    19: _f19, 20: _f20, 21: _f21, 22: _f21, 23: _f23, 24: _f24,
}


def bounds(D: int) -> List[Tuple[float, float]]:
    """Section 0.1: the functions are defined on R^D, the search domain is [-5,5]^D."""
    return [(-5.0, 5.0)] * D
