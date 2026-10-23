# SPDX-License-Identifier: Apache-2.0
# ============================================================================
# The ten single-objective bbob functions the bbob-biobj suite is built from.
# N. Hansen, S. Finck, R. Ros, A. Auger. "Real-Parameter Black-Box
# Optimization Benchmarking 2009: Noiseless Functions Definitions."
# INRIA RR-6829, with the 2019 errata.
#        (source: bbob-functions_hansen2009, bbob-functions-def_hansen2009)
#
# Written from the report. Only the ten f1, f2, f6, f8, f13, f14, f15, f17,
# f20, f21 that Brockhoff et al. select for bbob-biobj are here; the other
# fourteen are not needed and are not implemented.
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
#     rint(100 c) / 100, which C++ repeats.
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
    1: "Sphere", 2: "Ellipsoid", 6: "AttractiveSector", 8: "Rosenbrock",
    13: "SharpRidge", 14: "DifferentPowers", 15: "Rastrigin",
    17: "SchaffersF7", 20: "Schwefel", 21: "Gallagher101",
}

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
        raise ValueError(f"bbob function f{fid} is not one of the ten: {BASE_IDS}")
    rng = _seeds(fid, iid, D)
    x_opt = _uniform(rng, -4.0, 4.0, D)
    if fid == 8:                                   # "Exceptionally, here x_opt in [-3,3]^D"
        x_opt = _uniform(rng, -3.0, 3.0, D)
    inst = Instance(fid=fid, iid=iid, D=D, x_opt=x_opt, f_opt=_draw_f_opt(rng))
    if fid in (6, 13, 14, 15, 17, 21):
        inst.R = _lrng.haar(rng, D)
    if fid in (6, 13, 15, 17):
        inst.Q = _lrng.haar(rng, D)
    if fid in (6, 13, 15, 17, 20):
        inst.l10 = lam(10.0, D)
    if fid == 20:
        s = _plus_minus(rng, D)
        inst.x_opt = 0.5 * _SCHWEFEL * s
        inst.peaks = {"sign": s}
    if fid == 21:
        inst.peaks = _gallagher(rng, D)
        inst.x_opt = inst.peaks["y"][0]            # the global optimum is y_1
    return inst


def _gallagher(rng: _lrng.LibraryRng, D: int) -> dict:
    """f21's 101 peaks: positions y_i, weights w_i and the diagonals of C_i."""
    n = 101
    w = np.empty(n)
    w[0] = 10.0
    w[1:] = 1.1 + 8.0 * np.arange(n - 1) / 99.0
    # alpha_1 = 1000; the rest a permutation of {1000^(2j/99)}, without replacement
    alpha = np.empty(n)
    alpha[0] = 1000.0
    alpha[1:] = rng.shuffle(list(1000.0 ** (2.0 * np.arange(n - 1) / 99.0)))
    C = np.empty((n, D))
    for i in range(n):
        C[i] = np.array(rng.shuffle(list(lam(alpha[i], D)))) / alpha[i] ** 0.25
    y = np.empty((n, D))
    y[0] = _uniform(rng, -4.0, 4.0, D)
    for i in range(1, n):                          # BBOB-1: the errata's box
        y[i] = _uniform(rng, -5.0, 5.0, D)
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


def _f17(s: Instance, X: np.ndarray) -> np.ndarray:
    Z = s.l10 * rows_matvec(s.Q, t_asy(rows_matvec(s.R, X - s.x_opt), 0.5))
    si = np.sqrt(Z[:, :-1] ** 2 + Z[:, 1:] ** 2)                 # BBOB-3: i = 1..D-1
    rt = np.sqrt(si)
    inner = np.sum(rt + rt * np.sin(50.0 * si ** 0.2) ** 2, axis=1) / (s.D - 1.0)
    return inner * inner + 10.0 * f_pen(X) + s.f_opt


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
        U = (X[i] - p["y"]) @ s.R.T                            # (101, D), one point's product
        quad = np.sum(p["C"] * U * U, axis=1)
        best[i] = np.max(p["w"] * np.exp(-quad / (2.0 * s.D)))
    return pow_each(t_osz(10.0 - best), 2) + f_pen(X) + s.f_opt


_EVAL: Dict[int, Callable[[Instance, np.ndarray], np.ndarray]] = {
    1: _f1, 2: _f2, 6: _f6, 8: _f8, 13: _f13,
    14: _f14, 15: _f15, 17: _f17, 20: _f20, 21: _f21,
}


def bounds(D: int) -> List[Tuple[float, float]]:
    """Section 0.1: the functions are defined on R^D, the search domain is [-5,5]^D."""
    return [(-5.0, 5.0)] * D
