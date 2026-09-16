# SPDX-License-Identifier: Apache-2.0
# ============================================================================
# The ten single-objective bbob functions the bbob-biobj suite is built from.
# N. Hansen, S. Finck, R. Ros, A. Auger. "Real-Parameter Black-Box
# Optimization Benchmarking 2009: Noiseless Functions Definitions."
# INRIA RR-6829, with the 2019 errata.        (source: ReadyMarkdown/RR-6829v2)
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
#     this library and NOT with the COCO data archives.
# ============================================================================
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple

import numpy as np

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
    """Asymmetry. The identity for negative entries, so 1/2^D of the box is untouched."""
    D = len(x)
    if D < 2:
        return x.copy()
    e = 1.0 + beta * (np.arange(D) / (D - 1.0)) * np.sqrt(np.abs(x))
    return np.where(x > 0.0, np.abs(x) ** e, x)


def lam(alpha: float, D: int) -> np.ndarray:
    """The diagonal of Lambda^alpha: lambda_ii = alpha^(1/2 * (i-1)/(D-1))."""
    if D < 2:
        return np.ones(1)
    return alpha ** (0.5 * np.arange(D) / (D - 1.0))


def f_pen(x: np.ndarray) -> float:
    return float(np.sum(np.maximum(0.0, np.abs(x) - 5.0) ** 2))


# ---- what an instance is (Section 0.1) ------------------------------
def _seeds(fid: int, iid: int, D: int) -> np.random.Generator:
    """Deterministic per (function, instance, dimension). See BBOB-4."""
    return np.random.default_rng(np.random.SeedSequence([20260916, fid, iid, D]))


def _rotation(rng: np.random.Generator, D: int) -> np.ndarray:
    """An orthogonal matrix from Gram-Schmidt on standard normal entries.

    numpy's QR is the same orthonormalisation; the sign fix-up is what makes it
    a uniform (Haar) draw rather than one biased by LAPACK's sign convention,
    which otherwise correlates the first row with the identity.
    """
    A = rng.standard_normal((D, D))
    Q, R = np.linalg.qr(A)
    return Q * np.sign(np.where(np.diag(R) == 0.0, 1.0, np.diag(R)))


def _draw_f_opt(rng: np.random.Generator) -> float:
    """Cauchy about zero, about half the mass in [-100,100], 2 decimals, |.| <= 1000."""
    c = 100.0 * math.tan(math.pi * (rng.random() - 0.5))
    return float(min(1000.0, max(-1000.0, round(c, 2))))


def _plus_minus(rng: np.random.Generator, D: int) -> np.ndarray:
    return np.where(rng.random(D) < 0.5, -1.0, 1.0)


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
        return _EVAL[self.fid](self, np.asarray(x, dtype=float))


def instance(fid: int, iid: int, D: int) -> Instance:
    """Build function `fid` at instance `iid` in `D` variables."""
    if fid not in NAMES:
        raise ValueError(f"bbob function f{fid} is not one of the ten: {BASE_IDS}")
    rng = _seeds(fid, iid, D)
    x_opt = rng.uniform(-4.0, 4.0, D)
    if fid == 8:                                   # "Exceptionally, here x_opt in [-3,3]^D"
        x_opt = rng.uniform(-3.0, 3.0, D)
    inst = Instance(fid=fid, iid=iid, D=D, x_opt=x_opt, f_opt=_draw_f_opt(rng))
    if fid in (6, 13, 14, 15, 17, 21):
        inst.R = _rotation(rng, D)
    if fid in (6, 13, 15, 17):
        inst.Q = _rotation(rng, D)
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


def _gallagher(rng: np.random.Generator, D: int) -> dict:
    """f21's 101 peaks: positions y_i, weights w_i and the diagonals of C_i."""
    n = 101
    w = np.empty(n)
    w[0] = 10.0
    w[1:] = 1.1 + 8.0 * np.arange(n - 1) / 99.0
    # alpha_1 = 1000; the rest a permutation of {1000^(2j/99)}, without replacement
    alpha = np.empty(n)
    alpha[0] = 1000.0
    alpha[1:] = rng.permutation(1000.0 ** (2.0 * np.arange(n - 1) / 99.0))
    C = np.empty((n, D))
    for i in range(n):
        C[i] = rng.permutation(lam(alpha[i], D)) / alpha[i] ** 0.25
    y = np.empty((n, D))
    y[0] = rng.uniform(-4.0, 4.0, D)
    y[1:] = rng.uniform(-5.0, 5.0, (n - 1, D))     # BBOB-1: the errata's box
    return {"w": w, "C": C, "y": y}


# ---- the ten functions ----------------------------------------------
def _f1(s: Instance, x: np.ndarray) -> float:
    z = x - s.x_opt
    return float(z @ z) + s.f_opt


def _f2(s: Instance, x: np.ndarray) -> float:
    z = t_osz(x - s.x_opt)
    return float(np.sum(10.0 ** (6.0 * np.arange(s.D) / (s.D - 1.0)) * z * z)) + s.f_opt


def _f6(s: Instance, x: np.ndarray) -> float:
    z = s.Q @ (s.l10 * (s.R @ (x - s.x_opt)))
    si = np.where(z * s.x_opt > 0.0, 100.0, 1.0)
    return float(t_osz(float(np.sum((si * z) ** 2))) ** 0.9) + s.f_opt


def _f8(s: Instance, x: np.ndarray) -> float:
    z = max(1.0, math.sqrt(s.D) / 8.0) * (x - s.x_opt) + 1.0
    a = z[:-1] ** 2 - z[1:]
    b = z[:-1] - 1.0
    return float(np.sum(100.0 * a * a + b * b)) + s.f_opt


def _f13(s: Instance, x: np.ndarray) -> float:
    z = s.Q @ (s.l10 * (s.R @ (x - s.x_opt)))
    return float(z[0] ** 2 + 100.0 * math.sqrt(float(z[1:] @ z[1:]))) + s.f_opt


def _f14(s: Instance, x: np.ndarray) -> float:
    z = s.R @ (x - s.x_opt)
    e = 2.0 + 4.0 * np.arange(s.D) / (s.D - 1.0)
    return float(math.sqrt(float(np.sum(np.abs(z) ** e)))) + s.f_opt


def _f15(s: Instance, x: np.ndarray) -> float:
    z = s.R @ (s.l10 * (s.Q @ t_asy(t_osz(s.R @ (x - s.x_opt)), 0.2)))
    return float(10.0 * (s.D - np.sum(np.cos(2.0 * math.pi * z))) + z @ z) + s.f_opt


def _f17(s: Instance, x: np.ndarray) -> float:
    z = s.l10 * (s.Q @ t_asy(s.R @ (x - s.x_opt), 0.5))
    si = np.sqrt(z[:-1] ** 2 + z[1:] ** 2)                    # BBOB-3: i = 1..D-1
    rt = np.sqrt(si)
    inner = float(np.sum(rt + rt * np.sin(50.0 * si ** 0.2) ** 2)) / (s.D - 1.0)
    return inner * inner + 10.0 * f_pen(x) + s.f_opt


def _f20(s: Instance, x: np.ndarray) -> float:
    sign = s.peaks["sign"]
    xh = 2.0 * sign * x
    two_abs = 2.0 * np.abs(s.x_opt)                           # BBOB-2
    zh = np.empty(s.D)
    zh[0] = xh[0]
    zh[1:] = xh[1:] + 0.25 * (xh[:-1] - two_abs[:-1])
    z = 100.0 * (s.l10 * (zh - two_abs) + two_abs)
    term = -float(np.sum(z * np.sin(np.sqrt(np.abs(z))))) / (100.0 * s.D)
    return term + 4.189828872724339 + 100.0 * f_pen(z / 100.0) + s.f_opt


def _f21(s: Instance, x: np.ndarray) -> float:
    p = s.peaks
    U = (x - p["y"]) @ s.R.T                                   # (101, D)
    quad = np.sum(p["C"] * U * U, axis=1)
    best = float(np.max(p["w"] * np.exp(-quad / (2.0 * s.D))))
    return float(t_osz(10.0 - best)) ** 2 + f_pen(x) + s.f_opt


_EVAL: Dict[int, Callable[[Instance, np.ndarray], float]] = {
    1: _f1, 2: _f2, 6: _f6, 8: _f8, 13: _f13,
    14: _f14, 15: _f15, 17: _f17, 20: _f20, 21: _f21,
}


def bounds(D: int) -> List[Tuple[float, float]]:
    """Section 0.1: the functions are defined on R^D, the search domain is [-5,5]^D."""
    return [(-5.0, 5.0)] * D
