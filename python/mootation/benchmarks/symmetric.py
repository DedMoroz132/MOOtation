# SPDX-License-Identifier: Apache-2.0
# ============================================================================
# The problems with symmetries at one objective (task 19, §19.1): the
# registry's instances of the C++ suite's families sumexp, lowrank, cp3,
# cp3_degenerate, rbfnet, mlp_plateau, chaboche1d and lj_cluster, and of task
# 20's (package 24-27, §24.6.2) morse_cluster, circles_in_circle and
# circles_in_square, the public sets to check PBH, MDE and IDTS against their
# papers (include/mootation/suite/sym.hpp and the headers it names, which define
# them and give the readings), and the same families in NumPy for the
# cross-check (python/test_symmetric.py): the same draws of the library's
# generator (library_rng.py), the same formulas. The registry evaluates them
# by C++ (the family "sym" specs, registry.py); this module's values agree
# with C++'s to a tolerance (NumPy's exp, log and pow are not dm::'s, the
# normal draws take math.log).
#
# f_opt: 0 ("exact") where the student has the teacher's size and there is no
# noise; cp3_degenerate's 0 is an infimum, not attained ("infimum"); the
# best known value otherwise ("best_known", of version SO_REFERENCE_VERSION):
# lm's (scalar/lm.hpp, ftol = xtol = 1e-15) least 1/2 ||r||^2 from the
# instance's start and 199 starts about it (SYMR-1); lj_cluster's are Wales &
# Doye's Table I, Table II's icosahedral minima its f_local (lj.hpp);
# morse_cluster's Grosso, Locatelli & Schoen's Table 1 (morse.hpp); the
# circles' 0 ("exact": a packing fits the container, the record's times
# 1 + eps, circles.hpp, CIRC-2).
#
# READINGS:
#   SYMR-1. The best known values are our lm's least over the 200 starts,
#     each reached from many of them; not proven minima.
#   SYMR-2. Budgets: the least-squares instances take MGH's 1000 evaluations
#     and pop_size 1 (least_squares.py, LSQ-4: problems for the local solvers,
#     with a start x0); lj_cluster bbob's 10^4 D (pop_size 100, n_gen 100 D)
#     and its own random start as x0; so do the global problems of task 20
#     (GLOBAL), the circles least squares or not.
#   SYMR-3. Targets (so_metrics.hpp): the owner's rule of 01.11 names bbob
#     (absolute) and MGH, NIST (relative from x0); the least-squares instances
#     here take the latter, lj_cluster, a global problem with a best known
#     energy, the former, and so do the rest of GLOBAL.
# ============================================================================
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import library_rng as _lrng

KEY = 20261102


@dataclass
class Instance:
    teacher: int = 0
    student: int = 0
    points: int = 0
    dim: int = 0
    shape: tuple = ()
    noise: float = 0.0
    ratio: float = 0.0
    observed: float = 1.0
    variant: str = ""
    embed: float = 0.5
    instance: int = 1
    rho: float = 0.0              # morse_cluster (task 20, §24.6.2)
    container: float = 0.0        # circles_in_circle: R; circles_in_square: L

    def spec(self) -> dict:
        d = {"teacher": self.teacher, "student": self.student, "points": self.points, "dim": self.dim,
             "shape": list(self.shape), "noise": self.noise, "ratio": self.ratio, "observed": self.observed,
             "variant": self.variant, "embed": self.embed, "instance": self.instance}
        if self.rho:                  # their families' only, as the extension's spec: the rest as before
            d["rho"] = self.rho
        if self.container:
            d["container"] = self.container
        return d


@dataclass
class Row:
    """One registry instance: its name stem, family, numbers, minimum."""
    name: str
    family: str
    inst: Instance
    f_opt: float
    f_opt_kind: str
    f_local: tuple = field(default=())


# SYMR-1: lm's least 1/2 ||r||^2
_BEST = {
    "sumexp_K3_r4_s2": 0.0071095338512318597,
    "sumexp_K3_r4_snr1e3": 0.0002673934049386662,
    "sumexp_K3_r1.5_snr1e3": 0.00025429855914584586,
    "sumexp_lanczos": 2.1453101033057022e-06,
    "lowrank_m20_n15_r3_snr1e2": 0.56927996342595,
    "chaboche1d_N3_s2": 4.6752718957006572,
    "chaboche1d_N2_snr1e2": 3977.9922921133671,
}

# Wales & Doye 1997, Table I and Table II (lj.hpp)
LJ_BEST = {13: -44.326801, 19: -72.659782, 26: -108.315616, 38: -173.928427, 55: -279.248470, 75: -397.492331}
LJ_ICOSAHEDRAL = {38: -173.252378, 75: -396.282249}

# Grosso, Locatelli & Schoen 2007 (experimental analysis), Table 1, rho = 14 (morse.hpp)
MORSE_BEST = {30: -106.835, 55: -220.646, 61: -249.587, 79: -336.798}
# Lai et al. 2022, Table 2: R* of N unit circles in a circle; Costa, Hansen &
# Liberti, Table 4: r* of N circles in the unit square, L* = 1 / r* (circles.hpp)
CIRCLE_R = {66: 9.096279427, 82: 10.050824224}
SQUARE_r = {20: 0.111382, 30: 0.091671, 40: 0.079186, 50: 0.071377}
EPS = (2, 3)                       # the container: the record's times 1 + 10^-e (CIRC-2)
# the global problems: bbob's budgets and absolute targets (SYMR-2, SYMR-3)
GLOBAL = ("lj_cluster", "morse_cluster", "circles_in_circle", "circles_in_square")


def _rows() -> list:
    I = Instance
    rows = [
        Row("sumexp_K3_r4", "sumexp", I(teacher=3, points=64, ratio=4.0), 0.0, "exact"),
        Row("sumexp_K3_r4_s2", "sumexp", I(teacher=3, student=2, points=64, ratio=4.0), None, "best_known"),
        Row("sumexp_K3_r4_snr1e3", "sumexp", I(teacher=3, points=64, ratio=4.0, noise=1e-3), None, "best_known"),
        Row("sumexp_K3_r1.5_snr1e3", "sumexp", I(teacher=3, points=64, ratio=1.5, noise=1e-3), None, "best_known"),
        Row("sumexp_lanczos", "sumexp", I(student=2, variant="lanczos"), None, "best_known"),
        Row("sumexp_lanczos_K3", "sumexp", I(student=3, variant="lanczos"), 0.0, "exact"),
        Row("lowrank_m20_n15_r3", "lowrank", I(teacher=3, shape=(20, 15)), 0.0, "exact"),
        Row("lowrank_m20_n15_r3_obs0.5", "lowrank", I(teacher=3, shape=(20, 15), observed=0.5), 0.0, "exact"),
        Row("lowrank_m20_n15_r3_snr1e2", "lowrank", I(teacher=3, shape=(20, 15), noise=1e-2), None, "best_known"),
        Row("cp3_I5_J4_K3_R3", "cp3", I(teacher=3, shape=(5, 4, 3)), 0.0, "exact"),
        Row("cp3_degenerate_I2_J2_K2", "cp3_degenerate", I(shape=(2, 2, 2)), 0.0, "infimum"),
        Row("cp3_degenerate_I3_J3_K3", "cp3_degenerate", I(shape=(3, 3, 3)), 0.0, "infimum"),
        Row("rbfnet_d2_K3_N100", "rbfnet", I(teacher=3, points=100, dim=2), 0.0, "exact"),
        Row("rbfnet_d5_K5_N200", "rbfnet", I(teacher=5, points=200, dim=5), 0.0, "exact"),
        Row("mlp_plateau_d2_H2_N100_random", "mlp_plateau", I(teacher=3, points=100, dim=2, variant="random"),
            0.0, "exact"),
        Row("mlp_plateau_d2_H2_N100_dup0.5", "mlp_plateau",
            I(teacher=3, points=100, dim=2, variant="duplicate", embed=0.5), 0.0, "exact"),
        Row("mlp_plateau_d2_H2_N100_dup1.5", "mlp_plateau",
            I(teacher=3, points=100, dim=2, variant="duplicate", embed=1.5), 0.0, "exact"),
        Row("mlp_plateau_d2_H2_N100_bias", "mlp_plateau",
            I(teacher=3, points=100, dim=2, variant="bias", embed=0.5), 0.0, "exact"),
        Row("chaboche1d_N2", "chaboche1d", I(teacher=2), 0.0, "exact"),
        Row("chaboche1d_N3_s2", "chaboche1d", I(teacher=3, student=2), None, "best_known"),
        Row("chaboche1d_N2_snr1e2", "chaboche1d", I(teacher=2, noise=1e-2), None, "best_known"),
    ]
    for r in rows:
        if r.f_opt is None:
            r.f_opt = _BEST[r.name]
    for N in sorted(LJ_BEST):
        rows.append(Row(f"lj_cluster_N{N}", "lj_cluster", I(dim=N), LJ_BEST[N], "best_known",
                        (LJ_ICOSAHEDRAL[N],) if N in LJ_ICOSAHEDRAL else ()))
    for N in sorted(MORSE_BEST):
        rows.append(Row(f"morse_cluster_N{N}_rho14", "morse_cluster", I(dim=N, rho=14.0), MORSE_BEST[N], "best_known"))
    for N in sorted(CIRCLE_R):
        for e in EPS:
            rows.append(Row(f"circles_in_circle_N{N}_e{e}", "circles_in_circle",
                            I(dim=N, container=CIRCLE_R[N] * (1.0 + 10.0 ** -e)), 0.0, "exact"))
    for N in sorted(SQUARE_r):
        for e in EPS:
            rows.append(Row(f"circles_in_square_N{N}_e{e}", "circles_in_square",
                            I(dim=N, container=(1.0 / SQUARE_r[N]) * (1.0 + 10.0 ** -e)), 0.0, "exact"))
    return rows


ROWS = _rows()
BUDGET = 1000                                    # SYMR-2


def name(row: Row) -> str:
    return row.name + "_1D"


def spec(row: Row, n_var: int = 0) -> dict:
    return {"family": row.family, "n_var": n_var, "n_obj": 1, "sym": row.inst.spec()}


# ── the families in NumPy (the cross-check) ─────────────────────────────────
def _rng(name_: str, *key) -> _lrng.LibraryRng:
    return _lrng.LibraryRng(_lrng.seed(KEY, _lrng.crc32(name_), *key))


def _noise(rng: _lrng.LibraryRng, noise: float, y: np.ndarray) -> np.ndarray:
    if noise <= 0.0:
        return y
    sigma = noise * float(np.max(np.abs(y)))
    return np.array([v + sigma * rng.normal01() for v in y])


def _sort_blocks(x: np.ndarray, offset: int, size: int, count: int, key) -> np.ndarray:
    blocks = [list(x[offset + k * size: offset + (k + 1) * size]) for k in range(count)]
    blocks.sort(key=lambda b: (key(b), b))
    out = x.copy()
    for k, b in enumerate(blocks):
        out[offset + k * size: offset + (k + 1) * size] = b
    return out


def resolution(snr: float) -> float:
    """Istratov & Vyvenko's resolution limit (12)-(13)."""
    return math.exp(math.pi / (math.acosh(math.pi * snr * snr) / math.pi))


class Sumexp:
    def __init__(self, inst: Instance):
        if inst.variant == "lanczos":
            a = [0.0951, 0.8607, 1.5576]
            s = [-math.log(r) for r in (1.0, 3.0, 5.0)]
            self.t = np.array([0.05 * i for i in range(24)])
        else:
            a, s, tau = [], [], 1.0
            for k in range(inst.teacher):
                a.append(1.0)
                s.append(math.log(tau))
                if k + 1 < inst.teacher:
                    tau = tau * inst.ratio
            lo, hi = math.log(0.1), math.log(10.0 * tau)
            self.t = np.array([math.exp(lo + (hi - lo) * (i / (inst.points - 1))) for i in range(inst.points)])
        y = self._model(np.array(a), np.array(s))
        self.y = _noise(_rng("sumexp", inst.instance, len(a), len(self.t)), inst.noise, y)
        self.K = inst.student or len(a)
        self.res_blocks = [len(self.t) // 2, len(self.t) - len(self.t) // 2]   # SE-4 (task 19, §19.5)

    def _model(self, a, s):
        return np.exp(-np.outer(self.t, np.exp(-s))) @ a

    def residuals(self, x):
        x = np.asarray(x, float)
        return self._model(x[0::2], x[1::2]) - self.y

    def jacobian(self, x):
        x = np.asarray(x, float)
        rate = np.exp(-x[1::2])
        E = np.exp(-np.outer(self.t, rate))
        J = np.empty((len(self.t), 2 * self.K))
        J[:, 0::2] = E
        J[:, 1::2] = E * x[0::2] * np.outer(self.t, rate)
        return J

    def canon(self, x):
        return _sort_blocks(np.asarray(x, float), 0, 2, self.K, lambda b: b[1])


class Lowrank:
    def __init__(self, inst: Instance):
        m, n = inst.shape
        rt = inst.teacher
        rng = _rng("lowrank", inst.instance, m, n, rt)
        U0 = np.array([rng.normal01() for _ in range(m * rt)]).reshape(m, rt)
        V0 = np.array([rng.normal01() for _ in range(n * rt)]).reshape(n, rt)
        self.obs = []
        for i in range(m):
            for j in range(n):
                if inst.observed < 1.0 and not rng.uniform01() < inst.observed:
                    continue
                self.obs.append((i, j))
        Y = U0 @ V0.T
        self.y = _noise(rng, inst.noise, np.array([Y[i, j] for i, j in self.obs]))
        self.m, self.n, self.r = m, n, inst.student or rt

    def _uv(self, x):
        w = self.m + self.n
        x = np.asarray(x, float).reshape(self.r, w)
        return x[:, :self.m].T, x[:, self.m:].T          # U (m x r), V (n x r)

    def residuals(self, x):
        U, V = self._uv(x)
        P = U @ V.T
        return np.array([P[i, j] for i, j in self.obs]) - self.y

    def jacobian(self, x):
        U, V = self._uv(x)
        w = self.m + self.n
        J = np.zeros((len(self.obs), self.r * w))
        for e, (i, j) in enumerate(self.obs):
            for k in range(self.r):
                J[e, k * w + i] = V[j, k]
                J[e, k * w + self.m + j] = U[i, k]
        return J

    def gauge_basis(self, x):
        x = np.asarray(x, float)
        w, r = self.m + self.n, self.r
        N = np.zeros((r * w, r * r))
        for a in range(r):
            for b in range(r):
                c = a * r + b
                N[b * w: b * w + self.m, c] += x[a * w: a * w + self.m]
                N[a * w + self.m: (a + 1) * w, c] -= x[b * w + self.m: (b + 1) * w]
        return N


class CP3:
    def __init__(self, inst: Instance, degenerate: bool):
        I, J, K = inst.shape
        self.I, self.J, self.K = I, J, K
        fam = "cp3_degenerate" if degenerate else "cp3"
        rt = 3 if degenerate else inst.teacher
        rng = _rng(fam, inst.instance, I, J, K, rt)
        if degenerate:
            self.xy = []
            for size in (I, J, K):
                self.xy.append((np.array([rng.normal01() for _ in range(size)]),
                                np.array([rng.normal01() for _ in range(size)])))
            (x1, y1), (x2, y2), (x3, y3) = self.xy
            T = (np.einsum("i,j,k->ijk", x1, x2, y3) + np.einsum("i,j,k->ijk", x1, y2, x3) +
                 np.einsum("i,j,k->ijk", y1, x2, x3))
        else:
            teacher = np.array([rng.normal01() for _ in range(rt * (I + J + K))])
            T = self._tensor(teacher, rt)
        self.T = _noise(rng, inst.noise, T.reshape(-1))
        self.R = 2 if degenerate else (inst.student or rt)

    def _parts(self, x, R):
        x = np.asarray(x, float).reshape(R, self.I + self.J + self.K)
        return x[:, :self.I], x[:, self.I:self.I + self.J], x[:, self.I + self.J:]

    def _tensor(self, x, R):
        A, B, C = self._parts(x, R)
        return np.einsum("ri,rj,rk->ijk", A, B, C)

    def residuals(self, x):
        return self._tensor(x, self.R).reshape(-1) - self.T

    def jacobian(self, x):
        A, B, C = self._parts(x, self.R)
        I, J, K, w = self.I, self.J, self.K, self.I + self.J + self.K
        Jm = np.zeros((I * J * K, self.R * w))
        for i in range(I):
            for j in range(J):
                for k in range(K):
                    row = (i * J + j) * K + k
                    for c in range(self.R):
                        Jm[row, c * w + i] = B[c, j] * C[c, k]
                        Jm[row, c * w + I + j] = A[c, i] * C[c, k]
                        Jm[row, c * w + I + J + k] = A[c, i] * B[c, j]
        return Jm

    def canon(self, x):
        x = np.asarray(x, float).copy()
        I, J, w = self.I, self.J, self.I + self.J + self.K
        for c in range(self.R):
            a, b, cc = x[c * w: c * w + I], x[c * w + I: c * w + I + J], x[c * w + I + J: (c + 1) * w]
            na, nb, nc = (float(np.sqrt(np.sum(v * v))) for v in (a, b, cc))
            if 0.0 in (na, nb, nc):
                x[c * w: (c + 1) * w] = 0.0
                continue
            if max(na, nb, nc) - min(na, nb, nc) > 64 * np.finfo(float).eps * max(na, nb, nc):
                nu = (na * nb * nc) ** (1.0 / 3.0)
                a *= nu / na
                b *= nu / nb
                cc *= nu / nc
            if a[np.argmax(np.abs(a))] < 0:
                a *= -1
                cc *= -1
            if b[np.argmax(np.abs(b))] < 0:
                b *= -1
                cc *= -1
        return _sort_blocks(x, 0, w, self.R, lambda v: -(float(np.linalg.norm(v[:I])) * float(np.linalg.norm(
            v[I:I + J])) * float(np.linalg.norm(v[I + J:]))))

    def gauge_basis(self, x):
        x = np.asarray(x, float)
        I, J, w = self.I, self.J, self.I + self.J + self.K
        N = np.zeros((self.R * w, 2 * self.R))
        for c in range(self.R):
            N[c * w: c * w + I, 2 * c] = N[c * w: c * w + I, 2 * c + 1] = x[c * w: c * w + I]
            N[c * w + I: c * w + I + J, 2 * c] = -x[c * w + I: c * w + I + J]
            N[c * w + I + J: (c + 1) * w, 2 * c + 1] = -x[c * w + I + J: (c + 1) * w]
        return N


class Rbfnet:
    def __init__(self, inst: Instance):
        d, w = inst.dim, inst.dim + 2
        rng = _rng("rbfnet", inst.instance, d, inst.teacher, inst.points)
        teacher = []
        for _ in range(inst.teacher):
            sign = -1.0 if rng.uniform01() < 0.5 else 1.0
            teacher.append(sign * (0.5 + rng.uniform01()))
            teacher += [-1.0 + 2.0 * rng.uniform01() for _ in range(d)]
            teacher.append(math.log(0.3 + 0.4 * rng.uniform01()))
        self.u = np.array([-1.0 + 2.0 * rng.uniform01() for _ in range(inst.points * d)]).reshape(inst.points, d)
        self.d, self.K, self.w = d, inst.student or inst.teacher, w
        self.y = _noise(rng, inst.noise, self._model(np.array(teacher), inst.teacher))

    def _model(self, x, K):
        p = np.asarray(x, float).reshape(K, self.w)
        q = np.sum((self.u[:, None, :] - p[None, :, 1:1 + self.d]) ** 2, axis=2) * np.exp(-2.0 * p[:, -1])
        return np.exp(-0.5 * q) @ p[:, 0]

    def residuals(self, x):
        return self._model(x, self.K) - self.y

    def jacobian(self, x):
        p = np.asarray(x, float).reshape(self.K, self.w)
        diff = self.u[:, None, :] - p[None, :, 1:1 + self.d]           # N x K x d
        inv = np.exp(-2.0 * p[:, -1])
        q = np.sum(diff ** 2, axis=2) * inv
        phi = np.exp(-0.5 * q)
        J = np.empty((len(self.u), self.K, self.w))
        J[:, :, 0] = phi
        J[:, :, 1:1 + self.d] = (p[:, 0] * phi * inv)[:, :, None] * diff
        J[:, :, -1] = p[:, 0] * phi * q
        return J.reshape(len(self.u), -1)

    def canon(self, x):
        return _sort_blocks(np.asarray(x, float), 0, self.w, self.K, lambda b: b[0])


def _tanh(z):
    t = 1.0 - 2.0 / (np.exp(2.0 * np.abs(z)) + 1.0)
    return np.where(z < 0.0, -t, t)


class MlpPlateau:
    def __init__(self, inst: Instance):
        d, units = inst.dim, inst.teacher
        self.d, self.H, self.w = d, units, d + 2
        rng = _rng("mlp_plateau", inst.instance, d, units, inst.points)
        teacher = np.array([rng.normal01() for _ in range(units * (d + 2) + 1)])
        self.u = np.array([rng.normal01() for _ in range(inst.points * d)]).reshape(inst.points, d)
        self.y = _noise(rng, inst.noise, self._model(teacher))

    def _model(self, x):
        x = np.asarray(x, float)
        p = x[:-1].reshape(self.H, self.w)
        h = _tanh(p[:, 1] + self.u @ p[:, 2:].T)                       # N x H
        return x[-1] + h @ p[:, 0]

    def residuals(self, x):
        return self._model(x) - self.y

    def jacobian(self, x):
        x = np.asarray(x, float)
        p = x[:-1].reshape(self.H, self.w)
        h = _tanh(p[:, 1] + self.u @ p[:, 2:].T)
        dz = p[:, 0] * (1.0 - h * h)
        J = np.empty((len(self.u), self.H, self.w))
        J[:, :, 0] = h
        J[:, :, 1] = dz
        J[:, :, 2:] = dz[:, :, None] * self.u[:, None, :]
        return np.hstack([J.reshape(len(self.u), -1), np.ones((len(self.u), 1))])

    def canon(self, x):
        x = np.asarray(x, float).copy()
        for j in range(self.H):
            blk = x[j * self.w: (j + 1) * self.w]
            if blk[1 + int(np.argmax(np.abs(blk[1:])))] < 0:
                blk *= -1
        return _sort_blocks(x, 0, self.w, self.H, lambda b: b[1])


class Chaboche:
    def __init__(self, inst: Instance):
        strain, self.res_blocks = [], []
        for a in (0.004, 0.008, 0.012):
            q = int(a / 0.0004 + 0.5)
            before = len(strain)
            for _ in range(2):
                strain += [0.0004 * i for i in range(1, q + 1)]
                strain += [0.0004 * i for i in range(q - 1, -q - 1, -1)]
                strain += [0.0004 * i for i in range(-q + 1, 1)]
            self.res_blocks.append(len(strain) - before)                    # CH-4 (task 19, §19.5)
        self.strain = strain
        teacher, scale = [200000.0, 200.0, 100.0, 10.0], 1.0
        for _ in range(inst.teacher):
            teacher += [60000.0 / scale, 600.0 / scale]
            scale *= 10.0
        y = self.stresses(teacher, inst.teacher)
        self.y = _noise(_rng("chaboche1d", inst.instance, inst.teacher, len(strain)), inst.noise, y)
        self.N = inst.student or inst.teacher

    def stresses(self, p, pairs):
        E, sy, Q, b = p[:4]
        eps_p = pp = 0.0
        X = [0.0] * pairs
        out = []
        for e in self.strain:
            trial = E * (e - eps_p)
            xi = trial - sum(X)
            if abs(xi) - Q * (1.0 - math.exp(-b * pp)) - sy <= 0.0:
                out.append(trial)
                continue
            s = -1.0 if xi < 0.0 else 1.0
            Xn, dp = list(X), 0.0
            for _ in range(100):
                g = abs(xi) - E * dp - Q * (1.0 - math.exp(-b * (pp + dp))) - sy
                dg = -E - Q * b * math.exp(-b * (pp + dp))
                for k in range(pairs):
                    C, gam = p[4 + 2 * k], p[5 + 2 * k]
                    den = 1.0 + gam * dp
                    g -= s * ((Xn[k] + C * dp * s) / den) - s * Xn[k]
                    dg -= (C - gam * s * Xn[k]) / (den * den)
                if g <= 0.0:
                    break
                step = -g / dg
                dp += step
                if step <= 4.0 * np.finfo(float).eps * dp:
                    break
            X = [(Xn[k] + p[4 + 2 * k] * dp * s) / (1.0 + p[5 + 2 * k] * dp) for k in range(pairs)]
            eps_p += dp * s
            pp += dp
            out.append(trial - E * dp * s)
        return np.array(out)

    def residuals(self, x):
        return self.stresses(list(np.asarray(x, float)), self.N) - self.y

    def canon(self, x):
        return _sort_blocks(np.asarray(x, float), 4, 2, self.N, lambda b: b[1])


class LJCluster:
    def __init__(self, inst: Instance):
        self.N = inst.dim
        self.L = 1.0 + (3.0 * self.N / (4.0 * math.pi)) ** (1.0 / 3.0)

    def value(self, x):
        r = np.asarray(x, float).reshape(self.N, 3)
        i, j = np.triu_indices(self.N, 1)
        s = 1.0 / np.sum((r[i] - r[j]) ** 2, axis=1)
        s3 = s ** 3
        return 4.0 * float(np.sum(s3 * s3 - s3))

    def gradient(self, x):
        r = np.asarray(x, float).reshape(self.N, 3)
        i, j = np.triu_indices(self.N, 1)
        diff = r[i] - r[j]
        s = 1.0 / np.sum(diff ** 2, axis=1)
        s3 = s ** 3
        t = (8.0 * s * s3 * (3.0 - 6.0 * s3))[:, None] * diff
        g = np.zeros_like(r)
        np.add.at(g, i, t)
        np.add.at(g, j, -t)
        return g.reshape(-1)

    def gauge_basis(self, x):
        r = np.asarray(x, float).reshape(self.N, 3)
        N = np.zeros((self.N, 3, 6))
        for c in range(3):
            N[:, c, c] = 1.0
        N[:, 1, 3], N[:, 2, 3] = -r[:, 2], r[:, 1]
        N[:, 0, 4], N[:, 2, 4] = r[:, 2], -r[:, 0]
        N[:, 0, 5], N[:, 1, 5] = -r[:, 1], r[:, 0]
        return N.reshape(3 * self.N, 6)


class MorseCluster(LJCluster):
    """morse.hpp: lj_cluster's box rule at the volume 1/sqrt 2 (MORSE-1), its gauge."""

    def __init__(self, inst: Instance):
        self.N, self.rho = inst.dim, inst.rho
        self.L = 1.0 + (3.0 * self.N * (1.0 / math.sqrt(2.0)) / (4.0 * math.pi)) ** (1.0 / 3.0)

    def value(self, x):
        r = np.asarray(x, float).reshape(self.N, 3)
        i, j = np.triu_indices(self.N, 1)
        e = np.exp(self.rho * (1.0 - np.sqrt(np.sum((r[i] - r[j]) ** 2, axis=1)))) - 1.0
        return float(np.sum(e * e - 1.0))

    def gradient(self, x):
        r = np.asarray(x, float).reshape(self.N, 3)
        i, j = np.triu_indices(self.N, 1)
        diff = r[i] - r[j]
        d = np.sqrt(np.sum(diff ** 2, axis=1))
        a = np.exp(self.rho * (1.0 - d))
        t = (-2.0 * self.rho * a * (a - 1.0) / d)[:, None] * diff
        g = np.zeros_like(r)
        np.add.at(g, i, t)
        np.add.at(g, j, -t)
        return g.reshape(-1)


class Circles:
    """circles.hpp: the overlaps of N unit circles, the pairs (i < j) then the walls."""

    def __init__(self, inst: Instance, circle: bool):
        self.N, self.size, self.circle = inst.dim, inst.container, circle

    def _walls(self, c):
        if self.circle:
            d = np.sqrt(np.sum(c ** 2, axis=1))
            return (d + 1.0 - self.size)[:, None], (c / np.where(d > 0.0, d, 1.0)[:, None])[:, None, :]
        t = np.stack([1.0 - c[:, 0], c[:, 0] + 1.0 - self.size, 1.0 - c[:, 1], c[:, 1] + 1.0 - self.size], axis=1)
        dt = np.zeros((self.N, 4, 2))
        dt[:, 0, 0], dt[:, 1, 0], dt[:, 2, 1], dt[:, 3, 1] = -1.0, 1.0, -1.0, 1.0
        return t, dt

    def residuals(self, x):
        c = np.asarray(x, float).reshape(self.N, 2)
        i, j = np.triu_indices(self.N, 1)
        d = np.sqrt(np.sum((c[i] - c[j]) ** 2, axis=1))
        t, _ = self._walls(c)
        return np.concatenate([np.maximum(2.0 - d, 0.0), np.maximum(t, 0.0).reshape(-1)])

    def jacobian(self, x):
        c = np.asarray(x, float).reshape(self.N, 2)
        i, j = np.triu_indices(self.N, 1)
        diff = c[i] - c[j]
        d = np.sqrt(np.sum(diff ** 2, axis=1))
        on = (d < 2.0) & (d > 0.0)                    # CIRC-4
        u = np.where(on[:, None], diff / np.where(d > 0.0, d, 1.0)[:, None], 0.0)
        P = len(i)
        t, dt = self._walls(c)
        W = t.shape[1]
        J = np.zeros((P + self.N * W, 2 * self.N))
        k = np.arange(P)
        J[k, 2 * i], J[k, 2 * i + 1], J[k, 2 * j], J[k, 2 * j + 1] = -u[:, 0], -u[:, 1], u[:, 0], u[:, 1]
        for a in range(self.N):
            for w in range(W):
                if t[a, w] > 0.0:
                    J[P + a * W + w, 2 * a:2 * a + 2] = dt[a, w]
        return J

    def gauge_basis(self, x):
        c = np.asarray(x, float).reshape(self.N, 2)
        return np.stack([-c[:, 1], c[:, 0]], axis=1).reshape(2 * self.N, 1)

    def canon(self, x):
        return _sort_blocks(np.asarray(x, float), 0, 2, self.N, lambda b: 0.0)


def numpy_family(family: str, inst: Instance):
    """The family in NumPy for an instance (the cross-check's)."""
    return {"sumexp": lambda: Sumexp(inst), "lowrank": lambda: Lowrank(inst),
            "cp3": lambda: CP3(inst, False), "cp3_degenerate": lambda: CP3(inst, True),
            "rbfnet": lambda: Rbfnet(inst), "mlp_plateau": lambda: MlpPlateau(inst),
            "chaboche1d": lambda: Chaboche(inst), "lj_cluster": lambda: LJCluster(inst),
            "morse_cluster": lambda: MorseCluster(inst), "circles_in_circle": lambda: Circles(inst, True),
            "circles_in_square": lambda: Circles(inst, False)}[family]()
