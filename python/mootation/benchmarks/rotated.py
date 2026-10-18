# SPDX-License-Identifier: Apache-2.0
"""Rotated ZDT, DTLZ, WFG, MaF, ZCAT and Polygon: the distance variables
rotated about their optimum.

A control written for this library, not a published suite (task 6 of
2026-10-09, item 1, with its supplement 2). ZDT, DTLZ and WFG keep every
distance variable on its own axis, which is a gift to the operators that work
coordinate by coordinate (SBX, polynomial mutation). These problems keep the
position/distance structure and the exact front, and mix the distance
variables, so that what an algorithm owes to axis-wise variation shows against
rotation-invariant steps (DE with CR = 1, CMA-ES).

THE CONSTRUCTION. The position variables are left alone; only the block of
distance variables is transformed, about its optimum:

  rDTLZ1-4  z the k distance variables, c = dtlz_variants.shift_centres(k),
            y = 1/2 + Q (z - c), f = DTLZ(positions, y). At z = c, y = 1/2 and
            g = 0: DTLZ's front, the optimum off the centre of the box. y
            leaves [0, 1] — at a fifth of the coordinates of uniform points,
            from -1.1 to 2.0 — and is not clipped: the g of DTLZ1-4 is defined
            on the whole line with its only minimum at 1/2. DTLZ1's and
            DTLZ3's cosines then have more periods inside the box, i.e. more
            local fronts, a property of the construction.
  rcDTLZ2   y = 1/2 + Lambda Q (z - c), Lambda = diag(kappa^((i-1)/(2(k-1)))),
            kappa = 10^3: the Hessian of g has condition number kappa. The
            scaling comes AFTER the rotation: with Q Lambda the norm of
            Q Lambda v is that of Lambda v, and the rotation would vanish again.
  rWFG4-7   u = z_i / 2i, the l = 20 normalised distance variables,
            v = 0.35 + Q (u - 0.35), clipped to [0, 1], z_i = 2i v_i into WFG:
            the rotation acts on the normalised raw variables, before any of
            WFG's transformations, which are defined on [0, 1] only (hence the
            clipping). Clipping gives 0 or 1, never the optimum 0.35, so it
            makes no new optimum, but it makes plateaus. Within distance 0.35
            of the optimum (in u) nothing is clipped: |Q w|_inf <= |w|_2.
            Measured on 10^5 uniform points of the box (2026-10-10), rWFG4-7
            at three and five objectives: 97-99 % of the points have a clipped
            variable, 2.8-3.7 of the 20 on average, 10-29 % five or more.
  rZDT1-3, rZDT6 (task 7 of 2026-10-10, §3)
            z = x_2..x_n, the k = n - 1 distance variables, c =
            shift_centres(k), m_i = max(c_i, 1 - c_i), and ZDT evaluated at
            x' = |Q (z - c)| / m, component by component. g is a sum of the
            x' (ZDT6: its fourth root), a sum of absolute values, which is not
            isotropic: the rotation changes the function. At Q = I, x' is the
            fold of shiftZDT (shifted.py), which is the control. x' leaves
            [0, 1] and is not clipped: g is defined for any x' >= 0, with its
            only minimum at x' = 0.
  rZDT4     c = 10 (shift_centres(9) - 1/2), ZDT4 evaluated at x' = Q (z - c):
            the Rastrigin term of ZDT4 rotated. At Q = I it is shiftZDT4. x'
            leaves [-5, 5], unclipped, as ZDT4's g allows.

THE OTHER FAMILIES (task 7, §7). One rule: only the block of distance
variables turns, about its own optimum; where g is isotropic a rotation would
change nothing, and the problem is registered conditioned only (Lambda Q,
kappa = 10^3), with its Q = I control.

  rWFG1-3   as rWFG4-7: the normalised distance variables about 0.35, clipped
            to [0, 1]. Measured as for rWFG4-7 (10^5 uniform points,
            2026-10-10): 97-99 % of the points have a clipped variable, 2.7-3.8
            of the 20 on average, 9-32 % five or more.
  rWFG8, rWFG9
            The optimum of a distance variable is not 0.35. b_param (Huband et
            al. 2006, Table XIV) raises y_i to e(u) = B + (C - B)(A - (1 - 2u)
            |floor(1/2 - u) + A|), A = 0.98/49.98, B = 0.02, C = 50, before
            s_linear (WFG8) or s_multi (WFG9) sends it to 0 at 0.35, so
            y*_i = 0.35^(1/e(u)): u the mean of y_1..y_{i-1} in WFG8 (the
            positions and the distance variables before i), of y_{i+1}..y_n in
            WFG9 (the distance variables after i; y*_n = 0.35). The centre of
            the rotation is that optimum for the current positions, computed
            in order (WFG8 from the first distance variable on, WFG9 from the
            last back): v = u*(pos) + Q (u - u*(pos)), clipped. For fixed
            positions the map is affine and invertible up to the clipping.
            WFG9's u* depends on no position: a constant vector, from 0.35 at
            the last variable down to 0.001 at the first; WFG8's lies between
            0 and 0.98, median 0.33, at random positions.
            pymoo 0.6.2 builds the same Pareto set (_positional_to_optimal,
            Apache-2.0): wfg8_optimum and wfg9_optimum checked against it
            at four vectors, reference in
            tests/data/pymoo-0.6.2_wfg-optimum.json; the tests check the set
            numerically too: its images lie on WFG8's and WFG9's front, the
            sphere sum (f_m / 2m)^2 = 1. With the optimum near 0 for many of
            the variables, the rotation sends about half of them below 0: every
            uniform point has clipped variables, 9-10.5 of the 20 on average.
            The difference from the optimum is taken in x's units, so that at
            x = 2i u* the map is the identity bit for bit — WFG1's b_poly(0.02)
            turns 1e-17 off 0.35 into 0.46 of the way to the worst value — and
            rWFG1-3 do the same; rWFG4-7 keep task 6's arithmetic.
  rcDTLZ5   y = 1/2 + Lambda Q (z - c), control c0DTLZ5: DTLZ5's g is DTLZ2's,
            isotropic. theta reads g; at the optimum g = 0 and the front is
            DTLZ5's.
  MaF       (Cheng et al. 2017, as maf.py and registry.py evaluate them) the
            distance variables of MaF1-6 have their optimum at 1/2, and the
            construction is rDTLZ's, y = 1/2 + Q (z - c) or 1/2 + Lambda Q
            (z - c), c = shift_centres(k), by the form of g:
              MaF1   sum (y - 1/2)^2, isotropic        -> rcMaF1, control c0MaF1
              MaF2   per objective, a sum over its own chunk of the distance
                     variables of ((y - 1/2)/2)^2: each isotropic, the vector
                     not — a rotation mixes the chunks -> rMaF2, control r0MaF2
              MaF3   100 (k + sum ((y - 1/2)^2 - cos 20 pi (y - 1/2)))
                                                       -> rMaF3, control r0MaF3
              MaF4   MaF3's g                          -> rMaF4, control r0MaF4
              MaF5   sum (y - 1/2)^2                   -> rcMaF5, control c0MaF5
              MaF6   sum (y - 1/2)^2, which theta reads -> rcMaF6, control c0MaF6
              MaF7   DTLZ7's, linear, optimum on the bound: not rotated
              MaF8, MaF9   two variables, no distance block: none
              MaF10-12     WFG1, WFG2, WFG9: see rWFG1, rWFG2, rWFG9
              MaF13  its optimum moves with x_1 and x_2: none
            The task put MaF2 with the quadratic, conditioned ones; read off
            the code its g is a vector of chunk sums, which a rotation alone
            changes.
  rZCAT1-20 (Zapotecas-Martinez et al. 2023, zcat.py) The Pareto set is
            y_II = g(y_I), a function of the positions, as in WFG8. On the
            normalised variables y = x/i + 1/2, y'_II = g(y_I) + Q (y_II -
            g(y_I)), clipped to [0, 1], and x' = (y' - 1/2) i. ZCAT19 and
            ZCAT20 switch m between M - 1 and 1 by y_1, so each has two
            matrices, one per size of the distance block. The control is ZCAT
            itself. Clipped: 98-100 % of uniform points, 3.4-18.8 of the 28
            (M = 3) or 46-49 (M = 5) distance variables on average (10^4
            points, 2026-10-10).
  rcPolygon (polygon.py) g = sum x_i^2 over x_3..x_n: isotropic, and its
            optimum, 0, is the centre of [-1, 1]. The optimum moves first,
            c = 2 shift_centres(k) - 1 (not cyclic, 0.3-0.7 from the centre),
            then y = Lambda Q (z - c); control c0Polygon, y = Lambda (z - c).

The controls, for "with and without the rotation": the same construction with
Q = I — r0DTLZ1, r0DTLZ3 (y = 1/2 + (z - c), a shift that is not cyclic) and
c0DTLZ2 (y = 1/2 + Lambda (z - c)). shiftDTLZ is not one: its shift is cyclic
and far from the optimum its function is another. For WFG the control is WFG
itself: at Q = I the map is the identity; for rZDT it is shiftZDT.

SIZES. Everything at M = 3 and 5, rZDT at M = 2; rcDTLZ2 and c0DTLZ2 at
M = 6 and 10 too, for a trend over the number of objectives (task 7, §3).

INSTANCES (task 7, §3 and §7). rZDT1-4, rZDT6, rDTLZ1, rDTLZ3, rcDTLZ2,
rWFG4-7 and every rotated problem of §7 (not the controls) come in three
instances of Q, <name>_i1, _i2, _i3 (rDTLZ1_3D_i2, rZDT4_i3), at
every size they have. Instance k draws Q from SeedSequence([20261009,
crc32(base family), M, k_dist, k]); instance 1 is the problem without the
suffix (the same matrix, whose hash the tests have held since task 6), under
the name with it, so that a campaign over the instances lists all three.

WHAT A ROTATION CANNOT DO. DTLZ2's and DTLZ4's g = |y - 1/2|^2 is isotropic,
and |Q (z - c)| = |z - c|: rDTLZ2 and rDTLZ4 are the same functions as their
Q = I construction (to 3e-15 on 10^4 points), checks of invariance and not
rotated problems. Not rotated at all: DTLZ6 (x^0.1 has no value below 0) and
DTLZ7 (its g is linear and runs to minus infinity off [0, 1]), nor MaF7.

Q is a Haar rotation: QR of a standard normal matrix with the signs fixed, as
bbob._rotation, from SeedSequence([20261009, crc32(base family), M, k,
instance]) — the same for every run. numpy does not promise its normal stream
bit for bit across versions; the tests hold a hash of every Q.

Front, frame and Pareto-set positions are the base problem's; the distance
variables of the Pareto set are c (rZDT, rDTLZ, rMaF, rcPolygon, r0, c0),
u*(pos) * 2i (rWFG) and g(y_I) (rZCAT), and the shift is not cyclic
(cyclic_vars = ()).

THE LITERATURE, and how this differs (task 6, items 1 and supplement 2):
  * Deb, Sinha, Kukkonen, "Multi-objective test problems, linkages, and
    evolutionary methodologies", GECCO 2006, doi:10.1145/1143997.1144179:
    L1-ZDT4 transforms the distance block alone, y = diag(A, D) x with x_1
    unchanged, so that the front stays the minimum of g — the nearest
    predecessor. Theirs is a random non-orthogonal D, without a shift, on one
    problem; here Q is orthogonal (or Lambda Q, conditioning stated), about a
    shifted optimum, on DTLZ and WFG at any M.
  * Iorio, Li, "Rotated test problems for assessing the performance of
    multi-objective optimization algorithms", GECCO 2006, 683-690,
    doi:10.1145/1143997.1144118: R1-R4, ZDT-like, rotate the WHOLE decision
    vector about the origin by a uniform random rotation (Gram-Schmidt on
    normal vectors, Algorithm 1), a new one in every run; where f_1 leaves its
    range both objectives get large values. They warn that a rotation can
    hand a function arguments it is not defined for (a square root of a
    negative) — DTLZ6 and DTLZ7 above.
  * Huang, Qin, Deb, Zitzler, Suganthan, Liang, Preuss, Huband, "Problem
    definitions for performance assessment of multi-objective optimization
    algorithms", CEC 2007 technical report: R_ZDT4 and R_DTLZ2 (R_DTLZ2 at
    five objectives is UF11 of CEC 2009) rotate all variables, z = M x with
    an orthogonal M; a z outside the box is reflected, z' = -lambda z below 0
    and lambda z above 1, and the objectives multiplied by S = 2/(1 +
    exp(-psum)) of the overshoot. Its shifted S_ZDT and S_DTLZ shift every
    variable and extend the box with a penalty.
  * MOEA Framework's RotatedProblem (rotates all variables about the centre
    of the box, widened by sqrt(2), clipped back with a constraint), and the
    "sphered rotation" of Vermetten and Rook, arXiv:2604.08173 (a bijection of
    the whole box).
Here the position block stays on its axes, so the front and the Pareto set
stay exactly known and the box cuts nothing out of them, with no penalty.
"""

from __future__ import annotations

import math
import zlib
from typing import List

import numpy as np

from . import zcat as _zcat
from ._rows import rows_matvec
from .dtlz import DTLZ_FUNCS, DTLZ_ROWS
from .dtlz_variants import shift_centres
from .wfg import WFG_FUNCS

SIZES = (3, 5)
MORE_SIZES = {"rcDTLZ2": (6, 10), "c0DTLZ2": (6, 10)}
INSTANCED = ("rZDT1", "rZDT2", "rZDT3", "rZDT4", "rZDT6", "rDTLZ1", "rDTLZ3", "rcDTLZ2",
             "rWFG4", "rWFG5", "rWFG6", "rWFG7",
             "rWFG1", "rWFG2", "rWFG3", "rWFG8", "rWFG9", "rcDTLZ5", "rcMaF1", "rMaF2",
             "rMaF3", "rMaF4", "rcMaF5", "rcMaF6", "rcPolygon") + tuple(
                 f"rZCAT{i}" for i in range(1, 21))
INSTANCES = (1, 2, 3)
KAPPA = 1.0e3
L_WFG = 20
# name stem -> (base family, Q a rotation, Lambda the conditioning)
SPECS = {
    "rZDT1": ("ZDT1", True, False), "rZDT2": ("ZDT2", True, False),
    "rZDT3": ("ZDT3", True, False), "rZDT4": ("ZDT4", True, False),
    "rZDT6": ("ZDT6", True, False),
    "rDTLZ1": ("DTLZ1", True, False), "rDTLZ2": ("DTLZ2", True, False),
    "rDTLZ3": ("DTLZ3", True, False), "rDTLZ4": ("DTLZ4", True, False),
    "rcDTLZ2": ("DTLZ2", True, True),
    "r0DTLZ1": ("DTLZ1", False, False), "r0DTLZ3": ("DTLZ3", False, False),
    "c0DTLZ2": ("DTLZ2", False, True),
    "rWFG4": ("WFG4", True, False), "rWFG5": ("WFG5", True, False),
    "rWFG6": ("WFG6", True, False), "rWFG7": ("WFG7", True, False),
    # task 7, §7
    "rWFG1": ("WFG1", True, False), "rWFG2": ("WFG2", True, False),
    "rWFG3": ("WFG3", True, False), "rWFG8": ("WFG8", True, False),
    "rWFG9": ("WFG9", True, False),
    "rcDTLZ5": ("DTLZ5", True, True), "c0DTLZ5": ("DTLZ5", False, True),
    "rcMaF1": ("MaF1", True, True), "c0MaF1": ("MaF1", False, True),
    "rMaF2": ("MaF2", True, False), "r0MaF2": ("MaF2", False, False),
    "rMaF3": ("MaF3", True, False), "r0MaF3": ("MaF3", False, False),
    "rMaF4": ("MaF4", True, False), "r0MaF4": ("MaF4", False, False),
    "rcMaF5": ("MaF5", True, True), "c0MaF5": ("MaF5", False, True),
    "rcMaF6": ("MaF6", True, True), "c0MaF6": ("MaF6", False, True),
    "rcPolygon": ("Polygon", True, True), "c0Polygon": ("Polygon", False, True),
}
SPECS.update({f"rZCAT{i}": (f"ZCAT{i}", True, False) for i in range(1, 21)})

# b_param of Huband et al. 2006, Table XIV, as WFG7-9 call it
_BP_A, _BP_B, _BP_C = 0.98 / 49.98, 0.02, 50.0


def rotation(family: str, M: int, k: int, instance: int = 1) -> np.ndarray:
    """The k x k Haar rotation of a family at M objectives (see above)."""
    seed = np.random.SeedSequence([20261009, zlib.crc32(family.encode()), M, k, instance])
    A = np.random.default_rng(seed).standard_normal((k, k))
    Q, R = np.linalg.qr(A)
    return Q * np.sign(np.where(np.diag(R) == 0.0, 1.0, np.diag(R)))


def conditioning(k: int, kappa: float = KAPPA) -> np.ndarray:
    """The diagonal of Lambda: kappa^((i - 1)/(2 (k - 1))), i = 1..k."""
    return kappa ** (np.arange(k) / (2.0 * (k - 1)))


def _b_param_root(u: float) -> float:
    """The y with b_param(y, u) = 0.35: 0.35^(1/e(u)) (rWFG8, rWFG9 above)."""
    v = _BP_A - (1.0 - 2.0 * u) * abs(math.floor(0.5 - u) + _BP_A)
    return 0.35 ** (1.0 / (_BP_B + (_BP_C - _BP_B) * v))


def wfg8_optimum(positions, l: int) -> np.ndarray:
    """WFG8's optimal normalised distance variables for normalised positions."""
    y = [float(v) for v in positions]
    for _ in range(l):
        y.append(_b_param_root(sum(y) / len(y)))
    return np.array(y[len(positions):])


def wfg9_optimum(l: int) -> np.ndarray:
    """WFG9's optimal normalised distance variables, the last first."""
    y = [0.35]
    for _ in range(l - 1):
        y.insert(0, _b_param_root(sum(y) / len(y)))
    return np.array(y)


def _kind(family: str) -> str:
    for k in ("ZDT", "DTLZ", "WFG", "MaF", "ZCAT", "Polygon"):
        if family.startswith(k):
            return k.lower()
    raise ValueError(family)


def sizes(stem: str) -> tuple:
    """The objective counts a stem is registered at."""
    if stem.startswith("rZDT"):
        return (2,)
    return SIZES + MORE_SIZES.get(stem, ())


class Rotated:
    """One rotated problem: map(x) is the point its base problem is evaluated at."""

    def __init__(self, stem: str, M: int, n_vars: int, instance: int = 1):
        family, rotate, condition = SPECS[stem]
        self.family, self.M, self.n_vars, self.instance = family, M, n_vars, instance
        self.rotate, self.condition = rotate, condition
        self.kind = _kind(family)
        self.wfg, self.zdt = self.kind == "wfg", self.kind == "zdt"
        # the first distance variable
        self.first = {"wfg": n_vars - L_WFG, "zdt": 1, "polygon": 2}.get(self.kind, M - 1)
        if family in _zcat.DEGENERATE:
            self.first = 1
        self._matrices = {}
        k = n_vars - self.first
        self.A = self.matrix(k)
        self.base = family if self.zdt else f"{family}_{M}D"
        if self.wfg:
            self.scale_all = 2.0 * np.arange(1, n_vars + 1)
            self.scale = self.scale_all[self.first:]
            self.centre_u = wfg9_optimum(k) if family == "WFG9" else np.full(k, 0.35)
            self.optimum = None if family == "WFG8" else self.scale * self.centre_u
        elif self.kind == "zcat":
            self.optimum = None                                   # g(y_I), as WFG8
        elif family == "ZDT4":
            self.optimum = 10.0 * (shift_centres(k) - 0.5)
        elif self.kind == "polygon":
            self.optimum = 2.0 * shift_centres(k) - 1.0
        else:
            self.optimum = shift_centres(k)
            self.spread = np.maximum(self.optimum, 1.0 - self.optimum)    # m (rZDT)

    def matrix(self, k: int) -> np.ndarray:
        """Q, Lambda Q or I for a distance block of k variables."""
        if k not in self._matrices:
            A = rotation(self.family, self.M, k, self.instance) if self.rotate else np.eye(k)
            if self.condition:
                A = conditioning(k)[:, None] * A                 # Lambda Q, not Q Lambda
            self._matrices[k] = A
        return self._matrices[k]

    def matrices(self) -> list:
        """Every matrix the problem uses: ZCAT19 and ZCAT20 one per block size."""
        if self.family in _zcat.REGION_M:
            return [self.A, self.matrix(self.n_vars - 1)]
        return [self.A]

    def _wfg_v(self, x) -> np.ndarray:
        """rWFG's distance block, normalised, before the clipping. rWFG4-7 as in
        task 6, bit for bit; the others take the difference from the optimum in
        x's units first, so that at the optimum, x = 2i u*, the map is the
        identity bit for bit: WFG1's b_poly(0.02) turns 1e-17 off 0.35 into
        0.46 of the way to the worst value."""
        x = np.asarray(x, float)
        if self.family in ("WFG4", "WFG5", "WFG6", "WFG7"):
            return 0.35 + self.A @ (x[self.first:] / self.scale - 0.35)
        if self.family == "WFG8":
            c = wfg8_optimum(x[:self.first] / self.scale_all[:self.first], len(self.scale))
        else:
            c = self.centre_u
        return c + self.A @ ((x[self.first:] - self.scale * c) / self.scale)

    def _zcat_v(self, x):
        """rZCAT's normalised point, its m and its distance block before the clipping."""
        x = np.asarray(x, float)
        idx = np.arange(1, len(x) + 1, dtype=float)
        u = np.clip(x / idx + 0.5, 0.0, 1.0)
        m = _zcat.position_count(self.family, u, self.M)
        g = _zcat.topology(_zcat.G_OF[self.family], u[:m], m, len(x))
        return idx, m, g + self.matrix(len(x) - m) @ (u[m:] - g)

    def map(self, x) -> np.ndarray:
        y = np.array(x, dtype=float)
        if self.wfg:
            y[self.first:] = self.scale * np.clip(self._wfg_v(x), 0.0, 1.0)
        elif self.kind == "zcat":
            idx, m, v = self._zcat_v(x)
            y[m:] = (np.clip(v, 0.0, 1.0) - 0.5) * idx[m:]
        elif self.family == "ZDT4":
            y[self.first:] = self.A @ (y[self.first:] - self.optimum)
        elif self.zdt:
            y[self.first:] = np.abs(self.A @ (y[self.first:] - self.optimum)) / self.spread
        elif self.kind == "polygon":
            y[self.first:] = self.A @ (y[self.first:] - self.optimum)
        else:
            y[self.first:] = 0.5 + self.A @ (y[self.first:] - self.optimum)
        return y

    def map_rows(self, X) -> np.ndarray:
        """map of every row of X: one product with the matrix per row (gemv, as
        map does), the rest elementwise; rWFG's and rZCAT's row by row."""
        Y = np.array(X, dtype=float).reshape(-1, self.n_vars)
        if self.wfg or self.kind == "zcat":
            for i in range(len(Y)):
                Y[i] = self.map(Y[i])
            return Y
        D = rows_matvec(self.A, Y[:, self.first:] - self.optimum)
        if self.family == "ZDT4" or self.kind == "polygon":
            Y[:, self.first:] = D
        elif self.zdt:
            Y[:, self.first:] = np.abs(D) / self.spread
        else:
            Y[:, self.first:] = 0.5 + D
        return Y

    def evaluate_batch(self, X) -> np.ndarray:
        """evaluate of every row of X (task 10 of 2026-10-18)."""
        Y = self.map_rows(X)
        if self.kind == "dtlz":
            return DTLZ_ROWS[self.family](Y, self.M)
        if self.wfg:
            out = np.empty((len(Y), self.M))
            for i, y in enumerate(Y):
                out[i] = WFG_FUNCS[self.family](y.tolist(), self.M)
            return out
        from .registry import PROBLEMS
        return PROBLEMS[self.base].evaluate_batch(Y)

    def evaluate(self, x) -> List[float]:
        if self.wfg:
            return list(WFG_FUNCS[self.family](list(self.map(x)), self.M))
        if self.kind == "dtlz":
            return list(DTLZ_FUNCS[self.family](list(self.map(x)), self.M))
        from .registry import PROBLEMS     # ZDT's, MaF's, ZCAT's and Polygon's are the registry's
        return list(PROBLEMS[self.base].evaluate(list(self.map(x))))

    def distance_optimum(self, x) -> np.ndarray:
        """The distance block of the Pareto set at the positions of x, in x's units."""
        if self.family == "WFG8":
            pos = np.asarray(x, float)[:self.first] / self.scale_all[:self.first]
            return self.scale * wfg8_optimum(pos, len(self.scale))
        if self.kind == "zcat":
            idx, m, _ = self._zcat_v(x)
            u = np.clip(np.asarray(x, float) / idx + 0.5, 0.0, 1.0)
            return (_zcat.topology(_zcat.G_OF[self.family], u[:m], m, len(idx)) - 0.5) * idx[m:]
        return self.optimum

    def clipped(self, x) -> int:
        """How many distance variables the map clips at x (rWFG, rZCAT; 0 elsewhere)."""
        if self.wfg:
            v = self._wfg_v(x)
        elif self.kind == "zcat":
            v = self._zcat_v(x)[2]
        else:
            return 0
        return int(np.sum((v < 0.0) | (v > 1.0)))


def name(stem: str, M: int, instance: int = 0) -> str:
    """rDTLZ1_3D, rZDT4; with an instance rDTLZ1_3D_i2, rZDT4_i2."""
    base = stem if stem.startswith("rZDT") else f"{stem}_{M}D"
    return f"{base}_i{instance}" if instance else base
