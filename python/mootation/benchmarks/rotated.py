# SPDX-License-Identifier: Apache-2.0
"""Rotated DTLZ and WFG: the distance variables rotated about their optimum.

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

The controls, for "with and without the rotation": the same construction with
Q = I — r0DTLZ1, r0DTLZ3 (y = 1/2 + (z - c), a shift that is not cyclic) and
c0DTLZ2 (y = 1/2 + Lambda (z - c)). shiftDTLZ is not one: its shift is cyclic
and far from the optimum its function is another. For WFG the control is WFG
itself: at Q = I the map is the identity.

WHAT A ROTATION CANNOT DO. DTLZ2's and DTLZ4's g = |y - 1/2|^2 is isotropic,
and |Q (z - c)| = |z - c|: rDTLZ2 and rDTLZ4 are the same functions as their
Q = I construction (to 3e-15 on 10^4 points), checks of invariance and not
rotated problems. Rotated in earnest: rDTLZ1, rDTLZ3, rcDTLZ2 and rWFG4-7. Not
rotated at all: DTLZ6 (x^0.1 has no value below 0) and DTLZ7 (its g is linear
and runs to minus infinity off [0, 1]), nor WFG8 and WFG9, whose optimal
distance values depend on other variables (b_param) and are not the constant
0.35, nor WFG1-3.

Q is a Haar rotation: QR of a standard normal matrix with the signs fixed, as
bbob._rotation, from SeedSequence([20261009, crc32(base family), M, k, 1]) —
one instance, the same for every run. numpy does not promise its normal stream
bit for bit across versions; the tests hold a hash of every Q.

Front, frame and Pareto-set positions are the base problem's; the distance
variables of the Pareto set are c (rDTLZ, r0, c0) and 0.35 * 2i (rWFG), and
the shift is not cyclic (cyclic_vars = ()).

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

import zlib
from typing import List

import numpy as np

from .dtlz import DTLZ_FUNCS
from .dtlz_variants import shift_centres
from .wfg import WFG_FUNCS

SIZES = (3, 5)
KAPPA = 1.0e3
L_WFG = 20
# name stem -> (base family, Q a rotation, Lambda the conditioning)
SPECS = {
    "rDTLZ1": ("DTLZ1", True, False), "rDTLZ2": ("DTLZ2", True, False),
    "rDTLZ3": ("DTLZ3", True, False), "rDTLZ4": ("DTLZ4", True, False),
    "rcDTLZ2": ("DTLZ2", True, True),
    "r0DTLZ1": ("DTLZ1", False, False), "r0DTLZ3": ("DTLZ3", False, False),
    "c0DTLZ2": ("DTLZ2", False, True),
    "rWFG4": ("WFG4", True, False), "rWFG5": ("WFG5", True, False),
    "rWFG6": ("WFG6", True, False), "rWFG7": ("WFG7", True, False),
}


def rotation(family: str, M: int, k: int, instance: int = 1) -> np.ndarray:
    """The k x k Haar rotation of a family at M objectives (see above)."""
    seed = np.random.SeedSequence([20261009, zlib.crc32(family.encode()), M, k, instance])
    A = np.random.default_rng(seed).standard_normal((k, k))
    Q, R = np.linalg.qr(A)
    return Q * np.sign(np.where(np.diag(R) == 0.0, 1.0, np.diag(R)))


def conditioning(k: int, kappa: float = KAPPA) -> np.ndarray:
    """The diagonal of Lambda: kappa^((i - 1)/(2 (k - 1))), i = 1..k."""
    return kappa ** (np.arange(k) / (2.0 * (k - 1)))


class Rotated:
    """One rotated problem: map(x) is the point its base problem is evaluated at."""

    def __init__(self, stem: str, M: int, n_vars: int):
        family, rotate, condition = SPECS[stem]
        self.family, self.M, self.n_vars = family, M, n_vars
        self.wfg = family.startswith("WFG")
        self.first = n_vars - L_WFG if self.wfg else M - 1        # the first distance variable
        k = n_vars - self.first
        self.A = rotation(family, M, k) if rotate else np.eye(k)
        if condition:
            self.A = conditioning(k)[:, None] * self.A           # Lambda Q, not Q Lambda
        if self.wfg:
            self.scale = 2.0 * np.arange(self.first + 1, n_vars + 1)
            self.optimum = 0.35 * self.scale
        else:
            self.optimum = shift_centres(k)

    def map(self, x) -> np.ndarray:
        y = np.array(x, dtype=float)
        if self.wfg:
            u = y[self.first:] / self.scale
            v = 0.35 + self.A @ (u - 0.35)
            y[self.first:] = self.scale * np.clip(v, 0.0, 1.0)
        else:
            y[self.first:] = 0.5 + self.A @ (y[self.first:] - self.optimum)
        return y

    def evaluate(self, x) -> List[float]:
        f = WFG_FUNCS if self.wfg else DTLZ_FUNCS
        return list(f[self.family](list(self.map(x)), self.M))

    def clipped(self, x) -> int:
        """How many distance variables the map clips at x (rWFG; 0 for rDTLZ)."""
        if not self.wfg:
            return 0
        v = 0.35 + self.A @ (np.asarray(x, float)[self.first:] / self.scale - 0.35)
        return int(np.sum((v < 0.0) | (v > 1.0)))


def name(stem: str, M: int) -> str:
    return f"{stem}_{M}D"
