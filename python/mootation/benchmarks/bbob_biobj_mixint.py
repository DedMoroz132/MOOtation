# SPDX-License-Identifier: Apache-2.0
# ============================================================================
# bbob-biobj-mixint: bbob-biobj with four fifths of its variables integer
# (task 7 of 2026-10-10, supplement 3, §1.4).
# T. Tusar, D. Brockhoff, N. Hansen. "Mixed-Integer Benchmark Problems for
# Single- and Bi-Objective Optimization." GECCO 2019, 718-726,
# doi:10.1145/3321707.3321868.            (source: bbob-mixint_tusar2019, §3)
#
# THE DISCRETIZATION (§3.1.1-3.1.2). The n variables fall into five groups of
# n/5 consecutive ones, of arities l = 2, 4, 8, 16 and infinity (continuous).
# An integer variable i of arity l takes 0 .. l-1, and the value j - 1 stands
# for z_j: with the l equidistant y_j = -4 + 8 j / (l + 1), j = 1..l, y* the
# one nearest x_opt_i and d_i = y* - x_opt_i, z_j = y_j - d_i — so that one
# z_j is x_opt_i and the optimal value is kept. (The paper writes zeta, the
# map from a continuous value to its integer; an evaluator that receives the
# integer needs the other direction, j -> z_{j+1}.)
#
# READINGS (declared):
#   MIXINT-1. A bi-objective problem combines two bbob-mixint functions
#     (§3.2), each discretized about its own optimum: the integer j stands for
#     z^alpha_{j+1} in the first objective and z^beta_{j+1} in the second.
#     The paper says no more; COCO's code may align otherwise.
#   MIXINT-2. Table 1's scaling factors are left out. A factor multiplies one
#     objective, which changes neither dominance nor an indicator measured in
#     the frame (ideal, nadir) of the same objectives.
#   MIXINT-3. The pairs, their order and their instances are bbob_biobj.py's
#     — ours, not COCO's (its BIOBJ-2) — so nothing here can be checked
#     against COCO's archives; COCO's suite also takes the 92 pairs of
#     bbob-biobj-ext, this one the 55 of bbob-biobj, at n = 5 and 10.
#
# The ideal is (f_opt^alpha, f_opt^beta), which the alignment keeps; the nadir
# is (f^alpha(x^beta*), f^beta(x^alpha*)) at each objective's mixed-integer
# optimum (bbob_biobj.py's BIOBJ-1), x^alpha* the indices of its y* and its
# x_opt on the continuous part. No front: as for bbob-biobj, none is known.
# ============================================================================
from __future__ import annotations

from typing import List, Tuple

import numpy as np

from . import bbob_biobj

ARITIES: Tuple[int, ...] = (2, 4, 8, 16)


def arities(D: int) -> List[int]:
    """The arity of each integer variable: n/5 of each of 2, 4, 8, 16."""
    if D % 5:
        raise ValueError(f"bbob-biobj-mixint needs n a multiple of 5, not {D}")
    return [a for a in ARITIES for _ in range(D // 5)]


def levels(x_opt_i: float, l: int) -> np.ndarray:
    """z_1 .. z_l of a variable of arity l whose optimum is x_opt_i (§3.1.1)."""
    y = -4.0 + 8.0 * np.arange(1, l + 1) / (l + 1)
    d = y[int(np.argmin(np.abs(y - x_opt_i)))] - x_opt_i
    return y - d


class MixedInteger:
    """F`fnum` of bbob-biobj in D variables, the first 4 D / 5 integer."""

    def __init__(self, fnum: int, D: int):
        p = bbob_biobj.problem(fnum, D, 1)
        self.fnum, self.D = fnum, D
        self.alpha, self.beta = p.alpha, p.beta
        self.arities = arities(D)
        self.k = len(self.arities)
        self.z_alpha = [levels(p.alpha.x_opt[i], l) for i, l in enumerate(self.arities)]
        self.z_beta = [levels(p.beta.x_opt[i], l) for i, l in enumerate(self.arities)]
        best_a, best_b = self.optimum(self.alpha, self.z_alpha), self.optimum(self.beta, self.z_beta)
        self.ideal = (self.alpha.f_opt, self.beta.f_opt)
        self.nadir = tuple(self(best_b)[:1] + self(best_a)[1:])     # BIOBJ-1

    def optimum(self, f, z) -> list:
        """The mixed-integer point where f takes f_opt: the index of the z
        that is x_opt, and x_opt on the continuous part."""
        ints = [int(np.argmin(np.abs(zi - f.x_opt[i]))) for i, zi in enumerate(z)]
        return ints + list(f.x_opt[self.k:])

    def continuous(self, X, z) -> np.ndarray:
        """The rows of X with every integer value j replaced by z_{j+1} (a real
        there is rounded, half to even, and kept in 0 .. l-1, for a caller that
        does not decode)."""
        out = np.array(X, dtype=float).reshape(-1, self.D)
        for i, zi in enumerate(z):
            j = np.clip(np.rint(out[:, i]), 0, len(zi) - 1).astype(int)
            out[:, i] = zi[j]
        return out

    def __call__(self, x) -> List[float]:
        return self.batch([x])[0].tolist()

    def batch(self, X) -> np.ndarray:
        """(f_alpha, f_beta) at every row of X (integers and reals): n x 2."""
        return np.column_stack([self.alpha.batch(self.continuous(X, self.z_alpha)),
                                self.beta.batch(self.continuous(X, self.z_beta))])


def name(fnum: int, D: int) -> str:
    """bbobbiobjmixint07_n10_2D: F7 in ten variables, eight of them integer."""
    return f"bbobbiobjmixint{fnum:02d}_n{D:02d}_2D"


def bounds(D: int) -> List[Tuple[float, float]]:
    return [(0.0, float(l - 1)) for l in arities(D)] + [(-5.0, 5.0)] * (D // 5)


def var_types(D: int) -> list:
    return ["int"] * (4 * D // 5) + ["real"] * (D // 5)
