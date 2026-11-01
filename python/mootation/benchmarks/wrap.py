# SPDX-License-Identifier: Apache-2.0
"""Wrappers over any problem: a permutation, a rotation and a shift of the
whole decision vector (task 20 of 2026-11-02, §20.2.1), the cross-check of
include/mootation/suite/wrap.hpp, which is the reference: its header has the
construction and the readings WRAP-1 .. WRAP-6, and this module repeats them.

The run sees x, the problem is evaluated at y = T(x):

  shift    u = x - s
  rotate   v = c + Q (u - c)
  perm     y_i = v[pi[i]]

and back, v[pi[i]] = y_i, u = c + Q^T (v - c), x = u + s. A Jacobian J_y of
residuals becomes J_y P Q, row by row Q^T (P^T row); a gradient the same
(pull).

Each wrapper draws from its own key (20261102, kind, crc32(name), n, instance),
kind 1 perm, 2 rotate, 3 shift, with library_rng.py's generator: pi is its
shuffle of 0 .. n - 1, Q its haar(n), s_j = scale_j (2 U_j - 1). pi and s are
C++'s bit for bit, Q to a few ulps (NumPy's QR against the header's own
Householder). Q d and Q^T d are rows' pairwise sums (_rows.matvec_pairwise),
as C++ adds them.
"""

from __future__ import annotations

import math
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from . import library_rng as _lrng
from ._rows import matvec_pairwise

WRAP_KEY = 20261102
PERM, ROTATE, SHIFT = 1, 2, 3                      # the key's kinds


def _rng(kind: int, name: str, n: int, instance: int) -> _lrng.LibraryRng:
    return _lrng.LibraryRng(_lrng.seed(WRAP_KEY, kind, _lrng.crc32(name), n, instance))


class WrapMap:
    """The map x -> y over a problem of n variables (wrap.hpp, WrapMap).

    bounds: the problem's box, (lower, upper) per variable; x_opt and x0 its
    optimum and start where known. perm, rotate, shift: instance numbers
    (0: off)."""

    def __init__(self, name: str, n: int, n_obj: int, bounds: Sequence[Tuple[float, float]],
                 perm: int = 0, rotate: int = 0, shift: int = 0, shift_scale: float = 1.0,
                 defined_everywhere: bool = False, x_opt: Optional[Sequence[float]] = None,
                 x0: Optional[Sequence[float]] = None):
        def fail(why):
            raise ValueError(f"wrap: {name or 'a problem'}: {why}")
        if n < 1:
            fail("a problem of no variables")
        if len(bounds) != n:
            fail(f"the box has {len(bounds)} variables, not n")
        if x_opt is not None and len(x_opt) != n:
            fail("x_opt of n values")
        if x0 is not None and len(x0) != n:
            fail("x0 of n values")
        if min(perm, rotate, shift) < 0:
            fail("an instance number below 0")
        if not (shift_scale > 0.0 and math.isfinite(shift_scale)):
            fail("shift_scale: a finite number above 0")
        self.name, self.n = name, n
        self.perm_instance, self.rotate_instance, self.shift_instance = perm, rotate, shift
        self.shift_scale = shift_scale
        if self.active and not name:
            fail("the key needs the problem's name")

        self.pi = list(range(n))
        if perm > 0:
            _rng(PERM, name, n, perm).shuffle(self.pi)
        vbox = [None] * n                          # v[pi[i]] lies in the problem's box of y_i
        for i in range(n):
            vbox[self.pi[i]] = (float(bounds[i][0]), float(bounds[i][1]))

        self.Q = None
        self.centre = None
        if rotate > 0:
            if n_obj > 1:
                fail("rotate: a multi-objective problem (its rotations are the r* maps, WRAP-3)")
            boxed = any(math.isfinite(lo) or math.isfinite(hi) for lo, hi in bounds)
            if boxed and not defined_everywhere:
                fail("rotate: a problem with a box (a rotation carries points out of it, and bringing "
                     "them back changes the problem; WRAP-3)")
            self.Q = _lrng.haar(_rng(ROTATE, name, n, rotate), n)
            self.centre = np.zeros(n)              # WRAP-2
            if x_opt is not None:
                for i in range(n):
                    self.centre[self.pi[i]] = float(x_opt[i])

        self.s = np.zeros(n)
        if shift > 0:
            a = x_opt if x_opt is not None else (x0 if x0 is not None else ())
            size = 1.0                             # WRAP-4
            for v in a:
                size = max(size, abs(float(v)))
            rng = _rng(SHIFT, name, n, shift)
            for j, (lo, hi) in enumerate(vbox):
                scale = hi - lo if math.isfinite(lo) and math.isfinite(hi) else size
                self.s[j] = shift_scale * scale * (2.0 * rng.uniform01() - 1.0)
        self.box = [(lo + self.s[j], hi + self.s[j]) for j, (lo, hi) in enumerate(vbox)]

    @property
    def active(self) -> bool:
        return self.perm_instance > 0 or self.rotate_instance > 0 or self.shift_instance > 0

    def to_problem(self, x) -> np.ndarray:
        """y = T(x)."""
        v = np.array(x, dtype=float)
        if self.shift_instance > 0:
            v = v - self.s
        if self.Q is not None:
            v = self.centre + matvec_pairwise(self.Q, v - self.centre)
        return v[self.pi]

    def to_run(self, y) -> np.ndarray:
        """x = T^-1(y)."""
        u = np.empty(self.n)
        u[self.pi] = np.asarray(y, dtype=float)
        if self.Q is not None:
            u = self.centre + matvec_pairwise(self.Q.T, u - self.centre)
        return u + self.s if self.shift_instance > 0 else u

    def pull(self, g_y) -> np.ndarray:
        """(dy/dx)^T g_y = Q^T (P^T g_y): a gradient, or a row of a Jacobian."""
        w = np.empty(self.n)
        w[self.pi] = np.asarray(g_y, dtype=float)
        return matvec_pairwise(self.Q.T, w) if self.Q is not None else w

    def jacobian(self, J_y) -> np.ndarray:
        """J_y P Q, row by row."""
        J_y = np.asarray(J_y, dtype=float).reshape(-1, self.n)
        return np.array([self.pull(row) for row in J_y]).reshape(J_y.shape)

    def bounds(self) -> List[Tuple[float, float]]:
        return list(self.box)

    def describe(self) -> str:
        s = (f"shift {self.shift_instance}, rotate {self.rotate_instance}, perm {self.perm_instance} of "
             f"'{self.name}' (key {WRAP_KEY}, kind, crc32 {_lrng.crc32(self.name)}, n {self.n}, instance)")
        if self.shift_instance > 0 and self.shift_scale != 1.0:
            s += f", shift_scale {self.shift_scale:f}"
        return s


class Wrapped:
    """A problem's functions seen through a WrapMap (wrap.hpp, Wrapped):
    evaluate(x) and constraints(x) at T(x); residuals(x) -> (r, J) with J_y P Q
    where the problem's residuals give a Jacobian."""

    def __init__(self, wmap: WrapMap, evaluate: Callable, constraints: Optional[Callable] = None,
                 residuals: Optional[Callable] = None):
        self.map = wmap
        self._evaluate, self._constraints, self._residuals = evaluate, constraints, residuals

    def evaluate(self, x):
        return self._evaluate(self.map.to_problem(x))

    def constraints(self, x):
        return self._constraints(self.map.to_problem(x))

    def residuals(self, x):
        r, J = self._residuals(self.map.to_problem(x))
        return r, (None if J is None else self.map.jacobian(J))
