#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""The problems with symmetries (task 19, §19.1.6): C++ against Python.

Every registry instance of suite/sym.hpp's families (benchmarks/symmetric.py,
ROWS) is made by C++ (_core.Native) and by NumPy (symmetric.numpy_family), and
the two are compared at the instance's start and at two points about it: the
residuals, the Jacobian, the objective, the gradient, the gauge directions
and canon, to 1e-12 of each column's scale (NumPy's exp, log and pow are not
dm::'s; the normal draws take math.log). The C++ invariants themselves (canon
bit for bit, the gauge test, f(x_opt) = f_opt) are tests/test_suite.cpp's.
"""

from __future__ import annotations

import sys

import numpy as np

from mootation import _core
from mootation.benchmarks import symmetric as sym

TOL = 1e-12
failed = 0


def check(ok: bool, what: str) -> None:
    global failed
    if not ok:
        failed += 1
    print(("ok    " if ok else "FAIL  ") + what)


def close(a, b, scale=None) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    s = float(np.max(np.abs(b))) if scale is None else scale
    return float(np.max(np.abs(a - b))) / max(s, 1e-300) if a.size else 0.0


def main() -> int:
    for row in sym.ROWS:
        if row.family == "lj_cluster" and row.inst.dim > 26:
            continue                                   # the largest clusters: the same code as N <= 26
        nat = _core.Native(sym.spec(row))
        py = sym.numpy_family(row.family, row.inst)
        x0 = np.array(nat.start())
        rng = np.random.default_rng(7)
        worst = {"r": 0.0, "J": 0.0, "f": 0.0, "g": 0.0, "N": 0.0, "canon": 0.0}
        for t in range(3):
            x = x0 if t == 0 else x0 + 0.05 * (1.0 + np.abs(x0)) * rng.standard_normal(x0.size)
            lo, hi = np.array(nat.bounds).T
            x = np.clip(x, lo, hi)
            if nat.n_res:
                r, J = nat.residuals(list(x), nat.has_jacobian) if nat.has_jacobian else (nat.residuals(list(x)), None)
                worst["r"] = max(worst["r"], close(r, py.residuals(x)))
                if J is not None:
                    Jp = py.jacobian(x)
                    worst["J"] = max(worst["J"], max(close(np.array(J)[:, j], Jp[:, j], float(np.max(np.abs(Jp))))
                                                     for j in range(x.size)))
            else:
                worst["f"] = max(worst["f"], close([nat.evaluate(list(x))[0]], [py.value(x)]))
                worst["g"] = max(worst["g"], close(nat.gradient(list(x)), py.gradient(x)))
            if nat.gauge_dim:
                worst["N"] = max(worst["N"], close(nat.gauge_basis(list(x)), py.gauge_basis(x)))
            if nat.has_canon and hasattr(py, "canon"):
                worst["canon"] = max(worst["canon"], close(nat.canon(list(x)), py.canon(x)))
        seen = ", ".join(f"{k} {v:.1e}" for k, v in worst.items() if v)
        check(all(v <= TOL for v in worst.values()), f"{sym.name(row)}: C++ = NumPy to {TOL:g} ({seen or 'exact'})")
        # the spec travels: pickled and made again, the same problem
        again = _core.Native(nat.spec)
        check(again.evaluate(list(x0)) == nat.evaluate(list(x0)) and again.n_var == nat.n_var,
              f"{sym.name(row)}: its spec makes the same problem")
    check(abs(sym.resolution(1e2) - 2.44) < 0.005 and abs(sym.resolution(1e4) - 1.63) < 0.005,
          "resolution limit: Istratov & Vyvenko's Table I")
    print(f"test_symmetric: {'FAILED, ' + str(failed) + ' check(s)' if failed else 'all checks passed'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
