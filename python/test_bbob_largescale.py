#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""bbob-largescale (task 20, §24.6.5): C++ against NumPy.

Every function in 20, 80 and 640 variables, instance 1, is made by C++
(_core.Native, suite/bbob_largescale.hpp) and by NumPy
(benchmarks/bbob_largescale.py), each drawing the instance itself, and the
two are compared at x_opt and at three points of the box to 1e-10 of the
value's scale (NumPy's exp, log, pow and QR are not the library's). The
NumPy suite is also held to bbob.py at n = 20 and 40 with bbob's instance
handed over (one block, no permutations; f14 squared, LS-3). The C++
invariants (bbob.hpp's values at n <= 40, orthogonality, f(x_opt) = f_opt)
are tests/test_suite.cpp's.
"""

from __future__ import annotations

import sys

import numpy as np

from mootation import _core
from mootation.benchmarks import bbob as B
from mootation.benchmarks import bbob_largescale as L

TOL = 1e-10
failed = 0


def check(ok: bool, what: str) -> None:
    global failed
    if not ok:
        failed += 1
    print(("ok    " if ok else "FAIL  ") + what)


def main() -> int:
    worst, where = 0.0, ""
    for n in (20, 80, 640):
        for fid in range(1, 25):
            nat = _core.Native({"family": "bbob_largescale", "n_var": n, "n_obj": 1, "bbob_fid": fid, "bbob_iid": 1})
            py = L.instance(fid, 1, n)
            rng = np.random.default_rng(100 * n + fid)
            for x in [py.x_opt] + [rng.uniform(-5.0, 5.0, n) for _ in range(3)]:
                a, b = nat.evaluate(list(x))[0], py(x)
                e = abs(a - b) / max(1.0, abs(b - py.f_opt))
                if e > worst:
                    worst, where = e, f"f{fid} n{n}"
    check(worst <= TOL, f"C++ = NumPy at n = 20, 80, 640, every function, to {TOL:g} (worst {worst:.1e} at {where})")
    worst, where = 0.0, ""
    for n in (20, 40):
        for fid in range(1, 25):
            b = B.instance(fid, 1, n)
            ident = np.arange(n)
            s = L.Instance(fid, n, b.x_opt.copy(), b.f_opt)
            s.R = L.Rotation(n, [b.R], ident, ident) if b.R is not None else None
            s.Q = L.Rotation(n, [b.Q], ident, ident) if b.Q is not None else None
            s.l10, s.la = b.l10, b.la
            if fid in (5, 20, 24):
                s.sign = b.peaks["sign"]
            elif fid in (21, 22):
                s.peaks = b.peaks
            rng = np.random.default_rng(fid)
            for _ in range(5):
                x = rng.uniform(-5.0, 5.0, n)
                fb = b(x) - b.f_opt
                fb = fb * fb if fid == 14 else fb
                e = abs((s(x) - b.f_opt) - fb) / max(1.0, abs(fb))
                if e > worst:
                    worst, where = e, f"f{fid} n{n}"
    check(worst <= 1e-12, f"NumPy at n = 20, 40 with bbob's instance: bbob.py's values (worst {worst:.1e} at {where})")
    print(f"test_bbob_largescale: {'FAILED, ' + str(failed) + ' check(s)' if failed else 'all checks passed'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
