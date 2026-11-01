# SPDX-License-Identifier: Apache-2.0
"""The Python side of the wrappers' cross-check (task 20 of 2026-11-02,
§20.2.1): python/mootation/benchmarks/wrap.py over the NumPy code of bbob and
DTLZ2, written to tests/data/wrap_python.json, which tests/test_wrap.cpp holds
include/mootation/suite/wrap.hpp against: the permutations and the shifts bit
for bit, the rotations, the points y = T(x) and the values there to a
tolerance.

Run by hand; CI never runs it. From this repository's root (NumPy only, no
extension):

    PYTHONPATH=python python tools/refs/gen_wrap_refs.py

Three points per case, x = lo + (hi - lo) u, u_i = (0.1 j + 0.61803398875 i)
mod 1, j = 1..3, in the wrapped box (a box-less variable: lo, hi = -3, 3).
Before writing, the script checks wrap.py itself: T^-1(T(x)) = x, a
permutation alone exact, Q orthogonal.
"""

import datetime
import json
import math
from pathlib import Path

import numpy as np

from mootation.benchmarks import bbob, dtlz
from mootation.benchmarks.wrap import WrapMap

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests" / "data" / "wrap_python.json"

# (name, family, n, M, bbob fid, the wrappers' instances)
CASES = [
    ("bbob10_n05_1D", "bbob", 5, 1, 10, dict(perm=1, rotate=1, shift=1)),
    ("bbob08_n04_1D", "bbob", 4, 1, 8, dict(rotate=2)),
    ("bbob01_n03_1D", "bbob", 3, 1, 1, dict(perm=3)),
    ("bbob14_n06_1D", "bbob", 6, 1, 14, dict(shift=2, shift_scale=0.25)),
    ("DTLZ2_3D", "DTLZ2", 12, 3, 0, dict(perm=2, shift=1)),
]


def points(box, count=3):
    n = len(box)
    out = []
    for j in range(1, count + 1):
        x = []
        for i in range(1, n + 1):
            lo, hi = box[i - 1]
            if not (math.isfinite(lo) and math.isfinite(hi)):
                lo, hi = -3.0, 3.0
            x.append(lo + (hi - lo) * ((0.1 * j + 0.61803398875 * i) % 1.0))
        out.append(x)
    return out


def main():
    cases = []
    for name, family, n, M, fid, inst in CASES:
        if family == "bbob":
            f = bbob.instance(fid, 1, n)
            box, x_opt, fn = bbob.bounds(n), f.x_opt, (lambda y, _f=f: [_f(y)])
            everywhere = True
        else:
            box, x_opt, everywhere = [(0.0, 1.0)] * n, None, False
            fn = lambda y, _M=M: [float(v) for v in dtlz.dtlz2_rows(np.asarray(y)[None, :], _M)[0]]
        w = WrapMap(name, n, M, box, defined_everywhere=everywhere, x_opt=x_opt, **inst)
        rows = []
        for x in points(w.bounds()):
            y = w.to_problem(x)
            back = w.to_run(y)
            assert np.allclose(back, x, rtol=0, atol=1e-13 * (1 + np.max(np.abs(x)))), name
            if inst.get("rotate", 0) == 0 and inst.get("shift", 0) == 0:
                assert np.array_equal(back, np.asarray(x)), name
            rows.append({"x": [float(v) for v in x], "y": [float(v) for v in y], "f": fn(y)})
        if w.Q is not None:
            assert np.allclose(w.Q.T @ w.Q, np.eye(n), atol=1e-13), name
        case = {"name": name, "family": family, "n": n, "n_obj": M, "bbob_fid": fid,
                "perm": inst.get("perm", 0), "rotate": inst.get("rotate", 0), "shift": inst.get("shift", 0),
                "shift_scale": inst.get("shift_scale", 1.0),
                "pi": w.pi, "s": [float(v) for v in w.s],
                "Q": [] if w.Q is None else [float(v) for v in w.Q.ravel()],
                "centre": [] if w.centre is None else [float(v) for v in w.centre],
                "box": [[float(lo), float(hi)] for lo, hi in w.bounds()], "points": rows}
        if x_opt is not None:
            case["x_opt_run"] = [float(v) for v in w.to_run(x_opt)]
            case["f_opt"] = float(f.f_opt)
        cases.append(case)
    data = {"library": "MOOtation's own Python twin, python/mootation/benchmarks/wrap.py",
            "version": "NumPy " + np.__version__, "license": "Apache-2.0",
            "command": "PYTHONPATH=python python tools/refs/gen_wrap_refs.py",
            "date": datetime.date.today().isoformat(), "points": sum(len(c["points"]) for c in cases),
            "tolerance": "pi and s exact; Q, y, f to 1e-12 of their scale", "cases": cases}
    OUT.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"{OUT}: {len(cases)} cases, {data['points']} points")


if __name__ == "__main__":
    main()
