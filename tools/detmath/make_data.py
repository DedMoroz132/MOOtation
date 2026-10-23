# SPDX-License-Identifier: Apache-2.0
"""Writes tests/data/detmath_reference.txt: for every function of
detmath.hpp, its test points (points.py, 2 * 10^4 a function, plus the hard
inputs of hard_inputs.txt), the value to 60 digits and the bits dm:: gives
(task 14 of 2026-10-20, supplement 2, D7.4). Needs the extension
(_core.detmath) of the build whose bits it records; Python's standard library
otherwise.

    PYTHONPATH=python python tools/detmath/make_data.py

The file, one section a function:

    @ <f> <n generated> <n hard>
    <hi> <q> <d>                     one line a generated point, in order
    <x> [<y>] <hi> <q> <d>           one line a hard input

hi: the correctly rounded value (hex of its bits); q: (exact - hi) / ulp, in
thousandths of an ulp, the ulp of the exact value; d: dm::'s result is hi
moved by d ulps (nextafter steps). tests/test_detmath.cpp makes the same
points, checks dm:: against hi, q (the error, by the table of tolerances)
and against d (the bits: the same file on every platform).
"""

import math
import pathlib
import sys
from decimal import Decimal, localcontext

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import points as P           # noqa: E402
import reference as R        # noqa: E402

ROOT = HERE.parents[1]
OUT = ROOT / "tests" / "data" / "detmath_reference.txt"
N = 20000
ORDER = ("exp", "log", "pow", "sin", "cos", "tan", "acos", "hypot", "asin", "atan",
         "sin_big", "cos_big", "tan_big")


def steps_between(a, b):
    """How many nextafter steps from a to b (signed); a, b finite doubles."""
    def key(x):
        u = P.to_bits(x)
        return -(u & ((1 << 63) - 1)) if u >> 63 else u
    return key(b) - key(a)


def exact_of(f, p):
    """(hi, q) of the exact value; (inf or 0 with q 0) beyond the range."""
    if f == "pow" and p[0] == 0.0:
        raise ValueError("pow(0, y): not a test point")
    v = R.value(f, *p)
    with localcontext() as c:
        c.prec = R.DIGITS + 20
        if abs(v) > Decimal(2) ** 1024 * (1 - Decimal(2) ** -54):
            return math.copysign(math.inf, v), 0
        hi = float(v)
        if hi == 0.0:
            return hi, 0
        q = (v - Decimal(hi)) / R.ulp_of(v)
        return hi, int((q * 1000).to_integral_value())


def main():
    from mootation import _core
    dm = _core.detmath
    hard = {}
    for line in (HERE / "hard_inputs.txt").read_text(encoding="utf-8").splitlines():
        f, *vals = line.split()
        hard.setdefault(f, []).append(tuple(float.fromhex(v) for v in vals))
    out = ["# detmath.hpp reference values: tools/detmath/make_data.py (do not edit by hand)",
           "# @ f n_generated n_hard; then <hi> <q> <d> a point; then <x> [<y>] <hi> <q> <d> a hard input"]
    for f in ORDER:
        gen = P.points(f, N)
        hd = hard.get(f, [])
        out.append(f"@ {f} {len(gen)} {len(hd)}")
        worst = 0.0
        for kind, pts in (("gen", gen), ("hard", hd)):
            for p in pts:
                got = float(getattr(dm, f.replace("_big", ""))(*p))
                hi, q = exact_of(f, p)
                d = 0 if math.isinf(hi) and got == hi else steps_between(hi, got)
                err = abs(d - q / 1000) if not math.isinf(hi) else 0.0
                worst = max(worst, err)
                cols = [] if kind == "gen" else [v.hex() for v in p]
                out.append(" ".join(cols + [f"{P.to_bits(hi):016x}", str(q), str(d)]))
        print(f"{f}: {len(gen)} + {len(hd)} points, largest error {worst:.3f} ulp")
    OUT.write_text("\n".join(out) + "\n", encoding="utf-8", newline="\n")
    print("wrote", OUT, OUT.stat().st_size, "bytes")


if __name__ == "__main__":
    main()
