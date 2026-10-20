# SPDX-License-Identifier: Apache-2.0
"""The accuracy of detmath.hpp on 10^6 points a function (task 14 of
2026-10-20, supplement 2, D7.4): the points of points.py (other seeds than
the test file's), dm:: through _core.detmath, the exact values to 60 digits
(reference.py). Prints, per function and per block of points, the largest
error in ulps, where it is, and the share of results not correctly rounded.

    PYTHONPATH=python python tools/detmath/check.py [n] [processes] [function ...]

Python's standard library besides the extension; minutes on several processes.
"""

import math
import multiprocessing
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import points as P           # noqa: E402
import reference as R        # noqa: E402

ORDER = ("exp", "log", "pow", "sin", "cos", "tan", "acos", "hypot")


def _chunk(args):
    f, pts = args
    from mootation import _core
    fn = getattr(_core.detmath, f)
    worst, where, not_cr = 0.0, None, 0
    for p in pts:
        if f == "pow" and p[0] == 0.0:
            continue
        got = float(fn(*p))
        v = R.value(f, *p)
        if v.is_infinite() or abs(v) > R.Decimal(2) ** 1024:
            err = 0.0 if math.isinf(got) else math.inf
        elif abs(v) < R.Decimal(2) ** -1075:
            err = 0.0 if got == 0.0 else math.inf
        else:
            err = R.ulp_error(got, v)
            if got != float(v):
                not_cr += 1
        if err > worst:
            worst, where = err, p
    return worst, where, not_cr, len(pts)


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10 ** 6
    procs = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    funcs = sys.argv[3:] or ORDER
    with multiprocessing.Pool(procs) as pool:
        for f in funcs:
            gen, shares = P.BLOCKS[f]
            g = P.SplitMix64(P.SEEDS[f] + 1000)          # not the test file's points
            print(f"{f}:")
            total_bad, total = 0, 0
            for block, share in enumerate(shares):
                pts = [gen(g, block) for _ in range(n * share // 10)]
                chunks = [(f, pts[i:i + 5000]) for i in range(0, len(pts), 5000)]
                worst, where, bad, cnt = 0.0, None, 0, 0
                for w, at, b, c in pool.map(_chunk, chunks):
                    bad += b
                    cnt += c
                    if w > worst:
                        worst, where = w, at
                total_bad += bad
                total += cnt
                print(f"  block {block}: {cnt} points, largest error {worst:.4f} ulp at "
                      f"{tuple(v.hex() for v in where) if where else '-'}, "
                      f"not correctly rounded {100 * bad / max(cnt, 1):.3f} %")
            print(f"  all: {total} points, not correctly rounded {100 * total_bad / max(total, 1):.3f} %",
                  flush=True)


if __name__ == "__main__":
    main()
