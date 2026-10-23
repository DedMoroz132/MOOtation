# SPDX-License-Identifier: Apache-2.0
"""Prints the hard inputs of the 10 functions of detmath.hpp: the double-
precision inputs with the largest known errors of the libraries in
B. Gladman, V. Innocente, J. Mather, P. Zimmermann, "Accuracy of Mathematical
Functions in Single, Double, Double Extended, and Quadruple Precision"
(report, HAL hal-03141101, version of 2026-08-24), Tables 4-11 (task 14 of
2026-10-20, supplement 2, D7.4: test inputs, not code).

    python tools/detmath/hard_inputs.py <the report as Markdown> > tools/detmath/hard_inputs.txt

One line per input: the function, x (and y) as hex floats. Inputs of sin,
cos and tan beyond 2^20 pi/2 are left out (outside the area of detmath.hpp).
"""

import re
import sys

FUNCS = ("acos", "asin", "atan", "cos", "exp", "hypot", "log", "pow", "sin", "tan")
HEX = re.compile(r"-?0x[0-9a-fA-F.]+p[+-]?\d+")
TRIG_MAX = 1647099.3291652855        # 2^20 pi/2


def main():
    lines = open(sys.argv[1], encoding="utf-8").read().splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith("Table 3: Double precision"))
    end = next(i for i, l in enumerate(lines) if l.startswith("Table 12: Double extended"))
    seen = set()
    for line in lines[start:end]:
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 2 or cells[0] not in FUNCS:
            continue
        f = cells[0]
        for cell in cells[1:]:
            vals = [float.fromhex(h) for h in HEX.findall(cell)]
            if f in ("hypot", "pow"):
                if len(vals) != 2:
                    continue
                key = (f, vals[0], vals[1])
            else:
                if len(vals) != 1:
                    continue
                if f in ("sin", "cos", "tan") and abs(vals[0]) > TRIG_MAX:
                    continue
                key = (f, vals[0])
            if key not in seen:
                seen.add(key)
                print(f, " ".join(v.hex() for v in key[1:]))


if __name__ == "__main__":
    main()
