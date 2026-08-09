# SPDX-License-Identifier: Apache-2.0
"""A stand-in solver, so the demo config runs anywhere Python does.

Reads the decision vector from the generated input file, writes a CSV with one
row. Real solvers do the same thing with a mesh in between.
"""
import csv
import math
import sys


def main(argv):
    if len(argv) != 3:
        print("usage: solver_demo.py <input> <output.csv>", file=sys.stderr)
        return 2
    x = [float(t) for t in open(argv[1], encoding="utf-8").read().split()]
    g = 1.0 + 9.0 * sum(x[1:]) / max(1, len(x) - 1)
    with open(argv[2], "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["mass", "drag", "stress_max"])
        w.writerow([x[0], g * (1.0 - math.sqrt(x[0] / g)), sum(x) - len(x) * 0.9])
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
