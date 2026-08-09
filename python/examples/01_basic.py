#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""The smallest useful thing: minimize a Python function.

    python python/examples/01_basic.py

Write the objectives as an ordinary function of one decision vector, say where
the variables live, and call minimize. Everything else has a default.
"""

import math

import mootation


def zdt1(x):
    """ZDT1: two objectives, a convex front, thirty variables' worth of trap.

    f1 is x[0] alone; f2 punishes every other variable for being away from 0.
    The Pareto front is f2 = 1 - sqrt(f1), reached only when x[1:] are all 0.
    """
    g = 1.0 + 9.0 * sum(x[1:]) / (len(x) - 1)
    return [x[0], g * (1.0 - math.sqrt(x[0] / g))]


def main():
    result = mootation.minimize(
        zdt1,
        bounds=[(0.0, 1.0)] * 10,
        n_objs=2,
        algorithm="nsga2",
        pop_size=100,
        n_gen=200,
        seed=42,             # omit for a different run every time
    )

    print(f"{result.active_n} solutions on the approximated front\n")
    print(f"  {'f1':>8}  {'f2':>8}")
    # Sorted by f1 so the printout reads as a front rather than as a heap.
    for f in sorted(result.objectives, key=lambda f: f[0])[:10]:
        print(f"  {f[0]:8.4f}  {f[1]:8.4f}")
    print("  ...")

    # The analytic front is f2 = 1 - sqrt(f1); the distance to it is how far
    # this run got. On ZDT1 that is a fair summary because the front is known —
    # for a real problem it is not, which is the whole reason to keep the
    # population rather than a single point.
    worst = max(abs(f[1] - (1.0 - math.sqrt(f[0]))) for f in result.objectives)
    print(f"\nfurthest any solution sits from the true front: {worst:.4f}")


if __name__ == "__main__":
    main()
