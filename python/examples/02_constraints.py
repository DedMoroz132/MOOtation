#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Constrained optimization: a second function that says what is allowed.

    python python/examples/02_constraints.py

The convention throughout MOOtation is that a constraint value <= 0 means
SATISFIED, and a positive value is the size of the violation. Returning the
amount by which a limit is exceeded — rather than True/False — is what lets the
algorithm prefer a nearly-feasible point over a wildly infeasible one while
both are still infeasible.
"""

import math

import mootation


def objectives(x):
    """BNH: minimize both. A textbook constrained bi-objective problem."""
    return [4.0 * x[0] ** 2 + 4.0 * x[1] ** 2,
            (x[0] - 5.0) ** 2 + (x[1] - 5.0) ** 2]


def constraints(x):
    """Two circles. Each returns violation > 0, or <= 0 when satisfied.

    C1: (x0-5)^2 + x1^2      <= 25
    C2: (x0-8)^2 + (x1+3)^2  >= 7.7
    """
    c1 = (x[0] - 5.0) ** 2 + x[1] ** 2 - 25.0
    c2 = 7.7 - ((x[0] - 8.0) ** 2 + (x[1] + 3.0) ** 2)
    return [c1, c2]


def main():
    result = mootation.minimize(
        objectives,
        bounds=[(0.0, 5.0), (0.0, 3.0)],
        n_objs=2,
        constraints=constraints,     # supplying this turns constraint handling on
        algorithm="nsga2",
        pop_size=100,
        n_gen=150,
        seed=1,
    )

    feasible = [i for i, cv in enumerate(result.cv) if cv <= 0.0]
    print(f"{len(feasible)} of {result.active_n} solutions are feasible "
          f"(cv <= 0)\n")

    print(f"  {'f1':>9}  {'f2':>9}  {'cv':>8}")
    rows = sorted(feasible, key=lambda i: result.objectives[i][0])
    for i in rows[:8]:
        f = result.objectives[i]
        print(f"  {f[0]:9.3f}  {f[1]:9.3f}  {result.cv[i]:8.3f}")
    print("  ...")

    # The check that matters: the reported cv must agree with the constraints
    # as written. If it does not, the run optimized something else.
    bad = 0
    for i in feasible:
        c = constraints(result.variables[i])
        if any(v > 1e-9 for v in c):
            bad += 1
    print(f"\nsolutions reported feasible that actually violate a constraint: "
          f"{bad}")
    if bad:
        raise SystemExit("constraint handling disagrees with the constraints")


if __name__ == "__main__":
    main()
