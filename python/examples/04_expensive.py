#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""When each evaluation is expensive, or lives outside Python.

    python python/examples/04_expensive.py

Two things matter once an evaluation costs real time:

  1. Evaluate a whole generation at once. `batch=` hands the optimizer's
     candidates over as a list, so the boundary is crossed once per generation
     instead of once per individual. That is where NumPy vectorization, a
     thread pool, or a queue of cluster jobs goes.

  2. Never evaluate the same point twice. Evolutionary algorithms re-propose
     identical individuals more often than one expects, and a dictionary
     keyed on the decision vector is the entire fix.

The "solver" here is deliberately slowed down so the difference is visible.
"""

import time

import mootation


N_VARS = 8
CALLS = {"individual": 0, "batched": 0, "cached": 0}


def slow_objectives(x):
    """Stands in for a solver: a fixed cost per call, plus a little arithmetic."""
    time.sleep(0.0002)
    g = 1.0 + 9.0 * sum(x[1:]) / (len(x) - 1)
    return [x[0], g * (1.0 - (x[0] / g) ** 0.5)]


def run_plain():
    CALLS["individual"] = 0

    def fn(x):
        CALLS["individual"] += 1
        return slow_objectives(x)

    t0 = time.perf_counter()
    res = mootation.minimize(fn, [(0.0, 1.0)] * N_VARS, 2,
                             pop_size=40, n_gen=30, seed=3)
    return time.perf_counter() - t0, res


def run_batched():
    CALLS["batched"] = 0

    def batch(X):
        CALLS["batched"] += len(X)
        # A real one would submit X to a pool, a queue, or a vectorized kernel.
        return [slow_objectives(x) for x in X]

    t0 = time.perf_counter()
    res = mootation.minimize(slow_objectives, [(0.0, 1.0)] * N_VARS, 2,
                             batch=batch, pop_size=40, n_gen=30, seed=3)
    return time.perf_counter() - t0, res


def run_cached():
    CALLS["cached"] = 0
    seen = {}

    def batch(X):
        out = []
        for x in X:
            # Rounded before hashing: two vectors that differ only in float
            # noise are the same point as far as any solver is concerned.
            key = tuple(round(v, 12) for v in x)
            if key not in seen:
                CALLS["cached"] += 1
                seen[key] = slow_objectives(x)
            out.append(seen[key])
        return out

    t0 = time.perf_counter()
    res = mootation.minimize(slow_objectives, [(0.0, 1.0)] * N_VARS, 2,
                             batch=batch, pop_size=40, n_gen=30, seed=3)
    return time.perf_counter() - t0, res


def main():
    print("40 individuals x 30 generations, ~0.2 ms of fake solver per call\n")

    t_plain, r_plain = run_plain()
    print(f"  one at a time   {t_plain:6.2f}s   "
          f"{CALLS['individual']:>5} evaluations")

    t_batch, r_batch = run_batched()
    print(f"  batched         {t_batch:6.2f}s   "
          f"{CALLS['batched']:>5} evaluations")

    t_cache, r_cache = run_cached()
    print(f"  batched+cached  {t_cache:6.2f}s   "
          f"{CALLS['cached']:>5} evaluations   "
          f"({CALLS['batched'] - CALLS['cached']} repeats avoided)")

    # Same seed, same algorithm, same problem: the three must agree. If caching
    # changed the answer, the cache key would be wrong.
    a = sorted(map(tuple, r_plain.objectives))
    b = sorted(map(tuple, r_batch.objectives))
    c = sorted(map(tuple, r_cache.objectives))
    assert a == b == c, "batching or caching changed the result"
    print("\nall three produced identical populations, as they must")

    print("\nFor an evaluator that is a separate PROGRAM rather than a Python\n"
          "function, use the TOML layer instead — it handles scratch\n"
          "directories, timeouts, output parsing, failure policy and resume:\n"
          "    python -m mootation.run --check python/examples/demo.toml")


if __name__ == "__main__":
    main()
