#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Saving a run, and starting the next one from it.

    python python/examples/06_restart.py

Three files, all optional:

    population    what SURVIVED — the answer set
    evaluations   what was COMPUTED — every call, including the discards
    (both carry the settings that produced them, in a `#` preamble)

The point is restarting. Finish a run, keep the population, continue — with
the same settings, or with different ones, or with a different algorithm.

This is a warm start, not a checkpoint. No RNG position and no per-algorithm
state are saved, only the decision variables and objectives — which is exactly
what every algorithm shares, and therefore what lets a population produced by
NSGA-II seed a MOEA/D run. A resumed run does not reproduce what the
uninterrupted one would have done; it starts from the same place.

Seeding costs ZERO evaluations: the objectives come from the file rather than
being recomputed. That is the whole point when one evaluation is a solver run.
"""

import math
import tempfile
from pathlib import Path

import mootation


def zdt1(x):
    g = 1.0 + 9.0 * sum(x[1:]) / (len(x) - 1)
    return [x[0], g * (1.0 - math.sqrt(x[0] / g))]


def front_gap(result):
    """How far the worst solution sits from ZDT1's analytic front."""
    return max(abs(f[1] - (1.0 - math.sqrt(f[0]))) for f in result.objectives)


def main():
    bounds = [(0.0, 1.0)] * 10
    tmp = Path(tempfile.mkdtemp(prefix="mootation_restart_"))
    pop_file = tmp / "population.csv"
    log_file = tmp / "evaluations.csv"

    # ── 1. A short run, saving both files ───────────────────────────────────
    first = mootation.minimize(
        zdt1, bounds, 2,
        algorithm="nsga2", pop_size=60, n_gen=40, seed=1,
        save_population=pop_file,
        log_evaluations=log_file,
    )
    print(f"run 1  nsga2, 40 generations  ->  gap to the true front "
          f"{front_gap(first):.4f}")

    evals = sum(1 for line in log_file.read_text(encoding="utf-8").splitlines()
                if line and not line.startswith("#")) - 1     # minus the header
    print(f"       {len(first.objectives)} survivors kept, "
          f"{evals} evaluations recorded")
    print(f"       the log holds every call, including the {evals - len(first.objectives)} "
          f"the optimizer discarded")

    # ── 2. What the file remembers ──────────────────────────────────────────
    loaded = mootation.load_population(pop_file)
    print(f"\nreloaded: {loaded!r}")
    print(f"       settings recorded: {loaded.meta}")

    # ── 3. Continue: same settings, more generations ────────────────────────
    more = mootation.minimize(
        zdt1, bounds, 2,
        algorithm="nsga2", pop_size=60, n_gen=40, seed=2,
        seed_population=pop_file,
    )
    print(f"\nrun 2  same settings, 40 more     ->  gap {front_gap(more):.4f}")

    # ── 4. Continue with a DIFFERENT algorithm ──────────────────────────────
    # Only variables and objectives were saved, and those are the fields every
    # algorithm shares — so the population crosses between families freely.
    # pop_size changes too, which is why on_size_mismatch exists: 91 is a
    # Das-Dennis lattice size for 2 objectives and 60 is not what NSGA-III wants.
    switched = mootation.minimize(
        zdt1, bounds, 2,
        algorithm="nsga3", pop_size=91, n_gen=60, seed=3,
        seed_population=pop_file,
        on_size_mismatch="pad",     # 60 saved -> 91 needed
    )
    print(f"run 3  switched to nsga3, padded  ->  gap {front_gap(switched):.4f}")

    # ── 5. A cold start on the same budget, for comparison ──────────────────
    cold = mootation.minimize(zdt1, bounds, 2,
                              algorithm="nsga2", pop_size=60, n_gen=40, seed=2)
    print(f"\ncold   nsga2, 40 generations from random  ->  gap "
          f"{front_gap(cold):.4f}")
    print(f"\nSame budget, same seed, same algorithm: warm {front_gap(more):.4f}"
          f" against cold {front_gap(cold):.4f}."
          f"\nThe 40 generations behind the warm start are real work that did"
          f"\nnot have to be repeated — which is the whole point when one"
          f"\nevaluation is a solver run rather than arithmetic.")

    assert front_gap(more) <= front_gap(first) + 1e-9, \
        "continuing should not lose ground"
    assert len(switched.objectives) == 91, "nsga3 should return its pop_size"

    print(f"\nfiles left in {tmp}")


if __name__ == "__main__":
    main()
