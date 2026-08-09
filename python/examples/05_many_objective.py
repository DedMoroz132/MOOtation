#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Many objectives, where the choice of algorithm starts to matter.

    python python/examples/05_many_objective.py

Past three or four objectives almost every solution is nondominated, so
Pareto dominance stops discriminating and NSGA-II-style algorithms stall. That
is what the reference-vector and decomposition families exist for.

It also introduces a practical trap: NSGA-III, MOEA/D and their relatives
require the population size to be an exact Das-Dennis lattice count, which
depends on the objective count. 91 is one for M=3 and is NOT one for M=5.
`lattice_sizes` below is the same arithmetic the config validator uses.
"""

import mootation
import mootation.benchmarks as bench
from mootation.run.algorithms import lattice_sizes, check_pop


def show_lattice_sizes():
    print("Das-Dennis lattice sizes — the populations NSGA-III can take:\n")
    for m in (3, 5, 8, 10):
        sizes = [n for n in lattice_sizes(m, 400)]
        print(f"  M={m:<3} {', '.join(str(s) for s in sizes[:9])}"
              + (" ..." if len(sizes) > 9 else ""))

    print("\nAsk for one it cannot take and you are told, before anything runs:")
    print("  ", check_pop("nsga3", 100, 5))


def compare_at(m=5, problem=None, pop=126, n_gen=120):
    problem = problem or f"DTLZ2_{m}D"
    print(f"\n{problem}: {m} objectives, pop={pop}, {n_gen} generations\n")
    print(f"  {'algorithm':<14} {'family':<22} {'IGD':>9}")

    families = {
        "nsga2":      "Pareto dominance",
        "nsga3":      "reference points",
        "moead_de":   "decomposition",
        "rvea":       "reference vectors",
        "ibea_eplus": "indicator",
        "hype":       "indicator (HV)",
    }
    rows = []
    for alg, family in families.items():
        res = bench.solve(problem, alg, pop_size=pop, n_gen=n_gen, seed=1)
        rows.append((alg, family, bench.igd(problem, res.objectives)))

    for alg, family, score in sorted(rows, key=lambda r: r[2]):
        print(f"  {alg:<14} {family:<22} {score:>9.5f}")

    print("\nOne seed, one problem — this ranks nothing on its own. What it does\n"
          "show is that the spread between families is real at five objectives,\n"
          "where at two it would be noise.")


def main():
    show_lattice_sizes()
    compare_at(m=5, pop=126, n_gen=120)   # 126 = C(9,4), a lattice size for M=5


if __name__ == "__main__":
    main()
