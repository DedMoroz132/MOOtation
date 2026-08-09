#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""The standard suites, and comparing algorithms on them.

    python python/examples/03_benchmarks.py

216 problems ship with the package — ZDT, DTLZ, WFG, MaF, Polygon, MOP, BT and
the inverted/scaled/minus DTLZ variants — each carrying the population size and
generation count its paper used, and where a closed form exists a sampler of
the true Pareto front. That last part is what makes IGD meaningful: without a
reference front you can only report that a run finished, not that it worked.

Needs NumPy (`pip install "mootation[bench]"`).
"""

import mootation.benchmarks as bench


def list_some():
    fams = bench.families()
    print(f"{sum(len(v) for v in fams.values())} problems in "
          f"{len(fams)} families:\n")
    for name, members in sorted(fams.items()):
        sizes = sorted({m.split('_')[-1] for m in members if '_' in m})
        print(f"  {name:<12} {len(members):>3} problems"
              + (f"   sizes: {', '.join(sizes)}" if sizes else ""))


def compare(problem="DTLZ2_3D", algorithms=("nsga2", "nsga3", "moead_de",
                                            "spea2", "ibea_eplus")):
    p = bench.get(problem)
    print(f"\n{problem}: M={p.n_obj}, n_vars={p.n_vars}, "
          f"paper budget pop={p.pop_size} x {p.n_gen} generations")
    print("running a short 100 generations each, seed 1\n")

    print(f"  {'algorithm':<14} {'solutions':>9}  {'IGD':>9}")
    rows = []
    for alg in algorithms:
        # NSGA-III and MOEA/D need pop to be an exact Das-Dennis lattice size.
        # 91 is one for three objectives; picking it for everyone keeps the
        # comparison honest, since population size is itself a variable.
        res = bench.solve(problem, alg, pop_size=91, n_gen=100, seed=1)
        score = bench.igd(problem, res.objectives)
        rows.append((alg, res.active_n, score))

    for alg, n, score in sorted(rows, key=lambda r: r[2]):
        print(f"  {alg:<14} {n:>9}  {score:>9.5f}")

    print("\nLower IGD is better. One seed on one problem for 100 generations "
          "ranks\nnothing — a real comparison needs 31 runs and a Wilcoxon "
          "test, which is\nwhat the K_runs field on each problem is for.")


def main():
    list_some()
    compare()


if __name__ == "__main__":
    main()
