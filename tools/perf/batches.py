#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""How every core asks for its evaluations: in batches, or one point at a time.

    python tools/perf/batches.py                    # every core, a Markdown table
    python tools/perf/batches.py --algorithms nsga2,moead_de --budget 3000

Each core runs on a cheap problem (DTLZ2 at three objectives, twelve
variables) through minimize(batch=...), whose batch function notes the size of
every call: the core hands it a whole batch where it evaluates one, and a
single row where it evaluates one point (the binding sends a single point
through the same function). The table says, per core, how many calls the run
made, the share of the evaluations that came in calls of one point, and the
usual size of the others. A batch can be spread over processes by the caller
(minimize's `batch`); single calls cannot. docs/running.md carries the table.
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))


def dtlz2(x, m=3):
    g = sum((v - 0.5) ** 2 for v in x[m - 1:])
    f = []
    for i in range(m):
        v = 1.0 + g
        for j in range(m - 1 - i):
            v *= math.cos(x[j] * math.pi / 2)
        if i > 0:
            v *= math.sin(x[m - 1 - i] * math.pi / 2)
        f.append(v)
    return f


def probe(name: str, budget: int, pop: int, seed: int = 1) -> dict:
    import mootation
    sizes = []

    def batch(X):
        sizes.append(len(X))
        return [dtlz2(list(x)) for x in X]

    p = pop
    if name in ("nsga3", "a_nsga3", "theta_dea", "mombi2", "srv_nsga3", "nrv_moea"):
        p = 91
    from mootation.run.campaign import fit_pop
    p, _ = fit_pop(name, p, 3, {})
    mootation.minimize(dtlz2, bounds=[(0.0, 1.0)] * 12, n_objs=3, algorithm=name, pop_size=p,
                       max_evaluations=budget, seed=seed, batch=batch)
    total = sum(sizes)
    single = sum(s for s in sizes if s == 1)
    big = [s for s in sizes if s > 1]
    return {"algorithm": name, "pop": p, "calls": len(sizes), "evaluations": total,
            "single_share": single / total if total else 0.0,
            "batch_median": statistics.median(big) if big else 0,
            "batch_max": max(big) if big else 0}


def main(argv=None) -> int:
    import mootation
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--algorithms", default="", help="comma-separated; default every core")
    ap.add_argument("--budget", type=int, default=3000)
    ap.add_argument("--pop", type=int, default=100)
    args = ap.parse_args(argv)
    names = args.algorithms.split(",") if args.algorithms else list(mootation.algorithms())
    print("| core | population | calls | evaluations | in single calls | batch size (median, max) |")
    print("|---|---|---|---|---|---|")
    for n in names:
        r = probe(n, args.budget, args.pop)
        kind = (f"{r['batch_median']:g}, {r['batch_max']}" if r["batch_max"] else "—")
        print(f"| {n} | {r['pop']} | {r['calls']} | {r['evaluations']} | "
              f"{100.0 * r['single_share']:.0f} % | {kind} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
