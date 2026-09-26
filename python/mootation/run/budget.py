# SPDX-License-Identifier: Apache-2.0
"""Which algorithms behave differently under a different total budget.

A run's trajectory record at 10 000 evaluations is the same population as a
separate run of 10 000 — unless the algorithm schedules something by the share
of the budget spent: RVEA's APD penalty (t/t_max), the adaptation periods of
MOEA/D-AWA and AdaW, and so on. Those are budget-dependent: a budget ladder
needs a separate run per rung for them ([campaign] ladder), and the analysis
(--at, --cover) reads those runs instead of the trajectory.

Two lists, and how each was made:

  * BY_HEADERS — read off the code: the binding passes the budget to every core
    that has set_t_max (19 of them), and these use it in a schedule. Of the
    other two, IF-MaOEA stores t_max and never reads it, and CLIA uses it once,
    for theta = min(20, max(5, ceil(t_max*N / 2e4))) (§IV-B), which is 5 at
    every budget up to 100 000 evaluations.
  * BUDGET_DEPENDENT — measured by check(): each algorithm twice on one
    problem, seed and population, with max_evaluations `short` and `long`; the
    algorithm depends on the budget when some record of the answer set up to
    `short` evaluations differs, bit for bit, between the two runs. This is the
    list the campaign and the analysis use. `python -m mootation.run.budget`
    runs the check and says where the two lists disagree.

A run that ends exactly at its budget cuts its last step short (the sampling
baselines do; a generational core overshoots by less than a generation
instead), so its last record has spent fewer evaluations than the other run's
at the same step. That is not a schedule: the check stops comparing there and
says "same, cut".
"""

from __future__ import annotations

import argparse
import json
import sys
import time

BY_HEADERS = frozenset({
    "rvea", "rvea_star", "moead_awa", "adaw", "dea_gng", "mbra", "nrv_moea",
    "hlmea", "dhea", "moead_ds", "srv", "srv_nsga3", "dcea", "maoea_3c",
    "moead_m2m", "moead_am2m", "liu_gu2011",
})

# check() on DTLZ2_3D, seed 1, 10 000 against 25 000 evaluations, 2026-09-27,
# every algorithm and baseline: exactly the 17 above differ — MOEA/D-M2M first,
# at 180 evaluations, MOEA/D-AWA last, at 9 109 — and the other 46 and the four
# baselines are the same up to 10 000 (the baselines up to their last, cut step).
BUDGET_DEPENDENT = BY_HEADERS


def _records(name: str, problem: str, seed: int, budget: int, pop: int) -> list:
    """[(gen, fe, F)] of every record of one run: the answer set after setup and
    after every pop_size evaluations, and at the end."""
    import numpy as np
    from ..benchmarks import get as bench_get
    from .. import minimize
    from .archive import GridArchive
    from .baselines import BASELINES, run_baseline

    p = bench_get(problem)
    fe = 0
    recs = []
    arc = (GridArchive(p.n_obj, p.n_vars, ideal=p.ideal, nadir=p.nadir)
           if name in BASELINES else None)

    def evaluate(x):
        nonlocal fe
        fe += 1
        f = p.evaluate(x)
        if arc is not None:
            arc.add(f, x)
        return f

    def on_gen(gen, objectives):
        recs.append((int(gen), fe, np.asarray(objectives, float)))

    if name in BASELINES:
        run_baseline(name, evaluate, p.bounds, pop=pop, max_evaluations=budget, seed=seed,
                     archive=arc, on_generation=on_gen, record_every=1)
    else:
        minimize(evaluate, bounds=p.bounds, n_objs=p.n_obj, algorithm=name, pop_size=pop,
                 max_evaluations=budget, seed=seed, on_generation=on_gen, record_every=1)
    return recs


def _check_one(args) -> dict:
    name, problem, seed, short, long_, pop = args
    import numpy as np
    t0 = time.perf_counter()
    try:
        a = _records(name, problem, seed, short, pop)
        b = {g: (fe, F) for g, fe, F in _records(name, problem, seed, long_, pop)}
    except Exception as e:                                   # noqa: BLE001
        return {"algorithm": name, "error": f"{type(e).__name__}: {e}"}
    compared, first_diff, cut = 0, None, False
    for g, fe, F in a:
        if g not in b:
            continue
        fe_b, F_b = b[g]
        if fe != fe_b:
            # the short run cut its last step to end exactly at its budget (the
            # sampling baselines do): not a schedule, and nothing left to compare
            cut = True
            break
        compared += 1
        if F.shape != F_b.shape or not np.array_equal(F, F_b):
            first_diff = fe
            break
    return {"algorithm": name, "pop": pop, "compared": compared,
            "dependent": first_diff is not None, "first_diff_fe": first_diff,
            "cut_at_budget": cut, "last_fe": a[-1][1] if a else None,
            "by_headers": name in BY_HEADERS, "seconds": round(time.perf_counter() - t0, 2)}


def check(algorithms, *, problem: str = "DTLZ2_3D", seed: int = 1, short: int = 10_000,
          long: int = 25_000, workers: int = 1) -> list[dict]:
    """One dict per algorithm: `dependent` when a record up to `short`
    evaluations differs between the runs of `short` and `long` evaluations."""
    from ..benchmarks import get as bench_get
    from .campaign import fit_pop
    p = bench_get(problem)
    tasks = [(a, problem, seed, short, long, fit_pop(a, p.pop_size, p.n_obj, {})[0])
             for a in algorithms]
    if workers > 1 and len(tasks) > 1:
        import multiprocessing as mp
        with mp.Pool(processes=workers) as pool:
            out = list(pool.imap_unordered(_check_one, tasks))
    else:
        out = [_check_one(t) for t in tasks]
    order = {a: i for i, a in enumerate(algorithms)}
    return sorted(out, key=lambda r: order[r["algorithm"]])


def main(argv=None) -> int:
    from .algorithms import algorithm_names
    from .baselines import BASELINES
    ap = argparse.ArgumentParser(prog="mootation.run.budget",
                                 description="Measure which algorithms depend on the total budget.")
    ap.add_argument("--problem", default="DTLZ2_3D")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--short", type=int, default=10_000)
    ap.add_argument("--long", type=int, default=25_000)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--algorithms", help="comma-separated; default every algorithm and baseline")
    ap.add_argument("--json", help="write the per-algorithm results here")
    args = ap.parse_args(argv)
    algs = (args.algorithms.split(",") if args.algorithms
            else list(algorithm_names()) + list(BASELINES))
    res = check(algs, problem=args.problem, seed=args.seed, short=args.short,
                long=args.long, workers=args.workers)
    print(f"{args.problem}, seed {args.seed}: every record up to {args.short} evaluations "
          f"of a {args.short}- and a {args.long}-evaluation run")
    print(f"{'algorithm':<22}{'records':>8}{'result':>14}{'first diff':>12}{'headers':>9}{'s':>8}")
    for r in res:
        if "error" in r:
            print(f"{r['algorithm']:<22}  ERROR {r['error']}")
            continue
        same = "same, cut" if r.get("cut_at_budget") else "same"
        print(f"{r['algorithm']:<22}{r['compared']:>8}"
              f"{'dependent' if r['dependent'] else same:>14}"
              f"{r['first_diff_fe'] if r['first_diff_fe'] is not None else '-':>12}"
              f"{'yes' if r['by_headers'] else '':>9}{r['seconds']:>8}")
    measured = {r["algorithm"] for r in res if r.get("dependent")}
    checked = {r["algorithm"] for r in res if "error" not in r}
    only_measured = sorted(measured - BY_HEADERS)
    only_headers = sorted((BY_HEADERS & checked) - measured)
    print(f"\ndependent, measured: {len(measured)}: {', '.join(sorted(measured)) or '-'}")
    print(f"measured but not by the headers: {', '.join(only_measured) or 'none'}")
    print(f"by the headers but measured the same: {', '.join(only_headers) or 'none'}")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump({"problem": args.problem, "seed": args.seed, "short": args.short,
                       "long": args.long, "results": res,
                       "dependent": sorted(measured)}, fh, indent=1)
    return 0 if not any("error" in r for r in res) else 1


if __name__ == "__main__":
    sys.exit(main())
