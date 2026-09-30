# SPDX-License-Identifier: Apache-2.0
"""Which sets of algorithms cover the problems at a level of quality.

    python -m mootation.run.campaign camp.toml --cover
    python -m mootation.run.campaign camp.toml --cover --cover-taus 0.1,0.03 \\
        --cover-floor-taus 1,0.5 --cover-budgets 10000,25000 --cover-seeds 7 \\
        --cover-bootstrap 200

Reads finished runs; runs nothing.

COVERED. An algorithm covers a problem at a level and budget b when it reaches
the level in at least `min_seeds` of its seeds (7, meant for ten). The levels
go from coarse to fine, k = 0, 1, ...; level k:
  * on a problem with a reference front — criterion "igdp": igdp_norm ≤
    (1 + FLOOR_TAUS[k]) · floor, within 2, 1.5, 1.25 and 1.1 times the
    problem's floor; criterion "gdp": gdp_norm ≤ TAUS[k], the progress towards
    the front alone (GD+ in igdp_norm's frame, so that one τ reads alike on
    every problem);
  * on a problem without one (bbob-biobj), under both criteria: the relative
    gap of hv_h to the best value any run of the campaign reached on it,
    (best − hv_h)/best ≤ TAUS[k]. The best known values depend on what the
    campaign ran, so they are written out with the result (best_known_hv.csv).

THE FLOOR (owner and reviewer, 2026-09-28; the greedy floor 2026-09-29). N
points on the front itself leave IGD+ above zero, so absolute levels of
igdp_norm cannot all be reached: about 0.002 at two objectives, 0.02 at three
and 0.06 at five. A problem's floor is the IGD+, in igdp_norm's frame and
against the very reference front the runs were measured against, of the best N
points of that reference as far as a greedy search finds them: points added
one at a time, each the one that lowers IGD+ most, then single swaps while one
improves — a local optimum, an upper bound on the best N-subset. N is the
run's population: the problem's, or a multiple of K for the M2M-like cores
(campaign.fit_pop), whose floor is then its own. Beside it the IGD+ of the N
points DSS selects, "an ideal archive reduced by DSS", and the ratio of the
two, DSS's tax (on stage 3 1.05-3.7, median 1.39). The population scenario's
levels read the greedy floor; the archive scenario's the DSS floor, since that
answer is itself a DSS selection (on DTLZ2 the best archives stand at 0.97-0.99
of the DSS floor, 1.36-1.41 of the greedy one). 1.1 x floor (tau = 0.1) is the
ideal level: in stage 3 the min_seeds-th best seed reached it on 4 of 39 clean
regular problems; read it as the best attainable, not as a target. GD+ has a
floor of 0, so its levels stay absolute. The floors are written out
(igdp_floor.csv, with the reference version) and read back by the next
analysis of the same results.

KNOWN REFERENCE PROBLEMS (owner and reviewer, 2026-09-29). Where the reference
front itself is doubtful, no level says anything: DTLZ5, DTLZ6 and MaF6 from
four objectives (degenerate fronts the sample does not fill), WFG3 (its
degenerate front, incomplete in the sample at three objectives and more), DTLZ1
at five objectives (a fifth of the reference on the front's boundary); and,
measured against references of version 1 (benchmarks.registry,
REFERENCE_VERSION), IPolygon (polygons too close: most of the reference
dominated) and WFG1 and WFG2 where the grid of x1 aliases their last shape (at
n_ref = 1000: five objectives, WFG1 at eight, both at ten). A problem some of
whose runs were measured against an older reference than the current one, where
the reference changed since (REFERENCE_CHANGED), is set aside too, until
--recompute-reference brings them up to date. These problems are left out of
the coverage (cover_excluded.csv says which and why) unless keep_caveats is set,
and marked in floor_review.csv.

MARKED PROBLEMS (reviewer, 2026-09-29) stay in, and every table is written
twice, over all problems ("all") and over the unmarked ones ("unmarked"), to
see whether the conclusions hold without them: a reference of other than n_ref
rows or with repeated rows, the Das-Dennis lattice topped up with Dirichlet
points from four objectives (DTLZ1-4 and their variants, WFG4-9: at five
objectives the lattice has H = 6), and fronts of a lower dimension than M - 1
(properties.front_dim: DTLZ5/6 at three objectives, ZCAT14-16, Polygon from
four). cover_marks.csv lists them. The dimension also gives floor_review.csv's
eta = ratio^(-d), which reads alike across numbers of objectives.

WHAT IS WRITTEN. cover_meta.json: the commit of the analysis code (and whether
its tree differed), the scheme of the levels, the lists and the problems left
out; every CSV row says its level in words ("igdp_norm <= 2 x floor | hv_h gap
<= 0.1") and its subset, and cover_summary.csv the commit too, so that absolute
and floor-based levels are never confused. floor_review.csv: per problem with a
front, both floors at the problem's population and the level's own, beside the
best igdp_norm any run reached at the full budget, the best algorithm's median
over its seeds and the value that decides coverage (its min_seeds-th best
seed), as ratios to the level's floor: a ratio below 1 says the floor is too
high, far above 2 on an easy problem says it is too strict.

AT A BUDGET (read_at: the one reading of --cover, --portfolio,
--instance-space and the tables' --at). A run's value at b evaluations is its
final value when b is its budget, the budget asked for (budget_nominal: the
evaluations spent round it up to whole generations); for a budget-dependent
algorithm (budget.py), the final value of its ladder rung of nominal budget b;
for any other, the first trajectory record at or after b evaluations
([campaign] record_at puts one at every rung). A run that stopped before
spending b evaluations (DMS once its step is small enough: on BT6 and BT8 of
stage 3 after 630) is read at its end: its answer did not change after.

SCENARIO. With --scenario archive every value is the run archive reduced by
DSS instead of the population: final_archive at a run's own budget, the
archive checkpoint of budget b below it ([campaign] archive_checkpoints), and
a rung's final_archive for a budget-dependent algorithm. CSV under
<results>/_cover_archive/.

THE ANSWER'S SIZE (task 5, 2026-10-01). Beside the indicators every budget
reads how many points the answer has there ("n": n_final of a final
population, n of a trajectory record or a reduced archive). GD+ measures only
how close the answer's points are to the front, not how much of it they
cover: on stage 3 DMS reached the finest gdp level on BT6-BT8 with 2 points
of 100. So a gdp level counts only for an answer of at least N/2 points, N
the population the algorithm ran with; the igdp levels and the hv_h gap
penalise a short answer themselves. cover_short_answers.csv lists the runs
shorter than N/2 at every budget.

DETERMINISTIC ALGORITHMS (task 5, 2026-10-01). An algorithm whose seeds give
the same values on every problem at every budget (on stage 3 DMS alone,
which draws no random number) has its ten seeds as one run: "7 of 10" is
then "1 of 1" for it and "7 of 10" for everybody else. They are named in the
report and in cover_meta.json, and beside every level the sensitivity row
ONE RUN EACH (the fifth variant of report 06 of the stage-3 analysis) reads
every algorithm from one seed at a time — covered = the level in that seed — and
gives the mean over the seeds of what the main rule gives (cover_one_run.csv:
per seed, the smallest set and the problems each deterministic algorithm
covers alone).

FOR EVERY CRITERION, LEVEL AND BUDGET:
  * the problems no algorithm covers;
  * the smallest sets of algorithms covering every problem some algorithm
    covers: exact (branch and bound), all of them up to `max_sets`, with the
    greedy set beside them;
  * the curve k -> the largest share of those problems k algorithms cover,
    exact up to the smallest cover's size, with problems weighted alike and
    with families weighted alike (bbob-biobj is more than half the problems);
  * the coverage of every family along the curve;
  * with bootstrap > 0: how often each algorithm is in a smallest set when the
    seeds are drawn again with replacement.
Tables on the terminal, CSV under <results>/_cover/.
"""

from __future__ import annotations

import csv
import json
import math
import random
from pathlib import Path

TAUS = (0.1, 0.03, 0.01, 0.003)
FLOOR_TAUS = (1.0, 0.5, 0.25, 0.1)                       # igdp_norm <= (1 + tau) * floor
BUDGETS = (2500, 5000, 10000, 25000)
CRITERIA = {"igdp": "igdp_norm", "gdp": "gdp_norm"}      # criterion -> metric with a front
METRICS = ("igdp_norm", "gdp_norm", "hv_h", "eps_norm")  # read at every budget (portfolio.py too)


IDEAL_FLOOR_TAU = 0.1                                    # 1.1 x floor: the ideal level
# the reference fronts built on a Das-Dennis lattice topped up with Dirichlet
# points (dtlz_variants._simplex), coarse from four objectives (MARKED PROBLEMS)
LATTICE_FAMILIES = ("DTLZ1", "DTLZ2", "DTLZ3", "DTLZ4", "IDTLZ1", "IDTLZ2", "SDTLZ1", "SDTLZ2",
                    "shiftDTLZ1", "shiftDTLZ2", "shiftDTLZ3", "shiftDTLZ4", "WFG4", "WFG5",
                    "WFG6", "WFG7", "WFG8", "WFG9")


def reference_caveat(problem: str, version: int | None = None) -> str | None:
    """Why a problem's reference front is doubtful (KNOWN REFERENCE PROBLEMS),
    or None; `version` is the reference version the runs were measured
    against, the current one when None."""
    from ..benchmarks.registry import REFERENCE_VERSION
    version = REFERENCE_VERSION if version is None else version
    family, _, rest = problem.partition("_")
    m = int(rest[:-1]) if rest.endswith("D") and rest[:-1].isdigit() else None
    if family in ("DTLZ5", "DTLZ6", "MaF6") and m is not None and m >= 4:
        return "degenerate front; the reference sample does not fill it from four objectives"
    if family == "WFG3" and m is not None and m >= 3:
        return "degenerate WFG3 front; the reference sample is incomplete"
    if family == "DTLZ1" and m == 5:
        return "a fifth of the reference lies on the front's boundary"
    if version < 2 and family == "IPolygon":
        return "reference v1: the polygons are too close, most of the reference is dominated"
    if version < 2 and (family, m) in (("WFG1", 5), ("WFG1", 8), ("WFG1", 10), ("WFG2", 5),
                                       ("WFG2", 10)):
        return "reference v1: x1 on a grid where the last shape is linear"
    return None


def reference_marks(problem: str, rows: int | None, distinct: int | None,
                    n_ref: int) -> list:
    """Why a problem is marked (MARKED PROBLEMS); empty when it is not.
    `rows` and `distinct` count the reference front's rows."""
    from ..benchmarks import get as bench_get
    from ..benchmarks.properties import properties
    marks = []
    if rows is not None and rows != n_ref:
        marks.append(f"{rows} reference rows, not {n_ref}")
    if rows is not None and distinct is not None and distinct < rows:
        marks.append(f"{rows - distinct} repeated reference rows")
    m = bench_get(problem).n_obj
    if problem.partition("_")[0] in LATTICE_FAMILIES and m >= 4:
        marks.append("Das-Dennis lattice topped up with Dirichlet points, coarse from four "
                     "objectives")
    d = (properties(problem) or {}).get("front_dim")
    if d is not None and d < m - 1:
        marks.append(f"front of dimension {d}")
    return marks


NODE_LIMIT = 2_000_000


# ── reading the runs ────────────────────────────────────────────────────────


def nominal(row: dict) -> int:
    """The budget a run was asked for (budget_nominal; budget_fe for older runs)."""
    return int(row.get("budget_nominal") or row.get("budget_fe") or 0)


def needs_trajectory(b: int, budget: int, spent) -> bool:
    """Does read_at look into the trajectory at b: below the budget, and not
    after the run stopped?"""
    return b < budget and not (spent is not None and spent < b)


def answer_size(answer: dict):
    """How many points an answer has (THE ANSWER'S SIZE): n_final of a final
    population, n of a trajectory record or a reduced archive; None where the
    run did not record it."""
    v = answer.get("n_final", answer.get("n"))
    return None if v is None else int(v)


def read_at(b: int, budget: int, spent, answer: dict, traj=(), archive_at=None,
            metrics=METRICS) -> dict:
    """{metric: value} of a run at b evaluations (AT A BUDGET), and "n", the
    size of the answer there. budget: the run's nominal budget; spent: the
    evaluations it spent (meta.json "fe"); answer: its final values (of the
    population, or in the archive scenario of the reduced archive); archive_at:
    its archive checkpoints in the archive scenario, None otherwise. None past
    the budget, or where the run has no record at or after b below it."""
    if b > budget:
        return dict({m: None for m in metrics}, n=None)
    if not needs_trajectory(b, budget, spent):
        return dict({m: answer.get(m) for m in metrics}, n=answer_size(answer))
    if archive_at is not None:
        point = archive_at.get(str(b)) or {}
        return dict({m: point.get(m) for m in metrics}, n=answer_size(point))
    for rec in traj:
        if rec.get("fe", 0) >= b:
            return dict({m: rec.get(m) for m in metrics}, n=answer_size(rec))
    return dict({m: None for m in metrics}, n=None)


def _values_of_run(task) -> tuple:
    """(key, {budget: {metric: value}}) of one full-budget run, population."""
    key, run_dir, final, budget, spent, budgets = task
    from .campaign import read_trajectory
    traj = (read_trajectory(Path(run_dir))
            if any(needs_trajectory(b, budget, spent) for b in budgets) else ())
    return key, {b: read_at(b, budget, spent, final, traj) for b in budgets if b <= budget}


def _end(row: dict, scenario: str) -> dict:
    """A run's answer at its own budget: the population's indicators, or in the
    archive scenario the reduced archive's."""
    return (row.get("final_archive") or {}) if scenario == "archive" else row["final"]


def collect(root: Path, budgets=BUDGETS, workers: int = 1, scenario: str = "final") -> dict:
    """{(problem, algorithm, seed): {budget: {metric: value, "n": answer size}}}
    with the facts the analysis needs: which problems have a reference front,
    which algorithms are budget-dependent, and the rungs a campaign lacks."""
    from .budget import BUDGET_DEPENDENT
    from .campaign import scan_results
    rows = [r for r in scan_results(Path(root)) if r["status"] == "done"]
    rungs = {(r["problem"], r["ladder_of"], r["seed"], nominal(r)): r
             for r in rows if r.get("ladder_of")}
    main = [r for r in rows if not r.get("ladder_of")]
    tasks, values, missing = [], {}, []
    dependent = set()
    for r in main:
        key = (r["problem"], r["algorithm"], r["seed"])
        dep = r.get("budget_dependent")
        if dep is None:                                  # a campaign from before the flag
            dep = (r.get("core") or r["algorithm"]) in BUDGET_DEPENDENT
        # the budget asked for; budget_fe rounds it up to whole generations
        # (25 025 at a population of 91), which no budget of the list equals
        bfe = nominal(r)
        if dep:
            dependent.add(r["algorithm"])
            end = _end(r, scenario)
            values[key] = ({bfe: dict({m: end.get(m) for m in METRICS}, n=answer_size(end))}
                           if bfe in budgets else {})
            for b in budgets:
                if b < bfe:
                    rung = rungs.get((r["problem"], r["algorithm"], r["seed"], b))
                    if rung is None:
                        missing.append((r["problem"], r["algorithm"], r["seed"], b))
                    else:
                        end = _end(rung, scenario)
                        values[key][b] = dict({m: end.get(m) for m in METRICS},
                                              n=answer_size(end))
        elif scenario == "archive":
            # the run's own archive checkpoints below its budget
            values[key] = {b: read_at(b, bfe, r.get("fe"), _end(r, scenario),
                                      archive_at=r.get("archive_at") or {})
                           for b in budgets if b <= bfe}
        else:
            tasks.append((key, str(r["dir"]), r["final"], bfe, r.get("fe"), tuple(budgets)))
    if workers > 1 and len(tasks) > 1:
        import multiprocessing as mp
        from .campaign import single_threaded_blas
        single_threaded_blas()
        with mp.Pool(processes=workers) as pool:
            for key, v in pool.imap_unordered(_values_of_run, tasks, chunksize=32):
                values[key] = v
    else:
        for t in tasks:
            key, v = _values_of_run(t)
            values[key] = v
    # whether a problem has a reference front: the first run that says so, in
    # whatever order the file system lists them (the registry only where none does)
    front = {}
    for r in main:
        if front.get(r["problem"]) is None:
            front[r["problem"]] = r.get("has_reference_front")
    for p, has in list(front.items()):
        if has is None:                                  # older meta.json: ask the registry
            from ..benchmarks import get as bench_get
            front[p] = callable(bench_get(p).pareto_front)
    # the population each algorithm ran with (a floor per N) and the reference
    # versions each problem's runs were measured against (a meta.json without
    # them: the problem's population, version 1)
    pop, versions = {}, {}
    for r in main:
        if r.get("pop"):
            pop[(r["problem"], r["algorithm"])] = int(r["pop"])
        versions.setdefault(r["problem"], set()).add(int(r.get("reference_version") or 1))
    return {"values": values, "front": front, "dependent": dependent, "missing": missing,
            "main": main, "scenario": scenario, "pop": pop, "versions": versions}


FLOOR_SWEEPS = 3                                         # passes of single swaps at most


def _dplus_matrix(G):
    """D[c, z] = d+(z, c) = ||max(c - z, 0)|| between the rows of G: how far
    candidate c leaves reference point z uncovered; a block of rows at a time."""
    import numpy as np
    D = np.empty((len(G), len(G)))
    for a in range(0, len(G), 256):
        D[a:a + 256] = np.sqrt((np.maximum(G[a:a + 256, None, :] - G[None, :, :], 0.0) ** 2)
                               .sum(axis=2))
    return D


def greedy_subset(G, n: int, sweeps: int = FLOOR_SWEEPS) -> list:
    """Indices of n rows of the normalized reference G chosen for the smallest
    IGD+ against G itself (THE FLOOR): added one at a time, each the row that
    lowers IGD+ most, then single swaps while one improves, at most `sweeps`
    passes. Ties go to the lowest index, so the choice is deterministic."""
    import numpy as np
    D = _dplus_matrix(G)
    cur = np.full(len(G), np.inf)
    chosen: list = []
    for _ in range(min(n, len(G))):
        cost = np.minimum(cur[None, :], D).mean(axis=1)
        cost[chosen] = np.inf
        c = int(np.argmin(cost))
        chosen.append(c)
        cur = np.minimum(cur, D[c])
    for _ in range(sweeps):
        improved = False
        for i in range(len(chosen)):
            rest = chosen[:i] + chosen[i + 1:]
            base = D[rest].min(axis=0) if rest else np.full(len(G), np.inf)
            cost = np.minimum(base[None, :], D).mean(axis=1)
            cost[rest] = np.inf
            c = int(np.argmin(cost))
            if cost[c] < np.minimum(base, D[chosen[i]]).mean() - 1e-15:
                chosen[i] = c
                improved = True
        if not improved:
            break
    return chosen


def igdp_floors(problem: str, n_ref: int, sizes) -> dict | None:
    """The floors of a problem (THE FLOOR) at every population size N in
    `sizes`: {N: (greedy, dss)}, and under "rows" and "distinct" how many rows
    the reference front has and how many of them differ; None without a front."""
    import numpy as np
    from ..benchmarks import get as bench_get
    from .archive import dss_order
    from .metrics import _normalised, igd_plus
    p = bench_get(problem)
    if not callable(p.pareto_front):
        return None
    ref = np.asarray(p.pareto_front(n_ref), float)
    G, _ = _normalised(ref, ref, p.ideal, p.nadir)
    out: dict = {"rows": len(ref), "distinct": len(np.unique(np.round(ref, 12), axis=0))}
    for n in sorted(set(sizes)):
        dss = igd_plus(G[dss_order(ref, k=n, ideal=p.ideal, nadir=p.nadir)], G)
        greedy = igd_plus(G[greedy_subset(G, n)], G)
        out[n] = (greedy, dss)
    return out


def _median(values) -> float:
    v = sorted(values)
    return (v[(len(v) - 1) // 2] + v[len(v) // 2]) / 2 if v else math.nan


def _floor_task(task) -> tuple:
    problem, n_ref, sizes = task
    return problem, igdp_floors(problem, n_ref, sizes)


FLOOR_COLUMNS = ["problem", "pop", "n_ref", "reference_version", "floor_greedy", "floor_dss",
                 "dss_tax", "ref_rows", "ref_distinct"]


def floors(sizes: dict, n_ref: int, path: Path, workers: int = 1) -> dict:
    """{(problem, N): {"greedy", "dss", "rows", "distinct"}} for sizes =
    {problem: the population sizes its runs used}, over the problems with a
    reference front: from the CSV at `path` where it has them for this n_ref
    and the current reference version, computed and added otherwise (a
    reference front takes up to a minute to sample: ZCAT's are verified)."""
    from ..benchmarks import get as bench_get
    from ..benchmarks.registry import REFERENCE_VERSION
    have: dict = {}
    if path.is_file():
        with path.open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                if (row.get("floor_greedy") and int(row["n_ref"]) == n_ref
                        and int(row.get("reference_version") or 1) == REFERENCE_VERSION):
                    have[(row["problem"], int(row["pop"]))] = {
                        "greedy": float(row["floor_greedy"]), "dss": float(row["floor_dss"]),
                        "rows": int(row["ref_rows"]), "distinct": int(row["ref_distinct"])}
    todo = []
    for prob in sorted(sizes):
        missing = sorted(n for n in sizes[prob] if (prob, n) not in have)
        if missing and callable(bench_get(prob).pareto_front):
            todo.append((prob, n_ref, tuple(missing)))
    if workers > 1 and len(todo) > 1:
        import multiprocessing as mp
        from .campaign import single_threaded_blas
        single_threaded_blas()
        with mp.Pool(processes=workers) as pool:
            done = list(pool.imap_unordered(_floor_task, todo))
    else:
        done = [_floor_task(t) for t in todo]
    for prob, res in done:
        if res is not None:
            for n in (k for k in res if isinstance(k, int)):
                have[(prob, n)] = {"greedy": res[n][0], "dss": res[n][1],
                                   "rows": res["rows"], "distinct": res["distinct"]}
    if done:
        with path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(FLOOR_COLUMNS)
            for (prob, n) in sorted(have):
                f = have[(prob, n)]
                tax = f["dss"] / f["greedy"] if f["greedy"] > 0 else math.nan
                w.writerow([prob, n, n_ref, REFERENCE_VERSION, f"{f['greedy']:.10g}",
                            f"{f['dss']:.10g}", f"{tax:.6g}", f["rows"], f["distinct"]])
    return have


def level_floor(scenario: str) -> str:
    """Which floor a scenario's levels read (THE FLOOR): "greedy" for the
    population, "dss" for the archive, itself a DSS selection."""
    return "dss" if scenario == "archive" else "greedy"


def floor_review(data: dict, fl: dict, min_seeds: int, caveats: dict | None = None,
                 marks: dict | None = None) -> list:
    """Rows of floor_review.csv (WHAT IS WRITTEN): per problem with a floor at its
    population, both floors and DSS's tax; the best igdp_norm of any full-budget
    run, the best algorithm's median over its seeds and the smallest
    min_seeds-th best seed over the algorithms (what a level has to admit for
    somebody to cover the problem), each beside its ratio to the floor the
    scenario's levels read, and the best and deciding ratios as eta =
    ratio^(-d), d the front's dimension."""
    from ..benchmarks import get as bench_get
    from ..benchmarks.properties import properties
    scenario = data.get("scenario", "final")
    which = level_floor(scenario)
    caveats, marks = caveats or {}, marks or {}
    at_pop = {prob: f for (prob, n), f in fl.items() if n == bench_get(prob).pop_size}
    runs: dict = {}                                     # problem -> algorithm -> [(v, seed, n)]
    for r in data["main"]:
        if r["problem"] not in at_pop:
            continue
        end = _end(r, scenario)
        v = end.get("igdp_norm")
        if v is None or not math.isfinite(float(v)):
            continue
        runs.setdefault(r["problem"], {}).setdefault(r["algorithm"], []).append(
            (float(v), r["seed"], end.get("n_final") or end.get("n")))
    rows = []
    for prob in sorted(at_pop):
        f = at_pop[prob][which]
        by_alg = runs.get(prob, {})
        p = bench_get(prob)
        d = (properties(prob) or {}).get("front_dim")
        g, s = at_pop[prob]["greedy"], at_pop[prob]["dss"]
        row = {"problem": prob, "n_obj": p.n_obj, "pop": p.pop_size, "floor_greedy": g,
               "floor_dss": s, "dss_tax": s / g if g > 0 else math.nan, "level_floor": which,
               "floor": f, "front_dim": d, "caveat": caveats.get(prob, ""),
               "marks": "; ".join(marks.get(prob, []))}
        if by_alg and f > 0:
            best = min((v, a, s_, n) for a, vs in by_alg.items() for v, s_, n in vs)
            med = min((_median([v for v, _, _ in vs]), a) for a, vs in by_alg.items())
            kth = [(sorted(v for v, _, _ in vs)[min_seeds - 1], a) for a, vs in by_alg.items()
                   if len(vs) >= min_seeds]
            row.update(best=best[0], best_ratio=best[0] / f, best_algorithm=best[1],
                       best_seed=best[2], best_n=best[3],
                       median=med[0], median_ratio=med[0] / f, median_algorithm=med[1])
            if d:
                row["best_eta"] = row["best_ratio"] ** (-d)
            if kth:
                k = min(kth)
                row.update(kth=k[0], kth_ratio=k[0] / f, kth_algorithm=k[1])
                if d:
                    row["kth_eta"] = row["kth_ratio"] ** (-d)
        rows.append(row)
    return rows


def best_known_hv(data: dict) -> dict:
    """{problem: (best final hv_h, algorithm, seed)} over the full-budget runs."""
    best = {}
    for r in data["main"]:
        v = _end(r, data.get("scenario", "final")).get("hv_h")
        if v is None or not math.isfinite(float(v)):
            continue
        if r["problem"] not in best or float(v) > best[r["problem"]][0]:
            best[r["problem"]] = (float(v), r["algorithm"], r["seed"])
    return best


def reached(data: dict, criterion: str, level: tuple, budget: int, best_hv: dict,
            floor: dict | None = None) -> dict:
    """{problem: {algorithm: {seed: bool}}}: did the run reach the level at budget.
    `level` = (tau, tau_hv): igdp_norm <= (1 + tau) * floor, or gdp_norm <= tau
    with an answer of at least N/2 points (THE ANSWER'S SIZE), where the
    problem has a reference front; the hv_h gap <= tau_hv where not.
    `floor`: {(problem, N): floor}, N the population the algorithm ran with
    (data["pop"]; the problem's own where the run did not say)."""
    from ..benchmarks import get as bench_get
    metric = CRITERIA[criterion]
    tau, tau_hv = level
    pop = data.get("pop", {})
    own: dict = {}                                       # problem -> its population
    out: dict = {}
    for (prob, alg, seed), by_b in data["values"].items():
        v = by_b.get(budget, {})
        ok = False
        if data["front"].get(prob):
            x = v.get(metric)
            n = pop.get((prob, alg))
            if n is None:
                n = own.setdefault(prob, bench_get(prob).pop_size)
            limit = tau if criterion == "gdp" else (1.0 + tau) * (floor or {}).get((prob, n),
                                                                                  math.nan)
            ok = x is not None and math.isfinite(float(x)) and float(x) <= limit
            if criterion == "gdp" and v.get("n") is not None and v["n"] < n / 2:
                ok = False                               # GD+ does not see a short answer
        else:
            x, best = v.get("hv_h"), best_hv.get(prob, (0.0,))[0]
            ok = (x is not None and best > 0.0 and math.isfinite(float(x))
                  and (best - float(x)) / best <= tau_hv)
        out.setdefault(prob, {}).setdefault(alg, {})[seed] = ok
    return out


def successes(hit: dict, min_seeds: int, seeds=None) -> dict:
    """{algorithm: set of problems it covers}: the level in >= min_seeds seeds.
    `seeds` (a list, repeats allowed) reads those seeds instead of all."""
    cov: dict = {}
    for prob, algs in hit.items():
        for alg, by_seed in algs.items():
            n = (sum(by_seed.get(s, False) for s in seeds) if seeds is not None
                 else sum(by_seed.values()))
            cov.setdefault(alg, set())
            if n >= min_seeds:
                cov[alg].add(prob)
    return cov


def deterministic(values: dict) -> list:
    """The algorithms whose seeds are one run (DETERMINISTIC ALGORITHMS): on
    every problem where they ran more than one seed, every seed's values alike
    at every budget, the answer's size included."""
    runs: dict = {}                                  # (problem, algorithm) -> [seeds, values]
    for (prob, alg, _), by_b in values.items():
        seen = runs.setdefault((prob, alg), [0, set()])
        seen[0] += 1
        seen[1].add(json.dumps(by_b, sort_keys=True))
    alike: dict = {}
    for (prob, alg), (n_seeds, distinct) in runs.items():
        if n_seeds > 1:
            alike[alg] = alike.get(alg, True) and len(distinct) == 1
    return sorted(a for a, same in alike.items() if same)


def short_answers(data: dict, budgets) -> list:
    """[(problem, algorithm, budget, N, seeds, short seeds, median size)] of every
    problem with a front and budget where some run's answer has fewer than N/2
    points (THE ANSWER'S SIZE), N the population the algorithm ran with."""
    from ..benchmarks import get as bench_get
    sizes: dict = {}
    for (prob, alg, _), by_b in data["values"].items():
        if not data["front"].get(prob):
            continue
        for b in budgets:
            n = by_b.get(b, {}).get("n")
            if n is not None:
                sizes.setdefault((prob, alg, b), []).append(n)
    out = []
    for (prob, alg, b), ns in sorted(sizes.items()):
        pop = data.get("pop", {}).get((prob, alg)) or bench_get(prob).pop_size
        short = sum(1 for n in ns if n < pop / 2)
        if short:
            out.append((prob, alg, b, pop, len(ns), short, _median(ns)))
    return out


def _alone(sets: dict, a: str) -> int:
    """How many problems of the bit masks `sets` algorithm a covers alone."""
    others = 0
    for b, m in sets.items():
        if b != a:
            others |= m
    return (sets.get(a, 0) & ~others).bit_count()


def one_run(hit: dict, problems: list, seeds: list, watch=()) -> list:
    """ONE RUN EACH: per seed s, every algorithm covering what it reaches in s
    ("1 of 1"): [(s, covered by someone, smallest size, exact, the first
    smallest set, {a: problems a covers alone} for a in watch)]."""
    out = []
    for s in seeds:
        cov = successes(hit, 1, seeds=[s])
        sets = _masks(cov, problems)
        universe = 0
        for m in sets.values():
            universe |= m
        size, optimal, exact = smallest_covers(sets, universe, max_sets=1)
        out.append((s, universe.bit_count(), size, exact, optimal[0],
                    {a: _alone(sets, a) for a in watch}))
    return out


# ── set cover on bit masks ──────────────────────────────────────────────────


def _masks(cov: dict, problems: list) -> dict:
    idx = {p: i for i, p in enumerate(problems)}
    return {a: sum(1 << idx[p] for p in ps) for a, ps in cov.items()}


def greedy_cover(sets: dict, universe: int) -> list:
    """Repeatedly the set covering most of what is left (ties: the name)."""
    chosen, covered = [], 0
    while covered & universe != universe:
        a = max(sorted(sets), key=lambda s: (sets[s] & universe & ~covered).bit_count())
        if not (sets[a] & universe & ~covered):
            break
        chosen.append(a)
        covered |= sets[a]
    return chosen


def smallest_covers(sets: dict, universe: int, max_sets: int = 20,
                    node_limit: int = NODE_LIMIT) -> tuple:
    """(size, [sorted name tuples], exact): every smallest set of `sets` whose
    union contains `universe`, up to max_sets of them. Branches on the uncovered
    problem the fewest sets cover, bounded by the largest set still useful;
    `exact` is False when the node budget ran out (the greedy size is then an
    upper bound, not the minimum)."""
    names = sorted(sets)
    greedy = greedy_cover(sets, universe)
    if not universe:
        return 0, [()], True
    covering = {}
    bit, u = 0, universe
    while u:
        if u & 1:
            covering[bit] = [a for a in names if sets[a] >> bit & 1]
        u >>= 1
        bit += 1
    nodes = 0
    for size in range(1, len(greedy) + 1):
        found: set = set()
        exhausted = False

        def dfs(depth, covered, chosen):
            nonlocal nodes, exhausted
            nodes += 1
            if nodes > node_limit:
                exhausted = True
                return
            left = universe & ~covered
            if not left:
                found.add(tuple(sorted(chosen)))
                return
            if depth == size or len(found) >= max_sets:
                return
            gain = max((sets[a] & left).bit_count() for a in names)
            if gain * (size - depth) < left.bit_count():
                return
            e = min((b for b in covering if left >> b & 1), key=lambda b: len(covering[b]))
            for a in covering[e]:
                if a not in chosen:
                    dfs(depth + 1, covered | sets[a], chosen + [a])
                    if exhausted or len(found) >= max_sets:
                        return

        dfs(0, 0, [])
        if exhausted:
            return len(greedy), [tuple(sorted(greedy))], False
        if found:
            return size, sorted(found)[:max_sets], True
    return len(greedy), [tuple(sorted(greedy))], True


def _weigher(problems: list, weights: list):
    """w(mask) = sum of the weights of its problems, by family masks for speed."""
    groups: dict = {}
    for i, w in enumerate(weights):
        groups.setdefault(w, 0)
        groups[w] |= 1 << i
    items = list(groups.items())
    return lambda mask: sum(w * (mask & g).bit_count() for w, g in items)


CURVE_NODE_LIMIT = 50_000


def best_k(sets: dict, universe: int, k: int, weigh,
           node_limit: int = CURVE_NODE_LIMIT) -> tuple:
    """(value, names, exact): the k sets whose union weighs most within universe.
    Branch and bound over combinations, starting from the greedy k; the bound
    adds the k - depth largest marginal gains still available. `exact` is
    False when the node budget ran out: the value is then the best found, at
    least the greedy one."""
    names = sorted(sets, key=lambda a: (-weigh(sets[a] & universe), a))
    greedy, covered = [], 0
    for _ in range(k):
        a = max(names, key=lambda s: (weigh(sets[s] & universe & ~covered), -names.index(s)))
        greedy.append(a)
        covered |= sets[a]
    best = [weigh(covered & universe), list(greedy)]
    nodes = 0
    exhausted = False

    def dfs(start, depth, covered, chosen):
        nonlocal nodes, exhausted
        nodes += 1
        if nodes > node_limit:
            exhausted = True
            return
        value = weigh(covered & universe)
        if depth == k:
            if value > best[0] + 1e-12:
                best[0], best[1] = value, list(chosen)
            return
        gains = sorted((weigh(sets[a] & universe & ~covered) for a in names[start:]),
                       reverse=True)
        if value + sum(gains[:k - depth]) <= best[0] + 1e-12:
            return
        for i in range(start, len(names)):
            if len(names) - i < k - depth:
                break
            dfs(i + 1, depth + 1, covered | sets[names[i]], chosen + [names[i]])
            if exhausted:
                return

    dfs(0, 0, 0, [])
    return best[0], sorted(best[1]), not exhausted


# ── one analysis ────────────────────────────────────────────────────────────

_CURVE_CACHE: dict = {}


def analyse(cov: dict, problems: list, family_of, max_sets: int = 20) -> dict:
    """Everything item 5 asks of one criterion, level and budget."""
    if not problems or not cov:                    # nothing to cover, or nobody to cover it
        return {"n_problems": len(problems), "n_covered": 0, "nobody": list(problems),
                "size": 0, "optimal": [()], "exact": True, "greedy": [], "curve": [],
                "families": {}}
    sets = _masks(cov, problems)
    universe = 0
    for m in sets.values():
        universe |= m
    nobody = [p for i, p in enumerate(problems) if not universe >> i & 1]
    size, optimal, exact = smallest_covers(sets, universe, max_sets=max_sets)
    greedy = greedy_cover(sets, universe)
    fams = sorted({family_of(p) for p in problems})
    per_fam = {f: sum(1 for p in problems if family_of(p) == f) for f in fams}
    alike = _weigher(problems, [1.0] * len(problems))
    by_family = _weigher(problems, [1.0 / (len(fams) * per_fam[family_of(p)]) for p in problems])
    n_cov = universe.bit_count()
    total_fam = by_family(universe)
    frozen = tuple(sorted(sets.items()))
    curve = []
    for k in range(1, max(size, 1) + 1):
        # the same success sets recur across levels and budgets: solve each once
        for tag, weigh in (("alike", alike), ("families", by_family)):
            if (frozen, universe, k, tag) not in _CURVE_CACHE:
                _CURVE_CACHE[(frozen, universe, k, tag)] = best_k(sets, universe, k, weigh)
        v1, s1, e1 = _CURVE_CACHE[(frozen, universe, k, "alike")]
        v2, s2, e2 = _CURVE_CACHE[(frozen, universe, k, "families")]
        fam_cov = {}
        for f in fams:
            mask = sum(1 << i for i, p in enumerate(problems) if family_of(p) == f)
            got = 0
            for a in s1:
                got |= sets[a]
            fam_cov[f] = (got & mask).bit_count()
        curve.append({"k": k, "share": v1 / n_cov if n_cov else 0.0, "set": s1, "exact": e1,
                      "share_families": v2 / total_fam if total_fam else 0.0,
                      "set_families": s2, "exact_families": e2, "per_family": fam_cov})
    return {"n_problems": len(problems), "n_covered": n_cov, "nobody": nobody,
            "size": size, "optimal": optimal, "exact": exact, "greedy": greedy,
            "curve": curve, "families": per_fam}


def bootstrap(hit: dict, problems: list, seeds: list, min_seeds: int, replicates: int,
              rng_seed: int = 20260927) -> dict:
    """{algorithm: share of replicates in which it is in the (first) smallest cover}."""
    rng = random.Random(rng_seed)
    counts: dict = {}
    for _ in range(replicates):
        draw = [rng.choice(seeds) for _ in seeds]
        cov = successes(hit, min_seeds, seeds=draw)
        sets = _masks(cov, problems)
        universe = 0
        for m in sets.values():
            universe |= m
        _, optimal, _ = smallest_covers(sets, universe, max_sets=1)
        for a in optimal[0]:
            counts[a] = counts.get(a, 0) + 1
    return {a: c / replicates for a, c in sorted(counts.items(), key=lambda t: -t[1])}


# ── the report ──────────────────────────────────────────────────────────────


def prepare(data: dict, n_ref: int, floor_path: Path, workers: int = 1,
            keep_caveats: bool = False) -> dict:
    """What every analysis of the coverage needs beside the values (collect):
    the problems with a front, their floors at every population the runs used
    (floors(), cached at floor_path), which floor the scenario's levels read
    and {(problem, N): that floor}, the problems with a doubtful or outdated
    reference (KNOWN REFERENCE PROBLEMS) and those left out, and the marked
    ones (MARKED PROBLEMS)."""
    from ..benchmarks import get as bench_get
    from ..benchmarks.registry import REFERENCE_CHANGED, REFERENCE_VERSION
    problems = sorted({k[0] for k in data["values"]})
    with_front = [p for p in problems if data["front"].get(p)]
    sizes = {p: {bench_get(p).pop_size} for p in with_front}
    for (p, _), n in data["pop"].items():
        if p in sizes:
            sizes[p].add(n)
    fl = floors(sizes, n_ref, floor_path, workers=workers)
    which = level_floor(data.get("scenario", "final"))
    caveats = {}
    for p in problems:
        oldest = min(data["versions"].get(p, {REFERENCE_VERSION}))
        why = reference_caveat(p, oldest)
        stale = [v for v in REFERENCE_CHANGED if v > oldest and p in REFERENCE_CHANGED[v]]
        if stale and not why:
            why = (f"runs measured against reference v{oldest}, changed in v{max(stale)}: "
                   f"--recompute-reference")
        if why:
            caveats[p] = why
    marks = {}
    for p in with_front:
        f = fl.get((p, bench_get(p).pop_size))
        found = reference_marks(p, f and f["rows"], f and f["distinct"], n_ref)
        if found:
            marks[p] = found
    return {"problems": problems, "with_front": with_front, "fl": fl, "which": which,
            "floor": {key: f[which] for key, f in fl.items()}, "caveats": caveats,
            "excluded": {} if keep_caveats else caveats, "marks": marks}


def run(root: Path, *, taus=TAUS, floor_taus=FLOOR_TAUS, budgets=BUDGETS, min_seeds: int = 7,
        max_sets: int = 20, replicates: int = 0, workers: int = 1,
        out_dir: Path | None = None, scenario: str = "final", n_ref: int = 1000,
        keep_caveats: bool = False) -> str:
    """The whole analysis: text for the terminal; CSV files in out_dir. Level k
    is floor_taus[k] for igdp, taus[k] for gdp, taus[k] for the hv_h gap."""
    if len(floor_taus) != len(taus):
        raise ValueError(f"{len(floor_taus)} floor levels against {len(taus)} levels: "
                         f"level k pairs floor_taus[k] with taus[k]")
    from .campaign import problem_family
    root = Path(root)
    out_dir = (Path(out_dir) if out_dir else
               root / ("_cover_archive" if scenario == "archive" else "_cover"))
    out_dir.mkdir(parents=True, exist_ok=True)
    data = collect(root, budgets, workers=workers, scenario=scenario)
    problems = sorted({k[0] for k in data["values"]})
    seeds = sorted({k[2] for k in data["values"]})
    best_hv = best_known_hv(data)
    lines = [f"cover: {len(problems)} problems, {len({k[1] for k in data['values']})} "
             f"algorithms, seeds {seeds[0]}-{seeds[-1]} ({len(seeds)}); covered = the level "
             f"in >= {min_seeds} seeds" if problems else "cover: no finished runs"]
    if not problems:
        return lines[0]
    if data["missing"]:
        lines.append(f"  {len(data['missing'])} ladder rungs missing (budget-dependent runs "
                     f"read as not reaching any level there), e.g. {data['missing'][0]}")
    fixed = deterministic(data["values"])
    if fixed:
        lines.append(f"  deterministic, every seed the same run on every problem: "
                     f"{', '.join(fixed)}; 'one run each' reads every algorithm from one seed "
                     f"at a time (cover_one_run.csv)")
    from ..benchmarks import get as bench_get
    from ..benchmarks.registry import REFERENCE_VERSION
    prep = prepare(data, n_ref, out_dir / "igdp_floor.csv", workers=workers,
                   keep_caveats=keep_caveats)
    with_front, fl, which, floor = prep["with_front"], prep["fl"], prep["which"], prep["floor"]
    caveats, excluded, marks = prep["caveats"], prep["excluded"], prep["marks"]
    at_pop = [fl[(p, bench_get(p).pop_size)] for p in with_front
              if (p, bench_get(p).pop_size) in fl]
    lines.append(f"  igdp levels from each problem's {which} floor (igdp_floor.csv): median "
                 f"{_median([f[which] for f in at_pop]):.4g}; DSS's tax median "
                 f"{_median([f['dss'] / f['greedy'] for f in at_pop if f['greedy'] > 0]):.3g}"
                 if at_pop else "  no problem with a front")
    # what the levels are and which code drew them (WHAT IS WRITTEN)
    import time
    from .. import __version__
    from .provenance import revision
    rev = revision()
    commit = (rev.get("commit") or "unknown")[:12] + ("+dirty" if rev.get("dirty") else "")
    meta = {
        "analysis": {"mootation": __version__, "commit": rev.get("commit"),
                     "dirty": rev.get("dirty"), "source": rev.get("source")},
        "results": str(root), "scenario": scenario,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "levels": {
            "igdp": "igdp_norm <= (1 + tau) x floor, tau in floor_taus; floor = IGD+ in "
                    "igdp_norm's frame, against the n_ref-point reference front, of N points "
                    "of that reference: chosen greedily for the smallest IGD+ and improved "
                    "by swaps (floor_method greedy, scenario final) or selected by DSS "
                    "(floor_method dss, scenario archive, whose answer is a DSS selection); "
                    "N = the population the algorithm ran with",
            "ideal": f"igdp_norm <= {1 + IDEAL_FLOOR_TAU:g} x floor is the ideal level: read it "
                     f"as the best attainable, not as a target",
            "gdp": "gdp_norm <= tau, tau in taus (absolute: GD+ has a floor of 0), with an "
                   "answer of at least N/2 points (GD+ does not see a short answer)",
            "hv_h": "problems without a reference front, both criteria: (best - hv_h) / best "
                    "<= tau, tau in taus (absolute); best = the campaign's best final hv_h",
            "level_k": "level k pairs floor_taus[k] (igdp) or taus[k] (gdp) with taus[k] (hv_h)",
        },
        "floor_method": which, "reference_version": REFERENCE_VERSION,
        "subsets": {"all": "every problem not left out",
                    "unmarked": "without the marked problems (cover_marks.csv)"},
        "taus": list(taus), "floor_taus": list(floor_taus), "budgets": list(budgets),
        "min_seeds": min_seeds, "n_ref": n_ref, "keep_caveats": keep_caveats,
        "excluded": excluded, "marked": {p: "; ".join(m) for p, m in sorted(marks.items())},
        "floor_file": "igdp_floor.csv",
        "deterministic": fixed,
        "one_run_each": "cover_one_run.csv: every algorithm read from one seed at a time "
                        "(covered = the level in that seed), per seed",
        "short_answers": "cover_short_answers.csv: runs whose answer has fewer than N/2 points",
    }
    (out_dir / "cover_meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    with (out_dir / "cover_excluded.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["problem", "reason", "excluded"])
        for p_ in sorted(caveats):
            w.writerow([p_, caveats[p_], p_ in excluded])
    with (out_dir / "cover_marks.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["problem", "marks"])
        for p_ in sorted(marks):
            w.writerow([p_, "; ".join(marks[p_])])
    if caveats:
        lines.append(f"  {len(caveats)} problems with a doubtful reference front "
                     f"(cover_excluded.csv): {'kept' if keep_caveats else 'left out'}: "
                     f"{', '.join(sorted(caveats))}")
    if marks:
        lines.append(f"  {len(marks)} problems marked (cover_marks.csv): every table twice, "
                     f"with them and without")
    review = floor_review(data, fl, min_seeds, caveats, marks)
    review_cols = ["problem", "n_obj", "pop", "floor_greedy", "floor_dss", "dss_tax",
                   "level_floor", "floor", "best", "best_ratio", "best_algorithm", "best_seed",
                   "best_n", "median", "median_ratio", "median_algorithm", "kth", "kth_ratio",
                   "kth_algorithm", "front_dim", "best_eta", "kth_eta", "caveat", "marks"]
    with (out_dir / "floor_review.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(review_cols)
        for r_ in review:
            w.writerow([f"{r_[c]:.6g}" if isinstance(r_.get(c), float) else r_.get(c, "")
                        for c in review_cols])
    ratios = [r_["best_ratio"] for r_ in review if "best_ratio" in r_ and not r_["caveat"]]
    deciding = [r_["kth_ratio"] for r_ in review if "kth_ratio" in r_ and not r_["caveat"]]
    if ratios:
        lines.append(f"  best igdp_norm / {which} floor (floor_review.csv): median "
                     f"{_median(ratios):.3g}; below 1 on {sum(v < 1 for v in ratios)}, above 2 on "
                     f"{sum(v > 2 for v in ratios)} of {len(ratios)} problems; the deciding "
                     f"{min_seeds}th seed below 1 on {sum(v < 1 for v in deciding)}")
    kept = [p_ for p_ in problems if p_ not in excluded]
    kept_set = set(kept)
    subsets = {"all": kept, "unmarked": [p_ for p_ in kept if p_ not in marks]}
    no_front = [p for p in problems if not data["front"].get(p)]
    if no_front:
        lines.append(f"  {len(no_front)} problems without a reference front: the relative gap "
                     f"of hv_h to the campaign's best (best_known_hv.csv)")
    with (out_dir / "best_known_hv.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["problem", "best_hv_h", "algorithm", "seed"])
        for p in sorted(best_hv):
            w.writerow([p, f"{best_hv[p][0]:.10g}", best_hv[p][1], best_hv[p][2]])
    short = short_answers(data, budgets)
    with (out_dir / "cover_short_answers.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["problem", "algorithm", "budget", "pop", "seeds", "seeds_short",
                    "median_size"])
        w.writerows([p, a, b, n, k, s, f"{m:g}"] for p, a, b, n, k, s, m in short)
    at_full = [r for r in short if r[2] == max(budgets) and r[0] in kept_set]
    if at_full:
        lines.append(f"  answers of fewer than N/2 points at {max(budgets)}, which no gdp level "
                     f"counts: {len(at_full)} problem-algorithm pairs, "
                     f"{len({r[0] for r in at_full})} problems, "
                     f"{len({r[1] for r in at_full})} algorithms (cover_short_answers.csv)")

    summary, curves, uncovered, boot, single = [], [], [], [], []
    levels = {"igdp": list(zip(floor_taus, taus)), "gdp": list(zip(taus, taus))}
    for crit in CRITERIA:
        for tau, tau_hv in levels[crit]:
            ideal = crit == "igdp" and abs(tau - IDEAL_FLOOR_TAU) < 1e-12
            name = (f"igdp <= {1 + tau:g} x floor{' (ideal)' if ideal else ''}" if crit == "igdp"
                    else f"gdp <= {tau:g}")
            level = (f"igdp_norm <= {1 + tau:g} x {which} floor{' (ideal)' if ideal else ''}"
                     if crit == "igdp" else f"gdp_norm <= {tau:g}") + f" | hv_h gap <= {tau_hv:g}"
            for b in budgets:
                hit_all = {p_: v for p_, v in reached(data, crit, (tau, tau_hv), b, best_hv,
                                                        floor).items() if p_ in kept_set}
                for subset, members in subsets.items():
                    member_set = set(members)
                    hit = {p_: v for p_, v in hit_all.items() if p_ in member_set}
                    cov = successes(hit, min_seeds)
                    res = analyse(cov, members, problem_family, max_sets=max_sets)
                    opt = ["+".join(s) for s in res["optimal"]]
                    head = [crit, tau, tau_hv, level, subset, b]
                    summary.append(head + [res["n_problems"], res["n_covered"],
                                           len(res["nobody"]), res["size"], len(opt), res["exact"],
                                           " | ".join(opt), "+".join(res["greedy"]), commit])
                    for c in res["curve"]:
                        curves.append(head + [c["k"], f"{c['share']:.6g}", "+".join(c["set"]),
                                              c["exact"], f"{c['share_families']:.6g}",
                                              "+".join(c["set_families"]), c["exact_families"],
                                              json.dumps(c["per_family"], sort_keys=True)])
                    for p in res["nobody"]:
                        uncovered.append(head + [p, problem_family(p)])
                    if subset == "all":
                        lines.append(
                            f"{name} (hv_h gap <= {tau_hv:g}) at {b}: {res['n_covered']}/"
                            f"{res['n_problems']} covered by someone; smallest set "
                            f"{res['size']}{'' if res['exact'] else ' (greedy, bound)'}"
                            f" x{len(opt)}{'+' if len(opt) >= max_sets else ''}: "
                            f"{opt[0] if opt else '-'}; greedy {len(res['greedy'])}")
                    else:
                        lines.append(f"    unmarked: {res['n_covered']}/{res['n_problems']}; "
                                     f"smallest set {res['size']}: {opt[0] if opt else '-'}")
                    if fixed:
                        per_seed = one_run(hit, members, seeds, fixed)
                        for s, n_cov, size, exact, first, alone in per_seed:
                            single.append(head + [s, len(members), n_cov, size, exact,
                                                  "+".join(first), json.dumps(alone)])
                        if subset == "all":
                            covs = [r_[1] for r_ in per_seed]
                            sizes = [r_[2] for r_ in per_seed]
                            masks = _masks(cov, members)
                            lines.append(
                                f"    one run each ({len(seeds)} seeds, each alone): covered "
                                f"{sum(covs) / len(covs):.1f}/{len(members)}; smallest set "
                                f"{sum(sizes) / len(sizes):.1f} ({min(sizes)}-{max(sizes)}); "
                                + "; ".join(
                                    f"{a} covers alone {_alone(masks, a)} here, "
                                    f"{sum(r_[5][a] for r_ in per_seed) / len(per_seed):.1f} "
                                    f"one run each" for a in fixed))
                    if replicates:
                        for a, share in bootstrap(hit, members, seeds, min_seeds,
                                                  replicates).items():
                            boot.append(head + [a, f"{share:.4f}"])
    key = ["criterion", "tau", "tau_hv", "level", "subset", "budget"]
    heads = {
        "cover_summary.csv": key + ["problems", "covered_by_someone", "covered_by_nobody",
                                    "smallest_size", "smallest_sets_listed", "exact",
                                    "smallest_sets", "greedy_set", "analysis_commit"],
        "cover_curve.csv": key + ["k", "share", "set", "exact", "share_families_alike",
                                  "set_families_alike", "exact_families", "covered_per_family"],
        "cover_nobody.csv": key + ["problem", "family"],
        "cover_bootstrap.csv": key + ["algorithm", "share_in_smallest"],
        "cover_one_run.csv": key + ["seed", "problems", "covered_by_someone", "smallest_size",
                                    "exact", "smallest_set", "deterministic_alone"],
    }
    for name, rows in (("cover_summary.csv", summary), ("cover_curve.csv", curves),
                       ("cover_nobody.csv", uncovered), ("cover_bootstrap.csv", boot),
                       ("cover_one_run.csv", single)):
        if (name == "cover_bootstrap.csv" and not replicates) or (
                name == "cover_one_run.csv" and not fixed):
            continue
        with (out_dir / name).open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(heads[name])
            w.writerows(rows)
    lines.append(f"CSV: {out_dir} (analysis code {commit}; levels in cover_meta.json)")
    return "\n".join(lines)
