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

THE FLOOR (owner and reviewer, 2026-09-28). N points on the front itself leave
IGD+ above zero, so absolute levels of igdp_norm cannot all be reached: the
floor is about 0.002 at two objectives, 0.02 at three and 0.06 at five, above
0.01 and 0.003 at three objectives and above every level but 0.1 at five.
A problem's floor is the IGD+, in igdp_norm's frame, of the N points (N = its
population) that DSS selects from the reference front the runs were measured
against: the best N-point subset of that reference, as far as DSS finds it. A
run can go below it (DSS is not the best subset); GD+ has a floor of 0, so its
levels stay absolute. The floors are written out (igdp_floor.csv) and read back
by the next analysis of the same results.

AT A BUDGET. A run's value at b evaluations is its final value when b is its
budget; for a budget-dependent algorithm (budget.py), the final value of its
ladder rung of budget b; for any other, the first trajectory record at or
after b evaluations ([campaign] record_at puts one at every rung).

SCENARIO. With --scenario archive every value is the run archive reduced by
DSS instead of the population: final_archive at a run's own budget, the
archive checkpoint of budget b below it ([campaign] archive_checkpoints), and
a rung's final_archive for a budget-dependent algorithm. CSV under
<results>/_cover_archive/.

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
METRICS = ("igdp_norm", "gdp_norm", "hv_h")
NODE_LIMIT = 2_000_000


# ── reading the runs ────────────────────────────────────────────────────────


def _first_at_or_after(traj: list, b: int, metrics) -> dict:
    for rec in traj:
        if rec.get("fe", 0) >= b:
            return {m: rec.get(m) for m in metrics}
    return {m: None for m in metrics}


def _values_of_run(task) -> tuple:
    """(key, {budget: {metric: value}}) of one full-budget run."""
    key, run_dir, final, budget_fe, from_trajectory, budgets = task
    from .campaign import read_trajectory
    out = {}
    traj = read_trajectory(Path(run_dir)) if from_trajectory else None
    for b in budgets:
        if b == budget_fe:
            out[b] = {m: final.get(m) for m in METRICS}
        elif b < budget_fe and traj is not None:
            out[b] = _first_at_or_after(traj, b, METRICS)
    return key, out


def _end(row: dict, scenario: str) -> dict:
    """A run's answer at its own budget: the population's indicators, or in the
    archive scenario the reduced archive's."""
    return (row.get("final_archive") or {}) if scenario == "archive" else row["final"]


def collect(root: Path, budgets=BUDGETS, workers: int = 1, scenario: str = "final") -> dict:
    """{(problem, algorithm, seed): {budget: {metric: value}}} with the facts
    the analysis needs: which problems have a reference front, which
    algorithms are budget-dependent, and the rungs a campaign lacks."""
    from .budget import BUDGET_DEPENDENT
    from .campaign import scan_results
    rows = [r for r in scan_results(Path(root)) if r["status"] == "done"]
    rungs = {(r["problem"], r["ladder_of"], r["seed"], r.get("budget_fe")): r
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
        bfe = int(r.get("budget_nominal") or r.get("budget_fe") or 0)
        if dep:
            dependent.add(r["algorithm"])
            values[key] = ({bfe: {m: _end(r, scenario).get(m) for m in METRICS}}
                           if bfe in budgets else {})
            for b in budgets:
                if b < bfe:
                    rung = rungs.get((r["problem"], r["algorithm"], r["seed"], b))
                    if rung is None:
                        missing.append((r["problem"], r["algorithm"], r["seed"], b))
                    else:
                        values[key][b] = {m: _end(rung, scenario).get(m) for m in METRICS}
        elif scenario == "archive":
            # the run's own archive checkpoints below its budget
            values[key] = {b: ({m: _end(r, scenario).get(m) for m in METRICS} if b == bfe
                               else {m: r["archive_at"][str(b)].get(m) for m in METRICS})
                           for b in budgets
                           if b == bfe or (b < bfe and str(b) in r.get("archive_at", {}))}
        else:
            tasks.append((key, str(r["dir"]), r["final"], bfe, any(b < bfe for b in budgets),
                          tuple(budgets)))
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
    front = {}
    for r in main:
        if r["problem"] not in front:
            front[r["problem"]] = r.get("has_reference_front")
    for p, has in list(front.items()):
        if has is None:                                  # older meta.json: ask the registry
            from ..benchmarks import get as bench_get
            front[p] = callable(bench_get(p).pareto_front)
    return {"values": values, "front": front, "dependent": dependent, "missing": missing,
            "main": main, "scenario": scenario}


def igdp_floor(problem: str, n_ref: int) -> float | None:
    """The IGD+ floor of a problem (THE FLOOR above); None without a front."""
    import numpy as np
    from ..benchmarks import get as bench_get
    from .archive import dss_order
    from .metrics import _normalised, igd_plus
    p = bench_get(problem)
    if not callable(p.pareto_front):
        return None
    ref = np.asarray(p.pareto_front(n_ref), float)
    idx = dss_order(ref, k=p.pop_size, ideal=p.ideal, nadir=p.nadir)
    return igd_plus(*_normalised(ref[idx], ref, p.ideal, p.nadir))


def _median(values) -> float:
    v = sorted(values)
    return (v[(len(v) - 1) // 2] + v[len(v) // 2]) / 2 if v else math.nan


def _floor_task(task) -> tuple:
    problem, n_ref = task
    return problem, igdp_floor(problem, n_ref)


def floors(problems, n_ref: int, path: Path, workers: int = 1) -> dict:
    """{problem: floor} for the problems with a reference front, from the CSV at
    `path` where it has them for this n_ref, computed and added otherwise (a
    reference front takes up to a minute to sample: ZCAT's are verified)."""
    from ..benchmarks import get as bench_get
    have = {}
    if path.is_file():
        with path.open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                if int(row["n_ref"]) == n_ref:
                    have[row["problem"]] = float(row["floor"])
    todo = [(p, n_ref) for p in problems
            if p not in have and callable(bench_get(p).pareto_front)]
    if workers > 1 and len(todo) > 1:
        import multiprocessing as mp
        from .campaign import single_threaded_blas
        single_threaded_blas()
        with mp.Pool(processes=workers) as pool:
            done = list(pool.imap_unordered(_floor_task, todo))
    else:
        done = [_floor_task(t) for t in todo]
    have.update({p: f for p, f in done if f is not None})
    if done:
        with path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["problem", "pop", "n_ref", "floor"])
            for prob in sorted(have):
                w.writerow([prob, bench_get(prob).pop_size, n_ref, f"{have[prob]:.10g}"])
    return have


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
    `level` = (tau, tau_hv): igdp_norm <= (1 + tau) * floor, or gdp_norm <= tau,
    where the problem has a reference front; the hv_h gap <= tau_hv where not."""
    metric = CRITERIA[criterion]
    tau, tau_hv = level
    out: dict = {}
    for (prob, alg, seed), by_b in data["values"].items():
        v = by_b.get(budget, {})
        ok = False
        if data["front"].get(prob):
            x = v.get(metric)
            limit = tau if criterion == "gdp" else (1.0 + tau) * (floor or {}).get(prob, math.nan)
            ok = x is not None and math.isfinite(float(x)) and float(x) <= limit
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


def run(root: Path, *, taus=TAUS, floor_taus=FLOOR_TAUS, budgets=BUDGETS, min_seeds: int = 7,
        max_sets: int = 20, replicates: int = 0, workers: int = 1,
        out_dir: Path | None = None, scenario: str = "final", n_ref: int = 1000) -> str:
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
    floor = floors([p for p in problems if data["front"].get(p)], n_ref,
                   out_dir / "igdp_floor.csv", workers=workers)
    lines.append(f"  igdp levels from each problem's floor (igdp_floor.csv): median "
                 f"{_median(floor.values()):.4g}" if floor else "  no problem with a front")
    no_front = [p for p in problems if not data["front"].get(p)]
    if no_front:
        lines.append(f"  {len(no_front)} problems without a reference front: the relative gap "
                     f"of hv_h to the campaign's best (best_known_hv.csv)")
    with (out_dir / "best_known_hv.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["problem", "best_hv_h", "algorithm", "seed"])
        for p in sorted(best_hv):
            w.writerow([p, f"{best_hv[p][0]:.10g}", best_hv[p][1], best_hv[p][2]])

    summary, curves, uncovered, boot = [], [], [], []
    levels = {"igdp": list(zip(floor_taus, taus)), "gdp": list(zip(taus, taus))}
    for crit in CRITERIA:
        for tau, tau_hv in levels[crit]:
            name = (f"igdp <= {1 + tau:g} x floor" if crit == "igdp" else f"gdp <= {tau:g}")
            for b in budgets:
                hit = reached(data, crit, (tau, tau_hv), b, best_hv, floor)
                cov = successes(hit, min_seeds)
                res = analyse(cov, problems, problem_family, max_sets=max_sets)
                opt = ["+".join(s) for s in res["optimal"]]
                summary.append([crit, tau, tau_hv, b, res["n_problems"], res["n_covered"],
                                len(res["nobody"]), res["size"], len(opt), res["exact"],
                                " | ".join(opt), "+".join(res["greedy"])])
                for c in res["curve"]:
                    curves.append([crit, tau, tau_hv, b, c["k"], f"{c['share']:.6g}",
                                   "+".join(c["set"]),
                                   c["exact"], f"{c['share_families']:.6g}",
                                   "+".join(c["set_families"]), c["exact_families"],
                                   json.dumps(c["per_family"], sort_keys=True)])
                for p in res["nobody"]:
                    uncovered.append([crit, tau, tau_hv, b, p, problem_family(p)])
                lines.append(
                    f"{name} (hv_h gap <= {tau_hv:g}) at {b}: {res['n_covered']}/"
                    f"{res['n_problems']} covered by "
                    f"someone; smallest set {res['size']}{'' if res['exact'] else ' (greedy, bound)'}"
                    f" x{len(opt)}{'+' if len(opt) >= max_sets else ''}: {opt[0] if opt else '-'}"
                    f"; greedy {len(res['greedy'])}")
                if replicates:
                    for a, share in bootstrap(hit, problems, seeds, min_seeds,
                                              replicates).items():
                        boot.append([crit, tau, tau_hv, b, a, f"{share:.4f}"])
    heads = {
        "cover_summary.csv": ["criterion", "tau", "tau_hv", "budget", "problems",
                              "covered_by_someone",
                              "covered_by_nobody", "smallest_size", "smallest_sets_listed",
                              "exact", "smallest_sets", "greedy_set"],
        "cover_curve.csv": ["criterion", "tau", "tau_hv", "budget", "k", "share", "set", "exact",
                            "share_families_alike", "set_families_alike", "exact_families",
                            "covered_per_family"],
        "cover_nobody.csv": ["criterion", "tau", "tau_hv", "budget", "problem", "family"],
        "cover_bootstrap.csv": ["criterion", "tau", "tau_hv", "budget", "algorithm",
                                "share_in_smallest"],
    }
    for name, rows in (("cover_summary.csv", summary), ("cover_curve.csv", curves),
                       ("cover_nobody.csv", uncovered), ("cover_bootstrap.csv", boot)):
        if name == "cover_bootstrap.csv" and not replicates:
            continue
        with (out_dir / name).open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(heads[name])
            w.writerows(rows)
    lines.append(f"CSV: {out_dir}")
    return "\n".join(lines)
