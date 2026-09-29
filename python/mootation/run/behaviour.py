# SPDX-License-Identifier: Apache-2.0
"""How the algorithms behave along a run, read off the trajectories.

    python -m mootation.run.campaign camp.toml --behaviour [--workers 8]

Reads finished runs; runs nothing. CSV under <results>/_behaviour/.

DESCRIPTORS of one full-budget run, from its trajectory.jsonl (the population's
records; ladder rungs are left out). "Settles" means: from that record on the
value never gets more than SETTLE (10 %) better again.
  t_gdp_<tau>     evaluations until gdp_norm <= tau (progress towards the front),
                  tau in TAUS; the first record at or past the target (no
                  interpolation); empty when never within the budget
  t_igdp_<k>      evaluations until igdp_norm <= (1 + k) x the best final
                  igdp_norm any run of the campaign reached on the problem (the
                  coverage), k in BEST_TAUS. Relative to the best run, not to the
                  IGD+ floor of cover.py, so that nothing here waits on the floor
  settle_gdp, settle_igdp
                  the evaluations at which gdp_norm and igdp_norm settle
  slope           the two-point slope of log10 gdp_norm against log10 fe from the
                  first record to settle_gdp: -1 halves gdp_norm while the
                  evaluations double; 0 when it settles at once
  spread          log10(settle_igdp / settle_gdp): above 0 the coverage still
                  improves after the convergence has stopped ("converged, now
                  spreading"), below 0 it stopped first
  rc_drop         the largest range_cover of the run minus the last: how far the
                  population's extent collapsed
  stall           the share of the budget spent after the population became
                  non-dominated for good (nd_share >= ND_FULL in every later
                  record) and the quality settled (igdp_norm where there is a
                  front, hv_h where there is none): stagnation at nd_share ~ 1
  dup_end, dup_max  the last and the largest dup_share
  igdx_end, settle_igdx, cr_end   where the problem has a Pareto set
Descriptors of gdp_norm and igdp_norm exist only where the problem has a
reference front; the others everywhere.

PER ALGORITHM AND PROBLEM the median over the seeds (a run that never reached
a target counts as infinitely late; "inf" in the CSV).

PROFILES AND GROUPS. On each problem every descriptor's medians are ranked
across the algorithms, as a share from 0 (the smallest value) to 1 (the
largest), so that problems of different scales weigh alike; a descriptor on
which all algorithms are equal on a problem (a target nobody reaches) is left
out there. Two algorithms are as far apart as the mean absolute difference of
their shares over the (problem, descriptor) pairs both have. The algorithms
are grouped by average linkage (the distance between two groups is the mean
distance between their members), cut into as many groups as there are
families — those algorithms.def declares, and "Baseline" for the run layer's
baselines — and the groups set beside the families. behaviour_profile.csv
gives each algorithm's median share per descriptor: its behaviour in one row.

WHAT IS WRITTEN (<results>/_behaviour/):
  behaviour_meta.json     the commit of the analysis code, the settings
  behaviour_runs.csv      one row per run
  behaviour_medians.csv   one row per problem and algorithm
  behaviour_profile.csv   one row per algorithm: family, group, median shares
  behaviour_merges.csv    the grouping step by step: distance, the two groups
  behaviour_report.txt    what the terminal shows
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

TAUS = (0.1, 0.03, 0.01, 0.003)          # gdp_norm targets, as cover.py's levels
BEST_TAUS = (1.0, 0.5, 0.25, 0.1)        # igdp_norm <= (1 + k) x the campaign's best
SETTLE = 0.1
ND_FULL = 0.99


def descriptor_names() -> list:
    return ([f"t_gdp_{t:g}" for t in TAUS] + [f"t_igdp_{k:g}" for k in BEST_TAUS]
            + ["settle_gdp", "settle_igdp", "slope", "spread", "rc_drop", "stall",
               "dup_end", "dup_max", "igdx_end", "settle_igdx", "cr_end"])


# ── one run ────────────────────────────────────────────────────────────────


def _series(recs: list, metric: str) -> list:
    """[(fe, value)] of the records with a finite value of metric."""
    out = []
    for r in recs:
        v = r.get(metric)
        if v is not None and math.isfinite(float(v)):
            out.append((float(r["fe"]), float(v)))
    return out


def settle_index(values: list, higher: bool = False) -> int | None:
    """The first index from which the series never gets more than SETTLE better
    again: v[i] <= (1 + SETTLE) min(v[i:]), or for a higher-is-better series
    v[i] >= max(v[i:]) / (1 + SETTLE). None for an empty series."""
    best = math.inf if not higher else -math.inf
    ok = []
    for v in reversed(values):
        best = max(best, v) if higher else min(best, v)
        ok.append(v * (1.0 + SETTLE) >= best if higher else v <= (1.0 + SETTLE) * best)
    ok.reverse()
    return next((i for i, good in enumerate(ok) if good), None)


def _first_fe(recs: list, metric: str, target: float) -> float | None:
    for fe, v in _series(recs, metric):
        if v <= target:
            return fe
    return None


def run_descriptors(recs: list, best_igdp: float | None) -> dict:
    """The DESCRIPTORS of one run from its trajectory records (in order)."""
    d = dict.fromkeys(descriptor_names())
    if not recs:
        return d
    end = float(recs[-1]["fe"])
    g, q = _series(recs, "gdp_norm"), _series(recs, "igdp_norm")
    if g:
        for t in TAUS:
            d[f"t_gdp_{t:g}"] = _first_fe(recs, "gdp_norm", t)
        s = settle_index([v for _, v in g])
        d["settle_gdp"] = g[s][0]
        if s == 0:
            d["slope"] = 0.0
        else:
            (fe0, v0), (fe1, v1) = g[0], g[s]
            d["slope"] = ((math.log10(max(v1, 1e-12)) - math.log10(max(v0, 1e-12)))
                          / (math.log10(fe1) - math.log10(fe0)))
    if q:
        if best_igdp is not None and best_igdp > 0:
            for k in BEST_TAUS:
                d[f"t_igdp_{k:g}"] = _first_fe(recs, "igdp_norm", (1.0 + k) * best_igdp)
        d["settle_igdp"] = q[settle_index([v for _, v in q])][0]
    if d["settle_gdp"] and d["settle_igdp"]:
        d["spread"] = math.log10(d["settle_igdp"] / d["settle_gdp"])
    rc = [v for _, v in _series(recs, "range_cover")]
    if rc:
        d["rc_drop"] = max(rc) - rc[-1]
    # stagnation: non-dominated for good, and the quality settled
    nd = _series(recs, "nd_share")
    t_nd = None
    for fe, v in reversed(nd):
        if v < ND_FULL:
            break
        t_nd = fe
    if q:
        settled = d["settle_igdp"]
    else:
        h = _series(recs, "hv_h")
        settled = h[settle_index([v for _, v in h], higher=True)][0] if h else None
    if nd:
        d["stall"] = ((end - max(t_nd, settled)) / end
                      if t_nd is not None and settled is not None and end > 0 else 0.0)
    dup = [v for _, v in _series(recs, "dup_share")]
    if dup:
        d["dup_end"], d["dup_max"] = dup[-1], max(dup)
    x = _series(recs, "igdx")
    if x:
        d["igdx_end"] = x[-1][1]
        d["settle_igdx"] = x[settle_index([v for _, v in x])][0]
    cr = _series(recs, "cr")
    if cr:
        d["cr_end"] = cr[-1][1]
    return d


def _describe(task) -> tuple:
    key, run_dir, best_igdp = task
    from .campaign import read_trajectory
    return key, run_descriptors(read_trajectory(Path(run_dir)), best_igdp)


def collect(root: Path, workers: int = 1, problems=None) -> dict:
    """{(problem, algorithm, seed): descriptors} of the finished full-budget runs."""
    from .campaign import main_rows, scan_results
    from .postprocess import best_known
    rows = [r for r in main_rows(scan_results(Path(root))) if r["status"] == "done"
            and (problems is None or r["problem"] in problems)]
    best = best_known(rows, "igdp_norm")
    tasks = [((r["problem"], r["algorithm"], r["seed"]), str(r["dir"]), best.get(r["problem"]))
             for r in rows]
    out = {}
    if workers > 1 and len(tasks) > 1:
        import multiprocessing as mp
        from .campaign import single_threaded_blas
        single_threaded_blas()
        with mp.Pool(processes=workers) as pool:
            for key, d in pool.imap_unordered(_describe, tasks, chunksize=64):
                out[key] = d
    else:
        for t in tasks:
            key, d = _describe(t)
            out[key] = d
    return out


# ── per algorithm and problem ──────────────────────────────────────────────


def _median(values: list) -> float | None:
    v = sorted(values)
    if not v:
        return None
    return (v[(len(v) - 1) // 2] + v[len(v) // 2]) / 2


def medians(runs: dict) -> dict:
    """{(problem, algorithm): {descriptor: median over the seeds}}; a target
    never reached counts as infinitely late."""
    by: dict = {}
    for (prob, alg, _), d in runs.items():
        by.setdefault((prob, alg), []).append(d)
    out = {}
    for key, ds in by.items():
        row = {"n_runs": len(ds)}
        for name in descriptor_names():
            if name.startswith("t_"):
                have = [x[name] for x in ds if x[name] is not None or _asked(x, name)]
                vals = [math.inf if v is None else v for v in have]
            else:
                vals = [x[name] for x in ds if x[name] is not None]
            row[name] = _median(vals)
        out[key] = row
    return out


def _asked(d: dict, name: str) -> bool:
    """Was the target of a t_ descriptor asked of this run at all (the run has
    the metric)? A run without gdp_norm has settle_gdp None."""
    return d["settle_gdp" if name.startswith("t_gdp") else "settle_igdp"] is not None


# ── profiles and groups ────────────────────────────────────────────────────


def shares(med: dict) -> dict:
    """{algorithm: {(problem, descriptor): share}}: on every problem each
    descriptor's medians ranked across the algorithms, 0 the smallest, 1 the
    largest, ties sharing their mean rank. A descriptor on which every
    algorithm has the same value there (no algorithm reaching a target: all
    infinitely late) tells them apart in nothing and is left out."""
    from .stats import average_ranks
    out: dict = {}
    algs_of: dict = {}
    for p, a in med:
        algs_of.setdefault(p, []).append(a)
    for prob in sorted(algs_of):
        algs = sorted(algs_of[prob])
        for name in descriptor_names():
            have = [(a, med[(prob, a)][name]) for a in algs if med[(prob, a)][name] is not None]
            if len(have) < 2 or len({v for _, v in have}) == 1:
                continue
            ranks = average_ranks([v for _, v in have])
            for (a, _), r in zip(have, ranks):
                out.setdefault(a, {})[(prob, name)] = (r - 1.0) / (len(have) - 1)
    return out


def distance(a: dict, b: dict) -> float:
    """The mean absolute difference of two algorithms' shares over what both have."""
    common = a.keys() & b.keys()
    if not common:
        return math.nan
    return sum(abs(a[k] - b[k]) for k in common) / len(common)


def average_linkage(names: list, dist: dict) -> list:
    """The merges of agglomerative grouping with average linkage:
    [(distance, group_a, group_b)], smallest first. dist[(x, y)] for x < y."""
    def d(x, y):
        return dist[(x, y) if x < y else (y, x)]
    groups = [[n] for n in sorted(names)]
    merges = []
    while len(groups) > 1:
        best = None
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                dd = [d(x, y) for x in groups[i] for y in groups[j]]
                dd = [v for v in dd if not math.isnan(v)]
                v = sum(dd) / len(dd) if dd else math.inf
                if best is None or v < best[0]:
                    best = (v, i, j)
        v, i, j = best
        merges.append((v, groups[i], groups[j]))
        groups[i] = groups[i] + groups[j]
        del groups[j]
    return merges


def cut(names: list, merges: list, k: int) -> dict:
    """{name: group number 1..k}: the merges replayed until k groups remain,
    numbered in the order of their first member."""
    groups = [[n] for n in sorted(names)]
    for _, a, b in merges:
        if len(groups) <= k:
            break
        ga = next(g for g in groups if a[0] in g)
        gb = next(g for g in groups if b[0] in g)
        groups.remove(gb)
        ga.extend(gb)
    groups.sort(key=lambda g: min(g))
    return {n: i + 1 for i, g in enumerate(groups) for n in g}


def family_of() -> dict:
    """{algorithm: family} as algorithms.def declares them; the run layer's
    baselines (random and Sobol sampling, the ablations) as "Baseline"."""
    from .algorithms import algorithm_families
    from .baselines import BASELINES
    out = {a: fam for fam, algs in algorithm_families() for a in algs}
    out.update({b: "Baseline" for b in BASELINES})
    return out


# ── the whole analysis ─────────────────────────────────────────────────────


def _fmt(v) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return "inf" if math.isinf(v) else f"{v:.6g}"
    return str(v)


def run(root: Path, *, workers: int = 1, out_dir: Path | None = None, problems=None) -> str:
    """The whole analysis: text for the terminal; CSV files in out_dir."""
    import time
    from .. import __version__
    from .postprocess import budget_note
    from .provenance import revision
    root = Path(root)
    out_dir = Path(out_dir) if out_dir else root / "_behaviour"
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = collect(root, workers=workers, problems=problems)
    if not runs:
        return "behaviour: no finished runs"
    names = descriptor_names()
    rev = revision()
    commit = (rev.get("commit") or "unknown")[:12] + ("+dirty" if rev.get("dirty") else "")
    meta = {"analysis": {"mootation": __version__, "commit": rev.get("commit"),
                         "dirty": rev.get("dirty"), "source": rev.get("source")},
            "results": str(root), "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "taus": list(TAUS), "best_taus": list(BEST_TAUS), "settle": SETTLE,
            "nd_full": ND_FULL, "descriptors": names,
            "igdp_targets": "igdp_norm <= (1 + k) x the best final igdp_norm of the campaign "
                            "on the problem (not the IGD+ floor of cover.py)"}
    (out_dir / "behaviour_meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    with (out_dir / "behaviour_runs.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["problem", "algorithm", "seed"] + names)
        for key in sorted(runs, key=lambda k: (k[0], k[1], k[2])):
            w.writerow(list(key) + [_fmt(runs[key][n]) for n in names])
    med = medians(runs)
    with (out_dir / "behaviour_medians.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["problem", "algorithm", "n_runs"] + names)
        for key in sorted(med):
            w.writerow(list(key) + [med[key]["n_runs"]] + [_fmt(med[key][n]) for n in names])
    sh = shares(med)
    algs = sorted(sh)
    dist = {(a, b): distance(sh[a], sh[b]) for i, a in enumerate(algs) for b in algs[i + 1:]}
    merges = average_linkage(algs, dist)
    fam = family_of()
    families = sorted({fam.get(a, "?") for a in algs})
    group = cut(algs, merges, len(families))
    profile = {a: {n: _median([v for (p, m), v in sh[a].items() if m == n]) for n in names}
               for a in algs}
    with (out_dir / "behaviour_profile.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["algorithm", "family", "group", "analysis_commit"] + names)
        for a in sorted(algs, key=lambda a: (group[a], a)):
            w.writerow([a, fam.get(a, "?"), group[a], commit] + [_fmt(profile[a][n]) for n in names])
    with (out_dir / "behaviour_merges.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["step", "distance", "group_a", "group_b"])
        for i, (v, a, b) in enumerate(merges, 1):
            w.writerow([i, f"{v:.6g}", " ".join(a), " ".join(b)])

    problems_seen = sorted({k[0] for k in runs})
    lines = [f"behaviour: {len(runs)} runs, {len(problems_seen)} problems, {len(algs)} "
             f"algorithms; analysis code {commit}; CSV in {out_dir}",
             f"  settles = never again more than {SETTLE:.0%} better; igdp targets relative to "
             f"the campaign's best run; '*' = budget-dependent (read by share of budget)", ""]
    lines.append(f"{len(families)} groups by average linkage beside the declared families "
                 f"(how many of each family fall in each group):")
    head = f"{'family':<36}" + "".join(f"{g:>4}" for g in range(1, len(families) + 1))
    lines.append(head)
    for f_ in families:
        members = [a for a in algs if fam.get(a, "?") == f_]
        counts = [sum(1 for a in members if group[a] == g) for g in range(1, len(families) + 1)]
        lines.append(f"{f_[:35]:<36}" + "".join(f"{c if c else '.':>4}" for c in counts))
    lines.append("")
    show = ["t_gdp_0.1", "t_gdp_0.01", "t_igdp_1", "slope", "spread", "rc_drop", "stall",
            "dup_max", "igdx_end"]
    lines.append("median share per descriptor (0 = smallest value among the algorithms, "
                 "1 = largest), by group:")
    lines.append(f"{'group':>5} {'algorithm':<16}" + "".join(f"{n:>11}" for n in show))
    for a in sorted(algs, key=lambda a: (group[a], a)):
        lines.append(f"{group[a]:>5} {(a + budget_note(a))[:16]:<16}" + "".join(
            f"{profile[a][n]:>11.2f}" if profile[a][n] is not None else f"{'':>11}"
            for n in show))
    lines.append("")
    lines.append("last merges (distance, the two groups joined):")
    for v, a, b in merges[-min(10, len(merges)):]:
        lines.append(f"  {v:.3f}  [{' '.join(a[:6])}{' ...' if len(a) > 6 else ''}] + "
                     f"[{' '.join(b[:6])}{' ...' if len(b) > 6 else ''}]")
    text = "\n".join(lines)
    (out_dir / "behaviour_report.txt").write_text(text + "\n", encoding="utf-8")
    return text
