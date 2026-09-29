# SPDX-License-Identifier: Apache-2.0
"""What a portfolio of algorithms gains over its members (task 4, item 5).

    python -m mootation.run.campaign camp.toml --portfolio [--scenario archive] [--workers 8]

Reads finished runs; runs nothing. The levels, the floors and the problems
left out or marked are --cover's (cover.py: collect, prepare, reached); CSV
under <results>/_portfolio/ (<results>/_portfolio_archive/ for the archive).

TWO GROUPS OF PROBLEMS, never mixed (reviewer, 2026-09-30): "front", the
problems with a reference front, whose levels read igdp_norm against the floor
(criterion igdp) or gdp_norm (criterion gdp); and "bbob", those without one
(bbob-biobj), whose single level reads the gap of hv_h to the best run of the
campaign. That target is relative, and mixed with the fronts it would inflate
whoever wins there.

FOR EVERY group, criterion, level, budget and number of seeds that must reach
the level for an algorithm to cover a problem (5, 7 and 10 of the ten: how much
the conclusions hang on --cover's 7):
  * the oracle, the virtual best solver: the problems some algorithm covers;
  * the single best solver: the algorithm that covers most, and the gap;
  * every algorithm's Shapley value in the coverage game v(S) = the problems
    some member of S covers. v is the sum over the problems p of the games
    v_p(S) = 1 when S holds an algorithm covering p: in v_p the c_p algorithms
    covering p are symmetric and the others null, so each gets 1 / c_p
    (Shapley's symmetry, null-player and additivity axioms). Hence
    phi_a = sum of 1 / c_p over the problems a covers — a problem a covers alone
    counts 1, one all 67 cover 1/67 — and the values add up to the oracle;
  * the problems each algorithm covers alone.
COMPLEMENTARITY (7 seeds): for every two algorithms A and B, how many problems
A covers and B does not (portfolio_complementarity.csv, the pairs where some).
THE ORACLE GAP PER PROBLEM, at every budget: the virtual best solver's median
over the seeds (igdp_norm in "front", hv_h in "bbob") beside that of the single
best solver on average over the group — the smallest mean of log(median / best
median) for igdp_norm, of the relative gap (best - median) / best for hv_h — as
their ratio or relative gap.
THE MATRIX (portfolio_matrix.csv): problem x algorithm x budget in the scenario,
how many seeds reach every level, and the median and the 7th best of
igdp_norm, gdp_norm, hv_h and eps_norm — the coverage without reading the runs.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

SEED_RULES = (5, 7, 10)                   # seeds of ten that must reach a level
MATRIX_METRICS = ("igdp_norm", "gdp_norm", "hv_h", "eps_norm")
HIGHER = {"hv_h"}                         # the others: lower is better


def shapley(cov: dict) -> dict:
    """{algorithm: Shapley value} in the coverage game of `cov` = {algorithm:
    set of problems it covers}: the sum over its problems of 1 / (how many
    algorithms cover each)."""
    count: dict = {}
    for probs in cov.values():
        for p in probs:
            count[p] = count.get(p, 0) + 1
    return {a: sum(1.0 / count[p] for p in probs) for a, probs in cov.items()}


def oracle(cov: dict) -> dict:
    """The oracle (virtual best solver) and the single best solver of `cov`."""
    union = set().union(*cov.values()) if cov else set()
    sbs = max(sorted(cov), key=lambda a: len(cov[a])) if cov else None
    alone = {a: {p for p in probs if all(p not in cov[b] for b in cov if b != a)}
             for a, probs in cov.items()}
    return {"oracle": len(union), "sbs": sbs, "sbs_covers": len(cov[sbs]) if sbs else 0,
            "unique": alone}


def complementarity(cov: dict) -> dict:
    """{(A, B): problems A covers and B does not}, the pairs where some."""
    return {(a, b): len(cov[a] - cov[b]) for a in sorted(cov) for b in sorted(cov)
            if a != b and cov[a] - cov[b]}


def medians(data: dict, metric: str, budget: int, problems) -> dict:
    """{problem: {algorithm: median over the seeds of metric at budget}}."""
    from .cover import _median
    by: dict = {}
    for (p, a, _), vals in data["values"].items():
        if p not in problems:
            continue
        v = vals.get(budget, {}).get(metric)
        if v is not None and math.isfinite(float(v)):
            by.setdefault(p, {}).setdefault(a, []).append(float(v))
    return {p: {a: _median(vs) for a, vs in algs.items()} for p, algs in by.items()}


def oracle_gap(med: dict, metric: str) -> tuple:
    """(single best solver, [(problem, best algorithm, best, sbs value, gap)]):
    the SBS minimizes the mean over the problems of log(median / best) for a
    lower-is-better metric, of (best - median) / best for hv_h; the gap is that
    same ratio (>= 1) or relative gap (>= 0) on each problem."""
    higher = metric in HIGHER
    best = {p: (max if higher else min)(algs.values()) for p, algs in med.items() if algs}

    def loss(p, v):
        b = best[p]
        if higher:
            return (b - v) / b if b > 0 else 0.0
        return math.log(v / b) if b > 0 and v > 0 else (0.0 if v == b else math.inf)

    algs = sorted({a for m in med.values() for a in m})
    mean = {}
    for a in algs:
        losses = [loss(p, med[p][a]) for p in best if a in med[p]]
        mean[a] = sum(losses) / len(losses) if len(losses) == len(best) else math.inf
    sbs = min(algs, key=lambda a: (mean[a], a)) if algs else None
    rows = []
    for p in sorted(best):
        b = best[p]
        win = (max if higher else min)(sorted(med[p]), key=lambda a: med[p][a])
        v = med[p].get(sbs)
        gap = (None if v is None else (b - v) / b if higher and b > 0 else
               v / b if not higher and b > 0 else None)
        rows.append((p, win, b, v, gap))
    return sbs, rows


def _fmt(v) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return "inf" if math.isinf(v) else f"{v:.6g}"
    return str(v)


def run(root: Path, *, taus=None, floor_taus=None, budgets=None, workers: int = 1,
        out_dir: Path | None = None, scenario: str = "final", n_ref: int = 1000,
        keep_caveats: bool = False) -> str:
    """The whole analysis: text for the terminal; CSV files in out_dir."""
    import time
    from .. import __version__
    from . import cover as V
    from .behaviour import family_of
    from .provenance import revision
    taus = tuple(taus or V.TAUS)
    floor_taus = tuple(floor_taus or V.FLOOR_TAUS)
    budgets = tuple(budgets or V.BUDGETS)
    root = Path(root)
    out_dir = (Path(out_dir) if out_dir else
               root / ("_portfolio_archive" if scenario == "archive" else "_portfolio"))
    out_dir.mkdir(parents=True, exist_ok=True)
    cover_dir = root / ("_cover_archive" if scenario == "archive" else "_cover")
    cover_dir.mkdir(parents=True, exist_ok=True)
    data = V.collect(root, budgets, workers=workers, scenario=scenario)
    if not data["values"]:
        return "portfolio: no finished runs"
    prep = V.prepare(data, n_ref, cover_dir / "igdp_floor.csv", workers=workers,
                     keep_caveats=keep_caveats)
    best_hv = V.best_known_hv(data)
    kept = [p for p in prep["problems"] if p not in prep["excluded"]]
    groups = {"front": [p for p in kept if data["front"].get(p)],
              "bbob": [p for p in kept if not data["front"].get(p)]}
    # level k of each criterion; "bbob" has one, the hv_h gap (both of cover's
    # criteria read it there)
    levels = {("front", "igdp"): list(zip(floor_taus, taus)),
              ("front", "gdp"): list(zip(taus, taus)),
              ("bbob", "hv"): list(zip(taus, taus))}
    rev = revision()
    commit = (rev.get("commit") or "unknown")[:12] + ("+dirty" if rev.get("dirty") else "")
    fam = family_of()

    # {(criterion, k, budget): {problem: {algorithm: {seed: reached}}}} over every
    # problem, left out or not (the matrix has them all): cover's "igdp" reads
    # igdp_norm against the floor, its "gdp" gdp_norm, and both the hv_h gap
    # where there is no front
    hits = {}
    for k, tau in enumerate(taus):
        for b in budgets:
            hits[("igdp", k, b)] = V.reached(data, "igdp", (floor_taus[k], tau), b, best_hv,
                                             prep["floor"])
            hits[("gdp", k, b)] = V.reached(data, "gdp", (tau, tau), b, best_hv)
            hits[("hv", k, b)] = hits[("gdp", k, b)]

    summary, values, pairs = [], [], []
    for (group, crit), lv in levels.items():
        members = set(groups[group])
        if not members:
            continue
        for k, (tau, tau_hv) in enumerate(lv):
            level = (f"igdp_norm <= {1 + tau:g} x {prep['which']} floor" if crit == "igdp"
                     else f"gdp_norm <= {tau:g}" if crit == "gdp" else f"hv_h gap <= {tau_hv:g}")
            for b in budgets:
                hit = {p: v for p, v in hits[(crit, k, b)].items() if p in members}
                for rule in SEED_RULES:
                    cov = V.successes(hit, rule)
                    o = oracle(cov)
                    phi = shapley(cov)
                    summary.append([group, crit, k, level, b, rule, len(members), o["oracle"],
                                    o["sbs"], o["sbs_covers"], o["oracle"] - o["sbs_covers"],
                                    commit])
                    for a in sorted(cov):
                        values.append([group, crit, k, level, b, rule, a, fam.get(a, "?"),
                                       f"{phi[a]:.6g}", len(cov[a]), len(o["unique"][a])])
                    if rule == 7:
                        for (a, c), n in complementarity(cov).items():
                            pairs.append([group, crit, k, level, b, a, c, n])

    gaps, sbs_rows = [], []
    for group, metric in (("front", "igdp_norm"), ("bbob", "hv_h")):
        members = set(groups[group])
        for b in budgets:
            med = medians(data, metric, b, members)
            sbs, rows_ = oracle_gap(med, metric)
            sbs_rows.append([group, metric, b, sbs])
            for p, win, best, v, gap in rows_:
                gaps.append([group, metric, b, p, win, _fmt(best), sbs, _fmt(v), _fmt(gap)])

    # the matrix: problem x algorithm x budget, seeds per level, median and 7th best
    matrix = []
    by_run: dict = {}
    for (p, a, s), vals in data["values"].items():
        by_run.setdefault((p, a), {})[s] = vals
    for (p, a) in sorted(by_run):
        crits = ("igdp", "gdp") if data["front"].get(p) else ("hv",)
        for b in budgets:
            row = [p, a, b, scenario, len(by_run[(p, a)])]
            for crit in ("igdp", "gdp", "hv"):
                for k in range(len(taus)):
                    seeds = hits[(crit, k, b)].get(p, {}).get(a, {})
                    row.append(sum(bool(x) for x in seeds.values()) if crit in crits else "")
            for m in MATRIX_METRICS:
                vs = sorted(float(v[b][m]) for v in by_run[(p, a)].values()
                            if b in v and v[b].get(m) is not None and math.isfinite(float(v[b][m])))
                if m in HIGHER:
                    vs.reverse()
                row += [_fmt(V._median(vs) if vs else None), _fmt(vs[6] if len(vs) >= 7 else None)]
            matrix.append(row)

    key = ["group", "criterion", "level", "level_text", "budget", "seeds_required"]
    heads = {
        "portfolio_summary.csv": key + ["problems", "oracle_covers", "single_best",
                                        "single_best_covers", "gap", "analysis_commit"],
        "portfolio_shapley.csv": key + ["algorithm", "family", "shapley", "covers",
                                        "covers_alone"],
        "portfolio_complementarity.csv": key[:5] + ["algorithm_a", "algorithm_b",
                                                    "a_covers_b_does_not"],
        "portfolio_oracle_gap.csv": ["group", "metric", "budget", "problem", "best_algorithm",
                                     "best_median", "single_best", "single_best_median", "gap"],
    }
    for name, rows_ in (("portfolio_summary.csv", summary), ("portfolio_shapley.csv", values),
                        ("portfolio_complementarity.csv", pairs),
                        ("portfolio_oracle_gap.csv", gaps)):
        with (out_dir / name).open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(heads[name])
            w.writerows(rows_)
    mhead = ["problem", "algorithm", "budget", "scenario", "seeds"]
    mhead += [f"{c}_level{k}_seeds" for c in ("igdp", "gdp", "hv") for k in range(len(taus))]
    mhead += [f"{m}_{s}" for m in MATRIX_METRICS for s in ("median", "7th")]
    with (out_dir / "portfolio_matrix.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(mhead)
        w.writerows(matrix)
    meta = {"analysis": {"mootation": __version__, "commit": rev.get("commit"),
                         "dirty": rev.get("dirty"), "source": rev.get("source")},
            "results": str(root), "scenario": scenario,
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "groups": {g: len(m) for g, m in groups.items()},
            "levels": {"igdp": f"igdp_norm <= (1 + tau) x the {prep['which']} floor, tau in "
                               f"{list(floor_taus)}",
                       "gdp": f"gdp_norm <= tau, tau in {list(taus)}",
                       "hv": f"(best - hv_h) / best <= tau, tau in {list(taus)}, best = the "
                             f"campaign's best final hv_h (group bbob)"},
            "seed_rules": list(SEED_RULES), "budgets": list(budgets),
            "excluded": prep["excluded"], "single_best_by_metric": sbs_rows}
    (out_dir / "portfolio_meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")

    lines = [f"portfolio ({scenario}): {len(groups['front'])} problems with a front, "
             f"{len(groups['bbob'])} without; analysis code {commit}; CSV in {out_dir}"]
    full = max(budgets)
    for (group, crit), lv in levels.items():
        for k in range(len(lv)):
            cells = []
            for rule in SEED_RULES:
                r = next((r for r in summary if r[:3] == [group, crit, k] and r[4] == full
                          and r[5] == rule), None)
                if r:
                    cells.append(f"{rule}/10: oracle {r[7]}/{r[6]}, best {r[8]} {r[9]}")
            if cells:
                lines.append(f"  {group} {crit} level {k} at {full}: " + "; ".join(cells))
    for group, metric, b, sbs in sbs_rows:
        if b == full:
            g = [float(r[8]) for r in gaps if r[0] == group and r[2] == b and r[8] not in ("", "inf")]
            if g:
                g.sort()
                lines.append(f"  oracle gap, {group} ({metric}) at {b}: single best {sbs}; "
                             f"median gap {g[len(g) // 2]:.3g}, worst {g[-1]:.3g}")
    text = "\n".join(lines)
    (out_dir / "portfolio_report.txt").write_text(text + "\n", encoding="utf-8")
    return text
