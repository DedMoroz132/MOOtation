# SPDX-License-Identifier: Apache-2.0
"""Reading a finished campaign beyond medians: tests, groups and runtimes.

The campaign CLI (python -m mootation.run.campaign) calls these:

  reference_report  --compare/--ranks METRIC --reference ALG. On every
                    problem each algorithm's runs against the reference's:
                    the exact Wilcoxon rank-sum test, Holm-corrected over the
                    problems, and the Vargha-Delaney A12; across problems the
                    exact signed-rank test on the per-problem medians,
                    Holm-corrected over the algorithms (stats.py).
  rank_intervals    --ranks METRIC --ci: a 95 % bootstrap interval for each
                    mean rank, resampling the problems.
  property_ranks    --ranks METRIC --by KEY: mean ranks within each group of
                    problems sharing a property (benchmarks/properties.py:
                    front, multimodal, deceptive, bias, scaled, separable,
                    centre).
  gap_report        --gap METRIC: each algorithm's distance to the best value
                    any run reached, per problem, in the metric's own units.
  zero_share_report --zero-share: the share of seeds whose hypervolume is 0.
  ecdf_report       --ecdf METRIC: the COCO-style runtime ECDF over (run,
                    target) pairs, targets at fixed distances from the best
                    known value, with the interpolation between records named.

Every table marks with '*' the algorithms whose curves depend on the share of
the budget spent (postprocess.BUDGET_SCHEDULED).

`scenario` "final" reads what a run returned; "archive" reads what its run
archive holds, reduced to the problem's population size by DSS
(meta["final_archive"], written from 1.9's archive.csv).
"""

from __future__ import annotations

import math

from . import postprocess as PP
from . import stats as ST
from .metric_names import HIGHER_IS_BETTER


def _lower(metric: str) -> bool:
    return metric not in HIGHER_IS_BETTER


def _value(row: dict, metric: str, at, scenario: str):
    if scenario == "archive":
        v = (row.get("final_archive") or {}).get(metric)
        return None if v is None else float(v)
    from .campaign import value_at
    return value_at(row, metric, at)


def values(rows: list, metric: str, *, at=None, scenario: str = "final") -> dict:
    """{problem: {algorithm: [one value per finished run]}}."""
    out: dict = {}
    for r in rows:
        if r.get("status") != "done":
            continue
        v = _value(r, metric, at, scenario)
        if v is None or not math.isfinite(v):
            continue
        out.setdefault(r["problem"], {}).setdefault(r["algorithm"], []).append(v)
    return out


def _median(v):
    return ST.summarize(v)[0]


def problem_ranks(vals: dict, lower_better: bool) -> dict:
    """{problem: {algorithm: rank of its median, 1 = best, ties averaged}}."""
    out = {}
    for prob, algs in vals.items():
        names = sorted(algs)
        med = [_median(algs[a]) for a in names]
        key = [m if lower_better else -m for m in med]
        out[prob] = dict(zip(names, ST.average_ranks(key)))
    return out


def mark_budget(name: str) -> str:
    return name + PP.budget_note(name)


# ── 3.1 against a reference ─────────────────────────────────────────────────
def reference_report(rows: list, metric: str, reference: str, *, at=None,
                     scenario: str = "final", alpha: float = 0.05) -> dict:
    lower = _lower(metric)
    vals = values(rows, metric, at=at, scenario=scenario)
    probs = sorted(p for p, a in vals.items() if reference in a)
    algs = sorted({a for p in probs for a in vals[p]} - {reference})
    per_alg = []
    for alg in algs:
        tests = []
        for p in probs:
            if alg not in vals[p]:
                continue
            rs = ST.rank_sum(vals[p][alg], vals[p][reference], lower_better=lower)
            tests.append((p, rs, ST.a12(vals[p][alg], vals[p][reference], lower_better=lower)))
        adj = ST.holm([t[1]["p"] for t in tests])
        marks = [ST.mark(pa, t[1]["better"], alpha) for t, pa in zip(tests, adj)]
        a = [_median(vals[p][alg]) for p, _, _ in tests]
        b = [_median(vals[p][reference]) for p, _, _ in tests]
        sr = ST.wilcoxon_signed_rank(a, b, lower_better=lower)
        a12s = [t[2] for t in tests if t[2] is not None]
        per_alg.append({
            "algorithm": alg, "problems": len(tests),
            "wins": marks.count("+"), "losses": marks.count("-"), "ties": marks.count("="),
            "signed_rank": sr, "a12_median": _median(a12s) if a12s else None,
            "per_problem": {p: {"p_holm": pa, "mark": mk, "a12": eff}
                            for (p, _, eff), pa, mk in zip(tests, adj, marks)},
        })
    padj = ST.holm([e["signed_rank"]["p"] for e in per_alg])
    for e, pa in zip(per_alg, padj):
        e["p_holm"] = pa
        e["mark"] = ST.mark(pa, e["signed_rank"]["better"], alpha)
    per_alg.sort(key=lambda e: (-(e["wins"] - e["losses"]), e["algorithm"]))
    return {"metric": metric, "reference": reference, "scenario": scenario, "at": at,
            "alpha": alpha, "problems": len(probs), "algorithms": per_alg}


def format_reference(rep: dict) -> str:
    at = f" at {rep['at']:.0%} of the budget" if rep.get("at") is not None else ""
    lines = [
        f"{rep['metric']}{at} ({rep['scenario']}) against {rep['reference']} on "
        f"{rep['problems']} problem(s). Per problem: exact Wilcoxon rank-sum, Holm over the "
        f"problems, alpha {rep['alpha']}; across problems: exact signed-rank test on the "
        f"medians, Holm over the algorithms. '+' = better than the reference.",
        f"{'algorithm':<16}{'+':>5}{'-':>5}{'=':>5}{'W+':>9}{'W-':>9}{'p(Holm)':>10}  "
        f"{'across':>6}{'A12 med':>9}",
    ]
    for e in rep["algorithms"]:
        sr = e["signed_rank"]
        a12 = "-" if e["a12_median"] is None else f"{e['a12_median']:.2f}"
        lines.append(f"{mark_budget(e['algorithm'])[:16]:<16}{e['wins']:>5}{e['losses']:>5}"
                     f"{e['ties']:>5}{sr['w_plus']:>9.1f}{sr['w_minus']:>9.1f}"
                     f"{e['p_holm']:>10.3g}  {e['mark']:>6}{a12:>9}")
    lines.append("* its curve depends on the share of the budget spent (read by fraction).")
    return "\n".join(lines)


def rank_intervals(rows: list, metric: str, *, at=None, scenario: str = "final",
                   n_boot: int = 2000, level: float = 0.95) -> dict:
    """{algorithm: (low, high)} of the mean rank, problems resampled."""
    vals = values(rows, metric, at=at, scenario=scenario)
    return ST.bootstrap_mean_ranks(problem_ranks(vals, _lower(metric)), n_boot=n_boot,
                                   level=level)


# ── 3.2 by property ─────────────────────────────────────────────────────────
def property_ranks(rows: list, metric: str, key: str, *, at=None,
                   scenario: str = "final") -> dict:
    """{group label: {algorithm: (mean rank, problems)}} for one property."""
    from ..benchmarks.properties import KEYS, label
    if key not in KEYS:
        raise ValueError(f"unknown property '{key}'; known: {', '.join(KEYS)}")
    ranks = problem_ranks(values(rows, metric, at=at, scenario=scenario), _lower(metric))
    out: dict = {}
    for prob, per in ranks.items():
        g = label(prob, key)
        for alg, r in per.items():
            out.setdefault(g, {}).setdefault(alg, []).append(r)
    return {g: {a: (sum(v) / len(v), len(v)) for a, v in algs.items()}
            for g, algs in out.items()}


def format_property_ranks(groups: dict, metric: str, key: str) -> str:
    labels = sorted(groups)
    algs = sorted({a for g in groups.values() for a in g},
                  key=lambda a: sum(groups[g][a][0] for g in labels if a in groups[g])
                  / max(1, sum(1 for g in labels if a in groups[g])))
    n = {g: max(v[1] for v in groups[g].values()) for g in labels}
    lines = [f"mean rank by median {metric}, problems grouped by {key} "
             f"(problems per group in brackets)",
             f"{'algorithm':<16}" + "".join(f"{(g.split('=', 1)[1] + f'[{n[g]}]')[:13]:>14}"
                                            for g in labels)]
    for a in algs:
        lines.append(f"{mark_budget(a)[:16]:<16}" + "".join(
            f"{groups[g][a][0]:>14.2f}" if a in groups[g] else f"{'-':>14}" for g in labels))
    return "\n".join(lines)


# ── 3.4 post-processing ─────────────────────────────────────────────────────
def gap_report(rows: list, metric: str) -> str:
    gaps = PP.gap_to_best([r for r in rows], metric)
    algs = sorted({a for g in gaps.values() for a in g})
    lines = [f"median gap to the best {metric} any run reached, per problem "
             f"(0 = the best known)", f"{'problem':<18}" + "".join(
                 f"{mark_budget(a)[:11]:>12}" for a in algs)]
    for p in sorted(gaps):
        lines.append(f"{p[:18]:<18}" + "".join(
            f"{_median(gaps[p][a]):>12.3g}" if a in gaps[p] else f"{'-':>12}" for a in algs))
    return "\n".join(lines)


def zero_share_report(rows: list, metric: str = "hv") -> str:
    share = PP.zero_share(rows, metric)
    algs = sorted({a for s in share.values() for a in s})
    lines = [f"share of seeds whose {metric} is exactly 0 (a split a median hides)",
             f"{'problem':<18}" + "".join(f"{mark_budget(a)[:11]:>12}" for a in algs)]
    for p in sorted(share):
        if not any(v > 0 for v in share[p].values()):
            continue
        lines.append(f"{p[:18]:<18}" + "".join(
            f"{share[p][a]:>12.0%}" if a in share[p] else f"{'-':>12}" for a in algs))
    if len(lines) == 2:
        lines.append("(no problem has a seed at 0)")
    return "\n".join(lines)


def ecdf_report(rows: list, metric: str, *, precisions=(1.0, 0.1, 0.01, 0.001, 1e-4),
                interpolation: str = "step", grid=None) -> dict:
    """{algorithm: {"fe": grid, "share": [...], "pairs": n}}: the runtime ECDF.

    Targets on each problem are the best known value plus (minus, for a
    higher-is-better metric) each precision, in the metric's own units, so
    use a normalised one (igdp_norm, eps_norm, hv, hv_h). A run that never
    reaches a target counts in the denominator only.
    """
    from pathlib import Path
    lower = _lower(metric)
    best = PP.best_known(rows, metric)
    runs: dict = {}
    for r in rows:
        if r.get("status") != "done" or r["problem"] not in best:
            continue
        runs.setdefault(r["algorithm"], []).append((r["problem"], PP.read_trajectory(Path(r["dir"]))))
    if grid is None:
        grid = [round(10 ** (k / 2)) for k in range(4, 11)]          # 100 .. 100 000
    out = {}
    for alg, lst in runs.items():
        times, pairs = [], 0
        for prob, recs in lst:
            for d in precisions:
                t = best[prob] + d if lower else best[prob] - d
                pairs += 1
                times.append(PP.runtime_to_target(recs, metric, t, interpolation=interpolation))
        solved = sorted(x for x in times if x is not None)
        share = [sum(1 for x in solved if x <= g) / pairs if pairs else 0.0 for g in grid]
        out[alg] = {"fe": list(grid), "share": share, "pairs": pairs}
    return {"metric": metric, "interpolation": interpolation, "precisions": list(precisions),
            "algorithms": out}


def format_ecdf(rep: dict) -> str:
    algs = sorted(rep["algorithms"], key=lambda a: -rep["algorithms"][a]["share"][-1])
    grid = next(iter(rep["algorithms"].values()))["fe"] if rep["algorithms"] else []
    lines = [f"runtime ECDF of {rep['metric']}: share of (run, target) pairs solved by each "
             f"evaluation count; targets = best known ± {rep['precisions']}; between records: "
             f"{rep['interpolation']}",
             f"{'algorithm':<16}" + "".join(f"{g:>9}" for g in grid)]
    for a in algs:
        lines.append(f"{mark_budget(a)[:16]:<16}" + "".join(
            f"{s:>9.2f}" for s in rep["algorithms"][a]["share"]))
    return "\n".join(lines)
