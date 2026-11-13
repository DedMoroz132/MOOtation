# SPDX-License-Identifier: Apache-2.0
"""The report of a campaign at one objective (task 18, stage 5).

Read off every finished run's improvement records (improvements.csv: the
evaluations, gradient calls, f and cv of each new best point) and its
meta.json (`so`: f_opt, the kind of targets, f(x0), grad_cost), with the
measures of include/mootation/so_metrics.hpp (_core.so_metrics: the runtime
to a target, ERT, the ECDF, the value at a budget), as COCO defines them
(N. Hansen et al., "COCO: Performance Assessment", arXiv:1605.03560, H16;
"Anytime Performance Assessment in Blackbox Optimization Benchmarking", IEEE
TEVC 26(6) 2022, H22):

  ert_table     per problem and algorithm the ERT (H16 §4.1, H22 §V-A: every
                evaluation of every run until it hit the target, over the
                hits; inf without a hit) at the table's precisions (H22 Table
                I: 1e1 ... 1e-7 absolute; Moré and Wild's 1e-1 ... 1e-7 of
                f(x0) - f_opt relative), and the hits.
  ecdf          per dimension, per bbob class and over all problems, the share
                of (run, target) pairs hit by a budget of x D evaluations, x
                on a grid of 10^(i/5), the 51 targets of COCO (or Moré and
                Wild's four); a miss stays in the denominator (H16 §4.3).
                Never over dimensions (H16 §4.3, H22 §VI).
  values_at     per problem and algorithm the median f - f_opt of the best
                point at budgets of 10^k D evaluations (an ordinal scale
                only, H22 §III-D).
  indistinct    per dimension, class and budget, the algorithms no worse than
                the best: the blocks (problem, seed — an instance under
                bbob_instance = "by_run") with a run of every algorithm, f -
                f_opt at the budget ranked within each block; Friedman's test
                over the blocks; if it rejects at alpha, the best by mean rank
                against each other by the exact Wilcoxon signed-rank test,
                Holm-adjusted (stats.py); every algorithm the test cannot tell
                from the best is listed. If Friedman does not reject, all are.
                The test of the atlas's protocol replaces it once decided
                (task 18, §5).
  confidence    per problem and algorithm the share R of runs that reach the
                finest target (COCO's f_opt + 1e-8, Moré and Wild's 1e-7 of
                f(x0) - f_opt), the number of runs M_95 that finds it with
                95 % confidence and its cost M_95 x the mean cost of a run
                (D. Kavetski, Y. Qin, G. Kuczera, Water Resour. Res. 54,
                2018, (6), (12), (13); include/mootation/so_confidence.hpp);
                per algorithm typ M_95 and spread M_95, the median and the
                interquartile range over the problems ((16), (17)).

READINGS:
  REP-1. The bbob classes are RR-6829's headings (Hansen et al. 2009):
    separable f1-f5, low or moderate conditioning f6-f9, high conditioning
    unimodal f10-f14, multimodal with adequate global structure f15-f19,
    multimodal with weak global structure f20-f24. Other problems at one
    objective form their family's class (MGH, NIST, cec2013, ...).
  REP-2. A run's cost is evaluations + grad_cost x gradient calls (H22 §III-B);
    its length (for ERT and for the budgets) the cost of its last evaluation,
    meta's so.cost.
  REP-3. f - f_opt of a run with no feasible point, or of a problem without
    f_opt, is not a value: such runs are left out of values_at, and a block
    with one is left out of indistinct. A run without a feasible point at a
    budget counts as +inf there (worse than any value).
  REP-4. Kavetski's success ("within a tolerance of the best known value")
    is the run's reaching the finest target of its kind; R is safeguarded to
    [1/(n + 1), 1 - 1/(n + 1)] over its n runs (K18 after (12)), so a cell
    without a success has a finite M_95, the safeguard's.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

BBOB_CLASSES = (("separable", range(1, 6)), ("low/moderate conditioning", range(6, 10)),
                ("high conditioning, unimodal", range(10, 15)),
                ("multimodal, adequate structure", range(15, 20)), ("multimodal, weak structure", range(20, 25)))


def problem_class(name: str) -> str:
    """REP-1: the bbob class of bbobNN_nDD_1D, else the family (the name up to
    its first digit or underscore)."""
    if name.startswith("bbob") and name[4:6].isdigit():
        fid = int(name[4:6])
        for label, fids in BBOB_CLASSES:
            if fid in fids:
                return label
    stem = ""
    for ch in name:
        if ch.isdigit() or ch == "_":
            break
        stem += ch
    return stem or name


def dimension(row: dict) -> int:
    return int(row["n_vars"])


def _records(run_dir) -> list:
    text = (run_dir / "improvements.csv").read_text(encoding="utf-8")
    out = []
    header = True
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        if header:
            header = False
            continue
        e, g, f, cv = line.split(",")
        out.append((int(float(e)), int(float(g)), float(f), float(cv)))
    return out


def runs(root: Path, problems=None) -> list:
    """The finished runs at one objective under a campaign's root: {problem,
    algorithm, seed, n_vars, so, records}."""
    from .campaign import scan_results
    out = []
    for r in scan_results(Path(root)):
        if r.get("n_objs") != 1 or r.get("status") != "done":
            continue
        if problems is not None and r["problem"] not in problems:
            continue
        m = json.loads((r["dir"] / "meta.json").read_text(encoding="utf-8"))
        out.append({"problem": r["problem"], "algorithm": r["algorithm"], "seed": r["seed"],
                    "n_vars": m.get("n_vars"), "so": m.get("so") or {}, "records": _records(r["dir"])})
    return out


def _so():
    from .. import _core
    return _core.so_metrics


def _targets(run: dict, precisions) -> list | None:
    so = run["so"]
    if so.get("f_opt") is None or not so.get("targets"):
        return None
    f_x0 = so.get("f_x0")
    return list(_so().targets(so["targets"], list(precisions), so["f_opt"], 0.0 if f_x0 is None else f_x0))


def _table_precisions(kind: str) -> list:
    so = _so()
    return list(so.coco_table_precisions()) if kind == "absolute" else list(so.more_wild_factors())


def _cost(run: dict) -> float:
    return float(run["so"].get("cost") or (run["records"][-1][0] if run["records"] else 0))


def ert_table(rs: list) -> dict:
    """{(problem, algorithm): {"precisions", "ert", "hits", "runs"}} (H16 §4.1)."""
    so = _so()
    by = {}
    for r in rs:
        by.setdefault((r["problem"], r["algorithm"]), []).append(r)
    out = {}
    for key, group in sorted(by.items()):
        kind = group[0]["so"].get("targets")
        if group[0]["so"].get("f_opt") is None or not kind:
            continue
        eps = _table_precisions(kind)
        erts, hits = [], []
        for i in range(len(eps)):
            times, lengths = [], []
            for r in group:
                t = _targets(r, eps)[i]
                times.append(so.runtime(r["records"], t, float(r["so"].get("grad_cost", 1.0))))
                lengths.append(_cost(r))
            erts.append(so.ert(times, lengths))
            hits.append(sum(1 for t in times if t is not None))
        out[key] = {"precisions": eps, "ert": erts, "hits": hits, "runs": len(group)}
    return out


def ecdf(rs: list) -> dict:
    """{D: {class or "all": {algorithm: {"x": [evaluations / D], "share": [...]}}}}
    over COCO's 51 targets (absolute) or Moré and Wild's four (relative)."""
    so = _so()
    out = {}
    by_d = {}
    for r in rs:
        if r["so"].get("f_opt") is not None and r["so"].get("targets"):
            by_d.setdefault(dimension(r), []).append(r)
    for D, group in sorted(by_d.items()):
        longest = max(_cost(r) for r in group) / D
        xs = [10 ** (i / 5) for i in range(0, int(math.floor(5 * math.log10(max(longest, 1.0)))) + 2)]
        cells = {}
        for r in group:
            eps = list(so.default_precisions(r["so"]["targets"]))
            gc = float(r["so"].get("grad_cost", 1.0))
            times = [so.runtime(r["records"], t, gc) for t in _targets(r, eps)]
            scaled = [None if t is None else t / D for t in times]
            for c in (problem_class(r["problem"]), "all"):
                cells.setdefault(c, {}).setdefault(r["algorithm"], []).extend(scaled)
        out[D] = {c: {a: {"x": xs, "share": list(so.ecdf(v, xs))} for a, v in sorted(algs.items())}
                  for c, algs in sorted(cells.items())}
    return out


def _value(run: dict, budget: float) -> float | None:
    v = _so().value_at(run["records"], budget, float(run["so"].get("grad_cost", 1.0)))
    return math.inf if v is None else v - run["so"]["f_opt"]       # REP-3


def budgets_of(D: int, longest: float) -> list:
    return [10 ** k * D for k in range(1, 9) if 10 ** k * D <= longest]


def values_at(rs: list) -> dict:
    """{(problem, algorithm): {budget: the median f - f_opt}} (REP-3)."""
    by = {}
    for r in rs:
        if r["so"].get("f_opt") is not None:
            by.setdefault((r["problem"], r["algorithm"]), []).append(r)
    longest = {}
    for (p, _), group in by.items():
        longest[p] = max(longest.get(p, 0.0), max(_cost(r) for r in group))
    out = {}
    for key, group in sorted(by.items()):
        D = dimension(group[0])
        cells = {}
        for b in budgets_of(D, longest[key[0]]):     # the problem's longest run: a run that ended keeps its value
            v = sorted(_value(r, b) for r in group)
            n = len(v)
            cells[b] = v[n // 2] if n % 2 else (v[n // 2 - 1] + v[n // 2]) / 2
        out[key] = cells
    return out


def indistinct(rs: list, *, alpha: float = 0.05) -> dict:
    """{(D, class, budget): {"blocks", "friedman_p", "best", "same": [algorithms],
    "p": {algorithm: Holm-adjusted p against the best}}}."""
    from .stats import friedman, holm, wilcoxon_signed_rank
    algs = sorted({r["algorithm"] for r in rs})
    groups = {}
    for r in rs:
        if r["so"].get("f_opt") is None:
            continue
        groups.setdefault((dimension(r), problem_class(r["problem"])), []).append(r)
    out = {}
    for (D, c), group in sorted(groups.items()):
        cell = {}
        for r in group:
            cell.setdefault((r["problem"], r["seed"]), {})[r["algorithm"]] = r
        longest = max(_cost(r) for r in group)
        for b in budgets_of(D, longest):
            blocks = [[_value(cell[k][a], b) for a in algs] for k in sorted(cell) if len(cell[k]) == len(algs)]
            if not blocks:
                continue
            fr = friedman(blocks)
            best = min(range(len(algs)), key=lambda j: fr["mean_ranks"][j])
            same, adj = list(algs), {}
            if fr["p"] < alpha:
                others = [j for j in range(len(algs)) if j != best]
                ps = [wilcoxon_signed_rank([bl[best] for bl in blocks], [bl[j] for bl in blocks])["p"] for j in others]
                hp = holm(ps)
                adj = {algs[j]: p for j, p in zip(others, hp)}
                same = [algs[best]] + [algs[j] for j, p in zip(others, hp) if p >= alpha]
            out[(D, c, b)] = {"blocks": len(blocks), "friedman_p": fr["p"], "best": algs[best],
                              "same": sorted(same), "p": adj}
    return out


def confidence(rs: list, *, alpha: float = 0.95) -> dict:
    """{"cells": {(problem, algorithm): {"R", "M", "cost", "runs"}},
    "algorithms": {algorithm: {"typ", "spread", "problems"}}} (REP-4)."""
    so = _so()
    by = {}
    for r in rs:
        if r["so"].get("f_opt") is not None and r["so"].get("targets"):
            by.setdefault((r["problem"], r["algorithm"]), []).append(r)
    cells, efforts = {}, {}
    for key, group in sorted(by.items()):
        finest = list(so.default_precisions(group[0]["so"]["targets"]))[-1:]
        hits = [so.runtime(r["records"], _targets(r, finest)[0], float(r["so"].get("grad_cost", 1.0))) is not None
                for r in group]
        R = so.reliability(hits)
        M = so.characteristic_effort(R, alpha, len(group))
        mean_cost = sum(_cost(r) for r in group) / len(group)
        cells[key] = {"R": R, "M": M, "cost": so.characteristic_cost(M, mean_cost), "runs": len(group)}
        efforts.setdefault(key[1], []).append(M)
    algs = {a: {"typ": so.typ(v), "spread": so.spread(v), "problems": len(v)} for a, v in sorted(efforts.items())}
    return {"cells": cells, "algorithms": algs}


def _g(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, float) and math.isinf(v):
        return "inf"
    return f"{v:.3g}"


def report(root: Path, *, alpha: float = 0.05, problems=None) -> str:
    """The five tables as text (markdown)."""
    rs = runs(root, problems)
    if not rs:
        return "no finished run at one objective"
    lines = ["## ERT (evaluations; hits/runs) by problem", ""]
    for (p, a), row in ert_table(rs).items():
        cells = ", ".join(f"{e:g}: {_g(v)} ({h}/{row['runs']})" for e, v, h in
                          zip(row["precisions"], row["ert"], row["hits"]))
        lines.append(f"- {p} | {a}: {cells}")
    lines += ["", "## ECDF of (run, target) pairs by evaluations / D", ""]
    for D, classes in ecdf(rs).items():
        for c, algs in classes.items():
            for a, e in algs.items():
                pts = ", ".join(f"{x:.3g}: {s:.3f}" for x, s in zip(e["x"][::5], e["share"][::5]))
                lines.append(f"- D {D} | {c} | {a}: {pts}")
    lines += ["", "## Median f - f_opt at budgets (an ordinal scale)", ""]
    for (p, a), cells in values_at(rs).items():
        lines.append(f"- {p} | {a}: " + ", ".join(f"{b:g}: {_g(v)}" for b, v in cells.items()))
    lines += ["", f"## Methods no worse than the best (Friedman, then Wilcoxon + Holm, alpha = {alpha:g})", ""]
    for (D, c, b), cell in indistinct(rs, alpha=alpha).items():
        lines.append(f"- D {D} | {c} | budget {b:g}: {', '.join(cell['same'])} "
                     f"(best {cell['best']}, Friedman p = {cell['friedman_p']:.3g}, {cell['blocks']} blocks)")
    conf = confidence(rs)
    lines += ["", "## Runs and cost to the finest target with 95 % confidence (Kavetski et al. 2018)", ""]
    for (p, a), cell in conf["cells"].items():
        lines.append(f"- {p} | {a}: R = {cell['R']:.3g} ({cell['runs']} runs), M_95 = {cell['M']}, "
                     f"cost {_g(cell['cost'])}")
    for a, row in conf["algorithms"].items():
        lines.append(f"- {a}: typ M_95 = {row['typ']:g}, spread M_95 = {row['spread']:g} ({row['problems']} problems)")
    return "\n".join(lines)
