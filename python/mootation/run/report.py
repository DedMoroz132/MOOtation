# SPDX-License-Identifier: Apache-2.0
"""Reading a finished campaign beyond medians: tests, groups and runtimes.

The campaign CLI (python -m mootation.run.campaign) calls these:

  reference_report  --compare/--ranks METRIC --reference ALG. On every
                    problem each algorithm's runs against the reference's:
                    the exact Wilcoxon rank-sum test, Holm-corrected over the
                    problems (or Benjamini-Hochberg, --adjust bh), and the
                    Vargha-Delaney A12 with its class; a warning when the
                    seeds and the problems leave no mark but '=' possible.
                    Across problems the exact signed-rank test on the
                    per-problem medians (their logarithms for igdp_norm and
                    eps_norm), Holm-corrected over the algorithms, with the
                    sign test and the Hodges-Lehmann estimate (stats.py).
  rank_intervals    --ranks METRIC --ci: a 95 % bootstrap interval for each
                    mean rank, resampling the problems.
  property_ranks    --ranks METRIC --by KEY: mean ranks within each group of
                    problems sharing a property (benchmarks/properties.py:
                    front, multimodal, deceptive, bias, scaled, separable,
                    centre, boundary, degenerate, disconnected, linkage, n_obj,
                    n_vars, bbob_groups).
  gap_report        --gap METRIC: each algorithm's distance to the best value
                    any run reached, per problem, in the metric's own units.
  zero_share_report --zero-share: the share of seeds whose hypervolume is 0.
  ecdf_report       --ecdf METRIC: the COCO-style runtime ECDF over (run,
                    target) pairs, targets at fixed distances from the best
                    known value, with the interpolation between records named.
  seed_distance     --seed-distance: whether the seeds find the same solutions,
                    the chamfer distance between runs' non-dominated sets in
                    normalised variables (task 2, D4).
  eps_table         --eps-table: the binary multiplicative epsilon between
                    every two algorithms, per problem (D6).
  magnitude_experiment  --magnitude: the magnitude of the dominated set
                    against hv_h, at up to three objectives (experiment D7).

Every table marks with '*' the algorithms whose curves depend on the share of
the budget spent (postprocess.BUDGET_SCHEDULED).

`scenario` "final" reads what a run returned; "archive" reads what its run
archive holds, reduced to the problem's population size by DSS
(meta["final_archive"], written from 1.9's archive.csv).
"""

from __future__ import annotations

import json
import math
import textwrap
from pathlib import Path

from . import postprocess as PP
from . import stats as ST
from .metric_names import HIGHER_IS_BETTER


def _lower(metric: str) -> bool:
    return metric not in HIGHER_IS_BETTER


def _value(row: dict, metric: str, at, scenario: str):
    """The tables' reading (campaign.value_at) in both scenarios: in the archive
    one the reduced archive, at `at` from its checkpoints like the population
    from its trajectory."""
    from .campaign import value_at
    if scenario == "archive" and row.get("scenario") != "archive":
        row = dict(row, final=row.get("final_archive") or {}, scenario="archive")
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
# The positive scale-free indicators: across problems they are compared by the
# logarithm of their medians, so that a problem counts by the ratio of two
# medians, not by how large its values happen to be (task 6, item 10).
LOG_METRICS = frozenset({"igdp_norm", "eps_norm"})
ADJUSTMENTS = ("holm", "bh")


def reference_report(rows: list, metric: str, reference: str, *, at=None,
                     scenario: str = "final", alpha: float = 0.05,
                     adjust: str = "holm") -> dict:
    """Every algorithm against `reference` (task 6 of 2026-10-09, item 10).

    Per problem: the exact rank-sum test of the seeds, the p-values adjusted
    over the problems by Holm, or by Benjamini & Hochberg (`adjust` "bh"),
    a mark +, - or = at `alpha`, and beside it A12 and its class. Whether a
    mark can be anything but = at all: `labels_possible` — under Holm only if
    the smallest p a problem can give (both seed samples wholly apart, 2 /
    C(n_a + n_b, n_a)) times the number of problems is below alpha, so at k
    seeds a side not beyond 0.025 C(2k, k) problems at alpha 0.05; under BH
    only if that p is below alpha. Across problems: the exact signed-rank
    test on the per-problem medians (Holm over the algorithms) — on their
    logarithms for igdp_norm and eps_norm (LOG_METRICS) — with the sign test
    and the Hodges-Lehmann estimate of the difference beside it, as a ratio
    of medians for the logarithmic ones."""
    if adjust not in ADJUSTMENTS:
        raise ValueError(f"unknown adjustment '{adjust}'; known: {', '.join(ADJUSTMENTS)}")
    correct = ST.holm if adjust == "holm" else ST.benjamini_hochberg
    lower = _lower(metric)
    log = metric in LOG_METRICS
    vals = values(rows, metric, at=at, scenario=scenario)
    probs = sorted(p for p, a in vals.items() if reference in a)
    algs = sorted({a for p in probs for a in vals[p]} - {reference})
    p_min = 1.0
    per_alg = []
    for alg in algs:
        tests = []
        for p in probs:
            if alg not in vals[p]:
                continue
            rs = ST.rank_sum(vals[p][alg], vals[p][reference], lower_better=lower)
            tests.append((p, rs, ST.a12(vals[p][alg], vals[p][reference], lower_better=lower)))
            p_min = min(p_min, ST.smallest_rank_sum_p(len(vals[p][alg]),
                                                      len(vals[p][reference])))
        adj = correct([t[1]["p"] for t in tests])
        marks = [ST.mark(pa, t[1]["better"], alpha) for t, pa in zip(tests, adj)]
        classes = [ST.a12_magnitude(t[2]) for t in tests]
        a = [_median(vals[p][alg]) for p, _, _ in tests]
        b = [_median(vals[p][reference]) for p, _, _ in tests]
        if log:
            pairs = [(math.log(x), math.log(y)) for x, y in zip(a, b) if x > 0 and y > 0]
            a, b = [x for x, _ in pairs], [y for _, y in pairs]
        sr = ST.wilcoxon_signed_rank(a, b, lower_better=lower)
        hl = ST.hodges_lehmann([x - y for x, y in zip(a, b)])
        a12s = [t[2] for t in tests if t[2] is not None]
        a12_med = _median(a12s) if a12s else None
        per_alg.append({
            "algorithm": alg, "problems": len(tests),
            "wins": marks.count("+"), "losses": marks.count("-"), "ties": marks.count("="),
            "ties_large": sum(1 for mk, c in zip(marks, classes) if mk == "=" and c == "large"),
            "a12_classes": {c: classes.count(c)
                            for c in ("negligible", "small", "medium", "large")},
            "signed_rank": sr, "sign": ST.sign_test(a, b, lower_better=lower),
            "hl": hl, "hl_ratio": math.exp(hl) if log and hl is not None else None,
            "a12_median": a12_med, "a12_median_class": ST.a12_magnitude(a12_med),
            "per_problem": {p: {"p_adj": pa, "mark": mk, "a12": eff, "a12_class": c}
                            for (p, _, eff), pa, mk, c in zip(tests, adj, marks, classes)},
        })
    padj = ST.holm([e["signed_rank"]["p"] for e in per_alg])
    for e, pa in zip(per_alg, padj):
        e["p_holm"] = pa
        e["mark"] = ST.mark(pa, e["signed_rank"]["better"], alpha)
    per_alg.sort(key=lambda e: (-(e["wins"] - e["losses"]), e["algorithm"]))
    m = len(probs)
    possible = (m * p_min < alpha) if adjust == "holm" else (p_min < alpha)
    return {"metric": metric, "reference": reference, "scenario": scenario, "at": at,
            "alpha": alpha, "adjust": adjust, "log": log, "problems": m,
            "smallest_p": p_min, "labels_possible": bool(m == 0 or possible),
            "algorithms": per_alg}


def format_reference(rep: dict) -> str:
    at = f" at {rep['at']:.0%} of the budget" if rep.get("at") is not None else ""
    adjust = {"holm": "Holm", "bh": "Benjamini-Hochberg"}[rep.get("adjust", "holm")]
    on = "the logarithms of the medians" if rep.get("log") else "the medians"
    lines = textwrap.wrap(
        f"{rep['metric']}{at} ({rep['scenario']}) against {rep['reference']} on "
        f"{rep['problems']} problem(s). Per problem: exact Wilcoxon rank-sum, {adjust} over "
        f"the problems, alpha {rep['alpha']}, with A12 (Vargha-Delaney; classes at 0.06, "
        f"0.14, 0.21 from 0.5); across problems: exact signed-rank test on {on}, Holm over "
        f"the algorithms, the sign test and the Hodges-Lehmann estimate beside it. "
        f"'+' = better than the reference; =L counts the '=' with a large A12. Sorted by + "
        f"minus -: an order of these counts, not a rating of the algorithms.", 100)
    if not rep.get("labels_possible", True):
        lines += textwrap.wrap(
            f"WARNING: no mark but '=' is possible here: the smallest p a problem can give "
            f"with these seeds is {rep['smallest_p']:.3g}"
            + (f", and {adjust} multiplies it by the {rep['problems']} problems"
               if rep.get("adjust", "holm") == "holm" else "")
            + f"; that is not below alpha {rep['alpha']}. Read A12 and its class instead.", 100)
    hl_head = "HL ratio" if rep.get("log") else "HL diff"
    lines.append(f"{'algorithm':<16}{'+':>4}{'-':>4}{'=':>4}{'=L':>4}{'A12 med':>12}"
                 f"{'W+':>8}{'W-':>8}{'p(Holm)':>9}{'across':>7}{'sign':>9}{'p':>8}"
                 f"{hl_head:>10}")
    for e in rep["algorithms"]:
        sr, sg = e["signed_rank"], e["sign"]
        a12 = ("-" if e["a12_median"] is None
               else f"{e['a12_median']:.2f} {e['a12_median_class'][:3]}")
        hl = e["hl_ratio"] if rep.get("log") else e["hl"]
        lines.append(f"{mark_budget(e['algorithm'])[:16]:<16}{e['wins']:>4}{e['losses']:>4}"
                     f"{e['ties']:>4}{e['ties_large']:>4}{a12:>12}"
                     f"{sr['w_plus']:>8.1f}{sr['w_minus']:>8.1f}{e['p_holm']:>9.3g}"
                     f"{e['mark']:>7}{sg['wins']:>5}:{sg['losses']:<3}{sg['p']:>8.3g}"
                     f"{'-' if hl is None else format(hl, '.3g'):>10}")
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


# ── structural bias (task 2, A3) ─────────────────────────────────────────────
def _chi2_sf(x: float, dof: int) -> float:
    """P(chi^2_dof >= x): the regularized upper incomplete gamma Q(dof/2, x/2),
    by its series below a + 1 and its continued fraction above (Numerical
    Recipes §6.2), to about 1e-12."""
    a, z = dof / 2.0, x / 2.0
    if z <= 0.0:
        return 1.0
    gln = math.lgamma(a)
    if z < a + 1.0:
        term = total = 1.0 / a
        ap = a
        for _ in range(1000):
            ap += 1.0
            term *= z / ap
            total += term
            if abs(term) < abs(total) * 1e-15:
                break
        return max(0.0, 1.0 - total * math.exp(-z + a * math.log(z) - gln))
    b = z + 1.0 - a
    c = 1.0 / 1e-300
    d = 1.0 / b
    h = d
    for i in range(1, 1000):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        d = 1e-300 if abs(d) < 1e-300 else d
        c = b + an / c
        c = 1e-300 if abs(c) < 1e-300 else c
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-15:
            break
    return min(1.0, math.exp(-z + a * math.log(z) - gln) * h)


def _final_variables(run_dir, m: int, n: int):
    from .campaign import _read_points
    from pathlib import Path
    _, X = _read_points(Path(run_dir) / "final.csv", m, n)
    return X


BIAS_BINS = 20


def _bin_measures(hist, at_lower: int, at_upper: int, at_half: int) -> dict | None:
    """Kudela et al.'s binsizes from the counts of one point set's coordinates
    in BIAS_BINS bins: binsize_k = n_k / (N / K); bound = the mean of the two
    outer bins, centre of the two middle ones (1 = no bias); with the shares
    of coordinates exactly on a bound and exactly at 1/2."""
    import numpy as np
    h = np.asarray(hist, float)
    total = h.sum()
    if total == 0:
        return None
    b = h / (total / len(h))
    k = len(h)
    return {"bound": float((b[0] + b[-1]) / 2), "centre": float((b[k // 2 - 1] + b[k // 2]) / 2),
            "at_bound": (at_lower + at_upper) / total, "at_half": at_half / total,
            "binsizes": [round(float(v), 4) for v in b]}


def bias_class(bound: float, centre: float) -> str:
    """Our thresholds; Kudela et al. give none: "bound" when the outer bins
    exceed 1.5 and the middle ones do not, "centre" the other way round,
    "mixed" when both do, "empty edges" when the outer bins fall below 0.67."""
    if bound > 1.5 and centre > 1.5:
        return "mixed"
    if bound > 1.5:
        return "bound"
    if centre > 1.5:
        return "centre"
    if bound < 0.67:
        return "empty edges"
    return "none"


def structural_bias(rows: list) -> dict:
    """Where each algorithm searched, per problem (Kudela, van Stein, Bäck &
    Kononova, GECCO 2026, Section 3.3; benchmarks/uninformative.py).

    Per (problem, algorithm), over its finished runs, variables mapped to
    [0, 1] by the bounds, three sets of points:
      L  the final populations (final.csv): binsizes, the shares exactly on a
         bound and at 1/2, and its class (bias_class);
      P  the points non-dominated over each whole run and
      A  every point evaluated — both from the counts the uninformative
         problems leave in meta.json (structural_bias), where a run has them;
      chi2_p   the chi-squared test on ONE point drawn from every run's final
               population (independent across runs, unlike the points of one
               population), its coordinates pooled into min(10, max(2, n/5))
               bins;
      cei      at two variables, the median over the runs of the Clark-Evans
               index of the final population: mean nearest-neighbour distance
               over 0.5 / sqrt(N), ~1 uniform, < 1 clustered, > 1 grid-like.
    """
    import numpy as np
    from ..benchmarks import get as bench_get
    groups: dict = {}
    for r in rows:
        if r.get("status") != "done":
            continue
        groups.setdefault((r["problem"], r["algorithm"]), []).append(r)
    out: dict = {}
    for (prob, alg), runs in sorted(groups.items()):
        p = bench_get(prob)
        lo = np.array([b[0] for b in p.bounds], float)
        hi = np.array([b[1] for b in p.bounds], float)
        U_all, picks, ceis = [], [], []
        counts = {"all": None, "nondominated": None}
        for r in runs:
            X = _final_variables(r["dir"], p.n_obj, p.n_vars)
            if X is None or not len(X):
                continue
            U = (X - lo) / (hi - lo)
            U_all.append(U)
            picks.append(U[np.random.default_rng(int(r.get("seed") or 0)).integers(len(U))])
            if U.shape[1] == 2 and len(U) > 1:
                D = np.sqrt(((U[:, None, :] - U[None, :, :]) ** 2).sum(axis=2))
                np.fill_diagonal(D, np.inf)
                ceis.append(float(D.min(axis=1).mean() / (0.5 / np.sqrt(len(U)))))
            meta_path = Path(r["dir"]) / "meta.json"
            sb = (json.loads(meta_path.read_text(encoding="utf-8")).get("structural_bias")
                  if meta_path.is_file() else None)
            for key in counts:
                if sb and sb.get(key):
                    c = sb[key]
                    old = counts[key] or {"hist": [0] * len(c["hist"]), "at_lower": 0,
                                          "at_upper": 0, "at_half": 0}
                    counts[key] = {"hist": [a + b for a, b in zip(old["hist"], c["hist"])],
                                   **{f: old[f] + c[f] for f in ("at_lower", "at_upper", "at_half")}}
        if not U_all:
            continue
        U = np.vstack(U_all)
        v = U.ravel()
        hist = np.histogram(np.clip(v, 0.0, 1.0), bins=BIAS_BINS, range=(0.0, 1.0))[0]
        L = _bin_measures(hist, int(np.sum(v == 0.0)), int(np.sum(v == 1.0)),
                          int(np.sum(v == 0.5)))
        w = np.concatenate(picks)
        nb = int(min(10, max(2, len(w) // 5)))
        c = np.histogram(np.clip(w, 0.0, 1.0), bins=nb, range=(0.0, 1.0))[0]
        expected = len(w) / nb
        chi = float(((c - expected) ** 2).sum() / expected)
        out[(prob, alg)] = {
            "runs": len(U_all), "points": int(len(U)), "L": L, "class": bias_class(L["bound"],
                                                                               L["centre"]),
            "P": _bin_measures(**counts["nondominated"]) if counts["nondominated"] else None,
            "A": _bin_measures(**counts["all"]) if counts["all"] else None,
            "chi2_p": _chi2_sf(chi, nb - 1), "cei": float(np.median(ceis)) if ceis else None,
        }
    return out


def format_bias(rep: dict) -> str:
    lines = ["where each algorithm searched, per problem (Kudela et al. 2026): binsizes over 20 "
             "bins of all coordinates of all runs, bound = the two outer bins, centre = the two "
             "middle ones, 1 = no bias; L the final populations, P the run's non-dominated "
             "points, A every evaluated point; '=bd' / '=0.5' the shares exactly on a bound "
             "and at 1/2; chi2 p on one point per run; CEI the Clark-Evans index at n = 2; "
             "class by our thresholds (bound or centre > 1.5, edges < 0.67); strongest first"]

    def cell(m, key, digits=2, width=7):
        return f"{m[key]:>{width}.{digits}f}" if m and m.get(key) is not None else "-".rjust(width)

    probs = sorted({p for p, _ in rep})
    for prob in probs:
        rows = [(a, e) for (p, a), e in rep.items() if p == prob]
        rows.sort(key=lambda t: -max(abs(math.log2(max(t[1]["L"]["bound"], 1e-3))),
                                     abs(math.log2(max(t[1]["L"]["centre"], 1e-3)))))
        lines.append(f"\n{prob}")
        lines.append(f"  {'algorithm':<20}{'runs':>5}{'class':>13}{'L bd':>7}{'L ctr':>7}"
                     f"{'=bd':>7}{'=0.5':>7}{'P bd':>7}{'P ctr':>7}{'A bd':>7}{'A ctr':>7}"
                     f"{'A =bd':>7}{'chi2 p':>9}{'CEI':>6}")
        for a, e in rows:
            L, P, A = e["L"], e["P"], e["A"]
            lines.append(f"  {mark_budget(a)[:20]:<20}{e['runs']:>5}{e['class']:>13}"
                         + cell(L, "bound") + cell(L, "centre") + cell(L, "at_bound", 3)
                         + cell(L, "at_half", 3) + cell(P, "bound") + cell(P, "centre")
                         + cell(A, "bound") + cell(A, "centre") + cell(A, "at_bound", 3)
                         + f"{e['chi2_p']:>9.2g}" + cell(e, "cei", 2, 6))
    return "\n".join(lines)


# ── task 2, D4, D6, D7: set comparisons read from the final populations ─────
def _final_sets(rows: list):
    """{(problem, algorithm): [(F, X) of every finished run]}, from final.csv."""
    from ..benchmarks import get as bench_get
    from .campaign import _read_points
    from pathlib import Path
    out: dict = {}
    for r in rows:
        if r.get("status") != "done":
            continue
        p = bench_get(r["problem"])
        F, X = _read_points(Path(r["dir"]) / "final.csv", p.n_obj, p.n_vars)
        if F is None or not len(F):
            continue
        out.setdefault((r["problem"], r["algorithm"]), []).append((F, X))
    return out


def seed_distance(rows: list) -> dict:
    """D4: do different seeds find the same solutions?

    Per (problem, algorithm): the chamfer distance between the non-dominated
    solutions of two runs, variables normalised by the bounds (cyclic ones
    wrapped), averaged over every pair of seeds, beside the median pdist — the
    spread WITHIN one run's set. The measure needs no reference front, so it
    would suit a calibration problem as well; a campaign, and so this table,
    runs registry problems only.
    """
    import itertools
    from ..benchmarks import get as bench_get
    from . import metrics as M
    out: dict = {}
    for (prob, alg), runs in sorted(_final_sets(rows).items()):
        p = bench_get(prob)
        sets = [X[M.nondominated_mask(F)] for F, X in runs if X is not None and len(X)]
        if len(sets) < 2:
            continue
        pairs = [M.chamfer(a, b, bounds=p.bounds, cyclic=p.cyclic_vars)
                 for a, b in itertools.combinations(sets, 2)]
        spread = [M.pairwise_distance(s, bounds=p.bounds, cyclic=p.cyclic_vars)
                  for s in sets if len(s) > 1]
        out[(prob, alg)] = {"runs": len(sets), "pairs": len(pairs),
                            "chamfer": sum(pairs) / len(pairs),
                            "chamfer_max": max(pairs),
                            "pdist": _median(spread) if spread else None}
    return out


def format_seed_distance(rep: dict) -> str:
    lines = ["do the seeds find the same solutions? chamfer distance between the "
             "non-dominated sets of two runs in normalised variables, mean (and worst) "
             "over the pairs of seeds, beside pdist, the spread inside one run's set; "
             "chamfer well below pdist: the seeds agree"]
    if not rep:
        lines.append("(no algorithm with two finished runs on a problem)")
    for prob in sorted({p for p, _ in rep}):
        rs = sorted(((a, e) for (p, a), e in rep.items() if p == prob),
                    key=lambda t: t[1]["chamfer"])
        lines.append(f"\n{prob}")
        lines.append(f"  {'algorithm':<20}{'runs':>5}{'chamfer':>10}{'worst':>10}{'pdist':>10}")
        for a, e in rs:
            pd = f"{e['pdist']:>10.4f}" if e["pdist"] is not None else f"{'-':>10}"
            lines.append(f"  {mark_budget(a)[:20]:<20}{e['runs']:>5}{e['chamfer']:>10.4f}"
                         f"{e['chamfer_max']:>10.4f}{pd}")
    return "\n".join(lines)


def eps_table(rows: list) -> dict:
    """D6: the binary multiplicative ε between every two algorithms, per problem.

    {problem: {(A, B): median over every pair of runs (one of A, one of B) of
    I_ε×(A, B)}}, on the raw objective values: no reference front and no frame,
    and a rescaled objective changes nothing. A problem with a negative
    objective value anywhere is left out (the indicator needs values ≥ 0).
    """
    from . import metrics as M
    sets = _final_sets(rows)
    out: dict = {}
    for prob in sorted({p for p, _ in sets}):
        algs = sorted(a for p, a in sets if p == prob)
        cell: dict = {}
        ok = True
        for a in algs:
            for b in algs:
                if a == b:
                    continue
                vals = [M.eps_mult(Fa, Fb) for Fa, _ in sets[(prob, a)]
                        for Fb, _ in sets[(prob, b)]]
                if any(v is None for v in vals):
                    ok = False
                    break
                cell[(a, b)] = _median(vals)
            if not ok:
                break
        if ok and cell:
            out[prob] = cell
    return out


def format_eps_table(rep: dict) -> str:
    lines = ["binary multiplicative epsilon I(row, column): the factor by which the row's "
             "set, stretched, weakly dominates the column's; median over the pairs of "
             "runs, raw objective values (none negative). The row is better than the "
             "column where I(row, col) <= 1 < I(col, row), marked '<'"]
    if not rep:
        lines.append("(no problem whose objective values are all non-negative)")
    for prob, cell in rep.items():
        algs = sorted({a for a, _ in cell})
        lines.append(f"\n{prob}")
        lines.append(f"  {'':<16}" + "".join(f"{mark_budget(b)[:11]:>12}" for b in algs))
        for a in algs:
            row = []
            for b in algs:
                if a == b:
                    row.append(f"{'-':>12}")
                    continue
                v, w = cell[(a, b)], cell[(b, a)]
                mark = "<" if v <= 1.0 < w else " "
                row.append(f"{v:>11.4g}{mark}")
            lines.append(f"  {mark_budget(a)[:16]:<16}" + "".join(row))
    return "\n".join(lines)


def _kendall_tau(x, y):
    """Kendall's tau-b of two paired sequences; None when either is constant."""
    n = len(x)
    conc = disc = tx = ty = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx, dy = x[i] - x[j], y[i] - y[j]
            if dx == 0 and dy == 0:
                continue
            if dx == 0:
                tx += 1
            elif dy == 0:
                ty += 1
            elif (dx > 0) == (dy > 0):
                conc += 1
            else:
                disc += 1
    den = math.sqrt((conc + disc + tx) * (conc + disc + ty))
    return None if den == 0 else (conc - disc) / den


def magnitude_experiment(rows: list, *, max_m: int = 3) -> dict:
    """D7: would the magnitude of the dominated set rank the algorithms as hv_h does?

    On every problem with at most max_m objectives, each finished run's final
    population is measured by magnitude (metrics.magnitude) in hv_h's frame —
    objectives normalised by the problem's ideal and nadir, anchor 1 + 1/H —
    and by hv_h (recorded, or computed when the run has none). Per problem:
    the algorithms' median values ranked both ways, and Kendall's tau between
    the two orders; over the problems, each algorithm's mean rank by each.
    """
    import numpy as np
    from ..benchmarks import get as bench_get
    from . import metrics as M
    mag: dict = {}
    hvh: dict = {}
    for (prob, alg), runs in _final_sets(rows).items():
        p = bench_get(prob)
        if p.n_obj > max_m or p.ideal is None or p.nadir is None:
            continue
        scale = 1.0 + 1.0 / M.lattice_h(p.n_obj, int(p.pop_size))
        for F, _ in runs:
            mag.setdefault(prob, {}).setdefault(alg, []).append(
                M.magnitude(F, p.ideal, p.nadir, ref_scale=scale))
            h, _ = M.hypervolume(np.asarray(F, float), p.ideal, p.nadir, ref_scale=scale)
            hvh.setdefault(prob, {}).setdefault(alg, []).append(float(h))
    rm = problem_ranks(mag, lower_better=False)
    rh = problem_ranks(hvh, lower_better=False)
    per: dict = {}
    for prob in sorted(rm):
        algs = sorted(rm[prob])
        tau = _kendall_tau([rm[prob][a] for a in algs], [rh[prob][a] for a in algs])
        moved = sorted(((a, rh[prob][a], rm[prob][a]) for a in algs),
                       key=lambda t: -abs(t[1] - t[2]))[:3]
        per[prob] = {"tau": tau, "algorithms": len(algs), "moved": moved}
    mean: dict = {}
    for alg in sorted({a for d in rm.values() for a in d}):
        a = [rm[p][alg] for p in rm if alg in rm[p]]
        b = [rh[p][alg] for p in rh if alg in rh[p]]
        mean[alg] = (sum(b) / len(b), sum(a) / len(a), len(a))
    return {"per_problem": per, "mean_ranks": mean, "max_m": max_m}


def format_magnitude(rep: dict) -> str:
    per, mean = rep["per_problem"], rep["mean_ranks"]
    lines = [f"magnitude against hv_h on the problems with at most {rep['max_m']} objectives "
             "(both in hv_h's frame; experiment D7, not a campaign metric): Kendall's tau "
             "between the two orders of the algorithms' medians, and the algorithms whose "
             "rank moves most (hv_h rank -> magnitude rank)"]
    if not per:
        lines.append("(no finished run on such a problem)")
        return "\n".join(lines)
    taus = [e["tau"] for e in per.values() if e["tau"] is not None]
    for prob, e in per.items():
        t = f"{e['tau']:6.3f}" if e["tau"] is not None else "     -"
        moved = ", ".join(f"{mark_budget(a)[:14]} {h:g}->{m:g}"
                          for a, h, m in e["moved"] if h != m)
        lines.append(f"  {prob[:18]:<18} tau {t}   {moved or 'same order'}")
    if taus:
        lines.append(f"\nmean tau over {len(taus)} problems: {sum(taus) / len(taus):.3f}")
    lines.append(f"\n  {'algorithm':<20}{'rank hv_h':>10}{'rank mag':>10}{'problems':>10}")
    for alg, (h, m, n) in sorted(mean.items(), key=lambda t: t[1][0]):
        lines.append(f"  {mark_budget(alg)[:20]:<20}{h:>10.2f}{m:>10.2f}{n:>10}")
    return "\n".join(lines)
