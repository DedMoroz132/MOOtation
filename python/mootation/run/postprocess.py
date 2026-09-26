# SPDX-License-Identifier: Apache-2.0
"""Reading a finished campaign beyond medians and mean ranks.

  * gap_to_best — each run's distance to the best value any run reached on
    that problem: a common zero across problems whose raw values differ by
    orders of magnitude, and the targets COCO measures runtimes against.
  * zero_share — the share of seeds whose hypervolume is exactly 0: a split
    between seeds that a median hides (DTLZ3, DTLZ4, IDTLZ1_5D).
  * runtime_to_target / runtime_ecdf — how many evaluations a run needed to
    reach a target, and the ECDF of that over runs and targets, with the way
    between two recorded points STATED: "step" charges a target to the first
    record that reached it (no assumption, a pessimistic runtime); "linear"
    interpolates the evaluation count on the metric between the last record
    short of the target and the first past it. With 11-12 records per run the
    two differed by up to 49 % (research note, 2026-09); the logarithmic
    recording grid narrows that, the method is still printed with every result.
  * BUDGET_SCHEDULED — the algorithms whose behaviour depends on the SHARE of
    the budget spent (a t/t_max schedule): their curves at different budgets
    are not the same run cut short, so a 10 000- and a 25 000-evaluation
    trajectory of one of them must be overlaid by fraction of budget, never by
    absolute evaluation count.
"""

from __future__ import annotations

import math
from pathlib import Path

from .metric_names import HIGHER_IS_BETTER

# Checked 2026-09-22 against the code: 18 cores have set_t_max; these 16 use
# t/t_max in a schedule. Of the other two, IF-MaOEA stores t_max and never
# reads it, and CLIA uses it once, for theta = min(20, max(5, ceil(t_max*N /
# 2e4))) (§IV-B), which is 5 at every budget up to 100 000 evaluations — so on
# the 10 000 / 25 000 / 50 000 ladder CLIA behaves as unscheduled. RVEA*
# (rvea_star, added 2026-09-27) has RVEA's APD schedule: 19 and 17.
BUDGET_SCHEDULED = frozenset({
    "rvea", "rvea_star", "moead_awa", "adaw", "dea_gng", "mbra", "nrv_moea",
    "hlmea", "dhea", "moead_ds", "srv", "srv_nsga3", "dcea", "maoea_3c",
    "moead_m2m", "moead_am2m", "liu_gu2011",
})

INTERPOLATIONS = ("step", "linear")


def _lower_better(metric: str) -> bool:
    return metric not in HIGHER_IS_BETTER


def best_known(rows: list, metric: str) -> dict:
    """{problem: the best final `metric` any finished run reached}."""
    lower = _lower_better(metric)
    best: dict = {}
    for r in rows:
        if r.get("status") != "done":
            continue
        v = (r.get("final") or {}).get(metric)
        if v is None or not math.isfinite(float(v)):
            continue
        v = float(v)
        p = r["problem"]
        if p not in best or (v < best[p] if lower else v > best[p]):
            best[p] = v
    return best


def gap_to_best(rows: list, metric: str) -> dict:
    """{problem: {algorithm: [gap of each run]}}, gap >= 0, 0 = the best known."""
    lower = _lower_better(metric)
    best = best_known(rows, metric)
    out: dict = {}
    for r in rows:
        if r.get("status") != "done" or r["problem"] not in best:
            continue
        v = (r.get("final") or {}).get(metric)
        if v is None or not math.isfinite(float(v)):
            continue
        g = (float(v) - best[r["problem"]]) if lower else (best[r["problem"]] - float(v))
        out.setdefault(r["problem"], {}).setdefault(r["algorithm"], []).append(max(0.0, g))
    return out


def zero_share(rows: list, metric: str = "hv") -> dict:
    """{problem: {algorithm: share of its finished runs whose final metric is exactly 0}}."""
    count: dict = {}
    for r in rows:
        if r.get("status") != "done":
            continue
        v = (r.get("final") or {}).get(metric)
        if v is None:
            continue
        c = count.setdefault(r["problem"], {}).setdefault(r["algorithm"], [0, 0])
        c[0] += float(v) == 0.0
        c[1] += 1
    return {p: {a: z / n for a, (z, n) in algs.items()} for p, algs in count.items()}


def runtime_to_target(records: list, metric: str, target: float, *,
                      interpolation: str = "step") -> float | None:
    """Evaluations a run needed for `metric` to reach `target`; None if it never did.

    `records` are trajectory lines in order ({"fe": ..., metric: ...}). "Reach"
    is <= target for a lower-is-better metric and >= for the others.
    """
    if interpolation not in INTERPOLATIONS:
        raise ValueError(f"interpolation must be one of {INTERPOLATIONS}")
    lower = _lower_better(metric)

    def reached(v):
        return v <= target if lower else v >= target
    prev = None
    for rec in records:
        v = rec.get(metric)
        if v is None or not math.isfinite(float(v)):
            continue
        v, fe = float(v), float(rec["fe"])
        if reached(v):
            if interpolation == "step" or prev is None or prev[1] == v:
                return fe
            fe0, v0 = prev
            t = (target - v0) / (v - v0)            # where between the two records
            return fe0 + min(1.0, max(0.0, t)) * (fe - fe0)
        prev = (fe, v)
    return None


def runtime_ecdf(runs: list, metric: str, targets: list, *, interpolation: str = "step",
                 grid: list | None = None) -> dict:
    """The ECDF of runtimes over (run, target) pairs, COCO-style.

    `runs` is a list of trajectories (each a list of records); every run is
    asked every target. Returns {"interpolation", "fe": grid, "share": the
    share of pairs solved within each grid count, "pairs": their number}. A
    pair never solved counts in the denominator and never in the numerator.
    """
    times = []
    for recs in runs:
        for t in targets:
            times.append(runtime_to_target(recs, metric, t, interpolation=interpolation))
    solved = sorted(x for x in times if x is not None)
    if grid is None:
        grid = sorted(set(solved)) or [0.0]
    share = []
    j = 0
    for g in grid:
        while j < len(solved) and solved[j] <= g:
            j += 1
        share.append(j / len(times) if times else 0.0)
    return {"interpolation": interpolation, "fe": list(grid), "share": share,
            "pairs": len(times)}


def read_trajectory(run_dir: Path) -> list:
    import json
    path = Path(run_dir) / "trajectory.jsonl"
    out = []
    if not path.is_file():
        return out
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except ValueError:
                    break                              # a torn last line
    return out


def budget_note(algorithm: str) -> str:
    """'*' for an algorithm whose curve must be read by fraction of budget, else ''."""
    return "*" if algorithm in BUDGET_SCHEDULED else ""
