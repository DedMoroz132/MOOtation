# SPDX-License-Identifier: Apache-2.0
"""One page on a campaign (task 16, §4): `mootation report <results>`, from
run folders or a store alike (campaign.scan_results), written as one HTML
file with its plots drawn inline (SVG; nothing beyond the standard library and
NumPy).

The problems are never pooled across families: every table and plot is per
family of problems (families.family) and number of objectives M.

  levels    a run meets a level at b evaluations when its IGD+ (igdp_norm)
            there is at most level x the problem's floor (cover.py, THE
            FLOOR: the IGD+ of the best N points of the reference front, N
            the run's population; the DSS floor in the archive scenario).
            Levels 1.1, 2, 4, 10 by default (--levels). A problem without a
            reference front has no floor and is left out of these.
  ECDF      per algorithm, the share of its (run, level) pairs met at each
            point of the budget grid (campaign.budget_grid), read as
            --cover reads a run at a budget (cover.read_at: the first record
            at or after it).
  tables    per level, the share of the group's problems on which the median
            over the seeds meets it, at each budget.
  curves    the median over the runs of IGD+ / floor at each budget.
  archive   the final population's IGD+ beside the archive scenario's (the
            run archive reduced to N), medians, where the runs kept both.
  failures  every run not "done": failed, timeout (run_timeout), an
            incompatible pair (meta "incompatible"), still running.
"""

from __future__ import annotations

import html
import json
import math
from pathlib import Path

LEVELS = (1.1, 2.0, 4.0, 10.0)
COLOURS = ("#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b", "#e377c2",
           "#7f7f7f", "#bcbd22", "#17becf")


def _median(v):
    v = sorted(x for x in v if x is not None and not (isinstance(x, float) and math.isnan(x)))
    return (v[(len(v) - 1) // 2] + v[len(v) // 2]) / 2 if v else None


def _svg_lines(series: dict, xs: list, *, ylog: bool, ylabel: str, title: str,
               ylim=None) -> str:
    """A plot of {name: [y per x]} against xs on a log x axis, as SVG."""
    W, L, R, T, B = 560, 60, 150, 26, 40
    H = max(300, T + 13 * len(series) + B)          # the legend, one line a series
    pts = [y for ys in series.values() for y in ys if y is not None and (y > 0 or not ylog)]
    if not pts or len(xs) < 2:
        return f"<p><i>{html.escape(title)}: nothing to draw</i></p>"
    lo, hi = ylim if ylim else (min(pts), max(pts))
    if ylog:
        lo, hi = math.log10(lo), math.log10(hi)
    if hi <= lo:
        hi = lo + 1.0
    x0, x1 = math.log10(xs[0]), math.log10(xs[-1])

    def X(x):
        return L + (math.log10(x) - x0) / (x1 - x0) * (W - L - R)

    def Y(y):
        v = math.log10(y) if ylog else y
        return T + (1 - (v - lo) / (hi - lo)) * (H - T - B)
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-size="11">',
           f'<text x="{L}" y="15" font-weight="bold">{html.escape(title)}</text>',
           f'<rect x="{L}" y="{T}" width="{W - L - R}" height="{H - T - B}" fill="none" stroke="#999"/>']
    for x in xs:
        if x in (xs[0], xs[-1]) or math.log10(x) == int(math.log10(x)):
            out.append(f'<text x="{X(x):.1f}" y="{H - B + 14}" text-anchor="middle">{x:g}</text>')
    for k in range(5):
        v = lo + (hi - lo) * k / 4
        lab = f"{10 ** v:.3g}" if ylog else f"{v:.2f}"
        y = T + (1 - k / 4) * (H - T - B)
        out.append(f'<text x="{L - 4}" y="{y + 4:.1f}" text-anchor="end">{lab}</text>')
    out.append(f'<text x="12" y="{(T + H - B) / 2:.0f}" transform="rotate(-90 12 {(T + H - B) / 2:.0f})" '
               f'text-anchor="middle">{html.escape(ylabel)}</text>')
    out.append(f'<text x="{(L + W - R) / 2:.0f}" y="{H - 6}" text-anchor="middle">evaluations</text>')
    for i, (name, ys) in enumerate(series.items()):
        c = COLOURS[i % len(COLOURS)]
        dash = ("", ' stroke-dasharray="6 3"', ' stroke-dasharray="2 2"', ' stroke-dasharray="8 2 2 2"')[
            (i // len(COLOURS)) % 4]                     # past ten series: the colours again, dashed
        path = [f"{X(x):.1f},{Y(y):.1f}" for x, y in zip(xs, ys)
                if y is not None and (y > 0 or not ylog)]
        if path:
            out.append(f'<polyline fill="none" stroke="{c}" stroke-width="1.6"{dash} '
                       f'points="{" ".join(path)}"/>')
        out.append(f'<text x="{W - R + 8}" y="{T + 12 + 13 * i}" fill="{c}">{html.escape(name)}</text>')
    out.append("</svg>")
    return "\n".join(out)


def _table(head: list, rows: list) -> str:
    out = ["<table><tr>" + "".join(f"<th>{html.escape(str(h))}</th>" for h in head) + "</tr>"]
    for r in rows:
        out.append("<tr>" + "".join(f"<td>{html.escape(str(c))}</td>" for c in r) + "</tr>")
    return "\n".join(out) + "</table>"


def build(root: Path, *, levels=LEVELS, scenario: str = "population", workers: int = 1) -> str:
    """The report of the campaign under root, as HTML text."""
    from ..families import family
    from . import cover as V
    from .campaign import budget_grid, main_rows, read_trajectory, scan_results, scenario_rows
    root = Path(root)
    rows_all = main_rows(scan_results(root))
    rows = scenario_rows([r for r in rows_all if r["status"] == "done"], scenario)
    try:
        n_ref = int(json.loads((root / "campaign.json").read_text(encoding="utf-8"))["spec"]["n_ref"])
    except (OSError, ValueError, KeyError):
        n_ref = 1000
    sizes: dict = {}
    for r in rows:
        if r.get("pop"):
            sizes.setdefault(r["problem"], set()).add(int(r["pop"]))
    floor_dir = root / "_report"
    floor_dir.mkdir(exist_ok=True)
    fl = V.floors(sizes, n_ref, floor_dir / "igdp_floor.csv", workers=workers)
    which = V.level_floor(scenario)
    limit = max((int(r.get("budget_nominal") or r.get("budget_fe") or 0) for r in rows), default=0)
    grid = [b for b in budget_grid(limit) if b <= limit] if limit else []

    # every run's IGD+ / floor at the grid
    by_group: dict = {}
    for r in rows:
        f = (fl.get((r["problem"], int(r.get("pop") or 0))) or {}).get(which)
        if not f:
            continue
        budget = int(r.get("budget_nominal") or r.get("budget_fe") or 0)
        traj = read_trajectory(r["dir"]) if any(V.needs_trajectory(b, budget, r.get("fe"))
                                                for b in grid) else ()
        at = r.get("archive_at") if scenario == "archive" else None
        ratio = []
        for b in grid:
            v = V.read_at(b, budget, r.get("fe"), r["final"], traj, at, metrics=("igdp_norm",))["igdp_norm"]
            ratio.append(None if v is None else v / f)
        key = (family(r["problem"]), int(r.get("n_objs") or 0))
        by_group.setdefault(key, {}).setdefault(r["algorithm"], []).append((r["problem"], ratio))

    parts = ["<!doctype html><meta charset='utf-8'><title>Campaign report</title>",
             "<style>body{font-family:sans-serif;max-width:1150px;margin:20px}table{border-collapse:collapse;"
             "margin:6px 0 16px}td,th{border:1px solid #ccc;padding:2px 6px;text-align:right}"
             "th{background:#f3f3f3}td:first-child{text-align:left}</style>",
             f"<h1>Campaign report: {html.escape(root.name)}</h1>",
             f"<p>{len(rows_all)} runs, {len(rows)} done; scenario {scenario}; levels "
             f"{', '.join(f'{x:g}' for x in levels)} x the IGD+ floor ({which}); budgets {grid}. "
             "Families are never pooled.</p>"]
    for (fam, m) in sorted(by_group):
        algs = by_group[(fam, m)]
        names = sorted(algs)
        problems = sorted({p for runs in algs.values() for p, _ in runs})
        parts.append(f"<h2>{html.escape(fam)}, M = {m} ({len(problems)} problems)</h2>")
        ecdf = {}
        for a in names:
            share = []
            for k in range(len(grid)):
                pairs = [(x[k] is not None and x[k] <= lev) for _, x in algs[a] for lev in levels]
                share.append(sum(pairs) / len(pairs) if pairs else None)
            ecdf[a] = share
        parts.append(_svg_lines(ecdf, grid, ylog=False, ylabel="share of (run, level) met",
                                title="ECDF over the levels", ylim=(0.0, 1.0)))
        curves = {a: [_median([x[k] for _, x in algs[a]]) for k in range(len(grid))] for a in names}
        parts.append(_svg_lines(curves, grid, ylog=True, ylabel="IGD+ / floor (median)",
                                title="median IGD+ / floor"))
        for lev in levels:
            body = []
            for a in names:
                cells = []
                for k in range(len(grid)):
                    per = {}
                    for p, x in algs[a]:
                        per.setdefault(p, []).append(x[k])
                    med = [_median(v) for v in per.values()]
                    met = [v is not None and v <= lev for v in med]
                    cells.append(f"{sum(met) / len(problems):.2f}" if med else "")
                body.append([a] + cells)
            parts.append(f"<h3>share of the problems met at {lev:g} x floor (median over the seeds)</h3>")
            parts.append(_table(["algorithm"] + [f"{b:g}" for b in grid], body))
    # the archive beside the population
    body = []
    for key in sorted({(family(r["problem"]), int(r.get("n_objs") or 0), r["algorithm"]) for r in rows_all
                       if r["status"] == "done"}):
        fam, m, a = key
        rs = [r for r in rows_all if r["status"] == "done" and r["algorithm"] == a
              and family(r["problem"]) == fam and int(r.get("n_objs") or 0) == m]
        pop = _median([(r.get("final") or {}).get("igdp_norm") for r in rs])
        arc = _median([(r.get("final_archive") or {}).get("igdp_norm") for r in rs])
        if pop is not None or arc is not None:
            body.append([fam, m, a, len(rs), "" if pop is None else f"{pop:.4g}",
                         "" if arc is None else f"{arc:.4g}"])
    parts.append("<h2>The archive beside the population (final IGD+, medians)</h2>")
    parts.append(_table(["family", "M", "algorithm", "runs", "population", "archive"], body))
    # what did not finish
    body = []
    for r in sorted((r for r in rows_all if r["status"] != "done"),
                    key=lambda r: (r["status"], r["problem"], r["algorithm"], r.get("seed") or 0)):
        try:
            meta = json.loads((r["dir"] / "meta.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            meta = {}
        kind = ("incompatible" if meta.get("incompatible") else r["status"])
        why = meta.get("error") or (f"{meta.get('timeout_seconds'):g} s, last record at "
                                    f"{meta.get('fe')} evaluations" if r["status"] == "timeout" else "")
        body.append([kind, r["problem"], r["algorithm"], r.get("seed"), str(why)[:160]])
    parts.append(f"<h2>Not done ({len(body)})</h2>")
    parts.append(_table(["kind", "problem", "algorithm", "seed", "why"], body) if body else "<p>none</p>")
    return "\n".join(parts) + "\n"


def main(argv: list | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="mootation report", description="one page on a campaign (task 16, §4)")
    ap.add_argument("campaign", help="the campaign's results folder (run folders or a store)")
    ap.add_argument("--out", default="", help="the HTML file (default: <campaign>/_report/report.html)")
    ap.add_argument("--levels", default=",".join(f"{x:g}" for x in LEVELS),
                    help="levels as multiples of the IGD+ floor")
    ap.add_argument("--scenario", default="population", choices=("population", "archive"))
    ap.add_argument("--workers", type=int, default=1, help="for the floors not yet known")
    a = ap.parse_args(argv)
    root = Path(a.campaign)
    text = build(root, levels=tuple(float(x) for x in a.levels.split(",")), scenario=a.scenario,
                 workers=a.workers)
    out = Path(a.out) if a.out else root / "_report" / "report.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(out)
    return 0
