#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Where a campaign run's time goes, per component, for a set of algorithms x
problems x seeds at one budget.

    python tools/perf/profile.py OUT                       # the default set
    python tools/perf/profile.py OUT --algorithms nsga2,moead_de --problems ZDT1 --seeds 1
    python tools/perf/profile.py OUT --set record_grid=budgets --set archive_mode=posthoc
    python tools/perf/profile.py OUT --report               # read OUT again, run nothing

Every run is an ordinary campaign job (mootation.run.campaign.run_job), so what
is measured is what a campaign spends. The job records its own clocks in
meta["timing_us"]; this script writes the campaign config, runs it and turns the
clocks into components:

    evaluation the problem's objectives and constraints: Python's, or the C++
               suite's inside the core (the direct path, timing native)
    crossing   the calls from the core into Python and back: Result.callback_seconds
               less the time measured inside them — the binding's conversions,
               minimize()'s wrappers and the campaign evaluator's own bookkeeping
               (on the direct path, the rows handed over a batch at a time)
    core       the algorithm: minimize() less the evaluator calls, the C++
               evaluations and the records
    archive    the run archive and the oracle (and the constraints the archive
               asks for again)
    records    the trajectory records: indicators, snapshots, checkpoints
    final      the final indicators and the archive's selection
    write      the files after the run
    setup      the reference data before the run (the front, the frame, the
               Pareto set: built, or read from the campaign's cache)
    other      the rest of the job

OUT gets profile.toml (the campaign), results/ (its runs), profile.csv (one
row per run, seconds) and profile.md (median over the seeds per algorithm and
problem). Timings depend on the machine and on what else runs there: compare
two tables only from the same machine, with nothing heavy beside them.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

ALGORITHMS = ("nsga2", "ibea_eplus", "spea2_sde", "sms_emoa", "moead_de", "mosa")
PROBLEMS = ("ZDT1", "DTLZ2_3D", "DTLZ2_5D", "WFG4_3D", "ZCAT9_10D", "bbobbiobj55_n05_2D",
            "DAS-CMOP8_t01", "MaF9_10D")
SEEDS = (1, 2, 3)
BUDGET = 25_000
COMPONENTS = ("evaluation", "crossing", "core", "archive", "records", "final", "write", "setup",
              "other")


def toml_value(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(toml_value(x) for x in v) + "]"
    return json.dumps(str(v))


def parse_set(items) -> dict:
    """KEY=VALUE pairs for [campaign], VALUE read as TOML would read it (a bare
    word, as a shell leaves a quoted one, as a string)."""
    import tomllib
    out = {}
    for item in items or ():
        key, _, value = item.partition("=")
        try:
            out[key.strip()] = tomllib.loads(f"v = {value.strip()}")["v"]
        except tomllib.TOMLDecodeError:
            out[key.strip()] = value.strip()
    return out


def write_config(out: Path, algorithms, problems, seeds, budget, extra: dict) -> Path:
    campaign = {"out": "results", "budget_fe": int(budget), "seeds": list(seeds)}
    campaign.update(extra)
    lines = ["# written by tools/perf/profile.py", "[run]", 'name = "perf"', "",
             "[problem]", 'kind = "builtin"', "", "[benchmarks]",
             f"problems = {toml_value(list(problems))}", "", "[campaign]"]
    lines += [f"{k} = {toml_value(v)}" for k, v in campaign.items()]
    for a in algorithms:
        lines += ["", "[[algorithms]]", f'name = "{a}"', "pop = 0", "gens = 0"]
    path = out / "profile.toml"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def components(meta: dict) -> dict | None:
    """The components of one run, in seconds (module docstring); None without
    clocks (a run of a version that kept none). meta["timing_us"], whole
    microseconds (task 17), or meta["timing"], seconds (before)."""
    t = meta.get("timing")
    if meta.get("timing_us"):
        t = {k: (v if k == "calls" else v / 1e6) for k, v in meta["timing_us"].items()}
    if not t:
        return None
    total = float(meta.get("seconds", 0.0))
    native = float(t.get("native", 0.0))          # the C++ evaluations, outside the callbacks
    inside = (t["evaluate"] - native + t["constraints"] + t["feasible"] + t["archive"]
              + t["checkpoints"])
    c = {
        "evaluation": t["evaluate"] + t["constraints"],
        "crossing": max(0.0, t["callbacks"] - inside),
        "core": max(0.0, t["minimize"] - t["callbacks"] - native - t["records"]),
        "archive": t["archive"] + t["feasible"],
        "records": t["records"] + t["checkpoints"],
        "final": t["final"],
        "write": t["write"],
    }
    c["other"] = max(0.0, total - t["minimize"] - t["final"] - t["write"])
    c["setup"] = float(t.get("setup", 0.0))       # before the run's own clock starts
    c["total"] = total + c["setup"]
    c["calls"] = int(t.get("calls", 0))
    return c


def collect(results: Path) -> list[dict]:
    rows = []
    for meta_path in sorted(results.glob("*/*/run_*/meta.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("status") != "done":
            continue
        c = components(meta)
        if c is None:
            continue
        rows.append(dict(problem=meta["problem"], algorithm=meta["algorithm"], seed=meta["seed"],
                         fe=meta.get("fe"), **c))
    return rows


def write_tables(out: Path, rows: list[dict]) -> str:
    keys = ["problem", "algorithm", "seed", "fe", "total", "calls", *COMPONENTS]
    with (out / "profile.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(r[k], 4) if isinstance(r[k], float) else r[k]) for k in keys})
    groups: dict = {}
    for r in rows:
        groups.setdefault((r["problem"], r["algorithm"]), []).append(r)
    head = ("| problem | algorithm | runs | total, s | calls | "
            + " | ".join(f"{c}, %" for c in COMPONENTS) + " |")
    lines = [head, "|" + "---|" * (5 + len(COMPONENTS))]
    for (prob, alg), rs in sorted(groups.items()):
        med = {k: statistics.median(r[k] for r in rs) for k in ("total", "calls", *COMPONENTS)}
        tot = med["total"] or 1.0
        lines.append(f"| {prob} | {alg} | {len(rs)} | {med['total']:.2f} | {int(med['calls'])} | "
                     + " | ".join(f"{100.0 * med[c] / tot:.0f}" for c in COMPONENTS) + " |")
    whole = {c: sum(r[c] for r in rows) for c in ("total", *COMPONENTS)}
    lines += ["", f"all runs: {len(rows)}, {whole['total']:.1f} s; "
              + ", ".join(f"{c} {100.0 * whole[c] / (whole['total'] or 1.0):.0f} %"
                          for c in COMPONENTS)]
    text = "\n".join(lines) + "\n"
    (out / "profile.md").write_text(text, encoding="utf-8")
    return text


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("out", help="directory for the config, the runs and the tables")
    ap.add_argument("--algorithms", default=",".join(ALGORITHMS))
    ap.add_argument("--problems", default=",".join(PROBLEMS))
    ap.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    ap.add_argument("--budget", type=int, default=BUDGET)
    ap.add_argument("--workers", type=int, default=1,
                    help="worker processes; 1 keeps the clocks cleanest")
    ap.add_argument("--set", action="append", metavar="KEY=VALUE",
                    help="a [campaign] key, e.g. --set record_grid=\"budgets\"")
    ap.add_argument("--force", action="store_true", help="run the jobs that are done again")
    ap.add_argument("--report", action="store_true", help="only read OUT/results again")
    args = ap.parse_args(argv)

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if not args.report:
        from mootation.run.campaign import run_campaign
        from mootation.run.config import load, validate
        path = write_config(out, args.algorithms.split(","), args.problems.split(","),
                            [int(s) for s in args.seeds.split(",")], args.budget,
                            parse_set(args.set))
        cfg = load(path)
        problems = validate(cfg)
        if problems:
            raise SystemExit("the campaign config is not valid:\n  " + "\n  ".join(problems))
        run_campaign(cfg, workers=args.workers, force=args.force)
    print(write_tables(out, collect(out / "results")), end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
