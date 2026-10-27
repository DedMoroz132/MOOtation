#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Do two campaign result directories hold the same results?

    python tools/compare_results.py BEFORE AFTER            # every run, every file
    python tools/compare_results.py BEFORE AFTER --common-grid

For every run present in either (<problem>/<algorithm>/run_<seed>/), the files
are compared value for value: meta.json without what depends on the machine or
the moment (the clocks, the host, the revision, the process), the trajectory
record by record without its clock ("t"), final.csv, the archive files, the
archive checkpoints and snapshots (.npz, array by array). Beside the runs, the
CSV files the analyses wrote (--cover, --portfolio, ...: every *.csv under a
directory whose name starts with "_") are compared line by line, without the
column naming the code that wrote them (analysis_commit).

--common-grid compares the trajectories only at the evaluation counts both
have (a run recorded on another grid, [campaign] record_grid), and reads the
trajectory settings in meta.json, and a record's share of the evaluations
since the record before it (fail_share), as the grid's, not as results. --ignore-added
leaves out what only AFTER has (the values of a feature BEFORE did not have):
keys of meta.json and of the records, files, tables.

Exit status 0 when nothing differs, 1 otherwise; every difference is listed.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

# meta.json keys that say when, where and how fast a run ran, not what it found
# (fp_env: the process's floating-point state, whose sticky flags depend on
# what the process did before; the C++ engine's and Python's differ there)
VOLATILE = {"started", "finished", "seconds", "record_seconds", "pid", "host", "platform",
            "python", "revision", "timing", "timing_us", "mootation", "fp_env"}
# meta.json keys that describe the trajectory's grid (--common-grid)
GRID_KEYS = {"record_grid", "record_per_decade", "record_every", "records", "record_at",
             "trajectory_metrics_at_record_at"}
# keys of a trajectory record that are a share of the evaluations since the
# record before it, and so depend on the grid (--common-grid)
INTERVAL_KEYS = {"fail_share"}


def _text(path: Path) -> str:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            return fh.read()
    return path.read_text(encoding="utf-8")


IGNORE_ADDED = False
# columns of the analyses' tables that name the code that wrote them, not a result
VOLATILE_COLUMNS = {"analysis_commit"}


def _table(path: Path) -> list:
    """An analysis table's rows without its VOLATILE_COLUMNS."""
    import csv
    rows = list(csv.reader(_text(path).splitlines()))
    if not rows:
        return rows
    keep = [i for i, c in enumerate(rows[0]) if c not in VOLATILE_COLUMNS]
    return [[r[i] for i in keep if i < len(r)] for r in rows]


def _diff_json(a, b, where: str, out: list, skip=frozenset()) -> None:
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k in skip:
                continue
            if k not in a and IGNORE_ADDED:
                continue
            if k not in a or k not in b:
                out.append(f"{where}.{k}: only in {'after' if k not in a else 'before'}")
                continue
            _diff_json(a[k], b[k], f"{where}.{k}", out)
        return
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append(f"{where}: {len(a)} items before, {len(b)} after")
            return
        for i, (u, v) in enumerate(zip(a, b)):
            _diff_json(u, v, f"{where}[{i}]", out)
        return
    same = a == b or (isinstance(a, float) and isinstance(b, float) and a != a and b != b)
    if not same:
        out.append(f"{where}: {a!r} -> {b!r}")


def _records(path: Path) -> list:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            r.pop("t", None)
            rows.append(r)
    return rows


def _diff_npz(a: Path, b: Path, where: str, out: list) -> None:
    import numpy as np
    with np.load(a) as A, np.load(b) as B:
        for k in sorted(set(A.files) | set(B.files)):
            if k not in A.files or k not in B.files:
                out.append(f"{where}:{k}: only in one")
            elif A[k].shape != B[k].shape or A[k].tobytes() != B[k].tobytes():
                out.append(f"{where}:{k}: arrays differ")


def compare_run(a: Path, b: Path, common_grid: bool) -> list:
    out: list = []
    rel = f"{a.parents[1].name}/{a.parent.name}/{a.name}"
    ma = json.loads((a / "meta.json").read_text(encoding="utf-8")) if (a / "meta.json").is_file() else {}
    mb = json.loads((b / "meta.json").read_text(encoding="utf-8")) if (b / "meta.json").is_file() else {}
    skip = VOLATILE | (GRID_KEYS if common_grid else set())
    _diff_json(ma, mb, f"{rel} meta", out, frozenset(skip))
    ra, rb = _records(a / "trajectory.jsonl"), _records(b / "trajectory.jsonl")
    if common_grid:
        by_fe = {r["fe"]: r for r in rb}
        shared = [r for r in ra if r["fe"] in by_fe]
        if not shared:
            out.append(f"{rel} trajectory: no evaluation count in common")
        for r in shared:
            _diff_json(r, by_fe[r["fe"]], f"{rel} trajectory@{r['fe']}", out,
                       frozenset(INTERVAL_KEYS))
    else:
        _diff_json(ra, rb, f"{rel} trajectory", out)
    names = {p.name for p in a.iterdir()} | {p.name for p in b.iterdir()}
    for name in sorted(names - {"meta.json", "trajectory.jsonl"}):
        pa, pb = a / name, b / name
        if not pa.is_file() and IGNORE_ADDED:
            continue
        if not pa.is_file() or not pb.is_file():
            out.append(f"{rel}/{name}: only in {'after' if not pa.is_file() else 'before'}")
        elif name.endswith(".npz"):
            _diff_npz(pa, pb, f"{rel}/{name}", out)
        elif _text(pa) != _text(pb):
            out.append(f"{rel}/{name}: the contents differ")
    return out


def compare(before: Path, after: Path, common_grid: bool = False) -> list:
    out: list = []
    runs_a = {p.parent.relative_to(before) for p in before.glob("*/*/run_*/meta.json")}
    runs_b = {p.parent.relative_to(after) for p in after.glob("*/*/run_*/meta.json")}
    for r in sorted(runs_a ^ runs_b):
        out.append(f"{r}: only in {'before' if r in runs_a else 'after'}")
    for r in sorted(runs_a & runs_b):
        out += compare_run(before / r, after / r, common_grid)
    tables_a = {p.relative_to(before) for p in before.glob("_*/**/*.csv")}
    tables_b = {p.relative_to(after) for p in after.glob("_*/**/*.csv")}
    for t in sorted(tables_a ^ tables_b):
        if t not in tables_a and IGNORE_ADDED:
            continue
        out.append(f"{t}: only in {'before' if t in tables_a else 'after'}")
    for t in sorted(tables_a & tables_b):
        if _table(before / t) != _table(after / t):
            out.append(f"{t}: the contents differ")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("before")
    ap.add_argument("after")
    ap.add_argument("--common-grid", action="store_true",
                    help="trajectories only at the evaluation counts both recorded")
    ap.add_argument("--ignore-added", action="store_true",
                    help="leave out what only AFTER has (a new feature's values)")
    args = ap.parse_args(argv)
    global IGNORE_ADDED
    IGNORE_ADDED = args.ignore_added
    before, after = Path(args.before), Path(args.after)
    diffs = compare(before, after, args.common_grid)
    for d in diffs:
        print(d)
    runs = len(list(before.glob("*/*/run_*/meta.json")))
    print(f"{runs} runs compared: {len(diffs)} differences")
    return 1 if diffs else 0


if __name__ == "__main__":
    sys.exit(main())
