#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Do the benchmark problems give the values they gave at a baseline commit?

    python tools/compat_problems.py [--baseline REF] [--points 1000] [--only PREFIX[,PREFIX]]
                                    [--tree DIR] [--cache DIR]

Every problem of the registry is evaluated at the same points twice, once by
the python/ tree of the BASELINE commit (written out with git archive, run in
a separate interpreter) and once by the working tree's, and the values are
compared bit for bit: the frame (ideal, nadir, the raw hypervolume reference),
the objectives through `evaluate`, and through
`evaluate_batch` where the working tree has it; the constraints through
`constraints` (and `constraints_batch`, and both through
`evaluate_with_constraints`); the equalities; a noisy copy's
noise-free values; a run's own evaluator (make_evaluator, seed 1) over the
points in order. The points: `points` per problem, uniform in the box (the
integer, discrete, binary and categorical variables drawn from their values),
the box's corners and centre first. The same interpreter, NumPy and machine
run both sides, so a difference is a difference of the code.

--tree checks another python/ tree than the checkout's (a copy taken while
the checkout changes); --cache keeps the baseline's dump there and reads it
again while the baseline, the points and the prefix stay the same (a dump of
the whole registry serves every prefix).

A problem that only one side has is listed, not failed. Exit status: 0 when
every value is the same, 1 otherwise. On Windows and elsewhere alike; nothing
compiled is needed (the problems are Python).
"""

from __future__ import annotations

import argparse
import io
import json
import os
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# The dump, run in a fresh interpreter with the tree to check first on sys.path.
DUMP = r'''
import hashlib, json, math, sys, zlib
import numpy as np
sys.path.insert(0, sys.argv[1])
from mootation.benchmarks import registry as R

n_points, only, out = int(sys.argv[2]), sys.argv[3], sys.argv[4]


def points(p, n):
    rng = np.random.default_rng(zlib.crc32(p.name.encode()))
    lo = np.array([b[0] for b in p.bounds], float)
    hi = np.array([b[1] for b in p.bounds], float)
    X = [lo, hi, (lo + hi) / 2] + [lo + rng.random(len(lo)) * (hi - lo) for _ in range(n)]
    rows = []
    for x in X:
        row = []
        for j, v in enumerate(x):
            t = p.var_types[j] if p.var_types is not None else "real"
            if t == "int":
                row.append(int(round(min(max(v, math.ceil(lo[j])), math.floor(hi[j])))))
            elif t == "bin":
                row.append(int(rng.integers(0, 2)))
            elif isinstance(t, (tuple, list)) and t[0] == "discrete":
                row.append(float(t[1][int(rng.integers(0, len(t[1])))]))
            elif isinstance(t, (tuple, list)) and t[0] == "cat":
                labels = t[1] if isinstance(t[1], (list, tuple)) else list(range(int(t[1])))
                row.append(labels[int(rng.integers(0, len(labels)))])
            else:
                row.append(float(v))
        rows.append(row)
    return rows


def digest(values):
    """A text of every value, exact (repr of the floats), and of every failure."""
    h = hashlib.sha256()
    for v in values:
        h.update((v if isinstance(v, str) else repr([float(u) for u in v])).encode())
        h.update(b"\n")
    return h.hexdigest()


def each(fn, X):
    out = []
    for x in X:
        try:
            out.append(list(fn(list(x))))
        except Exception as e:                   # noqa: BLE001
            out.append(f"raises {type(e).__name__}")
    return out


res = {}
for name in R.names():
    if only and not name.startswith(tuple(only.split(","))):
        continue
    p = R.get(name)
    X = points(p, n_points)
    d = {"frame": repr([list(map(float, p.ideal)), list(map(float, p.nadir)),
                        list(map(float, p.hv_ref_raw))]),
         "evaluate": digest(each(p.evaluate, X))}
    if p.has_cons:
        d["constraints"] = digest(each(p.constraints, X))
    if p.equalities is not None:
        d["equalities"] = digest(each(p.equalities, X))
    if p.true_objectives is not None:
        d["true_objectives"] = digest(each(p.true_objectives, X))
    if p.make_evaluator is not None:
        d["evaluator"] = digest(each(p.make_evaluator(1), X))
    for key, fn in (("evaluate_batch", getattr(p, "evaluate_batch", None)),
                    ("constraints_batch", getattr(p, "constraints_batch", None))):
        if fn is not None:
            try:
                rows = [list(r) for r in fn(X)]
            except Exception as e:               # noqa: BLE001
                rows = [f"raises {type(e).__name__}"]
            d[key] = digest(rows)
    both = getattr(p, "evaluate_with_constraints", None)
    if both is not None:
        try:
            F, G = both(X)
            d["with_constraints:F"] = digest([list(r) for r in F])
            d["with_constraints:G"] = digest([list(r) for r in G])
        except Exception as e:                   # noqa: BLE001
            d["with_constraints:F"] = d["with_constraints:G"] = f"raises {type(e).__name__}"
    res[name] = d
json.dump(res, open(out, "w"))
'''


def baseline_ref() -> str:
    for line in (ROOT / "tests" / "compat_baseline.txt").read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            return line
    raise SystemExit("tests/compat_baseline.txt names no baseline commit")


def export_python(ref: str, dest: Path) -> Path:
    archive = subprocess.run(["git", "archive", "--format=tar", ref, "python"], cwd=ROOT,
                             capture_output=True, check=True).stdout
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        tar.extractall(dest, filter="data")
    return dest / "python"


def dump(tree: Path, n: int, only: str, out: Path) -> dict:
    env = dict(os.environ, PYTHONPATH="")
    script = out.with_suffix(".py")
    script.write_text(DUMP, encoding="utf-8")
    subprocess.run([sys.executable, str(script), str(tree), str(n), only, str(out)],
                   check=True, env=env)
    return json.loads(out.read_text(encoding="utf-8"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--baseline", help="commit to compare against (default: tests/compat_baseline.txt)")
    ap.add_argument("--points", type=int, default=1000)
    ap.add_argument("--only", default="",
                    help="only the problems whose name starts so (several: comma-separated)")
    ap.add_argument("--tree", help="the python/ tree to check (default: the checkout's)")
    ap.add_argument("--cache", help="keep the baseline's dump here and reuse it")
    args = ap.parse_args(argv)
    ref = args.baseline or baseline_ref()
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        cached = None
        if args.cache:
            sha = subprocess.run(["git", "rev-parse", ref], cwd=ROOT, capture_output=True,
                                 text=True, check=True).stdout.strip()
            cached = Path(args.cache) / f"problems_{sha[:12]}_{args.points}_{args.only or 'all'}.json"
            cached.parent.mkdir(parents=True, exist_ok=True)
        whole = cached.with_name(cached.name.replace(f"_{args.only}.json", "_all.json"))             if cached is not None and args.only else None
        if cached is not None and cached.is_file():
            before = json.loads(cached.read_text(encoding="utf-8"))
        elif whole is not None and whole.is_file():        # the whole registry's, cut down
            prefixes = tuple(args.only.split(","))
            before = {k: v for k, v in json.loads(whole.read_text(encoding="utf-8")).items()
                      if k.startswith(prefixes)}
        else:
            before = dump(export_python(ref, td / "base"), args.points, args.only, td / "before.json")
            if cached is not None:
                cached.write_text(json.dumps(before), encoding="utf-8")
        tree = Path(args.tree) if args.tree else ROOT / "python"
        after = dump(tree, args.points, args.only, td / "after.json")
    bad = 0
    for name in sorted(set(before) | set(after)):
        if name not in before or name not in after:
            print(f"{'new' if name in after else 'gone':<8} {name}")
            continue
        b, a = before[name], after[name]
        for key, value in b.items():
            if a.get(key) != value:
                bad += 1
                print(f"CHANGED  {name}: {key}")
        # what only the working tree has must agree with what both have
        for key, same_as in (("evaluate_batch", "evaluate"), ("constraints_batch", "constraints"),
                             ("with_constraints:F", "evaluate"),
                             ("with_constraints:G", "constraints")):
            if key in a and same_as in a and a[key] != a[same_as]:
                bad += 1
                print(f"CHANGED  {name}: {key} is not {same_as} row by row")
    print(f"{len(before)} problems against {ref}: {bad} differences")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
