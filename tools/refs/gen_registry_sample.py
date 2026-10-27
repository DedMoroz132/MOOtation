# SPDX-License-Identifier: Apache-2.0
"""tests/data/registry_sample.json: a few problems of the exported registry
(run/registry_export.py) with the values the registry's problems give at one
point each, the reference of tests/test_registry.cpp
(include/mootation/campaign/registry.hpp): a shifted problem (its centre
handed over), a constrained one (C-DTLZ's constants), a rotated one (its
matrix drawn by C++), a bbob-biobj one (its instance drawn by C++), one
the C++ engine refuses, and two failing copies (the band at the faces, the
random balls) with the values at points inside and outside the region
(null: NaN, a failed evaluation), and a noisy copy of each kind with the
values its run evaluator (seed 7, cell 0.02) gives at a few points in turn;
two uninformative problems with their evaluator's values and summary.

    python tools/refs/gen_registry_sample.py
"""
import json
from pathlib import Path

from mootation import __version__
from mootation.benchmarks import registry as R
from mootation.run.campaign import _code_stamp
from mootation.run.registry_export import FORMAT, entry, names

NAMES = ("shiftDTLZ2_3D", "C2-DTLZ2_3D", "rDTLZ2_3D", "bbobbiobj07_n10_2D", "catZDT1")
FAILING = ("ZDT1_fail30b", "DTLZ2_3D_fail30r")
NOISY = ("ZDT1_n1", "DTLZ2_3D_d5", "ZDT1_s5", "DTLZ2_3D_h1")
UNINFORMATIVE = ("uninformative_n10_2D", "uninformative_f2b_n02_2D", "uninformative_f3b_n02_2D",
                 "uninformative_f4b_n02_2D")


def box_point(p, u):
    """The point of p's box at the normalised point u."""
    return [lo + v * (hi - lo) for v, (lo, hi) in zip(u, p.bounds)]

problems = []
for name in NAMES:
    e = entry(name)
    p = R.PROBLEMS[name]
    x = [(lo + hi) / 2 + 0.1 * (i % 3) * (hi - lo) / 3 for i, (lo, hi) in enumerate(p.bounds)]
    if e["spec"] is not None:
        F, G = p.native.evaluate_with_constraints([x])
        e["check"] = {"x": x, "F": [float(v) for v in F[0]], "G": [float(v) for v in G[0]]}
    problems.append(e)
for name in FAILING:
    e = entry(name)
    p = R.PROBLEMS[name]
    n = p.n_vars
    region = e["failing_region"]
    if region["kind"] == "b":
        points = [[0.5] * n, [region["t"] / 2] + [0.5] * (n - 1), [0.5] * (n - 1) + [1.0 - region["t"] / 2]]
    else:
        c = region["centres"]
        points = [list(c[0]), [min(1.0, v + 0.5 * region["radius"] / n ** 0.5) for v in c[1]],
                  [0.5] * n, [0.0] * n]
    checks = []
    for u in points:
        x = box_point(p, u)
        F = [float(v) for v in p.evaluate(x)]
        checks.append({"x": x, "F": [None if v != v else v for v in F]})
    e["fail_checks"] = checks
    problems.append(e)
for name in NOISY:
    e = entry(name)
    p = R.PROBLEMS[name]
    ev = p.make_evaluator(7, cell=0.02)
    n = p.n_vars
    points = [box_point(p, [0.1 + 0.8 * ((3 * i + k) % 7) / 6 for i in range(n)]) for k in range(4)]
    points.append(list(points[0]))                    # the same x again: n and h draw anew, d and s not
    rows = ev.batch(points)
    e["noise_checks"] = {"seed": 7, "cell": 0.02,
                         "points": [{"x": x, "F": [float(v) for v in f]} for x, f in zip(points, rows)]}
    problems.append(e)
for name in UNINFORMATIVE:
    e = entry(name)
    p = R.PROBLEMS[name]
    ev = p.make_evaluator(7)
    n = p.n_vars
    points = [[((3 * i + k) % 5) / 4 for i in range(n)] for k in range(6)]   # 0, 1/2 and 1 among them
    rows = [ev(x) for x in points]
    e["uninformative_checks"] = {"seed": 7, "points": [{"x": x, "F": [float(v) for v in f]}
                                                       for x, f in zip(points, rows)],
                                 "summary": ev.summary()}
    problems.append(e)
doc = {"format": FORMAT, "code_stamp": _code_stamp(), "reference_version": R.REFERENCE_VERSION,
       "mootation": __version__, "names": names(), "problems": problems}
out = Path(__file__).resolve().parents[2] / "tests" / "data" / "registry_sample.json"
out.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8", newline="\n")
print(out)
