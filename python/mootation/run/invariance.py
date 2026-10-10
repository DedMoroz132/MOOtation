# SPDX-License-Identifier: Apache-2.0
"""Is an algorithm invariant to the scale and to a shift of the objectives?

`python -m mootation.run.campaign c.toml --invariance-check` (task 7 of
2026-10-10, §5.4, with the correction of its supplement 2, §5.1): every
algorithm entry of the campaign on every problem of it, at its budget.

SCALE, BIT FOR BIT. Multiplying an objective by a power of two is exact in
floating point, and so is every comparison, sum of such values and ratio of
two of them: an algorithm whose arithmetic on the objectives is invariant to
the scale runs the same trajectory on f and on 2^k f. Two copies: every
objective times 2^10 ("scale"), and the objectives times 2^-6, 2^0, 2^6, ...
in turn ("scale_each"). One run of each and of the problem itself, seed 1:
the points evaluated, in order, compared bit for bit, with the first
evaluation where they part.

SHIFT, BY STATISTICS. f + 2^e is rounded (the low bits of f are lost to the
larger exponent), so a trajectory can part from the original's without any
hidden constant in the algorithm. The copy f + 2^4 is therefore run on
`seeds` seeds (30) beside the problem itself, and the final populations'
IGD+ (the shift taken back off, normalised by the problem's frame, against
its reference front) compared by the rank-sum test, Holm-adjusted over the
algorithms of a problem. Three outcomes: "invariant" (every seed's
trajectory the same bit for bit), "rounding" (they part, but the
distributions cannot be told apart, p_holm >= 0.05), "not invariant".

The question is the algorithm's, not the run layer's: the runs go through
mootation.minimize with the entry's parameters, with no archive, no records
and no campaign files. The results are the reader's; the repository keeps
the tool and a test of two cases known in advance (NSGA-II is invariant to a
common power-of-two scale, its crowding distance normalised; SMS-EMOA at
three objectives is not, its reference point nad + 1 is absolute).
"""

from __future__ import annotations

import csv
from pathlib import Path

SCALE_COMMON = 2.0 ** 10
SHIFT = 2.0 ** 4
SCALE_EACH = (2.0 ** -6, 1.0, 2.0 ** 6)


def scale_each(m: int) -> list:
    """The per-objective factors of the "scale_each" copy: 2^-6, 1, 2^6, ..."""
    return [SCALE_EACH[i % len(SCALE_EACH)] for i in range(m)]


def run(name: str, problem, *, pop: int, budget: int, seed: int, params: dict,
        transform=None):
    """One run: (the evaluated points in order, as an array, and the final
    objectives taken back through the inverse of `transform`)."""
    import numpy as np
    from .. import minimize
    xs = []
    a = b = None
    if transform is not None:
        a, b = (np.asarray(v, float) for v in transform)

    def f(x):
        xs.append(np.array(x, float))
        v = np.asarray(problem.evaluate(list(x)), float)
        return list(v if a is None else a * v + b)
    res = minimize(f, problem.bounds, problem.n_obj, algorithm=name, pop_size=pop,
                   max_evaluations=budget, seed=seed, **params)
    F = np.asarray(res.objectives, float)
    if a is not None and len(F):
        F = (F - b) / a
    return np.array(xs), F


def first_difference(A, B) -> int | None:
    """The first evaluation (from 1) where two runs part, None if they never do."""
    import numpy as np
    n = min(len(A), len(B))
    same = np.all(A[:n] == B[:n], axis=1) if n else np.zeros(0, bool)
    if not same.all():
        return int(np.argmin(same)) + 1
    return None if len(A) == len(B) else n + 1


def check(name: str, problem_name: str, *, pop: int, budget: int, params: dict | None = None,
          seeds: int = 30, shift: bool = True) -> dict:
    """The three comparisons for one algorithm on one problem."""
    import numpy as np
    from ..benchmarks import get
    from .metrics import _normalised, igd_plus
    p = get(problem_name)
    params = dict(params or {})
    m = p.n_obj
    X0, F0 = run(name, p, pop=pop, budget=budget, seed=1, params=params)
    out = {"algorithm": name, "problem": problem_name, "evaluations": len(X0)}
    for key, a in (("scale", [SCALE_COMMON] * m), ("scale_each", scale_each(m))):
        X1, _ = run(name, p, pop=pop, budget=budget, seed=1, params=params,
                    transform=(a, [0.0] * m))
        d = first_difference(X0, X1)
        out[key] = "invariant" if d is None else "not invariant"
        out[key + "_first_diff"] = d
    if not shift:
        return out
    ref = np.asarray(p.pareto_front(1000), float) if callable(p.pareto_front) else None
    plain, moved, parted = [], [], []
    for s in range(1, seeds + 1):
        Xa, Fa = (X0, F0) if s == 1 else run(name, p, pop=pop, budget=budget, seed=s,
                                             params=params)
        Xb, Fb = run(name, p, pop=pop, budget=budget, seed=s, params=params,
                     transform=([1.0] * m, [SHIFT] * m))
        parted.append(first_difference(Xa, Xb))
        if ref is not None:
            for F, into in ((Fa, plain), (Fb, moved)):
                G, R = _normalised(F, ref, p.ideal, p.nadir)
                into.append(igd_plus(G, R) if len(G) else float("inf"))
    out["shift_parted"] = sum(d is not None for d in parted)
    out["shift_first_diff_median"] = (float(np.median([d for d in parted if d is not None]))
                                      if out["shift_parted"] else None)
    out["igdp_plain"] = plain
    out["igdp_shifted"] = moved
    return out


def classify_shift(rows: list) -> None:
    """Holm over the rows of one problem: shift = invariant | rounding | not invariant."""
    from .stats import holm, rank_sum
    ps = []
    for r in rows:
        if r.get("shift_parted") is None:
            continue
        r["shift_p"] = (rank_sum(r["igdp_plain"], r["igdp_shifted"])["p"]
                        if r["igdp_plain"] else 1.0)
        ps.append(r)
    for r, pa in zip(ps, holm([r["shift_p"] for r in ps])):
        r["shift_p_holm"] = pa
        r["shift"] = ("invariant" if r["shift_parted"] == 0 else
                      "rounding" if pa >= 0.05 else "not invariant")


def run_campaign(cfg, spec, root: Path, *, seeds: int = 30, workers: int = 1,
                 problems=None) -> str:
    """--invariance-check over a campaign's algorithm entries and problems
    (`problems`: only these)."""
    import numpy as np
    from .campaign import expand_jobs
    entries = {}                             # one job per (problem, entry): its seed is ours
    for job in expand_jobs(cfg, spec):
        if not job.ladder_of and (problems is None or job.problem in problems):
            entries.setdefault((job.problem, job.key), job)
    tasks = [(j.algorithm, j.problem, j.pop, j.budget, dict(j.params), seeds, j.key)
             for j in entries.values()]
    if workers > 1 and len(tasks) > 1:
        import multiprocessing as mp
        with mp.Pool(processes=workers) as pool:
            rows = pool.map(_one, tasks)
    else:
        rows = [_one(t) for t in tasks]
    for prob in sorted({r["problem"] for r in rows}):
        classify_shift([r for r in rows if r["problem"] == prob])
    out = root / "_invariance"
    out.mkdir(parents=True, exist_ok=True)
    cols = ["problem", "label", "algorithm", "evaluations", "scale", "scale_first_diff",
            "scale_each", "scale_each_first_diff", "shift", "shift_parted",
            "shift_first_diff_median", "shift_p", "shift_p_holm", "igdp_plain_median",
            "igdp_shifted_median"]
    with (out / "invariance.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for r in rows:
            r["igdp_plain_median"] = float(np.median(r["igdp_plain"])) if r["igdp_plain"] else None
            r["igdp_shifted_median"] = (float(np.median(r["igdp_shifted"]))
                                        if r["igdp_shifted"] else None)
            w.writerow([r.get(c) for c in cols])
    lines = [f"invariance of {len(rows)} runs' trajectories: scale x 2^10 and x (2^-6, 1, "
             f"2^6, ...) bit for bit; shift + 2^4 over {seeds} seeds, rank-sum, Holm",
             f"{'problem':16}{'label':22}{'scale':>16}{'scale_each':>16}{'shift':>16}"]
    for r in rows:
        lines.append(f"{r['problem']:16}{r['label']:22}{r['scale']:>16}{r['scale_each']:>16}"
                     f"{r.get('shift', '-'):>16}")
    lines.append(f"-> {out / 'invariance.csv'}")
    return "\n".join(lines)


def _one(task) -> dict:
    name, problem, pop, budget, params, seeds, label = task
    r = check(name, problem, pop=pop, budget=budget, params=params, seeds=seeds)
    r["label"] = label
    return r
