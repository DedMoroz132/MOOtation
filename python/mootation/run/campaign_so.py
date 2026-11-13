# SPDX-License-Identifier: Apache-2.0
"""One job of a campaign at one objective (task 18, stage 1, §1.4): what
campaign._run_job does for a problem of the registry with n_obj = 1 (bbob,
MGH, NIST), its own path beside the multi-objective one, which it leaves as
it is. include/mootation/campaign/job_run.hpp runs the same job in C++, and
the two write the same files (tools/compare_results.py).

THE RUN. The algorithm (minimize) sees the problem of the C++ suite through
counting functions: every value it asks for is a row of the run's
_core.RunLedger, which keeps the best point and a record each time it
improves (scalar/improvements.hpp), with the grad_calls made so far (a call
of the problem's Jacobian is one). A least-squares problem (MGH, NIST) hands
its residuals, and MGH its Jacobian, to the algorithms that take them (the
single-objective ones), and its start x0 to lm, lbfgsb and eki; a problem
with symmetries (task 19) its gradient where it is not a sum of squares
(lj_cluster), its gauge basis (for lm_gauge = "on") and, to the
single-objective algorithms, its blocks and canon (for sym). A bbob problem
runs at the instance [campaign] bbob_instance says: "by_run" (the default)
its run's seed (seeds 1, 2, ... by default: instance k on run k), "fixed"
instance 1; its f_opt is that instance's (the owner's decision of 01.11,
§1.1; the numbers are the library's, not COCO's). With [campaign]
wrap_shift, wrap_rotate or wrap_perm (task 20, §20.2.1) the run sees the
problem through the wrappers of suite/wrap.hpp (_core.Wrapped, keyed by the
problem's name; "by_run": the run's seed as the instance): the box, the
start and the answer in the wrapped coordinates, f_opt the problem's,
meta.json's so.wrap the instances; a wrapper the problem refuses (a rotation
of a problem with a box, WRAP-3; blocks, WRAP-7) makes the job incompatible.

WHAT IT WRITES (docs/running.md, "A campaign at one objective"):
  improvements.csv  every record: evaluations, grad_calls, f, cv
  trajectory.jsonl  at each count b of the sample grid (record_grid; the end
                    of the run too): the last record at or below b — fe = b,
                    its grad_calls, f, delta_f = f - f_opt, cv and the best
                    point's x1 ... xn (one key each: the store keeps the
                    trajectory column by column)
  final.csv         the answer: the best point, one row (f1, x1 ... xn)
  meta.json         the job, and "so": f_opt (and how it is known), the kind
                    of targets and their precisions, f(x0), the runtime to
                    each target (so_metrics.hpp), the run's cost; "final":
                    the best point's f, delta_f, cv and the evaluations and
                    grad_calls at it (fe_best, grad_calls_best)
The multi-objective keys of [campaign] (metrics, the archive, the second
answer, the hard budgets, the snapshots) do not apply: meta.json names them
(ignored_at_m1).

REFUSALS (incompatible, as campaign.py records them): an algorithm that is
not a single-objective one (mootation.scalar.SINGLE, stage 4's population
methods included); lm, eki or data_continuation on a problem without
residuals; an algorithm other than lm, lbfgsb and eki on a problem without a
box (bounds -inf / +inf).
"""

from __future__ import annotations

import json
import math
import os
import platform
import socket
import sys
import time
from pathlib import Path

# the [campaign] keys a job at one objective does not read, where they are not
# their defaults
MO_KEYS = ("metrics", "final_metrics", "archive", "set_answer", "set_metrics", "hard_budgets",
           "snapshots", "archive_checkpoints", "all_evaluated", "dense_reference", "operator_stats")
LOCAL = ("lm", "lbfgsb", "eki")


def instance_of(problem, seed: int, how: str) -> int | None:
    """The bbob instance of a run: its seed ("by_run") or 1 ("fixed"); None off bbob."""
    if problem.native is None or problem.native.spec.get("family") != "bbob":
        return None
    return int(seed) if how == "by_run" else 1


def refusal(algorithm: str, problem) -> str | None:
    """Why the pair cannot run at all (the job is recorded incompatible), or None."""
    from ..scalar import SINGLE
    if algorithm not in SINGLE:
        return f"{algorithm}: not a single-objective algorithm (mootation.scalar.SINGLE; incompatible by construction)"
    if algorithm in ("lm", "eki", "data_continuation") and problem.native.n_res == 0:
        return f"{algorithm}: the problem gives no residuals (incompatible by construction)"
    if algorithm not in LOCAL and any(math.isinf(v) for b in problem.bounds for v in b):
        return (f"{algorithm}: a problem without a box (bounds -inf / +inf) is for lm, lbfgsb and eki "
                "(incompatible by construction)")
    return None


def grid_of(spec, budget: int, pop: int) -> list:
    """The sample grid's counts up to the budget (record_grid)."""
    from .campaign import budget_grid, log_grid
    if spec.record_grid == "log":
        return sorted(set(log_grid(budget, spec.record_per_decade)) | {v for v in spec.record_at if v <= budget})
    if spec.record_grid == "budgets":
        return budget_grid(budget, spec.record_at)
    stride = max(1, spec.record_every) * max(1, pop)
    return list(range(stride, budget + 1, stride))


def _peaks(p, res, so) -> dict:
    """A CEC 2013 problem's peaks found (so_metrics.peaks_found) at the
    competition's accuracies on the run's final set: a strategy's minima
    log, else the answer (job_so.hpp, peaks)."""
    cec = p.native.spec.get("cec_problem") if p.native is not None else None
    if not cec:
        return {}
    from .. import _core
    settings = dict(res.settings)
    if "minima" in settings:
        rows = json.loads(settings["minima"])
        X, F, kind = [r[3:] for r in rows], [-r[0] for r in rows], "minima"
    else:
        X, F, kind = [list(v) for v in res.variables], [-o[0] for o in res.objectives], "answer"
    info = _core.cec2013_info(int(cec))
    eps = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
    return {"peaks": {"set": kind, "eps": eps,
                      "found": [so.peaks_found(X, F, info["height"], e, info["r"]) for e in eps],
                      "n_global": int(info["n_global"])}}


def run_job_so(job, spec, d: Path, meta_path: Path, traj_path: Path, put_table, *, quiet: bool) -> str:
    import numpy as np
    from .. import __version__, _core, minimize
    from ..benchmarks.properties import modification_of
    from ..benchmarks.registry import PROBLEMS, problem_version
    from .campaign import TRAJECTORY_VERSION, _code_stamp, _fp_env, _write_json
    from .provenance import revision
    p = PROBLEMS[job.problem]
    so = _core.so_metrics
    t0 = time.perf_counter()
    iid = instance_of(p, job.seed, spec.bbob_instance)
    native, f_opt = p.native, p.f_opt
    if iid is not None:
        spec_n = dict(p.native.spec, bbob_iid=iid)
        native = _core.Native({k: spec_n[k] for k in ("family", "n_var", "n_obj", "bbob_fid", "bbob_iid")})
        f_opt = float(native.spec["bbob"][0]["f_opt"])               # the suite's instance (bbob.hpp)
    from .campaign import CampaignSpec
    default = CampaignSpec()
    def plain(v):
        return list(v) if isinstance(v, (list, tuple)) else v
    ignored = [k for k in MO_KEYS if plain(getattr(spec, k)) != plain(getattr(default, k))]
    meta = {
        "status": "running", "problem": job.problem, "algorithm": job.key, "core": job.algorithm,
        "seed": job.seed, "pop": job.pop, "gens": job.gens, "budget_fe": job.budget,
        "budget_nominal": job.evaluations or spec.budget_fe or job.budget,
        "params": job.params, "n_objs": 1, "n_vars": p.n_vars, "pop_note": job.pop_note,
        "record_grid": spec.record_grid,
        "record_per_decade": spec.record_per_decade if spec.record_grid == "log" else None,
        "record_at": list(spec.record_at), "record_every": spec.record_every,
        "problem_version": problem_version(job.problem), "modification": modification_of(job.problem),
        "code_stamp": _code_stamp(), "trajectory_version": TRAJECTORY_VERSION,
        "ignored_at_m1": ignored,
        "mootation": __version__, "revision": revision(), "host": socket.gethostname(),
        "platform": platform.platform(), "python": sys.version.split()[0], "pid": os.getpid(),
        "fp_env": _fp_env(), "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    _write_json(meta_path, meta)
    traj_path.write_text("", encoding="utf-8")     # there from the start, as the multi-objective run's

    def failed(why: str, incompatible: bool) -> str:
        meta.update(status="failed", error=why, fe=fe, finished=time.strftime("%Y-%m-%dT%H:%M:%S"),
                    seconds=round(time.perf_counter() - t0, 3))
        if incompatible:
            meta["incompatible"] = True
        _write_json(meta_path, meta)
        if not quiet:
            print(f"[{job.index}] {job.problem} {job.key} seed {job.seed}: FAILED {why}", file=sys.stderr)
        return "failed"

    fe, grad_calls = 0, 0
    why = refusal(job.algorithm, p)
    if why is not None:
        return failed(why, True)
    bounds, x0, wrap = p.bounds, (list(p.x0) if p.x0 is not None else None), None
    w = {k: (job.seed if v == "by_run" else int(v))
         for k, v in (("shift", spec.wrap_shift), ("rotate", spec.wrap_rotate), ("perm", spec.wrap_perm))}
    if any(w.values()):
        bbob = native.spec.get("family") == "bbob"
        try:
            native = _core.Wrapped(native, job.problem, w["shift"], w["rotate"], w["perm"],
                                   spec.wrap_shift_scale, bbob, [tuple(b) for b in p.bounds],
                                   list(native.spec["bbob"][0]["x_opt"]) if bbob else [],
                                   x0 or [])
        except ValueError as e:
            return failed(f"{e} (incompatible by construction)", True)
        bounds = [tuple(b) for b in native.wrap_bounds]
        x0 = native.to_run(x0) if x0 is not None else None
        wrap = {**w, "shift_scale": spec.wrap_shift_scale, "instance": native.instance}

    # the counting functions (THE RUN)
    ledger = _core.RunLedger(1, p.n_vars, False, None, None, None, None)
    ledger.set_reads([], [])
    # the counts of the box at each count of the grid (task 20, §20.2.2): the
    # thread's counter, which only grows, since the run began (job_so.hpp)
    grid = grid_of(spec, job.budget, job.pop)
    counts0, snaps = _core.repair_counts(), []      # exact on any threads (SCALAR-15)

    # the count of an evaluation or of a gradient call: at once, or at the
    # round's end where a strategy's start runs in a thread (strategy.hpp,
    # THE DEFERRED COUNT; job_so.hpp the same)
    defer = getattr(_core.scalar, "defer", None)

    def account(X, F):
        if defer is None or not defer(lambda: account_now(X, F)):
            account_now(X, F)

    def count_grad():
        def one():
            nonlocal grad_calls
            grad_calls += 1
        if defer is None or not defer(one):
            one()

    def account_now(X, F):
        nonlocal fe
        ledger.set_grad_calls(grad_calls)
        ledger.begin(np.asarray(X, float).reshape(-1, p.n_vars), np.asarray(F, float).reshape(-1, 1))
        while ledger.next() >= 0:
            pass
        fe = ledger.fe
        while len(snaps) < len(grid) and grid[len(snaps)] <= fe:
            snaps.append(tuple(a - b for a, b in zip(_core.repair_counts(), counts0)))

    def fn(x):
        f = native.evaluate(x)
        account([x], [f])
        return f

    def residuals(x):
        r = native.residuals(x)
        s = 0.0
        for v in r:                                # in order, as the suite and the evaluator add
            s += v * v
        account([x], [[0.5 * s]])
        return r

    def jacobian(x):
        count_grad()
        return native.residuals(x, True)[1]

    from ..scalar import SINGLE
    derivatives = {}
    def gradient(x):
        count_grad()
        return list(native.gradient(x))

    if native.n_res and job.algorithm in SINGLE:
        derivatives = {"residuals": residuals, "jac": jacobian if native.has_jacobian else None}
    elif native.has_gradient and job.algorithm in SINGLE:
        derivatives = {"grad": gradient}
    if native.gauge_dim > 0:
        derivatives["gauge_basis"] = lambda x: [list(row) for row in native.gauge_basis(x)]
    if "residuals" in derivatives and native.res_blocks:  # its residual blocks at weight 1 (task 19; data_continuation)
        derivatives["residual_blocks"] = list(native.res_blocks)
    if native.blocks and job.algorithm in SINGLE:      # its symmetries, for sym (task 19, §19.2)
        derivatives["sym_blocks"] = list(native.blocks)
        if native.has_canon:
            derivatives["canon"] = lambda x: list(native.canon(x))
    if native.linear and job.algorithm in SINGLE:      # for lm_varpro (task 25; job_so.hpp as well)
        derivatives["linear"] = list(native.linear)
    if job.algorithm in LOCAL and x0 is not None:
        derivatives["x0"] = list(x0)
    try:
        res = minimize(None if "residuals" in derivatives else fn, bounds=bounds, n_objs=1,
                 algorithm=job.algorithm, pop_size=job.pop, n_gen=job.gens, seed=job.seed,
                 max_evaluations=job.budget, **{k: v for k, v in derivatives.items() if v is not None},
                 f_stop=f_opt + spec.stop_at if spec.stop_at > 0 and f_opt is not None else None,
                 **job.params)
    except Exception as e:                          # one bad job must not kill the shard
        return failed(f"{type(e).__name__}: {e}", "incompatible by construction" in str(e))

    records = [tuple(r) for r in ledger.improvements]
    xs = [list(x) for x in ledger.record_x]
    grad_cost = float(job.params.get("grad_cost", 1.0))
    delta = (lambda f: None if f_opt is None else f - f_opt)          # noqa: E731
    put_table("improvements.csv", ["# mootation campaign improvement records v1: a record each time "
                                   "the best point improved (scalar/improvements.hpp)"],
              ["evaluations", "grad_calls", "f", "cv"], records, ints=(0, 1))
    with traj_path.open("w", encoding="utf-8") as fh:
        counts = [b for b in grid_of(spec, job.budget, job.pop) if b <= fe]
        if fe and (not counts or counts[-1] != fe):
            counts.append(fe)
        k, n_records = -1, 0
        totals = tuple(int(res.repairs.get(key, 0)) for key in ("checked", "out_children", "out_vars"))
        for row, b in enumerate(counts):
            while k + 1 < len(records) and records[k + 1][0] <= b:
                k += 1
            rec = {"fe": b}
            if k >= 0:
                e, g, f, cv = records[k]
                rec.update(grad_calls=g, f=f, delta_f=delta(f), cv=cv)
                rec.update({f"x{j + 1}": v for j, v in enumerate(xs[k])})
            else:
                rec.update(grad_calls=None, f=None, delta_f=None, cv=None)
                rec.update({f"x{j + 1}": None for j in range(p.n_vars)})
            # the counts of the box at b; the last row the run's totals (the points
            # made but not evaluated included); the same on any threads (SCALAR-15)
            c = totals if row + 1 == len(counts) else snaps[row] if row < len(snaps) else None
            rec.update(repair_checked=None if c is None else c[0], repair_out_children=None if c is None else c[1],
                       repair_out_vars=None if c is None else c[2])
            fh.write(json.dumps(rec, separators=(",", ":")) + "\n")
            n_records += 1
    best = records[-1] if records else None
    if best is not None:
        put_table("final.csv", ["# mootation campaign answer v1 (one objective): the best point"],
                  ["f1", *(f"x{j + 1}" for j in range(p.n_vars))], [[best[2], *xs[-1]]])
    kind = p.targets
    f_x0 = records[0][2] if records else None
    precisions = so.default_precisions(kind) if kind else []
    runtimes = []
    if f_opt is not None and f_x0 is not None:
        runtimes = [so.runtime(records, t, grad_cost) for t in so.targets(kind, precisions, f_opt, f_x0)]
    meta.update(
        status="done", fe=fe, grad_calls=grad_calls, records=n_records,
        so={"f_opt": f_opt, "f_opt_kind": p.f_opt_kind,
            "reference_version": so.REFERENCE_VERSION if p.f_opt_kind == "best_known" else None,
            "instance": iid, "targets": kind, "precisions": precisions, "f_x0": f_x0,
            "runtimes": runtimes, "grad_cost": grad_cost, "cost": fe + grad_cost * grad_calls,
            "improvements": len(records),
            "repairs": {k: int(res.repairs.get(k, 0)) for k in ("checked", "out_children", "out_vars")},
            **_peaks(p, res, so),
            **({"stop_at": spec.stop_at} if spec.stop_at > 0 else {}),
            **({"restarts": json.loads(dict(res.settings)["restarts"])}
               if spec.keep_restarts and "restarts" in dict(res.settings) else {}),
            **({"wrap": wrap} if wrap is not None else {})},
        final=({} if best is None else {"f": best[2], "delta_f": delta(best[2]), "cv": best[3],
                                         "fe_best": best[0], "grad_calls_best": best[1]}),
        finished=time.strftime("%Y-%m-%dT%H:%M:%S"), seconds=round(time.perf_counter() - t0, 3))
    _write_json(meta_path, meta)
    return "done"
