# SPDX-License-Identifier: Apache-2.0
"""A campaign: every selected algorithm on every selected benchmark, several
seeds each, with a convergence trajectory recorded per run.

    python -m mootation.run.campaign camp.toml --list
    python -m mootation.run.campaign camp.toml                   # run all jobs here
    python -m mootation.run.campaign camp.toml --workers 8       # local pool; <results>/_workers.txt resizes it live
    python -m mootation.run.campaign camp.toml --shard 3/40      # one slice, for an array job
    python -m mootation.run.campaign camp.toml --job 17          # exactly one job
    python -m mootation.run.campaign camp.toml --emit-slurm 40   # write submit.sh + jobs.txt
    python -m mootation.run.campaign camp.toml --compare igd     # median per problem x algorithm
    python -m mootation.run.campaign camp.toml --ranks igd       # mean rank per algorithm

The config is the same TOML the rest of `mootation.run` reads
with `[problem] kind = "builtin"`, a `[benchmarks]` selection, `[[algorithms]]`
entries and a `[campaign]` table:

    [campaign]
    out          = "results/{name}"   # results root, relative to the config
    budget_fe    = 0                  # 0 -> the problem's own pop*gens; else this many evaluations
    record_every = 1                  # generations between trajectory points
    metrics      = ["igd", "igdp", "hv"]
    n_ref        = 1000               # reference-front sample size for IGD/IGD+
    seeds        = []                 # explicit list; empty -> 1..benchmarks.runs

An algorithm entry with `pop = 0` / `gens = 0` takes the problem's own budget
(the values its paper used); a `pop` the algorithm cannot accept — an
NSGA-III lattice, an M2M multiple of K — is rounded to the nearest one it can,
and the adjustment is written into the run's meta.json.

Layout on disk, the contract the TUI reads:

    <out>/<problem>/<algorithm>/run_<seed>/trajectory.jsonl   one line per record
    <out>/<problem>/<algorithm>/run_<seed>/meta.json          status, budget, timing
    <out>/<problem>/<algorithm>/run_<seed>/final.csv          objectives then variables

and with the run archive on, archive.csv (every archive point, objectives
then variables) or, with archive_variables = "selected", archive.csv.gz (the
objectives of every point) and final_archive.csv.gz (the N points DSS selects
for the archive scenario, objectives then variables); with snapshots on,
snapshots.npz.

A job whose meta.json says "done" is skipped, so a killed campaign resumes by
being started again; `--force` reruns everything.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import socket
import sys
import time
import traceback
from dataclasses import dataclass, asdict
from pathlib import Path

from .config import Config, ConfigError, load, validate
from .algorithms import (EXACT_LATTICE, K_DIVISIBLE, check_pop,
                         nearest_lattice_sizes)
from .metric_names import (HIGHER_IS_BETTER, HV_NAMES, METRIC_NAMES, NEEDS_FRONT, NEEDS_SET,
                           NEEDS_X, RUN_STATS, TABLE_NAMES)

METRICS_DEFAULT = ("igd", "igdp", "hv")


@dataclass
class Job:
    index: int
    problem: str
    algorithm: str
    seed: int
    pop: int
    gens: int
    params: dict
    n_objs: int
    pop_note: str = ""
    # the variant's label (results are filed and reported under it) and a
    # budget in evaluations that overrides pop * gens (0 = pop * gens)
    label: str = ""
    evaluations: int = 0
    # a rung of the budget ladder: the label of the full-budget run it
    # shortens ([campaign] ladder); empty for every other job
    ladder_of: str = ""

    @property
    def key(self) -> str:
        return self.label or self.algorithm

    @property
    def budget(self) -> int:
        return self.evaluations or self.pop * self.gens

    @property
    def rel_dir(self) -> Path:
        return Path(self.problem) / self.key / f"run_{self.seed}"


@dataclass
class CampaignSpec:
    out: str = "results/{name}"
    budget_fe: int = 0
    record_every: int = 1
    metrics: tuple = METRICS_DEFAULT
    final_metrics: tuple = ()          # empty: the same as metrics
    n_ref: int = 1000
    seeds: tuple = ()
    # "generations": a trajectory point every record_every generations.
    # "log": at fixed evaluation counts 10^(j/record_per_decade), the SAME
    # counts at every budget, so runs of a budget ladder line up (log_grid).
    record_grid: str = "generations"
    record_per_decade: int = 10
    # Extra evaluation counts for the "log" grid — a budget ladder's rungs
    # (2 500, 5 000, 10 000), which 10^(j/10) misses: a record is taken at the
    # first generation that reaches each, like the grid's own counts.
    record_at: tuple = ()
    # Trajectory indicators (of `metrics`) recorded only at the record_at
    # counts and at the end of a run, not on the whole grid: the costly ones.
    # --recompute-trajectory fills the other records from the snapshots.
    record_at_metrics: tuple = ()
    # Budget ladder: for every budget-dependent algorithm (budget.py), a
    # separate run at each of these budgets besides the full one, filed under
    # "<label>@<budget>" and marked ladder_of in meta.json. For the others a
    # rung is read off the full run's trajectory (record_at puts a record
    # there). The ladder jobs follow all the others in the job list.
    ladder: tuple = ()
    # The hypervolumes on the trajectory exactly only up to this many objectives
    # (0 = always). Above it they are recorded as null, or, with
    # trajectory_hv_mc_samples > 0, estimated by Monte Carlo from that many
    # points (the same points at every record, so the curve is smooth).
    trajectory_hv_max_m: int = 0
    trajectory_hv_mc_samples: int = 0
    # The final hypervolume: exact up to hv_exact_max_m objectives, Monte Carlo
    # from hv_mc_samples points above (metrics.hypervolume). The exact one is the
    # compiled WFG when the extension is built: milliseconds at 5 objectives.
    hv_exact_max_m: int = 5
    hv_mc_samples: int = 100_000
    # Scenario "archive" (report.py): the final metrics once more, on the run
    # archive reduced to the problem's population size by DSS
    # (meta["final_archive"]). Needs archive = true.
    archive_scenario: bool = True
    # What the variation operators did (metric_names.RUN_STATS): per trajectory
    # record and over the run in `final`. Off by default; it never changes a
    # run's result. meta["operators"] lists the operators and their bound
    # repair either way.
    operator_stats: bool = False
    # The run archive (archive.GridArchive -> archive.csv); archive_delta 0 is
    # the default grid step, 1e-3 below five objectives and 1e-2 from five.
    archive: bool = True
    archive_delta: float = 0.0
    # Whose variables the archive keeps: "all" — every point's, in archive.csv;
    # "selected" — only those of the N points DSS selects for the archive
    # scenario, in final_archive.csv.gz, with the objectives of every point in
    # archive.csv.gz. Over the stage-3 problems "selected" takes a tenth of the
    # room of "all" (a quarter of it gzipped).
    archive_variables: str = "all"
    # Population objectives at every trajectory record (snapshots.npz, float32),
    # to look at the front's shape later or compute a metric the run did not
    # record (--recompute-trajectory): False, True, or a list of the problems to
    # keep them for.
    snapshots: object = False
    # Whose snapshots keep the variables too (float32), so that igdx, cr and
    # pdist can be recomputed along the trajectory: "none", "pareto_set" (the
    # problems with a Pareto-set sample, which igdx and cr need) or "all". The
    # archive checkpoints keep their variables by the same rule.
    snapshot_variables: str = "none"
    # The archive scenario also at every record_at count below the budget: the
    # archive reduced by DSS there, its trajectory metrics in
    # meta["archive_at"] and its points in archive_at.npz, so that --cover and
    # the tables read the archive scenario at those budgets. Needs the archive.
    archive_checkpoints: bool = False


def _snapshots_wanted(spec: "CampaignSpec", problem: str) -> bool:
    s = spec.snapshots
    return s if isinstance(s, bool) else problem in s


def log_grid(limit: int, per_decade: int) -> list:
    """Evaluation counts round(10^(j/per_decade)) up to `limit`, without repeats.

    Absolute, not relative to the budget: a 10 000- and a 25 000-evaluation run
    share every threshold up to 10 000, so their trajectories can be compared
    point by point instead of through interpolation.
    """
    out = []
    j = 0
    while True:
        v = int(round(10.0 ** (j / per_decade)))
        if v > limit:
            return out
        if not out or v > out[-1]:
            out.append(v)
        j += 1


# ── reading the [campaign] table ────────────────────────────────────────────


def campaign_spec(cfg: Config) -> CampaignSpec:
    raw = dict(cfg.campaign or {})
    spec = CampaignSpec()
    known = {"out", "budget_fe", "record_every", "metrics", "final_metrics", "n_ref", "seeds",
             "record_grid", "record_per_decade", "record_at", "record_at_metrics", "ladder",
             "trajectory_hv_max_m",
             "trajectory_hv_mc_samples", "hv_exact_max_m", "hv_mc_samples",
             "archive", "archive_delta", "archive_variables", "archive_scenario", "snapshots",
             "snapshot_variables", "archive_checkpoints", "operator_stats"}
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ConfigError("campaign", f"unknown key(s): {', '.join(unknown)}. "
                                      f"Known: {', '.join(sorted(known))}")
    spec.out = str(raw.get("out", spec.out))
    spec.budget_fe = int(raw.get("budget_fe", 0))
    spec.record_every = int(raw.get("record_every", 1))
    # `metrics` are recorded along the trajectory, `final_metrics` once on the
    # final population. Splitting them is what lets a costly indicator — the
    # hypervolume at five objectives — be measured on the result without
    # being paid for at every trajectory point.
    for key, default in (("metrics", METRICS_DEFAULT), ("final_metrics", None)):
        names = raw.get(key, default)
        if names is None:
            continue
        for m in names:
            if m not in METRIC_NAMES:
                raise ConfigError(f"campaign.{key}",
                                  f"unknown metric '{m}'; known: {', '.join(METRIC_NAMES)}")
        setattr(spec, key, tuple(names))
    spec.n_ref = int(raw.get("n_ref", 1000))
    spec.seeds = tuple(int(s) for s in raw.get("seeds", ()))
    spec.record_grid = str(raw.get("record_grid", "generations"))
    spec.record_per_decade = int(raw.get("record_per_decade", 10))
    spec.trajectory_hv_max_m = int(raw.get("trajectory_hv_max_m", 0))
    if spec.record_grid not in ("generations", "log"):
        raise ConfigError("campaign.record_grid",
                          f"'generations' or 'log', not '{spec.record_grid}'")
    if spec.record_per_decade < 1:
        raise ConfigError("campaign.record_per_decade", "must be >= 1")
    at = raw.get("record_at", [])
    if (not isinstance(at, (list, tuple))
            or not all(isinstance(v, int) and not isinstance(v, bool) and v > 0 for v in at)):
        raise ConfigError("campaign.record_at", f"a list of positive evaluation counts, not {at!r}")
    if at and spec.record_grid != "log":
        raise ConfigError("campaign.record_at", "needs record_grid = \"log\"")
    spec.record_at = tuple(sorted(set(at)))
    sparse = raw.get("record_at_metrics", [])
    if not isinstance(sparse, (list, tuple)) or any(m not in spec.metrics for m in sparse):
        raise ConfigError("campaign.record_at_metrics",
                          f"a list of names from `metrics`, not {sparse!r}")
    if sparse and not spec.record_at:
        raise ConfigError("campaign.record_at_metrics", "needs record_at")
    spec.record_at_metrics = tuple(sparse)
    ladder = raw.get("ladder", [])
    if (not isinstance(ladder, (list, tuple))
            or not all(isinstance(v, int) and not isinstance(v, bool) and v > 0 for v in ladder)):
        raise ConfigError("campaign.ladder", f"a list of positive evaluation counts, not {ladder!r}")
    spec.ladder = tuple(sorted(set(ladder)))
    if spec.trajectory_hv_max_m < 0:
        raise ConfigError("campaign.trajectory_hv_max_m", "must be >= 0 (0 = no limit)")
    for key, lo in (("trajectory_hv_mc_samples", 0), ("hv_exact_max_m", 0),
                    ("hv_mc_samples", 1)):
        v = raw.get(key, getattr(spec, key))
        if not isinstance(v, int) or isinstance(v, bool) or v < lo:
            raise ConfigError(f"campaign.{key}", f"an integer >= {lo}, not {v!r}")
        setattr(spec, key, v)
    spec.operator_stats = raw.get("operator_stats", False)
    if not isinstance(spec.operator_stats, bool):
        raise ConfigError("campaign.operator_stats", "true or false")
    spec.archive_scenario = raw.get("archive_scenario", True)
    if not isinstance(spec.archive_scenario, bool):
        raise ConfigError("campaign.archive_scenario", "true or false")
    spec.archive = raw.get("archive", True)
    if not isinstance(spec.archive, bool):
        raise ConfigError("campaign.archive", "true or false")
    spec.archive_delta = float(raw.get("archive_delta", 0.0))
    if spec.archive_delta < 0:
        raise ConfigError("campaign.archive_delta", "must be >= 0 (0 = the default step)")
    spec.archive_variables = raw.get("archive_variables", "all")
    if spec.archive_variables not in ("all", "selected"):
        raise ConfigError("campaign.archive_variables",
                          f"\"all\" or \"selected\", not {spec.archive_variables!r}")
    snaps = raw.get("snapshots", False)
    if isinstance(snaps, bool):
        spec.snapshots = snaps
    elif isinstance(snaps, (list, tuple)):
        spec.snapshots = tuple(str(s) for s in snaps)
    else:
        raise ConfigError("campaign.snapshots", "true, false, or a list of problem names")
    spec.snapshot_variables = raw.get("snapshot_variables", "none")
    if spec.snapshot_variables not in ("none", "pareto_set", "all"):
        raise ConfigError("campaign.snapshot_variables",
                          f"\"none\", \"pareto_set\" or \"all\", not {spec.snapshot_variables!r}")
    spec.archive_checkpoints = raw.get("archive_checkpoints", False)
    if not isinstance(spec.archive_checkpoints, bool):
        raise ConfigError("campaign.archive_checkpoints", "true or false")
    if spec.record_every < 1:
        raise ConfigError("campaign.record_every", "must be >= 1")
    if spec.budget_fe < 0:
        raise ConfigError("campaign.budget_fe", "must be >= 0")
    return spec


def out_root(cfg: Config, spec: CampaignSpec) -> Path:
    base = cfg.source_path.parent if cfg.source_path else Path.cwd()
    p = Path(spec.out.format(name=cfg.name))
    return p if p.is_absolute() else base / p


# ── population-size adjustment ──────────────────────────────────────────────


def _two_layer_sizes(m: int, limit: int) -> set:
    from .algorithms import das_dennis_count
    sizes = set()
    h1 = 1
    while das_dennis_count(m, h1) <= limit:
        for h2 in range(1, h1):
            s = das_dennis_count(m, h1) + das_dennis_count(m, h2)
            if s <= limit:
                sizes.add(s)
        h1 += 1
    return sizes


def fit_pop(name: str, pop: int, n_objs: int, params: dict) -> tuple[int, str]:
    """The population size this algorithm will accept, and a note if it changed."""
    if name in EXACT_LATTICE:
        from .algorithms import lattice_sizes
        ok = set(lattice_sizes(n_objs, pop)) | _two_layer_sizes(n_objs, pop)
        if pop in ok:
            return pop, ""
        below = max((s for s in ok if s <= pop), default=None)
        if below is None:
            _, above = nearest_lattice_sizes(n_objs, pop)
            return above, f"pop {pop} -> {above} (smallest Das-Dennis lattice for M={n_objs})"
        return below, f"pop {pop} -> {below} (largest lattice size <= {pop} for M={n_objs})"
    if name in K_DIVISIBLE:
        k = int(params.get("K", params.get("n_clusters", K_DIVISIBLE[name])))
        if pop % k == 0:
            return pop, ""
        new = max(k, (pop // k) * k)
        return new, f"pop {pop} -> {new} (multiple of K={k})"
    return pop, ""


# ── job expansion ───────────────────────────────────────────────────────────


def expand_jobs(cfg: Config, spec: CampaignSpec) -> list[Job]:
    from ..benchmarks import PROBLEMS
    problems = list(cfg.benchmark_problems)
    if not problems:
        raise ConfigError("benchmarks", "the campaign selects no problems")
    if spec.seeds:
        seeds = list(spec.seeds)
    else:
        runs = int((cfg.benchmarks or {}).get("runs", 31))
        seeds = list(range(1, runs + 1))
    jobs: list[Job] = []
    idx = 0
    for pname in problems:
        p = PROBLEMS[pname]
        for a in cfg.algorithms:
            pop = a.pop if a.pop > 0 else p.pop_size
            pop, note = fit_pop(a.name, pop, p.n_obj, a.params)
            if a.evaluations > 0:                  # the algorithm's own budget
                gens = max(1, math.ceil(a.evaluations / pop))
            elif spec.budget_fe > 0:
                gens = max(1, math.ceil(spec.budget_fe / pop))
            else:
                gens = a.gens if a.gens > 0 else p.n_gen
            for s in seeds:
                jobs.append(Job(index=idx, problem=pname, algorithm=a.name, seed=s,
                                pop=pop, gens=gens, params=dict(a.params),
                                n_objs=p.n_obj, pop_note=note, label=a.key,
                                evaluations=a.evaluations))
                idx += 1
    if spec.ladder:
        # The rungs, after every full-budget job so that the job numbers of a
        # campaign without them stay as they were.
        from .budget import BUDGET_DEPENDENT
        full = {(j.problem, j.key): j for j in jobs}
        for pname in problems:
            for a in cfg.algorithms:
                if a.name not in BUDGET_DEPENDENT or a.evaluations > 0:
                    continue
                base = full[(pname, a.key)]
                for b in spec.ladder:
                    if b >= base.budget:
                        continue
                    gens = max(1, math.ceil(b / base.pop))
                    for s in seeds:
                        jobs.append(Job(index=idx, problem=pname, algorithm=a.name, seed=s,
                                        pop=base.pop, gens=gens, params=dict(a.params),
                                        n_objs=base.n_objs, pop_note=base.pop_note,
                                        label=f"{a.key}@{b}", evaluations=b,
                                        ladder_of=a.key))
                        idx += 1
    return jobs


def job_status(root: Path, job: Job) -> str:
    meta = root / job.rel_dir / "meta.json"
    if not meta.is_file():
        return "pending"
    try:
        with meta.open("r", encoding="utf-8") as fh:
            return str(json.load(fh).get("status", "pending"))
    except (OSError, ValueError):
        return "pending"


# ── running one job ─────────────────────────────────────────────────────────


def run_job(job: Job, root: Path, spec: CampaignSpec, *, force: bool = False,
            quiet: bool = False) -> str:
    """Run one job to completion; returns the final status."""
    import numpy as np
    from ..benchmarks import get as bench_get
    from .. import __version__, minimize
    from . import metrics as M
    from .provenance import revision

    d = root / job.rel_dir
    meta_path = d / "meta.json"
    if not force and job_status(root, job) == "done":
        return "done"
    d.mkdir(parents=True, exist_ok=True)
    traj_path = d / "trajectory.jsonl"
    if traj_path.exists():
        traj_path.unlink()

    p = bench_get(job.problem)
    ref = None
    final_metrics = spec.final_metrics or spec.metrics
    if any(m in NEEDS_FRONT for m in (*spec.metrics, *final_metrics)) and callable(p.pareto_front):
        ref = np.asarray(p.pareto_front(spec.n_ref), float)
    ideal, nadir = p.ideal, p.nadir
    pset = None
    if any(m in NEEDS_SET for m in (*spec.metrics, *final_metrics)) and callable(p.pareto_set):
        pset = np.asarray(p.pareto_set(spec.n_ref), float)
    final_hv = {"exact_max_m": spec.hv_exact_max_m, "mc_samples": spec.hv_mc_samples}

    from .budget import BUDGET_DEPENDENT
    meta = {
        "status": "running", "problem": job.problem, "algorithm": job.key,
        "core": job.algorithm,
        # measured in budget.py: a record of this run at a smaller budget is
        # not what a separate run of that budget gives (ladder_of: the full
        # run a ladder rung shortens)
        "budget_dependent": job.algorithm in BUDGET_DEPENDENT,
        "ladder_of": job.ladder_of or None,
        "seed": job.seed, "pop": job.pop, "gens": job.gens, "budget_fe": job.budget,
        # the budget asked for: budget_fe rounds it up to whole generations
        # (25 025 at a population of 91), --at and the checkpoints count from it
        "budget_nominal": job.evaluations or spec.budget_fe or job.budget,
        "params": job.params, "n_objs": p.n_obj, "n_vars": p.n_vars,
        "pop_note": job.pop_note, "metrics": list(spec.metrics),
        "final_metrics": list(final_metrics),
        "record_every": spec.record_every, "n_ref": spec.n_ref,
        "has_reference_front": ref is not None,
        "has_pareto_set": callable(p.pareto_set),
        "mootation": __version__,
        # the version never changes between commits; these do (provenance.py)
        "revision": revision(),
        "host": socket.gethostname(), "platform": platform.platform(),
        "python": sys.version.split()[0], "pid": os.getpid(),
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    _write_json(meta_path, meta)

    from .archive import GridArchive
    from .baselines import BASELINES, run_baseline
    baseline = job.algorithm in BASELINES
    # Every nondominated point the run evaluates, whatever the algorithm keeps
    # in its population (archive.GridArchive; archive.csv). A baseline's
    # answer IS its archive, so it gets one even when archiving is off.
    arc = (GridArchive(p.n_obj, p.n_vars, ideal=ideal, nadir=nadir,
                       delta=spec.archive_delta or None)
           if (spec.archive or baseline) else None)

    def feasible(x) -> bool:
        return all(c <= 0.0 for c in p.constraints(x))

    fe = 0
    # a problem whose values depend on the run (uninformative.py) gets its
    # evaluator from the run's seed; every other problem is its `evaluate`
    objective = (p.make_evaluator(job.seed) if getattr(p, "make_evaluator", None)
                 else p.evaluate)

    def evaluate(x):
        nonlocal fe
        fe += 1
        f = objective(x)
        if arc is not None and (not p.has_cons or feasible(x)):
            arc.add(f, x)
        return f

    snap = _snapshots_wanted(spec, job.problem)
    # the variables go into the snapshots and checkpoints where snapshot_variables says so
    keep_x = (spec.snapshot_variables == "all"
              or (spec.snapshot_variables == "pareto_set" and callable(p.pareto_set)))
    op_stats = spec.operator_stats and not baseline
    if op_stats:
        from .. import _core
    snap_fe, snap_gen, snap_n, snap_F, snap_X = [], [], [], [], []

    t0 = time.perf_counter()
    fh = traj_path.open("a", encoding="utf-8")
    n_records = 0
    budget = job.budget

    # Above trajectory_hv_max_m objectives the hypervolumes stay off the
    # trajectory (recorded as null, so the absence is explicit), or are
    # estimated by Monte Carlo when trajectory_hv_mc_samples asks for it.
    traj_metrics = list(spec.metrics)
    hv_skipped = []
    traj_hv = {"exact_max_m": spec.hv_exact_max_m, "mc_samples": spec.hv_mc_samples}
    if spec.trajectory_hv_max_m and p.n_obj > spec.trajectory_hv_max_m:
        if spec.trajectory_hv_mc_samples > 0:
            traj_hv = {"exact_max_m": 0, "mc_samples": spec.trajectory_hv_mc_samples}
        else:
            hv_skipped = [m for m in traj_metrics if m in HV_NAMES]
            traj_metrics = [m for m in traj_metrics if m not in HV_NAMES]
    grid = (sorted(set(log_grid(budget, spec.record_per_decade))
                   | {v for v in spec.record_at if v <= budget})
            if spec.record_grid == "log" else None)
    next_i = 0
    meta.update(record_grid=spec.record_grid,
                record_per_decade=spec.record_per_decade if grid is not None else None,
                record_at=list(spec.record_at),
                trajectory_hv_max_m=spec.trajectory_hv_max_m,
                trajectory_hv=("not recorded" if hv_skipped else
                               "exact" if traj_hv["exact_max_m"] >= p.n_obj else
                               f"monte-carlo, {traj_hv['mc_samples']} samples"),
                trajectory_metrics=traj_metrics, trajectory_hv_skipped=hv_skipped,
                trajectory_metrics_at_record_at=[m for m in traj_metrics
                                                 if m in spec.record_at_metrics])
    # The archive scenario at the record_at counts below the budget (its end is
    # final_archive): taken at the record that first reaches each count.
    checkpoints = ([b for b in spec.record_at if b < budget]
                   if spec.archive_checkpoints and arc is not None and grid is not None else [])
    archive_at = {}
    ck_budget, ck_fe, ck_n, ck_F, ck_X = [], [], [], [], []
    last_fe = 0                    # the evaluations of the previous record
    record_seconds = 0.0           # what the records cost: indicators, statistics, snapshots
    need_x = (snap and keep_x) or any(m in NEEDS_X for m in traj_metrics)
    if need_x and not baseline:
        from .. import _core                   # current_variables() at the records

    def measure(F, X, which):
        vals = M.compute(F, ref_front=ref, ideal=ideal, nadir=nadir, which=which,
                         pop=p.pop_size, hv_options=traj_hv, X=X, pareto_set=pset,
                         bounds=p.bounds, cyclic=p.cyclic_vars)
        for m in hv_skipped:
            vals[m] = None
        return vals

    # the costly indicators only at the record_at counts and at the end
    everywhere = [m for m in traj_metrics if m not in spec.record_at_metrics]

    def on_gen(gen, objectives, variables=None):
        nonlocal n_records, next_i, last_fe, record_seconds
        if grid is not None:
            # Generation 0 and the last call are always recorded; in between,
            # the first call at or past the next count of the grid.
            if gen > 0 and fe < budget and (next_i >= len(grid) or fe < grid[next_i]):
                return
            while next_i < len(grid) and grid[next_i] <= fe:
                next_i += 1
        t_rec = time.perf_counter()
        F = np.asarray(objectives, float)
        X = None
        if need_x and F.size:
            # a baseline hands its variables over; a core is asked for them
            if variables is None:
                variables = _core.current_variables()
            X = np.asarray(variables, float).reshape(len(F), p.n_vars)
        rec = {"gen": gen, "fe": fe, "t": round(t_rec - t0, 3), "n": int(len(F))}
        finite = bool(np.all(np.isfinite(F))) if F.size else True
        rec["finite"] = finite
        mark = fe >= budget or any(last_fe < b <= fe for b in spec.record_at)
        if finite and F.size:
            rec.update(measure(F, X, traj_metrics if mark else everywhere))
        if op_stats:
            rec.update(_core.operator_stats())
        if snap:
            snap_fe.append(fe); snap_gen.append(gen); snap_n.append(len(F))
            snap_F.append(F.astype(np.float32).reshape(len(F), p.n_obj))
            if keep_x:
                snap_X.append(np.zeros((0, p.n_vars), np.float32) if X is None
                              else X.astype(np.float32))
        crossed = [b for b in checkpoints if last_fe < b <= fe]
        if crossed and fe < budget and len(arc):         # the end is final_archive
            AF, AX, _ = arc.points()
            idx = arc.select(p.pop_size)
            vals = dict(measure(AF[idx], AX[idx], traj_metrics), fe=fe, n=int(len(idx)))
            for b in crossed:
                archive_at[str(b)] = vals
                ck_budget.append(b); ck_fe.append(fe); ck_n.append(len(idx))
                ck_F.append(AF[idx].astype(np.float32))
                if keep_x:
                    ck_X.append(AX[idx].astype(np.float32))
        last_fe = fe
        fh.write(json.dumps(rec, separators=(",", ":")) + "\n")
        fh.flush()
        n_records += 1
        record_seconds += time.perf_counter() - t_rec

    try:
        if baseline:
            res = run_baseline(job.algorithm, evaluate, p.bounds, pop=job.pop,
                               max_evaluations=budget, seed=job.seed, archive=arc,
                               on_generation=on_gen,
                               record_every=(1 if grid is not None else spec.record_every))
        else:
            res = minimize(
                evaluate, bounds=p.bounds, n_objs=p.n_obj,
                algorithm=job.algorithm, pop_size=job.pop, n_gen=job.gens,
                seed=job.seed,
                constraints=(p.constraints if p.has_cons else None),
                # The budget is spent in evaluations, not steps. NIMMO
                # evaluates one offspring per step and MOEA/D-DRA and -AWA a
                # fifth of the population, so a budget in generations gave them
                # 2 % and 21 % of what every other algorithm spent.
                max_evaluations=budget,
                # Under the log grid the observer must see every stride of one
                # population; on_gen itself decides which calls become records.
                on_generation=on_gen,
                record_every=(1 if grid is not None else spec.record_every),
                operator_stats=op_stats,
                **job.params,
            )
    except Exception as e:                       # one bad job must not kill the shard
        fh.close()
        meta.update(status="failed", error=f"{type(e).__name__}: {e}",
                    traceback=traceback.format_exc()[-4000:], fe=fe,
                    finished=time.strftime("%Y-%m-%dT%H:%M:%S"),
                    seconds=round(time.perf_counter() - t0, 3))
        _write_json(meta_path, meta)
        if not quiet:
            print(f"[{job.index}] {job.problem} {job.key} seed {job.seed}: FAILED {e}",
                  file=sys.stderr)
        return "failed"
    fh.close()

    F = np.asarray(res.objectives, float)
    X = np.asarray(res.variables, float)
    with (d / "final.csv").open("w", encoding="utf-8") as out:
        out.write("# mootation campaign final population v1\n")
        out.write(",".join([f"f{i+1}" for i in range(F.shape[1] if F.ndim == 2 else 0)]
                           + [f"x{i+1}" for i in range(X.shape[1] if X.ndim == 2 else 0)]) + "\n")
        for i in range(len(F)):
            out.write(",".join(f"{v:.10g}" for v in list(F[i]) + list(X[i])) + "\n")

    arc_sel = None                  # the archive scenario's points: indices into arc.points()
    if arc is not None:
        AF, AX, AE = arc.points()
        # "all": every point with its variables; "selected": the objectives of
        # every point, and the variables only of the points DSS selects
        whole = spec.archive_variables == "all"
        name = "archive.csv" if whole else "archive.csv.gz"
        with _open_text(d / name, "w") as out:
            out.write(f"# mootation campaign archive v1: every nondominated point evaluated"
                      f"{' (feasible only)' if p.has_cons else ''}, at most one per cell of a "
                      f"{arc.delta:g} grid in {arc.normalization}-normalized objectives, "
                      f"each objective's best point kept whatever its cell holds"
                      + ("" if whole else "; objectives only, the variables of the points "
                         "the archive scenario selects are in final_archive.csv.gz") + "\n")
            out.write(",".join([f"f{i+1}" for i in range(p.n_obj)]
                               + [f"x{i+1}" for i in range(p.n_vars) if whole]
                               + ["extreme"]) + "\n")
            for i in range(len(AF)):
                row = list(AF[i]) + (list(AX[i]) if whole else [])
                out.write(",".join(f"{v:.10g}" for v in row) + f",{int(AE[i])}\n")
        if len(AF) and (spec.archive_scenario or not whole):
            arc_sel = arc.select(p.pop_size)
        if not whole and arc_sel is not None:
            with _open_text(d / "final_archive.csv.gz", "w") as out:
                out.write("# mootation campaign archive selection v1: the points DSS selects "
                          "from the archive for the archive scenario\n")
                out.write(",".join([f"f{i+1}" for i in range(p.n_obj)]
                                   + [f"x{i+1}" for i in range(p.n_vars)]) + "\n")
                for i in arc_sel:
                    out.write(",".join(f"{v:.10g}" for v in list(AF[i]) + list(AX[i])) + "\n")
        lo, hi = arc.frame()
        # The frame DSS normalises by when it reduces the archive (scenario
        # "archive"), so that --recompute --scenario archive selects the same points.
        meta["archive"] = dict(arc.info(), extremes=int(AE.sum()), file=name,
                               variables=("all, in archive.csv" if whole else
                                          "the selected points', in final_archive.csv.gz"),
                               feasible_only=bool(p.has_cons),
                               frame=None if lo is None else [np.asarray(lo).tolist(),
                                                              np.asarray(hi).tolist()])
    if snap and snap_F:
        payload = dict(fe=np.asarray(snap_fe, np.int64), gen=np.asarray(snap_gen, np.int64),
                       n=np.asarray(snap_n, np.int64), F=np.vstack(snap_F))
        if keep_x:
            payload["X"] = np.vstack(snap_X)
        np.savez_compressed(d / "snapshots.npz", **payload)
        meta["snapshots"] = {"file": "snapshots.npz", "records": len(snap_F),
                             "dtype": "float32", "variables": keep_x,
                             "layout": "F (and X) rows of every record, stacked in order; "
                                       "n gives each record's row count"}
    if checkpoints:
        # the archive scenario at the record_at counts: indicators and points
        meta["archive_at"] = archive_at
        if ck_F:
            payload = dict(budget=np.asarray(ck_budget, np.int64),
                           fe=np.asarray(ck_fe, np.int64), n=np.asarray(ck_n, np.int64),
                           F=np.vstack(ck_F))
            if keep_x:
                payload["X"] = np.vstack(ck_X)
            np.savez_compressed(d / "archive_at.npz", **payload)

    extra = {"pop": p.pop_size, "hv_options": final_hv, "pareto_set": pset,
             "bounds": p.bounds, "cyclic": p.cyclic_vars}
    final = {}
    if F.size and np.all(np.isfinite(F)):
        final = M.compute(F, ref_front=ref, ideal=ideal, nadir=nadir, which=final_metrics,
                          X=X, **extra)
    if arc_sel is not None and spec.archive_scenario:
        AF, AX, _ = arc.points()
        meta["final_archive"] = M.compute(AF[arc_sel], ref_front=ref, ideal=ideal, nadir=nadir,
                                          which=final_metrics, X=AX[arc_sel], **extra)
        meta["final_archive"]["n"] = int(len(arc_sel))
    if not baseline:
        meta["operators"] = [{"operator": o, "bound_repair": r}
                             for o, r in getattr(res, "operators", [])]
    if op_stats and final:
        final.update({k: float(v) for k, v in dict(res.operator_totals).items()})
    meta["operator_stats"] = op_stats
    meta.update(status="done", fe=fe, records=n_records, final=final,
                ignored_knobs=list(res.ignored), active_n=int(res.active_n),
                finished=time.strftime("%Y-%m-%dT%H:%M:%S"),
                seconds=round(time.perf_counter() - t0, 3),
                # of `seconds`, what the trajectory records took (indicators,
                # operator statistics, snapshots, archive checkpoints)
                record_seconds=round(record_seconds, 3))
    _write_json(meta_path, meta)
    if not quiet:
        summary = ", ".join(f"{k}={v:.4g}" for k, v in final.items() if isinstance(v, float))
        print(f"[{job.index}] {job.problem} {job.key} seed {job.seed}: done "
              f"fe={fe} {summary}", file=sys.stderr)
    return "done"


def _write_text(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        fh.write(text)
    # On Windows a file another process has open cannot be replaced, and the
    # TUI and --list read every meta.json every few seconds: one collision
    # used to abort a whole campaign 8 400 jobs in. A reader holds the file
    # for milliseconds, so a few seconds of retries is plenty; on POSIX the
    # first attempt always succeeds.
    delay = 0.01
    for attempt in range(40):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == 39:
                raise
            time.sleep(delay)
            delay = min(2 * delay, 0.25)


def _write_json(path: Path, obj: dict) -> None:
    _write_text(path, json.dumps(obj, indent=1))


# ── live control of a running campaign ──────────────────────────────────────
#
# Two small files in the results root. `_workers.txt` holds the number of
# worker processes the runner keeps. It is re-read every 1.5 s: a higher number
# starts workers at once, a lower one lets the surplus finish the job it is on
# and leave, and 0 drains the campaign and stops it cleanly, with unfinished
# jobs left pending. Anything may write it — the TUI, or by hand,
# `echo 8 > results/<name>/_workers.txt`. `_runner.json` is the runner's
# heartbeat, rewritten every two seconds, so a UI can show and steer a run it
# did not start.

WORKERS_FILE = "_workers.txt"
RUNNER_FILE = "_runner.json"
_HEARTBEAT_STALE_S = 15.0


def read_workers(root: Path, default: int) -> int:
    """The worker count the running campaign should keep, as `_workers.txt` says."""
    try:
        return max(0, int((root / WORKERS_FILE).read_text(encoding="utf-8").strip()))
    except (OSError, ValueError):
        return default


def write_workers(root: Path, n: int) -> None:
    root.mkdir(parents=True, exist_ok=True)
    _write_text(root / WORKERS_FILE, f"{max(0, int(n))}\n")


def runner_state(root: Path) -> dict | None:
    """The last heartbeat of this results root's runner, with `alive` added.

    None when no runner ever wrote one. `alive` is False once the runner has
    finished or stopped, or has not written for 15 s — killed without a chance
    to say so.
    """
    try:
        with (root / RUNNER_FILE).open("r", encoding="utf-8") as fh:
            st = json.load(fh)
    except (OSError, ValueError):
        return None
    st["alive"] = (st.get("state") == "running"
                   and time.time() - float(st.get("updated", 0)) <= _HEARTBEAT_STALE_S)
    return st


# ── the whole campaign ──────────────────────────────────────────────────────


def _pool_worker(args):
    cfg_path, index, force = args
    try:
        cfg = load(cfg_path)
        validate(cfg)
        spec = campaign_spec(cfg)
        jobs = expand_jobs(cfg, spec)
        return index, run_job(jobs[index], out_root(cfg, spec), spec, force=force)
    except Exception as e:                           # noqa: BLE001
        return index, _job_crashed(index, e)


def single_threaded_blas() -> None:
    """One BLAS thread in every worker process started after this call.

    The pool is the parallelism. Left alone, NumPy's OpenBLAS reserves a buffer
    for every core in every process at import — 756 MB each on a 24-thread
    machine — and the stage-3 probe's twenty workers exhausted the commit limit
    of a 31 GB Windows machine with it (MemoryError on a 5 MB array). A value
    the user has set is kept.
    """
    for var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, "1")


def _job_crashed(index: int, e: BaseException) -> str:
    # run_job already records a failing optimisation as "failed". This is
    # everything around it — the file system, a config edited mid-run — and it
    # must cost one job, not the campaign. The job's meta.json still says
    # "running", so starting the campaign again reruns it.
    print(f"[{index}] FAILED outside the run: {type(e).__name__}: {e}", file=sys.stderr)
    return "failed"


def _worker_main(cfg_path, force, conn) -> None:
    """One worker process: asks the runner for a job, runs it, reports, repeats.

    The runner answers each request over this worker's own pipe with a job
    index, or with None when the worker is surplus or nothing is left, so it
    always knows which job a worker holds. The config is read and expanded once
    per process, not once per job.
    """
    pid = os.getpid()
    try:
        cfg = load(cfg_path)
        validate(cfg)
        spec = campaign_spec(cfg)
        jobs = expand_jobs(cfg, spec)
        root = out_root(cfg, spec)
    except Exception as e:                           # noqa: BLE001
        conn.send(("broken", pid, f"{type(e).__name__}: {e}"))
        return
    while True:
        conn.send(("ready", pid))
        index = conn.recv()
        if index is None:
            return
        try:
            status = run_job(jobs[index], root, spec, force=force)
        except Exception as e:                       # noqa: BLE001
            status = _job_crashed(index, e)
        conn.send(("done", pid, index, status))


def _readable(conns: list, timeout: float) -> list:
    """The pipes in `conns` with something to read, waiting up to `timeout`.

    One wait on Windows takes at most 63 handles, so a larger pool is polled in
    slices of 60.
    """
    from multiprocessing.connection import wait
    if len(conns) <= 60:
        return wait(conns, timeout=timeout)
    ready: list = []
    for i in range(0, len(conns), 60):
        ready += wait(conns[i:i + 60], timeout=0)
    if not ready:
        time.sleep(min(timeout, 0.05))
    return ready


def _run_dynamic(cfg_path: str, todo: list, root: Path, *, workers: int,
                 force: bool, label: str, worker=None) -> dict:
    """Run `todo` with a pool whose size follows `_workers.txt` while it runs.

    Each worker has its own pipe and asks for one job at a time, so the runner
    always knows which job a worker holds, and a worker that dies holding one
    fails exactly that job. `worker` stands in for `_worker_main` in tests.
    """
    import multiprocessing as mp
    from collections import deque
    # Nothing but a pipe crosses to a worker. The pool used to share queues,
    # whose semaphores are duplicated into a process that is still starting;
    # on a loaded Windows machine they arrived dead in every worker of a run
    # ("The handle is invalid" on the first report), the job each worker had
    # taken was lost with it, and the campaign waited forever.
    ctx = mp.get_context()
    pending = deque(j.index for j in todo)
    ceiling = os.cpu_count() or 1
    write_workers(root, workers)

    pool: dict = {}          # the runner's end of a worker's pipe -> [process, job it holds]
    retired: list = []       # workers told to leave, joined at the end
    counts = {"done": 0, "failed": 0}
    finished = 0
    want = 0
    started = time.time()
    last_read = last_beat = 0.0
    state = "running"

    def beat(st: str) -> None:
        _write_json(root / RUNNER_FILE, {
            "state": st, "pid": os.getpid(), "slice": label,
            "started": started, "updated": time.time(),
            "target": want, "active": len(pool),
            "todo": len(todo), "done": counts["done"], "failed": counts["failed"],
            "running": sorted(job for _, job in pool.values() if job is not None),
        })

    def spawn() -> None:
        here, there = ctx.Pipe()
        p = ctx.Process(target=worker or _worker_main, daemon=True,
                        args=(cfg_path, force, there))
        p.start()
        there.close()
        pool[here] = [p, None]

    def lose(conn) -> None:
        # A worker gone without being told to leave crashed inside a core or
        # was killed. The job it held failed, and the next read of
        # _workers.txt starts a replacement.
        nonlocal finished
        p, job = pool.pop(conn)
        conn.close()
        p.join(timeout=1)
        if job is not None:
            counts["failed"] += 1
            finished += 1
            print(f"[{job}] FAILED: its worker exited with code {p.exitcode}",
                  file=sys.stderr)

    try:
        while finished < len(todo) and state == "running":
            now = time.time()
            if now - last_read >= 1.5:
                last_read = now
                want = min(read_workers(root, workers), ceiling)
                busy = sum(1 for _, job in pool.values() if job is not None)
                for _ in range(min(want, busy + len(pending)) - len(pool)):
                    spawn()
                if want == 0 and not pool:
                    state = "stopped"                # drained on request
                    break
            if now - last_beat >= 2.0:
                last_beat = now
                beat("running")
            for conn in _readable(list(pool), 0.5):
                try:
                    msg = conn.recv()
                except (EOFError, OSError):
                    lose(conn)
                    continue
                if msg[0] == "ready":
                    if pending and len(pool) <= want:
                        pool[conn][1] = index = pending.popleft()
                        try:
                            conn.send(index)
                        except OSError:
                            pool[conn][1] = None
                            pending.appendleft(index)
                            lose(conn)
                        continue
                    try:                             # surplus, or nothing left: leave
                        conn.send(None)
                    except OSError:
                        pass
                    retired.append(pool.pop(conn)[0])
                    conn.close()
                elif msg[0] == "done":
                    pool[conn][1] = None
                    counts[msg[3]] = counts.get(msg[3], 0) + 1
                    finished += 1
                elif msg[0] == "broken":
                    print(f"a worker could not start: {msg[2]}", file=sys.stderr)
                    state = "broken"
                    break
            # A worker that died before it could take its end of the pipe leaves
            # no end of file to read, so the processes are checked as well.
            for conn, (p, _) in list(pool.items()):
                if p.is_alive():
                    continue
                try:
                    if conn.poll():
                        continue                     # its last words are read first
                except OSError:
                    pass
                lose(conn)
    except KeyboardInterrupt:
        state = "stopped"
        raise
    finally:
        for p, _ in pool.values():
            if p.is_alive():
                p.terminate()
        for p in [p for p, _ in pool.values()] + retired:
            p.join(timeout=5)
        beat("finished" if state == "running" else state)
    counts["pending"] = len(todo) - finished
    return counts


def run_campaign(cfg: Config, *, shard: tuple[int, int] | None = None,
                 only: list[int] | None = None, workers: int = 1,
                 force: bool = False) -> dict:
    spec = campaign_spec(cfg)
    jobs = expand_jobs(cfg, spec)
    root = out_root(cfg, spec)
    root.mkdir(parents=True, exist_ok=True)
    _write_json(root / "campaign.json", {
        "name": cfg.name, "config": str(cfg.source_path or ""),
        "problems": sorted({j.problem for j in jobs}),
        "algorithms": [a.key for a in cfg.algorithms],
        "seeds": sorted({j.seed for j in jobs}),
        "jobs": len(jobs), "spec": asdict(spec),
    })
    selected = jobs
    if only is not None:
        selected = [jobs[i] for i in only if 0 <= i < len(jobs)]
    elif shard is not None:
        i, n = shard
        selected = [j for j in jobs if j.index % n == i]
    todo = [j for j in selected if force or job_status(root, j) != "done"]
    print(f"{cfg.name}: {len(jobs)} jobs total, {len(selected)} in this slice, "
          f"{len(todo)} to run -> {root}", file=sys.stderr)

    counts = {"done": 0, "failed": 0, "skipped": len(selected) - len(todo)}
    if len(todo) > 1 and cfg.source_path is not None:
        # Worker processes whenever there is more than one job, even with
        # workers = 1, so the pool can be grown while the campaign runs.
        single_threaded_blas()
        label = (f"shard {shard[0]}/{shard[1]}" if shard is not None
                 else f"{len(only)} job(s)" if only is not None else "all")
        got = _run_dynamic(str(cfg.source_path), todo, root, workers=workers,
                           force=force, label=label)
        for k, v in got.items():
            counts[k] = counts.get(k, 0) + v
    else:
        for j in todo:
            try:
                status = run_job(j, root, spec, force=force)
            except Exception as e:                   # noqa: BLE001
                status = _job_crashed(j.index, e)
            counts[status] = counts.get(status, 0) + 1
    print(f"finished: {counts}", file=sys.stderr)
    return counts


# ── cluster helpers ─────────────────────────────────────────────────────────


def emit_slurm(cfg: Config, n_shards: int, *, time_limit: str = "24:00:00",
               partition: str | None = None, python: str = "python") -> Path:
    spec = campaign_spec(cfg)
    jobs = expand_jobs(cfg, spec)
    root = out_root(cfg, spec)
    root.mkdir(parents=True, exist_ok=True)
    cfg_path = Path(cfg.source_path).resolve()
    n_shards = max(1, min(n_shards, len(jobs)))
    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={cfg.name}",
        f"#SBATCH --array=0-{n_shards - 1}",
        "#SBATCH --ntasks=1",
        "#SBATCH --cpus-per-task=1",
        f"#SBATCH --time={time_limit}",
        f"#SBATCH --output={root.as_posix()}/slurm_%A_%a.out",
    ]
    if partition:
        lines.append(f"#SBATCH --partition={partition}")
    lines += [
        "set -u",
        f"cd {cfg_path.parent.as_posix()}",
        f'{python} -m mootation.run.campaign "{cfg_path.as_posix()}" '
        f"--shard ${{SLURM_ARRAY_TASK_ID}}/{n_shards}",
    ]
    script = root / "submit.sh"
    script.write_text("\n".join(lines) + "\n", encoding="utf-8")
    # one command per job, for GNU parallel / xargs / a PBS loop
    jobs_txt = root / "jobs.txt"
    with jobs_txt.open("w", encoding="utf-8") as fh:
        for j in jobs:
            fh.write(f'{python} -m mootation.run.campaign "{cfg_path.as_posix()}" --job {j.index}\n')
    return script


# ── reading results back (Compare / Explore) ────────────────────────────────


def read_trajectory(run_dir: Path) -> list[dict]:
    p = run_dir / "trajectory.jsonl"
    out = []
    if not p.is_file():
        return out
    with p.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def scan_results(root: Path) -> list[dict]:
    """Every run under root as {problem, algorithm, seed, status, final, seconds, fe, n_objs}."""
    rows = []
    if not root.is_dir():
        return rows
    for meta in root.glob("*/*/run_*/meta.json"):
        try:
            with meta.open("r", encoding="utf-8") as fh:
                m = json.load(fh)
        except (OSError, ValueError):
            continue
        rows.append({
            "problem": m.get("problem", meta.parents[2].name),
            "algorithm": m.get("algorithm", meta.parents[1].name),
            "seed": m.get("seed"), "status": m.get("status", "?"),
            "final": m.get("final", {}) or {}, "seconds": m.get("seconds"),
            "fe": m.get("fe"), "n_objs": m.get("n_objs"), "budget_fe": m.get("budget_fe"),
            "budget_nominal": m.get("budget_nominal"),
            "final_archive": m.get("final_archive") or {},
            "archive_at": m.get("archive_at") or {},
            "core": m.get("core"), "budget_dependent": m.get("budget_dependent"),
            "ladder_of": m.get("ladder_of"),
            "has_reference_front": m.get("has_reference_front"),
            "dir": meta.parent,
        })
    return rows


def main_rows(rows: list[dict]) -> list[dict]:
    """The full-budget runs: without the rungs of a budget ladder, which are read
    in their place at a smaller budget (rungs_in_place, cover.py), not beside them."""
    return [r for r in rows if not r.get("ladder_of")]


def rungs_in_place(rows: list[dict], at: float | None) -> list[dict]:
    """The full-budget runs, each read at the fraction `at` of its budget the way
    the budget asks: a budget-dependent algorithm's run is replaced by its
    ladder rung of exactly that budget when the campaign has one (the rung's
    final answer, filed under the full run's label), every other run is read
    off its trajectory (value_at)."""
    main = main_rows(rows)
    if at is None:
        return main
    rungs = {(r["problem"], r["ladder_of"], r["seed"], r.get("budget_fe")): r
             for r in rows if r.get("ladder_of")}
    out = []
    for r in main:
        b = int(round(float(at) * float(r.get("budget_nominal") or r.get("budget_fe") or 0)))
        rung = rungs.get((r["problem"], r["algorithm"], r["seed"], b))
        out.append(dict(rung, algorithm=r["algorithm"], rung_of=r.get("budget_fe"))
                   if rung is not None else r)
    return out


def value_at(row: dict, metric: str, at: float | None) -> float | None:
    """A run's `metric`: its final value, or at a fraction `at` of its budget.

    At a fraction it is the last trajectory record that had spent no more than
    `at * budget_fe` evaluations — the anytime reading of the same run, so ranks
    at 10 %, 25 % and 100 % of the budget come out of one campaign. It needs the
    metric among the campaign's `metrics`, which are what a trajectory records.
    A ladder rung put in a full run's place (rungs_in_place) is read at its end.
    In the archive scenario (scenario_rows) a fraction is read from the archive
    checkpoint of exactly that budget (archive_checkpoints), or is missing.
    """
    if at is None or row.get("rung_of"):
        v = row["final"].get(metric)
        return None if v is None else float(v)
    limit = float(at) * float(row.get("budget_fe") or 0)
    if row.get("scenario") == "archive":
        # the archive checkpoint of exactly that budget, if the run took one
        b = int(round(float(at) * float(row.get("budget_nominal") or row.get("budget_fe") or 0)))
        v = (row.get("archive_at") or {}).get(str(b), {}).get(metric)
        return None if v is None else float(v)
    found = None
    for rec in read_trajectory(Path(row["dir"])):
        if rec.get("fe", 0) <= limit and rec.get(metric) is not None:
            found = rec[metric]
    return None if found is None else float(found)


def compare_table(rows: list[dict], metric: str = "igd", at: float | None = None) -> dict:
    """{problem: {algorithm: (median, q1, q3, n)}} over finished runs.

    `at` reads every run at that fraction of its budget instead of at the end.
    """
    import statistics
    table: dict = {}
    grouped: dict = {}
    for r in rows:
        if r["status"] != "done":
            continue
        v = value_at(r, metric, at)
        if v is None:
            continue
        grouped.setdefault(r["problem"], {}).setdefault(r["algorithm"], []).append(float(v))
    for prob, algs in grouped.items():
        table[prob] = {}
        for alg, vals in algs.items():
            vals = sorted(vals)
            med = statistics.median(vals)
            q1 = vals[max(0, int(0.25 * (len(vals) - 1)))]
            q3 = vals[min(len(vals) - 1, int(math.ceil(0.75 * (len(vals) - 1))))]
            table[prob][alg] = (med, q1, q3, len(vals))
    return table


def write_compare_csv(table: dict, path: Path, metric: str) -> None:
    algs = sorted({a for d in table.values() for a in d})
    with path.open("w", encoding="utf-8") as fh:
        fh.write("problem," + ",".join(f"{a}_{metric}_median,{a}_q1,{a}_q3,{a}_n" for a in algs) + "\n")
        for prob in sorted(table):
            cells = []
            for a in algs:
                if a in table[prob]:
                    med, q1, q3, n = table[prob][a]
                    cells.append(f"{med:.6g},{q1:.6g},{q3:.6g},{n}")
                else:
                    cells.append(",,,")
            fh.write(prob + "," + ",".join(cells) + "\n")


def problem_family(name: str) -> str:
    """The family of a registry name: DTLZ2_3D -> DTLZ, MaF10_8D -> MaF, ZDT1 -> ZDT.

    The same stem rule as `mootation.benchmarks.families()`.
    """
    stem = name.split("_")[0]
    return "".join(c for c in stem if not c.isdigit()) or stem


def rank_table(rows: list[dict], metric: str = "igd", at: float | None = None) -> dict:
    """Where each algorithm stands, as one mean rank per group of problems.

    On every problem the algorithms are ranked by their median `metric` over
    seeds (1 is best; equal medians share the average of their ranks; a
    non-finite median ranks last). The ranks are then averaged over all
    problems, over each family and over each objective count. A group column
    appears only when the campaign has more than one family, or more than one
    objective count. A win is a problem on which the algorithm's median is the
    best one, a shared best included.

    An algorithm with no finished run on some problem is averaged over the
    problems it has, and the count next to every mean says how many that was,
    so a partial campaign cannot pass for a complete one.

    A mean rank rewards consistency, not margin: an algorithm second everywhere
    outranks one that alternates between first and last. Read it next to the
    medians, not instead of them.

        {"metric": "igd", "lower_better": True, "n_problems": 45,
         "groups": ["DTLZ", "WFG", "ZDT", "M=2", "M=3", "M=5"],
         "algorithms": [(name, {"all": (mean, n), "DTLZ": (mean, n), ...,
                                "wins": w}), ...]}      # best mean rank first
    """
    table = compare_table(rows, metric, at)
    lower_better = metric not in HIGHER_IS_BETTER
    n_objs: dict = {}
    for r in rows:
        if r.get("n_objs") is not None:
            n_objs.setdefault(r["problem"], int(r["n_objs"]))

    def key(med: float) -> tuple:
        if not math.isfinite(med):
            return (1, 0.0)
        return (0, med if lower_better else -med)

    ranks: dict = {}
    wins: dict = {}
    for prob, algs in table.items():
        labels = ["all", problem_family(prob)]
        if prob in n_objs:
            labels.append(f"M={n_objs[prob]}")
        order = sorted(algs, key=lambda a: key(algs[a][0]))
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and key(algs[order[j + 1]][0]) == key(algs[order[i]][0]):
                j += 1
            rank = (i + j) / 2 + 1
            for a in order[i:j + 1]:
                per = ranks.setdefault(a, {})
                for g in labels:
                    per.setdefault(g, []).append(rank)
                if i == 0 and math.isfinite(algs[a][0]):
                    wins[a] = wins.get(a, 0) + 1
            i = j + 1

    fams = sorted({problem_family(p) for p in table})
    ms = sorted({n_objs[p] for p in table if p in n_objs})
    groups = ((fams if len(fams) > 1 else [])
              + ([f"M={m}" for m in ms] if len(ms) > 1 else []))
    out = []
    for a, per in ranks.items():
        entry: dict = {g: (sum(v) / len(v), len(v)) for g, v in per.items()}
        entry["wins"] = wins.get(a, 0)
        out.append((a, entry))
    out.sort(key=lambda t: (t[1]["all"][0], -t[1]["wins"], t[0]))
    return {"metric": metric, "at": at, "lower_better": lower_better,
            "n_problems": len(table), "groups": groups, "algorithms": out}


def format_ranks(ranks: dict) -> str:
    """The rank table as fixed-width text, for the terminal."""
    groups = ranks["groups"]
    at = f" at {ranks['at']:.0%} of the budget" if ranks.get("at") is not None else ""
    lines = [f"mean rank by median {ranks['metric']}{at} over {ranks['n_problems']} problem(s): "
             f"1 = best, ties share the average rank; probs = problems ranked on",
             f"{'algorithm':<14}{'mean':>7}{'wins':>6}{'probs':>6}"
             + "".join(f"{g[:9]:>10}" for g in groups)]
    from .postprocess import budget_note
    marked = False
    for alg, e in ranks["algorithms"]:
        mean, n = e["all"]
        name = alg[:13] + budget_note(alg)
        marked |= name != alg[:13]
        line = f"{name:<14}{mean:>7.2f}{e['wins']:>6}{n:>6}"
        for g in groups:
            line += f"{e[g][0]:>10.2f}" if g in e else f"{'-':>10}"
        lines.append(line)
    if marked:
        lines.append("* runs on a schedule of the budget share spent: compare its runs at "
                     "different budgets by fraction of budget, not by evaluations")
    return "\n".join(lines)


def write_rank_csv(ranks: dict, path: Path) -> None:
    groups = ranks["groups"]
    with path.open("w", encoding="utf-8") as fh:
        fh.write("algorithm,mean_rank,problems,wins"
                 + "".join(f",{g}_mean_rank,{g}_problems" for g in groups) + "\n")
        for alg, e in ranks["algorithms"]:
            mean, n = e["all"]
            cells = [alg, f"{mean:.6g}", str(n), str(e["wins"])]
            for g in groups:
                cells += [f"{e[g][0]:.6g}", str(e[g][1])] if g in e else ["", ""]
            fh.write(",".join(cells) + "\n")


# ── indicators added after the fact ─────────────────────────────────────────


def _open_text(path: Path, mode: str = "r"):
    """A results text file, gzip-compressed when its name ends in .gz."""
    if path.suffix == ".gz":
        import gzip
        return gzip.open(path, mode + "t", compresslevel=6, encoding="utf-8", newline="\n")
    return path.open(mode, encoding="utf-8")


def _read_points(path: Path, m: int, n: int):
    """(F, X) from a final.csv, archive.csv or final_archive.csv.gz: m objective
    columns, then n variables."""
    import numpy as np
    rows = []
    with _open_text(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("f1"):
                continue
            rows.append([float(v) for v in line.split(",")])
    width = max((len(r) for r in rows), default=m + n)
    A = np.asarray([r + [np.nan] * (width - len(r)) for r in rows], float).reshape(-1, width)
    X = A[:, m:m + n] if width >= m + n else None
    return A[:, :m], X


def _recompute_one(args) -> str:
    run_dir, names, n_ref, hv_options, scenario = args
    import numpy as np
    from ..benchmarks import get as bench_get
    from . import metrics as M
    from .archive import dss_order
    run_dir = Path(run_dir)
    meta_path = run_dir / "meta.json"
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("status") != "done":
            return "skipped"
        p = bench_get(meta["problem"])
        m = int(meta.get("n_objs") or p.n_obj)
        n = int(meta.get("n_vars") or p.n_vars)
        k = int(meta.get("n_ref") or n_ref)
        ref = pset = None
        if any(name in NEEDS_FRONT for name in names) and callable(p.pareto_front):
            ref = np.asarray(p.pareto_front(k), float)
        if any(name in NEEDS_SET for name in names) and callable(p.pareto_set):
            pset = np.asarray(p.pareto_set(k), float)
        kw = dict(ref_front=ref, ideal=p.ideal, nadir=p.nadir, which=names, pop=p.pop_size,
                  hv_options=hv_options, pareto_set=pset, bounds=p.bounds,
                  cyclic=p.cyclic_vars)
        if scenario == "archive" and (run_dir / "final_archive.csv.gz").is_file():
            # archive_variables = "selected": the run kept its selection
            F, X = _read_points(run_dir / "final_archive.csv.gz", m, n)
            out = dict(meta.get("final_archive") or {})
            out.update(M.compute(F, X=X, **kw))
            out["n"] = int(len(F))
            meta["final_archive"] = out
        elif scenario == "archive":
            path = run_dir / "archive.csv"
            if not path.is_file():
                return "skipped"
            F, X = _read_points(path, m, n)
            if not len(F):
                return "skipped"
            arc_meta = meta.get("archive") or {}
            frame = arc_meta.get("frame")
            if arc_meta.get("normalization") == "archive":
                lo = hi = None                   # the kept points' range, as GridArchive.select
            elif frame:
                lo, hi = frame
            elif p.ideal is not None and p.nadir is not None:
                lo, hi = p.ideal, p.nadir
            else:
                lo = hi = None                   # the archive's own range
            idx = (np.arange(len(F)) if len(F) <= p.pop_size
                   else dss_order(F, k=p.pop_size, ideal=lo, nadir=hi))
            out = dict(meta.get("final_archive") or {})
            out.update(M.compute(F[idx], X=None if X is None else X[idx], **kw))
            out["n"] = int(len(idx))
            meta["final_archive"] = out
        else:
            F, X = _read_points(run_dir / "final.csv", m, n)
            if not F.size or not np.all(np.isfinite(F)):
                return "skipped"
            out = dict(meta.get("final") or {})
            out.update(M.compute(F, X=X, **kw))
            meta["final"] = out
        _write_json(meta_path, meta)
        return "done"
    except Exception as e:                           # noqa: BLE001
        print(f"{run_dir}: {type(e).__name__}: {e}", file=sys.stderr)
        return "failed"


def recompute_final(root: Path, names: list, *, workers: int = 1, n_ref: int = 1000,
                    hv_options: dict | None = None, scenario: str = "final",
                    problems=None) -> dict:
    """Compute `names` from every finished run's final.csv and merge them into its meta.json.

    The final population is on disk, so an indicator added after a campaign
    ran costs its own arithmetic, not a rerun. Trajectories keep what they
    recorded. With scenario "archive" the same is done for the run archive
    reduced to the population size by DSS, into meta["final_archive"]: the
    selection the run kept (final_archive.csv.gz, archive_variables =
    "selected"), or archive.csv reduced now — which also gives that scenario
    to campaigns run before it existed, as long as they kept an archive.
    """
    tasks = [(str(m.parent), list(names), n_ref, hv_options, scenario)
             for m in root.glob("*/*/run_*/meta.json")
             if problems is None or m.parents[2].name in problems]
    counts: dict = {}
    if workers > 1 and len(tasks) > 1:
        import multiprocessing as mp
        single_threaded_blas()
        with mp.Pool(processes=workers) as pool:
            for status in pool.imap_unordered(_recompute_one, tasks, chunksize=8):
                counts[status] = counts.get(status, 0) + 1
    else:
        for task in tasks:
            status = _recompute_one(task)
            counts[status] = counts.get(status, 0) + 1
    return counts


def _retrajectory_one(args) -> str:
    run_dir, names, hv_settings = args
    import numpy as np
    from ..benchmarks import get as bench_get
    from . import metrics as M
    run_dir = Path(run_dir)
    meta_path = run_dir / "meta.json"
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        snap_path = run_dir / "snapshots.npz"
        if meta.get("status") != "done" or not snap_path.is_file():
            return "skipped"
        with np.load(snap_path) as z:
            S = {k: z[k] for k in z.files}
        traj_path = run_dir / "trajectory.jsonl"
        recs = [json.loads(line) for line in traj_path.read_text(encoding="utf-8").splitlines()
                if line.strip()]
        if [r.get("fe") for r in recs] != S["fe"].tolist():
            print(f"{run_dir}: the snapshots and the trajectory do not line up", file=sys.stderr)
            return "failed"
        # the decision-space indicators only where the snapshots kept the variables
        todo = [m for m in names if m not in NEEDS_X or "X" in S]
        if not todo:
            return "skipped"
        p = bench_get(meta["problem"])
        k = int(meta.get("n_ref") or 1000)
        ref = (np.asarray(p.pareto_front(k), float)
               if any(m in NEEDS_FRONT for m in todo) and callable(p.pareto_front) else None)
        pset = (np.asarray(p.pareto_set(k), float)
                if any(m in NEEDS_SET for m in todo) and callable(p.pareto_set) else None)
        # the trajectory's hypervolume settings (run_job): exact up to
        # trajectory_hv_max_m objectives, Monte Carlo or nothing above
        exact_max_m, mc_samples, traj_max_m, traj_mc = hv_settings
        hvo = {"exact_max_m": exact_max_m, "mc_samples": mc_samples}
        if traj_max_m and p.n_obj > traj_max_m:
            if traj_mc > 0:
                hvo = {"exact_max_m": 0, "mc_samples": traj_mc}
            else:
                todo = [m for m in todo if m not in HV_NAMES]
        start = 0
        for rec, n in zip(recs, S["n"].tolist()):
            F = S["F"][start:start + n].astype(float)
            X = S["X"][start:start + n].astype(float) if "X" in S else None
            start += n
            if n and rec.get("finite", True):
                rec.update(M.compute(F, ref_front=ref, ideal=p.ideal, nadir=p.nadir, which=todo,
                                     pop=p.pop_size, hv_options=hvo, X=X, pareto_set=pset,
                                     bounds=p.bounds, cyclic=p.cyclic_vars))
        tmp = traj_path.with_suffix(".jsonl.tmp")
        tmp.write_text("".join(json.dumps(r, separators=(",", ":")) + "\n" for r in recs),
                       encoding="utf-8")
        os.replace(tmp, traj_path)
        meta["trajectory_metrics"] = list(dict.fromkeys([*meta.get("trajectory_metrics", []),
                                                         *todo]))
        _write_json(meta_path, meta)
        return "done"
    except Exception as e:                           # noqa: BLE001
        print(f"{run_dir}: {type(e).__name__}: {e}", file=sys.stderr)
        return "failed"


def recompute_trajectory(root: Path, names: list, *, workers: int = 1,
                         hv_settings: tuple = (5, 100_000, 0, 0), problems=None) -> dict:
    """Compute `names` at every trajectory record of every finished run that kept
    snapshots ([campaign] snapshots), from snapshots.npz, into its
    trajectory.jsonl, and add them to meta["trajectory_metrics"]. An indicator
    thought of after a campaign costs its arithmetic, not a rerun. igdx, cr and
    pdist need the snapshots' variables (snapshot_variables); a run without them
    gets the others. `hv_settings`: (hv_exact_max_m, hv_mc_samples,
    trajectory_hv_max_m, trajectory_hv_mc_samples), as the campaign recorded.
    """
    tasks = [(str(m.parent), list(names), tuple(hv_settings))
             for m in root.glob("*/*/run_*/meta.json")
             if problems is None or m.parents[2].name in problems]
    counts: dict = {}
    if workers > 1 and len(tasks) > 1:
        import multiprocessing as mp
        single_threaded_blas()
        with mp.Pool(processes=workers) as pool:
            for status in pool.imap_unordered(_retrajectory_one, tasks, chunksize=8):
                counts[status] = counts.get(status, 0) + 1
    else:
        for task in tasks:
            status = _retrajectory_one(task)
            counts[status] = counts.get(status, 0) + 1
    return counts


def scenario_rows(rows: list[dict], scenario: str) -> list[dict]:
    """The rows as the tables read them: "archive" puts final_archive in place of
    final, and value_at reads such a row at a smaller budget from its archive
    checkpoints (archive_at) instead of its trajectory."""
    if scenario != "archive":
        return rows
    return [dict(r, final=r.get("final_archive") or {}, scenario="archive") for r in rows]


# ── CLI ─────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="mootation.run.campaign",
                                 description="Run a benchmark campaign described in TOML.")
    ap.add_argument("config")
    ap.add_argument("--list", action="store_true", help="print the job list and exit")
    ap.add_argument("--shard", help="i/n: run the jobs with index %% n == i")
    ap.add_argument("--job", type=int, action="append", help="run one job by index (repeatable)")
    ap.add_argument("--workers", type=int, default=1,
                    help="worker processes to start with; change it while the campaign runs by "
                         "writing the number into <results>/_workers.txt (0 drains and stops)")
    ap.add_argument("--force", action="store_true", help="rerun jobs already marked done")
    ap.add_argument("--emit-slurm", type=int, metavar="N",
                    help="write <out>/submit.sh (an array of N shards) and jobs.txt, then exit")
    ap.add_argument("--time", default="24:00:00", help="SLURM time limit for --emit-slurm")
    ap.add_argument("--partition", default=None, help="SLURM partition for --emit-slurm")
    ap.add_argument("--python", default="python", help="interpreter name in the emitted scripts")
    ap.add_argument("--compare", metavar="METRIC", help="print the median table for METRIC and exit")
    ap.add_argument("--ranks", metavar="METRIC",
                    help="print each algorithm's mean rank by median METRIC, overall, per "
                         "family and per objective count, and exit")
    ap.add_argument("--at", type=float, metavar="FRACTION",
                    help="with --compare or --ranks: read every run at this fraction of its "
                         "budget, from its trajectory, instead of at the end")
    ap.add_argument("--problems", metavar="NAMES",
                    help="only these comma-separated problems of the campaign's selection, for "
                         "--list, running (with --shard and --force), the tables and "
                         "--recompute: e.g. rerun the problems whose reference front changed "
                         "with --problems ZDT3,WFG3_5D --force")
    ap.add_argument("--recompute", metavar="METRICS",
                    help="compute these comma-separated metrics from every finished run's "
                         "final.csv (with --scenario archive: its final_archive.csv.gz, or "
                         "its archive.csv reduced by DSS), "
                         "store them in its meta.json, and exit (honours --workers)")
    ap.add_argument("--recompute-trajectory", metavar="METRICS",
                    help="compute these comma-separated metrics at every trajectory record of "
                         "every finished run that kept snapshots (snapshots.npz), store them in "
                         "its trajectory.jsonl, and exit (honours --workers); igdx, cr and pdist "
                         "need the snapshots' variables")
    ap.add_argument("--scenario", choices=("final", "archive"), default="final",
                    help="which answer of a run the tables, --cover and --recompute read: "
                         "'final', what the algorithm returned, or 'archive', the run archive "
                         "reduced to the problem's population size by DSS — with --at, at its "
                         "archive checkpoints (default final)")
    ap.add_argument("--reference", metavar="ALG",
                    help="with --compare or --ranks: test every algorithm against ALG — the "
                         "exact Wilcoxon rank-sum per problem (Holm over the problems) with "
                         "the Vargha-Delaney A12, and the exact signed-rank test on the "
                         "medians across problems (Holm over the algorithms)")
    ap.add_argument("--alpha", type=float, default=0.05,
                    help="significance level for --reference (default 0.05)")
    ap.add_argument("--ci", action="store_true",
                    help="with --ranks: a 95 %% bootstrap interval for every mean rank, "
                         "resampling the problems")
    ap.add_argument("--by", metavar="KEY",
                    help="with --ranks: mean ranks within groups of problems sharing a "
                         "property: front, multimodal, deceptive, bias, scaled, separable, "
                         "centre (benchmarks/properties.py)")
    ap.add_argument("--gap", metavar="METRIC",
                    help="print each algorithm's median gap to the best final METRIC any run "
                         "reached, per problem, and exit")
    ap.add_argument("--zero-share", nargs="?", const="hv", metavar="METRIC",
                    help="print the share of seeds whose final METRIC (default hv) is exactly "
                         "0, per problem, and exit")
    ap.add_argument("--ecdf", metavar="METRIC",
                    help="print the runtime ECDF of METRIC over (run, target) pairs, targets "
                         "at fixed distances from the best known value, and exit")
    ap.add_argument("--bias", action="store_true",
                    help="print where each algorithm's final populations sit in the box: "
                         "per variable a chi-square against uniform over 10 bins, and the "
                         "shares near the bounds and in the centre (the uninformative "
                         "problems make any preference the algorithm's own)")
    ap.add_argument("--seed-distance", action="store_true",
                    help="print whether each algorithm's seeds find the same solutions: the "
                         "chamfer distance between the non-dominated sets of two runs in "
                         "normalised variables, over every pair of seeds, beside pdist")
    ap.add_argument("--eps-table", action="store_true",
                    help="print the binary multiplicative epsilon between every two "
                         "algorithms per problem, median over the pairs of runs, on the raw "
                         "objectives (no reference, no frame; values must be >= 0)")
    ap.add_argument("--magnitude", action="store_true",
                    help="experiment D7: rank the algorithms by the magnitude of the "
                         "dominated set and by hv_h on the problems with at most three "
                         "objectives, and print how far the two orders agree")
    ap.add_argument("--interpolation", choices=("step", "linear"), default="step",
                    help="with --ecdf: a target reached between two records is charged to the "
                         "later record (step, the default) or interpolated (linear)")
    ap.add_argument("--cover", action="store_true",
                    help="the smallest sets of algorithms that cover the problems at each level "
                         "and budget, exact and greedy, the curve k -> share covered, and the "
                         "problems nobody covers (cover.py); CSV under <results>/_cover/")
    ap.add_argument("--cover-taus", default="0.1,0.03,0.01,0.003", metavar="LIST",
                    help="with --cover: the levels of gdp_norm and of the hv_h gap, coarse to "
                         "fine, comma-separated (default 0.1,0.03,0.01,0.003)")
    ap.add_argument("--cover-floor-taus", default="1,0.5,0.25,0.1", metavar="LIST",
                    help="with --cover: the levels of igdp_norm, as igdp_norm <= (1 + tau) x the "
                         "problem's floor, as many as --cover-taus (default 1,0.5,0.25,0.1)")
    ap.add_argument("--cover-keep-caveats", action="store_true",
                    help="with --cover: keep the problems whose reference front is doubtful "
                         "(DTLZ5/6 and MaF6 from four objectives, WFG3, DTLZ1 at five), which "
                         "are left out of the coverage otherwise")
    ap.add_argument("--cover-budgets", metavar="LIST",
                    help="with --cover: the budgets, comma-separated (default the campaign's "
                         "ladder and full budget)")
    ap.add_argument("--cover-seeds", type=int, default=7, metavar="N",
                    help="with --cover: a problem is covered when N seeds reach the level (7)")
    ap.add_argument("--cover-max-sets", type=int, default=20, metavar="N",
                    help="with --cover: list at most N smallest sets per case (20)")
    ap.add_argument("--cover-bootstrap", type=int, default=0, metavar="N",
                    help="with --cover: N resamplings of the seeds, and how often each "
                         "algorithm is in a smallest set (0: off)")
    args = ap.parse_args(argv)

    try:
        cfg = load(args.config)
    except ConfigError as e:
        print(f"config error — {e}", file=sys.stderr)
        return 1
    problems = validate(cfg)
    if problems:
        print(f"{len(problems)} problem(s) in the config:", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        return 1
    if cfg.kind != "builtin":
        print("a campaign needs [problem] kind = \"builtin\" and a [benchmarks] selection",
              file=sys.stderr)
        return 1
    try:
        spec = campaign_spec(cfg)
        jobs = expand_jobs(cfg, spec)
    except ConfigError as e:
        print(f"config error — {e}", file=sys.stderr)
        return 1
    root = out_root(cfg, spec)

    only_problems = None
    if args.problems is not None:
        only_problems = [q.strip() for q in args.problems.split(",") if q.strip()]
        have = {j.problem for j in jobs}
        missing = [q for q in only_problems if q not in have]
        if missing or not only_problems:
            print(f"--problems: not in this campaign's selection: "
                  f"{', '.join(missing) or '(none given)'}", file=sys.stderr)
            return 1
        if args.job:
            print("--problems and --job both pick jobs; give one of them", file=sys.stderr)
            return 1
        if args.emit_slurm:
            print("--problems does not combine with --emit-slurm, whose array covers the "
                  "whole campaign; use --shard with --problems on each node instead",
                  file=sys.stderr)
            return 1
        only_problems = set(only_problems)
        jobs = [j for j in jobs if j.problem in only_problems]

    if args.list:
        for j in jobs:
            st = job_status(root, j)
            note = f"  [{j.pop_note}]" if j.pop_note else ""
            print(f"{j.index:5d} {j.problem:<14} {j.key:<14} seed {j.seed:<3} "
                  f"pop {j.pop:<4} gens {j.gens:<5} fe {j.budget:<8} {st}{note}")
        print(f"\n{len(jobs)} jobs -> {root}", file=sys.stderr)
        return 0
    for flag, metric in (("--compare", args.compare), ("--ranks", args.ranks),
                         ("--gap", args.gap), ("--zero-share", args.zero_share),
                         ("--ecdf", args.ecdf)):
        if metric is not None and metric not in TABLE_NAMES:
            print(f"{flag}: unknown metric '{metric}'; known: {', '.join(TABLE_NAMES)}",
                  file=sys.stderr)
            return 1
    if args.at is not None and not 0.0 < args.at <= 1.0:
        print("--at wants a fraction of the budget in (0, 1], e.g. 0.25", file=sys.stderr)
        return 1
    if args.scenario == "archive" and args.ecdf:
        print("--scenario archive has no trajectory, only the end of a run and its archive "
              "checkpoints: it does not combine with --ecdf", file=sys.stderr)
        return 1
    set_tables = args.seed_distance or args.eps_table or args.magnitude
    if set_tables and args.scenario == "archive":
        print("--seed-distance, --eps-table and --magnitude read the final populations "
              "(final.csv); drop --scenario archive", file=sys.stderr)
        return 1
    if (args.ci or args.by) and not args.ranks:
        print("--ci and --by go with --ranks METRIC", file=sys.stderr)
        return 1
    if args.reference and not (args.ranks or args.compare):
        print("--reference goes with --ranks or --compare METRIC", file=sys.stderr)
        return 1
    if args.recompute:
        names = [m.strip() for m in args.recompute.split(",") if m.strip()]
        bad = [m for m in names if m not in METRIC_NAMES]
        if bad or not names:
            stats = [m for m in bad if m in RUN_STATS]
            why = (f" ({', '.join(stats)} describe the run itself: only a run with "
                   f"operator_stats = true records them)" if stats else "")
            print(f"--recompute: unknown metric(s) {', '.join(bad) or '(none given)'}{why}; "
                  f"known: {', '.join(METRIC_NAMES)}", file=sys.stderr)
            return 1
        counts = recompute_final(root, names, workers=args.workers, n_ref=spec.n_ref,
                                 hv_options={"exact_max_m": spec.hv_exact_max_m,
                                             "mc_samples": spec.hv_mc_samples},
                                 scenario=args.scenario, problems=only_problems)
        print(f"recomputed {', '.join(names)} ({args.scenario}): {counts}", file=sys.stderr)
        return 0 if counts.get("failed", 0) == 0 else 2
    if args.recompute_trajectory:
        names = [m.strip() for m in args.recompute_trajectory.split(",") if m.strip()]
        bad = [m for m in names if m not in METRIC_NAMES]
        if bad or not names:
            print(f"--recompute-trajectory: unknown metric(s) {', '.join(bad) or '(none given)'}"
                  f"; known: {', '.join(METRIC_NAMES)}", file=sys.stderr)
            return 1
        counts = recompute_trajectory(
            root, names, workers=args.workers, problems=only_problems,
            hv_settings=(spec.hv_exact_max_m, spec.hv_mc_samples,
                         spec.trajectory_hv_max_m, spec.trajectory_hv_mc_samples))
        print(f"recomputed {', '.join(names)} along the trajectories: {counts}", file=sys.stderr)
        return 0 if counts.get("failed", 0) == 0 else 2
    if args.cover:
        from . import cover as C
        try:
            taus = tuple(float(v) for v in args.cover_taus.split(",") if v.strip())
            floor_taus = tuple(float(v) for v in args.cover_floor_taus.split(",") if v.strip())
            budgets = (tuple(int(v) for v in args.cover_budgets.split(",") if v.strip())
                       if args.cover_budgets else
                       tuple(sorted(set(spec.ladder) | {spec.budget_fe})) if spec.budget_fe
                       else C.BUDGETS)
        except ValueError:
            print("--cover-taus and --cover-floor-taus want numbers and --cover-budgets "
                  "integers, comma-separated", file=sys.stderr)
            return 1
        if len(floor_taus) != len(taus):
            print(f"--cover-floor-taus gives {len(floor_taus)} levels and --cover-taus "
                  f"{len(taus)}: level k pairs the two k-th values", file=sys.stderr)
            return 1
        print(C.run(root, taus=taus, floor_taus=floor_taus, budgets=budgets,
                    min_seeds=args.cover_seeds, max_sets=args.cover_max_sets,
                    replicates=args.cover_bootstrap, workers=args.workers,
                    scenario=args.scenario, n_ref=spec.n_ref,
                    keep_caveats=args.cover_keep_caveats))
        return 0
    rows = None
    if args.bias or set_tables or any(v is not None for v in (
            args.ranks, args.compare, args.gap, args.zero_share, args.ecdf)):
        from . import report as R
        rows = rungs_in_place(scenario_rows(scan_results(root), args.scenario), args.at)
        if only_problems is not None:
            rows = [r for r in rows if r["problem"] in only_problems]
        try:
            if args.reference:
                metric = args.ranks or args.compare
                if not any(r["algorithm"] == args.reference for r in rows):
                    print(f"--reference: no run of '{args.reference}' under {root}",
                          file=sys.stderr)
                    return 1
                print(R.format_reference(R.reference_report(
                    rows, metric, args.reference, at=args.at, scenario=args.scenario,
                    alpha=args.alpha)))
                print()
            if args.ranks and args.by:
                print(R.format_property_ranks(R.property_ranks(
                    rows, args.ranks, args.by, at=args.at, scenario=args.scenario),
                    args.ranks, args.by))
            elif args.ranks:
                table = rank_table(rows, args.ranks, args.at)
                print(format_ranks(table))
                if args.ci:
                    ci = R.rank_intervals(rows, args.ranks, at=args.at, scenario=args.scenario)
                    print("\n95 % bootstrap interval of each mean rank (problems resampled "
                          "2000 times)")
                    for alg, _ in table["algorithms"]:
                        if alg in ci:
                            lo, hi = ci[alg]
                            print(f"  {R.mark_budget(alg)[:16]:<16} [{lo:5.2f}, {hi:5.2f}]")
            if args.gap:
                print(R.gap_report(rows, args.gap))
            if args.zero_share:
                print(R.zero_share_report(rows, args.zero_share))
            if args.ecdf:
                print(R.format_ecdf(R.ecdf_report(rows, args.ecdf,
                                                  interpolation=args.interpolation)))
            if args.bias:
                print(R.format_bias(R.structural_bias(rows)))
            if args.seed_distance:
                print(R.format_seed_distance(R.seed_distance(rows)))
            if args.eps_table:
                print(R.format_eps_table(R.eps_table(rows)))
            if args.magnitude:
                print(R.format_magnitude(R.magnitude_experiment(rows)))
        except ValueError as e:
            print(str(e), file=sys.stderr)
            return 1
        if not args.compare:
            return 0
    if args.compare:
        from .postprocess import budget_note
        at = f" at {args.at:.0%} of the budget" if args.at is not None else ""
        print(f"median {args.compare}{at} ({args.scenario}), runs per cell in brackets")
        table = compare_table(rows, args.compare, args.at)
        algs = sorted({a for d in table.values() for a in d})
        print("problem".ljust(16) + "".join((a[:13] + budget_note(a)).rjust(15) for a in algs))
        for prob in sorted(table):
            cells = []
            for a in algs:
                if a in table[prob]:
                    med, q1, q3, n = table[prob][a]
                    cells.append(f"{med:.4g} ({n})".rjust(15))
                else:
                    cells.append("-".rjust(15))
            print(prob.ljust(16) + "".join(cells))
        return 0
    if args.emit_slurm:
        script = emit_slurm(cfg, args.emit_slurm, time_limit=args.time,
                            partition=args.partition, python=args.python)
        print(f"wrote {script} and {script.parent / 'jobs.txt'} ({len(jobs)} jobs)")
        return 0

    shard = None
    if args.shard:
        try:
            i, n = (int(v) for v in args.shard.split("/"))
        except ValueError:
            print("--shard wants i/n, e.g. 3/40", file=sys.stderr)
            return 1
        if not (0 <= i < n):
            print("--shard: i must satisfy 0 <= i < n", file=sys.stderr)
            return 1
        shard = (i, n)
    only = args.job
    if only_problems is not None:
        # the problems' jobs, by their indices in the whole campaign, so that
        # --shard i/n splits them the same way on every machine
        only = [j.index for j in jobs if shard is None or j.index % shard[1] == shard[0]]
        shard = None
    counts = run_campaign(cfg, shard=shard, only=only, workers=args.workers,
                          force=args.force)
    return 0 if counts.get("failed", 0) == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
