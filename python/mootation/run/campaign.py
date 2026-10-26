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
import bisect
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
from .algorithms import (EXACT_LATTICE, FREE_H, K_DIVISIBLE, check_pop, seed_independent,
                         nearest_lattice_sizes)
from .metric_names import (DENSE_SUFFIX, POOLED_SUFFIX, HIGHER_IS_BETTER, HV_NAMES,
                           METRIC_NAMES, NEEDS_FRONT, NEEDS_SET, NEEDS_X, RUN_STATS, DE_STATS,
                           SA_STATS, SET_METRICS, TABLE_NAMES, base_name, is_table_name)

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
    # the first population (config.Algorithm: init, init_share)
    init: str = "uniform"
    init_share: float = 0.0

    @property
    def key(self) -> str:
        return self.label or self.algorithm

    @property
    def budget(self) -> int:
        return self.evaluations or self.pop * self.gens

    @property
    def rel_dir(self) -> Path:
        return Path(self.problem) / self.key / f"run_{self.seed}"


def shard_of(job: Job, k: int) -> int:
    """The part of k a job belongs to (--shard i/k; task 11 of 2026-10-18, §5):
    a hash of its problem, algorithm and seed (its rel_dir), the same on every
    machine and whatever else the campaign holds, so that k machines running
    parts 0..k-1 of one config run every job once."""
    import hashlib
    digest = hashlib.sha256(job.rel_dir.as_posix().encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % k


@dataclass
class CampaignSpec:
    out: str = "results/{name}"
    budget_fe: int = 0
    record_every: int = 1
    metrics: tuple = METRICS_DEFAULT
    final_metrics: tuple = ()          # empty: the same as metrics
    n_ref: int = 1000
    # The front indicators against the dense reference too, where a problem
    # has one (registry.dense_reference: 4 to 10 objectives), as
    # "<indicator>_ref_v3" beside the usual values (task 6, item 9); on by
    # default — the usual values do not change.
    dense_reference: bool = True
    # IGDX, CR and PSP along the trajectory and in `final` wherever the
    # problem has a Pareto-set sample, besides `metrics` (supplement 3 of task
    # 7, §5.3); on by default.
    set_metrics: bool = True
    # The second answer for those indicators (task 15, the owner's word of
    # 2026-10-22), on the multi-modal problems (properties: multimodal_ps,
    # several Pareto-set parts): N points spread out in the decision space
    # (Tanabe's rule, archive.spread_select), from "population" — the final population's
    # non-dominated points — or "evaluated" — every point evaluated that no
    # other dominates, equal objective vectors with other variables kept apart
    # (archive.SetArchive), which an archive in the objective space glues
    # together (the default since 2026-10-22, task 16). Its IGDX, CR and PSP go
    # to meta["set_answer"]; "off" makes none.
    set_answer: str = "evaluated"
    seeds: tuple = ()
    # "budgets" (the default since task 10 of 2026-10-18): a trajectory point
    # at each count of budget_grid — 100, 150, 200, 300, 500, 700, 1 000, ...,
    # 25 000 among them — and at the end, taken at exactly that many
    # evaluations: the population the run had after at most b evaluations (the
    # last generation that did not pass b), the run archive and everything
    # evaluated after exactly b (the population of a core that evaluates one
    # point at a time: as the observer last saw it, every population size of
    # evaluations; operator statistics and coordinate diagnostics: as they are
    # when the record is taken, at the first generation past b).
    # "generations": a trajectory point every record_every generations.
    # "log": at fixed evaluation counts
    # 10^(j/record_per_decade), the SAME counts at every budget, so runs of a
    # budget ladder line up (log_grid), at the first generation that reaches
    # each. A campaign whose results directory was started under another grid
    # keeps it when the config does not say (campaign_spec); one with
    # hard_budgets defaults to "generations", which they need.
    record_grid: str = "budgets"
    record_per_decade: int = 10
    # Extra evaluation counts for the "log" and "budgets" grids — a budget
    # ladder's rungs (2 500, 5 000, 10 000), which the grid misses: under
    # "log" a record is taken at the first generation that reaches each, like
    # the grid's own counts; under "budgets" at exactly each.
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
    # No rung whose budget is under this many populations (task 5, run B1:
    # a configuration with fewer than three generations is not run); 0: all.
    ladder_min_gens: int = 0
    # The hypervolumes on the trajectory exactly only up to this many objectives
    # (0 = always). Above it they are recorded as null, or, with
    # trajectory_hv_mc_samples > 0, estimated by Monte Carlo from that many
    # points (the same points at every record, so the curve is smooth).
    trajectory_hv_max_m: int = 0
    trajectory_hv_mc_samples: int = 0
    # The hypervolumes (task 17, supplement 2, §2; the owner's table): of the
    # final population and of the archive scenario's N points exact up to
    # hv_exact_max_m objectives, on the trajectory up to hv_grid_exact_max_m,
    # Monte Carlo from hv_mc_samples points above (metrics.hypervolume; its
    # standard error beside it, hv_se). hv_archive, of the whole archive the
    # run writes (archive_cap x N, or all of it), exactly: at the records up to
    # hv_archive_grid_max_m objectives, at the end up to hv_archive_final_max_m.
    hv_exact_max_m: int = 10
    hv_grid_exact_max_m: int = 8
    hv_archive_grid_max_m: int = 6
    hv_archive_final_max_m: int = 8
    hv_mc_samples: int = 100_000
    # Scenario "archive" (report.py): the final metrics once more, on the run
    # archive reduced to the problem's population size by DSS
    # (meta["final_archive"]). Needs archive = true.
    archive_scenario: bool = True
    # How the archive is reduced to that answer, here and at the checkpoints
    # and hard budgets: "dss" (archive.dss_order), or "igdp_self", the points of
    # the smallest IGD+ against the archive thinned by DSS to 1 000
    # (archive.igdp_self_order; task 6 of 2026-10-09, item 13). The runs
    # record which in meta["archive"]["answer_select"].
    answer_select: str = "dss"
    # What the variation operators did (metric_names.RUN_STATS): per trajectory
    # record and over the run in `final`. Off by default; it never changes a
    # run's result. meta["operators"] lists the operators and their bound
    # repair either way.
    operator_stats: bool = False
    # With coords = pca, pca_nd or randrot: the learned basis's spectrum and
    # its principal angles against the problem's distance block at every
    # record (coords_diag.py; task 7, supplement 1).
    coords_diag: bool = False
    # The run archive (archive.GridArchive -> archive.csv); archive_delta 0 is
    # the default grid step, 1e-3 below five objectives and 1e-2 from five.
    archive: bool = True
    archive_delta: float = 0.0
    # When the archive takes the points: "online", after every batch the core
    # evaluates (a generation, or a point), or "posthoc", only when it is read
    # (the checkpoints, the hard budgets, the end), all the points since the
    # last read at once. The same archive either way, point for point; posthoc
    # saves the calls of a campaign that reads it only at the end.
    archive_mode: str = "online"
    # "auto": a problem of the C++ suite (benchmarks/native.py) is evaluated by
    # the core itself, without calling Python, where nothing of the run's
    # stands between (minimize.direct_path; no noise, typed variables, own
    # evaluator or baseline); the evaluator below then takes its rows a batch
    # at a time. "off": every evaluation through the evaluator, as before task
    # 11. The same files either way.
    native: str = "auto"
    # The cell of the step noise (benchmarks/noisy.py, <base>_s1 and _s5), a
    # share of every variable's range.
    noise_cell: float = 0.01
    # Whose variables the archive keeps: "all" — every point's, in archive.csv;
    # "selected" — only those of the N points DSS selects for the archive
    # scenario, in final_archive.csv.gz, with the objectives of every point in
    # archive.csv.gz. On a large campaign "selected" takes about a tenth of the
    # room of "all" (a quarter of it gzipped).
    archive_variables: str = "all"
    # The archive written: every point (0), or at most archive_cap x N, cut by
    # DSS in the frame the archive scenario selects in (task 16, supplement 1,
    # §2; the atlas: 10). The archive scenario's N points are the first N of
    # that DSS order, so the cut keeps them (with answer_select = "igdp_self"
    # its N points are kept besides); the indicators are not changed by it.
    archive_cap: int = 0
    # IGD+ and the additive epsilon of EVERY point the run evaluated (the
    # feasible ones, noise-free on a noisy copy), in igdp_norm's frame, kept up
    # to date at every evaluation (_core.IncrementalFront; task 10, §5.2): in
    # every trajectory record as igdp_norm_all and eps_norm_all, in `final`,
    # and in meta["all_evaluated"] the evaluation at which igdp_norm_all first
    # reached each of all_levels. Needs a reference front and the extension.
    all_evaluated: bool = True
    all_levels: tuple = (1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001)
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
    # Taken after exactly b evaluations, inside a generation if need be (task
    # 7, §5.5: at 100 or 300 evaluations a generation is most of the budget;
    # until then it was the first record at or past b).
    archive_checkpoints: bool = False
    # Budgets read the hard way (task 5, run B1: at 100-1 000 evaluations a
    # generation is a large share of the budget). The population at b is the
    # last trajectory record at or below b evaluations: a generation that
    # would pass b does not count, where the first record at or after b
    # (record_at) would add up to a whole population. The archive at b is the
    # run archive after exactly b evaluations, reduced by DSS to answer_k
    # points (0: the problem's population size), one answer size for every
    # configuration. Both in meta["hard_at"], the evaluations of the record
    # read beside them; a record every generation needed.
    hard_budgets: tuple = ()
    answer_k: int = 0
    # A population an algorithm does not accept (a Das-Dennis lattice, a
    # multiple of K) is fitted to "below", the largest accepted size up to it,
    # or "nearest", the accepted size nearest to it (fit_pop); the size run
    # is in meta.json, pop_note says why.
    pop_fit: str = "below"
    # Where the runs go (task 16, §1): "files" — a folder of files per run, as
    # before; "store" — a few large files per worker (mootation.store), the
    # atlas's. keep: "standard" leaves the archives at the intermediate points
    # of the budget grid (archive_at.npz) out of the store, "full" keeps them.
    # segment_mb: a worker starts a new segment past this size.
    storage: str = "files"
    keep: str = "standard"
    segment_mb: int = 1024
    # A run's seconds at most (task 16, §3.1; 0: no limit), with the worker
    # pool: past it the runner stops the worker, and the run is kept with the
    # status "timeout" and its trajectory up to the last record it wrote (the
    # last point of the budget grid it reached). A campaign resumed does not
    # run it again; --force does.
    run_timeout: float = 0.0


def _snapshots_wanted(spec: "CampaignSpec", problem: str) -> bool:
    s = spec.snapshots
    return s if isinstance(s, bool) else problem in s


BUDGET_STEPS = (1.0, 1.5, 2.0, 3.0, 5.0, 7.0)     # per decade, from 100 on


def budget_grid(limit: int, extra=()) -> list:
    """The evaluation counts of record_grid = "budgets" up to `limit`: 100,
    150, 200, 300, 500, 700 and the same per decade on, 25 000 besides (the
    budget most campaigns run at), and the `extra` counts (record_at)."""
    out = {25_000} | {int(v) for v in extra}
    k = 2
    while 10 ** k <= limit:
        out |= {int(round(s * 10 ** k)) for s in BUDGET_STEPS}
        k += 1
    return sorted(v for v in out if 0 < v <= limit)


def _started_grid(cfg: Config, spec: "CampaignSpec") -> str | None:
    """The record_grid the results directory was started with, if it was."""
    try:
        path = out_root(cfg, spec) / "campaign.json"
        if path.is_file():
            got = json.loads(path.read_text(encoding="utf-8")).get("spec", {})
            return str(got.get("record_grid") or "generations")
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return None


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
             "trajectory_hv_mc_samples", "hv_exact_max_m", "hv_mc_samples", "hv_grid_exact_max_m",
             "hv_archive_grid_max_m", "hv_archive_final_max_m",
             "archive", "archive_delta", "archive_variables", "archive_scenario", "snapshots",
             "snapshot_variables", "archive_checkpoints", "operator_stats", "hard_budgets",
             "answer_k", "pop_fit", "ladder_min_gens", "dense_reference", "answer_select",
             "noise_cell", "coords_diag", "set_metrics", "set_answer", "all_evaluated", "all_levels",
             "archive_mode", "native", "storage", "keep", "segment_mb", "run_timeout",
             "archive_cap"}
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
    spec.dense_reference = raw.get("dense_reference", True)
    if not isinstance(spec.dense_reference, bool):
        raise ConfigError("campaign.dense_reference", "true or false")
    spec.set_metrics = raw.get("set_metrics", True)
    if not isinstance(spec.set_metrics, bool):
        raise ConfigError("campaign.set_metrics", "true or false")
    spec.set_answer = raw.get("set_answer", "evaluated")
    if spec.set_answer not in ("off", "population", "evaluated"):
        raise ConfigError("campaign.set_answer",
                          f"\"off\", \"population\" or \"evaluated\", not {spec.set_answer!r}")
    spec.seeds = tuple(int(s) for s in raw.get("seeds", ()))
    # the default grid: "budgets"; "generations" with hard budgets (they read
    # every generation); the grid a results directory was started with, so
    # that a resumed campaign does not mix two grids
    default_grid = "generations" if raw.get("hard_budgets") else "budgets"
    if "record_grid" not in raw:
        default_grid = _started_grid(cfg, spec) or default_grid
    spec.record_grid = str(raw.get("record_grid", default_grid))
    spec.record_per_decade = int(raw.get("record_per_decade", 10))
    spec.trajectory_hv_max_m = int(raw.get("trajectory_hv_max_m", 0))
    if spec.record_grid not in ("budgets", "generations", "log"):
        raise ConfigError("campaign.record_grid",
                          f"'budgets', 'generations' or 'log', not '{spec.record_grid}'")
    if spec.record_per_decade < 1:
        raise ConfigError("campaign.record_per_decade", "must be >= 1")
    at = raw.get("record_at", [])
    if (not isinstance(at, (list, tuple))
            or not all(isinstance(v, int) and not isinstance(v, bool) and v > 0 for v in at)):
        raise ConfigError("campaign.record_at", f"a list of positive evaluation counts, not {at!r}")
    if at and spec.record_grid not in ("log", "budgets"):
        raise ConfigError("campaign.record_at", "needs record_grid = \"budgets\" or \"log\"")
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
                    ("hv_mc_samples", 1), ("hv_grid_exact_max_m", 0), ("hv_archive_grid_max_m", 0),
                    ("hv_archive_final_max_m", 0)):
        v = raw.get(key, getattr(spec, key))
        if not isinstance(v, int) or isinstance(v, bool) or v < lo:
            raise ConfigError(f"campaign.{key}", f"an integer >= {lo}, not {v!r}")
        setattr(spec, key, v)
    spec.operator_stats = raw.get("operator_stats", False)
    if not isinstance(spec.operator_stats, bool):
        raise ConfigError("campaign.operator_stats", "true or false")
    spec.coords_diag = raw.get("coords_diag", False)
    if not isinstance(spec.coords_diag, bool):
        raise ConfigError("campaign.coords_diag", "true or false")
    spec.archive_scenario = raw.get("archive_scenario", True)
    if not isinstance(spec.archive_scenario, bool):
        raise ConfigError("campaign.archive_scenario", "true or false")
    spec.archive = raw.get("archive", True)
    if not isinstance(spec.archive, bool):
        raise ConfigError("campaign.archive", "true or false")
    spec.archive_delta = float(raw.get("archive_delta", 0.0))
    spec.archive_mode = raw.get("archive_mode", "online")
    if spec.archive_mode not in ("online", "posthoc"):
        raise ConfigError("campaign.archive_mode",
                          f"\"online\" or \"posthoc\", not {spec.archive_mode!r}")
    spec.native = raw.get("native", "auto")
    if spec.native not in ("auto", "off"):
        raise ConfigError("campaign.native", f"\"auto\" or \"off\", not {spec.native!r}")
    if spec.archive_delta < 0:
        raise ConfigError("campaign.archive_delta", "must be >= 0 (0 = the default step)")
    spec.noise_cell = raw.get("noise_cell", 0.01)
    if (not isinstance(spec.noise_cell, (int, float)) or isinstance(spec.noise_cell, bool)
            or not 0 < spec.noise_cell <= 1):
        raise ConfigError("campaign.noise_cell", f"a share in (0, 1], not {spec.noise_cell!r}")
    spec.archive_cap = raw.get("archive_cap", 0)
    if not isinstance(spec.archive_cap, int) or isinstance(spec.archive_cap, bool) or spec.archive_cap < 0:
        raise ConfigError("campaign.archive_cap", f"an integer >= 0 (x N; 0: the whole archive), "
                                                  f"not {spec.archive_cap!r}")
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
    spec.all_evaluated = raw.get("all_evaluated", True)
    if not isinstance(spec.all_evaluated, bool):
        raise ConfigError("campaign.all_evaluated", "true or false")
    levels = raw.get("all_levels", list(spec.all_levels))
    if (not isinstance(levels, (list, tuple)) or not levels
            or not all(isinstance(v, (int, float)) and not isinstance(v, bool) and v >= 0
                       for v in levels)):
        raise ConfigError("campaign.all_levels", f"a list of levels of igdp_norm >= 0, not {levels!r}")
    spec.all_levels = tuple(float(v) for v in levels)
    if spec.record_every < 1:
        raise ConfigError("campaign.record_every", "must be >= 1")
    hard = raw.get("hard_budgets", [])
    if (not isinstance(hard, (list, tuple))
            or not all(isinstance(v, int) and not isinstance(v, bool) and v > 0 for v in hard)):
        raise ConfigError("campaign.hard_budgets", f"a list of positive evaluation counts, "
                                                   f"not {hard!r}")
    if hard and (spec.record_grid != "generations" or spec.record_every != 1):
        raise ConfigError("campaign.hard_budgets", "reads the last record at or below each "
                                                   "budget: needs a record every generation "
                                                   "(record_grid = \"generations\", "
                                                   "record_every = 1)")
    spec.hard_budgets = tuple(sorted(set(hard)))
    spec.answer_k = raw.get("answer_k", 0)
    if not isinstance(spec.answer_k, int) or isinstance(spec.answer_k, bool) or spec.answer_k < 0:
        raise ConfigError("campaign.answer_k", f"an integer >= 0, not {spec.answer_k!r}")
    spec.answer_select = raw.get("answer_select", "dss")
    if spec.answer_select not in ("dss", "igdp_self"):      # archive.ANSWER_SELECT, without NumPy
        raise ConfigError("campaign.answer_select",
                          f"\"dss\" or \"igdp_self\", not {spec.answer_select!r}")
    spec.pop_fit = raw.get("pop_fit", "below")
    if spec.pop_fit not in ("below", "nearest"):
        raise ConfigError("campaign.pop_fit", f"\"below\" or \"nearest\", not {spec.pop_fit!r}")
    spec.ladder_min_gens = raw.get("ladder_min_gens", 0)
    if (not isinstance(spec.ladder_min_gens, int) or isinstance(spec.ladder_min_gens, bool)
            or spec.ladder_min_gens < 0):
        raise ConfigError("campaign.ladder_min_gens",
                          f"an integer >= 0, not {spec.ladder_min_gens!r}")
    if spec.budget_fe < 0:
        raise ConfigError("campaign.budget_fe", "must be >= 0")
    spec.storage = raw.get("storage", "files")
    if spec.storage not in ("files", "store"):
        raise ConfigError("campaign.storage", f"\"files\" or \"store\", not {spec.storage!r}")
    spec.keep = raw.get("keep", "standard")
    if spec.keep not in ("standard", "full"):
        raise ConfigError("campaign.keep", f"\"standard\" or \"full\", not {spec.keep!r}")
    spec.segment_mb = raw.get("segment_mb", 1024)
    if not isinstance(spec.segment_mb, int) or isinstance(spec.segment_mb, bool) or spec.segment_mb < 1:
        raise ConfigError("campaign.segment_mb", f"an integer >= 1, not {spec.segment_mb!r}")
    spec.run_timeout = raw.get("run_timeout", 0.0)
    if (not isinstance(spec.run_timeout, (int, float)) or isinstance(spec.run_timeout, bool)
            or spec.run_timeout < 0):
        raise ConfigError("campaign.run_timeout", f"seconds >= 0 (0: no limit), not {spec.run_timeout!r}")
    spec.run_timeout = float(spec.run_timeout)
    return spec


def out_root(cfg: Config, spec: CampaignSpec) -> Path:
    base = cfg.source_path.parent if cfg.source_path else Path.cwd()
    p = Path(spec.out.format(name=cfg.name))
    return p if p.is_absolute() else base / p


# ── population-size adjustment ──────────────────────────────────────────────


def _two_layer_sizes(m: int, limit: int) -> set:
    from .algorithms import das_dennis_count
    if m < 2:                     # one objective: two one-point layers
        return {2} if limit >= 2 else set()
    sizes = set()
    h1 = 1
    while das_dennis_count(m, h1) <= limit:
        for h2 in range(1, h1):
            s = das_dennis_count(m, h1) + das_dennis_count(m, h2)
            if s <= limit:
                sizes.add(s)
        h1 += 1
    return sizes


def fit_pop(name: str, pop: int, n_objs: int, params: dict,
            rule: str = "below") -> tuple[int, str]:
    """The population size this algorithm will accept, and a note if it changed.
    rule "below": the largest accepted size up to pop (the smallest above when
    there is none); "nearest": the accepted size nearest to pop, the smaller of
    two as near (CampaignSpec.pop_fit)."""
    # Task 15, §2: given n_ref_points, these take any population size (their
    # reference points no longer come from it), as check_pop says; rounding
    # such a pop to a lattice ran another population than the one asked for.
    if name in FREE_H and params.get("n_ref_points"):
        return pop, ""
    if name in EXACT_LATTICE:
        from .algorithms import lattice_sizes
        ok = set(lattice_sizes(n_objs, 2 * pop)) | _two_layer_sizes(n_objs, 2 * pop)
        if pop in ok:
            return pop, ""
        if rule == "nearest":
            _, above = nearest_lattice_sizes(n_objs, pop)
            near = min(ok | ({above} if above is not None else set()), key=lambda s: (abs(s - pop), s))
            return near, f"pop {pop} -> {near} (nearest lattice size for M={n_objs})"
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
        if rule == "nearest" and pop - new > new + k - pop:
            new += k
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
            if a.problems and pname not in a.problems:      # an entry on a part of them
                continue
            pop = a.pop if a.pop > 0 else p.pop_size
            pop, note = fit_pop(a.name, pop, p.n_obj, a.params, spec.pop_fit)
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
                                evaluations=a.evaluations, init=a.init,
                                init_share=a.init_share))
                idx += 1
    if spec.ladder:
        # The rungs, after every full-budget job so that the job numbers of a
        # campaign without them stay as they were.
        from .budget import depends_on_budget
        full = {(j.problem, j.key): j for j in jobs}
        for pname in problems:
            for a in cfg.algorithms:
                if a.problems and pname not in a.problems:
                    continue
                # a plan taking a share of the budget depends on the budget too
                if (not depends_on_budget(a.name, a.params) and not a.init_share)                         or a.evaluations > 0:
                    continue
                base = full[(pname, a.key)]
                for b in spec.ladder:
                    if b >= base.budget or b < spec.ladder_min_gens * base.pop:
                        continue
                    gens = max(1, math.ceil(b / base.pop))
                    for s in seeds:
                        jobs.append(Job(index=idx, problem=pname, algorithm=a.name, seed=s,
                                        pop=base.pop, gens=gens, params=dict(a.params),
                                        n_objs=base.n_objs, pop_note=base.pop_note,
                                        label=f"{a.key}@{b}", evaluations=b,
                                        ladder_of=a.key, init=a.init,
                                        init_share=a.init_share))
                        idx += 1
    return jobs


def constraint_count(p) -> int:
    """How many constraint values the problem gives: its constraints at the
    point minimize() probed before (the lower bounds of the core's genome,
    decoded), which is not an evaluation of the run."""
    bounds = list(p.bounds)
    n_bin, decode = 0, None
    if p.var_types is not None:
        from ..var_types import Layout
        layout = Layout(p.var_types, bounds)
        if not layout.trivial():
            bounds, n_bin, decode = layout.core_bounds(), len(layout.genes), layout.to_user
    x = [lo for lo, _ in bounds] + [0.0] * n_bin
    return len(list(p.constraints(decode(x) if decode else x)))


# ── the reference data, once per campaign (task 10 of 2026-10-18, §8) ────────
# A problem's frame, reference front, dense reference and Pareto-set sample are
# the same for every run of it, and some take long to build (ZCAT's front and
# set at ten objectives, DAS-CMOP's frame), which every worker process paid
# again for every problem it met. Those of the registry's problems with an
# exact front are kept in the user's cache of references (refs.py; task 12 of
# 2026-10-19, §3), built before a campaign's first run where missing. The
# others (approximate fronts: RE, CRE, RWMOP; a problem without a front) are
# kept in <results>/_references/<problem>.npz the first time and read from
# there after, as long as the stamp matches: the reference version and a hash
# of the benchmark code (mootation/benchmarks/), so that a change of the code
# builds them again. The same arrays, so the same values.
REFERENCES_DIR = "_references"

# The version of the algorithms' trajectories (task 14 of 2026-10-20, §10): a
# run at a seed is the same run only within one version. 1 — before task 14
# (the standard library's distributions, the C library's mathematics: the runs
# differed between platforms); 2 — task 14: the library's own random numbers
# and mathematics, the same runs on every supported platform. meta.json holds
# it (absent: 1); a campaign refuses to put runs of two versions in one folder
# and to merge parts of two versions.
TRAJECTORY_VERSION = 2

# A run's accounting of its evaluations in C++ where the extension has it
# (run_ledger.hpp; task 17, stage 2b); False: in Python (account_numpy), the
# reference the tests compare it with.
USE_LEDGER = True


def trajectory_version_of(meta: dict) -> int:
    return int(meta.get("trajectory_version") or 1)


def check_trajectory_versions(root: Path, jobs: list) -> None:
    """RuntimeError when the folder holds finished runs of these jobs made with
    another trajectory version than this one's: a campaign would mix them."""
    other = []
    for j in jobs:
        mp = root / j.rel_dir / "meta.json"
        if not mp.is_file():
            continue
        try:
            m = json.loads(mp.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if m.get("status") == "done" and trajectory_version_of(m) != TRAJECTORY_VERSION:
            other.append(f"{j.rel_dir.as_posix()} (v{trajectory_version_of(m)})")
    from .. import store
    s = store.store_dir(root)
    keys = {store.job_key(j) for j in jobs} if s.is_dir() else set()
    for raw in (store.packs(s) + store.segments(s)) if keys else []:
        # a segment's or pack's header names the trajectory versions it holds
        versions = store.header_of(raw).get("trajectory_version", "")
        if {int(v) for v in versions.split(",") if v} - {TRAJECTORY_VERSION} and \
                any(k in keys and loc.status == "done" for k, loc in store.segment_entries(raw)):
            other.append(f"store/{raw.name} (v{versions})")
    if other:
        raise RuntimeError(
            f"{root} holds {len(other)} finished run(s) of another trajectory version than "
            f"this library's ({TRAJECTORY_VERSION}): {', '.join(other[:5])}"
            + (" ..." if len(other) > 5 else "")
            + ". Runs of two versions are not comparable seed by seed; use another folder, "
              "or run them again (--force).")
# Workers that start on the same problem at once: the first takes
# <problem>.lock and builds, the others wait for the file it writes (checked
# every REFERENCE_POLL seconds) and read it; a lock older than REFERENCE_WAIT
# seconds (its worker gone) is taken over, and a waiter that sees no file in
# that time builds its own.
REFERENCE_POLL = 0.5
REFERENCE_WAIT = 900.0
_CODE_STAMP: list = []


def _code_stamp() -> str:
    """REFERENCE_VERSION and a hash of every source file of the benchmarks."""
    if not _CODE_STAMP:
        import hashlib
        from ..benchmarks.registry import REFERENCE_VERSION
        h = hashlib.sha256()
        base = Path(__file__).resolve().parent.parent / "benchmarks"
        for f in sorted(base.rglob("*.py")):
            h.update(f.relative_to(base).as_posix().encode())
            h.update(f.read_bytes())
        _CODE_STAMP.append(f"v{REFERENCE_VERSION}-{h.hexdigest()[:16]}")
    return _CODE_STAMP[0]


def reference_hash(p, ref, ref3, pset) -> str:
    """A hash of everything a run is measured against: the frame, the reference
    front, the dense reference and the Pareto-set sample (None where absent), as
    the run reads them (meta.json; --merge checks that the parts agree)."""
    import hashlib
    import numpy as np
    h = hashlib.sha256()
    for name, a in (("ideal", p.ideal), ("nadir", p.nadir), ("ref", ref), ("ref3", ref3),
                    ("pset", pset)):
        h.update(name.encode())
        if a is None:
            h.update(b"none")
        else:
            a = np.ascontiguousarray(a, dtype=float)
            h.update(str(a.shape).encode())
            h.update(a.tobytes())
    return h.hexdigest()[:16]


def reference_data(root: Path, name: str, n_ref: int, front: bool, dense: bool, pset: bool):
    """(problem, ref, (ref3, info3), pareto_set) of a registry problem for a
    run: the problem with its frame read off its front where the registry does
    that, the reference front of n_ref points, the dense reference (or None,
    None) and the Pareto-set sample, each only if asked for and if the problem
    has it — from the cache of references (refs.py) for a problem with an exact
    front, from the campaign's cache (REFERENCES_DIR) for the others."""
    import numpy as np
    from ..benchmarks import registry as R
    from . import refs
    if refs.shared(name):
        return _shared_reference(name, n_ref, front, dense, pset)
    path = root / REFERENCES_DIR / f"{name}.npz"
    lock = path.with_suffix(".lock")
    stamp = _code_stamp()
    want = f"n{int(n_ref)}-f{int(front)}-d{int(dense)}-s{int(pset)}"

    def cached():
        if not path.is_file():
            return None
        try:
            with np.load(path, allow_pickle=False) as z:
                if str(z["stamp"]) == stamp and str(z["want"]) == want:
                    p = R.get_framed(name, z["ideal"], z["nadir"])
                    get = lambda k: z[k] if k in z.files else None       # noqa: E731
                    info = json.loads(str(z["info3"])) if "info3" in z.files else None
                    return p, get("ref"), (get("ref3"), info), get("pset")
        except (OSError, ValueError, KeyError):
            pass                                  # unreadable: build it again
        return None

    got = cached()
    if got is not None:
        return got
    mine = False                                  # this worker holds the lock
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        waited = 0.0
        while not mine:
            try:
                os.close(os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
                mine = True
            except FileExistsError:
                try:
                    stale = time.time() - lock.stat().st_mtime > REFERENCE_WAIT
                except OSError:
                    stale = False                 # just released: look again
                if stale or waited > REFERENCE_WAIT:
                    break                         # build without it
                time.sleep(REFERENCE_POLL)
                waited += REFERENCE_POLL
                got = cached()
                if got is not None:
                    return got
        if mine:
            got = cached()                        # written while this one waited
            if got is not None:
                return got
    except OSError:
        pass                                      # no lock to be had: build
    try:
        return _build_reference(path, name, n_ref, front, dense, pset, stamp, want)
    finally:
        if mine:
            try:
                lock.unlink()
            except OSError:
                pass


def _shared_reference(name: str, n_ref: int, front: bool, dense: bool, pset: bool):
    """reference_data from the cache of references: the entry of `name` at
    n_ref (built here if it is missing or old — a campaign builds them before
    its first run, so this is a run started on its own), and of it what the
    run asks for, as _build_reference gives it."""
    from ..benchmarks import registry as R
    from . import refs
    got = refs.load_entry(name, n_ref)
    if got is None or refs.status(name, got[1]) != "ok":
        refs.build_entry(name, n_ref)
        got = refs.load_entry(name, n_ref)
    a, meta = got
    p = R.get_framed(name, a["ideal"], a["nadir"])
    ref = a["ref"] if front else None
    with3 = dense and ref is not None and "ref3" in a
    return (p, ref, (a["ref3"], meta["info3"]) if with3 else (None, None),
            a.get("pset") if pset else None)


def _build_reference(path: Path, name: str, n_ref: int, front: bool, dense: bool, pset: bool,
                     stamp: str, want: str):
    """reference_data's arrays built, and written to the cache (path)."""
    import numpy as np
    from ..benchmarks import registry as R
    p = R.get(name)
    ref = np.asarray(p.pareto_front(n_ref), float) if front and callable(p.pareto_front) else None
    ref3 = info3 = None
    if dense and ref is not None and R.has_dense_reference(name):
        ref3, info3 = R.dense_reference(name, p.pop_size)
    ps = np.asarray(p.pareto_set(n_ref), float) if pset and callable(p.pareto_set) else None
    payload = dict(stamp=np.asarray(stamp), want=np.asarray(want),
                   ideal=np.asarray(p.ideal, float), nadir=np.asarray(p.nadir, float))
    for k, v in (("ref", ref), ("ref3", ref3), ("pset", ps)):
        if v is not None:
            payload[k] = np.asarray(v, float)
    if info3 is not None:
        payload["info3"] = np.asarray(json.dumps(info3))
    tmp = path.with_name(f"{path.stem}.{os.getpid()}.tmp.npz")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(tmp, **payload)
        os.replace(tmp, path)
    except OSError:                               # another worker's copy will do
        try:
            tmp.unlink()
        except OSError:
            pass
    return p, ref, (ref3, info3), ps


def job_status(root: Path, job: Job) -> str:
    return job_statuses(root, [job])[job.index]


def job_statuses(root: Path, jobs: list) -> dict:
    """{job index: "done", "failed", "running" or "pending"}: the run's record
    in the store (mootation.store) where it has one, else its meta.json."""
    from .. import store
    stored = store.index(store.store_dir(root)) if store.store_dir(root).is_dir() else {}
    out = {}
    for job in jobs:
        loc = stored.get(store.job_key(job))
        if loc is not None:
            out[job.index] = loc.status
            continue
        meta = root / job.rel_dir / "meta.json"
        try:
            with meta.open("r", encoding="utf-8") as fh:
                out[job.index] = str(json.load(fh).get("status", "pending"))
        except (OSError, ValueError):
            out[job.index] = "pending"
    return out


def run_dir_of(root: Path, job: Job):
    """The run's folder, to read: its record in the store where it has one
    (store.StoredRun: `run / "meta.json"` reads as a file does), else
    root / rel_dir."""
    from .. import store
    s = store.store_dir(root)
    loc = store.index(s).get(store.job_key(job)) if s.is_dir() else None
    return store.StoredRun(loc, store.job_key(job)) if loc is not None else root / job.rel_dir


def _record_timeout(root: Path, spec: CampaignSpec, job: Job, pid: int, seconds: float) -> None:
    """A run the runner stopped at run_timeout: its meta.json says "timeout",
    with the evaluations and the values of its last trajectory record (the
    last point of the budget grid it reached); its trajectory stays as
    written. In a store it is packed from the stopped worker's scratch folder
    into the runner's segment."""
    import shutil
    from .. import store
    mine = store.store_dir(root) / store.SCRATCH_DIR / f"p{pid}"
    d = mine / job.rel_dir if spec.storage == "store" else root / job.rel_dir
    d.mkdir(parents=True, exist_ok=True)
    try:
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):                    # stopped before it wrote one
        meta = {"problem": job.problem, "algorithm": job.key, "core": job.algorithm,
                "seed": job.seed, "pop": job.pop, "gens": job.gens, "budget_fe": job.budget,
                "params": job.params, "n_objs": job.n_objs}
    traj = read_trajectory(d)
    meta.update(status="timeout", timeout_seconds=seconds,
                fe=int(traj[-1]["fe"]) if traj else 0, last_record=traj[-1] if traj else None,
                finished=time.strftime("%Y-%m-%dT%H:%M:%S"))
    _write_json(d / "meta.json", meta)
    if spec.storage == "store":
        w = store.writer(store.store_dir(root), spec.segment_mb << 20, header={
            "code_stamp": _code_stamp(), "trajectory_version": TRAJECTORY_VERSION,
            "detmath_path": _detmath_path()})
        w.append(store.job_key(job), [store.format_block()] + store.pack_run_dir(d, keep=spec.keep))
        shutil.rmtree(mine, ignore_errors=True)


def _detmath_path() -> str:
    """The problems' elementary functions: the library's own where the
    extension computes the suite (detmath.hpp), the C library's without it."""
    try:
        from .. import _core  # noqa: F401
        return "detmath"
    except ImportError:
        return "libm"


# ── running one job ─────────────────────────────────────────────────────────


def run_job(job: Job, root: Path, spec: CampaignSpec, *, force: bool = False,
            quiet: bool = False) -> str:
    """Run one job to completion; returns the final status. With [campaign]
    storage = "store" the run's files are written to a scratch folder and
    packed into this process's segment of the store (mootation.store); a run
    stopped before that leaves no record and runs again on resuming."""
    if not force and job_status(root, job) == "done":
        return "done"
    if spec.storage != "store":
        return _run_job(job, root, spec, root / job.rel_dir, quiet=quiet)
    import shutil
    from .. import store
    mine = store.store_dir(root) / store.SCRATCH_DIR / f"p{os.getpid()}"   # one run at a time
    shutil.rmtree(mine, ignore_errors=True)
    d = mine / job.rel_dir
    tables: dict = {}
    status = _run_job(job, root, spec, d, quiet=quiet, tables=tables)
    w = store.writer(store.store_dir(root), spec.segment_mb << 20, header={
        "code_stamp": _code_stamp(), "trajectory_version": TRAJECTORY_VERSION,
        "detmath_path": _detmath_path()})
    blocks = [store.format_block()] + store.pack_run_dir(d, keep=spec.keep)
    for name, (kind, *what) in tables.items():
        if spec.keep == "standard" and name in store.FULL_ONLY:
            continue
        blocks += store.table_blocks(name, *what) if kind == "table" else store.array_blocks(name, *what)
    w.append(store.job_key(job), blocks)
    shutil.rmtree(mine)
    return status


def _run_job(job: Job, root: Path, spec: CampaignSpec, d: Path, *, quiet: bool = False,
             tables: dict | None = None) -> str:
    """run_job's run, its files written to d; with `tables` (a store) its
    point sets go there from memory as float64 instead of CSV files:
    {name: ("table", head, array) or ("arrays", {name: array})}."""
    import numpy as np
    from ..benchmarks import get as bench_get
    from .. import __version__, minimize
    from . import metrics as M
    from .provenance import revision

    meta_path = d / "meta.json"
    d.mkdir(parents=True, exist_ok=True)

    def put_table(name, comments, columns, A, ints=()):
        """A point set of the run: its CSV ("{:.10g}", the files layout), or
        its numbers as float64 for the store (store.table_blocks)."""
        from ..store import _csv_text
        A = np.asarray(A, float).reshape(-1, len(columns)) if len(columns) else np.zeros((0, 0))
        head = {"comments": comments, "columns": columns, "ints": list(ints),
                "newline": "\n" if name.endswith(".gz") else os.linesep}
        if tables is not None:
            tables[name] = ("table", head, A)
            return
        with _open_text(d / name, "w") as out:
            out.write(_csv_text(dict(head, newline="\n"), A))

    def put_arrays(name, payload):
        if tables is not None:
            tables[name] = ("arrays", payload)
        else:
            np.savez_compressed(d / name, **payload)
    traj_path = d / "trajectory.jsonl"
    if traj_path.exists():
        traj_path.unlink()

    t_setup = time.perf_counter()
    from ..benchmarks.registry import PROBLEMS as _registry
    p = _registry[job.problem]               # the problem's own fields; its frame comes below
    final_metrics = spec.final_metrics or spec.metrics
    # IGDX, CR and PSP wherever the problem has a Pareto set ([campaign] set_metrics)
    set_extra = []
    if spec.set_metrics and callable(p.pareto_set):
        set_extra = [m for m in SET_METRICS if m not in spec.metrics]
        final_metrics = (*final_metrics, *(m for m in SET_METRICS if m not in final_metrics))
    # the frame, the reference front, the dense reference ([campaign]
    # dense_reference) and the Pareto-set sample, from the campaign's cache
    # (reference_data)
    p, ref, (ref3, info3), pset = reference_data(
        root, job.problem, spec.n_ref,
        front=any(m in NEEDS_FRONT for m in (*spec.metrics, *final_metrics)),
        dense=spec.dense_reference,
        pset=any(m in NEEDS_SET for m in (*spec.metrics, *final_metrics)))
    ref_hash = reference_hash(p, ref, ref3, pset)
    ideal, nadir = p.ideal, p.nadir
    final_hv = {"exact_max_m": spec.hv_exact_max_m, "mc_samples": spec.hv_mc_samples}

    from .budget import depends_on_budget
    from ..benchmarks.registry import REFERENCE_VERSION, problem_version
    from ..benchmarks.properties import modification_of
    meta = {
        "status": "running", "problem": job.problem, "algorithm": job.key,
        "core": job.algorithm,
        # measured in budget.py: a record of this run at a smaller budget is
        # not what a separate run of that budget gives (ladder_of: the full
        # run a ladder rung shortens); so is a plan taking a share of it
        "budget_dependent": depends_on_budget(job.algorithm, job.params) or bool(job.init_share),
        "ladder_of": job.ladder_of or None,
        "init": job.init, "init_share": job.init_share,
        "seed": job.seed, "pop": job.pop, "gens": job.gens, "budget_fe": job.budget,
        # one run whatever the seed (algorithms.seed_independent; how the atlas
        # counts such runs is the owner's, its protocol's question P13)
        "seed_independent": seed_independent(job.algorithm, job.params, p.n_vars, job.init),
        # the budget asked for: budget_fe rounds it up to whole generations
        # (25 025 at a population of 91), --at and the checkpoints count from it
        "budget_nominal": job.evaluations or spec.budget_fe or job.budget,
        "params": job.params, "n_objs": p.n_obj, "n_vars": p.n_vars,
        "pop_note": job.pop_note, "metrics": [*spec.metrics, *set_extra],
        "final_metrics": list(final_metrics),
        "record_every": spec.record_every, "n_ref": spec.n_ref,
        "has_reference_front": ref is not None,
        # which reference the metrics were measured against (registry.py,
        # REFERENCE_VERSION) and how many rows it had
        "reference_version": REFERENCE_VERSION,
        # the problem's version (registry.py, PROBLEM_VERSION): a run of an
        # older one answered another problem
        "problem_version": problem_version(job.problem),
        # what the problem changes of another — a rotation, a shift, noise
        # (properties.modification_of); None for one of its own. A campaign's
        # repeats are its seeds; a modification is another problem (task 16, §3.4)
        "modification": modification_of(job.problem),
        # what the run was measured against and with (reference_hash), and the
        # benchmark sources' stamp: --merge compares them across the parts
        "reference_hash": ref_hash, "code_stamp": _code_stamp(),
        "trajectory_version": TRAJECTORY_VERSION,
        "reference_rows": None if ref is None else int(len(ref)),
        # the dense reference's rows and step, and the population's step
        "reference_v3": info3,
        "has_pareto_set": callable(p.pareto_set),
        "mootation": __version__,
        # the version never changes between commits; these do (provenance.py)
        "revision": revision(),
        "host": socket.gethostname(), "platform": platform.platform(),
        "python": sys.version.split()[0], "pid": os.getpid(),
        # denormals flushed or read as zero change results (task 14, §5 item 3)
        "fp_env": _fp_env(),
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if pset is not None and not len(pset):
        # the decision-space indicators measure nothing (metrics.compute: None)
        meta["pareto_set_empty"] = ("the Pareto-set sample is empty (a failing copy whose "
                                    "failed region takes all of it): no decision-space "
                                    "indicators")
    _write_json(meta_path, meta)

    from .archive import GridArchive
    from .baselines import BASELINES, run_baseline
    from ..scalar import NAMES as SCALAR_NAMES
    baseline = job.algorithm in BASELINES
    # the scalarization methods (mootation.scalar, _core.scalar): no core
    # state to read at the records
    in_python = baseline or job.algorithm in SCALAR_NAMES
    # Every nondominated point the run evaluates, whatever the algorithm keeps
    # in its population (archive.GridArchive; archive.csv). A baseline's
    # answer IS its archive, so it gets one even when archiving is off.
    arc = (GridArchive(p.n_obj, p.n_vars, ideal=ideal, nadir=nadir,
                       delta=spec.archive_delta or None)
           if (spec.archive or baseline) else None)
    # the second answer's candidates when they are every point evaluated
    sarc = None
    # kept only where the Pareto set has several parts (properties: multimodal_ps)
    from ..benchmarks.properties import properties
    multimodal = bool((properties(job.problem) or {}).get("multimodal_ps"))
    if spec.set_answer == "evaluated" and multimodal and pset is not None and len(pset) and p.n_vars:
        from .archive import SetArchive
        sarc = SetArchive(p.n_obj, p.n_vars)

    # Where the run's time goes (meta["timing_us"], whole microseconds — the
    # store's integer codec keeps them exactly, the owner's word of 2026-10-24,
    # task 17, supplement 5, §2.6; tools/perf reads it):
    # the problem's objectives with its constraints (one computation, the
    # batch's), the run archive, the oracle and everything evaluated, the
    # evaluator around all that, the checkpoints taken inside it, the
    # trajectory records, minimize() as a whole, the final indicators and the
    # files. The core's own share is minimize less the evaluator calls
    # (Result.callback_seconds) and the records. (constraints and feasible:
    # the constraints asked for apart, which no run does any more.)
    tm = dict(evaluate=0.0, constraints=0.0, feasible=0.0, archive=0.0, evaluator=0.0,
              checkpoints=0.0, records=0.0, minimize=0.0, final=0.0, write=0.0, set_answer=0.0,
              setup=time.perf_counter() - t_setup)

    # A constrained problem's indicators are measured on the feasible part of
    # each answer (metrics.compute), which needs its variables; fe_first_feas
    # is the evaluation that first gave a feasible point (task 6, item 5.2).
    cons = p.constraints if p.has_cons else None
    fe_first_feas = None
    # A noisy copy (benchmarks/noisy.py): the algorithm sees the noisy values,
    # every indicator is measured on the noise-free ones at the same x, and the
    # files keep those (task 6, item 7).
    truth = p.true_objectives

    def true_values(F, X):
        if truth is None:
            return F
        return np.array([truth(list(x)) for x in X], float).reshape(len(X), p.n_obj)
    fe = 0
    # a problem whose values depend on the run (uninformative.py) gets its
    # evaluator from the run's seed; every other problem is its `evaluate`
    from ..benchmarks.noisy import kind_of as noise_kind
    if noise_kind(job.problem) == "s":
        objective = p.make_evaluator(job.seed, cell=float(spec.noise_cell))
    else:
        objective = (p.make_evaluator(job.seed) if getattr(p, "make_evaluator", None)
                     else p.evaluate)
    # A noisy copy's oracle (task 7, §4): every non-dominated point by the
    # noise-free values of all the points evaluated, as the run archive keeps
    # the noisy ones.
    oracle = None
    if truth is not None:
        from .archive import GridArchive as _Grid
        oracle = _Grid(p.n_obj, p.n_vars, ideal=ideal, nadir=nadir,
                       delta=spec.archive_delta or None)
    # The hard budgets (CampaignSpec.hard_budgets): the archive after exactly
    # b evaluations is taken by the evaluator, the population's last record at
    # or below b by on_gen.
    hard = [b for b in spec.hard_budgets if b <= job.budget]
    hard_pop, hard_arc = {}, {}

    def take_hard_archive(b):
        t_hd = time.perf_counter()
        AF, AX, _ = arc.points()
        if len(AF) or cons is not None:     # a constrained problem's empty archive: feas_share 0
            idx = arc.select(spec.answer_k or p.pop_size, spec.answer_select)
            hard_arc[str(b)] = dict(measure(true_values(AF[idx], AX[idx]), AX[idx], traj_metrics),
                                    fe=fe, n=int(len(idx)))
        tm["checkpoints"] += time.perf_counter() - t_hd

    # Typed variables (mootation.var_types, task 7, §10): the points the
    # algorithm asked for again after the rounding, dup_x_share in final.
    typed = p.var_types is not None and any(t != "real" for t in p.var_types)
    seen_x, dup_x = set(), 0
    # a categorical variable reaches the objective as its label and the run
    # archive as its index (var_types.Layout.numeric; supplement 3, §1)
    numeric = None
    layout = None
    if typed:
        from ..var_types import Layout
        layout = Layout(p.var_types, p.bounds)
        if "cat" in layout.kinds:
            numeric = layout.numeric

    def baseline_point(x):
        """A baseline's point as the problem reads it: a categorical variable's
        value its label (the nearest index, as categorical.label_of reads a
        number), the rest as drawn."""
        return [layout.values[i][min(max(int(round(v)), 0), len(layout.values[i]) - 1)]
                if k == "cat" else v for i, (k, v) in enumerate(zip(layout.kinds, x))]

    def user_rows(rows):
        """The rows a core or a baseline hands over (Result.variables and
        binary_variables, current_variables(), a baseline's points) as the
        problem's variables in their order, numbers: a categorical variable's
        index. Unchanged without a categorical variable (a problem of bits is
        its genome)."""
        if numeric is None:
            return rows
        if baseline:
            return [numeric(baseline_point(r)) for r in rows]
        return [numeric(layout.to_user(r)) for r in rows]
    # A problem of bits with an exact front (benchmarks/binary.py): the first
    # generation and evaluation at which the population, and the inner
    # archive, hold every front vector (supplement 2, §3.4: the paper's
    # measure), cover_gen and cover_evals in final.
    cover_front = None
    if (p.var_types is not None and all(t == "bin" for t in p.var_types)
            and p.front_exact and callable(p.pareto_front)):
        cover_front = {tuple(float(v) for v in row) for row in np.asarray(p.pareto_front(1000))}
    cover = {"gen": None, "fe": None, "archive_gen": None, "archive_fe": None}
    if cover_front is not None and not baseline:
        from .. import _core                   # inner_archive_state() at every call

    # Failed evaluations (mootation.failures; supplement 3, §2): a non-finite
    # value is counted, kept out of the run archive and the oracle, and goes to
    # the core as it is, which marks it failed; the trajectory records the
    # share of failures since the last record, final n_failed.
    n_failed, failed_at_record = 0, 0
    failed_fe: list = []                 # the counts of the failed evaluations ("budgets")

    # The archive's points the run writes, and whose hypervolume hv_archive is:
    # all, or archive_cap x N cut by DSS in the frame select() uses — their
    # first N the archive scenario's, joined to `sel` (select()'s, where it is
    # not DSS's prefix: igdp_self).
    def archive_kept(AF, sel):
        cap = spec.archive_cap * p.pop_size
        if not cap or len(AF) <= cap:
            return np.arange(len(AF))
        from .archive import dss_order
        lo_c, hi_c = arc.frame() if arc.normalization == "problem" else (None, None)
        return np.union1d(dss_order(AF, k=cap, ideal=lo_c, nadir=hi_c),
                          np.asarray(sel if sel is not None else [], np.intp))

    def archive_hv():
        """hv_archive: the exact hypervolume of archive_kept's points, noise-free."""
        AF, AX, _ = arc.points()
        if not len(AF):
            return 0.0
        sel = arc.select(p.pop_size, spec.answer_select) if spec.answer_select != "dss" else None
        kept = archive_kept(AF, sel)
        T = true_values(AF[kept], AX[kept])
        return M.hypervolume(T, ideal, nadir, exact_max_m=p.n_obj)[0]

    # hv_archive at the records, read at exactly each count of the grid
    arc_hv = (arc is not None and ideal is not None and nadir is not None
              and "hv" in spec.metrics and p.n_obj <= spec.hv_archive_grid_max_m)
    arc_hv_at: dict = {}

    # Everything evaluated (CampaignSpec.all_evaluated): IGD+ and epsilon of all
    # the feasible, finite points so far (noise-free on a noisy copy), and their
    # values at exactly each count of the "budgets" grid.
    inc = None
    if spec.all_evaluated and ref is not None and ideal is not None and nadir is not None:
        from .. import _core as _c
        if hasattr(_c, "IncrementalFront"):
            inc = _c.IncrementalFront(np.ascontiguousarray(ref, float), np.asarray(ideal, float),
                                      np.asarray(nadir, float), list(spec.all_levels))
    all_at: dict = {}
    grid_set: set = set()

    # THE EVALUATOR, A BATCH AT A TIME (task 10, §4.2). The core hands over
    # every batch it evaluates — a generation, or one point — and the rows are
    # computed together: the problem's batch, with its constraints in the same
    # computation where it has them (evaluate_with_constraints; the core gets
    # them from there, the archive's feasibility too); a run's own evaluator
    # (a noisy or an uninformative copy) row by row, its state following the
    # points in order. Then every row in order, as one point at a time before:
    # its count, a failure, its feasibility, the archive, the oracle,
    # everything evaluated, the hard budgets and the checkpoints at exactly
    # their count. The archive and the oracle take their points a batch at a
    # time (add_many; "posthoc": when they are read), which keeps them point
    # for point what adding one at a time gave.
    both = p.evaluate_with_constraints if (cons is not None and objective is p.evaluate) else None
    if objective is p.evaluate:
        rows_of = p.evaluate_batch
    else:
        rows_of = getattr(objective, "batch", None)
    posthoc = spec.archive_mode == "posthoc" and not baseline    # a baseline reads its archive
    pend_a, pend_o, pend_i = [], [], []          # (f, x) for the archive and oracle, (f, fe)
    pend_s: list = []                            # (f, x) for the second answer's archive

    def flush(read: bool = False) -> None:
        """Hand the pending points over: to the archive and the oracle after
        every batch online and when they are read posthoc, to everything
        evaluated always (its values are read at the grid)."""
        t_a = time.perf_counter()
        if pend_a and (read or not posthoc):
            arc.add_many([f for f, _ in pend_a], [x for _, x in pend_a] if p.n_vars else None)
            pend_a.clear()
        if pend_o and (read or not posthoc):
            oracle.add_many([f for f, _ in pend_o], [x for _, x in pend_o] if p.n_vars else None)
            pend_o.clear()
        if pend_i:
            inc.update(np.asarray([f for f, _ in pend_i], float),
                       np.asarray([n for _, n in pend_i], np.int64))
            pend_i.clear()
        if pend_s:
            t_s = time.perf_counter()
            sarc.add_many([f for f, _ in pend_s], [x for _, x in pend_s])
            pend_s.clear()
            tm["set_answer"] += time.perf_counter() - t_s
        tm["archive"] += time.perf_counter() - t_a

    def objectives_of(X):
        """(F, G, T) of the rows X: objectives, constraints (None without),
        noise-free values (None but on a noisy copy)."""
        if both is not None:
            F, G = both(X)
            return F, G, None
        if rows_of is not None:
            F = rows_of(X)
            T = getattr(objective, "last_true_rows", None) if truth is not None else None
        else:
            F, T = [], []
            for x in X:
                F.append(objective(x))
                T.append(getattr(objective, "last_true", None))
            T = None if truth is None or any(t is None for t in T) else T
        G = p.constraints_batch(X) if cons is not None else None
        return F, G, T

    read_at = None          # the counts where account reads something, sorted

    # The accounting in C++ (run_ledger.hpp; task 17, stage 2b) where the
    # archives are the extension's: the same counts, failures and points;
    # account_numpy below is the reference, and the path without it.
    ledger = None
    if USE_LEDGER:
        try:
            from .. import _core as _cl
        except ImportError:
            _cl = None
        compiled = _cl is not None and hasattr(_cl, "RunLedger") and all(
            a is None or isinstance(getattr(a, "_a", None), (_cl.GridArchive, _cl.SetArchive))
            for a in (arc, oracle, sarc))
        if compiled:
            ledger = _cl.RunLedger(p.n_obj, p.n_vars, cons is not None,
                                   arc._a if arc is not None else None,
                                   oracle._a if oracle is not None else None, inc,
                                   sarc._a if sarc is not None else None)

    def reads():
        """The counts where account reads something, made at the first call:
        they are all set before the run."""
        nonlocal read_at
        if read_at is None:
            read_at = sorted(set(checkpoints) | (set(hard) if arc is not None else set())
                             | (grid_set if (inc is not None or arc_hv) else set()))
            if ledger is not None:
                ledger.set_reads(read_at, sorted(grid_set) if inc is not None else [])
        return read_at

    def read(b):
        """What the run reads after exactly b evaluations, the archive's points offered up to b."""
        if arc_hv and b in grid_set:
            flush(read=True)
            t_hv = time.perf_counter()
            arc_hv_at[b] = archive_hv()
            tm["records"] += time.perf_counter() - t_hv
        if (arc is not None and b in hard) or b in checkpoints:
            flush(read=True)
            if arc is not None and b in hard:
                take_hard_archive(b)
            if b in checkpoints and (len(arc) or cons is not None):
                take_checkpoint(b)

    def account(X, F, G, T):
        """The rows X evaluated, (F, G, T) their values (objectives_of), in
        order (THE EVALUATOR above): the counts, the failures and the points of
        a batch at once, which every row is offered in order, and at each count
        where something is read (the grid of everything evaluated, the hard
        budgets, the checkpoints) the reading after exactly that many."""
        nonlocal fe, fe_first_feas, n_failed
        if ledger is None:
            return account_numpy(X, F, G, T)
        reads()
        n = len(X)
        F = np.asarray(F, float).reshape(n, p.n_obj)
        # a categorical variable's label is a string: the archives keep its index
        Xa = (np.asarray(X, float) if numeric is None else
              np.asarray([numeric(x) for x in X], float)).reshape(n, p.n_vars)
        if oracle is not None and T is None:
            finite = np.isfinite(F).all(axis=1)
            T = [truth(list(x)) if finite[i] else [np.nan] * p.n_obj for i, x in enumerate(X)]
        t_a = time.perf_counter()
        ledger.begin(Xa, F, None if G is None else np.asarray(G, float).reshape(n, -1),
                     None if T is None else np.asarray(T, float).reshape(n, p.n_obj))
        while True:
            b = ledger.next()
            if b < 0:
                break
            fe = b
            tm["archive"] += time.perf_counter() - t_a
            read(b)
            t_a = time.perf_counter()
        tm["archive"] += time.perf_counter() - t_a
        fe, n_failed, fe_first_feas = ledger.fe, ledger.n_failed, ledger.first_feasible
        if len(failed_fe) != n_failed:
            failed_fe[:] = ledger.failed_fe

    def account_numpy(X, F, G, T):
        """account() in Python: the reference."""
        nonlocal fe, fe_first_feas, n_failed
        n = len(X)
        F = np.asarray(F, float).reshape(n, p.n_obj)
        finite = np.isfinite(F).all(axis=1)
        ok = finite if G is None else \
            finite & (np.asarray(G, float).reshape(n, -1) <= 0.0).all(axis=1)
        start = fe
        failed = np.flatnonzero(~finite)
        if len(failed):
            n_failed += len(failed)
            failed_fe.extend((start + 1 + failed).tolist())
        if cons is not None and fe_first_feas is None and ok.any():
            fe_first_feas = start + 1 + int(np.argmax(ok))

        def pend(lo, hi):
            """rows lo..hi-1 to the archive, the oracle and everything evaluated"""
            for i in (lo + np.flatnonzero(finite[lo:hi])).tolist():
                x = X[i]
                xn = numeric(x) if numeric is not None else x
                if arc is not None and ok[i]:
                    pend_a.append((F[i], xn))
                if sarc is not None and ok[i]:
                    pend_s.append((F[i], xn))
                t_true = None
                if oracle is not None:
                    t_true = T[i] if T is not None else truth(list(x))
                    pend_o.append((t_true, xn))
                if inc is not None and ok[i]:
                    pend_i.append((F[i] if t_true is None else t_true, start + 1 + i))
        at = reads()
        lo = 0
        for b in at[bisect.bisect_right(at, start):bisect.bisect_right(at, start + n)]:
            pend(lo, b - start)
            lo = b - start
            fe = b
            if inc is not None and fe in grid_set:
                flush()
                all_at[fe] = (inc.igd_plus(), inc.eps_plus())
            read(b)
        pend(lo, n)
        fe = start + n
        flush()

    def evaluate_rows(X):
        nonlocal dup_x
        t_in = time.perf_counter()
        X = [list(x) for x in X]
        if typed:
            for x in X:
                key = tuple(x)
                dup_x += key in seen_x
                seen_x.add(key)
        F, G, T = objectives_of(X)
        F = np.asarray(F, float).reshape(len(X), p.n_obj)
        tm["evaluate"] += time.perf_counter() - t_in
        account(X, F, G, T)
        tm["evaluator"] += time.perf_counter() - t_in
        rows = F.tolist()
        if cons is not None:
            return rows, [list(map(float, g)) for g in G]
        return rows

    def evaluate(x):
        """One point (a baseline's, or a core's single evaluation)."""
        if baseline and numeric is not None:
            x = baseline_point(x)
        got = evaluate_rows([x])
        return got[0][0] if cons is not None else got[0]

    # THE DIRECT PATH (task 11 of 2026-10-18, §1.3): a problem of the C++ suite
    # the core evaluates itself (minimize.direct_path), where the run has no
    # evaluator of its own; its rows come to account a batch at a time, before
    # every observer call and at the end, so that the counts, the archive, the
    # hard budgets and the checkpoints are what the evaluator above gives.
    from ..minimize import direct_path
    direct = None
    if not baseline and objective is p.evaluate and truth is None:
        direct, _ = direct_path(objective, native=spec.native, algorithm=job.algorithm,
                                **job.params)

    def on_rows(X, F, G):
        nonlocal dup_x
        t_in = time.perf_counter()
        if typed:                   # the decoded rows, as evaluate_rows counts them
            for x in X.tolist():
                key = tuple(x)
                dup_x += key in seen_x
                seen_x.add(key)
        account(X, F, G if cons is not None else None, None)
        tm["evaluator"] += time.perf_counter() - t_in

    def take_checkpoint(b):
        """The archive scenario after exactly b evaluations (archive_checkpoints)."""
        nonlocal record_seconds
        t_ck = time.perf_counter()
        AF, AX, _ = arc.points()
        idx = arc.select(p.pop_size, spec.answer_select)
        T = true_values(AF[idx], AX[idx])
        archive_at[str(b)] = dict(measure(T, AX[idx], traj_metrics), fe=b, n=int(len(idx)))
        ck_budget.append(b); ck_fe.append(b); ck_n.append(len(idx))
        ck_F.append(T.astype(np.float32))
        if keep_x:
            ck_X.append(AX[idx].astype(np.float32))
        record_seconds += time.perf_counter() - t_ck
        tm["checkpoints"] += time.perf_counter() - t_ck

    snap = _snapshots_wanted(spec, job.problem)
    # the variables go into the snapshots and checkpoints where snapshot_variables
    # says so, and always on a constrained problem, whose indicators need them
    keep_x = (spec.snapshot_variables == "all" or cons is not None
              or (spec.snapshot_variables == "pareto_set" and callable(p.pareto_set)))
    op_stats = spec.operator_stats and not in_python
    if op_stats:
        from .. import _core
    diag = spec.coords_diag and not in_python
    if diag:
        from .. import _core
        from . import coords_diag
        diag_start = coords_diag.distance_start(job.problem)
    snap_fe, snap_gen, snap_n, snap_F, snap_X = [], [], [], [], []

    t0 = time.perf_counter()
    fh = traj_path.open("a", encoding="utf-8")
    n_records = 0
    budget = job.budget

    # Above trajectory_hv_max_m objectives the hypervolumes stay off the
    # trajectory (recorded as null, so the absence is explicit), or are
    # estimated by Monte Carlo when trajectory_hv_mc_samples asks for it.
    traj_metrics = [*spec.metrics, *set_extra]
    hv_skipped = []
    traj_hv = {"exact_max_m": spec.hv_grid_exact_max_m, "mc_samples": spec.hv_mc_samples}
    if spec.trajectory_hv_max_m and p.n_obj > spec.trajectory_hv_max_m:
        if spec.trajectory_hv_mc_samples > 0:
            traj_hv = {"exact_max_m": 0, "mc_samples": spec.trajectory_hv_mc_samples}
        else:
            hv_skipped = [m for m in traj_metrics if m in HV_NAMES]
            traj_metrics = [m for m in traj_metrics if m not in HV_NAMES]
    if spec.record_grid == "log":
        grid = sorted(set(log_grid(budget, spec.record_per_decade))
                      | {v for v in spec.record_at if v <= budget})
    elif spec.record_grid == "budgets":
        grid = budget_grid(budget, spec.record_at)
        grid_set.update(grid)
    else:
        grid = None
    next_i = 0
    meta.update(record_grid=spec.record_grid,
                record_per_decade=spec.record_per_decade if spec.record_grid == "log" else None,
                record_at=list(spec.record_at),
                trajectory_hv_max_m=spec.trajectory_hv_max_m,
                trajectory_hv=("not recorded" if hv_skipped else
                               "exact" if traj_hv["exact_max_m"] >= p.n_obj else
                               f"monte-carlo, {traj_hv['mc_samples']} samples"),
                # the indicators' version (metrics.INDICATOR_VERSION)
                indicator_version=M.INDICATOR_VERSION,
                # the Monte-Carlo estimate's version, where one may be taken
                hv_mc_version=(M.HV_MC_VERSION if min(traj_hv["exact_max_m"], final_hv["exact_max_m"])
                               < p.n_obj else None),
                trajectory_metrics=traj_metrics, trajectory_hv_skipped=hv_skipped,
                trajectory_metrics_at_record_at=[m for m in traj_metrics
                                                 if m in spec.record_at_metrics])
    # The archive scenario at the record_at counts below the budget, nominal
    # and actual (its end is final_archive; budget_fe rounded up to whole
    # generations ends inside the last one): taken after exactly each count of
    # evaluations (take_checkpoint).
    checkpoints = ([b for b in spec.record_at if b < min(budget, meta["budget_nominal"])]
                   if spec.archive_checkpoints and arc is not None and grid is not None else [])
    archive_at = {}
    ck_budget, ck_fe, ck_n, ck_F, ck_X = [], [], [], [], []
    last_fe = 0                    # the evaluations of the previous record
    record_seconds = 0.0           # what the records cost: indicators, statistics, snapshots
    need_x = ((snap and keep_x) or any(m in NEEDS_X for m in traj_metrics)
              or cons is not None or truth is not None)
    if need_x and not baseline:
        from .. import _core                   # current_variables() at the records

    def dense(F, X, which) -> dict:
        """The front indicators among `which` against the dense reference."""
        front = [m for m in which if m in NEEDS_FRONT]
        if ref3 is None or not front or not len(F):
            return {}
        got = M.compute(F, ref_front=ref3, ideal=ideal, nadir=nadir, which=front, X=X,
                        constraints=cons)
        return {m + DENSE_SUFFIX: got[m] for m in front}

    def measure(F, X, which):
        vals = M.compute(F, ref_front=ref, ideal=ideal, nadir=nadir, which=which,
                         pop=p.pop_size, hv_options=traj_hv, X=X, pareto_set=pset,
                         bounds=p.bounds, cyclic=p.cyclic_vars, distance_fn=p.distance_fn,
                         constraints=cons)
        vals.update(dense(F, X, which))
        if truth is not None and len(F):
            vals["true_dominated_share"] = 1.0 - M.nd_share(F)
        for m in hv_skipped:
            vals[m] = None
        return vals

    # the costly indicators only at the record_at counts and at the end
    everywhere = [m for m in traj_metrics if m not in spec.record_at_metrics]

    held = None     # "budgets": the last call's (gen, fe, rows, variables), not recorded yet

    def variables_of(objectives, variables):
        """The variables of the rows on_generation hands over, when the record needs them."""
        if need_x and len(objectives) and variables is None:
            variables = _core.current_variables()         # a baseline hands them over
        if variables is not None and len(variables):
            variables = user_rows(variables)
        return variables

    def on_gen(gen, objectives, variables=None):
        nonlocal next_i, held
        if cover_front is not None:            # at every call, before the grid's choice
            if cover["gen"] is None and cover_front <= {tuple(map(float, r)) for r in objectives}:
                cover["gen"], cover["fe"] = gen, fe
            if cover["archive_gen"] is None and not in_python:
                A = {tuple(map(float, r)) for r in _core.inner_archive_state()}
                if A and cover_front <= A:
                    cover["archive_gen"], cover["archive_fe"] = gen, fe
        if spec.record_grid == "budgets":
            # Every count b of the grid passed since the previous call gets the
            # population the run had at b: this call's when it is at exactly b,
            # else the previous call's, the last that did not pass b (none
            # before the first population is complete). The end of the run is
            # recorded after minimize() returns (held).
            variables = variables_of(objectives, variables)
            while next_i < len(grid) and grid[next_i] <= fe:
                b = grid[next_i]
                next_i += 1
                if b == fe:
                    record(gen, b, objectives, variables)
                elif held is not None:
                    record(held[0], b, held[2], held[3], fe_pop=held[1])
            held = (gen, fe, objectives, variables)
            return
        if grid is not None:
            # Generation 0 and the last call are always recorded; in between,
            # the first call at or past the next count of the grid.
            if gen > 0 and fe < budget and (next_i >= len(grid) or fe < grid[next_i]):
                return
            while next_i < len(grid) and grid[next_i] <= fe:
                next_i += 1
        record(gen, fe, objectives, variables_of(objectives, variables))

    def record(gen, fe_rec, objectives, variables, fe_pop=None):
        """One trajectory record: the population (objective rows, variables)
        at fe_rec evaluations; fe_pop, the evaluations it cost when that is
        fewer (the "budgets" grid)."""
        nonlocal n_records, last_fe, record_seconds, failed_at_record
        t_rec = time.perf_counter()
        F = np.asarray(objectives, float)
        X = None
        if need_x and F.size:
            X = np.asarray(variables, float).reshape(len(F), p.n_vars)
        if truth is not None and X is not None:
            F = true_values(F, X)                 # snapshot and indicators: noise-free
        rec = {"gen": gen, "fe": fe_rec, "t": round(t_rec - t0, 3), "n": int(len(F))}
        if fe_pop is not None:
            rec["fe_pop"] = fe_pop
        if spec.record_grid == "budgets":
            if p.failing is not None or failed_fe:
                k = sum(1 for v in failed_fe if last_fe < v <= fe_rec)
                rec["fail_share"] = k / max(fe_rec - last_fe, 1)
        elif p.failing is not None or n_failed:
            rec["fail_share"] = (n_failed - failed_at_record) / max(fe_rec - last_fe, 1)
            failed_at_record = n_failed
        finite = bool(np.all(np.isfinite(F))) if F.size else True
        rec["finite"] = finite
        mark = fe_rec >= budget or any(last_fe < b <= fe_rec for b in spec.record_at)
        if finite and F.size:
            rec.update(measure(F, X, traj_metrics if mark else everywhere))
        if arc_hv and finite and F.size:
            if fe_rec not in arc_hv_at:
                flush(read=True)
                arc_hv_at[fe_rec] = archive_hv()
            rec["hv_archive"] = arc_hv_at[fe_rec]
        if inc is not None:
            v = ((ledger.all_at(fe_rec) if ledger is not None else all_at.get(fe_rec))
                 if fe_rec != fe else None)
            v = v or (inc.igd_plus(), inc.eps_plus())
            rec["igdp_norm_all"], rec["eps_norm_all"] = (None if not math.isfinite(u) else u
                                                         for u in v)
        if op_stats:
            rec.update(_core.operator_stats())
        if diag:
            rec.update(coords_diag.record(job.problem, _core.coords_state(), diag_start))
        if snap:
            snap_fe.append(fe_rec); snap_gen.append(gen); snap_n.append(len(F))
            snap_F.append(F.astype(np.float32).reshape(len(F), p.n_obj))
            if keep_x:
                snap_X.append(np.zeros((0, p.n_vars), np.float32) if X is None
                              else X.astype(np.float32))
        for b in hard:                                   # the last record at or below b
            if fe_rec <= b:
                hard_pop[str(b)] = {k: rec.get(k) for k in ("gen", "fe", "n", *traj_metrics)}
        last_fe = fe_rec
        fh.write(json.dumps(rec, separators=(",", ":")) + "\n")
        fh.flush()
        n_records += 1
        record_seconds += time.perf_counter() - t_rec
        tm["records"] += time.perf_counter() - t_rec

    t_min = time.perf_counter()
    try:
        if baseline:
            res = run_baseline(job.algorithm, evaluate, p.bounds, pop=job.pop,
                               max_evaluations=budget, seed=job.seed, archive=arc,
                               on_generation=on_gen,
                               record_every=(1 if grid is not None else spec.record_every))
        else:
            # every batch the core evaluates, with the constraints in the same
            # call (THE EVALUATOR above); or the problem itself (THE DIRECT PATH)
            how = (dict(native="auto", on_rows=on_rows) if direct is not None else
                   dict(batch=evaluate_rows, batch_with_constraints=p.has_cons,
                        n_cons=constraint_count(p) if p.has_cons else 0))
            res = minimize(
                objective if direct is not None else evaluate, bounds=p.bounds,
                n_objs=p.n_obj, algorithm=job.algorithm, pop_size=job.pop, n_gen=job.gens,
                seed=job.seed, **how,
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
                init=job.init, init_share=job.init_share,
                var_types=p.var_types,
                stochastic=p.stochastic,
                **job.params,
            )
    except Exception as e:                       # one bad job must not kill the shard
        fh.close()
        meta.update(status="failed", error=f"{type(e).__name__}: {e}",
                    traceback=traceback.format_exc()[-4000:], fe=fe,
                    finished=time.strftime("%Y-%m-%dT%H:%M:%S"),
                    seconds=round(time.perf_counter() - t0, 3))
        # the pair cannot run at all (a core without the integer genome on
        # binary variables, BC's n + 1 parents above pop_size): not the
        # algorithm failing; the reports keep these apart (task 16, §2.5)
        if "incompatible by construction" in str(e):
            meta["incompatible"] = True
        _write_json(meta_path, meta)
        if not quiet:
            print(f"[{job.index}] {job.problem} {job.key} seed {job.seed}: FAILED {e}",
                  file=sys.stderr)
        return "failed"
    if spec.record_grid == "budgets" and held is not None and held[1] > last_fe:
        record(held[0], held[1], held[2], held[3])
    flush(read=True)                   # posthoc: the archive takes its points now
    fh.close()
    tm["minimize"] = time.perf_counter() - t_min
    t_out = time.perf_counter()

    F = np.asarray(res.objectives, float)
    genes = getattr(res, "binary_variables", None) if numeric is not None and not baseline else None
    rows = ([list(x) + list(b) for x, b in zip(res.variables, genes)] if genes
            else res.variables)
    X = np.asarray(user_rows(rows) if len(rows) else rows, float)
    seen = None
    if truth is not None and len(F):
        seen, F = F, true_values(F, X)       # f: noise-free; f_seen: what the algorithm saw
    columns = ([f"f{i+1}" for i in range(F.shape[1] if F.ndim == 2 else 0)]
               + [f"x{i+1}" for i in range(X.shape[1] if X.ndim == 2 else 0)]
               + ([f"f{i+1}_seen" for i in range(p.n_obj)] if seen is not None else []))
    parts = [F, X] + ([seen] if seen is not None else [])
    put_table("final.csv", ["# mootation campaign final population v1"
                            + ("; f noise-free, f_seen as the algorithm saw it" if seen is not None
                               else "")],
              columns, np.column_stack(parts) if columns else np.zeros((0, 0)))
    cv = getattr(res, "cv", None)
    if cons is not None and cv is not None and len(cv) == len(F):
        # the final population's constraint violation, row by row with final.csv
        # (task 16, supplement 1)
        put_table("final_cv.csv", ["# mootation campaign final population's constraint violation "
                                   "v1: the sum of the violations, the rows of final.csv"],
                  ["cv"], np.asarray(cv, float).reshape(-1, 1))

    arc_sel = None                  # the archive scenario's points: indices into arc.points()
    if arc is not None:
        AF, AX, AE = arc.points()
        whole = spec.archive_variables == "all"
        if (len(AF) or cons is not None) and (spec.archive_scenario or not whole):
            t_sel = time.perf_counter()
            arc_sel = arc.select(p.pop_size, spec.answer_select)
            tm["final"] += time.perf_counter() - t_sel
        # the points written: all, or archive_cap x N cut by DSS (archive_kept)
        cap = spec.archive_cap * p.pop_size
        kept = archive_kept(AF, arc_sel)
        # a noisy copy's archive is written noise-free, and beside it the
        # values the archive scenario selected by (f*_seen, task 16,
        # supplement 1), so that its N points are found again from the file
        seen_af = AF if truth is not None else None
        AF = true_values(AF, AX)
        # "all": every point with its variables; "selected": the objectives of
        # every point, and the variables only of the points DSS selects
        name = "archive.csv" if whole else "archive.csv.gz"
        put_table(name, [f"# mootation campaign archive v1: every nondominated point evaluated"
                         f"{' (feasible only)' if p.has_cons else ''}, at most one per cell of a "
                         f"{arc.delta:g} grid in {arc.normalization}-normalized objectives, "
                         f"each objective's best point kept whatever its cell holds"
                         + ("" if whole else "; objectives only, the variables of the points "
                            "the archive scenario selects are in final_archive.csv.gz")
                         + (f"; cut to {cap} points by DSS" if len(kept) < len(AF) else "")
                         + ("; f noise-free, f_seen as the algorithm saw it" if seen_af is not None
                            else "")],
                  [f"f{i+1}" for i in range(p.n_obj)] + [f"x{i+1}" for i in range(p.n_vars) if whole]
                  + ["extreme"] + ([f"f{i+1}_seen" for i in range(p.n_obj)] if seen_af is not None else []),
                  np.column_stack([AF[kept]] + ([AX[kept]] if whole else []) + [AE[kept]]
                                  + ([seen_af[kept]] if seen_af is not None else [])),
                  ints=[p.n_obj + (p.n_vars if whole else 0)])
        if not whole and arc_sel is not None:
            how = "DSS" if spec.answer_select == "dss" else spec.answer_select
            put_table("final_archive.csv.gz",
                      [f"# mootation campaign archive selection v1: the points {how} selects "
                       f"from the archive for the archive scenario"],
                      [f"f{i+1}" for i in range(p.n_obj)] + [f"x{i+1}" for i in range(p.n_vars)],
                      np.column_stack([AF[arc_sel], AX[arc_sel]]))
        lo, hi = arc.frame()
        # The frame DSS normalises by when it reduces the archive (scenario
        # "archive"), so that --recompute --scenario archive selects the same points.
        meta["archive"] = dict(arc.info(), extremes=int(AE.sum()), file=name,
                               written=int(len(kept)), cap=cap or None,
                               answer_select=spec.answer_select,
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
        put_arrays("snapshots.npz", payload)
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
            put_arrays("archive_at.npz", payload)
    if hard:
        for b in hard:
            if str(b) not in hard_arc and arc is not None and (len(arc) or cons is not None):
                take_hard_archive(b)         # the run stopped before b: its archive at the end
        meta["hard_at"] = {str(b): {"population": hard_pop.get(str(b)),
                                    "archive": hard_arc.get(str(b))} for b in hard}
        meta["answer_k"] = spec.answer_k or p.pop_size

    extra = {"pop": p.pop_size, "hv_options": final_hv, "pareto_set": pset,
             "bounds": p.bounds, "cyclic": p.cyclic_vars, "distance_fn": p.distance_fn,
             "constraints": cons}
    final = {}
    t_fin = time.perf_counter()
    if F.size and np.all(np.isfinite(F)):
        final = M.compute(F, ref_front=ref, ideal=ideal, nadir=nadir, which=final_metrics,
                          X=X, **extra)
        final.update(dense(F, X, final_metrics))
    if inc is not None and final:
        final["igdp_norm_all"] = inc.igd_plus() if inc.seen else None
        final["eps_norm_all"] = inc.eps_plus() if inc.seen else None
    if inc is not None:
        meta["all_evaluated"] = {
            "points": int(inc.seen), "levels": list(spec.all_levels),
            "fe_to_level": [None if v < 0 else int(v) for v in inc.reached],
            "measured": "IGD+ and the additive epsilon of every feasible, finite point "
                        "evaluated (noise-free on a noisy copy) against the reference, in "
                        "igdp_norm's frame; fe_to_level: the evaluation at which igdp_norm_all "
                        "first reached each level"}
    if cons is not None:
        final["fe_first_feas"] = fe_first_feas
    if p.failing is not None or n_failed:
        final["n_failed"] = n_failed
    if typed and final:
        final["dup_x_share"] = dup_x / max(fe, 1)
    if cover_front is not None:
        # not covered within the budget: NaN, and cover_reached false
        nan = float("nan")
        final.update(cover_gen=nan if cover["gen"] is None else cover["gen"],
                     cover_evals=nan if cover["fe"] is None else cover["fe"],
                     cover_reached=cover["gen"] is not None,
                     cover_archive_gen=(nan if cover["archive_gen"] is None
                                        else cover["archive_gen"]),
                     cover_archive_evals=(nan if cover["archive_fe"] is None
                                          else cover["archive_fe"]))
    if truth is not None:
        if final:
            final["true_dominated_share"] = 1.0 - M.nd_share(F)
            # the observed answer: its noisy values, as the algorithm saw them
            obs = M.compute(seen, ref_front=ref, ideal=ideal, nadir=nadir,
                            which=final_metrics, X=X, **extra)
            final.update({m + "_observed": v for m, v in obs.items()})
        OF, OX, _ = oracle.points()
        if len(OF):
            # the oracle: the noise-free non-dominated set of every evaluation,
            # reduced to the answer's size by the archive scenario's rule
            sel = oracle.select(p.pop_size, spec.answer_select)
            orc = M.compute(OF[sel], ref_front=ref, ideal=ideal, nadir=nadir,
                            which=final_metrics, X=OX[sel], **extra)
            final.update({m + "_oracle": v for m, v in orc.items()})
        kind = noise_kind(job.problem)
        meta["noise"] = {"kind": kind, "scale": list(objective.scale),
                         "cell": float(spec.noise_cell) if kind == "s" else None,
                         "measured": "every indicator on the noise-free objectives of the "
                                     "answer; _oracle on the noise-free non-dominated set "
                                     "of every evaluation; _observed on the noisy values "
                                     "of the answer",
                         "oracle_n": int(len(OF))}
    if arc_sel is not None and spec.archive_scenario:
        AF, AX, _ = arc.points()
        T = true_values(AF[arc_sel], AX[arc_sel])
        meta["final_archive"] = M.compute(T, ref_front=ref, ideal=ideal, nadir=nadir,
                                          which=final_metrics, X=AX[arc_sel], **extra)
        meta["final_archive"].update(dense(T, AX[arc_sel], final_metrics))
        meta["final_archive"]["n"] = int(len(arc_sel))
        if truth is not None and len(T):
            meta["final_archive"]["true_dominated_share"] = 1.0 - M.nd_share(T)
        if ("hv" in final_metrics and ideal is not None and nadir is not None
                and p.n_obj <= spec.hv_archive_final_max_m):
            # the whole archive written, exactly (CampaignSpec.hv_archive_final_max_m)
            meta["final_archive"]["hv_archive"] = archive_hv()
    if spec.set_answer != "off" and multimodal and pset is not None and len(pset) and p.n_vars:
        # the second answer (CampaignSpec.set_answer): its decision-space indicators
        from .archive import SET_ANSWER_VERSION, SetArchive, spread_select
        t_s = time.perf_counter()
        if sarc is None:                    # "population": the final population's non-dominated
            sarc = SetArchive(p.n_obj, p.n_vars)
            keep = np.all(np.isfinite(F), axis=1) if F.size else np.zeros(0, bool)
            if cons is not None and getattr(res, "cv", None) is not None and len(res.cv) == len(F):
                keep &= np.asarray(res.cv, float) <= 0.0
            if keep.any():
                sarc.add_many(F[keep], X[keep])
        if len(sarc):
            sel = spread_select(sarc.X, p.pop_size, p.bounds, job.seed)
            SF = true_values(sarc.F[sel], sarc.X[sel])
            got = M.compute(SF, ref_front=ref, ideal=ideal,
                            nadir=nadir, which=SET_METRICS, X=sarc.X[sel], **extra)
            # its points (task 16, supplement 1): objectives noise-free, then variables
            put_table("set_answer.csv", ["# mootation campaign second answer v1: the points "
                                         "spread_select takes for the decision-space indicators"],
                      [f"f{i+1}" for i in range(p.n_obj)] + [f"x{i+1}" for i in range(p.n_vars)],
                      np.column_stack([SF, sarc.X[sel]]))
            meta["set_answer"] = {"candidates": spec.set_answer, "n_candidates": len(sarc),
                                  "n": int(len(sel)), "version": SET_ANSWER_VERSION,
                                  **{m: got.get(m) for m in SET_METRICS}}
        else:
            meta["set_answer"] = {"candidates": spec.set_answer, "n_candidates": 0, "n": 0,
                                  "version": SET_ANSWER_VERSION}
        tm["set_answer"] += time.perf_counter() - t_s
    tm["final"] += time.perf_counter() - t_fin
    if not baseline:
        meta["operators"] = [{"operator": o, "bound_repair": r}
                             for o, r in getattr(res, "operators", [])]
        # a core with presets (smpso): every knob as the run used it
        if getattr(res, "settings", None):
            meta["settings"] = dict(res.settings)
            # SMS-EMOA / MO-CMA-ES: steps whose race for the least contributor
            # stopped at race_max_samples, which its guarantee does not cover
            capped = float(meta["settings"].get("race_capped_share", 0.0) or 0.0)
            if capped > RACE_CAPPED_WARN:
                print(f"warning: {job.problem} {job.key} seed {job.seed}: the race stopped at "
                      f"race_max_samples in {capped:.1%} of its steps", file=sys.stderr)
    if op_stats and final:
        final.update({k: float(v) for k, v in dict(res.operator_totals).items()})
    meta["operator_stats"] = op_stats
    if hasattr(objective, "summary"):
        # the uninformative problems count where every evaluated point and
        # every non-dominated one fell (benchmarks/uninformative.py; --bias)
        meta["structural_bias"] = objective.summary()
    # the files and the rest after the run: what is not the final indicators
    tm["write"] = time.perf_counter() - t_out - tm["final"]
    # the direct path: the evaluations' time is C++'s inside the core
    native_s = float(getattr(res, "native_seconds", 0.0))
    if direct is not None:
        tm["evaluate"] = native_s
        tm["evaluator"] += native_s
    meta["timing_us"] = dict({k: int(round(1e6 * v)) for k, v in tm.items()},
                             callbacks=int(round(1e6 * float(getattr(res, "callback_seconds", 0.0)))),
                             calls=int(getattr(res, "callbacks", 0)), native=int(round(1e6 * native_s)))
    meta.update(status="done", fe=fe, records=n_records, final=final,
                ignored_knobs=list(res.ignored), active_n=int(res.active_n),
                finished=time.strftime("%Y-%m-%dT%H:%M:%S"),
                seconds=round(time.perf_counter() - t0, 3),
                # of `seconds`, what the trajectory records took (indicators,
                # operator statistics, snapshots, archive checkpoints)
                record_seconds=round(record_seconds, 3))
    if getattr(res, "finished", False) and fe < budget:
        # the core stopped by its own rule before the budget (DMS, DMulti-MADS:
        # the step below its tolerance; task 16, §2): a done run, marked
        meta["core_stopped"] = {"evaluations": int(fe), "budget": int(budget)}
    if getattr(res, "unique_evaluations", None) is not None:
        # the distinct points among the evaluations: the budget counts calls,
        # a point asked for again included (task 16, §3.2)
        meta["unique_fe"] = int(res.unique_evaluations)
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


def prepare_references(todo: list, n_ref: int, *, workers: int = 1, build: bool = True) -> None:
    """The entries of the cache of references (refs.py) of every problem of
    `todo` with an exact front, current before the first run: the missing and
    old ones built here, `workers` at a time; build=False (--no-build-refs)
    raises refs.MissingReferences listing them instead."""
    from . import refs
    need = sorted({j.problem for j in todo if refs.shared(j.problem)})
    if not need:
        return
    st = refs.check(need, n_ref)
    missing = [n for n in need if st[n] != "ok"]
    if not missing:
        return
    if not build:
        raise refs.MissingReferences(missing, n_ref, st)
    refs.build(missing, n_ref, workers=max(1, workers),
               log=lambda line: print(line, file=sys.stderr))


def single_threaded_blas() -> None:
    """One BLAS thread in every worker process started after this call.

    The pool is the parallelism. Left alone, NumPy's OpenBLAS reserves a buffer
    for every core in every process at import — 756 MB each on a 24-thread
    machine — and twenty workers can exhaust the commit limit of a 31 GB
    Windows machine with it (MemoryError on a 5 MB array). A value
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
                 force: bool, label: str, worker=None, timeout: float = 0.0,
                 on_timeout=None) -> dict:
    """Run `todo` with a pool whose size follows `_workers.txt` while it runs.

    Each worker has its own pipe and asks for one job at a time, so the runner
    always knows which job a worker holds, and a worker that dies holding one
    fails exactly that job. `worker` stands in for `_worker_main` in tests.
    A job held past `timeout` seconds (0: never) stops its worker and counts
    as "timeout"; on_timeout(job index, the worker's pid) records it.
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
    taken: dict = {}         # the same -> when it took its job
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
        taken.pop(conn, None)
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
                        taken[conn] = time.time()
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
                    taken.pop(conn, None)
                    counts[msg[3]] = counts.get(msg[3], 0) + 1
                    finished += 1
                elif msg[0] == "broken":
                    print(f"a worker could not start: {msg[2]}", file=sys.stderr)
                    state = "broken"
                    break
            # a job past the time limit: its worker stopped where it is
            for conn, (p, job) in list(pool.items()):
                if timeout <= 0 or job is None or time.time() - taken[conn] <= timeout:
                    continue
                pid = p.pid
                p.kill()
                p.join(timeout=5)
                pool.pop(conn)
                taken.pop(conn)
                conn.close()
                counts["timeout"] = counts.get("timeout", 0) + 1
                finished += 1
                print(f"[{job}] TIMEOUT: stopped after {timeout:g} s", file=sys.stderr)
                if on_timeout is not None:
                    on_timeout(job, pid)
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


# ── the queue: the longest runs first (task 11 of 2026-10-18, §6.2) ─────────
# A campaign of many workers ends when its last run ends: a long run started
# last keeps all but one worker idle. The queue is ordered by the expected
# seconds of every run, the longest first; the runs are independent, so the
# order changes when the campaign ends, not what it finds. Expected: the
# seconds per evaluation of the finished runs of the same algorithm and
# problem in this results directory (a campaign resumed, or run again with
# --force), times the run's budget; without such runs a model, per evaluation:
# the core's (CORE_US: the cores that compute hypervolumes cost the most and
# grow fastest with the objectives) and the problem's (PROBLEM_US, by family:
# the C++ suite's cost nearly nothing, a family evaluated row by row in Python
# the most), both rough, measured with tools/perf in task 10 and 11.
CORE_US = {"sms_emoa": 300.0, "hype": 300.0, "mo_cma_es": 300.0, "sms_m2m": 300.0,
           "r2ibea": 150.0, "mosa": 120.0}
CORE_US_DEFAULT = 50.0
PROBLEM_US = (("WFG", 20.0), ("MaF10", 1000.0), ("MaF11", 1000.0), ("MaF12", 1000.0),
              ("bbobbiobj", 30.0), ("RE", 50.0), ("CRE", 50.0), ("RCM", 80.0),
              ("RWMOP", 80.0), ("DOC", 50.0), ("MW", 20.0), ("DAS-CMOP", 20.0),
              ("LIR-CMOP", 20.0), ("BT", 20.0), ("MOP", 20.0), ("MMF", 30.0),
              ("SYM-PART", 30.0), ("Omni", 30.0), ("IPolygon", 30.0), ("Polygon", 20.0))
PROBLEM_US_DEFAULT = 5.0


def expected_seconds(root: Path, jobs: list) -> dict:
    """{job index: the run's expected seconds} (the comment above)."""
    import re
    # a noisy copy (<base>_n1 .. _h5, benchmarks/noisy.py) or a failing one
    # (<base>_fail...): its own evaluator, row by row; read off the name, so
    # that ordering the queue needs no NumPy
    own_evaluator = re.compile(r".*_([ndsh][15]|fail\w+)$")
    rate: dict = {}
    for m in root.glob("*/*/run_*/meta.json"):
        try:
            meta = json.loads(m.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if meta.get("status") != "done" or not meta.get("fe") or not meta.get("seconds"):
            continue
        key = (m.parents[2].name, m.parents[1].name)
        rate.setdefault(key, []).append(float(meta["seconds"]) / float(meta["fe"]))
    out = {}
    for j in jobs:
        seen = rate.get((j.problem, j.key))
        if seen:
            per = sum(seen) / len(seen)
        else:
            core = CORE_US.get(j.algorithm, CORE_US_DEFAULT) * (1.0 + j.n_objs / 3.0)
            prob = next((us for stem, us in PROBLEM_US if j.problem.startswith(stem)),
                        PROBLEM_US_DEFAULT)
            if own_evaluator.fullmatch(j.problem):
                prob = max(prob, 100.0)             # a run's own evaluator, row by row
            per = 1e-6 * (core + prob)
        out[j.index] = per * j.budget
    return out


def longest_first(root: Path, jobs: list) -> list:
    """The jobs in the order a campaign runs them: the longest expected first,
    ties in the order of the campaign."""
    est = expected_seconds(root, jobs)
    return sorted(jobs, key=lambda j: (-est[j.index], j.index))


def run_campaign(cfg: Config, *, shard: tuple[int, int] | None = None,
                 only: list[int] | None = None, workers: int = 1,
                 force: bool = False, build_refs: bool = True) -> dict:
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
        selected = [j for j in jobs if shard_of(j, n) == i]
    from .. import store
    if store.store_dir(root).is_dir() and not (runner_state(root) or {}).get("alive"):
        # a campaign stopped while a worker wrote: its unfinished record goes
        # (store.log says so) and the run is done again. Not while a runner
        # of this folder beats: its workers may be writing
        store.repair(store.store_dir(root))
    status = job_statuses(root, selected)
    # a run stopped at run_timeout is not run again on resuming (it would
    # stop there again); --force runs it
    todo = longest_first(root, [j for j in selected
                                if force or status[j.index] not in ("done", "timeout")])
    print(f"{cfg.name}: {len(jobs)} jobs total, {len(selected)} in this slice, "
          f"{len(todo)} to run -> {root}", file=sys.stderr)
    prepare_references(todo, spec.n_ref, workers=workers, build=build_refs)

    if todo and not force:
        check_trajectory_versions(root, selected)       # task 14, §10: no mixing
    _warn_exact_hv_time(todo)
    counts = {"done": 0, "failed": 0, "skipped": len(selected) - len(todo)}
    if len(todo) > 1 and cfg.source_path is not None:
        # Worker processes whenever there is more than one job, even with
        # workers = 1, so the pool can be grown while the campaign runs.
        single_threaded_blas()
        label = (f"shard {shard[0]}/{shard[1]}" if shard is not None
                 else f"{len(only)} job(s)" if only is not None else "all")
        got = _run_dynamic(str(cfg.source_path), todo, root, workers=workers,
                           force=force, label=label, timeout=spec.run_timeout,
                           on_timeout=lambda index, pid: _record_timeout(
                               root, spec, jobs[index], pid, spec.run_timeout))
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
    _warn_race_capped(root, [j for j in selected])
    return counts


def _fp_env() -> dict | None:
    """The floating-point environment of this process (_core.fp_env), with a
    warning when denormals are flushed to zero or read as zero: a library
    built with -ffast-math, loaded into the same process, can switch that on
    for everything. None without the extension."""
    try:
        from .. import _core
        env = dict(_core.fp_env())
    except (ImportError, AttributeError):
        return None
    if env.get("ftz") or env.get("daz"):
        print("warning: denormals are flushed to zero or read as zero in this process "
              f"({env}): results differ from a process without it", file=sys.stderr)
    return env


def _warn_exact_hv_time(jobs: list) -> None:
    """Before the runs: the jobs of SMS-EMOA and MO-CMA-ES whose hypervolume
    contributions are exact at EXACT_HV_SLOW_M objectives or more (minimize.py,
    exact_hv_time_warning): their runs can take hours."""
    from .._hv_time import exact_hv_time_warning
    slow = [j for j in jobs if exact_hv_time_warning(j.algorithm, j.n_objs, dict(j.params or {}))]
    if slow:
        names = sorted({f"{j.algorithm} on {j.problem}" for j in slow})
        print(f"warning: {len(slow)} job(s) compute the hypervolume contributions exactly at eight "
              f"objectives or more and can take hours a run: " + "; ".join(names[:10])
              + (" ..." if len(names) > 10 else ""), file=sys.stderr)


# A run whose race for the least hypervolume contributor (SMS-EMOA, MO-CMA-ES
# above hv_exact_max_m; task 14, supplement 1, Д5) stopped at race_max_samples
# in more than this share of its steps is named in the campaign's summary.
RACE_CAPPED_WARN = 0.01


def _warn_race_capped(root: Path, jobs: list) -> None:
    """The summary's warnings: the runs whose race stopped at race_max_samples in
    more than RACE_CAPPED_WARN of their steps, and the runs with denormals
    flushed to zero (fp_env)."""
    many, flushed = [], []
    for j in jobs:
        meta = root / j.rel_dir / "meta.json"
        try:
            d = json.loads(meta.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        share = float((d.get("settings") or {}).get("race_capped_share", 0.0) or 0.0)
        if share > RACE_CAPPED_WARN:
            many.append(f"{j.problem} {j.key} seed {j.seed} ({share:.1%})")
        env = d.get("fp_env") or {}
        if env.get("ftz") or env.get("daz"):
            flushed.append(f"{j.problem} {j.key} seed {j.seed}")
    if many:
        print(f"warning: the race stopped at race_max_samples in more than "
              f"{RACE_CAPPED_WARN:.0%} of the steps of {len(many)} run(s): " + "; ".join(many[:10])
              + (" ..." if len(many) > 10 else ""), file=sys.stderr)
    if flushed:
        print(f"warning: {len(flushed)} run(s) ran with denormals flushed to zero or read as "
              f"zero (meta.json fp_env): " + "; ".join(flushed[:10])
              + (" ..." if len(flushed) > 10 else ""), file=sys.stderr)


# ── the parts of a campaign run apart (task 11 of 2026-10-18, §5) ─────────────


def merge_parts(cfg: Config, parts: list, *, out: Path | None = None) -> dict:
    """--merge: a campaign run in parts (--shard i/k, each part on its own
    machine into its own results directory) gathered into one results directory
    (the config's, or `out`): every job's run copied from the one part that has
    it finished, the parts' reference caches (_references/) from the first that
    has each, campaign.json, and merge.json saying where each run came from.
    The parts stay as they are. Nothing is written, and ValueError says why,
    when a job has no finished run or has one in two parts, when the parts' runs
    were made by different code (revision, the benchmark sources' code stamp, a
    problem's version, the reference version), were measured against different
    references (reference_hash) or have different reference caches, or when
    the target already holds a run of the campaign."""
    import shutil
    from ..benchmarks.registry import REFERENCE_VERSION, problem_version
    spec = campaign_spec(cfg)
    jobs = expand_jobs(cfg, spec)
    root = Path(out) if out is not None else out_root(cfg, spec)
    parts = [Path(q) for q in parts]
    problems: list = []
    found: dict = {}
    for j in jobs:
        have = []
        for q in parts:
            mp = q / j.rel_dir / "meta.json"
            if mp.is_file():
                m = json.loads(mp.read_text(encoding="utf-8"))
                if m.get("status") == "done":
                    have.append((q, m))
        if not have:
            problems.append(f"{j.rel_dir.as_posix()}: no part has it finished")
        elif len(have) > 1:
            problems.append(f"{j.rel_dir.as_posix()}: in {len(have)} parts "
                            f"({', '.join(str(q) for q, _ in have)})")
        else:
            found[j.index] = have[0]
    same = {"revision": set(), "code_stamp": set(), "reference_version": set(),
            "trajectory_version": set()}
    ref_of: dict = {}
    for j in jobs:
        if j.index not in found:
            continue
        q, m = found[j.index]
        for k in same:
            v = trajectory_version_of(m) if k == "trajectory_version" else m.get(k)
            same[k].add(json.dumps(v, sort_keys=True))
        if int(m.get("problem_version") or 1) != problem_version(j.problem):
            problems.append(f"{j.rel_dir.as_posix()}: problem version "
                            f"{m.get('problem_version') or 1}, now {problem_version(j.problem)}")
        ref_of.setdefault(j.problem, set()).add(m.get("reference_hash"))
    for k, vals in same.items():
        if len(vals) > 1:
            problems.append(f"the parts' runs differ in {k}: {', '.join(sorted(vals))}")
    if same["trajectory_version"] and same["trajectory_version"] != {json.dumps(TRAJECTORY_VERSION)}:
        problems.append(f"trajectory version {', '.join(sorted(same['trajectory_version']))}, "
                        f"now {TRAJECTORY_VERSION}")
    if same["reference_version"] and same["reference_version"] != {json.dumps(REFERENCE_VERSION)}:
        problems.append(f"reference version {', '.join(sorted(same['reference_version']))}, "
                        f"now {REFERENCE_VERSION}")
    for name, hashes in sorted(ref_of.items()):
        if len(hashes) > 1:
            problems.append(f"{name}: measured against {len(hashes)} different references")
    caches: dict = {}                                 # a reference cache's file -> its part
    for q in parts:
        d = q / REFERENCES_DIR
        if not d.is_dir():
            continue
        for f in sorted(d.glob("*.npz")):
            if f.name in caches:
                if f.read_bytes() != (caches[f.name] / REFERENCES_DIR / f.name).read_bytes():
                    problems.append(f"{REFERENCES_DIR}/{f.name} differs between {caches[f.name]} "
                                    f"and {q}")
            else:
                caches[f.name] = q
    for j in jobs:
        if (root / j.rel_dir).exists():
            problems.append(f"{root}: holds {j.rel_dir.as_posix()} already")
            break
    if problems:
        raise ValueError("--merge: " + "; ".join(problems[:20]) +
                         (f"; and {len(problems) - 20} more" if len(problems) > 20 else ""))
    root.mkdir(parents=True, exist_ok=True)
    origin: dict = {}
    for j in jobs:
        q, _ = found[j.index]
        shutil.copytree(q / j.rel_dir, root / j.rel_dir)
        origin[str(q)] = origin.get(str(q), 0) + 1
    if caches:
        (root / REFERENCES_DIR).mkdir(parents=True, exist_ok=True)
        for name, q in caches.items():
            target = root / REFERENCES_DIR / name
            if not target.exists():
                shutil.copy2(q / REFERENCES_DIR / name, target)
    _write_json(root / "campaign.json", {
        "name": cfg.name, "config": str(cfg.source_path or ""),
        "problems": sorted({j.problem for j in jobs}),
        "algorithms": [a.key for a in cfg.algorithms],
        "seeds": sorted({j.seed for j in jobs}),
        "jobs": len(jobs), "spec": asdict(spec),
    })
    _write_json(root / "merge.json", {"parts": [str(q) for q in parts], "runs": len(jobs),
                                      "from": origin,
                                      "merged": time.strftime("%Y-%m-%dT%H:%M:%S")})
    return {"runs": len(jobs), "from": origin}


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
    from ..store import as_run_dir
    p = as_run_dir(run_dir) / "trajectory.jsonl"
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
    """Every run under root as {problem, algorithm, seed, status, final, seconds, fe, n_objs}:
    the run folders and the runs of the store (mootation.store; "dir" a
    store.StoredRun then, read as a folder is), a run in both the store's."""
    from .. import store
    rows = []
    if not root.is_dir():
        return rows
    stored = store.stored_runs(root)
    in_store = {s.rel_dir.as_posix() for s in stored}
    runs = [m.parent for m in root.glob("*/*/run_*/meta.json")
            if m.parent.relative_to(root).as_posix() not in in_store] + stored
    for run_dir in runs:
        rel = run_dir.rel_dir if isinstance(run_dir, store.StoredRun) else run_dir.relative_to(root)
        try:
            m = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        rows.append({
            "problem": m.get("problem", rel.parts[0]),
            "algorithm": m.get("algorithm", rel.parts[1]),
            "seed": m.get("seed"), "status": m.get("status", "?"),
            "final": m.get("final", {}) or {}, "seconds": m.get("seconds"),
            "fe": m.get("fe"), "n_objs": m.get("n_objs"), "budget_fe": m.get("budget_fe"),
            "budget_nominal": m.get("budget_nominal"),
            "final_archive": m.get("final_archive") or {},
            "archive_at": m.get("archive_at") or {},
            "core": m.get("core"), "budget_dependent": m.get("budget_dependent"),
            "ladder_of": m.get("ladder_of"),
            "has_reference_front": m.get("has_reference_front"),
            "pop": m.get("pop"), "reference_version": m.get("reference_version"),
            "dir": run_dir,
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
    from .cover import nominal
    main = main_rows(rows)
    if at is None:
        return main
    rungs = {(r["problem"], r["ladder_of"], r["seed"], nominal(r)): r
             for r in rows if r.get("ladder_of")}
    out = []
    for r in main:
        b = int(round(float(at) * nominal(r)))
        rung = rungs.get((r["problem"], r["algorithm"], r["seed"], b))
        out.append(dict(rung, algorithm=r["algorithm"], rung_of=r.get("budget_fe"))
                   if rung is not None else r)
    return out


def value_at(row: dict, metric: str, at: float | None) -> float | None:
    """A run's `metric`: its final value, or at a fraction `at` of its budget.

    At a fraction it is read at `at` times the budget asked for (budget_nominal)
    the way --cover reads a budget (cover.read_at): the first trajectory record
    at or after it — the anytime reading of the same run, so ranks at 10 %,
    25 % and 100 % of the budget come out of one campaign — or in the archive
    scenario (scenario_rows) the archive checkpoint of exactly that budget
    (archive_checkpoints); a run that stopped before spending it, at its end;
    missing where the run has neither. It needs the metric among the
    campaign's `metrics`, which are what a trajectory records. A ladder rung
    put in a full run's place (rungs_in_place) is read at its end.
    """
    v = read_row(row, (metric,), at)[metric]
    return None if v is None else float(v)


def read_row(row: dict, metrics, at: float | None) -> dict:
    """{metric: value} of a run where value_at reads it, and "n", the size of
    its answer there (cover.py, THE ANSWER'S SIZE)."""
    from .cover import answer_size, needs_trajectory, nominal, read_at
    if at is None or row.get("rung_of"):
        return dict({m: row["final"].get(m) for m in metrics}, n=answer_size(row["final"]))
    budget = nominal(row)
    b = int(round(float(at) * budget))
    archive = row.get("scenario") == "archive"
    traj = (read_trajectory(row["dir"])
            if not archive and needs_trajectory(b, budget, row.get("fe")) else ())
    return read_at(b, budget, row.get("fe"), row["final"], traj,
                   (row.get("archive_at") or {}) if archive else None, tuple(metrics))


def is_short(size, pop) -> bool:
    """An answer of fewer than N/2 points, N the population (THE ANSWER'S SIZE)."""
    return size is not None and bool(pop) and size < int(pop) / 2


def compare_table(rows: list[dict], metric: str = "igd", at: float | None = None,
                  sizes: dict | None = None, empty: dict | None = None) -> dict:
    """{problem: {algorithm: (median, q1, q3, n)}} over finished runs.

    `at` reads every run at that fraction of its budget instead of at the end.
    `sizes`, when given, is filled with {(problem, algorithm): (median answer
    size, population)} from the same reading.

    An empty run, a constrained problem's whose answer has no feasible point
    there (feas_share 0, the metric None; metrics.compute), counts as the
    worst value, infinitely bad, in the median and the quartiles and in n
    (task 6 of 2026-10-09, item 5.2). `empty`, when given, is filled with
    {(problem, algorithm): the number of such runs}, where there is any.
    """
    import statistics
    table: dict = {}
    grouped: dict = {}
    counts: dict = {}
    worst = -math.inf if metric in HIGHER_IS_BETTER else math.inf
    for r in rows:
        if r["status"] != "done":
            continue
        got = read_row(r, (metric, "feas_share"), at)
        if got["n"] is not None:
            counts.setdefault((r["problem"], r["algorithm"]), (r.get("pop"), []))[1].append(
                got["n"])
        v = got[metric]
        if v is None and got["feas_share"] == 0:
            v = worst
            if empty is not None:
                key = (r["problem"], r["algorithm"])
                empty[key] = empty.get(key, 0) + 1
        if v is None:
            continue
        grouped.setdefault(r["problem"], {}).setdefault(r["algorithm"], []).append(float(v))
    if sizes is not None:
        for key, (pop, ns) in counts.items():
            sizes[key] = (statistics.median(ns), pop)
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
    """The family of a registry name: DTLZ2_3D -> DTLZ, MaF10_8D -> MaF, ZDT1 -> ZDT
    (mootation.families, the rule of `mootation.benchmarks.families()`)."""
    from ..families import family
    return family(name)


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
    medians, not instead of them. "short" counts the problems where the
    algorithm's median answer has fewer than N/2 points (cover.py, THE
    ANSWER'S SIZE): a metric such as GD+ does not see it. "empty" counts its
    runs without a feasible point on the constrained problems, each ranked as
    the worst value (compare_table): the mark beside a rank that has them.

    The real-world problems (RE, CRE, RCM: mootation.families.is_engineering)
    are ranked apart from the synthetic ones (task 6, item 6): in a campaign
    with both, "all" averages the synthetic problems only and the group
    "engineering" the real-world ones. For a metric that reads the reference
    front or its frame (NEEDS_FRONT, HV_NAMES), the synthetic problems whose
    reference is an approximation (properties front_ref) go to the group
    "approx_front" where the campaign has both kinds, and "all" keeps those
    with an exact front.

        {"metric": "igd", "lower_better": True, "n_problems": 45,
         "groups": ["DTLZ", "WFG", "ZDT", "M=2", "M=3", "M=5"],
         "algorithms": [(name, {"all": (mean, n), "DTLZ": (mean, n), ...,
                                "wins": w, "short": s, "empty": e}), ...]}   # best first
    """
    sizes: dict = {}
    empty: dict = {}
    table = compare_table(rows, metric, at, sizes, empty)
    lower_better = metric not in HIGHER_IS_BETTER
    n_objs: dict = {}
    for r in rows:
        if r.get("n_objs") is not None:
            n_objs.setdefault(r["problem"], int(r["n_objs"]))

    def key(med: float) -> tuple:
        if not math.isfinite(med):
            return (1, 0.0)
        return (0, med if lower_better else -med)

    from ..families import is_engineering
    apart = len({is_engineering(p) for p in table}) == 2
    approx: set = set()
    if base_name(metric) in NEEDS_FRONT or metric in HV_NAMES:
        try:
            from ..benchmarks.properties import approximate_front
        except ImportError:                     # no NumPy, no registry to ask
            approximate_front = None
        if approximate_front is not None:
            synthetic = [p for p in table if not is_engineering(p)]
            approx = {p for p in synthetic if approximate_front(p)}
            if len(approx) == len(synthetic):
                approx = set()                  # nothing to set them apart from
    ranks: dict = {}
    wins: dict = {}
    for prob, algs in table.items():
        labels = ["engineering" if apart and is_engineering(prob)
                  else "approx_front" if prob in approx else "all",
                  problem_family(prob)]
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
              + ([f"M={m}" for m in ms] if len(ms) > 1 else [])
              + (["engineering"] if apart else [])
              + (["approx_front"] if approx else []))
    out = []
    for a, per in ranks.items():
        entry: dict = {g: (sum(v) / len(v), len(v)) for g, v in per.items()}
        entry.setdefault("all", (math.nan, 0))       # an algorithm on real-world problems only
        entry["wins"] = wins.get(a, 0)
        entry["short"] = sum(1 for p in table if a in table[p]
                             and is_short(*sizes.get((p, a), (None, None))))
        entry["empty"] = sum(v for (_, alg), v in empty.items() if alg == a)
        out.append((a, entry))
    out.sort(key=lambda t: (t[1]["all"][0] if math.isfinite(t[1]["all"][0]) else math.inf,
                            -t[1]["wins"], t[0]))
    return {"metric": metric, "at": at, "lower_better": lower_better,
            "n_problems": len(table), "groups": groups, "algorithms": out,
            "n_engineering": sum(1 for p in table if is_engineering(p)),
            "n_approx_front": len(approx)}


def format_ranks(ranks: dict) -> str:
    """The rank table as fixed-width text, for the terminal."""
    groups = ranks["groups"]
    at = f" at {ranks['at']:.0%} of the budget" if ranks.get("at") is not None else ""
    lines = [f"mean rank by median {ranks['metric']}{at} over {ranks['n_problems']} problem(s): "
             f"1 = best, ties share the average rank; probs = problems ranked on; short = "
             f"problems where the median answer has fewer than N/2 points",
             f"{'algorithm':<14}{'mean':>7}{'wins':>6}{'probs':>6}{'short':>6}"
             + "".join(f"{g[:9]:>10}" for g in groups)]
    from .postprocess import budget_note
    marked = False
    empties = any(e.get("empty") for _, e in ranks["algorithms"])
    if empties:
        lines[1] += f"{'empty':>6}"
    for alg, e in ranks["algorithms"]:
        mean, n = e["all"]
        name = alg[:13] + budget_note(alg)
        marked |= name != alg[:13]
        line = f"{name:<14}{mean:>7.2f}{e['wins']:>6}{n:>6}{e.get('short', 0):>6}"
        if empties:
            line += f"{e.get('empty', 0) or '':>6}"
        for g in groups:
            line += f"{e[g][0]:>10.2f}" if g in e else f"{'-':>10}"
        lines.append(line)
    if marked:
        lines.append("* runs on a schedule of the budget share spent: compare its runs at "
                     "different budgets by fraction of budget, not by evaluations")
    if empties:
        lines.append("empty = runs whose answer has no feasible point (constrained problems): "
                     "no value, ranked as the worst one")
    if "engineering" in groups:
        lines.append("all = the synthetic problems; engineering = the real-world RE, CRE and "
                     "RCM, ranked apart")
    if "approx_front" in groups:
        lines.append("approx_front = the synthetic problems whose reference front is an "
                     "approximation (properties front_ref), ranked apart; all = those with an "
                     "exact one")
    if ranks.get("n_engineering") and base_name(ranks["metric"]) in NEEDS_FRONT:
        lines.append(f"on RE, CRE and RCM, {ranks['metric']} is the distance to an "
                     f"approximation of the front, not to the front")
    return "\n".join(lines)


def write_rank_csv(ranks: dict, path: Path) -> None:
    groups = ranks["groups"]
    with path.open("w", encoding="utf-8") as fh:
        fh.write("algorithm,mean_rank,problems,wins"
                 + "".join(f",{g}_mean_rank,{g}_problems" for g in groups)
                 + ",short_answers,empty_runs\n")
        for alg, e in ranks["algorithms"]:
            mean, n = e["all"]
            cells = [alg, f"{mean:.6g}", str(n), str(e["wins"])]
            for g in groups:
                cells += [f"{e[g][0]:.6g}", str(e[g][1])] if g in e else ["", ""]
            cells += [str(e.get("short", 0)), str(e.get("empty", 0))]
            fh.write(",".join(cells) + "\n")


# ── indicators added after the fact ─────────────────────────────────────────


def _open_text(path: Path, mode: str = "r"):
    """A results text file, gzip-compressed when its name ends in .gz (a
    stored run's file reads as its text: store.StoredFile)."""
    from ..store import StoredFile
    if isinstance(path, StoredFile):
        if "r" not in mode:
            raise OSError(f"{path}: a stored run is read only (mootation store export --files)")
        return path.open(mode)
    if path.suffix == ".gz":
        import gzip
        return gzip.open(path, mode + "t", compresslevel=6, encoding="utf-8", newline="\n")
    return path.open(mode, encoding="utf-8")


def _read_points(path: Path, m: int, n: int):
    """(F, X) from a final.csv, archive.csv or final_archive.csv.gz: m objective
    columns, then n variables."""
    import numpy as np
    from ..store import StoredFile
    if isinstance(path, StoredFile) and path.table() is not None:
        # the numbers the store keeps are what parsing the text gives
        A = path.table() if len(path.table()) else np.zeros((0, m + n))
        return A[:, :m], (A[:, m:m + n] if A.shape[1] >= m + n else None)
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


def _archive_answer(run_dir: Path, meta: dict, p, m: int, n: int):
    """(F, X) of a run's archive scenario, from its files: the selection the
    run kept (final_archive.csv.gz, archive_variables = "selected"), or
    archive.csv reduced to the population size in the frame the run used, by
    DSS or the run's answer_select; None where it has neither, or an empty
    archive."""
    import numpy as np
    from .archive import dss_order, igdp_self_order
    if (run_dir / "final_archive.csv.gz").is_file():
        return _read_points(run_dir / "final_archive.csv.gz", m, n)
    path = run_dir / "archive.csv"
    if not path.is_file():
        return None
    F, X = _read_points(path, m, n)
    if not len(F):
        return None
    # a noisy copy's archive: selected by the values the run saw (f*_seen)
    S = _seen_columns(path, m)
    by = F if S is None else S
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
    if len(F) <= p.pop_size:
        idx = np.arange(len(F))
    elif arc_meta.get("answer_select", "dss") == "igdp_self":
        idx = igdp_self_order(by, p.pop_size, ideal=lo, nadir=hi)
    else:
        idx = dss_order(by, k=p.pop_size, ideal=lo, nadir=hi)
    return F[idx], (None if X is None else X[idx])


def _seen_columns(path, m: int):
    """The f1_seen .. fm_seen columns of a results CSV (a noisy copy's values as
    its algorithm saw them), or None where it has none."""
    import numpy as np
    from ..store import StoredFile
    if isinstance(path, StoredFile) and path.table() is not None:
        import json as _json
        head = _json.loads(path.run.blocks()[path.name + "#head"].data)
        cols, A = head["columns"], path.table()
    else:
        cols, rows = None, []
        with _open_text(path) as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if cols is None:
                    cols = line.split(",")
                    continue
                rows.append([float(v) for v in line.split(",")])
        if cols is None:
            return None
        A = np.asarray(rows, float).reshape(-1, len(cols))
    want = [f"f{i+1}_seen" for i in range(m)]
    if not all(c in cols for c in want):
        return None
    return A[:, [cols.index(c) for c in want]]


def _recompute_one(args) -> str:
    run_dir, names, n_ref, hv_options, scenario = args
    import numpy as np
    from ..benchmarks import get as bench_get
    from . import metrics as M
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
                  cyclic=p.cyclic_vars, distance_fn=p.distance_fn,
                  constraints=p.constraints if p.has_cons else None)
        if scenario == "archive":
            got = _archive_answer(run_dir, meta, p, m, n)
            if got is None:
                return "skipped"
            F, X = got
            if X is None:
                kw["constraints"] = None             # the archive keeps feasible points only
            out = dict(meta.get("final_archive") or {})
            out.update(M.compute(F, X=X, **kw))
            out["n"] = int(len(F))
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
        p = bench_get(meta["problem"])
        if not todo or (p.has_cons and "X" not in S):     # feasibility needs them too
            return "skipped"
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
                                     bounds=p.bounds, cyclic=p.cyclic_vars,
                                     distance_fn=p.distance_fn,
                                     constraints=p.constraints if p.has_cons else None))
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


def _rereference_one(args) -> str:
    """One run's reference-dependent values measured again against the current
    reference front (recompute_reference)."""
    run_dir, current = args
    import shutil
    import numpy as np
    from ..benchmarks import get as bench_get
    from . import metrics as M
    run_dir = Path(run_dir)
    meta_path = run_dir / "meta.json"
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        old = int(meta.get("reference_version") or 1)
        if meta.get("status") != "done" or old >= current:
            return "skipped"
        p = bench_get(meta["problem"])
        m = int(meta.get("n_objs") or p.n_obj)
        n = int(meta.get("n_vars") or p.n_vars)
        ref = np.asarray(p.pareto_front(int(meta.get("n_ref") or 1000)), float)
        kw = dict(ref_front=ref, ideal=p.ideal, nadir=p.nadir, pop=p.pop_size,
                  bounds=p.bounds, cyclic=p.cyclic_vars)

        def again(F, values: dict, X=None) -> dict:
            """The reference-dependent metrics among `values`, measured on F
            again: {name: old value}, `values` updated in place. A constrained
            problem's on the feasible rows, from X (metrics.compute)."""
            names = [x for x in values if x in NEEDS_FRONT]
            if not names or not len(F) or not np.all(np.isfinite(F)):
                return {}
            cons = p.constraints if p.has_cons and X is not None else None
            new = M.compute(F, which=names, X=X, constraints=cons, **kw)
            new.pop("feas_share", None)
            before = {x: values.get(x) for x in new}
            values.update(new)
            return before

        kept: dict = {}
        if meta.get("final"):
            F, X = _read_points(run_dir / "final.csv", m, n)
            kept["final"] = again(F, meta["final"], X)
        if meta.get("final_archive") and (run_dir / "final_archive.csv.gz").is_file():
            F, _ = _read_points(run_dir / "final_archive.csv.gz", m, n)
            kept["final_archive"] = again(F, meta["final_archive"])   # feasible points only
        if meta.get("archive_at") and (run_dir / "archive_at.npz").is_file():
            with np.load(run_dir / "archive_at.npz") as z:
                A = {k: z[k] for k in z.files}
            kept["archive_at"], start = {}, 0
            for b, cnt in zip(A["budget"].tolist(), A["n"].tolist()):
                F = A["F"][start:start + cnt].astype(float)
                start += cnt
                if str(b) in meta["archive_at"]:
                    kept["archive_at"][str(b)] = again(F, meta["archive_at"][str(b)])
        traj_path, snap_path = run_dir / "trajectory.jsonl", run_dir / "snapshots.npz"
        recs = None
        if traj_path.is_file() and snap_path.is_file():
            with np.load(snap_path) as z:
                S = {k: z[k] for k in z.files}
            recs = [json.loads(line) for line in traj_path.read_text(encoding="utf-8").splitlines()
                    if line.strip()]
            if [r.get("fe") for r in recs] != S["fe"].tolist():
                print(f"{run_dir}: the snapshots and the trajectory do not line up",
                      file=sys.stderr)
                return "failed"
            start = 0
            for rec, cnt in zip(recs, S["n"].tolist()):
                F = S["F"][start:start + cnt].astype(float)
                X = S["X"][start:start + cnt].astype(float) if "X" in S else None
                start += cnt
                if cnt and rec.get("finite", True):
                    again(F, rec, X)
            keep_old = run_dir / f"trajectory.ref_v{old}.jsonl"
            if not keep_old.is_file():
                shutil.copy2(traj_path, keep_old)
            meta["trajectory_ref_file"] = {f"v{old}": keep_old.name}
        meta[f"reference_v{old}"] = kept
        meta["reference_version"] = current
        meta["reference_rows"] = int(len(ref))
        if recs is not None:
            tmp = traj_path.with_suffix(".jsonl.tmp")
            tmp.write_text("".join(json.dumps(r, separators=(",", ":")) + "\n" for r in recs),
                           encoding="utf-8")
            os.replace(tmp, traj_path)
        _write_json(meta_path, meta)
        return "done"
    except Exception as e:                           # noqa: BLE001
        print(f"{run_dir}: {type(e).__name__}: {e}", file=sys.stderr)
        return "failed"


def recompute_reference(root: Path, *, workers: int = 1, problems=None) -> dict:
    """Bring every finished run measured against an older reference version
    than the current one (benchmarks.registry, REFERENCE_VERSION), on a problem
    whose reference changed since (REFERENCE_CHANGED), up to date: every
    reference-dependent metric (NEEDS_FRONT) it recorded, measured again from
    its saved answers — final.csv, final_archive.csv.gz, the archive
    checkpoints of archive_at.npz, the trajectory from snapshots.npz. The old
    values stay beside the new: meta["reference_v<old>"] for the final ones and
    the checkpoints, trajectory.ref_v<old>.jsonl for the trajectory. A problem
    whose DEFINITION changed (PROBLEM_CHANGED: IPolygon at version 2) is not
    measured again — its runs answered another problem — but counted, to be
    run again."""
    from ..benchmarks.registry import PROBLEM_CHANGED, REFERENCE_CHANGED, REFERENCE_VERSION
    changed = set().union(*REFERENCE_CHANGED.values()) if REFERENCE_CHANGED else set()
    redefined = set().union(*PROBLEM_CHANGED.values()) if PROBLEM_CHANGED else set()
    metas = [m for m in root.glob("*/*/run_*/meta.json")
             if (problems is None or m.parents[2].name in problems)
             and m.parents[2].name in changed]

    def older(meta_path) -> bool:
        try:
            m = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False
        return int(m.get("reference_version") or 1) < REFERENCE_VERSION

    counts: dict = {"to_run_again": sum(1 for m in metas
                                        if m.parents[2].name in redefined and older(m))}
    tasks = [(str(m.parent), REFERENCE_VERSION) for m in metas
             if m.parents[2].name not in redefined]
    if workers > 1 and len(tasks) > 1:
        import multiprocessing as mp
        single_threaded_blas()
        with mp.Pool(processes=workers) as pool:
            for status in pool.imap_unordered(_rereference_one, tasks, chunksize=8):
                counts[status] = counts.get(status, 0) + 1
    else:
        for task in tasks:
            status = _rereference_one(task)
            counts[status] = counts.get(status, 0) + 1
    return counts


def _measure_saved(run_dir: Path, meta: dict, p, ref, ideal, nadir, suffix: str) -> bool:
    """A finished run's front indicators against `ref`, in the frame (ideal,
    nadir), from its saved answers — final, archive, archive checkpoints and,
    from snapshots.npz, the trajectory — added as "<indicator><suffix>" beside
    the old values, which stay as they are. meta is updated in place, not
    written; trajectory.jsonl is rewritten. False when the snapshots and the
    trajectory do not line up."""
    import numpy as np
    from . import metrics as M
    m = int(meta.get("n_objs") or p.n_obj)
    n = int(meta.get("n_vars") or p.n_vars)
    cons = p.constraints if p.has_cons else None

    def add(F, X, values: dict) -> None:
        names = [x for x in values if x in NEEDS_FRONT]
        if not names or not len(F) or not np.all(np.isfinite(F)):
            return
        # an archive holds feasible points only: without its variables, as it is
        c = cons if X is not None else None
        got = M.compute(F, ref_front=ref, ideal=ideal, nadir=nadir, which=names, X=X,
                        constraints=c)
        values.update({x + suffix: got[x] for x in names})

    if meta.get("final") and (run_dir / "final.csv").is_file():
        F, X = _read_points(run_dir / "final.csv", m, n)
        add(F, X, meta["final"])
    got = _archive_answer(run_dir, meta, p, m, n) if meta.get("final_archive") else None
    if got is not None:
        F, X = got
        add(F, X, meta["final_archive"])
    if meta.get("archive_at") and (run_dir / "archive_at.npz").is_file():
        with np.load(run_dir / "archive_at.npz") as z:
            A = {k: z[k] for k in z.files}
        start = 0
        for b, cnt in zip(A["budget"].tolist(), A["n"].tolist()):
            F = A["F"][start:start + cnt].astype(float)
            X = A["X"][start:start + cnt].astype(float) if "X" in A else None
            start += cnt
            if str(b) in meta["archive_at"]:
                add(F, X, meta["archive_at"][str(b)])
    traj_path, snap_path = run_dir / "trajectory.jsonl", run_dir / "snapshots.npz"
    if traj_path.is_file() and snap_path.is_file():
        with np.load(snap_path) as z:
            S = {k: z[k] for k in z.files}
        recs = [json.loads(line) for line in traj_path.read_text(encoding="utf-8").splitlines()
                if line.strip()]
        if [r.get("fe") for r in recs] != S["fe"].tolist():
            print(f"{run_dir}: the snapshots and the trajectory do not line up",
                  file=sys.stderr)
            return False
        start = 0
        for rec, cnt in zip(recs, S["n"].tolist()):
            F = S["F"][start:start + cnt].astype(float)
            X = S["X"][start:start + cnt].astype(float) if "X" in S else None
            start += cnt
            if cnt and rec.get("finite", True):
                add(F, X, rec)
        tmp = traj_path.with_suffix(".jsonl.tmp")
        tmp.write_text("".join(json.dumps(r, separators=(",", ":")) + "\n" for r in recs),
                       encoding="utf-8")
        os.replace(tmp, traj_path)
    return True


def _dense_one(run_dir: str) -> str:
    """One finished run's front indicators against the dense reference, from
    its saved answers, as new "<indicator>_ref_v3" values beside the old ones
    (recompute_dense)."""
    from ..benchmarks import get as bench_get
    from ..benchmarks.registry import dense_reference
    run_dir = Path(run_dir)
    meta_path = run_dir / "meta.json"
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("status") != "done" or meta.get("reference_v3"):
            return "skipped"
        p = bench_get(meta["problem"])
        ref3, info3 = dense_reference(meta["problem"], p.pop_size)
        if not _measure_saved(run_dir, meta, p, ref3, p.ideal, p.nadir, DENSE_SUFFIX):
            return "failed"
        meta["reference_v3"] = info3
        _write_json(meta_path, meta)
        return "done"
    except Exception as e:                           # noqa: BLE001
        print(f"{run_dir}: {type(e).__name__}: {e}", file=sys.stderr)
        return "failed"


def recompute_dense(root: Path, *, workers: int = 1, problems=None) -> dict:
    """Add to every finished run on a problem with a dense reference
    (registry.dense_reference: 4 to 10 objectives) its front indicators
    against it — final, archive, archive checkpoints and, from snapshots.npz,
    the trajectory — as "<indicator>_ref_v3" beside the old values, which stay
    as they are; meta["reference_v3"] records the reference's rows and step.
    A run that has them is skipped."""
    from ..benchmarks.registry import has_dense_reference
    tasks = [str(m.parent) for m in root.glob("*/*/run_*/meta.json")
             if (problems is None or m.parents[2].name in problems)
             and has_dense_reference(m.parents[2].name)]
    counts: dict = {}
    if workers > 1 and len(tasks) > 1:
        import multiprocessing as mp
        single_threaded_blas()
        with mp.Pool(processes=workers) as pool:
            for status in pool.imap_unordered(_dense_one, tasks, chunksize=4):
                counts[status] = counts.get(status, 0) + 1
    else:
        for task in tasks:
            status = _dense_one(task)
            counts[status] = counts.get(status, 0) + 1
    return counts


# ── the campaign's own reference ────────────────────────────────────────────
#
# Where a problem's reference front is an approximation (properties front_ref)
# — or to read every problem alike — a campaign's runs can be measured against
# the non-dominated points of all its own runs on that problem: their final
# answers and, where they kept one, their run archives, the feasible points of
# a constrained problem's. pooled_reference/<problem>.csv holds the points,
# <problem>.json which runs went into it (algorithm entry, seed, budget, the
# commit each ran on, the rows each gave), the code that built it and its
# sha256: the version a run's meta["reference_pooled"] names. The values go
# beside the old ones as "<indicator>_pooled", in the pool's own frame (its
# corners). Runs added later change the pool: the next call builds it again
# and measures every run whose stamp names another version.

POOLED_DIR = "pooled_reference"


def merge_nondominated(P, R):
    """The non-dominated rows of P and R together, P already non-dominated
    and without copies; a row equal to one of P is kept once."""
    import numpy as np
    from . import metrics as M
    R = M.nondominated(R)
    if not len(P) or not len(R):
        return R if not len(P) else P
    le = (P[None, :, :] <= R[:, None, :]).all(axis=2)       # [i, j]: P_j <= R_i
    R = R[~le.any(axis=1)]                                  # dominated by or equal to P
    if not len(R):
        return P
    le = (R[None, :, :] <= P[:, None, :]).all(axis=2)       # [j, i]: R_i <= P_j
    lt = (R[None, :, :] < P[:, None, :]).any(axis=2)
    return np.vstack([P[~(le & lt).any(axis=1)], R])


def _run_points(run_dir: Path, meta: dict, p) -> dict:
    """{answer: F} of a finished run for its problem's pool: "final", the
    feasible rows of final.csv, and "archive", the run archive (archive.csv,
    feasible points only), or the archive answer it kept
    (final_archive.csv.gz); finite rows only."""
    import numpy as np
    from . import metrics as M
    m = int(meta.get("n_objs") or p.n_obj)
    n = int(meta.get("n_vars") or p.n_vars)
    out = {}
    if (run_dir / "final.csv").is_file():
        F, X = _read_points(run_dir / "final.csv", m, n)
        if p.has_cons:
            F = F[M.feasible_mask(X, p.constraints)] if X is not None else F[:0]
        out["final"] = F
    for name in ("archive.csv", "final_archive.csv.gz"):
        if (run_dir / name).is_file():
            out["archive"] = _read_points(run_dir / name, m, n)[0]
            break
    return {k: F[np.all(np.isfinite(F), axis=1)] for k, F in out.items()}


def build_pooled_reference(root: Path, problem: str, meta_paths: list) -> dict | None:
    """Pool one problem's finished runs (the block above): write
    pooled_reference/<problem>.csv and .json, return the json's content; None
    when no run gave a point."""
    import hashlib
    import time
    import numpy as np
    from .. import __version__
    from ..benchmarks import get as bench_get
    from .provenance import revision
    p = bench_get(problem)
    pool = np.zeros((0, p.n_obj))
    runs = []
    for meta_path in sorted(meta_paths):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("status") != "done":
            continue
        got = _run_points(meta_path.parent, meta, p)
        for F in got.values():
            pool = merge_nondominated(pool, F)
        runs.append({"run": f"{meta_path.parents[1].name}/{meta_path.parent.name}",
                     "algorithm": meta.get("algorithm"), "seed": meta.get("seed"),
                     "budget_fe": meta.get("budget_fe"),
                     "commit": (meta.get("revision") or {}).get("commit"),
                     "rows": {k: int(len(F)) for k, F in got.items()}})
    if not len(pool):
        return None
    pool = pool[np.lexsort(pool.T[::-1])]                   # one order, whatever the runs'
    text = ",".join(f"f{j + 1}" for j in range(p.n_obj)) + "\n" + "".join(
        ",".join(f"{v:.17g}" for v in row) + "\n" for row in pool)
    rev = revision()
    info = {"problem": problem, "rows": int(len(pool)),
            "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "ideal": pool.min(axis=0).tolist(), "nadir": pool.max(axis=0).tolist(),
            "answers": "final (feasible rows) and the run archive",
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "analysis": {"mootation": __version__, "commit": rev.get("commit"),
                         "dirty": rev.get("dirty"), "source": rev.get("source")},
            "n_runs": len(runs), "runs": runs}
    out = root / POOLED_DIR
    out.mkdir(exist_ok=True)
    _write_text(out / f"{problem}.csv", text)
    _write_json(out / f"{problem}.json", info)
    return info


def pooled_reference_of(root: Path, problem: str):
    """(rows, info) of a problem's pool as build_pooled_reference wrote it."""
    import numpy as np
    out = root / POOLED_DIR
    info = json.loads((out / f"{problem}.json").read_text(encoding="utf-8"))
    return np.loadtxt(out / f"{problem}.csv", delimiter=",", skiprows=1, ndmin=2), info


def _pooled_one(args) -> str:
    """One finished run measured against its problem's pool (pooled_reference)."""
    run_dir, root = Path(args[0]), Path(args[1])
    from ..benchmarks import get as bench_get
    meta_path = run_dir / "meta.json"
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("status") != "done":
            return "skipped"
        ref, info = pooled_reference_of(root, meta["problem"])
        if (meta.get("reference_pooled") or {}).get("sha256") == info["sha256"]:
            return "skipped"
        p = bench_get(meta["problem"])
        if not _measure_saved(run_dir, meta, p, ref, info["ideal"], info["nadir"],
                              POOLED_SUFFIX):
            return "failed"
        meta["reference_pooled"] = {"sha256": info["sha256"], "rows": info["rows"],
                                    "n_runs": info["n_runs"], "created": info["created"]}
        _write_json(meta_path, meta)
        return "done"
    except Exception as e:                           # noqa: BLE001
        print(f"{run_dir}: {type(e).__name__}: {e}", file=sys.stderr)
        return "failed"


def pooled_reference(root: Path, *, workers: int = 1, problems=None) -> dict:
    """Build every problem's pool from the campaign's finished runs and measure
    every run against it (the block above); a run already measured against
    the same version is skipped. {"pools": problems pooled, "done": ...}."""
    by_problem: dict = {}
    for meta_path in root.glob("*/*/run_*/meta.json"):
        prob = meta_path.parents[2].name
        if problems is None or prob in problems:
            by_problem.setdefault(prob, []).append(meta_path)
    counts: dict = {"pools": 0}
    tasks = []
    for prob, paths in sorted(by_problem.items()):
        if build_pooled_reference(root, prob, paths) is not None:
            counts["pools"] += 1
            tasks += [(str(m.parent), str(root)) for m in paths]
    if workers > 1 and len(tasks) > 1:
        import multiprocessing as mp
        single_threaded_blas()
        with mp.Pool(processes=workers) as pool:
            for status in pool.imap_unordered(_pooled_one, tasks, chunksize=4):
                counts[status] = counts.get(status, 0) + 1
    else:
        for task in tasks:
            status = _pooled_one(task)
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
    ap.add_argument("--shard", help="i/n: run part i of n (a job's part: a hash of its problem, "
                                    "algorithm and seed; every machine of n runs one part)")
    ap.add_argument("--merge", nargs="+", metavar="PART",
                    help="gather the results directories of the parts (--shard i/n, run "
                         "apart) into the config's results directory, after checking that "
                         "every run is there once and that the parts agree (code, problem "
                         "versions, references)")
    ap.add_argument("--job", type=int, action="append", help="run one job by index (repeatable)")
    ap.add_argument("--workers", type=int, default=1,
                    help="worker processes to start with; change it while the campaign runs by "
                         "writing the number into <results>/_workers.txt (0 drains and stops)")
    ap.add_argument("--force", action="store_true", help="rerun jobs already marked done")
    ap.add_argument("--no-build-refs", action="store_true",
                    help="stop with the list where the cache of references (mootation refs) lacks "
                         "a problem's entry, instead of building it before the first run")
    ap.add_argument("--refs-dir", default="", metavar="DIR",
                    help="the cache of references' root (default: the user's cache; "
                         "MOOTATION_REFS_DIR)")
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
    ap.add_argument("--recompute-reference", action="store_true",
                    help="measure every finished run whose problem's reference front changed "
                         "since the run (benchmarks registry: REFERENCE_VERSION, "
                         "REFERENCE_CHANGED) again against the current one — final, archive, "
                         "archive checkpoints and trajectory, from the saved answers — keeping "
                         "the old values beside the new; and measure every run on a problem "
                         "with a dense reference (4 to 10 objectives) against it too, as "
                         "<indicator>_ref_v3, the old values untouched; then exit (honours "
                         "--workers and --problems)")
    ap.add_argument("--pooled-reference", action="store_true",
                    help="the campaign's own reference: per problem, the non-dominated points "
                         "of all its finished runs (final answers and run archives, feasible "
                         "points only), written to pooled_reference/<problem>.csv with a .json "
                         "of the runs that went into it, the code that built it and its sha256; "
                         "every run measured against it as <indicator>_pooled, in the pool's "
                         "frame, the old values untouched; then exit (honours --workers and "
                         "--problems)")
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
    ap.add_argument("--adjust", choices=("holm", "bh"), default="holm",
                    help="with --reference: how the per-problem p-values are adjusted over "
                         "the problems — Holm (default) or Benjamini-Hochberg")
    ap.add_argument("--ci", action="store_true",
                    help="with --ranks: a 95 %% bootstrap interval for every mean rank, "
                         "resampling the problems")
    ap.add_argument("--by", metavar="KEY",
                    help="with --ranks: mean ranks within groups of problems sharing a "
                         "property: front, multimodal, deceptive, bias, scaled, separable, "
                         "centre, boundary, degenerate, disconnected, linkage, n_obj, "
                         "n_vars, bbob_groups, multimodal_ps, ps_parts "
                         "(benchmarks/properties.py)")
    ap.add_argument("--gap", metavar="METRIC",
                    help="print each algorithm's median gap to the best final METRIC any run "
                         "reached, per problem, and exit")
    ap.add_argument("--cec2019-ranks", action="store_true",
                    help="print each algorithm's mean rank by median IGD, IGDX and 1/PSP and "
                         "their mean, the CEC 2019 multimodal competition's protocol (the "
                         "problems with a Pareto-set sample)")
    ap.add_argument("--zero-share", nargs="?", const="hv", metavar="METRIC",
                    help="print the share of seeds whose final METRIC (default hv) is exactly "
                         "0, per problem, and exit")
    ap.add_argument("--ecdf", metavar="METRIC",
                    help="print the runtime ECDF of METRIC over (run, target) pairs, targets "
                         "at fixed distances from the best known value, and exit")
    ap.add_argument("--bias", action="store_true",
                    help="print where each algorithm searched (Kudela et al. 2026): binsizes "
                         "over 20 bins at the bounds and in the centre for the final "
                         "populations, the non-dominated and all evaluated points, exact "
                         "hits of a bound and of 1/2, chi-square on one point per run, the "
                         "Clark-Evans index (the uninformative problems make any preference "
                         "the algorithm's own)")
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
                         "(DTLZ5/6 and MaF6 from four objectives, WFG3, DTLZ1 at five; against "
                         "references of version 1 IPolygon and the aliased WFG1/WFG2) or older "
                         "than the current one where it changed, which are left out of the "
                         "coverage otherwise")
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
    ap.add_argument("--portfolio", action="store_true",
                    help="what the portfolio gains over its members, per group of problems (with "
                         "a front, bbob-biobj) at --cover's levels and budgets: the oracle and the "
                         "single best solver at 5, 7 and 10 of ten seeds, Shapley values, "
                         "complementarity, the oracle gap per problem, and the matrix problem x "
                         "algorithm x budget (portfolio.py); CSV under <results>/_portfolio/")
    ap.add_argument("--oracle", action="store_true",
                    help="the best N points of every run's archive under IGD+ against the "
                         "reference front, beside the population, the archive answer, "
                         "answer_select = \"igdp_self\" and the whole archive: what the search "
                         "found, apart from how the answer was chosen (oracle.py; not "
                         "--portfolio's oracle, the best algorithm per problem); CSV under "
                         "<results>/_oracle/")
    ap.add_argument("--invariance-check", action="store_true",
                    help="every algorithm entry on every problem of the campaign, at its "
                         "budget: the trajectory on the objectives times 2^10 and times "
                         "(2^-6, 1, 2^6, ...) against the original, bit for bit, and the "
                         "final IGD+ on the objectives plus 2^4 against the original over "
                         "--inv-seeds seeds, rank-sum and Holm (invariance.py); CSV under "
                         "<results>/_invariance/")
    ap.add_argument("--inv-seeds", type=int, default=30, metavar="N",
                    help="with --invariance-check: the seeds of the shift's comparison (30)")
    ap.add_argument("--instance-space", action="store_true",
                    help="the problems as points of a plane drawn from their properties "
                         "(PILOT), per group of problems, and each algorithm's footprint there "
                         "at --cover's levels and budgets (instance_space.py; needs SciPy); CSV "
                         "under <results>/_instance_space/")
    ap.add_argument("--is-calibration", choices=("whiten", "preserve"), default="whiten",
                    help="with --instance-space: the plane's axes, whitened or stretched to "
                         "the highest topological preservation (instance_space.py, THE PLANE)")
    ap.add_argument("--is-weight", choices=("blocks", "none"), default="blocks",
                    help="with --instance-space: the performance rows weighted sqrt(q/a) "
                         "(q features, a algorithms) or not")
    ap.add_argument("--is-bbob-features", choices=("sample", "properties"), default="sample",
                    help="with --instance-space: the bbob-biobj problems' features, from a "
                         "uniform sample of their box or their properties")
    ap.add_argument("--behaviour", action="store_true",
                    help="describe every run from its trajectory (time to targets, "
                         "convergence slope, converged-then-spreading, collapse, stagnation, "
                         "duplicates, igdx), and group the algorithms by those profiles beside "
                         "their declared families (behaviour.py); CSV under <results>/_behaviour/")
    args = ap.parse_args(argv)
    if args.refs_dir:
        from .refs import ENV_DIR
        os.environ[ENV_DIR] = args.refs_dir           # the workers inherit it

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
        from ..benchmarks.registry import problem_version
        statuses = job_statuses(root, jobs)
        for j in jobs:
            st = statuses[j.index]
            note = f"  [{j.pop_note}]" if j.pop_note else ""
            if st == "done":
                meta = json.loads((root / j.rel_dir / "meta.json").read_text(encoding="utf-8"))
                ran = int(meta.get("problem_version") or 1)
                if ran < problem_version(j.problem):
                    note += (f"  [problem v{ran}, now v{problem_version(j.problem)}: to run "
                             f"again, --job {j.index} --force]")
                tv = trajectory_version_of(meta)
                if tv != TRAJECTORY_VERSION:
                    note += f"  [trajectory v{tv}, now v{TRAJECTORY_VERSION}]"
            print(f"{j.index:5d} {j.problem:<14} {j.key:<14} seed {j.seed:<3} "
                  f"pop {j.pop:<4} gens {j.gens:<5} fe {j.budget:<8} {st}{note}")
        print(f"\n{len(jobs)} jobs -> {root}", file=sys.stderr)
        return 0
    for flag, metric in (("--compare", args.compare), ("--ranks", args.ranks),
                         ("--gap", args.gap), ("--zero-share", args.zero_share),
                         ("--ecdf", args.ecdf)):
        if metric is not None and not is_table_name(metric):
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
            stats = [m for m in bad if m in RUN_STATS + DE_STATS + SA_STATS]
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
    if args.pooled_reference:
        counts = pooled_reference(root, workers=args.workers, problems=only_problems)
        print(f"measured against the campaign's own reference (_pooled, {root / POOLED_DIR}): "
              f"{counts}", file=sys.stderr)
        return 0 if counts.get("failed", 0) == 0 else 2
    if args.recompute_reference:
        counts = recompute_reference(root, workers=args.workers, problems=only_problems)
        print(f"measured against the current reference: {counts}", file=sys.stderr)
        dense = recompute_dense(root, workers=args.workers, problems=only_problems)
        print(f"measured against the dense reference (_ref_v3): {dense}", file=sys.stderr)
        return 0 if counts.get("failed", 0) == 0 and dense.get("failed", 0) == 0 else 2
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
    if args.oracle:
        from . import oracle as O
        print(O.run(root, n_ref=spec.n_ref, workers=args.workers))
        return 0
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
    if args.invariance_check:
        from . import invariance as I
        print(I.run_campaign(cfg, spec, root, seeds=args.inv_seeds, workers=args.workers,
                             problems=only_problems))
        return 0
    if args.portfolio or args.instance_space:
        try:
            taus = tuple(float(v) for v in args.cover_taus.split(",") if v.strip())
            floor_taus = tuple(float(v) for v in args.cover_floor_taus.split(",") if v.strip())
            budgets = (tuple(int(v) for v in args.cover_budgets.split(",") if v.strip())
                       if args.cover_budgets else
                       tuple(sorted(set(spec.ladder) | {spec.budget_fe})) if spec.budget_fe
                       else None)
        except ValueError:
            print("--cover-taus and --cover-floor-taus want numbers and --cover-budgets "
                  "integers, comma-separated", file=sys.stderr)
            return 1
        if len(floor_taus) != len(taus):
            print(f"--cover-floor-taus gives {len(floor_taus)} levels and --cover-taus "
                  f"{len(taus)}: level k pairs the two k-th values", file=sys.stderr)
            return 1
        if args.portfolio:
            from . import portfolio as P
            print(P.run(root, taus=taus, floor_taus=floor_taus, budgets=budgets,
                        workers=args.workers, scenario=args.scenario, n_ref=spec.n_ref,
                        keep_caveats=args.cover_keep_caveats))
        if args.instance_space:
            from . import instance_space as S
            print(S.run(root, taus=taus, floor_taus=floor_taus, budgets=budgets,
                        min_seeds=args.cover_seeds, workers=args.workers,
                        scenario=args.scenario, n_ref=spec.n_ref,
                        keep_caveats=args.cover_keep_caveats,
                        calibration=args.is_calibration, weight=args.is_weight,
                        bbob_features=args.is_bbob_features))
        return 0
    if args.behaviour:
        if args.scenario == "archive":
            print("--behaviour reads the population's trajectories: drop --scenario archive",
                  file=sys.stderr)
            return 1
        from . import behaviour as B
        print(B.run(root, workers=args.workers, problems=only_problems))
        return 0
    rows = None
    if args.bias or args.cec2019_ranks or set_tables or any(v is not None for v in (
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
                    alpha=args.alpha, adjust=args.adjust)))
                print()
            if args.ranks and args.by:
                empty: dict = {}
                print(R.format_property_ranks(R.property_ranks(
                    rows, args.ranks, args.by, at=args.at, scenario=args.scenario, empty=empty),
                    args.ranks, args.by, empty))
            elif args.ranks:
                table = rank_table(rows, args.ranks, args.at)
                print(format_ranks(table))
                if args.ci:
                    empty: dict = {}
                    ci = R.rank_intervals(rows, args.ranks, at=args.at, scenario=args.scenario,
                                          empty=empty)
                    print("\n95 % bootstrap interval of each mean rank (problems resampled "
                          "2000 times)")
                    for alg, _ in table["algorithms"]:
                        if alg in ci:
                            lo, hi = ci[alg]
                            k = empty.get(alg)
                            print(f"  {R.mark_budget(alg)[:16]:<16} [{lo:5.2f}, {hi:5.2f}]"
                                  + (f"  {k} empty, ranked worst" if k else ""))
            if args.cec2019_ranks:
                print(R.format_cec2019_ranks(R.cec2019_ranks(rows, at=args.at,
                                                             scenario=args.scenario)))
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
        print(f"median {args.compare}{at} ({args.scenario}), runs per cell in brackets; "
              f"~ the median answer has fewer than N/2 points")
        sizes: dict = {}
        empty: dict = {}
        table = compare_table(rows, args.compare, args.at, sizes, empty)
        algs = sorted({a for d in table.values() for a in d})
        print("problem".ljust(16) + "".join((a[:13] + budget_note(a)).rjust(15) for a in algs))
        for prob in sorted(table):
            cells = []
            for a in algs:
                if a in table[prob]:
                    med, q1, q3, n = table[prob][a]
                    mark = "~" if is_short(*sizes.get((prob, a), (None, None))) else ""
                    k = empty.get((prob, a))
                    runs = f"{n}, {k} empty" if k else f"{n}"
                    cells.append(f"{med:.4g} ({runs}){mark}".rjust(15))
                else:
                    cells.append("-".rjust(15))
            print(prob.ljust(16) + "".join(cells))
        if empty:
            print(f"empty: runs without a feasible point in the answer, {sum(empty.values())} "
                  f"in all; counted as the worst value (inf) in the median")
        return 0
    if args.emit_slurm:
        script = emit_slurm(cfg, args.emit_slurm, time_limit=args.time,
                            partition=args.partition, python=args.python)
        print(f"wrote {script} and {script.parent / 'jobs.txt'} ({len(jobs)} jobs)")
        return 0

    if args.merge:
        try:
            got = merge_parts(cfg, args.merge)
        except ValueError as e:
            print(str(e), file=sys.stderr)
            return 1
        print(f"merged {got['runs']} runs into {out_root(cfg, campaign_spec(cfg))}: "
              + ", ".join(f"{n} from {q}" for q, n in got["from"].items()))
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
        only = [j.index for j in jobs if shard is None or shard_of(j, shard[1]) == shard[0]]
        shard = None
    from .refs import MissingReferences
    try:
        counts = run_campaign(cfg, shard=shard, only=only, workers=args.workers,
                              force=args.force, build_refs=not args.no_build_refs)
    except MissingReferences as e:
        print(e, file=sys.stderr)
        return 2
    return 0 if counts.get("failed", 0) == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
