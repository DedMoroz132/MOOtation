# SPDX-License-Identifier: Apache-2.0
"""A campaign: every selected algorithm on every selected benchmark, several
seeds each, with a convergence trajectory recorded per run.

    python -m mootation.run.campaign camp.toml --list
    python -m mootation.run.campaign camp.toml                   # run all jobs here
    python -m mootation.run.campaign camp.toml --workers 8       # local pool
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
    budget_fe    = 0                  # 0 -> the problem's own pop*gens; else gens = ceil(budget/pop)
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

    @property
    def rel_dir(self) -> Path:
        return Path(self.problem) / self.algorithm / f"run_{self.seed}"


@dataclass
class CampaignSpec:
    out: str = "results/{name}"
    budget_fe: int = 0
    record_every: int = 1
    metrics: tuple = METRICS_DEFAULT
    n_ref: int = 1000
    seeds: tuple = ()


# ── reading the [campaign] table ────────────────────────────────────────────


def campaign_spec(cfg: Config) -> CampaignSpec:
    raw = dict(cfg.campaign or {})
    spec = CampaignSpec()
    known = {"out", "budget_fe", "record_every", "metrics", "n_ref", "seeds"}
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ConfigError("campaign", f"unknown key(s): {', '.join(unknown)}. "
                                      f"Known: {', '.join(sorted(known))}")
    spec.out = str(raw.get("out", spec.out))
    spec.budget_fe = int(raw.get("budget_fe", 0))
    spec.record_every = int(raw.get("record_every", 1))
    metrics = tuple(raw.get("metrics", METRICS_DEFAULT))
    for m in metrics:
        if m not in ("igd", "igdp", "hv"):
            raise ConfigError("campaign.metrics", f"unknown metric '{m}'; known: igd, igdp, hv")
    spec.metrics = metrics
    spec.n_ref = int(raw.get("n_ref", 1000))
    spec.seeds = tuple(int(s) for s in raw.get("seeds", ()))
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
            if spec.budget_fe > 0:
                gens = max(1, math.ceil(spec.budget_fe / pop))
            else:
                gens = a.gens if a.gens > 0 else p.n_gen
            for s in seeds:
                jobs.append(Job(index=idx, problem=pname, algorithm=a.name, seed=s,
                                pop=pop, gens=gens, params=dict(a.params),
                                n_objs=p.n_obj, pop_note=note))
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
    if any(m in spec.metrics for m in ("igd", "igdp")) and callable(p.pareto_front):
        ref = np.asarray(p.pareto_front(spec.n_ref), float)
    ideal, nadir = p.ideal, p.nadir

    meta = {
        "status": "running", "problem": job.problem, "algorithm": job.algorithm,
        "seed": job.seed, "pop": job.pop, "gens": job.gens, "budget_fe": job.pop * job.gens,
        "params": job.params, "n_objs": p.n_obj, "n_vars": p.n_vars,
        "pop_note": job.pop_note, "metrics": list(spec.metrics),
        "record_every": spec.record_every, "n_ref": spec.n_ref,
        "has_reference_front": ref is not None,
        "mootation": __version__,
        "host": socket.gethostname(), "platform": platform.platform(),
        "python": sys.version.split()[0], "pid": os.getpid(),
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    _write_json(meta_path, meta)

    fe = 0
    def evaluate(x):
        nonlocal fe
        fe += 1
        return p.evaluate(x)

    t0 = time.perf_counter()
    fh = traj_path.open("a", encoding="utf-8")
    n_records = 0

    def on_gen(gen, objectives):
        nonlocal n_records
        F = np.asarray(objectives, float)
        rec = {"gen": gen, "fe": fe, "t": round(time.perf_counter() - t0, 3), "n": int(len(F))}
        finite = bool(np.all(np.isfinite(F))) if F.size else True
        rec["finite"] = finite
        if finite and F.size:
            rec.update(M.compute(F, ref_front=ref, ideal=ideal, nadir=nadir, which=spec.metrics))
        fh.write(json.dumps(rec, separators=(",", ":")) + "\n")
        fh.flush()
        n_records += 1

    try:
        res = minimize(
            evaluate, bounds=p.bounds, n_objs=p.n_obj,
            algorithm=job.algorithm, pop_size=job.pop, n_gen=job.gens,
            seed=job.seed,
            constraints=(p.constraints if p.has_cons else None),
            on_generation=on_gen, record_every=spec.record_every,
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
            print(f"[{job.index}] {job.problem} {job.algorithm} seed {job.seed}: FAILED {e}",
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

    final = {}
    if F.size and np.all(np.isfinite(F)):
        final = M.compute(F, ref_front=ref, ideal=ideal, nadir=nadir, which=spec.metrics)
    meta.update(status="done", fe=fe, records=n_records, final=final,
                ignored_knobs=list(res.ignored), active_n=int(res.active_n),
                finished=time.strftime("%Y-%m-%dT%H:%M:%S"),
                seconds=round(time.perf_counter() - t0, 3))
    _write_json(meta_path, meta)
    if not quiet:
        summary = ", ".join(f"{k}={v:.4g}" for k, v in final.items() if isinstance(v, float))
        print(f"[{job.index}] {job.problem} {job.algorithm} seed {job.seed}: done "
              f"fe={fe} {summary}", file=sys.stderr)
    return "done"


def _write_json(path: Path, obj: dict) -> None:
    tmp = path.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=1)
    os.replace(tmp, path)


# ── the whole campaign ──────────────────────────────────────────────────────


def _pool_worker(args):
    cfg_path, index, force = args
    cfg = load(cfg_path)
    validate(cfg)
    spec = campaign_spec(cfg)
    jobs = expand_jobs(cfg, spec)
    return index, run_job(jobs[index], out_root(cfg, spec), spec, force=force)


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
        "algorithms": [a.name for a in cfg.algorithms],
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
    if workers > 1 and len(todo) > 1:
        import multiprocessing as mp
        cfg_path = str(cfg.source_path)
        with mp.Pool(processes=workers) as pool:
            for _, status in pool.imap_unordered(
                    _pool_worker, [(cfg_path, j.index, force) for j in todo]):
                counts[status] = counts.get(status, 0) + 1
    else:
        for j in todo:
            status = run_job(j, root, spec, force=force)
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
            "fe": m.get("fe"), "n_objs": m.get("n_objs"), "dir": meta.parent,
        })
    return rows


def compare_table(rows: list[dict], metric: str = "igd") -> dict:
    """{problem: {algorithm: (median, q1, q3, n)}} over finished runs."""
    import statistics
    table: dict = {}
    grouped: dict = {}
    for r in rows:
        if r["status"] != "done":
            continue
        v = r["final"].get(metric)
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


def rank_table(rows: list[dict], metric: str = "igd") -> dict:
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
    table = compare_table(rows, metric)
    lower_better = metric != "hv"
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
    return {"metric": metric, "lower_better": lower_better, "n_problems": len(table),
            "groups": groups, "algorithms": out}


def format_ranks(ranks: dict) -> str:
    """The rank table as fixed-width text, for the terminal."""
    groups = ranks["groups"]
    lines = [f"mean rank by median {ranks['metric']} over {ranks['n_problems']} problem(s): "
             f"1 = best, ties share the average rank; probs = problems ranked on",
             f"{'algorithm':<14}{'mean':>7}{'wins':>6}{'probs':>6}"
             + "".join(f"{g[:9]:>10}" for g in groups)]
    for alg, e in ranks["algorithms"]:
        mean, n = e["all"]
        line = f"{alg[:14]:<14}{mean:>7.2f}{e['wins']:>6}{n:>6}"
        for g in groups:
            line += f"{e[g][0]:>10.2f}" if g in e else f"{'-':>10}"
        lines.append(line)
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


# ── CLI ─────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="mootation.run.campaign",
                                 description="Run a benchmark campaign described in TOML.")
    ap.add_argument("config")
    ap.add_argument("--list", action="store_true", help="print the job list and exit")
    ap.add_argument("--shard", help="i/n: run the jobs with index %% n == i")
    ap.add_argument("--job", type=int, action="append", help="run one job by index (repeatable)")
    ap.add_argument("--workers", type=int, default=1, help="local process pool size")
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

    if args.list:
        for j in jobs:
            st = job_status(root, j)
            note = f"  [{j.pop_note}]" if j.pop_note else ""
            print(f"{j.index:5d} {j.problem:<14} {j.algorithm:<14} seed {j.seed:<3} "
                  f"pop {j.pop:<4} gens {j.gens:<5} fe {j.pop * j.gens:<8} {st}{note}")
        print(f"\n{len(jobs)} jobs -> {root}", file=sys.stderr)
        return 0
    for flag, metric in (("--compare", args.compare), ("--ranks", args.ranks)):
        if metric is not None and metric not in ("igd", "igdp", "hv"):
            print(f"{flag}: unknown metric '{metric}'; known: igd, igdp, hv", file=sys.stderr)
            return 1
    if args.ranks:
        print(format_ranks(rank_table(scan_results(root), args.ranks)))
        return 0
    if args.compare:
        table = compare_table(scan_results(root), args.compare)
        algs = sorted({a for d in table.values() for a in d})
        print("problem".ljust(16) + "".join(a[:14].rjust(15) for a in algs))
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
    counts = run_campaign(cfg, shard=shard, only=args.job, workers=args.workers,
                          force=args.force)
    return 0 if counts.get("failed", 0) == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
