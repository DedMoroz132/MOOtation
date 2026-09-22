<!-- SPDX-License-Identifier: Apache-2.0 -->
# Running from a config file: `mootation.run`

`mootation.run` drives the optimizer from a TOML description of the run. It is
for the case where objective values come from *programs* rather than from a
function: a mesher, a solver, a measurement rig. It is pure Python with no
dependencies (`tomllib` is in the standard library since 3.11); only the
built-in benchmark suites need NumPy and only the terminal interface needs
Textual.

```bash
python -m mootation.run --show --check python/examples/demo.toml   # inspect, validate
python -m mootation.run                python/examples/demo.toml   # run it
python -m mootation.run --tui          python/examples/demo.toml   # watch it
```

| flag | does |
|---|---|
| `--check` | validate and exit 1 if the run cannot start; every complaint at once |
| `--show` | print the configuration as it resolved (paths, platform-specific steps) |
| `--algorithms` | list the 58 algorithm names |
| `--problems` | list the 434 benchmark problems (needs NumPy) |
| `--tui` | the terminal interface (needs Textual) |
| `--campaign` | run the benchmark campaign the file describes; sharding flags live in `python -m mootation.run.campaign --help` |

The two shipped examples: [`python/examples/demo.toml`](../python/examples/demo.toml)
passes `--check` on any machine with Python (its "solver" is a script next to
it); [`python/examples/airfoil.toml`](../python/examples/airfoil.toml) is the
real-world shape (gmsh, a solver binary, a CSV) and **deliberately fails**
`--check`, so you can see what the report looks like.

## The file

Every relative path resolves against the directory holding the file, and steps
run with that directory as their working directory. Move the directory, copy it
to a cluster, attach it to a paper: nothing inside needs editing.

```toml
[run]
name    = "airfoil"
scratch = "scratch/worker_{worker:02d}"     # per worker, overwritten
ledger  = "results/{name}/evaluations.jsonl" # per run, append-only
workers = 12
on_fail = "penalty"                          # penalty | skip | abort
penalty = 1e8
resume  = true

# [run.warm_start]                           # optional
# from             = "results/previous/final_population.jsonl"
# on_size_mismatch = "truncate"              # truncate | pad | error

[problem]
kind   = "external"                          # external | builtin
n_vars = 12
n_objs = 3
bounds = [[0.0, 1.0], [0.0, 1.0], ...]       # one [lower, upper] per variable

[problem.input]
template = "templates/geom.geo.in"           # {x[0]}, {x[1]}, ... are substituted
write_to = "{scratch}/geom.geo"

[[problem.steps]]
name    = "mesh"
run     = ["gmsh", "-3", "{scratch}/geom.geo", "-o", "{scratch}/mesh.msh"]
timeout = 300

[[problem.steps]]
name        = "solve"
run         = ["solver", "--mesh", "{scratch}/mesh.msh", "--out", "{scratch}/out.csv"]
run_windows = ["solver.exe", "--mesh", "{scratch}/mesh.msh", "--out", "{scratch}/out.csv"]
timeout     = 3600

[problem.output]
parser      = "csv"                          # csv | json | regex | columns | python:module.function
from        = "{scratch}/out.csv"
objectives  = ["mass", "drag", "deflection"]
constraints = ["stress_max"]                 # <= 0 means satisfied

[[algorithms]]
name   = "nsga2"
pop    = 92
gens   = 500
params = { eta_c = 20, pc = 0.9 }            # any knob from `--algorithms`
```

Commands are argv **arrays**: nothing to quote, nothing to escape, no
difference between `cmd` and `sh`. `run_windows` / `run_linux` / `run_darwin`
override one step on one platform; `run` is the fallback.

### Parsers

From clean formats to dirty ones: `csv` by column name; `json` by dotted path;
`regex` for solver logs, one pattern per named value, `take = "last"` because
solvers print the value every iteration; `columns` positionally, from a given
line; `python:module.function` for HDF5, VTU and anything else that cannot be
described declaratively. **A parse failure is an evaluation failure**: nothing
is substituted silently, because one fabricated point poisons the archive and
every metric taken from it. `on_fail` decides what happens instead: `penalty`
(the value in `penalty`), `skip` or `abort`.

### Scratch and journal

Two stores, deliberately separate. Scratch is per worker and overwritten on
every evaluation (meshes, VTU files, gigabytes). The journal is per run and
append-only: one JSON line per evaluation with the generation, index, decision
vector, objectives, constraint values, wall time, status and a hash of the
quantised decision vector, about 300 bytes. `resume = true` restarts a run off
the journal, and the journal doubles as a cache keyed by that hash. Knowing
that *x* maps to *f* does not require the mesh that produced it, and
evolutionary algorithms re-propose identical individuals more often than one
expects.

### What `--check` verifies

Before anything expensive starts: unknown keys (an error, never a silent
no-op); every step's executable exists on this platform, on PATH or as a file;
every `{x[i]}` in the input template stays within `n_vars`; `bounds` are
ordered and there is exactly one pair per variable; every algorithm name is in
`include/mootation/algorithms.def`; every `params` key is a knob the library
knows; every `pop` satisfies its algorithm's own constraint (an exact
Das–Dennis lattice size for NSGA-III and the MOEA/D family, `pop` divisible by
`K` for the M2M family), and for built-in sweeps that check is repeated for
every objective count the selection contains; a warm-start file exists and its
objective count matches. Sample report:

```
3 problem(s):

  - problem.bounds: has 2 entries but n_vars = 3; there must be exactly one [lower, upper] pair per variable
  - problem.steps.solve: executable not found on PATH and not an existing file: 'solver' (platform: linux)
  - algorithms[0] (nsga3): pop = 92 is not a Das-Dennis lattice size for n_objs = 3; nsga3 requires an exact lattice. Nearest attainable: 91, 105
```

That last one is a real trap: the NSGA-III paper's own Table I uses N = 92,
the nearest multiple of four above the lattice size 91, and this library wants
the lattice size exactly. Ten seconds here instead of ten minutes into a
day-long run.

## The built-in suites

ZDT, DTLZ, WFG, MaF, ZCAT, bbob-biobj, the Ishibuchi polygon family, MOP and
BT: 434 problems across 14 families, each with bounds, an evaluator, the
reference point a hypervolume needs and, where a closed form exists, a sampler
of the true Pareto front. Objective counts run from 2 to 15. Point a config at
them instead of an external program:

```toml
[run]
name = "suite"

[problem]
kind = "builtin"

[benchmarks]
families   = ["DTLZ", "WFG"]
objectives = [3, 5, 10]
runs       = 31
# problems = ["DTLZ2_3D", "WFG4_5D", "MaF3_8D"]   # ... or name them outright

[[algorithms]]
name = "nsga2"
pop  = 100
gens = 500
```

`--check` expands the cross product against the real registry and names what
does not exist, with near misses; the names are `DTLZ2_3D`, not `DTLZ2_M3`.

**Four problems use their FULL fronts, and three of them have very long ones.**
DTLZ5, DTLZ6, MaF6 and WFG3 were designed with a degenerate, curve-shaped
Pareto front, but the true fronts of DTLZ5 and DTLZ6 from 4 objectives and of
WFG3 from 3 also have a non-degenerate part (Ishibuchi, Masuda & Nojima, IEEE
TEC 20(5), 2016), and MaF6's from 7. At those sizes the reference front is now
the full one, built by `mootation.benchmarks.fronts_full` in the problem's
reduced coordinates and kept only if every point survives an independent
sample of 10^6, a local search for a dominating point and an exact test
against every attainable point (`python -m mootation.benchmarks.make_fronts`
rebuilds them). Things to know before reading a number on them:

- **The DTLZ5 and DTLZ6 fronts run to the largest g the box allows.** With
  f_M = 0, the smallest f_2 a point can have is (1 + g) sin²(π / (4(1 + g))),
  which falls as g grows, so the front's far end sits at the bound of g — 2.5
  for DTLZ5, 10 for DTLZ6 with their ten distance variables. The nadir in the
  objectives before the last is about 3.4 on DTLZ5 and 11 on DTLZ6 at four and
  five objectives, where the curve never exceeded 1; f_M itself stays at most 1,
  since the curve dominates every point above that. Every indicator
  normalised by the nadir (`hv`, `hv_h`, `igdp_norm`, `eps_norm`) changed
  accordingly, and the two problems no longer share a front.
- **WFG3's first objective reaches 3** at three objectives, where the curve
  stopped at 1: the paper's counterexample (3, 1, 1) is on the front.
- **MaF6 leaves the curve only from 7 objectives.** Its objectives carry
  (1 + 100 g), which makes leaving the curve pay only where an objective
  multiplies five cosines or more; at 5 the build finds no point off it, so
  MaF6_5D keeps the curve. At 8 its front, too, runs to the bound of g, with
  f_7 up to 245, and a quarter of it is off the curve at 10.
- **Runs scored against the old curve are not comparable** with runs scored
  against the full fronts on these problems: `--recompute` updates their final
  indicators from `final.csv`, but a trajectory has to be run again.

`mootation.benchmarks.get()` still warns, once per problem, for any size that
has no shipped full front — WFG3 at six and ten, DTLZ5 and DTLZ6 at ten and
fifteen, MaF6 at fifteen — and
`degenerate_subset_note(name)` returns that caveat. Every other reference front in the registry is exact. Where one is
sampled as the nondominated images of a grid of positions — DTLZ7 and MaF7,
WFG1, WFG2 and MaF11, ZDT3, ZCAT — every point is also checked against a
fresh sample and its own neighbours: until 2026-09-22 such fronts kept points
that merely no other grid point beat, up to a third of WFG2_3D's.
[`python/examples/bench.toml`](../python/examples/bench.toml) is the complete
example. The same suites are importable directly:

```python
import mootation.benchmarks as bench
res = bench.solve("DTLZ2_3D", "nsga3", pop_size=91, n_gen=200)
print(bench.igd("DTLZ2_3D", res.objectives))
```

## Campaigns

A campaign is every selected algorithm on every selected problem, several seeds
each, with a convergence trajectory per run. Add a `[campaign]` section
([`python/examples/campaign.toml`](../python/examples/campaign.toml) is the
template) and run it with `--campaign`, or through the module directly for the
cluster flags:

```bash
python -m mootation.run.campaign campaign.toml --list          # the job list
python -m mootation.run.campaign campaign.toml --workers 4     # local process pool
python -m mootation.run.campaign campaign.toml --shard 3/40    # one shard of forty
python -m mootation.run.campaign campaign.toml --emit-slurm 40 # writes submit.sh + jobs.txt
python -m mootation.run.campaign campaign.toml --compare igd   # median table
python -m mootation.run.campaign campaign.toml --ranks igd     # mean rank per algorithm
```

Each run writes `trajectory.jsonl` (IGD, IGD+ and hypervolume against the
evaluation count), `meta.json` and `final.csv`; a campaign is resumable, and
`pop = 0` / `gens = 0` take each problem's own published budget. Whatever the
budget, a run stops when it has spent that many evaluations, not after that
many steps: a step is a generation for most algorithms, but one offspring for
NIMMO and a fifth of the population for MOEA/D-DRA and MOEA/D-AWA, and a budget
in generations left those three with 2 % and 21 % of everyone else's
evaluations. The count each run actually spent is `fe` in its `meta.json`.
`mootation.run.metrics` carries the indicators. `metrics` names the ones
recorded along every trajectory, `final_metrics` the ones computed once on the
final population (the same list when omitted):

| name | what it measures | better |
|---|---|---|
| `igd` | mean distance from the reference front to the nearest point; not Pareto-compliant, so for comparison with published numbers rather than for ranking | lower |
| `igdp` | IGD+: only the dominated part of each offset counts | lower |
| `igdp_norm` | IGD+ with objectives and front divided by nadir − ideal: scale-free, for SDTLZ and anything in mixed units | lower |
| `eps`, `eps_norm` | additive ε: the smallest shift in every objective that makes the set weakly dominate the reference front, i.e. no worse than the front by more than ε anywhere; raw and normalised | lower |
| `hv` | hypervolume, objectives normalised by ideal and nadir, reference point 1.1 | higher |
| `hv_h` | hypervolume with the reference point at 1 + 1/H, H from the problem's default population (1.0101, 1.0833 and 1.2 at 2, 3 and 5 objectives), which evens out the contributions of a uniformly spread set (Ishibuchi, Imada, Setoguchi & Nojima, GECCO 2017) | higher |
| `gdp` | GD+: IGD+'s distance averaged over the set instead of the front — convergence only; beside IGD+ it tells "still converging" (both fall) from "converged, now spreading" (GD+ flat, IGD+ falling) | lower |
| `roi_dist` | distance from the set to the box [ideal, nadir] in normalised units, 0 once any point is inside; tells apart the runs whose hypervolume is exactly 0 (COCO scores bbob-biobj this way) | lower |
| `range_cover` | the worst objective's covered share of [ideal, nadir]; falls early when a population collapses onto part of the front, as on DTLZ4; per objective in `range_cover_each` | higher |
| `nd_share`, `dup_share` | the share of the set no other member dominates, and the share that repeats an objective vector already in it: stagnation and duplicates | — |

`gdp` needs a reference front like `igd`; `roi_dist` and `range_cover` need only
the problem's ideal and nadir, and `nd_share` and `dup_share` nothing at all, so
those four also describe the bbob-biobj problems, which have no front.

The hypervolume is exact up to five objectives and Monte-Carlo above, and exact
is not cheap at five: a well-spread set of 126 points takes two to three
seconds per call, so recording it along a trajectory is most of a campaign's
time. That is what `final_metrics` is for — `metrics = ["igdp"]` with
`final_metrics = ["igdp", "eps", "hv", "hv_h"]` pays for the hypervolume once
per run — and `trajectory_hv_max_m = 3` keeps it on the trajectory only where
it is cheap (about 50 ms per point at three objectives), recording `null`
above. The other indicators cost milliseconds.

**Where the trajectory is recorded.** `record_every = k` records every k
generations. `record_grid = "log"` records instead at fixed evaluation counts,
round(10^(j / `record_per_decade`)), 10 per decade by default: 100, 126, 158,
200, … The counts are absolute, so a 10 000- and a 25 000-evaluation run share
every point up to 10 000 and a budget ladder compares point by point, without
interpolating; a point is the first generation at or past its count.

**What else a run keeps.**
- `meta.json` records `revision`: the git commit and whether tracked files
  differed from it, for the source tree and, separately, for the compiled
  extension — a pulled commit whose C++ was never rebuilt shows as a mismatch.
  The version string alone never changes between commits.
- `archive.csv` holds every nondominated point the run evaluated, whatever the
  algorithm kept: at most one per cell of a grid in normalised objectives
  (step 1e-3, or 1e-2 from five objectives; `archive_delta`), with each
  objective's best point kept outside the grid so the ends of the front are
  never pruned. `archive = false` turns it off. `meta.json` says the step and
  the normalisation used.
- `snapshots = true`, or a list of problem names, stores the population's
  objectives at every trajectory point in `snapshots.npz` (float32), to see how
  the front's shape moved or to compute an indicator the run did not record.

**Baselines.** `random_search` and `sobol_search` (scrambled Sobol, needs
SciPy) go in the algorithm list like any core. They sample the box blindly,
keep the run archive, and answer with the problem's population size chosen
from it by distance-based subset selection on the IGD+ distance — the same
selection that thins every other point set here — so their indicators compare
with a population's. They spend the budget exactly. An algorithm that does not
clearly beat them on a problem says more about the problem or the budget than
about the algorithm.

Two more readings need no rerun. `--at 0.25` gives `--compare` and `--ranks`
every run as it stood at a quarter of its budget, read from its trajectory, so
ranks at several budgets come out of one campaign — they do differ with the
budget (Tanabe & Oyama, GECCO 2017). `--recompute eps,hv_h` computes indicators
a campaign did not record from each finished run's `final.csv` and stores them
in its `meta.json`; `--workers` spreads the work.

### Reading the results

`--compare` prints the median of a final indicator for every problem ×
algorithm pair. With four algorithms that is the whole story; with fifty-eight
it is a table nobody can read, and `--ranks` condenses it. On every problem the
algorithms are ranked by that median — 1 is best, equal medians share the
average rank — and each algorithm's ranks are averaged over all problems, over
each family and over each objective count, next to the number of problems it
won and the number it was ranked on.

Two cautions. A mean rank rewards consistency, not margin: an algorithm second
on every problem outranks one that alternates between first and last, and a
rank cannot tell whether two medians differ by more than the spread between
seeds (a Wilcoxon test is on the roadmap, not here yet). Use the ranks to find
where to look and the medians with their quartiles to decide. And a group mean
is only as broad as its group: best on WFG at five objectives means best on
those nine problems, at the budget the campaign gave them.

### Changing the number of workers while it runs

`--workers N` is only where a campaign starts. The runner re-reads
`<results>/_workers.txt` every 1.5 seconds: raise the number and workers start
at once, lower it and the surplus finish the job they are on and leave, write 0
and the campaign drains and stops with the unfinished jobs left for next time.
The TUI's Apply writes that file, and so can anything else:

```bash
echo 8 > results/all58/_workers.txt
```

`<results>/_runner.json` is the runner's heartbeat — workers alive and wanted,
jobs done, the jobs in flight — rewritten every two seconds, which is how the
TUI shows a campaign it did not start. A worker that dies inside an algorithm
fails the one job it holds, which is recorded, and is replaced.

### All 58 algorithms on another machine

[`python/examples/campaign_all.toml`](../python/examples/campaign_all.toml)
runs every algorithm and the two baselines on 203 problems — ZDT at two
objectives; DTLZ1–7, WFG1–9, the inverted IDTLZ1–2, the scaled SDTLZ1–2,
shiftDTLZ1–4 (DTLZ1–4 with the optimum moved off the centre of the box) and
ZCAT1–20 at three and five; and bbob-biobj F1–F55, every pair of ten bbob
functions, at 5 and 10 variables — five seeds each at 10 000 evaluations:
60 900 jobs and about 100 CPU-hours, some six hours on a 16-core machine. The
file records what that estimate is built from, which three algorithms are most
of it, where the time goes between the suites, and a preset for the papers'
own budgets. Its trajectories are on the logarithmic grid, so a second run of
the same file at 25 000 or 50 000 evaluations lines up with it point by point.

The bbob-biobj problems have no reference front — a pair of bbob functions has
no closed-form Pareto set — so `igd`, `igdp`, `gdp` and `eps` are recorded as
null for all 110 of them. `hv` and `hv_h` rank them, and `roi_dist`,
`range_cover`, `nd_share` and `dup_share` describe them. Use `--ranks hv` if you
want them counted; `--ranks igd` silently covers the other 93.

A clean machine needs Python 3.11 or newer and a C++17 compiler — on Windows,
the Visual Studio Build Tools with the "Desktop development with C++" workload.
pip fetches CMake if it is missing, and the extension compiles once, in a few
minutes.

```bash
git clone https://github.com/DedMoroz132/MOOtation.git
cd MOOtation
python -m pip install ".[all]"
cd python/examples
python -m mootation.run.campaign campaign_all.toml --list > jobs.txt
python -m mootation.run.campaign campaign_all.toml --workers 16
```

`--workers` is how many jobs run at once; each job is single-threaded, so use
one per core. While it runs, the Campaign tab of
`python -m mootation.run --tui campaign_all.toml` shows progress per problem ×
algorithm, and the results land in `python/examples/results/all58/`.

Across several machines, start the same file on each with its own shard:

```bash
python -m mootation.run.campaign campaign_all.toml --shard 0/3 --workers 16   # machine 1
python -m mootation.run.campaign campaign_all.toml --shard 1/3 --workers 16   # machine 2
python -m mootation.run.campaign campaign_all.toml --shard 2/3 --workers 16   # machine 3
```

then copy every machine's `results/all58/` into one directory. The trees do not
overlap — a job writes only its own `<problem>/<algorithm>/run_<seed>/` — and
Compare, `--compare` and `--ranks` read whatever `meta.json` files sit under the
results root, wherever they were produced. Use the same commit everywhere; each
`meta.json` records the library version, the host and the Python it ran under.
A killed shard resumes when started again, and `--force` reruns finished jobs.

```bash
python -m mootation.run.campaign campaign_all.toml --ranks igd
python -m mootation.run --tui campaign_all.toml       # Compare tab, then t
```

## The terminal interface

`python -m mootation.run --tui run.toml` opens four screens over the same
functions: the resolved config with its `--check` verdict, the problem registry
with a filter, the selected algorithms with each one's population objection,
and a monitor that reads the journal live. A built-in campaign adds three more
and opens on the first of them: a dashboard of the campaign with its controls,
a comparison table of medians with quartiles that `t` switches to the mean
ranks above (either view exported to CSV with `e`), and a per-run trajectory
plot.

The dashboard shows progress, speed and time left, the jobs in flight, every
algorithm grouped by family with its own progress bar, and every problem. It
reads the results in a background thread and never re-reads a finished job, so
it stays responsive on a campaign of thousands. Start (`s`) launches the
campaign with the worker count in the box, Stop (`x`) kills it — the jobs that
were running run again next time — and Apply, or Enter in the box, changes the
number of workers while it runs. A campaign started from the command line is
shown as well and can be resized the same way; Stop drains it instead of
killing it. Closing the app stops a campaign the app started.

The config itself stays read-only: you edit it in your own editor, because a
configuration assembled by clicking cannot be diffed, copied to a cluster or
attached to a paper.
