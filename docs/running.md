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
| `--problems` | list the 324 benchmark problems (needs NumPy) |
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

ZDT, DTLZ, WFG, MaF, ZCAT, the Ishibuchi polygon family, MOP and BT: 324
problems across 13 families, each with bounds, an evaluator, the reference
point a hypervolume needs and, where a closed form exists, a sampler of the
true Pareto front. Objective counts run from 2 to 15. Point a config at them
instead of an external program:

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

**One caveat, and it is reported at run time too.** DTLZ5, DTLZ6, MaF6 and
WFG3 were designed with a degenerate, curve-shaped Pareto front, and their
reference fronts here are exactly that curve. Their true fronts also have a
non-degenerate part — from 4 objectives for DTLZ5, DTLZ6 and MaF6, and already
from 3 for WFG3 (Ishibuchi, Masuda & Nojima, IEEE TEC 20(5), 2016). At those
objective counts the reference set is a proper subset of the true front, so IGD
and IGD+ on them compare algorithms run here against each other but do not
compare with published numbers. `mootation.benchmarks.get()` warns once per
affected problem, and `degenerate_subset_note(name)` returns the caveat for any
name. Every other reference front in the registry is exact.
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

The hypervolume is exact up to five objectives and Monte-Carlo above, and exact
is not cheap at five: a well-spread set of 126 points takes two to three
seconds per call, so recording it along a trajectory is most of a campaign's
time. That is what `final_metrics` is for — `metrics = ["igdp"]` with
`final_metrics = ["igdp", "eps", "hv", "hv_h"]` pays for the hypervolume once
per run. The other indicators cost hundredths of a second.

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
runs every algorithm on 53 problems — ZDT at two objectives; DTLZ1–7, WFG1–9,
the inverted IDTLZ1–2, the scaled SDTLZ1–2 and shiftDTLZ1–4 (DTLZ1–4 with the
optimum moved off the centre of the box) at three and five — five seeds each at
10 000 evaluations: 15 370 jobs and about 43 CPU-hours, some three hours on a
16-core machine. The file records what that estimate is built from, which
three algorithms are most of it, and a preset for the papers' own budgets.

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
