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
| `--problems` | list the 216 benchmark problems (needs NumPy) |
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

ZDT, DTLZ, WFG, MaF, the Ishibuchi polygon family, MOP and BT: 216 problems
across 11 families, each with bounds, an evaluator, the reference point a
hypervolume needs and, where a closed form exists, a sampler of the true
Pareto front. Objective counts run from 2 to 15. Point a config at them instead
of an external program:

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
```

Each run writes `trajectory.jsonl` (IGD, IGD+ and hypervolume against the
evaluation count), `meta.json` and `final.csv`; a campaign is resumable, and
`pop = 0` / `gens = 0` take each problem's own published budget.
`mootation.run.metrics` carries the indicators: IGD, IGD+ and an exact WFG
hypervolume up to five objectives, Monte-Carlo above that.

## The terminal interface

`python -m mootation.run --tui run.toml` opens four screens over the same
functions: the resolved config with its `--check` verdict, the problem registry
with a filter, the selected algorithms with each one's population objection,
and a monitor that reads the journal live. A built-in campaign adds three more:
progress per problem × algorithm, a comparison table of medians with quartiles
(exported to CSV with `e`), and a per-run trajectory plot. The interface is
read-only on purpose: you edit the config in your own editor, because a
configuration assembled by clicking cannot be diffed, copied to a cluster or
attached to a paper.
