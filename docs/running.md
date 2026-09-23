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
| `--algorithms` | list the 59 algorithm names |
| `--problems` | list the 436 benchmark problems (needs NumPy) |
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

An `[[algorithms]]` entry can also carry `evaluations = 30000`, a budget in
evaluations that replaces `pop` × `gens` (and `gens` may then be left out): a
step is not a generation for every core — one offspring for NIMMO, a fifth of
the population for MOEA/D-DRA and -AWA — so a budget in steps is not a budget.
And `label`, which names a variant: two entries of one core with different
`params` must be told apart, and a campaign files and reports each under its
label. A parameter can be a switch:

```toml
[[algorithms]]
name   = "r2ibea"
label  = "r2ibea_norm"                      # results/.../r2ibea_norm/
pop    = 0
gens   = 0
params = { normalize = true }                # IBEA, R2-IBEA, Two_Arch2, MOEA/D-AM2M
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
BT, and two uninformative probes: 436 problems across 15 families, each
with bounds, an evaluator, the
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
has no shipped full front — WFG3 at six and ten, DTLZ5 at fifteen, DTLZ6 at
ten and fifteen, MaF6 at fifteen — and
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
python -m mootation.run.campaign campaign.toml --problems DTLZ5_5D,WFG3_5D --force  # rerun two
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
| `igdx` | IGD in the decision space: the mean distance from a Pareto-SET sample to the nearest solution, variables normalised by the bounds (Tanabe & Ishibuchi 2019, Eq. 5) | lower |
| `cr` | cover rate: how much of the Pareto set's extent in each variable the solutions span, geometric mean over the variables, 1 = all (Tanabe & Ishibuchi 2019, Eqs. 7–8) | higher |
| `pdist` | the mean distance between two solutions in normalised variables: spread in the decision space, no reference needed | — |
| `tau90` | the 0.9 quantile of the IGD+ distances d⁺(z, A) = min over a of ‖max(a − z, 0)‖ over the reference sample, normalised like `igdp_norm`: 90 % of the front lies within `tau90` of the set. `igdp_norm` is the mean of the same distances; one reference point far from everything moves their maximum and not this. Weakly Pareto-compliant, like IGD+. With it, `coverage_curve` gives the share of the sample within 0.01, 0.02, 0.05, 0.1 and 0.2 | lower |
| `n_final` | the number of points in the answer. `hv`, IGD+ and ε never get worse when points are added, so when two algorithms answer with sets of different sizes, read them next to this | — |

`gdp` needs a reference front like `igd`; `roi_dist` and `range_cover` need only
the problem's ideal and nadir, and `nd_share` and `dup_share` nothing at all, so
those four also describe the bbob-biobj problems, which have no front.

`igdx` and `cr` need a sample of the Pareto SET, which only some problems
have: the Ishibuchi polygons (Polygon, IPolygon), DTLZ1–4, shiftDTLZ1–4 and
ZCAT1–20. On shiftDTLZ the distance variables are **cyclic** — the optimum
moved off the centre wraps around the box — so their differences are taken as
((x − c + 0.5) mod 1) − 0.5 in normalised units; a plain difference would call
two solutions on either side of the wrap far apart. On ZCAT the set is the
positions with every distance variable at its optimum g(y_I). Elsewhere they
are null. `pdist` needs nothing but the solutions. All three read the final
population (`final_metrics`), whose variables `final.csv` keeps, so
`--recompute igdx,cr,pdist` adds them to a campaign that ran without them.

The hypervolume is exact up to `hv_exact_max_m` objectives (5 by default) and
a Monte-Carlo estimate from `hv_mc_samples` points (100 000) above. The exact
one is the WFG algorithm (While, Bradstreet & Barone 2012) compiled into the
extension, `mootation._core.hypervolume`: about 20 ms for 210 points at five
objectives, where the Python recursion it replaces took seconds; without the
extension the Python one still runs. The Monte-Carlo points are drawn by
NumPy from a fixed seed and only counted in C++, so an estimate does not
depend on which of the two counted it, and every record of a trajectory uses
the same points, which keeps the curve free of sampling jitter. `hv_method` in
each record says which one ran.

Along a trajectory, `trajectory_hv_max_m = 3` keeps the exact hypervolume to
three objectives; above it the hypervolume is `null`, or, with
`trajectory_hv_mc_samples = 10000`, estimated from that many points (about
±0.01 at 95 %, ~15 ms per record at five objectives). The final population
always gets the exact value up to `hv_exact_max_m`. `final_metrics` pays for
anything costly once per run: `metrics = ["igdp"]` with `final_metrics =
["igdp", "eps", "hv", "hv_h"]`. The other indicators cost milliseconds.

The same C++ is available to a C++ program as `mootation/hypervolume.hpp`:
`hypervolume::wfg(F, n, m, ref)` for the exact value and
`hypervolume::covered(F, n, m, S, k)` for the Monte-Carlo count.

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

Two ablations take one ingredient of an evolutionary algorithm away.
`random_selection_ea` is NSGA-II's variation — SBX with η_c = 20 and p_c =
0.9, polynomial mutation with η_m = 20 and p_m = 1/n, formula for formula the
library's operators — with none of its selection: parents drawn uniformly, and
N of the 2N parents and children kept uniformly at random. What is left is
what the operators do on their own; its answer is its last population. `gsemo`
is global SEMO (Laumanns, Thiele & Zitzler, IEEE TEVC 8(2), 2004, with Giel's
global mutation, CEC 2003) carried to real variables: the population is every
nondominated point found, each evaluation mutates one member drawn uniformly
(every variable with probability 1/n, polynomial mutation), and the child
enters unless a member weakly dominates it. Selection by dominance alone, no
diversity mechanism and no population size; its answer is the population
reduced to N by DSS. The carrying-over to real variables is ours: both papers
define the algorithm on bit strings. None of the four baselines handles
constraints; only the run archive keeps to feasible points.

**Bound repair.** An operator that can put a variable outside the box repairs
it, and how is part of the algorithm: Kononova et al. (Evol. Comput. 32(1),
2024) show that results are not reproducible without saying which, and
Kudela, van Stein, Bäck & Kononova (GECCO 2026) trace a systematic pull
towards the bounds in 120 PlatEMO algorithms to its default clamp. The knob
`bound_repair` takes `clip`, `reflect` (mirrored at the violated bound, again
while outside), `random` (uniform in the box), `midpoint` (halfway between the
parent's value and the bound), `resample` (the operator draws that variable
again, at most ten times, then clips), `wrap` (the box as a torus) or `native`
(the rule the operator's own paper gives). Nine algorithms have an operator
that can leave the box, and each keeps its own default: `moead_de`,
`moead_dra` and `lis_lcs` (DE) `random`, the rule MOEA/D-DE's paper gives;
`dcea`, `hlmea` and `naemo` `clip`, as their code always did; `liu_gu2011`,
`moead_m2m` and `moead_am2m` `native`, Liu & Li's own formula. DE refuses
`resample` and `native`, Liu & Li's crossover `resample` (one scalar per
offspring cannot redraw one variable). Every other core reports the knob as
ignored: SBX and polynomial mutation cannot leave the box. Whatever the
settings, every run's `meta.json` lists the operators it used with their
repair under `operators`.

**Switchable operators.** NSGA-II, IBEA-ε+, SPEA2+SDE and AGE-MOEA take
`crossover` and `mutation`, MOEA/D-DE `mutation` alone; the defaults are
each paper's SBX and polynomial mutation, called exactly as before. The
alternatives are for ablations — a variant in a campaign is an
`[[algorithms]]` entry with its own `label`:

```toml
[[algorithms]]
name   = "nsga2"
label  = "nsga2_gauss01"
pop    = 0
gens   = 0
params = { mutation = "gaussian", mutation_scale = 0.1, bound_repair = "reflect" }
```

| knob | values |
|---|---|
| `crossover` | `sbx`; `uniform` (every variable from one parent at random, the other child the complement — JEGA's shuffle_random for two parents); `blx_alpha` (each variable uniform in [min − αI, max + αI], I the parents' distance, `blx_alpha` = 0.5) |
| `mutation` | `polynomial`; `gaussian`, x + s(ub − lb)·N(0, 1) with `mutation_scale` s = 0.1; `cauchy`, the same with C(0, 1) and s = 0.05; `uniform_reset`, U(lb, ub); `mixture` (`mixture_cauchy`): the polynomial step with probability 1 − `mixture_q` (0.1), the gaussian (Cauchy) one otherwise. Every one mutates each variable with probability `pm`, 1/n by default |
| `bound_repair` | for the operators above that can leave the box (BLX-α, gaussian, Cauchy): `reflect` by default, provisionally, until experiment E3 picks one |
| `sbx_var_prob` | SBX's share of crossed variables, 0.5 (the canonical realcross) or anything in [0, 1], for this run only; any core using SBX |

In MOEA/D-DE, `polynomial` is its literal Eq. 7 and every other mutation
leaves its steps raw for Step 2.3's repair of the whole offspring, as Eq. 7
does. The operators live in `operators/real_crossover.hpp` and
`operators/real_mutation.hpp`, with DE/rand/2/bin and DE/best/1/bin (Storn &
Price's DE/x/y/z notation) in `operators/de_mutation.hpp`.

**Crowding in the decision space.** `crowding_space = "decision"` makes
NSGA-II compute its crowding distance over the variables instead of the
objectives: the same formula, each variable's gaps over its range on the
front. Among equally ranked solutions it keeps those far apart in the
PARAMETERS, so that different parameter sets with the same objective values
survive side by side. It is the idea of DN-NSGA-II (Liang, Yue & Qu, CEC
2016), whose paper is not in the corpus: the knob implements the description
of it, not the paper's letter. Default `objectives`, the paper's.

**DMS.** Direct MultiSearch (Custódio, Madeira, Vaz & Vicente, SIAM J.
Optim. 21(3), 2011) is the one algorithm here that is not evolutionary: a
list of nondominated points, each with its own step size, polled one at a
time along ±every coordinate (2n evaluations a poll). It is deterministic —
the seed changes nothing — and stops by itself when every step size is below
10⁻³ of the variable's range, which can be before the budget runs out. Its
answer is the list reduced to `pop` by DSS, so `pop` is the size of the
answer, not of a population. `dms_init` chooses the initial list: `line`, n
points on the box's diagonal (the paper's best variant, the default), or
`single`, the box's centre. A problem whose optimum sits at the centre of the
box (ZDT4's g) is solved by `single` at the first evaluation; see the
structural-bias campaign before reading anything into that.

**What the operators did.** `operator_stats = true` in `[campaign]` adds to
every trajectory record, for the offspring evaluated since the previous one,
and to `final` for the whole run:

| field | what it counts |
|---|---|
| `oob_share`, `oob_var_share` | offspring with a variable outside the box before repair, and the share of their variables that were (the measure of Kononova, Caraffini & Bäck, Inf. Sci. 581, 2021). Liu & Li's offspring pass two repair sites, crossover and mutation, and count at each |
| `survival_share` | offspring that entered the next population: it holds more copies of their variable vector than the parents did, so a clone of a surviving parent does not count and a child copied into several MOEA/D neighbourhoods counts once |
| `offspring_nd_share` | offspring that no member of the parent population dominates |
| `step_mean` | the distance from an offspring to the nearest member of the parent population, over the box diagonal ‖ub − lb‖ |

The relations are to the parent POPULATION, the one before the step: the
library does not track which individuals an offspring came from across its
59 cores, so "not dominated by its parents" is read as "by any parent", and
the nearest parent as the nearest member. The statistics cost milliseconds
and never change a run: the populations are the same with them on or off.
They are off by default and not computable afterwards, so `--recompute`
refuses them; `--compare survival_share` and the other tables read them.

**Structural bias.** Where does an algorithm put its population when the
objectives say nothing? On `uninformative_n02_2D` and `uninformative_n10_2D`
every objective value is a U(0, 1) draw that does not depend on x — drawn
from the run's seed and the evaluation's number, so a campaign run gets its
own stream (`BenchProblem.make_evaluator`) — and no region of the box is
better than another. A final population piled at the bounds or drawn to the
centre shows a preference of the algorithm's own, which it carries onto every
problem; Kudela, van Stein, Bäck & Kononova (GECCO 2026) found such pulls in
120 PlatEMO algorithms and traced the one towards the bounds to the default
clamp. [`python/examples/structural_bias.toml`](../python/examples/structural_bias.toml)
runs every algorithm on both, thirty seeds at 10 000 evaluations (about an
hour on 16 cores), and `--bias` reads it: per problem and algorithm, a
chi-square against uniform over ten bins per variable (the mean over the
variables, and the smallest p-value) and the shares of coordinates within
10 % of a bound and in the central 20 % — 0.2 each when uniform — with their
spread over the runs. The points of one population are not independent
draws, so the p-values overstate the evidence: read them as a scale and the
shares as the effect.

**The archive scenario.** With the run archive on (the default), every run
also stores `final_archive` in its `meta.json`: the final indicators once more,
on the archive reduced to the problem's population size by the same DSS
selection — what the run found, not what the algorithm kept.
`archive_scenario = false` turns it off. `--scenario archive` makes
`--compare`, `--ranks` and every report below read it instead of the final
population, which answers a different question: how good the search was, as
opposed to how good the population it returns is. `meta.json` also records the
frame the selection normalised by, so `--recompute ... --scenario archive`
selects the same points again from `archive.csv` — or, on a campaign run before
the scenario existed, selects them for the first time.

Two more readings need no rerun. `--at 0.25` gives `--compare` and `--ranks`
every run as it stood at a quarter of its budget, read from its trajectory, so
ranks at several budgets come out of one campaign — they do differ with the
budget (Tanabe & Oyama, GECCO 2017). `--recompute eps,hv_h` computes indicators
a campaign did not record from each finished run's `final.csv` and stores them
in its `meta.json` — `--scenario archive` does the same from `archive.csv` into
`final_archive` — with the campaign's own hypervolume settings; `--workers`
spreads the work.

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
rank alone cannot tell whether two medians differ by more than the spread
between seeds. Use the ranks to find where to look and the tests below to
decide. And a group mean is only as broad as its group: best on WFG at five
objectives means best on those nine problems, at the budget the campaign gave
them.

The tests and groupings, all read from the same results:

```bash
python -m mootation.run.campaign c.toml --ranks hv_h --reference nsga2   # tests against a reference
python -m mootation.run.campaign c.toml --ranks hv_h --ci                # bootstrap intervals of the mean ranks
python -m mootation.run.campaign c.toml --ranks igdp_norm --by front     # ranks within groups of problems
python -m mootation.run.campaign c.toml --ranks hv --scenario archive    # the run archives instead
python -m mootation.run.campaign c.toml --gap igdp_norm                  # distance to the best known value
python -m mootation.run.campaign c.toml --zero-share                     # seeds whose hypervolume is 0
python -m mootation.run.campaign c.toml --ecdf igdp_norm --interpolation linear
```

- `--reference ALG` (with `--ranks` or `--compare`): on every problem, each
  algorithm's seeds against the reference's by the exact Wilcoxon rank-sum
  test, Holm-corrected over the problems, with the Vargha–Delaney A12 effect
  size; across problems, the exact Wilcoxon signed-rank test on the
  per-problem medians, Holm-corrected over the algorithms. `+` / `-` / `=`
  count the problems where an algorithm is significantly better, worse, or
  neither (`--alpha`, 0.05). Both tests are exact with ties — the null
  distribution is built by dynamic programming, not approximated — which
  matters on the tied medians of converged runs. With five seeds a side the
  smallest attainable rank-sum p is 1/126 ≈ 0.008, before Holm. The
  signed-rank test ranks the SIZE of the differences across problems, so use a
  scale-free metric for it (`hv`, `hv_h`, `igdp_norm`, `eps_norm`): in raw units
  the problem with the largest values decides.
- `--ci`: a 95 % percentile interval for each mean rank, resampling the
  problems 2000 times. Two algorithms whose intervals overlap widely are not
  ordered by this campaign.
- `--by KEY` groups the problems by one property and ranks within each group:
  `front` (linear, concave, convex, mixed, disconnected, degenerate),
  `multimodal`, `deceptive`, `bias`, `scaled`, `separable` and `centre`
  (the optimum in the middle of the box). The values are Huband, Hingston,
  Barone & While's (IEEE TEVC 10(5), 2006, Tables V, VII and XV) for ZDT, DTLZ
  and WFG and are argued in `mootation/benchmarks/properties.py` for the rest;
  unknown is `?`, never guessed.
- `--gap METRIC`: every algorithm's median distance to the best final value
  any run reached on each problem — a common zero across problems whose raw
  values differ by orders of magnitude.
- `--zero-share [METRIC]`: the share of seeds whose final hypervolume (or
  METRIC) is exactly 0, a split between seeds that a median hides.
- `--ecdf METRIC`: the COCO-style runtime ECDF — the share of (run, target)
  pairs solved by each evaluation count, the targets the best known value plus
  1, 0.1, …, 10⁻⁴ in the metric's own units (so use a normalised one). How a
  target reached between two records is charged is stated in the output:
  `step` (the default) charges it to the later record, a pessimistic runtime
  with no assumption; `linear` interpolates the evaluation count.

Every table marks with `*` the sixteen algorithms whose behaviour follows the
share of the budget spent (a t/t_max schedule): RVEA, MOEA/D-AWA, AdaW,
DEA-GNG, MBRA, NRV-MOEA, HLMEA, DHEA, MOEA/D-DS, SRV, SRV-NSGA-III, DCEA,
MaOEA-3C, MOEA/D-M2M, MOEA/D-AM2M and Liu–Gu 2011. Their runs at 10 000 and
25 000 evaluations are not one run cut at two points, so overlay their curves
by fraction of the budget (`--at`), never by absolute evaluations.

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
runs every algorithm, the two baselines and the two ablations on 203
problems — ZDT at two objectives; DTLZ1–7, WFG1–9, the inverted IDTLZ1–2,
the scaled SDTLZ1–2, shiftDTLZ1–4 (DTLZ1–4 with the optimum moved off the
centre of the box) and
ZCAT1–20 at three and five; and bbob-biobj F1–F55, every pair of ten bbob
functions, at 5 and 10 variables — five seeds each at 10 000 evaluations:
62 930 jobs and about 105 CPU-hours, some seven hours on a 16-core machine. The
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

`--problems` narrows everything to part of the selection — `--list`, the run
(with `--shard` and `--force`), the tables and `--recompute` — which is how a
campaign catches up when some problems' reference fronts change: their final
indicators can be recomputed, but their trajectories were normalised by the
old front, so those problems are run again and nothing else is.

```bash
P=ZDT3,DTLZ7_3D,DTLZ7_5D,WFG2_3D,WFG3_3D,WFG3_5D,DTLZ5_5D,DTLZ6_5D
P=$P,ZCAT11_3D,ZCAT12_3D,ZCAT13_3D,ZCAT16_3D,ZCAT11_5D,ZCAT12_5D,ZCAT13_5D
python -m mootation.run.campaign campaign_all.toml --problems $P --force --workers 16
```

The fifteen problems in `P` are those whose references changed on 2026-09-22
(the full fronts and the cleaned samples). Run without `--problems`
afterwards, the campaign fills in only what is still missing — the
baselines added since, on every other problem.

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
