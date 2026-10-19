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
| `--algorithms` | list the 75 algorithm names and the nine Python methods |
| `--problems` | list the 1340 benchmark problems (needs NumPy) |
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

### Failed evaluations

A finite-element model can crash, not converge or return NaN. `minimize`
(and so every campaign) runs the objective, the constraints and `batch=`
through a guard (`mootation/failures.py`): an exception of
`failure_exceptions` (`Exception` by default), a non-finite value or a row a
batch returned as `None` is a failed evaluation. With `on_failure =
"infeasible"`, the default, the core gives the point a penalty
(`data_vault.hpp`, FAILED EVALUATIONS): every objective one span above the
worst finite value evaluated so far and every constraint violated beyond the
worst seen — so it is dominated by every point that did not fail, worse than
any infeasible one, equal to every other failure — and leaves it out of the
answer and the observers' rows; outside the core its values are NaN.
`on_failure = "raise"` stops the run at the first failure, for debugging.
A failure is an evaluation: it counts in the budget, and `result.n_failed`
counts them. `max_failure_share = 0.3` with `failure_window = 100` stops the
run with `FailureShareExceeded` once more than 30 of the last 100 evaluations
failed. `log_evaluations` marks a failed row (`failed`, `error`). A hung
evaluation is not handled here: the batch function or the job that runs the
solver must stop it (a timeout) and return `None` or raise. The penalty
reaches an algorithm's nadir point only while a failure stays in its
population, which an elitist selection ends once it holds `pop_size` points
that did not fail.

### Equality constraints

`minimize(eq_constraints=h, n_eq=k)` takes equalities h(x) = 0 beside the
inequalities of `constraints`. An equality holds where |h| <= `eq_tol` (1e-4
by default) and adds max(|h| - `eq_tol`, 0) to the violation cv: Liu and
Wang's rule (IEEE TEVC 23(5), 2019, Eq. 3, with their DOC problems); the RWMOP
competition's guidelines (Kumar et al. 2021) take the same 1e-4 but count |h|
itself beyond it. The core does the arithmetic (`data_vault.hpp`, EQUALITY
CONSTRAINTS), so every algorithm sees an equality as the inequality
|h| - `eq_tol` <= 0, and with a fixed tolerance a run is, bit for bit, the run
with that inequality in its place. `eq_tol_schedule = "linear"` with
`eq_tol0` lets the tolerance fall from `eq_tol0` to `eq_tol` in proportion to
the evaluations spent (it needs `max_evaluations`): before every step the
core rewrites every point's violation from its |h|, and the answer's cv is
taken at `eq_tol` (`result.eq_tol`); an observer reads the tolerance of the
moment from `mootation._core.current_eq_tol()`. A core that keeps its own
copies of the constraint values (LIS-LCS, MOSA's archive) judges a point by
the tolerance of the moment it read them, so under the linear schedule its
answer can hold points that meet only a looser one; check them against
`eq_tol` there.

### The epsilon level

`minimize(constraints=g, constraint_mode="eps_level")` counts a violation up
to a level ε as none: the vault hands the algorithm cv = 0 for such a point
(`data_vault.hpp`, EPSILON LEVEL), so every core that ranks by feasibility
first does Takahama and Sakai's ε-level comparison — by the objectives where
both violations are at most ε, by the violation otherwise. ε follows Fan et
al.'s improved setting (Soft Computing 23(23), 2019, Eq. 8, the rule of
MOEA/D-IEpsilon) once per generation of `pop_size` evaluations: ε(0) is the
violation of the initial population's (0.05 N)-th worst point, then 0.9 of
itself while fewer than 95 % of the population are feasible, 1.1 times the
largest violation seen once that many are, and 0 from 80 % of the budget on.
The paper's Tc is 800 of its 1 000 generations; here it is 80 % of whatever
budget the run has — a fixed 800 generations would never reach ε = 0 on a
shorter budget. `eps_schedule = {"tau", "alpha", "theta", "tc"}` replaces any
of the four; with `tc = 0` ε stays 0 and the run is the feasibility-first one
bit for bit. `result.cv` is the violation itself; an observer reads ε from
`mootation._core.current_cv_level()`.

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
BT, the DTLZ variants, the uninformative probes, the constrained C-DTLZ, MW, DAS-CMOP, LIR-CMOP and DOC, rotated
ZDT, DTLZ, WFG, MaF, ZCAT and Polygon, noisy and failing copies and the real-world RE and CRE, bbob-biobj-mixint and catZDT1,
the multimodal SYM-PART, Omni-test and MMF of the CEC 2019 competition, and RWMOP's mechanical, chemical and process
problems RCM01–29: 1340 problems across 49 families, each
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

**A batch at a time.** Every problem evaluates a batch of points:
`p.evaluate_batch(X)` takes n rows of `n_vars` values and gives the n × M
objectives, `p.constraints_batch(X)` the n × K constraints, and a constrained
problem whose objectives and constraints come out of one computation (DAS-CMOP,
LIR-CMOP, MW, C-DTLZ, DOC, RWMOP, CRE) gives both at once,
`p.evaluate_with_constraints(X)` → `(F, G)`. A family written for batches
(ZDT, DTLZ and its variants, C-DTLZ, MaF but MaF8 and MaF10–12, ZCAT,
bbob-biobj and its mixint, DAS-CMOP, LIR-CMOP, MW, BT, MOP, and the rotated,
shifted and noisy copies of those) has one implementation, the batch, and
`p.evaluate(x)` is its batch of one row; every other problem's batch is the
loop over its point function. Either way a row's values are, bit for bit, what
the point code gave that point before batches existed (`tools/compat_problems.py`
compares the whole registry with a commit of before; `_rows.py` lists what had
to be kept one point at a time for that: BLAS's product per point, the C
library's `pow`, `sin`, `cos` and `exp` of one float where NumPy's of an array
may differ). A row whose evaluation fails is NaN in that row; the batch goes on.
`benchmarks.solve` and every campaign hand the batch to `minimize` (`batch=`,
`batch_with_constraints=`), so a generational core's whole generation is one
call.

**In C++: the suite.** ZDT1–4 and ZDT6, DTLZ1–7 and wideDTLZ3, shiftDTLZ,
IDTLZ, minusDTLZ2, SDTLZ, C-DTLZ, MaF1–7, MaF9 and MaF13, ZCAT1–20, shiftZDT
and the rotations over them (rZDT, rDTLZ, rcDTLZ, r0/c0DTLZ, rMaF, rcMaF,
r0/c0MaF, rZCAT) are computed in C++ when the extension is built
(`include/mootation/suite`, `benchmarks/native.py`): `p.evaluate`,
`p.evaluate_batch` and the constraints call it, the loop over a batch runs in
C++, and `p.native` is the `_core.Native` behind them. The NumPy code stays,
the reference in the tests (`p.reference`) and what runs without the
extension; the two give the same values bit for bit wherever NumPy's `sin`,
`cos`, `exp` and `pow` are the C library's (`_rows.agrees`), which the tests
check on every problem of these families. `minimize` hands such a problem to
the core, which then evaluates it without calling Python (`native = "auto"`,
the default; `"off"` calls it as any function), wherever nothing of the
caller's stands between (`minimize.direct_path` says what does: a batch or
constraints of the caller's own, `log_evaluations`, typed variables,
`budget_counts = "unique"`, `max_failure_share`, a seed population, polishing,
a scalarization method); `on_rows` then observes every evaluation a batch at a
time, and `result.native_seconds` is the time the evaluations took. A campaign
does the same (`[campaign] native`, below) and writes the same files either
way. Two parts of these families changed in the last bits with it and are
version 2 of themselves (`registry.PROBLEM_VERSION`, in every run's
`meta.json` as `problem_version`; `--list` marks a run of an older version):
the rotated problems whose matrix is not diagonal, which now take A·d as
NumPy's pairwise sums of the rows of A ⊙ d (`_rows.matvec_pairwise`, an order
C++ repeats) instead of BLAS's product, whose order is its own, and MaF9's
constraint from five objectives, its dot products of two coordinates written
out. The controls r0 and c0 are unchanged. MaF8 stays in Python (its
distances are CPython's `math.hypot`), and so do MaF10–12 (WFG).

**Where a problem is not defined.** RCM02 (EI / μ < 0, the layers out of
order) and RCM24 (the logarithms of its equalities, outside x8 > 100,
x7 < 600, x9 > x7) return NaN there — a failed evaluation (below), counted in
a run's `n_failed`; RE22 and RE22c, at x2 = 0, the face of the box where
their stress divides by zero, too. `benchmarks.properties` says so,
`undefined = true` and `undefined_share`, the share of the box estimated from
100 000 uniform points (`rwmop.undefined_share`).

**Four problems use their FULL fronts, and three of them have very long ones.**
DTLZ5, DTLZ6, MaF6 and WFG3 were designed with a degenerate, curve-shaped
Pareto front, but the true fronts of DTLZ5 and DTLZ6 from 4 objectives and of
WFG3 from 3 also have a non-degenerate part (Ishibuchi, Masuda & Nojima, IEEE
TEVC 20(5), 2016), and — by this library's own analysis — MaF6's from 7. At
those sizes the reference front is now
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
**The constrained problems** are Jain & Deb's C-DTLZ (IEEE TEVC 18(4), 2014,
doi:10.1109/TEVC.2013.2281534) at three and five objectives: `C1-DTLZ1_3D`,
`C1-DTLZ3_3D` (a feasible band, or an infeasible shell, beside the front),
`C2-DTLZ2_3D`, `C2-ConvexDTLZ2_3D` (only pieces of the unconstrained front
feasible) and `C3-DTLZ1_3D`, `C3-DTLZ4_3D` (the front on the constraints), and
the same at `_5D`. A constraint is a value that is at most 0 when satisfied,
as everywhere in the library: the paper's c(x) >= 0 with the sign turned. Two
of the paper's equations are read as pymoo reads them (`c_dtlz.py`, READINGS);
values checked against pymoo 0.6.2 at 60 points, reference in
`tests/data/pymoo-0.6.2_c-dtlz.json`. The fronts are analytic but for C2-ConvexDTLZ2's: behind the hole the cylinder cuts in
the convex front lie points of larger g that nothing feasible dominates, which
the paper's front leaves out; that front is a sample checked point
by point (`BenchProblem.front_exact` is False, and `--cover` says so). MaF9 from five
objectives is constrained too.

**The noisy copies** stand in for the noise of a mesh and a solver: `ZDT1_n1`
returns ZDT1's objectives plus 0.01 (nadir − ideal) N(0, 1) in each, and `_n5`
0.05 (`benchmarks/noisy.py`, a construction of this library). The noise is
drawn from the run's seed and the evaluation's number, so a seed repeats a run
exactly. Three more kinds, at the same two levels: `_d1`/`_d5`, deterministic —
the noise is a hash of x, the same in every run and at every repeat, as a
solver's tolerance gives, which repeats do not average away; `_s1`/`_s5`, a
step — the hash of the cell ⌊(x − lb)/h⌋, h = `noise_cell` (0.01, a
`[campaign]` key) of every variable's range, constant inside a cell and a jump
at its border, as a remeshing gives; `_h1`/`_h5`, relative — σ (f − ideal)
N(0, 1), as a solver with a relative tolerance gives. All four kinds of ZDT1,
ZDT4, DTLZ1_3D, DTLZ2_3D, DTLZ2_5D, WFG4_3D, WFG7_3D, rcDTLZ2_3D and of RE21,
RE25 and RE34, whose objectives differ in scale by orders of magnitude.

The algorithm sees the noisy values; a campaign measures every indicator on
the noise-free ones at the same x, keeps those in its files and snapshots —
`final.csv` the noisy ones too, as `f1_seen`, ... — and records
`true_dominated_share` beside them, the share of the answer another of its
points dominates by the true values. `final` then holds every indicator three
ways: as recommended (the noise-free values of the answer the algorithm chose,
without a suffix), `<metric>_oracle` (the noise-free non-dominated set of every
point evaluated, reduced to the answer's size by the archive's rule — what the
run found, recognised or not) and `<metric>_observed` (the noisy values of the
answer, biased towards better: the size of that bias). They are a family of
their own, `noisy`, picked by name: `families = ["DTLZ"]` does not bring them
in; the copies of RE are ranked with the real-world problems.

**The failing copies** fail in part of the box, as a solver that crashes
would (`benchmarks/failing.py`, a construction of this library):
`ZDT1_fail10r` is ZDT1 returning NaN in a region `r` holding 10 % of the box's
volume — four balls at random centres —, `_fail30r` 30 %; `b` the band along
the box's faces holding that share; `n` the neighbourhood of part of the
Pareto set, the share being the share of the set that fails. ZDT1, DTLZ2_3D
and WFG4_3D, 18 problems, a family of their own, `failing`. Where the region
misses the Pareto set, front and frame are the base's; where it takes part of
it, the reference is the image of the rest of a Pareto-set sample — a part of
the copy's front, whose new points at the region's edge are not known — and
`front_exact` is False (`--cover` says so); where it takes all of it (ZDT1's
optimum lies on a face, inside the band), the reference is the base's front,
which no point outside the band reaches. A campaign counts the failures:
`n_failed` in `final`, `fail_share` (since the previous record) in the
trajectory, and a failed point never enters the run archive or the oracle.

**The real-world problems** are Tanabe & Ishibuchi's RE and CRE (Applied Soft
Computing 89:106078, 2020): truss, beam, pressure vessel, welded beam, disc
brake, speed reducer, gear train, car side impact, ship design, water
resources — `RE21` … `RE91` without constraints (where the original has some,
the last objective is the sum of their violations) and `CRE21` … `CRE51` with
them, ported formula for formula from the authors' Python code
(github.com/ryojitanabe/reproblems, MIT; `benchmarks/re.py`, whose header
lists the defects kept on purpose — RE22's "3,10", RE33's range of x4 — and
those fixed upstream). Their true fronts are unknown: the reference is the
authors' approximation for RE, shipped as it is (`benchmarks/_re/`), and for
CRE the feasible points of RE's, which are thin — so IGD+ and ε there are a
distance to an approximation (`front_exact = False`; the rank table and
`--cover` say so). Integer variables are rounded inside the function, as in
the original. RE91 draws four normals at every evaluation; here they come
from the run's seed. In a rank table the real-world problems count apart:
with synthetic ones beside them, `all` is the synthetic problems only and
`engineering` the real-world ones (`--by family_kind` groups the same way).

`RE22c` and `RE33c` are their corrected copies, beside the originals, which
stay for the published fronts: RE22 with x1's list as meant (3.10 for the
original's "3,10", and the 7.00 it leaves out: the 77 values of the source's
table), RE33 with x4 in [2, 11] (reproblems' issue #9; the source's box is
[2, 20] with a constraint x4 <= 11, which the suite read as x4 >= 11).
reproblems has no front for them; theirs are built here from an exhaustive
grid refined locally and thinned by DSS to 500 M points
(`python -m mootation.benchmarks.re_fronts`, `benchmarks/_re/corrected/`), the
frame the front's own.

`RCM01` … `RCM29` are RWMOP's mechanical, chemical and process problems
(Kumar et al., Swarm and Evolutionary Computation 67:100961, 2021;
`benchmarks/rwmop.py`), constrained, two to five objectives, at the
competition's budget (20 000 evaluations at two objectives and up to ten
variables, 26 250 at three, 53 000 at five). The formulas are the paper's as
printed; the module's header lists the few places that cannot be read as they
stand and how they are read (constants the paper uses without a value, and
RCM06's g11, RCM08's g8 and RCM28's sign, which as printed leave nothing
feasible), and the places where the paper's problem is not the classic one,
kept. Six of them are Tanabe and Ishibuchi's CRE21–25 and CRE31, and agree
with `re.py` to 1e-12. RCM22–28 have equalities, counted as DOC's
(`|h| − 1e-4 ≤ 0`, `n_eq`). Not in the registry: RCM19 (its demands are never
given), RCM20 (as printed nothing is feasible), RCM24 and RCM27 (their fronts
are single points), and RCM30–50 (the paper leaves out the inverters'
frequencies and the networks' data). The references are this library's:
the feasible non-dominated points of ten long runs per problem, and for the
small problems with equalities or 0/1 variables the epsilon-constraint method
by SLSQP, or the exact front (`front_exact = False`; the header says how).

**Typed variables.** A problem's `var_types` declares each variable `"real"`,
`"int"`, `("discrete", values)` or `"bin"`: RE22, RE23, RE25, RE35, RE36, CRE24,
CRE25 and RE22c (RE33c's x4 stays real, as reproblems evaluates it), and the
bit-string problems below. A campaign hands it to `minimize`, whose operators
then work on reals and the vault rounds after every write — the integer
variables to the nearest integer (ties to even) in an operator space widened to
[lo − 0.5, hi + 0.5), the discrete ones to the nearest value of their list —
so the genome holds the rounded value (`int_repair = "lamarck"`, the default,
for every algorithm that writes through the vault) or keeps the real one and
only the evaluation sees it rounded (`"baldwin"`). Polynomial mutation takes
`eta_int` (3) for those variables. Both are `params` of an algorithm entry. A
run records `dup_x_share`, the share of its evaluations whose decoded point had
been evaluated before.

`("cat", labels)` — or `("cat", k)` for the labels 0 .. k−1 — is a
categorical variable: k ≥ 2 unordered values, which `fn` receives as the
label itself. Bits and categorical variables form the cores' integer genome
(DataVault's `binary_variables`: a bit 0 or 1, a label its index): uniform
crossover takes the genes as they are, and mutation gives each gene, with
probability 1/(number of genes) but at most 1/2 (pymoo's cap; with a single
gene 1 would change every child's), one of its other values, each alike —
a bit flips (`operators/bit_flip.hpp`). SBX, DE and polynomial mutation never
see them, and with more than two genes a binary run is what it was, bit for
bit. 50 of the 75 cores take the integer genome; these 25 refuse it by name:
`bc_rex`, `bc_spx`, `clia`, `dcea`, `dea_gng`, `dhea`, `dms`, `dmulti_mads`, `dn_nsga2`,
`etea`, `gde3`, `hcca`, `hlmea`, `if_maoea`, `isde_rd`, `maoea_3c`, `mo_cma_es`,
`moead_ds`, `mosa`, `multiglods`, `naemo`, `paes`, `rm_meda`, `smpso` and `two_arch2`
(`minimize` says so, naming the variables); so do the Python methods.

`result.evaluations` counts the calls, `result.unique_evaluations` the
distinct decoded points among them (`mootation/unique.py`; the key is the
point `fn` receives, so two reals that round onto one integer are one point).
`budget_counts = "unique"` answers a repeat from memory without calling `fn`
and counts `max_evaluations` in distinct points — what a solver run costs; a
run whose algorithm asks for nothing new in 10 000 steps ends there (a space
smaller than the budget). The default, `"calls"`, calls `fn` every time, as
before. The run journal's cache (`run/ledger.py`) keys the decoded point too.

**bbob-biobj-mixint** (Tušar, Brockhoff and Hansen, GECCO 2019;
`benchmarks/bbob_biobj_mixint.py`): the 55 bbob-biobj pairs at n = 5 and 10
with the first four fifths of the variables integer, n/5 each of arities 2,
4, 8 and 16, the rest continuous — `bbobbiobjmixint07_n10_2D`. The integer
value j stands for z_{j+1}, l equidistant points in (−4, 4) shifted so that
one is the objective's optimum (the paper's §3.1.1), each objective about its
own; the ideal and nadir are the pair's at its mixed-integer optima, and there
is no front. The paper's scaling factors are left out (a factor on one
objective changes neither dominance nor a normalized indicator), and the
instances are bbob-biobj's own, not COCO's, so nothing here compares with
COCO's archives. **catZDT1** (`benchmarks/categorical.py`, ours): ZDT1 whose
distance function g a categorical variable picks — ZDT4's multimodal one,
which alone reaches ZDT1's front, or one of two easy ones whose least g is
1.02 and 1.05, just above it.

**DAS-CMOP1-9** (Fan et al., Evolutionary Computation 28(3), 2020;
`benchmarks/dascmop.py`): constrained problems whose difficulty a triplet
(η, ζ, γ) sets — diversity (Type I constraints on the positions),
feasibility (Type II, a band [d, e] for g, an equality at ζ = 1) and
convergence (Type III, ellipses or spheres in the objective space). Each is
registered at the paper's sixteen triplets, `DAS-CMOP1_t01` … `DAS-CMOP9_t16`
(Table 3), and DAS-CMOP1-3 at Figure 6's (0, 0.5, 0.5), `_f6`;
`dascmop.problem(i, eta, zeta, gamma)` gives any other. Their fronts are built
on first use from a dense sample of the attainable set, each point's g brought
down to its feasibility boundary by bisection, and the frame read off them;
the properties' `difficulty` names the triplet's kinds. The formulas are the
paper's Table 2; values checked against pymoo 0.6.2 at 36 points, reference
in `tests/data/pymoo-0.6.2_das-cmop.json`; at ζ = 1, where the feasibility constraint becomes the equality g = d, the
library takes the paper's tolerance 10^-6 (its Eq. 13).

**LIR-CMOP1-14** (Fan et al., Soft Computing 23(23), 2019;
`benchmarks/lir_cmop.py`): constrained problems with large infeasible
regions, n = 30. In 1-4 each distance function g must lie in the band
[0.5, 0.51], so the feasible set is small; 5-8 add ellipses in the objective
space (the front of 5 and 6 is the unconstrained one, that of 7 and 8 lies on
an ellipse's boundary); 9-12 an ellipse and a sine constraint that cut the
front into pieces; 13 and 14 have three objectives and spherical shells. The
formulas are Table 6 of the paper's preprint (arXiv:1707.08767). The fronts
are built on first use from the problems' geometry — for 5-12 the least
feasible attainable f2 at every f1 of a fine grid, by bisection — and the
frame read off them; every front point is reached by a feasible x to 1e-12,
and feasible points made on purpose do not beat it. Two kinds of point
belong to 9-12's fronts though they lie apart from the rest: f1 = 0 only at
x1 = 0 and f2 = 0 only at x1 = 1, so the least feasible point on either axis
is Pareto-optimal (9: (0, 2.175...)), and the frame takes it.

**DOC1-9** (Liu and Wang, IEEE TEVC 23(5), 2019; `benchmarks/doc.py`):
constraints in both the decision and the objective space. Each takes a
problem of the CEC 2006 suite (g04, g19, g23, g09, g21, g07, g14, g10, g18)
with its constraints as the decision part, its objective shifted to a least
feasible value of 1 as the distance function g of a ZDT- or DTLZ-like pair or
triple, and constraints on the objectives that cut the front: an arc, pieces,
or 21 and 14 isolated points (DOC4, DOC5). DOC3, DOC5 and DOC7 hold
equalities (4, 5, 3): `constraints` gives them as |h| - 1e-4 <= 0, the
paper's Eq. 3 and, bit for bit, `eq_constraints` at `eq_tol = 1e-4`;
`equalities` and `n_eq` give h itself, and `benchmarks.solve` passes them on,
so that `eq_tol` and `eq_tol_schedule` reach them. The formulas are the
paper's supplementary file, S-I, with two readings where it contradicts the
front it states (DOC-7's sine constraint, DOC-9's f3; in `doc.py`). Checked:
the least g over the decision constraints is 1 for all nine (SciPy's SLSQP,
the equalities within 1e-4), and the share of feasible random points is
27.0 % for DOC1 and 0.52 % for DOC4 (the paper's Table I: 26.97 %, 0.53 %).
The fronts are the paper's; DOC2's ends are solved here, and its 0.3830 is
0.3803, the circle's edge.

**Bit strings.** OneMinMax (`OMM_n10`, `_n20`, `_n50`), LOTZ (`LOTZ_n…`) and
OneJumpZeroJump (`OJZJ_n10_k2`, `OJZJ_n10_k3`, `OJZJ_n20_k3`, `OJZJ_n20_k4`)
of the runtime theory (`benchmarks/binary.py`; maximized there, registered as
their negatives), with exact fronts checked against every string up to n = 12.
A run on one records the paper's measure (Bian et al., IJCAI 2023): `cover_gen`
and `cover_evals`, the first generation and evaluation at which the population
holds every front vector, and `cover_archive_gen`, `cover_archive_evals` for the
inner archive (`inner_archive = "nd"`) — NaN, with `cover_reached` false, when
the budget ran out first. Read them at every generation: `record_every = 1`.

**The rotated problems** mix the distance variables, which ZDT, DTLZ and WFG
keep each on its own axis — a gift to operators that work coordinate by
coordinate, SBX and polynomial mutation. The position variables are left
alone; the distance block is rotated about its optimum, so the front and the
Pareto set stay exactly known (`benchmarks/rotated.py`, a construction of this
library; its header sets it beside Deb, Sinha & Kukkonen's L1-ZDT4, Iorio &
Li's rotated problems and CEC 2007's R_DTLZ2). `rDTLZ1_3D` … `rDTLZ4_5D`
evaluate DTLZ at y = 1/2 + Q (z − c), c the shifted optimum, unclipped;
`rcDTLZ2` at y = 1/2 + Λ Q (z − c) with condition number 10³; `rWFG4`–`rWFG7`
rotate WFG's normalised distance variables about 0.35 and clip them to
[0, 1] (most random points have a few clipped; none within 0.35 of the
optimum). DTLZ2's and DTLZ4's g is isotropic, so `rDTLZ2` and `rDTLZ4` are the
same functions as without the rotation — checks of invariance; rotated in
earnest are `rDTLZ1`, `rDTLZ3`, `rcDTLZ2` and `rWFG4`–`rWFG7`. Their controls
are the same construction with Q = I: `r0DTLZ1`, `r0DTLZ3`, `c0DTLZ2` (each a
family of its own), and WFG itself.

Task 7 added `rZDT1`–`rZDT4` and `rZDT6` (ZDT at |Q (z − c)| / m, ZDT4 at
Q (z − c); their control is shiftZDT); `rWFG1`–`rWFG3`, and `rWFG8` and `rWFG9`,
whose optimum depends on other variables and which turn about that optimum
for the current positions; `rcDTLZ5`; MaF by the form of its g — `rcMaF1`,
`rcMaF5`, `rcMaF6` conditioned, `rMaF2`–`rMaF4` rotated, controls `c0MaF*` and
`r0MaF*`; `rZCAT1`–`rZCAT20`, about the Pareto set y_II = g(y_I), clipped to
the box; `rcPolygon` with its optimum moved off the centre (control
`c0Polygon`); and `rcDTLZ2` and `c0DTLZ2` at six and ten objectives. Every
rotated problem, the controls aside, comes in three instances of Q,
`<name>_i1` … `_i3`, the first being the problem without the suffix.

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
| `i_cmop` | Vodopija, Tušar and Filipič's indicator for a constrained problem (IEEE TEVC 29(1), 2025, Eq. 12), on the whole answer: with a feasible point, at most 1 — minus the normalised hypervolume (reference point 1) of the feasible points inside [ideal, nadir], or their distance to that box while none is; without one, 1 plus the least violation. One number for runs with and without feasible points, any feasible set better than any infeasible one | lower |
| `range_cover` | the worst objective's covered share of [ideal, nadir]; falls early when a population collapses onto part of the front, as on DTLZ4; per objective in `range_cover_each` | higher |
| `nd_share`, `dup_share` | the share of the set no other member dominates, and the share that repeats an objective vector already in it: stagnation and duplicates | — |
| `igdx` | IGD in the decision space: the mean distance from a Pareto-SET sample to the nearest solution, variables normalised by the bounds (Tanabe & Ishibuchi 2019, Eq. 5) | lower |
| `cr` | cover rate: how much of the Pareto set's extent in each variable the solutions span, geometric mean over the variables, 1 = all (Tanabe & Ishibuchi 2019, Eqs. 7–8) | higher |
| `psp` | Pareto sets proximity, `cr` / `igdx` (Yue, Qu & Liang, IEEE TEVC 22(5), 2018, Eq. 8): near the Pareto set and over all of its extent, in the variables. `igdx` is normalised by the bounds here, the paper's is in the raw variables, so with one range r for every variable the paper's PSP is this one over r; inf where `igdx` is 0. The CEC 2019 competition ranks by 1/PSP | higher |
| `pdist` | the mean distance between two solutions in normalised variables: spread in the decision space, no reference needed | — |
| `tau90` | the 0.9 quantile of the IGD+ distances d⁺(z, A) = min over a of ‖max(a − z, 0)‖ over the reference sample, normalised like `igdp_norm`: 90 % of the front lies within `tau90` of the set. `igdp_norm` is the mean of the same distances; one reference point far from everything moves their maximum and not this. Weakly Pareto-compliant, like IGD+. With it, `coverage_curve` gives the share of the sample within 0.01, 0.02, 0.05, 0.1 and 0.2 | lower |
| `n_final` | the number of points in the answer. `hv`, IGD+ and ε never get worse when points are added, so when two algorithms answer with sets of different sizes, read them next to this | — |
| `r2` | discrete R2 (the unary R2 of Brockhoff, Wagner & Trautmann, Evol. Comput. 2015, Def. 4): over the Das–Dennis weights of `hv_h`'s lattice, the mean of the set's best weighted Tchebycheff value max_i w_i·(f_i − ideal_i)/(nadir_i − ideal_i) (`r2_h` is the lattice's H). Weakly Pareto-compliant, and no more points count than there are weights; its cost grows linearly with the number of objectives where the hypervolume turns Monte-Carlo, and it is not 0 where `hv` is. Meant for four objectives and more; the R2 integrated over all weights is Pareto-compliant (Schäpermeier & Kerschke, arXiv:2407.01504, 2024, exact at two objectives; Jaszkiewicz & Zielniewicz, IEEE TEVC 29(4), 2025) | lower |
| `gdp_norm` | GD+ with objectives and front divided by nadir − ideal, like `igdp_norm`: how far the set is from the front, one scale on every problem | lower |
| `gap_max` | the largest hole: the maximum over the reference sample of d⁺(z, A), normalised like `igdp_norm` — `tau90` taken at 100 %. One reference point nothing comes near decides it | lower |
| `nn_cv` | evenness: the coefficient of variation of the distances from each distinct non-dominated point to its nearest other one, objectives normalised by ideal and nadir; 0 when evenly spaced, larger as the set clumps. No reference front needed | lower |
| `gdp_min` | has the set reached the front anywhere: the smallest d⁺(a, Z) of any point of the set, normalised like `igdp_norm`. It cannot tell values below the reference's resolution apart, so `gdp_min_ref_step` (the median distance between neighbouring reference points) goes beside it and `gdp_min_below_step` says when it is lower | lower |
| `g_min`, `g_median` | the smallest and the median g(x) − g* in the set: the problem's own distance to the Pareto set (`BenchProblem.distance_fn`: ZDT1–4 and 6, DTLZ1–4 and 7, DTLZ5–6 up to three objectives, IDTLZ, SDTLZ, wideDTLZ3, the shift variants, C1-DTLZ1, C1-DTLZ3 and C2-DTLZ2). Exact, no reference sample | lower |
| `ps_dmin` | the smallest distance from a solution to the Pareto-set sample, variables normalised by the bounds and the cyclic ones wrapped, as `igdx`: the decision-space counterpart where the problem has a Pareto set but no distance function | lower |

**Dense references.** From four objectives on, a reference of 1 000 points is
only 1.5–2 times finer than the population it measures (the median distance
from a point to its nearest neighbour, in the problem's frame, against that of
N points of the same sampler), so IGD+ there measures the reference as well
as the set. For the problems with an analytic front — DTLZ1–4, IDTLZ, SDTLZ,
minusDTLZ2, shiftDTLZ1–4, WFG4–9, Polygon, MaF1, 3, 4, 5, 12 and 13, and the
rotated DTLZ and WFG, at 4 to 10 objectives — a campaign measures the front
indicators also against a dense reference (`[campaign] dense_reference`, on
by default; `false` switches it off): the first of 1 000, 2 000, 5 000,
… 100 000 points whose step is at most a third of the population's, or
100 000 (5 000 at four objectives, 10 000 at five, 20 000 at six, 100 000 —
still only 2.3 times finer — at ten). The values go beside the usual ones as
`igdp_norm_ref_v3` and so on, which every table can be asked for
(`--ranks igdp_norm_ref_v3`), and `meta.json` records the reference's rows
and step (`reference_v3`). At five objectives and 126 points an IGD+ pass
over 10 000 reference points costs about 0.1 s, over 100 000 about 1 s.
IGD, IGD+, GD+ and ε are computed in blocks of reference points, whatever the
size; no KD-tree, since d⁺ is not a metric and the nearest point by it can be
far by the Euclidean distance.

On a constrained problem every indicator is measured on the feasible part of
the answer, and two more are recorded unasked: `feas_share`, the share of the
answer that satisfies every constraint (wherever an indicator is measured, the
trajectory included), and `fe_first_feas`, the evaluation that first gave a
feasible point (in `final`; none when the run found none). An answer without a
feasible point has no value at all — none, not 0 and not infinity. Such an
empty run is counted apart: `--compare` writes "n, k empty" in its cell,
`--ranks` and `--ranks --by` an `empty` column, `--ci` "k empty" beside the
interval and `--reference` a line of them; in the median, the tests and the
ranks it counts as the worst value. A run on a constrained problem keeps the variables in its
snapshots and checkpoints, whatever `snapshot_variables` says, since telling
the feasible points apart needs them (in float32 there, so a point within
rounding of a constraint may read differently when recomputed).

`gdp` needs a reference front like `igd`; `roi_dist` and `range_cover` need only
the problem's ideal and nadir, and `nd_share` and `dup_share` nothing at all, so
those four also describe the bbob-biobj problems, which have no front; so does
`nn_cv`.

What each one looks at — the question a table of them answers depends on it:

| looks at | indicators |
|---|---|
| progress towards the front | `gdp`, `gdp_norm` |
| whether any point reached the front | `gdp_min`, `g_min`, `ps_dmin` |
| how much of the front is covered, and how evenly | `igdp`, `igdp_norm`, `tau90`, `gap_max`, `range_cover`, `nn_cv` |
| both at once | `hv`, `hv_h`, `eps`, `eps_norm` |

GD+ falls as the set approaches the front whatever part of it the set sits on;
the coverage indicators fall only as the set spreads over it (`nn_cv` needs no
front, and sees the spacing but not the extent). The hypervolume and ε cannot
tell a set that is close but narrow from one that is wide but far.

`igdx`, `cr` and `psp` need a sample of the Pareto SET, which only some problems
have: the Ishibuchi polygons (Polygon, IPolygon), ZDT1–4 and ZDT6, DTLZ1–4 and
DTLZ7, DTLZ5–6 up to three objectives, IDTLZ, SDTLZ, minusDTLZ2, the shift and
wide variants, ZCAT1–20, and the multimodal problems below, every part of
their sets. Where a problem has one, a campaign records the three along the
trajectory and in `final` without being asked (`[campaign] set_metrics`, on
by default; `false` records `metrics` alone). On shiftDTLZ1–4 the distance variables are
**cyclic** — the optimum
moved off the centre wraps around the box — so their differences are taken as
((x − c + 0.5) mod 1) − 0.5 in normalised units; a plain difference would call
two solutions on either side of the wrap far apart. On ZCAT the set is the
positions with every distance variable at its optimum g(y_I). Elsewhere they
are null. `pdist` needs nothing but the solutions. All three read the final
population (`final_metrics`), whose variables `final.csv` keeps, so
`--recompute igdx,cr,pdist` adds them to a campaign that ran without them.

**Multimodal problems.** Several parts of the variables map onto one front:
SYM-PART simple and rotated (nine parts), Omni-test (27 at three variables)
and MMF1–15 with MMF1_z, MMF1_e, MMF14_a and MMF15_a, the 22 problems of the
CEC 2019 competition on multimodal multiobjective optimization (Liang, Qu,
Gong & Yue's technical report; `benchmarks/mmo.py`), at its parameters and
its budget: a population of 100 per variable, 5 000 evaluations per variable.
Each has its front and a sample of every part of its Pareto set, derived from
the formulas and checked against a fine grid; where the report's own pictures
or set descriptions disagree with its formulas, the formulas win and the
module's header lists the cases (MMF2 and MMF3's second branch is a local set,
MMF8's front is the quarter circle, MMF13's printed set lies outside its box,
and so on). `dn_nsga2` is NSGA-II with a niched mating pool and crowding in
the variables, built for them (DN-NSGA-II, Liang, Yue & Qu, CEC 2016).

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
The decision-space indicators (`pdist`, and `igdx` and `cr` where the problem
has a Pareto-set sample) can be recorded along the trajectory too: the
campaign asks the running core for the rows' variables only at the records
(`mootation._core.current_variables()`, from the observer), and a baseline
hands its own over. `meta.json` says what the records cost, `record_seconds`
of the run's `seconds`.

The same C++ is available to a C++ program as `mootation/hypervolume.hpp`:
`hypervolume::wfg(F, n, m, ref)` for the exact value and
`hypervolume::covered(F, n, m, S, k)` for the Monte-Carlo count.

**Where the trajectory is recorded.** By default (`record_grid = "budgets"`)
at a grid of evaluation counts — 100, 150, 200, 300, 500, 700, 1 000, 1 500,
…, 7 000, 10 000, … per decade, 25 000, the counts of `record_at`, and the
end of the run — and at exactly b evaluations: the population the run had at
b, the last one the observer reported without passing b (a generational core
reports one a generation, a core of one point at a time one a population of
evaluations); a record whose population was complete before b says at how many
in `fe_pop`. Runs of every budget and every population size share every count,
and nothing between them costs a record. A campaign whose results directory
was started under another grid keeps it when the config does not name one, so
older campaigns read and resume as they were; `hard_budgets` (which read every
generation) default to `"generations"`. `record_grid = "generations"` with
`record_every = k` records every k generations. `record_grid = "log"` records
at fixed evaluation counts,
round(10^(j / `record_per_decade`)), 10 per decade by default: 100, 126, 158,
200, … The counts are absolute, so a 10 000- and a 25 000-evaluation run share
every point up to 10 000 and a budget ladder compares point by point, without
interpolating; a point is the first generation at or past its count.
`record_at = [2500, 5000, 10000]` adds counts the decade grid misses (it has
2 512 and 5 012): the rungs of a budget ladder get a record of their own, at the
same generation a separate run of that budget stops at. `record_at_metrics`
names indicators of `metrics` that are recorded only at those counts and at the
end of the run, not on the whole grid — the costly ones; `--at` and `--cover`
read a rung at the record `record_at` puts there, which has them, and
`--recompute-trajectory` fills the other records from the snapshots.

**What else a run keeps.**
- `meta.json` records `revision`: the git commit and whether tracked files
  differed from it, for the source tree and, separately, for the compiled
  extension — a pulled commit whose C++ was never rebuilt shows as a mismatch.
  The version string alone never changes between commits.
- `all_evaluated = true` (the default; it needs a reference front) keeps IGD+
  and the additive ε of EVERY point the run evaluated — the feasible ones,
  noise-free on a noisy copy, in `igdp_norm`'s frame — up to date at every
  evaluation (`_core.IncrementalFront`: for each reference point the least
  d⁺ and ε so far, O(|Z| M) a point): every record has them as
  `igdp_norm_all` and `eps_norm_all`, `final` too, and `meta.json`'s
  `all_evaluated` gives, for each level of `all_levels` (1, 0.5, 0.2, …,
  0.001), the evaluation at which `igdp_norm_all` first reached it — the
  evaluations-to-level an ECDF reads.
- `archive.csv` holds every nondominated point the run evaluated, whatever the
  algorithm kept: at most one per cell of a grid in normalised objectives
  (step 1e-3, or 1e-2 from five objectives; `archive_delta`), with each
  objective's best point kept outside the grid so the ends of the front are
  never pruned. `archive = false` turns it off. `meta.json` says the step and
  the normalisation used. `archive_variables = "selected"` keeps the variables
  only of the points the archive scenario selects (below): the objectives of
  every archive point go to `archive.csv.gz`, and the selected points,
  objectives and variables, to `final_archive.csv.gz`. On a large campaign
  that is about a tenth of the room of `archive.csv` (a quarter of it
  gzipped); the default `"all"` writes `archive.csv` as before. The archive is
  C++ (`_core.GridArchive`, the same points as the Python archive point for
  point) and takes a batch at a time; `archive_mode = "posthoc"` hands it the
  points only when it is read (the checkpoints, the hard budgets, the end),
  the same archive either way, for a campaign that reads it only at the end.
- `snapshots = true`, or a list of problem names, stores the population's
  objectives at every trajectory point in `snapshots.npz` (float32), to see how
  the front's shape moved or to compute an indicator the run did not record.
  `snapshot_variables = "pareto_set"` stores the variables as well where the
  problem has a Pareto-set sample (`"all"`: everywhere), so that `igdx`, `cr`
  and `pdist` can be recomputed too. `--recompute-trajectory METRICS` computes
  indicators at every record from the snapshots into `trajectory.jsonl` — an
  indicator thought of after a campaign costs its arithmetic, not a rerun. The
  rows are float32, so a recomputed value agrees with one recorded during the
  run to about 1e-6 relative, not bit for bit.
- `archive_checkpoints = true` takes the archive scenario (below) also at the
  `record_at` counts below the budget, after exactly that many evaluations —
  inside a generation if need be, so that `record_at = [100, 300, 1000, 2500,
  ...]` reads small budgets too, where a generation is much of the budget:
  its trajectory indicators go to `meta.json` (`archive_at`, keyed by the
  count) and its points to `archive_at.npz` (objectives, and variables by the
  snapshots' rule). The end of the run is `final_archive`.
- `native = "auto"` (the default) lets the core evaluate a problem of the C++
  suite itself, without calling Python, where the run has no evaluator of its
  own (no noise, typed variables, baseline or scalarization method,
  `minimize.direct_path`); the rows reach the campaign's counts, archive,
  checkpoints and hard budgets a batch at a time, before every record and at
  the end, and the files are those of `native = "off"`, which evaluates
  through the campaign's evaluator as before. `meta.json`'s `timing.native` is
  the time of the C++ evaluations.
- `hard_budgets = [100, 300, 1000]` reads those budgets the hard way, for
  budgets of a few generations, where the first record at or after b (the
  rule of `record_at` and `--at`) can add a whole population — at N = 200
  and b = 300, 400 evaluations. The population at b is the last record at or
  below b, and the archive at b is the run archive after exactly b
  evaluations, reduced by DSS to `answer_k` points (0: the problem's
  population size), so that configurations of different N answer with as
  many points. Both go to `meta.json` (`hard_at`, keyed by the budget, the
  record's evaluations and generation beside its indicators); it needs a
  record every generation (`record_grid = "generations"`, `record_every =
  1`). A run is not cut at b: its last generation may pass it, and the
  reading ignores what it evaluated beyond.
- `pop_fit = "nearest"` fits a population an algorithm does not accept (a
  Das-Dennis lattice, a multiple of K) to the accepted size nearest to it —
  21 for 20 at six objectives, where the default `"below"` takes the largest
  size up to it, 6 — and lets the config ask for such a size at every
  objective count; `pop_note` in `meta.json` says what was run.
  `ladder_min_gens = 3` leaves out the rungs whose budget is under three
  populations.

**What a campaign keeps between runs.** The worker processes live for the
whole campaign, each taking one job after another, so imports and numba's
compiled WFG kernels (cached on disk) are paid once a process.

**The cache of references.** A problem's frame, reference front of `n_ref`
points, dense reference (four to ten objectives), Pareto-set sample and the
IGD+ floors of `--cover` at its population are the same for every run of it,
and some take minutes to build (ZCAT's at ten objectives, DAS-CMOP's). For
every registry problem with an exact reference front they are built once per
machine into the user's cache — `%LOCALAPPDATA%\mootation\references\v<reference
version>\` on Windows, `~/.cache/mootation/references/v<...>/` elsewhere
(`$XDG_CACHE_HOME` if set), or wherever `MOOTATION_REFS_DIR` (`--refs-dir`)
says — one file per problem and `n_ref`:

```bash
mootation refs build --workers 20                 # every problem with an exact front, n_ref = 1000
mootation refs build --problems ZDT1,DTLZ2_5D --n-ref 2000
mootation refs build --rebuild                    # build them again
mootation refs build --extend --n-ref 5000        # larger samples beside the smaller ones
mootation refs check                              # which are missing or old
```

(`python -m mootation refs ...` where the console script is not installed.)
`build` builds only what is missing or old: an entry is old when the problem's
version or the reference builder's version is not the one it was built with,
and `check` lists those, and the entries built by other benchmark sources of
the same versions. With `--extend`, where the builder's sample of n points is
the first n of its larger sample (the fronts taken in DSS order: DAS-CMOP,
DOC, LIR-CMOP, Polygon) the entry records that the smaller sample is kept in
the larger one, elsewhere that the larger was built anew. Each entry holds a
hash of its arrays; every machine builds the same arrays, so a cache can be
copied between machines.

A campaign checks before its first run that the entries of all its problems
are there and current, and builds the missing ones itself, in parallel on its
workers, before any run starts; `--no-build-refs` stops it instead with the
list. The problems without an exact front — the approximate fronts of RE,
CRE, RCM, their noisy and failing copies — keep theirs in the campaign's
`<results>/_references/<problem>.npz`, read back by every later run while the
reference version and a hash of the benchmark sources stay the same; of the
workers that start on such a problem together one builds it (it holds
`<problem>.lock`) and the others wait for its file. Either way the same arrays,
so the same values.

**The first population.** `init = "sobol"` in an algorithm's entry starts it
from a scrambled Sobol design (`mootation.designs`), `init = "lhs"` from a
maximin Latin hypercube (below), `init = "halton_rand"` from a randomized
Halton sequence (Owen, arXiv:1706.02808, 2017: every digit of every
coordinate through a random permutation of its own, seeded by the run),
`init = "sobol_owen"` from the Sobol sequence under Owen's nested uniform
scrambling (Joe & Kuo's direction numbers, Burley's hash, JCGT 2020 — where
`sobol` is SciPy's linear matrix scramble with a digital shift),
instead of its own random
population: the design is evaluated by the campaign's evaluator, costing what
the algorithm's own start would, and planted as the seed population.
`init_share = 0.5` makes the design a plan of half the budget before the
algorithm: its population is the plan's nondominated points first, the rest
chosen by DSS among the others, and it has the other half to run; such a run
depends on its budget, so a `ladder` gives it a run of its own at each rung.
`minimize(init=..., init_share=...)` does the same outside a campaign.

**Baselines.** `random_search`, `sobol_search` (scrambled Sobol, needs
SciPy), `lhs_search`, `halton_search` (the randomized Halton sequence) and
`sobol_owen_search` (Owen-scrambled Sobol) go in the algorithm list like any
core. They sample the box blindly,
keep the run archive, and answer with the problem's population size chosen
from it by distance-based subset selection on the IGD+ distance — the same
selection that thins every other point set here — so their indicators compare
with a population's. They spend the budget exactly. An algorithm that does not
clearly beat them on a problem says more about the problem or the budget than
about the algorithm.

`lhs_search` lays out one maximin Latin hypercube for the whole budget
(Morris & Mitchell, J. Statist. Plann. Inference 43, 1995): the search of
their Fig. 1, annealing φ_p of the squared Euclidean distances by swapping
two levels of a variable, once for each p of their Section 5 (1, 2, 5, 10,
20, 50, 100), the best by the maximin order itself. Two departures: the search
is cut at 20 000 perturbations (2·10⁷/n above 1 000 points), which still finds
the small optima of their catalog; and a level is a cell of McKay's Latin
hypercube with the point drawn inside it, not the paper's grid i/(n − 1),
whose 0, 1 and, at odd n, 1/2 are exactly the optima of DTLZ1–4's and ZDT's
distance variables (`designs.py`). A smaller budget is another hypercube, so
the baseline is budget-dependent and a `ladder` runs it at each rung.
`sobol_search`, `halton_search` and `sobol_owen_search` draw sequences, whose
first points do not depend on how many follow: a smaller budget is the start of the same run, and
a `ladder` reads it off the trajectory.

The randomized Halton design is stratified at n = b^m in the coordinate of
base b, and nowhere exactly 0 or 1/2. In many variables its bases are large
(71 is the 20th prime, 229 the 50th), and Owen warns that neighbouring
coordinates then line up; measured, at 30 and 50 variables and 100 to 1 000
points it is close to uniform points instead — its correlations no larger
than theirs, its discrepancy only 2-15 % lower — while at 10 variables its
discrepancy is a third to a sixth of theirs (`designs.py`). The Owen-scrambled
Sobol design is stratified at n = 2^m in every coordinate and nowhere exactly
0 or 1/2; unscrambled, its points are SciPy's `Sobol(scramble=False)` bit for
bit. Its correlations are lower than Halton's on average, but at 100 points
in 30 and 50 variables a pair of its coordinates can correlate strongly, as
in SciPy's Sobol — a property of the sequence that scrambling keeps.

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
define the algorithm on bit strings. None of the seven baselines handles
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
| `crossover` | `sbx`; `uniform` (every variable from one parent at random, the other child the complement — JEGA's shuffle_random for two parents); `blx_alpha` (each variable uniform in [min − αI, max + αI], I the parents' distance, `blx_alpha` = 0.5); and four multi-parent, rotation-invariant ones: `spx` (uniform in the parents' simplex expanded √(n+2) times about its centre, n + 1 parents), `rex` (centre + Σ ξ_j(x_j − centre), ξ_j ~ N(0, 1/n), n + 1 parents), `undx` (a normal distribution along the line of two parents, spread across it by a third; three parents, the symmetric pair of children), `pcx` (a normal distribution about one of three parents). The pair is the algorithm's own, the other parents come from its own mating selection, all different; their parameters are the papers' (`operators/multi_parent.hpp`). Differential evolution: `de`, DE/rand/1/bin (`F` = 0.5, `CR` = 1: the whole vector moves, rotation-invariant), and `de_pbest`, JADE's current-to-pbest/1 (x_pbest one of the algorithm's best `de_p` = 0.1 by its own order, r2 also from an archive of displaced parents, `de_archive`); with `de_adapt = "shade"` F and CR come from a SHADE memory (`de_memory` = 5 cells; success = the child survived the selection; `de_adapt_weight` `equal` or `contribution`), and the memory and the means of CR and F drawn and kept are recorded with `operator_stats` (`de_mu_cr_mean`, …, `de_cr_success_mean`) — `operators/de_crossover.hpp`. `portfolio`: every mating's operator drawn from `portfolio` (`["sbx", "de", "pcx", "undx"]`), `portfolio_credit` `uniform` or `borg` (∝ the operator's children among the non-dominated survivors + 1); each operator's share of the children and the share of them kept are recorded (`portfolio_share_<op>`, `portfolio_kept_<op>`) |
| `mutation` | `polynomial`; `gaussian`, x + s(ub − lb)·N(0, 1) with `mutation_scale` s = 0.1; `cauchy`, the same with C(0, 1) and s = 0.05; `uniform_reset`, U(lb, ub); `mixture` (`mixture_cauchy`): the polynomial step with probability 1 − `mixture_q` (0.1), the gaussian (Cauchy) one otherwise; `none`, no mutation. Every one mutates each variable with probability `pm`, 1/n by default; `mutation_scale` must be > 0 |
| `bound_repair` | for the operators above that can leave the box (BLX-α, the four multi-parent crossovers, de, de_pbest, gaussian, Cauchy): `reflect` by default, provisionally, until experiment E3 picks one; the multi-parent crossovers and DE refuse `resample` |
| `step_share`, `F_spread` | EMyO/C only, the knobs that test its mechanism: the clamp of the difference vector, ±`step_share`·(ub − lb) (0.5, the paper's), and F drawn from U[F − `F_spread`, F + `F_spread`] per offspring (0; `F` is 1, no factor, as in the paper). With `bound_repair` (`clip`, the paper's Eq. 5) they take the exact hits of 0.5 and of the bounds apart (`emyo_c.hpp`) |
| `sbx_var_prob` | SBX's share of crossed variables, 0.5 (the canonical realcross) or anything in [0, 1], for this run only; any core using SBX |
| `dn_cf` | DN-NSGA-II: the crowding factor of its niched mating pool — a random member meets the nearest, by Euclidean distance in the variables, of `dn_cf` others drawn without replacement (all of them from N − 1 on), and the better of the two by rank and crowding distance becomes a parent; a whole number ≥ 1, 20 by default (`algorithms/dn_nsga2.hpp`, DN-1) |
| `hv_algo` | SMS-EMOA and MO-CMA-ES: the exact contributions' hypervolumes by WFG (`wfg`, the default) or by box decomposition (`box`; Lacour, Klamroth & Fonseca, Comput. Oper. Res. 79, 2017; `hv_box.hpp`) — the same values up to the order of the sums, so two nearly equal contributions may compare the other way and the run take another path; `metrics.hypervolume(..., hv_algo=)` and `_core.hypervolume(points, ref, algo)` likewise. HypE's exact part is not a hypervolume (its I_h^k): it reports the knob ignored |
| `hv_contrib` | SMS-EMOA and MO-CMA-ES: the exact contributions at three and four objectives by WFG (`wfg`, the default) or by HVC3D and HVC4D (`hvc`; Guerreiro & Fonseca, IEEE TEVC 22(3), 2018; `hvc.hpp`). At M = 3 the HVC3D structure is kept from step to step and brought to the next worst front by the paper's single-point updates (one offspring in, one point out), then all contributions in linear time; at M = 4 HVC4D, O(μ²) a step. It acts only where the contributions are exact: at M = 3, and at M = 4 with `n_samples = 0` — by default M = 4 is Monte-Carlo and the knob changes nothing there, nor at M = 2 (EJOR's Eq. 6) or M ≥ 5. The values are WFG's up to the order of the sums, not bit for bit, so two nearly equal contributions may compare the other way and the run take another trajectory; at M = 3 and 4 it takes over from `hv_algo`. R2-EMOA and HypE report it ignored |
| `n_samples` | SMS-EMOA, MO-CMA-ES and HypE from four objectives: the Monte-Carlo sample points of a step, from which the hypervolume contributions (HypE: its fitness) are estimated — 10 000 by default, shared out over the points of the last front, each sampled in its own box. A step costs in proportion to it and the estimate's noise falls as its square root; SMS-EMOA and MO-CMA-ES take 0 for the exact contributions at every M (WFG, whose cost grows steeply with M and the front), HypE a whole number ≥ 1 (`sms_emoa.hpp` SMS-2, `mo_cma_es.hpp` MOCMA-5, `hype.hpp`) |
| `survival`, `spu_frac` | SMS-EMOA, NSGA-II and NSGA-III: `greedy` (the default, the papers' selection) or `stochastic`, the stochastic population update of Bian, Zhou, Li & Qian (IJCAI 2023): a random Q′ of ⌊`spu_frac`·\|Q\|⌋ members of parents and offspring (1/2 in SMS-EMOA, 3/4 in the NSGA's, the paper's) goes through the usual selection alone — its fronts, reference point, extremes, normalisation — and every member outside Q′ survives untouched; any member survives a step with probability at least 1/2 (SMS-EMOA) or 1/4. `spu_frac = 1` is greedy bit for bit. For NSGA-III Q′ has ⌈`spu_frac`·2N⌉ members, Opris's ⌈3µ/2⌉ (IJCAI 2025, Algorithm 1; NSGA-II takes Bian's floor, the same at an even N) |
| `inner_archive`, `inner_archive_cap`, `archive_reuse_p` | SMS-EMOA, NSGA-II, NSGA-III, IBEA-ε+: `inner_archive = "nd"` keeps every non-dominated point the algorithm evaluated (an equal objective vector is not added; under constraint handling feasible points only), cut to `inner_archive_cap` by DSS after every addition (0: no cap; unset: 10 μ, none for a purely binary genome); each parent, with probability `archive_reuse_p` (0), is a member drawn uniformly. The result's `inner_archive` holds it; `operator_stats` the share of the parents drawn from it (`archive_parent_share`). The answer's archive (`[campaign] archive`) is apart and unchanged (`operators/inner_archive.hpp`) |
| `bin_crossover`, `mating` | for the paper's theory runs: `bin_crossover = "none"` (SMS-EMOA, NSGA-II) leaves the binary genome uncrossed (`uniform` by default); NSGA-II's `mating = "each"` makes every member a parent exactly once, its child its mutation (`tournament` by default), and needs no crossover on any part of the genome — `crossover = "none"`, which gives the parents as they are, with real variables |
| `coords`, `coords_warmup`, `coords_update` | where the crossover and the mutation act: `axis` (the default, as always); `pca`, in y = (x − μ) R, R the eigenvectors of the population's covariance by decreasing eigenvalue, μ its mean; `pca_nd`, the same from its non-dominated members (all of it while fewer than three); `randrot`, a fresh random rotation at every update, the control. R = I for the first `coords_warmup` generations (10) — counted in generations, not in a share of the budget, so the algorithm stays budget-independent — then R and μ every `coords_update` generations (1). In y there is no box: every crossover runs unbounded, SBX in its unbounded form, and the mutation acts there too (else it is axis-wise again), within the projected population's range widened by half its span; the children go back to x, x = y Rᵀ + μ, and through `bound_repair` (reflect by default). The basis is complete with fewer points than variables. mosa's sbx step takes it too, the basis learned from its archive (`operators/learned_coords.hpp`, `variation.hpp`). With `operator_stats` the spectrum is recorded (`coords_ratio`, `coords_lead_share`); `[campaign] coords_diag = true` adds the principal angles between the learned minor subspace and the problem's distance block (`coords_angle_max`, `coords_angle_mean`; 0°: found), where the block is known (ZDT, DTLZ, WFG, MaF, ZCAT, Polygon and their shifted, rotated and noisy copies; `run/coords_diag.py`) |

In MOEA/D-DE, `polynomial` is its literal Eq. 7 and every other mutation
leaves its steps raw for Step 2.3's repair of the whole offspring, as Eq. 7
does. The operators live in `operators/real_crossover.hpp`,
`operators/multi_parent.hpp` and `operators/real_mutation.hpp`, with
DE/rand/2/bin and DE/best/1/bin (Storn & Price's DE/x/y/z notation) in
`operators/de_mutation.hpp`. The crossover and mutation are switchable in
NSGA-II, IBEA-ε+, SPEA2+SDE, AGE-MOEA and SMS-EMOA; MOEA/D-DE takes the
mutation.

**Crowding in the decision space.** `crowding_space = "decision"` makes
NSGA-II compute its crowding distance over the variables instead of the
objectives: the same formula, each variable's gaps over its range on the
front. Among equally ranked solutions it keeps those far apart in the
PARAMETERS, so that different parameter sets with the same objective values
survive side by side. It is one of the two changes of DN-NSGA-II (Liang, Yue
& Qu, CEC 2016), which swaps the crowding in the objectives for crowding in
the decision space without giving a formula; the other, a niched mating pool
(a random solution competes with the nearest of a few random others), is
DN-NSGA-II's own algorithm, `dn_nsga2`: NSGA-II with both, the mating pool's
crowding factor `dn_cf` (20 by default; the paper leaves it open,
`algorithms/dn_nsga2.hpp` argues the value). The knob alone does not make
NSGA-II into DN-NSGA-II. Default `objectives`, NSGA-II's (`decision` in
`dn_nsga2`).

**DMS.** Direct MultiSearch (Custódio, Madeira, Vaz & Vicente, SIAM J.
Optim. 21(3), 2011) is one of the two algorithms here that are not
evolutionary (the other is MOSA, below): a
list of nondominated points, each with its own step size, polled one at a
time along ±every coordinate (2n evaluations a poll). It is deterministic —
the seed changes nothing — and stops by itself when every step size is below
10⁻³ of the variable's range, which can be before the budget runs out. Its
answer is the list reduced to `pop` by DSS, so `pop` is the size of the
answer, not of a population. `dms_init` chooses the initial list: `line`, n
points on the box's diagonal (the paper's best variant, the default), or
`single`, the box's centre. A problem whose optimum sits at the centre of the
box (ZDT4's g) is solved by `single` at the first evaluation; see the
structural-bias campaign before reading anything into that. The line's points
and their halved steps land exactly on the middle and the bounds of the box,
where DTLZ1-4 and ZDT have their optima, and every seed is the same run; to
tell its search from both, `dms_init = "random"` starts from a Latin
hypercube of n points drawn from the seed and `dms_poll = "random"` polls a
point drawn among those still worth polling (`first`, the default, the first
in list order), the directions in random order (DMS-6 in `dms.hpp`).

**SMPSO.** Speed-constrained multi-objective PSO (Nebro et al., MCDM 2009;
`algorithms/smpso.hpp`): `pop` particles, a leaders archive of `pop`
non-dominated points by crowding distance, and the archive as the answer.
Each step moves the whole swarm (one batch of `pop` evaluations): a leader by
a binary tournament on crowding, the velocity of Eq. 2 times the
constriction factor χ, bounded to ±`vmax_frac`·(ub − lb), the position put
back on the box with the velocity multiplied by a factor, a turbulence on a
share of the swarm. What the paper leaves open is a preset:

| preset | χ | w | velocity bound | at the bound | turbulence | answer |
|---|---|---|---|---|---|---|
| `jmetal` (default) | as Eq. 3 prints it (negative for C1 + C2 > 4) | 0.1 | ½ width | clip, velocity × −1 | polynomial (η = 20, 1/n) on i % 6 == 0 | leaders, crowding, one removed at a time |
| `platemo` | its modulus | U[0.1, 0.5] | ½ width | clip, velocity × 0.001 | polynomial on each particle with probability 0.15 | leaders |
| `fair` | its modulus | 0.1 | ½ width | `bound_repair = reflect`, velocity × −1 | as jmetal | as jmetal |
| `omopso` | 1 | U[0.1, 0.5], C1, C2 ~ U[1.5, 2.0] | none | clip, velocity × −1 | non-uniform / uniform / none by thirds | leaders: only the particles that replaced their best, cut once a step; the answer an ε-dominance archive (ε = 0.0075) of every leader accepted |

Every cell is a knob (`chi_mode`, `w_min`/`w_max`, `c_min`/`c_max`,
`vmax_frac`, `bound_repair`, `vel_on_bound_low`/`vel_on_bound_up`,
`turbulence` with `turbulence_k`/`turbulence_p`, `eta_m`, `pm`,
`archive_size`, `archive_prune`, `leader_select`, `leader_offer`, `r_per_dim`,
`eps_archive_eta`), applied over the preset whatever the order. The run
reports every knob as it used it in `Result.settings`, and a campaign writes
them to `meta.json` under `settings`. The omopso preset's non-uniform mutation
narrows over the whole budget, so with `turbulence = thirds` the run depends
on `max_evaluations` (`run/budget.py`, `depends_on_budget`); the other
presets do not. Real variables with finite bounds only.

**R2-EMOA** (`algorithms/r2_emoa.hpp`): SMS-EMOA's steady state — two
parents uniformly, SBX and polynomial mutation, one offspring a step — with
the worst front's point of least R2 contribution removed instead of the least
hypervolume (Brockhoff, Wagner & Trautmann, Evol. Comput. 23(3), 2015,
Algorithm 1). The paper's settings: crossover probability 0.9, η_c = 15,
η_m = 20, 500 weights. `r2_mode = "exact"` (the default, two objectives) takes
the R2 of a continuous uniform distribution of weights (Schäpermeier et al.,
2024); `"discrete"` the paper's `r2_weights` weights, uniform in weight or,
with `r2_space = "angle"`, in angle space; from three objectives always
discrete, on a Das–Dennis lattice. The utopian point is the ideal point of
every evaluation so far, less `r2_offset` of the range; `normalize = true`
scales the objectives first.

**BC** (`bc_spx`, `bc_rex`; `algorithms/bc_spx.hpp`, `bc_rex.hpp`): Tanabe & Ishibuchi's
non-elitist best-children selection (GECCO 2019): k = n + 1 parents drawn
from the population, λ = 10n children of the same parents by SPX
(ε = √(n + 2)) or REX (σ² = 1/(k − 1)); the parents always leave, and the k
best children by non-domination level and crowding distance — ranked among
the rest of the population — take their places. `bc_selection = "bf"` (the
best k of children and parents) and `"ba"` (the best μ of all) are the
paper's two elitist comparisons; `bc_lambda` sets λ, `crossover` switches
spx and rex. The paper's population is ⌊100 ln n⌋, and pop_size must be at
least n + 1. The population is not elitist: the paper measures an unbounded
archive of every non-dominated point, a campaign's archive scenario here.

**PAES** (`algorithms/paes.hpp`): the (1 + 1) Pareto archived evolution
strategy of Knowles & Corne — one mutant of the current solution a step,
accepted by dominance and, between mutually non-dominated ones, by the
crowding of an adaptive grid over the archive (2^`paes_depth` intervals per
objective; 5, the paper's). The archive, bounded by `archive_size`
(pop_size), is the answer. Polynomial mutation (`eta_m`, `pm`) for the
papers' bit flips. A grid of 2^(5·M) cells is far finer than an archive of a
hundred points from three objectives on, where every point has a cell of its
own and a full archive loses its first member: a smaller `paes_depth` there.

**RM-MEDA** (`algorithms/rm_meda.hpp`; Zhang, Zhou & Jin, IEEE TEVC 2008):
an estimation of distribution algorithm. Each generation an (m − 1)-dimensional
local principal component analysis splits the population into `n_clusters`
(5) clusters; each cluster's model is the box spanned by its points along its
first m − 1 principal components, extended by `rm_extension` (0.25) of that
span on each side, plus Gaussian noise whose variance is the mean of the
other eigenvalues. `pop_size` offspring are sampled, a cluster drawn in
proportion to its box's volume, and `pop_size` survive from parents and
offspring by non-dominated sorting, the last front cut one member at a time by
the paper's crowding distance (the average side of the cuboid, not
normalized). A variable sampled outside the box is put on the bound
(`bound_repair = "clip"`); the paper's text resets it at random inside the
box (`native`, the same as `random`), but its own curves are reached with the
variable kept at the bound, and a random reset keeps the population off a
Pareto set that lies on a bound (RM-8 in the header). `pop_size` must exceed
`n_clusters`; the paper uses 100 at two objectives and 200 at three. Real
variables only.

**Scalarization with a local solver** (`python/mootation/scalar/`, task 9):
the algorithm `scalar` and its presets `eps_sqp`, `tch_sqp`, `nbi_sqp`,
`nnc_sqp`, `eps_nm`, `eps_cmaes`, Python with NumPy and SciPy. A run is a plan
of single-objective subproblems:

1. the initial plan — `pop_size` uniform points, or the design of `init`
   (with `init_share`), or the seed population;
2. the anchors: for each objective the lexicographic minimum of (f_k,
   f_{k+1}, ...), which gives the normalization (z* the utopia point, z^N the
   pseudo-nadir of the anchors);
3. the subproblems of `sub_plan`, each started from the archive's best point
   for it (feasible first, then its value; again from the initial plan's best
   when the solver does not move): `grid` the lattice in order, so that a
   subproblem starts next to its neighbour's solution; `laumanns` the adaptive
   ε-constraint scheme of Laumanns, Thiele & Zitzler (eps only); `gap` ours:
   of a lattice four times finer, the subproblem farthest from the points found.

| knob | values |
|---|---|
| `scalarization` | `eps` min f_k with the others bounded; `tch` the smooth Chebyshev form min t, w_i F̄_i <= t (`tch_weights`: `inverse`, w ∝ 1/d so that the solutions lie on the lattice's rays, or `lattice`, w = d); `nbi` Das & Dennis's max t, Φβ + t·n̂ = F̄ with the quasi-normal; `nnc` min F̄_m under the normal constraints; `ps` Pascoletti–Serafini, min t with F̄ <= Φβ + t·r |
| `local` | `sqp` (SLSQP; the problem's inequalities and equalities as its constraints), `nm`, `cmaes` (both with a penalty in which a feasible point is always better) |
| `n_sub` | subproblems after the anchors; 0: from the budget — what is left over 10 (n + 1) evaluations for sqp, 100 n for the others, at most `pop_size` |
| `sub_budget` | evaluations a subproblem may take; 0: what is left over the subproblems left (what one does not use, the next get) |
| `fd_step`, `local_tol`, `cma_sigma0`, `eps_objective` | sqp's forward-difference step in the box normalized to [0, 1] (1.49e-8); the solvers' tolerance (1e-14); CMA-ES's first step in the normalized box (0.3); the objective eps minimizes (1) |

The run ends when the plan is done or the budget spent: a plan that ends
early leaves evaluations unspent, as DMS does. `result.solutions` holds every
subproblem's solution in the order solved — the anchors first —, a diagnostic
beside the answer, which is the archive's feasible non-dominated points by
DSS. The plan is laid out over the budget, so these runs depend on
`max_evaluations` (`run/budget.py`). Only `constraint_mode = "feasibility"`;
real variables only. The header of each module lists what it reads where the
sources are silent (SCALAR-1 ... 12 in `scalar/__init__.py`).

**MOSQP** (`python/mootation/scalar/mosqp.py`; Fliege & Vaz, SIAM J.
Optim. 2016, Algorithm 4.1): the extreme points first (min f_i under the
constraints); then `mosqp_spread` (20) iterations that spread a set of
pop_size points from the line between the bounds — every point gives, for
every objective, the SQP direction of that objective under the linearized
constraints and a step by Armijo's rule on an ℓ1 merit function — the set
kept non-dominated and cut by crowding distance; then every point refined
along the direction that improves all the objectives' quadratic models at
once, until the direction is below `mosqp_tau` or the step below
√`mosqp_tau` (in the spread, a direction below `mosqp_tau`^(1/4) or a step
below √`mosqp_tau` gives no point). Gradients by forward differences
(`fd_step`), counted; H_i = I; `mosqp_mu` (10⁻⁴) and
`mosqp_beta` (1/2) are Armijo's constant and factor. The preprint leaves τ,
μ, β and the set's size open: the header says what is chosen. Equalities and
inequalities go to the subproblems as constraints; an infeasible
subproblem sends the point to a restoration.

**Polishing** (`polish`, every algorithm; `scalar/polish.py`): `polish = "sqp"`
or `"nm"` keeps `polish_share` (0.2) of `max_evaluations` for the end. The
algorithm runs on the rest; then `polish_points` (0: from what the share buys,
at most `pop_size`) of the feasible non-dominated points of everything it
evaluated, chosen by DSS, are each improved in one objective — round the
objectives from one point to the next — with the others no worse than they
were (to the subproblem's slack), and every point it evaluates is counted.
The answer is the non-dominated points of the algorithm's answer and the
polished points; cut to `pop_size` when there are more, it keeps first a
point weakly dominating each point of the algorithm's answer, then the rest
by DSS, so nothing of that answer is lost. `polish = "off"` (the default)
changes nothing. A run with polishing depends on its budget; real variables
only.

**DMulti-MADS** (`algorithms/dmulti_mads.hpp`; Bigeon, Le Digabel & Salomon,
Comput. Optim. Appl. 2021): a list of non-dominated points, each with its own
frame size Δ (mesh δ = min(Δ, Δ²)) in the box normalized to [0, 1]^n. A step
polls the point of the largest Δ — of several, the one in the least crowded
part of the list (`mads_select = "spread"`, or `"first"`) — along n + 1
directions rounded to the mesh, after a speculative point along the direction
that found it (`mads_search`); success is a point dominating it; new points
that dominate or extend the list get the frame Δ/τ, those that fill it Δ, a
failure shrinks the poll centre's frame to τΔ (`mads_tau` 1/2, `mads_delta0`
0.1). `mads_poll = "opportunistic"` stops a poll at its first success. The
poll directions come from the seed; the run ends when the mesh is below 1e-9.
The paper's granular mesh and its n + 1 directions with a target direction
come from papers that were not at hand; the header says what stands in for
them. Constraints by the extreme barrier.

**MultiGLODS** (`algorithms/multiglods.hpp`; Custódio & Madeira, J. Glob.
Optim. 2018): DMS's list with a comparison radius and an active flag for each
point. A new point far from every point of the list starts a search of its own;
one that dominates a nearby active point takes over, and a dominated one is
kept inactive (the searches merge). The search step — the next n points of a
Sobol sequence — runs after `glods_search_every` (3) failures in a row; the
poll centre is the active point of the largest step that is not yet local
(`glods_local_tol` 10⁻³). The answer is the non-dominated part of the list.
Deterministic, as dms: no random number below 101 variables.

**BoostDMS** (`python/mootation/scalar/boostdms.py`; Brás & Custódio,
Comput. Optim. Appl. 2020): DMS with a search step before each poll — quadratic
models of every objective from up to (n + 1)(n + 2) evaluated points, 80 %
from the ball of three trust radii around the poll centre; each model by
regression, minimum Frobenius norm or linear interpolation as the points
allow; minimized in the trust region by the adapted ε-constraint of Custódio,
Pozzi & Silva (2026; `boost_scalarization = "eps"`, the default: one
objective, the others no worse than at the centre) or by the paper's min-max
levels (`"minmax"`). A new non-dominated point skips the poll. Python with
SciPy; deterministic.

**MOSA.** Multi-objective simulated annealing (`algorithms/mosa.hpp`):
annealing chains over an archive of every non-dominated point evaluated. It is
not one paper's algorithm but a composition: DBMOSA's unconstrained archive,
energy (the share of the archive dominating a point), Laplace move and restarts
from the archive, and the chains of Pareto simulated annealing with weights that adapt to
push the chains apart; the header says which part is whose and where it
departs from the papers. One step is one evaluation, so `pop`
is the size of the answer — the archive reduced by DSS, published every `pop`
steps and at the last — not of a population, and annealing is sequential:
`batch=` and a cluster's evaluators see one point at a time. The temperature
and the step scale fall geometrically over the whole budget, so the run depends
on it (`run/budget.py`; a ladder runs each rung apart).

| knob | values |
|---|---|
| `sa_energy` | `cmb` (the default: the share of the archive dominating the point where it is above 0, minus the additive ε of the archive against the point elsewhere), `eps`, `dom`, `tch` (Chebyshev, each chain its own weights) |
| `sa_step` | `mix` (the default: SBX with a random archive member, then polynomial mutation, or a differential step, half and half), `laplace` (DBMOSA's move, here each variable with probability 1/n, scale s·(ub − lb)), `de` (x + F(x_a − x_b) with probability `p_de` = 0.7, otherwise Laplace), `sbx`, `adaptive` (laplace, de or sbx with probability ∝ its successes in the last `sa_adapt_window` = 200 steps + 1) |
| `F`, `p_de` | the differential step's factor (0.5) and, in `de`, its probability (0.7) |
| `sa_chains` | chains over one archive, advancing in turn: 1, or 10 with `tch` |
| `sa_reseed` | the probability per step of restarting a chain from the better of two random archive members (1/50) |
| `sa_t_ratio` | T at the last step over T_0 (10⁻⁴); T_0 is the median energy difference between start points over ln 2 |
| `sa_s0`, `sa_s_end` | the Laplace scale over ub − lb at the first and the last step (0.2, 0.002) |
| `sa_archive_cap` | 0 (the default) keeps every non-dominated point; above 0, DSS thins the archive to it |
| `sa_weights`, `sa_psa_alpha` | `tch` only: `psa` (the default) multiplies a chain's weight by α = 1.05 where it beats its nearest non-dominated neighbour chain and divides it elsewhere; `fixed` keeps the start weights |
| `sa_tch_accept` | `tch` only: `weighted_sum` (the default, PSA's) compares the weighted sum of the objectives' differences, `chebyshev` the energies |
| `eta_c`, `eta_m`, `pm`, `bound_repair` | SBX's and polynomial mutation's (20, 20, 1/n); `reflect` by default, `resample` and `native` refused |

With `operator_stats` a run records the temperature `sa_T`, the step scale
`sa_s`, `sa_archive_size` and `sa_reseeds`; with `tch` the weights' mean and
standard deviation over the chains, `sa_w_mean_<j>` and `sa_w_sd_<j>`; with
`adaptive` the probabilities `sa_p_laplace`, `sa_p_de` and `sa_p_sbx`. Under
constraints the archive keeps the feasible points (the least violating one
while none is), and a chain moves from a feasible point to a feasible one only.

**GDE3, SMS-EMOA, MO-CMA-ES.** Three classics the library lacked. GDE3
(Kukkonen & Lampinen, CEC 2005) builds a DE/rand/1/bin trial for every
member and keeps both when neither dominates, pruning back to `pop` by
non-dominated sorting and crowding distance recomputed one removal at a time;
`F` and `CR` are knobs, 0.2 and 0.2 by default (the paper's setting on its
test problems), and at `CR = 1` the trial is rotation-invariant. But with the
paper's F = 0.2 that step is small — a fifth of the difference of two
members, along every variable at once; give `CR = 1` a larger F (0.5, DE's
usual) when rotation invariance is the point. SMS-EMOA
(Emmerich, Beume & Naujoks, EMO 2005) and the steady-state MO-CMA-ES (Voß,
Hansen & Igel, GECCO 2010) spend one evaluation a step and drop the point
with the smallest hypervolume contribution in the worst front — exactly at
two and three objectives, by Monte-Carlo from four
(`include/mootation/hv_contribution.hpp`). SMS-EMOA's reference point is the
nadir plus one in every objective, as its author's thesis defines it, so it
depends on the objectives' units; MO-CMA-ES adapts a step size and a
covariance matrix per individual and is the first operator here that is
invariant to rotating and rescaling the search space. Both hypervolume
algorithms are slow by construction on five or more objectives.

**What the operators did.** `operator_stats = true` in `[campaign]` adds to
every trajectory record, for the offspring evaluated since the previous one,
and to `final` for the whole run:

| field | what it counts |
|---|---|
| `oob_share`, `oob_var_share` | offspring with a variable outside the box before repair, per evaluation, and the share of their variables that were (the measure of Kononova, Caraffini & Bäck, Inf. Sci. 581, 2021). Liu & Li's offspring pass two repair sites, crossover and mutation, and count at each. Every repair goes through the count: an operator's own, and since 2026-10-01 also the store's clamp of a value written outside the box, logged as `vault` — so 0 now says no offspring left the box (EMyO/C's step, HCCA's DE and MaOEA-IAMD's noise used to be clipped past it) |
| `survival_share` | offspring that entered the next population: it holds more copies of their variable vector than the parents did, so a clone of a surviving parent does not count and a child copied into several MOEA/D neighbourhoods counts once |
| `offspring_nd_share` | offspring that no member of the parent population dominates |
| `step_mean` | the distance from an offspring to the nearest member of the parent population, over the box diagonal ‖ub − lb‖ |

The relations are to the parent POPULATION, the one before the step: the
library does not track which individuals an offspring came from across its
75 cores, so "not dominated by its parents" is read as "by any parent", and
the nearest parent as the nearest member. The statistics cost milliseconds
and never change a run: the populations are the same with them on or off.
They are off by default and not computable afterwards, so `--recompute`
refuses them; `--compare survival_share` and the other tables read them.

**Structural bias.** Where does an algorithm search when the objectives say
nothing? On the uninformative problems every objective value is a random draw
that does not depend on x — drawn from the run's seed and the evaluation's
number, so a campaign run gets its own stream (`BenchProblem.make_evaluator`)
— and no region of the box is better than another. Points piled at the bounds
or drawn to the centre show a preference of the algorithm's own, which it
carries onto every problem; Kudela, van Stein, Bäck & Kononova (GECCO 2026)
found such pulls in 120 PlatEMO algorithms. The problems are theirs (Table 1),
in two and ten variables: `uninformative_n02_2D` is f1, [U, U]; f2b, f3b and
f4b (`uninformative_f2b_n02_2D`, ...) put a disconnected, a stepped and a
two-region front under a χ²-shaped noise (variant β); f5 is [a, −a], every
point non-dominated. The problem itself counts, in 20 bins of [0, 1] and
exactly on a bound or at ½, the coordinates of every point a run evaluated and
of those non-dominated over the run, and the campaign writes the counts into
`meta.json` (`structural_bias`).
[`python/examples/structural_bias.toml`](../python/examples/structural_bias.toml)
runs every algorithm on f1–f4b at the paper's 30 000 evaluations and thirty
seeds, [`structural_bias_f5.toml`](../python/examples/structural_bias_f5.toml)
on f5 with ten seeds and DMS at 10 000 (every algorithm truncates its
non-dominated set at every evaluation there). `--bias` reads them as the paper
does (Section 3.3): per problem and algorithm the binsizes n_k / (N/K) over 20
bins of all coordinates of all runs — `bound` the mean of the two outer bins,
`centre` of the two middle ones, 1 when uniform — for the final populations
(L), the runs' non-dominated points (P) and every evaluated point (A); the
shares of coordinates exactly on a bound and exactly at ½, the trace of a
clamp or of exact arithmetic; a χ² test on one point drawn from each run's
final population (independent across runs, unlike the points of one
population); the Clark–Evans index of the final populations at two variables
(the mean nearest-neighbour distance over 0.5/√N: ~1 uniform, below 1
clustered); and a class by our thresholds, which the paper does not give —
`bound` when the outer bins exceed 1.5 and the middle ones do not, `centre`
the other way round, `mixed` when both do, `empty edges` below 0.67.

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
the scenario existed, or with `archive_scenario = false`, selects them for the
first time: the archive is written either way. With `archive_variables =
"selected"` it reads the selection the run kept, `final_archive.csv.gz`.

How the archive becomes an answer is `answer_select`: `"dss"` (the default,
below) or `"igdp_self"`, the N points of the smallest IGD+ against the archive
itself thinned by DSS to 1 000 points — chosen greedily from the whole
archive, without a reference (`archive.igdp_self_order`); it applies at the
checkpoints and hard budgets too, and `meta.json` records it. `--oracle`
measures what the search found apart from that choice: for every finished
run, the best N points of its archive under IGD+ against the reference front
(a greedy choice that knows the reference, so no run could answer with it),
beside the population, the archive answer, what `"igdp_self"` would choose and
the whole archive, all as `igdp_norm` (`<results>/_oracle/oracle.csv`).

DSS (`archive.dss_order`, and `dss.hpp` for DMS's answer) starts from the best
point of every objective and then takes the point the selection covers worst.
Both steps used to favour *almost-dominated* points — the best value of one
objective bought with a much worse value of another. Such a point is now set
aside when some other point is better than it by more than 0.1 in some
objective while it is better than that point by at most 0.001 everywhere
(normalised units: a trade-off beyond 100 to 1 over a tenth of the range); the
points set aside come after all the others, so a selection of N contains none
of them while enough others remain. Without the problem's frame, the ideal and
nadir are taken from the points kept, after every round that sets points aside,
so one outlier no longer stretches the frame the rest is measured in. DSS is a
spread heuristic with no approximation guarantee: nothing bounds the covering
radius it leaves against the best subset of the same size.

Two more readings need no rerun. `--at 0.25` gives `--compare` and `--ranks`
every run as it stood at a quarter of its budget, read from its trajectory, so
ranks at several budgets come out of one campaign — they do differ with the
budget (Tanabe & Oyama, GECCO 2017). The fraction is of the budget asked for
(`budget_nominal` in `meta.json`; `budget_fe` rounds it up to whole
generations, 25 025 at a population of 91), and a run is read at the first
record at or after that many evaluations, exactly as `--cover` reads a budget:
at a population of 91 the record of "2 500" is at 2 548, the end of the
generation a separate run of 2 500 stops at. (Until 2026-10-01 `--at` took the
last record at or below `at × budget_fe` — 2 002 there — and the tables at 10 %
put the ordinary algorithms a fifth of a budget behind the budget-dependent
ones' rungs.) A run that stopped before spending the budget (DMS once its step
is small enough) is read at its end, its answer no longer changing. With
`--scenario archive`, `--at` reads the archive checkpoint of exactly that budget. `--recompute eps,hv_h` computes indicators
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
won and the number it was ranked on, and `short`, the problems where its
median answer has fewer than N/2 points (N its population): an indicator such
as GD+ does not see a short answer, so read its ranks with that column (the
CSV's `short_answers`). `--compare` marks such a cell with `~`.

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
python -m mootation.run.campaign c.toml --seed-distance                  # do the seeds agree in x?
python -m mootation.run.campaign c.toml --eps-table                      # A against B, no reference
python -m mootation.run.campaign c.toml --magnitude                      # experiment D7
python -m mootation.run.campaign c.toml --cover --workers 8              # sets that cover the problems
python -m mootation.run.campaign c.toml --cover --scenario archive       # the same for the run archives
python -m mootation.run.campaign c.toml --behaviour --workers 8          # how the runs behave, grouped
python -m mootation.run.campaign c.toml --portfolio --workers 8          # oracle, Shapley, the matrix
python -m mootation.run.campaign c.toml --oracle --workers 8             # the archives' best N under IGD+
python -m mootation.run.campaign c.toml --instance-space --workers 8     # the problems as a plane, footprints
python -m mootation.run.campaign c.toml --invariance-check --workers 8   # scale and shift of the objectives
python -m mootation.run.campaign c.toml --recompute-trajectory r2,pdist --workers 8   # from the snapshots
python -m mootation.run.campaign c.toml --recompute-reference --workers 8   # against a changed reference front
python -m mootation.run.campaign c.toml --pooled-reference --workers 8      # against the campaign's own front
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
  smallest attainable rank-sum p is 1/126 ≈ 0.008, before Holm — so beyond
  0.025·C(2k, k) problems at k seeds a side (6 at five seeds, 85 at seven) Holm
  leaves no mark but `=` possible at all, and the report says so in a warning;
  `--adjust bh` adjusts by Benjamini–Hochberg instead, which controls the false
  discovery rate and keeps marks possible. Beside the counts: `=L`, the `=`
  with a large effect, how many problems fall in each class of A12
  (`A12 n/s/m/l`: negligible, small, medium, large at 0.06, 0.14 and 0.21
  from 0.5, Vargha & Delaney), and the median A12 with its class. The
  signed-rank test ranks the SIZE of the differences across problems, so use a
  scale-free metric for it (`hv`, `hv_h`, `igdp_norm`, `eps_norm`): in raw units
  the problem with the largest values decides. For `igdp_norm` and `eps_norm`
  it runs on the logarithms of the medians, a problem counting by the ratio of
  two medians; beside it the sign test, and the Hodges–Lehmann estimate of the
  difference — for those two a ratio of medians, below 1 where the algorithm
  is better. The order of the table is by `+` minus `-`: an order of these
  counts, not a rating.
- `--recompute-reference` also measures every finished run on a problem with a
  dense reference against it (below), from the saved answers, into new values
  `<indicator>_ref_v3` beside the old ones, which stay as they are; a run that
  has them is skipped.
- `--pooled-reference`: the campaign's own reference, for problems whose
  reference front is an approximation (`--by front_ref`) or to read every
  problem alike. Per problem, the non-dominated points of all its finished
  runs — their final answers, feasible rows only, and their run archives —
  go to `pooled_reference/<problem>.csv`; `<problem>.json` beside it lists the
  runs that went into it (entry, seed, budget, the commit each ran on, the rows
  each gave), the code that built it and the file's sha256, which is the
  pool's version. Every run is measured against it from its saved answers —
  final, archive, archive checkpoints, trajectory — as `<indicator>_pooled`
  beside the old values, in the pool's own frame (its corners), and records
  the version in `meta.json` (`reference_pooled`). The tables read the new
  names (`--ranks igdp_norm_pooled`). Runs added later change the pool: the
  next call builds it again and measures again every run that names another
  version; a run already measured against the same one is skipped. The pool
  depends on which algorithms ran: compare values against one pool only.
- `--ci`: a 95 % percentile interval for each mean rank, resampling the
  problems 2000 times. Two algorithms whose intervals overlap widely are not
  ordered by this campaign.
- `--by KEY` groups the problems by one property and ranks within each group:
  `front` (linear, concave, convex, mixed, disconnected, degenerate),
  `multimodal`, `deceptive`, `bias`, `scaled`, `separable`, `centre` (the
  optimum in the middle of the box: Huband's medial parameters) and `boundary`
  (the optimum on a bound: his extremal parameters). The values are Huband,
  Hingston, Barone & While's (IEEE TEVC 10(5), 2006, Tables V, VII and XV) for
  ZDT, DTLZ and WFG and are argued in `mootation/benchmarks/properties.py` for
  the rest; unknown is `?`, never guessed. Where a problem has a Pareto-set
  sample, `properties.measured()` reads `centre` and `boundary` off it, and the
  tests hold the table to that. Besides these: `degenerate` and `disconnected`
  (read off the front), `linkage` (a distance variable's optimum depends on the
  position variables: a curved Pareto set), `n_obj`, `n_vars`, and for
  bbob-biobj `bbob_groups`, the function groups of the pair; `multimodal_ps`
  and `ps_parts`, a Pareto set of several parts that each map onto the whole
  front and how many (the multimodal problems, IPolygon, minusDTLZ2; unknown
  elsewhere); `front_ref`, `exact` or `approximate`, whether the reference
  front lies on the true front, and `front_built`, how an approximate one was
  built (RE and CRE: the authors'; RCM: this library's long runs, RCM28's
  exact; C2-ConvexDTLZ2: a checked sample; the failing copies). For a metric
  that reads the reference front or its frame, `--ranks` sets the synthetic
  problems with an approximate front apart, as the group `approx_front`, where
  the campaign has both kinds; `all` keeps those with an exact front.
- `--cec2019-ranks`: the CEC 2019 multimodal competition's protocol — each
  algorithm's mean rank by median IGD, by median IGDX and by median 1/PSP,
  and the mean of the three, over the problems with a Pareto-set sample.
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
- `--seed-distance`: whether different seeds find the same solutions — per
  problem and algorithm the chamfer distance ½[mean over x of min over y
  ‖x − y‖ + mean over y of min over x ‖x − y‖] between the non-dominated sets
  of two runs, variables normalised by the bounds, averaged over every pair of
  seeds, beside `pdist`, the spread inside one run's set. A chamfer well
  below `pdist` says the seeds land on the same solutions. The measure needs
  no reference front, so it would suit a calibration problem as well; a
  campaign, though, runs registry problems only.
- `--eps-table`: the binary multiplicative ε, I(A, B) = max over b of min over
  a of max_i a_i / b_i (Zitzler, Thiele, Laumanns, Fonseca & Grunert da
  Fonseca, IEEE TEVC 7(2), 2003), between every two algorithms on each
  problem, the median over the pairs of runs. A is better than B exactly when
  I(A, B) ≤ 1 < I(B, A), marked `<`. It needs no reference front and no
  frame and does not change when an objective is rescaled, which is what a
  calibration's residuals want ("no worse than B by more than 5 % in any
  residual" is I ≤ 1.05); it needs values ≥ 0, a zero in b being matched only
  by a zero, and a problem with a negative value is left out. On a benchmark
  front that touches 0 in some objective, ratios against values near 0 run
  into the thousands: read it on residuals, not there.
- `--magnitude`: experiment D7, not a metric. On the problems with at most
  three objectives, every final population is measured by the magnitude of
  its dominated region (Emmerich, arXiv 2604.18147, 2026: the sum over the
  subsets S of the objectives of 2^−|S| times the measure of the region's
  projection on S, so a point on the boundary of the reference box counts,
  which it does not for the hypervolume) in `hv_h`'s frame, and by `hv_h`;
  printed are Kendall's tau between the two orders of the algorithms per
  problem and the mean ranks by both. Magnitude joins the metrics only if the
  order it gives is clearly different and the difference can be explained.
- `--cover`: which few algorithms together do well everywhere. An algorithm
  covers a problem at a level and budget b when at least `--cover-seeds` of
  its seeds (7, meant for ten) reach it. The levels go from coarse to fine.
  Where the problem has a reference front, criterion `igdp` reads `igdp_norm ≤
  (1 + τ) × floor`, τ in `--cover-floor-taus` (1, 0.5, 0.25, 0.1: within 2,
  1.5, 1.25 and 1.1 times the floor): N points on the front itself leave IGD+
  above zero — about 0.002 at two objectives, 0.02 at three, 0.06 at five —
  so absolute levels of 0.01 and 0.003 could not be reached at all. The floor
  is the IGD+, against the reference front the runs were measured against, of
  the best N points of that front as far as a greedy search finds them (added
  one at a time for the smallest IGD+, then single swaps); N is the population
  the algorithm ran with, so an M2M core's multiple of K has its own floor.
  Beside it the IGD+ of the N points DSS selects, "an ideal archive reduced by
  DSS", and the ratio of the two, DSS's tax; the population's levels read the
  greedy floor, the archive scenario's the DSS floor, since that answer is
  itself a DSS selection. 1.1 × floor is the ideal level: the best attainable,
  not a target. Both floors go to `igdp_floor.csv`, with the reference version,
  and are read back next time. Criterion `gdp` reads `gdp_norm ≤ τ`, τ in
  `--cover-taus` (0.1, 0.03, 0.01, 0.003): the progress towards the front
  alone, whose floor is 0 — counted only for an answer of at least N/2 points,
  since GD+ does not see how much of the front the answer covers (two points
  on the front reach its finest level);
  `cover_short_answers.csv` lists the runs shorter than that at every budget.
  Where there is no front (bbob-biobj), both read the
  relative gap of `hv_h` to the best any run of the campaign reached on it,
  (best − hv_h)/best ≤ τ of `--cover-taus` — those best values depend on what
  the campaign ran, and are written to `best_known_hv.csv`; level k pairs the
  k-th values of the two lists. Problems whose reference front is doubtful —
  DTLZ5, DTLZ6 and MaF6 from four objectives, WFG3, DTLZ1 at five objectives,
  and against references of version 1 IPolygon and the aliased WFG1 and WFG2 —
  are left out of the coverage (`cover_excluded.csv`; `--cover-keep-caveats`
  keeps them), and so is a problem whose runs were measured against an older
  reference than the current one where it changed since (run
  `--recompute-reference`). Problems marked for their reference — other than
  n_ref rows, repeated rows, the Das-Dennis lattice from four objectives, a
  front of lower dimension than M − 1 — stay in, and every table is written
  twice, over all problems and over the unmarked ones (the `subset` column;
  `cover_marks.csv` says why), and a third time over the problems whose
  optimum is neither at the centre nor on the boundary (`interior`: the
  properties `centre` and `boundary` not true), where clipping a variable or
  stepping to the middle of its range finds no optimum by construction;
  `--portfolio` gives those as the group `front_interior`, on lines of its
  own. `cover_meta.json` records the commit of the
  analysis code and the scheme of the levels, and every CSV row says its level
  in words, so absolute and floor-based levels are never confused.
  `floor_review.csv` sets each problem's floors beside the best `igdp_norm` any
  run reached, the best algorithm's median and the value that decides coverage
  (its `--cover-seeds`-th best seed), as ratios to the level's floor, and as
  η = ratio^(−d), d the front's dimension, which reads alike across numbers of
  objectives: a ratio below 1 says the floor is too high. For every criterion,
  level and budget in `--cover-budgets` (the campaign's
  ladder and full budget), it reports the problems nobody covers, the smallest
  sets of algorithms covering every problem somebody does — exact, by branch
  and bound, all of them up to `--cover-max-sets`, and the greedy set beside
  them — and the curve k → the largest share k algorithms cover, exact up to
  the smallest set's size, with problems weighted alike and with families
  weighted alike (bbob-biobj is more than half of the problems), with each
  family's count along it. `--cover-bootstrap N` resamples the seeds N times
  and says how often each algorithm is in a smallest set. An algorithm whose
  seeds give the same values on every problem at every budget (DMS, which
  draws no random number) has its ten seeds as one run, so "7 of 10" is "1 of 1" for it: the
  report names it, and beside every level the row "one run each" reads every
  algorithm from one seed at a time (covered = the level in that seed) and
  gives the mean over the seeds, with the problems the deterministic algorithm
  then covers alone (`cover_one_run.csv`, per seed). Tables on the
  terminal, CSV in `<results>/_cover/`. A budget-dependent algorithm is read
  at a smaller budget from its ladder rung, every other from its trajectory
  (the first record at or after the budget). `--cover --scenario archive` reads
  the reduced archive instead: `final_archive` at the full budget, the archive
  checkpoints below it, a rung's `final_archive` for a budget-dependent
  algorithm; CSV in `<results>/_cover_archive/`.
- `--behaviour`: how each run behaved, read off its trajectory — the
  evaluations until `gdp_norm` reaches 0.1, 0.03, 0.01, 0.003 and until
  `igdp_norm` comes within 2, 1.5, 1.25, 1.1 times the best any run of the
  campaign reached (not the IGD+ floor of `--cover`); where `gdp_norm` and
  `igdp_norm` settle (never again more than 10 % better); the slope of
  log `gdp_norm` against log evaluations until it settles; `spread`, whether
  the coverage kept improving after the convergence stopped; the collapse of
  `range_cover`; `stall`, the share of the budget spent with the population
  non-dominated for good and the quality settled; `dup_share`; `igdx` and `cr`
  where the problem has a Pareto set. Medians over the seeds per problem and
  algorithm; then, per problem and descriptor, the algorithms ranked into
  shares from 0 to 1, and grouped by average linkage on the mean difference of
  their shares into as many groups as there are declared families, set beside
  the families. CSV in `<results>/_behaviour/` (`behaviour_meta.json` records
  the commit of the analysis code); `mootation/run/behaviour.py` defines every
  descriptor.
- `--portfolio`: what the set of algorithms gains over its members, at
  `--cover`'s levels, budgets and floors, with the problems with a reference
  front and the bbob-biobj ones kept apart (the latter's target, the gap to the
  campaign's best run, is relative): for every level, budget and number of
  seeds that must reach it (5, 7 and 10 of ten), the oracle (the problems some
  algorithm covers), the single best solver and the gap, every algorithm's
  Shapley value in the coverage game (each problem's 1 split among the
  algorithms covering it) and what it covers alone; for every two algorithms
  the problems one covers and the other does not; per problem the gap between
  the best median and that of the single best solver on average; and the
  matrix problem × algorithm × budget with the seeds reaching every level and
  the median and 7th best of `igdp_norm`, `gdp_norm`, `hv_h` and `eps_norm`,
  the answer's median size and the seeds whose answer is shorter than N/2
  (`portfolio_matrix.csv`), so that later analyses need not read every
  `meta.json`. With a deterministic algorithm in the campaign, the rule
  "one_run" beside 5, 7 and 10: per seed every algorithm covering what it
  reaches in that seed, the oracle, Shapley values and problems covered alone
  averaged over the seeds, and the report sets the deterministic algorithm's
  Shapley value and place under 7 of 10 beside them. CSV in
  `<results>/_portfolio/` (`_portfolio_archive/`).
- `--invariance-check`: is each algorithm entry invariant to the scale and to
  a shift of the objectives? It runs, rather than reads: every entry on every
  problem of the campaign at its budget, on the problem and on two copies with
  the objectives times 2^10 and times (2^-6, 1, 2^6, ...) — exact in floating
  point, so an invariant algorithm runs the same trajectory, compared bit for
  bit (the points evaluated, in order; the first evaluation where they part)
  — and on a copy shifted by 2^4 over `--inv-seeds` seeds (30). A shift is
  rounded, so trajectories may part without any hidden constant: the final
  IGD+ (the shift taken off) is compared by the rank-sum test, Holm-adjusted
  over the entries of a problem, and the shift is "invariant" (the same
  trajectories), "rounding" (they part, the distributions cannot be told
  apart) or "not invariant". CSV in `<results>/_invariance/` (`invariance.py`).
- `--instance-space`: instance space analysis (Smith-Miles & Muñoz, ACM
  Computing Surveys 55(12), 2023; the projection of Muñoz, Villanova, Baatar &
  Smith-Miles, Machine Learning 107(1), 2018; the footprints of Muñoz &
  Smith-Miles, Evolutionary Computation 25(4), 2017), with the problems with a reference
  front and the bbob-biobj ones in spaces of their own. The features are the
  problem properties of `--by` (yes/no as 1/0, unknown as 0.5, the front's
  shapes a column each, the bbob-biobj function groups as counts, M, log2 D,
  the front's dimension), z-scored, for the problems with a front; for the
  bbob-biobj ones, which their properties put on a few tens of points, 19
  features of a uniform sample of ten populations of their box (non-dominated
  share, dominated pairs, the objectives' correlation and scale ratios, skew,
  kurtosis, linear and quadratic R², fitness-distance correlation, local
  optima and nearest-better ratios, and their multi-objective analogues),
  through PRELIM's bounding, Box-Cox and z-score (`--is-bbob-features
  properties` takes the properties there too); the performance is PRELIM's relative one,
  each algorithm's median `igdp_norm` (or `hv_h`) at the full budget against
  the best median on the problem, Box-Cox-transformed and z-scored; SIFTED's
  first step keeps the features correlated with some algorithm's performance.
  PILOT places each problem at Z = A F so that linear models of Z predict the
  features and the performances best. Its optimum is a reduced-rank regression
  and is computed exactly; the authors' toolkit's BFGS from 30 random starts
  is run beside it as a check, and the analytical solution the papers give —
  not that optimum: it takes the eigenvectors of [F; Y] where those of its part
  in F's row space are needed — is reported too. The optimum fixes the plane
  only up to an invertible linear map, which changes the footprints, so its
  axes are fixed: whitened (the default) or, with `--is-calibration
  preserve`, stretched to preserve the problems' distances best; and every
  footprint's area share comes with its range over six such planes of the
  same optimum. The performance rows are weighted √(q/a), q features against
  a algorithms, so that the algorithms do not outweigh the features
  (`--is-weight none`: the papers' unweighted loss; the other weighting's
  preservation and R² are reported beside). The footprint of an algorithm at each of
  `--cover`'s levels and budgets (good = the level in `--cover-seeds` seeds) is
  the Delaunay triangulation of the problems where it is good, without
  triangles whose side exceeds a quarter of the largest distance or whose
  density or purity (the share of good problems inside, at least 0.75) is too
  low; its area and density are given as shares of the known region's, the
  same construction over all problems, and its area also as a share of the
  convex hull of all problems, as the 2018 paper and Yap, Muñoz & Smith-Miles
  (IEEE TEVC 26(6), 2022) give it. The density limit is relative to the known
  region's density (the papers' ρ = 10 per unit area was 4 % of it in 2017 and
  50 % of the hull's in 2018; PILOT's plane has no scale of its own). The footprints of the best algorithm per
  problem lose the triangles where another algorithm's overlapping ones are
  larger. With a deterministic algorithm in the campaign every good footprint
  is drawn once more per seed, good = the level in that one seed, and the mean
  of its area shares goes beside it (kind `good_one_run`). Needs SciPy. CSV
  and JSON in `<results>/_instance_space/`
  (`_instance_space_archive/`): the coordinates with the features, the relative
  performances, the projection with every feature's and algorithm's R², the
  footprints' measures and triangles.
- `--recompute-reference`: every run records which reference front its
  metrics were measured against (`reference_version` and `reference_rows` in
  `meta.json`; the registry's `REFERENCE_VERSION`, version 2 since
  2026-09-29: distinct rows, exactly `n_ref` of them where the sampler can
  give as many, WFG1 and WFG2 without the aliasing grid, a new IPolygon). A run
  measured against an older version, on a problem whose reference changed since
  (`REFERENCE_CHANGED`), is measured again from its saved answers — `final.csv`,
  `final_archive.csv.gz`, the archive checkpoints in `archive_at.npz`, the
  trajectory from `snapshots.npz` — every reference-dependent metric it
  recorded; the old values stay beside the new, in `meta.json` under
  `reference_v1` and in `trajectory.ref_v1.jsonl`. A problem whose definition
  changed (`PROBLEM_CHANGED`: IPolygon) is not measured again but counted: its
  runs answered another problem and are to be run again. Until then `--cover`
  leaves such problems out.

**Budget-dependent algorithms.** Seventeen algorithms schedule something by
the share of the budget spent — RVEA's angle penalty t/t_max, the adaptation
periods of MOEA/D-AWA and AdaW, and so on: RVEA, RVEA*, MOEA/D-AWA, AdaW,
DEA-GNG, MBRA, NRV-MOEA, HLMEA, DHEA, MOEA/D-DS, SRV, SRV-NSGA-III, DCEA,
MaOEA-3C, MOEA/D-M2M, MOEA/D-AM2M and Liu–Gu 2011. For them the record at
10 000 evaluations of a 25 000-evaluation run is not what a run of 10 000
gives, and every table marks them with `*`. The list is known two ways,
kept side by side in `mootation/run/budget.py`: read off the code — the
binding passes the budget to every core with `set_t_max`, and these use it in
a schedule (IF-MaOEA stores it without reading it, and CLIA's one use gives the
same value at every budget up to 100 000) — and measured:

```bash
python -m mootation.run.budget --workers 16          # DTLZ2_3D, seed 1, 10 000 against 25 000
```

runs every algorithm twice on one problem, seed and population and compares
every record of the answer set up to the shorter budget bit for bit. On
2026-09-27 exactly the seventeen differed, MOEA/D-M2M first (at 180
evaluations) and MOEA/D-AWA last (at 9 109); the other 46 and the four
baselines were identical — the baselines up to their last step, which they cut
to end exactly at the budget, and which the check reports as "same, cut".

A budget ladder therefore needs separate runs for these seventeen only.
`ladder = [2500, 5000, 10000]` in `[campaign]` adds them: for every
budget-dependent algorithm, problem and seed, a run of each budget, filed under
`<algorithm>@<budget>` and appended after all other jobs, so the full runs'
job numbers do not move. Every `meta.json` says `budget_dependent`, and a rung
also `ladder_of`. The analysis reads a budget-dependent algorithm at a smaller
budget from its rung and every other from its full run's trajectory: `--cover`
takes the first record at or after the budget, and `--at` puts the rung in the
full run's place when the campaign has one of exactly that budget. The rungs
never appear as algorithms of their own in `--compare`, `--ranks` or the TUI.
Without rungs, overlay a budget-dependent algorithm's curves by fraction of
the budget (`--at`), never by absolute evaluations.

### Changing the number of workers while it runs

Every worker runs NumPy with one BLAS thread (`OPENBLAS_NUM_THREADS`,
`OMP_NUM_THREADS` and `MKL_NUM_THREADS` are set to 1 unless you set them): the
pool is the parallelism, and OpenBLAS otherwise reserves a buffer for every
core in every process at import — 756 MB each on a 24-thread machine, enough
for twenty workers to exhaust a 31 GB Windows machine's commit limit.

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

A job's part is a hash of its problem, algorithm and seed: the same on every
machine and whatever else the campaign holds, the k parts together every job
once. Then copy every machine's `results/all58/` to one machine and merge them
into the config's results directory:

```bash
python -m mootation.run.campaign campaign_all.toml --merge part1/all58 part2/all58 part3/all58
```

`--merge` copies every run from the one part that has it finished, the parts'
reference caches (`_references/`) and writes `campaign.json` and `merge.json`
(where each run came from); the parts stay as they are. It refuses, writing
nothing, when a run is missing or in two parts, when the parts ran different
code (the revision, the benchmark sources' `code_stamp`, a problem's version,
the reference version) or measured against different references (each
`meta.json`'s `reference_hash`, a hash of the frame, the reference front, the
dense reference and the Pareto-set sample), or when the target already holds a
run of the campaign. A killed shard resumes when started again, and `--force`
reruns finished jobs.

The queue runs the longest jobs first (`campaign.longest_first`): the
expected seconds of every job, from the finished runs of the same algorithm
and problem in the results directory where there are some, else from a model
of the core's and the problem's cost per evaluation. The runs are independent:
the order changes when the campaign ends, not what it finds.

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
