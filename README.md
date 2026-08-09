# MOOtation

***English** · [Русский](README.ru.md)*

[![DOI](https://zenodo.org/badge/1328202748.svg)](https://doi.org/10.5281/zenodo.21864324)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

**Header-only C++17 library with 60 multi- and many-objective evolutionary algorithms, implemented from their original papers.**

*MOO* — multi-objective optimization; *mutation* — the operator every one of
these algorithms is built on. The name is both words at once.

No dependencies beyond the C++ standard library. Every algorithm was cross-checked line-by-line against its primary source (pseudocode, equations, default parameters); every conscious deviation, and every resolution of an in-paper ambiguity, is declared in the header of the corresponding file with section and equation references.

Two files are not transcriptions of a paper and say so in their own headers: `ibea_eplus_clean.hpp` is a documented composition of two published algorithms, and `srv_moead.hpp` is an explicitly experimental extrapolation of SRV onto a carrier the SRV paper does not cover.

## Highlights

- **60 algorithms** spanning Pareto-dominance, decomposition (MOEA/D family), indicator-based, reference-vector, clustering-based and archive-based approaches — one `#include`, no build step.
- **Source fidelity as a feature.** Each `.hpp` starts with the paper reference (authors, venue, DOI), a short scheme of the generational cycle, paper defaults, and an explicit numbered list of any deviations. The library went through several rounds of independent audit against the primary sources, including adversarial re-review and runtime sanitizer passes (ASan/UBSan clean).
- **Genomes:** real-valued everywhere; 45 of the 60 also handle binary and mixed real+binary. The remaining 15 are real-only because their reproduction operator is (their headers state this, and those that could silently drop a binary genome now refuse it instead).
- **Constraints:** constraint violation (CV) with feasibility-first / constrained-domination handling, in **all 60**, off by default as in the original (unconstrained) papers. Each header states where its constraint handling attaches — the non-dominated sort, the scalarizing comparison, the archive filter, the truncation rule — because the right place differs per algorithm and only one of them is the algorithm's actual preference relation.
- **Diagnostics:** some algorithms silently substitute a parameter you asked for — a subregion count that is not an attainable lattice size comes back rounded up, and the derived per-subregion quota changes with it. Each such case is declared in the file's header and, with `set_warn_handler`, also announced at runtime. Silent by default.
- **External / batch evaluation:** plug in your own evaluator (e.g., a simulator) through a batch-executor interface; objectives are evaluated lazily.
- **Reproducibility:** explicit seeding (`set_seed`), deterministic reference-point generators (Das–Dennis simplex lattice, two-layer), and defaults taken from the paper wherever the paper states one. Where it does not — some papers give no distribution index, or give a value only for one objective count — the header says so in as many words and names what was used instead. A handful of defaults deliberately differ from the paper for library consistency; those are numbered deviations, not silent choices.

## Which interface do I want?

| your situation | use | where |
|---|---|---|
| C++, one algorithm known at compile time | `Optimizer<Ind, Core>` | [Quick start](#quick-start) |
| C++, algorithm chosen at run time, or you own the evaluation loop | `mootation::Session` / `run` | [`include/mootation/embed.hpp`](include/mootation/embed.hpp) |
| Python, objectives are a Python function | `mootation.minimize` | [`python/examples/01_basic.py`](python/examples/01_basic.py) |
| Python, standard test suites | `mootation.benchmarks` | [`python/examples/03_benchmarks.py`](python/examples/03_benchmarks.py) |
| Objectives come from a separate PROGRAM (solver, mesher, rig) | TOML + `mootation.run` | [`python/examples/demo.toml`](python/examples/demo.toml) |
| Rust, C#, Julia, R, MATLAB, Fortran, Go | the C ABI | [`include/mootation/capi.h`](include/mootation/capi.h), wrappers in [`capi/wrappers/`](capi/wrappers/) |
| Continue a finished run, or switch algorithm mid-study | save / seed a population | [`python/examples/06_restart.py`](python/examples/06_restart.py) |
| No control over include paths at all | the single header | [amalgamation](#single-header) |

Every one of them reaches the same 60 algorithms. The differences are who owns
the loop and what crosses the boundary — not what you can run.

## Quick start

```cpp
// nsga2_zdt1.cpp — NSGA-II on the built-in ZDT1 problem
#include <mootation/mootation.hpp>
#include <iostream>

int main() {
    using namespace mootation;

    Problem<NSGAII_Individual>   prob;               // default: ZDT1, n=6
    DataVault<NSGAII_Individual> vault(100, prob);   // population size 100

    // `defer_setup` is what makes the run reproducible. The one-argument
    // constructor calls setup() from its own body, which draws the initial
    // population immediately — after that, set_seed() is too late to matter.
    Optimizer<NSGAII_Individual, NSGAIICore<NSGAII_Individual>>
        opt(std::move(vault), defer_setup);

    opt.get_algorithm().set_seed(12345);
    opt.setup();

    opt.optimize(500);                               // 500 generations

    auto& v = opt.get_vault();
    for (std::size_t i = 0; i < v.active_n(); ++i) {
        const auto& f = v.objectives_of(i);
        std::cout << f[0] << ' ' << f[1] << '\n';
    }
}
```

```bash
g++ -std=c++17 -O2 -Iinclude examples/nsga2_zdt1.cpp -o nsga2_zdt1 && ./nsga2_zdt1
```

### Embedding it in your own program

When the objective values come from somewhere the optimizer cannot call into —
a simulator, a solver process, a measurement rig, another language — drive the
loop yourself. `ask()` hands you candidates, `tell()` takes their objective
values and returns the next candidates.

```cpp
#include <mootation/embed.hpp>

mootation::Session s(mootation::Settings::from_file("run.cfg"));

std::vector<std::vector<double>> F;
for (auto X = s.ask(); !X.empty(); X = s.tell(F)) {
    F.assign(X.size(), std::vector<double>(s.n_objs()));
    for (std::size_t i = 0; i < X.size(); ++i)
        F[i] = my_simulator(X[i]);      // X[i] is one candidate's variables
}
mootation::Result r = s.result();
```

**The batch size is the algorithm's choice, not `pop_size`.** Generational
algorithms hand over a whole offspring generation at once; steady-state ones
(MOEA/D-DE, MOEA/DD, …) hand over one candidate at a time, because that is how
those algorithms are defined. Size your loop off `X.size()`.

If you can just hand over a function, `run()` owns the loop instead — same
`Settings`, same `Result`, bit-identical output for the same seed:

```cpp
auto r = mootation::run(settings,
    [](const std::vector<std::vector<double>>& X,
       std::vector<std::vector<double>>& F,
       std::vector<std::vector<double>>& /*G*/) {
        for (std::size_t i = 0; i < X.size(); ++i) F[i] = my_simulator(X[i]);
    });
```

Settings are a plain struct, and the same fields are a plain `key = value` file
— no schema, no builder, no registry:

```ini
algorithm   = nsga3        # any name from mootation::algorithm_names()
pop_size    = 92
max_gen     = 300
seed        = 42
n_vars      = 7
n_objs      = 3
lower       = 0            # one value broadcasts to all n_vars
upper       = 1
constraints = none         # none | feasibility | cdp | eps_constraint
eta_c       = 30           # optional knobs keep the paper default when absent
```

`examples/run.cfg` is a commented copy of every key. An unknown key is an
error, not a silent no-op: a typo must not quietly give you a different run
than you asked for. A knob the chosen algorithm does not
have is reported in `Result::ignored` for the same reason. Constraints go in as
a second matrix `G` alongside `F`, one row per candidate, each value `<= 0`
meaning satisfied.

Because the algorithm is picked by name at run time, `#include
<mootation/embed.hpp>` instantiates all 60 in that translation unit — a real
compile-time cost. A program that always uses one algorithm should build
`Optimizer<Ind, Core>` directly, as in the quick start above.
`examples/embed_asktell.cpp` runs both shapes and checks they agree.

### Running against your own solver (TOML)

When the objective values come from programs rather than a function — a mesher,
a solver, a measurement rig — describe the run in a file and let
`mootation.run` drive it. Pure Python, no dependencies: `tomllib` has been in
the standard library since 3.11.

```bash
python -m mootation.run --show --check python/examples/demo.toml
```

`--check` is the point of the whole layer. It verifies, before anything
expensive starts, that every step's executable exists on this platform, that
the input template's `{x[i]}` stay within `n_vars`, that `bounds` are ordered
and counted, that each algorithm name is in the registry, and that each `pop`
satisfies that algorithm's own constraint — NSGA-III wants an exact Das-Dennis
lattice size, MOEA/D-M2M wants `pop` divisible by `K`. Every complaint at once,
exit status 1, no UI required:

```
3 problem(s):

  - problem.bounds: has 2 entries but n_vars = 3; there must be exactly one [lower, upper] pair per variable
  - problem.steps.solve: executable not found on PATH and not an existing file: 'solver' (platform: linux)
  - algorithms[0] (nsga3): pop = 92 is not a Das-Dennis lattice size for n_objs = 3; nsga3 requires an exact lattice. Nearest attainable: 91, 105
```

That last one is a real trap: the NSGA-III paper's own Table I uses N = 92, the
nearest multiple of four above H = 91, and this library requires the lattice
size exactly (see `nsga3.hpp`). Ten seconds here instead of ten minutes into a
day-long run.

Commands are argv **arrays**, so there is nothing to quote and nothing to
escape, and `run_windows` / `run_linux` / `run_darwin` override one step on one
platform. Output is read by one of five strategies — `csv` by column name,
`json` by dotted path, `regex` for solver logs (`take = "last"`, because
solvers print the value every iteration), `columns` positionally, and
`python:module.function` for HDF5, VTU and anything else that cannot be
described declaratively.

**A parse failure is an evaluation failure.** Nothing is silently substituted:
one fabricated point poisons the archive and every metric taken from it. The
run's `on_fail` decides — `penalty`, `skip` or `abort`.

Two stores, deliberately separate: scratch is per worker and overwritten
(meshes, VTU, gigabytes); the journal is per run and append-only
(`{"gen":12,"idx":47,"x":[...],"f":[...],"status":"ok"}`, ~300 bytes an
evaluation). Resume works off the journal — knowing that x maps to f does not
require the mesh that produced it — and the journal doubles as a cache keyed by
a hash of the decision vector, which matters more than it sounds: evolutionary
algorithms re-propose identical individuals often.

The **standard suites** come with it: ZDT, DTLZ, WFG, MaF, the Ishibuchi
polygon family, MOP and BT — 216 problems across 11 families, each with bounds,
an evaluator, and where a closed form exists a sampler of the true Pareto front
so IGD and friends have a real reference. Point a config at them instead of an
external solver:

```toml
[problem]
kind = "builtin"

[benchmarks]
families   = ["DTLZ", "WFG"]
objectives = [3, 5, 10]
runs       = 31
# problems = ["DTLZ2_3D", "WFG4_5D", "MaF3_8D"]   # ... or name them outright
```

`--check` expands that cross product against the real registry and names what
does not exist, with near misses. `python -m mootation.run --problems` lists
everything. The suites need NumPy; nothing else here does.

There is a **terminal interface** over the same functions —
`python -m mootation.run --tui run.toml` — with four read-only screens: the
resolved config and its verdict, the problem registry with a filter, the
selected algorithms with each one's population objection, and a monitor that
reads the journal live. It needs Textual. It is read-only on purpose: you edit
the config in your own editor, because a configuration assembled by clicking
cannot be diffed, copied to a cluster, or attached to a paper.

### Saving a run, and continuing it

```python
res = mootation.minimize(f, bounds, 2, pop_size=100, n_gen=200,
                         save_population="pop.csv",     # what survived
                         log_evaluations="evals.csv")   # every call made

# later — same settings, or different ones, or a different algorithm
res = mootation.minimize(f, bounds, 2, algorithm="moead_de", pop_size=91,
                         n_gen=500, seed_population="pop.csv",
                         on_size_mismatch="pad")
```

Both files are CSV with a `#` preamble carrying the settings that produced
them, so `pandas.read_csv(path, comment="#")` reads either directly and the
population loads straight back. The evaluation log holds every call **including
the ones the optimizer discarded**, which for an expensive evaluator is the
bulk of the cost. Both are off unless asked for — off meaning no file is opened
and nothing is wrapped, not a flag checked per evaluation.

Seeding costs **zero function evaluations**: the objectives are read from the
file, not recomputed.

This is a warm start, not a checkpoint. No RNG position and no per-algorithm
state are saved — only the decision variables, objectives and constraint values,
which is exactly what all 60 algorithms share and therefore what lets a
population produced by NSGA-II seed a MOEA/D run. A resumed run does not
reproduce what the uninterrupted one would have done; it starts from the same
place.

In C++ it is a field, in both shapes:

```cpp
mootation::Settings s = mootation::Settings::from_file("run.cfg");
s.seed_population  = "pop.csv";                 // or set it in run.cfg
s.on_size_mismatch = mootation::SizeMismatch::Truncate;

auto r = mootation::run(s, my_evaluator);       // and Session(s) likewise
```

or straight from the previous run, with no file in between:

```cpp
mootation::Session next(s, mootation::as_population(previous));
```

Seeding is refused, with the mismatch named, when the population comes from a
different problem — a different variable or objective count — and when a
constrained run is handed a population that carries no per-constraint values.
That last one matters: `cv` is a sum and cannot be taken apart again, so
starting from it would plant a population every one of whose constraints reads
as satisfied. Saved files carry the individual values (`g1…gk`) for exactly this
reason. `mootation::io::save_population` / `load_population` are the file pair.

### From other languages (C ABI)

`MOOTATION_BUILD_C_API=ON` builds a shared library with a plain `extern "C"`
surface — the bridge for ctypes/cffi, P/Invoke, Rust's `extern "C"`, Julia's
`ccall`, MATLAB's `loadlibrary`, Fortran's `iso_c_binding`, anything with an
FFI. It is the only compiled artefact the project produces.

Nothing but C types crosses the boundary: configuration goes in as one string
in the settings format above, and candidates and objective values move as flat
row-major `double` arrays. There is no struct to keep in sync between
languages, which is the point.

```c
#include <mootation/capi.h>

moo_session* s = moo_open("algorithm = nsga2\n"
                          "pop_size  = 40\n"
                          "max_gen   = 40\n"
                          "n_vars    = 10\n"
                          "n_objs    = 2\n"
                          "lower     = 0\n"
                          "upper     = 1\n");
int nv = moo_n_vars(s), no = moo_n_objs(s);

for (int n = moo_ask_count(s); n > 0; ) {
    double* x = malloc((size_t)n * nv * sizeof(double));
    double* f = malloc((size_t)n * no * sizeof(double));
    moo_ask(s, x, n * nv);                     /* n rows of nv variables */
    for (int i = 0; i < n; ++i)
        my_simulator(&x[i * nv], &f[i * no]);
    n = moo_tell(s, f, n * no, NULL, 0);       /* returns the NEXT count */
    free(x); free(f);
}

int m = moo_result_count(s);                   /* the final population */
/* moo_result(s, x, ..., f, ..., cv, ...); */
moo_close(s);
```

Every call that can fail returns a negative int (or `NULL`) and leaves a
message in `moo_last_error()`; no exception escapes the library.
`capi/ctypes_demo.py` drives the same run from Python with `ctypes` alone — no
build step, no binding — and is about 60 lines meant to be copied into whatever
project needs it. `capi/smoke.c` is compiled as C99 in CI, so a C++-ism that
creeps into `capi.h` fails there rather than in your project.

[`capi/wrappers/`](capi/wrappers/) has the same loop for **Julia**, **R** and
**MATLAB/Octave**. Read that directory's README before relying on them: the ABI
underneath is tested on every CI run, but the three language wrappers were
written on a machine with none of those interpreters installed and have not
been executed.

### Python

```bash
pip install .            # the optimizer
pip install ".[all]"     # plus benchmarks (NumPy) and the TUI (Textual)
```

```python
import mootation

def zdt1(x):
    g = 1 + 9 * sum(x[1:]) / (len(x) - 1)
    return [x[0], g * (1 - (x[0] / g) ** 0.5)]

res = mootation.minimize(zdt1, bounds=[(0, 1)] * 10, n_objs=2,
                         algorithm="nsga2", pop_size=100, n_gen=250)
print(res.objectives[0], res.ignored)
```

Constraints are a second function returning violations (`<= 0` means
satisfied); `batch=` hands a whole generation over at once, for when an
evaluation is expensive. `mootation.algorithms()` lists all 60 names.

The **216 benchmark problems** are importable and runnable against any of them,
with the true Pareto front where a closed form exists:

```python
import mootation.benchmarks as bench

res = bench.solve("DTLZ2_3D", "nsga3", pop_size=91, n_gen=200)
print(bench.igd("DTLZ2_3D", res.objectives))
```

`python/examples/` has five worked scripts — basics, constraints, benchmark
comparison, expensive evaluators, many-objective — each runnable and each run
in CI. See [python/examples/README.md](python/examples/README.md).

`res.ignored` lists knobs you set that the chosen algorithm does not have —
`kappa` on NSGA-II, say. They are reported rather than dropped, because a run
configured with a silently ignored parameter is not the run you asked for.

For full control, `mootation.Problem` / `mootation.Config` / `mootation.run_raw`
map onto the C++ API one-to-one and hide nothing; `minimize` is a wrapper over
them, not a different capability. (The dispatch function is `run_raw` because
`mootation.run` is the TOML subpackage.)

A source build instead of `pip`:

```bash
cmake -S . -B build -DMOOTATION_BUILD_PYTHON=ON && cmake --build build
```

The dispatch is generated from the same `include/mootation/algorithms.def` the
test suite iterates over, so the binding cannot drift from the library.

If you cannot control include paths — a competition judge, a plugin SDK, a
Compiler Explorer link — `python tools/amalgamate.py` flattens the library into
one self-contained header, `embed.hpp` included. It is generated, not
committed: CI regenerates it each run, compiles it standalone at `-Wall -Wextra
-Werror`, checks that `run()` and `Session` still agree there, and fails if any
private file reached it. Comments are kept — the per-algorithm header block is
the point of the library, not overhead. Budget about a minute of compile time
per translation unit that includes it: it carries all 60 algorithms plus the
run-time dispatch over them.

Adding your **own algorithm** is one class template plus one line in
`include/mootation/algorithms.def` — no base class, no registry. The contract, the vault's
slot/function-evaluation rules and the header conventions are written up in
[docs/writing-an-algorithm.md](docs/writing-an-algorithm.md).

Custom problems are defined by specializing `Problem<>` for a tag individual (duck-typed interface: variable/objective counts, per-variable bounds, `calc_objs`) or by wiring a `batch_executor` for external/parallel evaluation. `examples/custom_problem.cpp` is a fully self-contained walkthrough: it defines the classic Kursawe benchmark in ~40 lines and solves it with NSGA-II.

## Algorithms

DOI links point to the primary source each implementation follows. File = `include/mootation/algorithms/<file>.hpp`.

### Pareto-dominance & diversity-based

| Algorithm | Year | File | DOI |
|---|---|---|---|
| NSGA-II | 2002 | `nsga2` | [10.1109/4235.996017](https://doi.org/10.1109/4235.996017) |
| SPEA2 | 2001 | `spea2` | [10.3929/ethz-a-004284029](https://doi.org/10.3929/ethz-a-004284029) |
| SPEA2+SDE | 2014 | `spea2_sde` | [10.1109/TEVC.2013.2262178](https://doi.org/10.1109/TEVC.2013.2262178) |
| GrEA | 2013 | `grea` | [10.1109/TEVC.2012.2227145](https://doi.org/10.1109/TEVC.2012.2227145) |
| VaEA | 2017 | `vaea` | [10.1109/TEVC.2016.2587808](https://doi.org/10.1109/TEVC.2016.2587808) |
| AGE-MOEA | 2019 | `agemoea` | [10.1145/3321707.3321839](https://doi.org/10.1145/3321707.3321839) |
| ETEA | 2014 | `etea` | [10.1162/evco_a_00106](https://doi.org/10.1162/evco_a_00106) |

### Reference-point based (NSGA-III family)

| Algorithm | Year | File | DOI |
|---|---|---|---|
| NSGA-III | 2014 | `nsga3` | [10.1109/TEVC.2013.2281535](https://doi.org/10.1109/TEVC.2013.2281535) |
| A-NSGA-III | 2014 | `a_nsga3` | [10.1109/TEVC.2013.2281534](https://doi.org/10.1109/TEVC.2013.2281534) |
| θ-DEA | 2016 | `theta_dea` | [10.1109/TEVC.2015.2420112](https://doi.org/10.1109/TEVC.2015.2420112) |
| AR-MOEA | 2018 | `ar_moea` | [10.1109/TEVC.2017.2749619](https://doi.org/10.1109/TEVC.2017.2749619) |

### Decomposition & region division (MOEA/D family)

| Algorithm | Year | File | DOI |
|---|---|---|---|
| MOEA/D | 2007 | `moead` | [10.1109/TEVC.2007.892759](https://doi.org/10.1109/TEVC.2007.892759) |
| MOEA/D-DE | 2009 | `moead_de` | [10.1109/TEVC.2008.925798](https://doi.org/10.1109/TEVC.2008.925798) |
| MOEA/D-DRA | 2009 | `moead_dra` | [10.1109/CEC.2009.4982949](https://doi.org/10.1109/CEC.2009.4982949) |
| MOEA/DD | 2015 | `moead_dd` | [10.1109/TEVC.2014.2373386](https://doi.org/10.1109/TEVC.2014.2373386) |
| MOEA/D-AWA | 2014 | `moead_awa` | [10.1162/EVCO_a_00109](https://doi.org/10.1162/EVCO_a_00109) |
| AdaW | 2020 | `adaw` | [10.1162/evco_a_00269](https://doi.org/10.1162/evco_a_00269) |
| MOEA/D-M2M | 2014 | `moead_m2m` | [10.1109/TEVC.2013.2281533](https://doi.org/10.1109/TEVC.2013.2281533) |
| MOEA/D-AM2M | 2018 | `moead_am2m` | [10.1109/TEVC.2017.2725902](https://doi.org/10.1109/TEVC.2017.2725902) |
| SMS-M2M | 2015 | `sms_m2m` | [10.1007/978-3-319-13356-0_35](https://doi.org/10.1007/978-3-319-13356-0_35) |
| MOEA/D-DS | 2023 | `moead_ds` | [10.1016/j.asoc.2023.110295](https://doi.org/10.1016/j.asoc.2023.110295) |
| RD-EMO | 2020 | `rd_emo` | [10.1016/j.knosys.2020.105518](https://doi.org/10.1016/j.knosys.2020.105518) |
| APRD | 2021 | `aprd` | [10.1109/ICACI52617.2021.9435909](https://doi.org/10.1109/ICACI52617.2021.9435909) |
| Liu–Gu sub-regional NSGA-II | 2011 | `liu_gu2011` | [10.1109/CEC.2011.5949848](https://doi.org/10.1109/CEC.2011.5949848) |
| I_SDE+ RD | 2018 | `isde_rd` | [10.1109/CIS2018.2018.00015](https://doi.org/10.1109/CIS2018.2018.00015) |
| DHEA | 2024 | `dhea` | [10.1007/s40747-024-01637-3](https://doi.org/10.1007/s40747-024-01637-3) |
| HLMEA | 2022 | `hlmea` | [10.1016/j.ins.2022.08.077](https://doi.org/10.1016/j.ins.2022.08.077) |

### Indicator-based

| Algorithm | Year | File | DOI |
|---|---|---|---|
| IBEA (ε+) | 2004 | `ibea_eplus` | [10.1007/978-3-540-30217-9_84](https://doi.org/10.1007/978-3-540-30217-9_84) |
| IBEA (HD) | 2004 | `ibea_hd` | [10.1007/978-3-540-30217-9_84](https://doi.org/10.1007/978-3-540-30217-9_84) |
| IBEA-ε+ with front filter | 2004/2017 | `ibea_eplus_clean` | composition: [10.1007/978-3-540-30217-9_84](https://doi.org/10.1007/978-3-540-30217-9_84) + [10.1109/CEC.2017.7969423](https://doi.org/10.1109/CEC.2017.7969423) |
| mIBEA | 2017 | `mibea` | [10.1109/CEC.2017.7969423](https://doi.org/10.1109/CEC.2017.7969423) |
| R2-IBEA | 2013 | `r2ibea` | [10.1109/CEC.2013.6557868](https://doi.org/10.1109/CEC.2013.6557868) |
| HypE | 2011 | `hype` | [10.1162/EVCO_a_00009](https://doi.org/10.1162/EVCO_a_00009) |
| MOMBI-II | 2015 | `mombi2` | [10.1145/2739480.2754776](https://doi.org/10.1145/2739480.2754776) |
| NIMMO | 2019 | `nimmo` | [10.1016/j.swevo.2019.06.001](https://doi.org/10.1016/j.swevo.2019.06.001) |
| IREA | 2018 | `irea` | [10.1016/j.knosys.2017.10.025](https://doi.org/10.1016/j.knosys.2017.10.025) |
| EDV | 2019 | `edv` | [10.1016/j.asoc.2018.11.041](https://doi.org/10.1016/j.asoc.2018.11.041) |
| IF-MaOEA | 2024 | `if_maoea` | [10.1016/j.asoc.2024.111872](https://doi.org/10.1016/j.asoc.2024.111872) |
| MaOEA-IAMD | 2026 | `maoea_iamd` | [10.1007/s11227-026-08362-3](https://doi.org/10.1007/s11227-026-08362-3) |

### Reference-vector / angle-based

| Algorithm | Year | File | DOI |
|---|---|---|---|
| RVEA | 2016 | `rvea` | [10.1109/TEVC.2016.2519378](https://doi.org/10.1109/TEVC.2016.2519378) |
| MaOEA-ARV | 2021 | `maoeaarv` | [10.1016/j.ins.2021.01.015](https://doi.org/10.1016/j.ins.2021.01.015) |
| MBRA | 2024 | `mbra` | [10.1007/s40747-023-01161-w](https://doi.org/10.1007/s40747-023-01161-w) |
| NRV-MOEA | 2024 | `nrv_moea` | [10.1007/s40747-024-01353-y](https://doi.org/10.1007/s40747-024-01353-y) |
| DEA-GNG | 2020 | `dea_gng` | [10.1109/TEVC.2019.2926151](https://doi.org/10.1109/TEVC.2019.2926151) |
| MaOEA/SRV | 2022 | `srv` (+ `srv_strategy`) | [10.1109/TCYB.2020.2971638](https://doi.org/10.1109/TCYB.2020.2971638) |
| SRV-NSGA-III (hybrid) | 2022 | `srv_nsga3` | [10.1109/TCYB.2020.2971638](https://doi.org/10.1109/TCYB.2020.2971638) |
| SRV-MOEA/D (experimental) | 2022 | `srv_moead` | [10.1109/TCYB.2020.2971638](https://doi.org/10.1109/TCYB.2020.2971638) |

### Clustering-based

| Algorithm | Year | File | DOI |
|---|---|---|---|
| CA-MOEA | 2019 | `camoea` | [10.1109/TCYB.2018.2834466](https://doi.org/10.1109/TCYB.2018.2834466) |
| CAVA-MOEA | 2025 | `cava_moea` | [10.1007/s11227-024-06496-w](https://doi.org/10.1007/s11227-024-06496-w) |
| MaOEA/AC | 2020 | `maoea_ac` | [10.1016/j.ins.2020.03.104](https://doi.org/10.1016/j.ins.2020.03.104) |
| MaOEA/C | 2019 | `maoea_c` | [10.1109/TEVC.2018.2866927](https://doi.org/10.1109/TEVC.2018.2866927) |
| MaOEA-3C | 2023 | `maoea_3c` | [10.1016/j.ins.2023.119289](https://doi.org/10.1016/j.ins.2023.119289) |
| EMyO/C | 2014 | `emyo_c` | [10.1007/978-3-319-10762-2_53](https://doi.org/10.1007/978-3-319-10762-2_53) |
| crEA | 2015 | `crea` | [10.1016/j.asoc.2015.06.020](https://doi.org/10.1016/j.asoc.2015.06.020) |
| DCEA | 2024 | `dcea` | [10.1016/j.ins.2024.120940](https://doi.org/10.1016/j.ins.2024.120940) |
| HCCA | 2023 | `hcca` | [10.1109/ACCESS.2023.3234226](https://doi.org/10.1109/ACCESS.2023.3234226) |
| LIS/LCS | 2023 | `lis_lcs` | [10.1016/j.ins.2022.12.076](https://doi.org/10.1016/j.ins.2022.12.076) |
| CLIA | 2019 | `clia` | [10.1109/TEVC.2018.2874465](https://doi.org/10.1109/TEVC.2018.2874465) |

### Archive-based

| Algorithm | Year | File | DOI |
|---|---|---|---|
| Two_Arch2 | 2015 | `two_arch2` | [10.1109/TEVC.2014.2350987](https://doi.org/10.1109/TEVC.2014.2350987) |
| NAEMO | 2019 | `naemo` | [10.1016/j.swevo.2018.12.002](https://doi.org/10.1016/j.swevo.2018.12.002) |

### Variation operators

SBX crossover (Deb & Agrawal, 1995) · polynomial mutation (NSGA-II reference-code variant) · DE `rand/1/bin` (Storn & Price, 1997) with Clip / RandomReset repair · Liu–Li annealing arithmetic crossover & mutation (Liu & Li, 2009) · uniform binary crossover · bit-flip mutation. Reference-point generators: Das–Dennis simplex lattice and two-layer variant.

## Permutation / ordering problems (e.g., packing)

For problems whose natural encoding is a permutation (packing order, scheduling, routing), use the **random-keys encoding**: keep a real-valued genome, and inside your objective function decode it with `argsort(keys) → permutation → greedy decoder` (e.g., First-Fit for packing). Offspring are always feasible, and all 60 algorithms work unchanged — this is the classic BRKGA scheme. Native ordinal operators (OX, insertion, scramble) are on the roadmap for cases where random-keys hits its quality ceiling.

## Roadmap

Ordered by what would remove the most friction, not by what is most
interesting to build.

- **A public metrics module.** Exact hypervolume already exists (HSO slicing)
  but only as private copies inside `hype`, `sms_m2m` and `hlmea` — the one
  number every paper reports is the one number a user cannot compute. Extract
  it, add IGD+ and GD+ beside `benchmarks.igd`, expose all of it in Python.
- **A true checkpoint.** Restarting from a population works everywhere now —
  `setup_with_seed` in the core, `seed_population` in `Settings`, so `run`,
  `Session`, the C ABI and Python all have it. What is still missing is the
  rest of the state: the RNG position and each algorithm's own bookkeeping are
  not serializable, so a three-day job that dies at hour sixty resumes from the
  right place but not on the same trajectory.
- **A built-in parallel evaluator.** `batch_executor` is the hook; almost every
  user then writes the same thread pool. `n_workers=8` should be a parameter.
- **NumPy-native Python signatures.** `minimize` takes and returns lists;
  arrays in and out is what the scientific ecosystem expects.
- **A benchmark campaign runner** (repeated seeded runs, IGD/HV tabulation,
  Wilcoxon) on top of the Python binding.
- **Running the Julia, R and MATLAB wrappers.** They are written and shipped in
  `capi/wrappers/`, but none of the three interpreters was installed on the
  machine that wrote them, so the marshalling in those files has never
  executed. Until someone runs them, only the ABI beneath them is tested.
- Native permutation operators (order crossover, insertion, scramble).
- The WFG hypervolume algorithm — the faster variant at large objective counts.
  A speed item, not a missing capability.

## Fidelity & verification

Implementation notes at the top of every algorithm file document: the paper (authors, venue, DOI), a 5–8 line scheme of the generational cycle, paper defaults with the section they come from, and a numbered list of declared deviations. Resolutions of in-paper contradictions are marked as such and quote the conflicting passages, so a reader can check the arbitration rather than take it on trust.

The library has been through: a line-by-line dual-blind audit against the primary sources with arbitration of disagreements; an adversarial refutation pass over every finding, followed by a re-verification of the refutations themselves (which reinstated a substantial fraction of them); compiler-warning review at `/W4 /WX` and `-Wall -Wextra -Werror`; and ASan/UBSan runtime passes over all algorithms, enforced by CI.

What that process does NOT give you: an independent reproduction of the published experimental results. The convergence suite in `tests/` checks that each algorithm converges on DTLZ2 and ZDT1 within a coarse tolerance — it is a smoke test, not a benchmark replication. Two algorithms currently carry a `known_issue` marker there (`liu_gu2011`, `moead_m2m`); their headers explain what is unresolved.

## How to cite

Each release is archived on Zenodo. Cite the **concept DOI** below rather than a
version-specific one: it always resolves to the newest release, which is what a
reader following the reference will want. Use the version DOI only when the
exact code matters — reporting an experiment, say — and Zenodo lists one for
every release.

```bibtex
@software{karavan_mootation,
  author    = {Karavan, Andrey},
  title     = {{MOOtation}: a header-only {C++17} library of multi- and
               many-objective evolutionary algorithms},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.21864324},
  url       = {https://github.com/DedMoroz132/MOOtation}
}
```

If you use an algorithm in published work, **cite its paper too**. Every header
carries the reference and DOI of the publication it follows, and that is the
work the method comes from — this library is only an implementation of it.

## License

[Apache License 2.0](LICENSE). See also [NOTICE](NOTICE) and, for
contributions, the [CLA](CLA.md).
