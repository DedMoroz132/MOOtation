# MOOtation

***English** · [Русский](README.ru.md)*

[![DOI](https://zenodo.org/badge/1328202748.svg)](https://doi.org/10.5281/zenodo.21864324)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

**58 multi- and many-objective evolutionary algorithms, each implemented from
its paper and checked against it line by line. Header-only C++17, no
dependencies; also reachable from Python, from a TOML file that drives an
external solver, and from any language with a C FFI.**

The library exists for one reason: when you compare against NSGA-III or
MOEA/D-DE in a paper, you want the algorithm the authors published, not
somebody's reading of it. Every header here names the paper, traces one
generation in the paper's own symbols, lists the paper's defaults with the
section they come from, and declares every deviation and every resolution of an
in-paper ambiguity by number. The name is *MOO* (multi-objective optimization)
plus *mutation*, the operator all of these algorithms are built on.

## What you get

- **58 algorithms** in seven families: Pareto-dominance, NSGA-III-style
  reference points, the MOEA/D family, indicator-based, reference-vector,
  clustering-based and archive-based. The full list with DOIs, and how to
  choose, is in [docs/algorithms.md](docs/algorithms.md).
- **A paper trail per file.** Paper defaults, declared deviations, resolved
  ambiguities, and the place where constraint handling attaches, all in the
  header of the algorithm that uses them.
- **Five ways in, one algorithm list.** A C++ template, a C++ ask/tell session,
  a Python `minimize`, a TOML run description for external programs, and a C
  ABI. All five are generated from the same `algorithms.def`, so none can drift.
- **216 benchmark problems** (ZDT, DTLZ, WFG, MaF, MOP, BT, the Ishibuchi
  polygons) with true Pareto fronts where a closed form exists, IGD / IGD+ /
  hypervolume, and a campaign runner that shards across a cluster.
- **Warm starts**, an evaluation log, constraint handling in every algorithm
  (off by default, as in the papers), and a `--check` that refuses a run ten
  seconds in rather than ten minutes into a day-long job.

## What it is not

- Not a framework of interchangeable selection and variation blocks. Each
  algorithm is written as its paper states it; the coupling between steps is
  usually the paper's contribution.
- Not a reproduction of the papers' experimental tables. Fidelity was checked
  against the text; the tests are convergence smoke tests on DTLZ2, not
  benchmark replications.
- Real-valued and binary genomes only (46 algorithms take binary or mixed
  genomes, 13 refuse them). No native integer, categorical or permutation
  types and no ordinal operators; permutations go through random keys.
- No dedicated constrained, surrogate-assisted, dynamic, noisy or large-scale
  algorithms. Constraint handling is bolted onto unconstrained methods and says
  so in each header.
- From Python and the C ABI the genome is real-valued only.

## Install

**C++.** Copy `include/` or use CMake; the library is header-only and needs a
C++17 compiler and nothing else.

```cmake
add_subdirectory(MOOtation)                       # or FetchContent, or find_package(MOOtation)
target_link_libraries(app PRIVATE MOOtation::MOOtation)
```

**Python.** Needs a C++ compiler and CMake on the machine; the wheel is built on
install.

```bash
pip install .            # the optimizer
pip install ".[all]"     # plus the benchmark suites (NumPy) and the terminal interface (Textual)
```

**C ABI.** One shared library, `libmootation.so` / `libmootation.dylib` /
`mootation.dll`, target `MOOtation::c`.

```bash
cmake -S . -B build -DMOOTATION_BUILD_C_API=ON && cmake --build build
```

## Which interface do I want?

| your situation | use | start at |
|---|---|---|
| C++, one algorithm known at compile time | `Optimizer<Ind, Core>` | [quick start](#quick-start) |
| C++, algorithm chosen at run time, or you own the evaluation loop | `Session` / `run` in `embed.hpp` | [docs/embedding.md](docs/embedding.md) |
| Python, objectives are a Python function | `mootation.minimize` | [`python/examples/01_basic.py`](python/examples/01_basic.py) |
| Python, the standard test suites | `mootation.benchmarks` | [`python/examples/03_benchmarks.py`](python/examples/03_benchmarks.py) |
| objectives come from a separate program (solver, mesher, rig) | a TOML file + `mootation.run` | [docs/running.md](docs/running.md) |
| Rust, C#, Julia, R, MATLAB, Fortran, Go | the C ABI | [docs/embedding.md](docs/embedding.md#the-c-abi) |
| continue a finished run, or switch algorithm mid-study | save and seed a population | [`python/examples/06_restart.py`](python/examples/06_restart.py) |
| no control over include paths | the single header | [docs/embedding.md](docs/embedding.md#the-single-header) |

All of them reach the same 58 algorithms. They differ in who owns the loop and
what crosses the boundary, not in what you can run.

## Quick start

**C++, one algorithm.** `examples/nsga2_zdt1.cpp`:

```cpp
#include <mootation/mootation.hpp>
#include <iostream>

int main() {
    using namespace mootation;

    Problem<NSGAII_Individual>   prob;               // default problem: ZDT1, n = 6
    DataVault<NSGAII_Individual> vault(100, prob);   // population size 100

    // defer_setup keeps the run reproducible: the one-argument constructor
    // would draw the initial population before set_seed() could matter.
    Optimizer<NSGAII_Individual, NSGAIICore<NSGAII_Individual>>
        opt(std::move(vault), defer_setup);
    opt.get_algorithm().set_seed(12345);
    opt.setup();
    opt.optimize(500);                               // 500 generations

    auto& v = opt.get_vault();
    for (std::size_t i = 0; i < v.active_n(); ++i)
        std::cout << v.objectives_of(i)[0] << ' ' << v.objectives_of(i)[1] << '\n';
}
```

```bash
g++ -std=c++17 -O2 -Iinclude examples/nsga2_zdt1.cpp -o nsga2_zdt1 && ./nsga2_zdt1
```

**C++, you own the loop.** The algorithm is picked by name from a settings
struct or a `key = value` file; `ask()` hands you candidates, `tell()` takes
their objective values and returns the next candidates. The batch size is the
algorithm's choice, not `pop_size`: generational algorithms hand over a whole
offspring generation, steady-state ones one candidate at a time.

```cpp
#include <mootation/embed.hpp>

mootation::Session s(mootation::Settings::from_file("run.cfg"));

std::vector<std::vector<double>> F;
for (auto X = s.ask(); !X.empty(); X = s.tell(F)) {
    F.assign(X.size(), std::vector<double>(s.n_objs()));
    for (std::size_t i = 0; i < X.size(); ++i)
        F[i] = my_simulator(X[i]);                   // one candidate's variables in
}
mootation::Result r = s.result();
```

`examples/run.cfg` documents every key of the settings file. Details, the
`run()` one-liner, warm starts and the compile-time cost of the run-time
dispatch: [docs/embedding.md](docs/embedding.md).

**Python.**

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
satisfied); `batch=` receives a whole generation in one call, for evaluators
that cost real time; `save_population=` / `seed_population=` are the warm
start; `mootation.algorithms()` lists the 58 names. `res.ignored` names any
knob you set that this algorithm does not have. Six worked scripts live in
[python/examples/](python/examples/README.md); `mootation.Problem` /
`mootation.Config` / `mootation.run_raw` map onto the C++ API one-to-one when
`minimize` is not enough.

**External programs, from a TOML file.**

```bash
python -m mootation.run --show --check python/examples/demo.toml   # validate first
python -m mootation.run python/examples/demo.toml                  # then run
```

The file names the steps (argv arrays, per-platform overrides), how to parse
their output (CSV, JSON, regex, columns, or a Python function), what to do on a
failed evaluation, and where the journal goes. `--check` verifies executables,
templates, bounds, algorithm names and population-size constraints before
anything expensive starts. [docs/running.md](docs/running.md).

**C, and everything with an FFI.**

```c
#include <mootation/capi.h>
#include <stdlib.h>

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
moo_close(s);
```

Nothing but C types crosses the boundary: the configuration is one string in
the settings format, candidates and objectives are flat row-major `double`
arrays, every failing call returns a negative int or `NULL` and leaves a message
in `moo_last_error()`. `capi/ctypes_demo.py` drives the same loop from Python
with `ctypes` alone; `capi/wrappers/` has Julia, R and MATLAB adapters that
have not yet been run by anyone. [docs/embedding.md](docs/embedding.md#the-c-abi).

## Choosing an algorithm

| family | count | reach for it when |
|---|---|---|
| Pareto-dominance & diversity (NSGA-II, SPEA2, ...) | 7 | two or three objectives, a baseline everyone recognises |
| Reference-point, NSGA-III family | 4 | many objectives, a regular front; `pop_size` must be a Das–Dennis lattice size |
| Decomposition, MOEA/D family | 16 | scalarisable problems, many objectives, cheap generations |
| Indicator-based (IBEA, HypE, R2, ...) | 11 | one quality indicator is what you care about; cost grows with the objective count |
| Reference-vector / angle-based (RVEA, ...) | 7 | irregular or badly scaled fronts |
| Clustering-based | 11 | disconnected or irregular fronts |
| Archive-based | 2 | the non-dominated history is the answer, not a fixed population |

Seventeen algorithms need `pop_size` to be an exact lattice size for the
objective count, and the M2M family needs it divisible by the subregion count;
`python -m mootation.run --check` names the constraint and the nearest valid
sizes, and every core reports the knobs it does not have instead of dropping
them. Tables with every algorithm, year, file and DOI: [docs/algorithms.md](docs/algorithms.md).

## Fidelity

Every algorithm was checked against its primary source in three passes: a
dual-blind line-by-line audit with arbitration, an adversarial refutation pass
over every finding, and a final pass with one checklist per algorithm mapping
every equation, pseudocode line and parameter of the full paper to a code
location. Where the paper is explicit, the code follows it; where the paper is
silent, ambiguous or contradicts itself, the header quotes the passages and
states which reading was taken and why. Where a literal reading measurably
breaks the method, the header records the measurement and the switch that
selects the letter. Every file transcribes a paper.

The code is warning-clean at `/W4 /WX` and `-Wall -Wextra -Werror`, and clean
under ASan / UBSan over all algorithms; CI enforces both. What none of this
gives you is a reproduction of the papers' published results: the test suite
checks that every algorithm converges on DTLZ2 within a coarse tolerance and
that constraint handling is live, nothing more. The verification reports quote
the papers at length and are therefore kept outside the repository; the
conclusions live in the headers.

## Benchmarks and campaigns

`mootation.benchmarks` holds 216 problems across 11 families, 2 to 15
objectives, each with bounds, an evaluator, the reference point a hypervolume
needs and, where a closed form exists, a sampler of the true Pareto front.
`mootation.run.metrics` computes IGD, IGD+ and hypervolume (exact up to five
objectives, Monte-Carlo above). A `[campaign]` section in a run file turns a
family × objective-count × algorithm × seed cross product into resumable jobs
with a convergence trajectory each, locally or as a SLURM array:
[docs/running.md](docs/running.md#campaigns).

## Project layout

```
include/mootation/
  algorithms/        58 headers, one algorithm each; srv_strategy.hpp is a shared helper
  algorithms.def     the X-macro list every interface is generated from
  operators/         SBX, polynomial mutation, DE, Liu–Li, binary crossover, bit-flip
  problems/          DTLZ1-4 and ZDT1-3 in C++, and the macro that defines a problem
  io/                population files and the evaluation log
  mootation.hpp      the umbrella header (compile-time algorithm choice)
  embed.hpp          Settings, run(), Session: run-time choice, ask/tell
  capi.h             the C ABI
examples/            seven C++ programs, run.cfg with every settings key
capi/                the C ABI implementation, a C99 smoke test, ctypes_demo.py, wrappers/
python/mootation/    the binding, benchmarks/, run/ (TOML layer, campaign, metrics), tui/
python/examples/     six scripts and four run descriptions
tests/               CTest suite; the long convergence and constraint tests run nightly
tools/amalgamate.py  flattens everything into one header
docs/                algorithms, embedding, running, writing-an-algorithm
```

## Roadmap

In order of how much friction each removes.

- A true checkpoint. Warm starts work everywhere; the RNG position and each
  algorithm's own bookkeeping are still not serialisable, so a resumed run
  starts from the same population but not on the same trajectory.
- A built-in parallel evaluator: `batch_executor` is the hook, `n_workers`
  should be a parameter.
- NumPy-native Python signatures, and binary genomes in the Python binding and
  the C ABI.
- Native permutation and integer operators.
- GD+ and a C++ metrics module (hypervolume exists only inside three algorithms
  and in Python); a Wilcoxon test in the campaign comparison.
- Exact hypervolume subset selection at two objectives, where it costs
  `O((n-k)k + n log n)` and the greedy removal the papers specify is the
  weakest of the three greedy schemes. It would be a declared deviation, not a
  fix, so it needs the switch-and-measure treatment.
- Exact reference fronts for the partially degenerate problems (DTLZ5, DTLZ6,
  MaF6 from 4 objectives, WFG3 from 3): either the published characterisation
  of the non-degenerate part or the constraints that make them truly
  degenerate. Sampling and filtering does not work — it returns points that are
  merely unbeaten within the sample. The limitation is warned about today.
- Running the Julia, R and MATLAB wrappers on a machine that has them.

## Contributing, citing, license

Contributions follow one rule, that an implementation must match the paper it
claims: see [CONTRIBUTING.md](CONTRIBUTING.md) and
[docs/writing-an-algorithm.md](docs/writing-an-algorithm.md) for the Core
contract and the header convention.

Each release is archived on Zenodo; cite the concept DOI, which always resolves
to the newest release, and cite the paper of every algorithm you use, since
that is where the method comes from.

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

[Apache License 2.0](LICENSE). See also [NOTICE](NOTICE) and, for
contributions, the [CLA](CLA.md).
