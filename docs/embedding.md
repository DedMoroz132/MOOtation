<!-- SPDX-License-Identifier: Apache-2.0 -->
# Embedding MOOtation in your own program

Three shapes, from the most static to the most portable: the algorithm fixed at
compile time (`Optimizer<Ind, Core>`), the algorithm chosen by name at run time
with either the library or you owning the loop (`embed.hpp`), and the same loop
behind a C ABI for other languages. All three run the same 63 cores from
`include/mootation/algorithms.def`.

## Compile-time choice: `Optimizer<Ind, Core>`

```cpp
#include <mootation/mootation.hpp>

using namespace mootation;
Problem<NSGAII_Individual>   prob;               // your Problem<> specialisation
DataVault<NSGAII_Individual> vault(100, prob);   // pop_size 100
Optimizer<NSGAII_Individual, NSGAIICore<NSGAII_Individual>> opt(std::move(vault), defer_setup);
opt.get_algorithm().set_seed(12345);
opt.setup();                                     // or opt.setup_with_seed(vars, objs) for a warm start
opt.optimize(500);                               // 500 generations; opt.step() runs one
auto& v = opt.get_vault();                       // v.active_n(), v.objectives_of(i), v.get_variable(i, j)
```

`defer_setup` matters for reproducibility: the one-argument constructor calls
`setup()` from its own body and draws the initial population before
`set_seed()` could apply. Each core has its own individual type (`NSGAII_Individual`,
`MOEAD_Individual`, ...); the pairing is the line in `algorithms.def`.

### Your problem

A problem is a specialisation of `Problem<>` for a tag individual, duck-typed:

```cpp
struct MyTag : mootation::NSGAII_Individual {};
namespace mootation {
template <> class Problem<MyTag> {
public:
    std::vector<std::pair<std::optional<double>, std::optional<double>>> bounds;  // per variable
    int  get_vars_n()     const { return 3; }
    int  get_bin_vars_n() const { return 0; }     // > 0 for a binary / mixed genome
    int  get_objs_n()     const { return 2; }
    int  get_lims_n()     const { return 0; }     // constraint values per individual
    void calc_objs(MyTag& ind) const {            // fill ind.objectives (and ind.limits) from ind.variables
        /* ... */
    }
};
}
```

[`examples/custom_problem.cpp`](../examples/custom_problem.cpp) does exactly
this for the Kursawe problem in about forty lines.
[`include/mootation/problems/benchmarks.hpp`](../include/mootation/problems/benchmarks.hpp)
carries DTLZ1-4 and ZDT1-3 as plain specs plus the macro
`MOOTATION_DEFINE_PROBLEM(Tag, Base, Spec)` that writes the specialisation for
you; the full WFG / MaF families live on the Python side. A binary or mixed
genome is the same interface with `get_bin_vars_n() > 0`
([`problems/zdt1_mixed.hpp`](../problems/zdt1_mixed.hpp),
[`examples/ibea_mixed.cpp`](../examples/ibea_mixed.cpp)); 47 of the 63 cores
accept one, the other 16 refuse it at `setup()` with a message.

### External or parallel evaluation

`DataVault::set_batch_executor` replaces per-individual `calc_objs` with one
call per generation over the whole set of unevaluated individuals
(`BatchRequest` in, `BatchResponse` out; the contract is at the top of
[`batch_executor.hpp`](../include/mootation/batch_executor.hpp)). This is where
a thread pool, a process pool or a simulator queue plugs in;
[`examples/batch_demo.cpp`](../examples/batch_demo.cpp) shows it. Steady-state
cores still evaluate one candidate per `step()`, because that is how those
algorithms are defined.

### Diagnostics

Some cores substitute a parameter that cannot be honoured: a subregion count
that is not a lattice size comes back rounded up, and the derived quota changes
with it. Each such case is named in the file's header and announced at run time
through `set_warn_handler` (silent by default). Where a core cannot honour the
configuration at all (an exact lattice size, `pop` divisible by `K`, a binary
genome) it throws `std::invalid_argument` from `setup()` with the nearest valid
sizes in the message.

## Run-time choice: `embed.hpp`

`#include <mootation/embed.hpp>` instantiates all 63 cores in that translation
unit so that the algorithm can be picked by name. That is a real compile-time
cost (about a minute); a program that always runs one algorithm should use
`Optimizer<Ind, Core>` directly.

### Settings

A plain struct, and the same fields as a `key = value` file:

| field / key | meaning |
|---|---|
| `algorithm` | any name from `mootation::algorithm_names()` |
| `pop_size`, `max_gen`, `seed` | the run |
| `max_evaluations` | a budget in evaluations instead of `max_gen` steps (0, the default, runs `max_gen`); checked between steps, so a generational core overshoots it by less than one generation. Use it whenever algorithms are compared or a budget is what you pay for: a step is one offspring for NIMMO and a fifth of the population for MOEA/D-DRA and -AWA |
| `n_vars`, `lower`, `upper` | per-variable bounds; one scalar broadcasts to all `n_vars` (`Settings::set_box(n, lo, hi)`) |
| `n_objs`, `n_cons` | objective and constraint values per candidate |
| `constraints` | `none`, `feasibility`, `cdp`, `eps_constraint` |
| `seed_population`, `on_size_mismatch` | warm start: a population file, and `error` / `truncate` / `pad` when its size differs from `pop_size` |
| `eta_c eta_m pc pm T delta nr kappa K n_clusters theta alpha F CR div` | optional knobs; anything absent keeps the paper's default |
| `normalize` | 0 or 1: the objective normalization of IBEA, R2-IBEA, Two_Arch2 and MOEA/D-AM2M; each header says which setting is the paper's |
| `crossover`, `mutation` | words: `sbx` / `uniform` / `blx_alpha` / `spx` / `rex` / `undx` / `pcx`, and `polynomial` / `gaussian` / `cauchy` / `uniform_reset` / `mixture` / `mixture_cauchy` — NSGA-II, IBEA-ε+, SPEA2+SDE, AGE-MOEA and SMS-EMOA take both, MOEA/D-DE the mutation; the defaults are the papers' SBX and polynomial mutation |
| `mutation_scale`, `mixture_q`, `blx_alpha` | their parameters: s of the gaussian (0.1) and Cauchy (0.05) steps as a share of ub − lb, the share of noisy steps in a mixture (0.1), BLX's α (0.5) |
| `sbx_var_prob` | SBX's share of crossed variables (0.5) for this run |
| `bound_repair` | a word: `clip`, `reflect`, `random`, `midpoint`, `resample`, `wrap` or `native` — what an operator that can leave the box does with the variables it put outside (DE in MOEA/D-DE, MOEA/D-DRA and LIS/LCS; Liu & Li's operators; DCEA, HLMEA and NAEMO); each keeps its own default, and the others report it as ignored (`operators/bound_repair.hpp`) |

`Settings::from_file("run.cfg")` reads the file; an unknown key is an error, not
a silent no-op. [`examples/run.cfg`](../examples/run.cfg) is a commented copy
of every key. A knob the chosen algorithm does not have is reported in
`Result::ignored` rather than dropped.

### `run()` and `Session`

```cpp
#include <mootation/embed.hpp>

mootation::Settings s = mootation::Settings::from_file("run.cfg");

// The library owns the loop: your evaluator fills F (and G, one row per
// candidate, each value <= 0 meaning satisfied) for a whole batch X.
mootation::Result r = mootation::run(s,
    [](const std::vector<std::vector<double>>& X,
       std::vector<std::vector<double>>& F,
       std::vector<std::vector<double>>& /*G*/) {
        for (std::size_t i = 0; i < X.size(); ++i) F[i] = my_simulator(X[i]);
    });

// You own the loop: ask() hands out candidates, tell() takes their values and
// returns the next batch; empty means the run is over.
mootation::Session sess(s);
std::vector<std::vector<double>> F;
for (auto X = sess.ask(); !X.empty(); X = sess.tell(F)) {
    F.assign(X.size(), std::vector<double>(sess.n_objs()));
    for (std::size_t i = 0; i < X.size(); ++i) F[i] = my_simulator(X[i]);
}
mootation::Result r2 = sess.result();
```

Both shapes give bit-identical output for the same seed;
[`examples/embed_asktell.cpp`](../examples/embed_asktell.cpp) runs both and
checks that they agree and that the run converged. **The batch size is the
algorithm's choice, not `pop_size`**: generational algorithms hand over a whole
offspring generation, steady-state ones (MOEA/D-DE, MOEA/DD, AdaW, HLMEA and
others) one candidate at a time. Size your loop off `X.size()`.

`Result` carries `variables`, `objectives`, `cv` (the summed violation),
`limits` (the individual constraint values, empty when `n_cons == 0`),
`ignored`, `generations`, `evaluations` and `size()`.

### Warm start

A population saved by one run can seed the next, with the same or a different
algorithm, because decision variables, objectives and constraint values are
what every core shares. Seeding costs zero function evaluations: the
objectives are read, not recomputed.

```cpp
s.seed_population  = "pop.csv";                       // or in run.cfg
s.on_size_mismatch = mootation::SizeMismatch::Truncate;
auto r = mootation::run(s, my_evaluator);             // Session(s) likewise

mootation::Session next(s, mootation::as_population(r));   // straight from a Result, no file
```

`mootation::io::save_population` / `load_population` are the file pair: CSV
with a `#` preamble carrying the settings that produced it, then `x1..`, `f1..`,
the individual constraint values `g1..gk` and `cv`. Seeding is refused, with the
mismatch named, when the file comes from a different variable or objective
count, and when a constrained run is handed a file without per-constraint
columns: `cv` is a sum and cannot be taken apart, so starting from it would
plant a population every one of whose constraints reads as satisfied.

This is a warm start, not a checkpoint: no RNG position and no per-algorithm
state are saved, so a resumed run starts from the same place but does not
follow the trajectory the uninterrupted run would have.

## The C ABI

`cmake -DMOOTATION_BUILD_C_API=ON` builds `libmootation.so` /
`libmootation.dylib` / `mootation.dll` (MSVC drops the `lib` prefix) from
`capi/mootation_c.cpp`; the header is `include/mootation/capi.h`, the CMake
target `MOOtation::c`. Only the `extern "C"` surface is exported. Nothing but C
types crosses the boundary: configuration goes in as one string in the settings
format above, candidates and objective values move as flat row-major `double`
arrays.

| function | does |
|---|---|
| `moo_version()`, `moo_version_string()` | library version |
| `moo_algorithm_count()`, `moo_algorithm_name(i)` | the registry |
| `moo_open(text)`, `moo_open_file(path)`, `moo_close(s)` | a session from settings text or a file |
| `moo_n_vars(s)`, `moo_n_objs(s)`, `moo_n_cons(s)`, `moo_generation(s)` | dimensions and progress |
| `moo_ask_count(s)` | candidates waiting; 0 means the run is over |
| `moo_ask(s, x, x_len)` | copies them out, `count` rows of `n_vars` doubles |
| `moo_tell(s, f, f_len, g, g_len)` | posts objectives (and constraint values, or `NULL, 0`); returns the size of the NEXT batch |
| `moo_result_count(s)`, `moo_result(s, x, x_len, f, f_len, cv, cv_len)` | the final population; any of the three pointers may be `NULL` with length 0 |
| `moo_ignored_count(s)`, `moo_ignored_name(s, i)` | knobs the algorithm does not have |
| `moo_last_error(s)` | the last error on the session, or on `NULL` the last `moo_open` failure |

Every call that can fail returns a negative int or `NULL`; no exception escapes
the library. Two consumers run on every CI build: [`capi/smoke.c`](../capi/smoke.c),
compiled as C99 so that a C++-ism creeping into the header fails there, and
[`capi/ctypes_demo.py`](../capi/ctypes_demo.py), which drives the loop from
Python with `ctypes` alone. [`capi/wrappers/`](../capi/wrappers/) has the same
loop for Julia, R (through a small C shim) and MATLAB; read that directory's
README first: the ABI beneath them is tested, the three wrappers themselves have
not yet been executed by anyone. MATLAB's `loadlibrary` needs a header without
the export macro; `cpp -P -DMOO_API= include/mootation/capi.h` produces one.

## The single header

`python tools/amalgamate.py` flattens the library, `embed.hpp` included, into
one self-contained header for the cases where you cannot control include paths
(a competition judge, a plugin SDK, a Compiler Explorer link). It is generated,
not committed: CI regenerates it, compiles it standalone at `-Wall -Wextra
-Werror`, checks that `run()` and `Session` still agree there, and fails if any
private file reached it. Comments are kept, because the per-algorithm header
block is the point. Budget about a minute of compile time per translation unit
that includes it.

## Writing an algorithm

One class template with `set_seed` / `setup` / `setup_seeded` / `step`, plus one
line in `algorithms.def`; no base class, no registry. The contract, the vault's
slot and function-evaluation rules, and the header convention are in
[writing-an-algorithm.md](writing-an-algorithm.md).
