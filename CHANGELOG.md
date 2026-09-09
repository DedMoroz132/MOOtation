# Changelog

All notable changes to MOOtation are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Until 1.0.0 the public API may change in a minor release; breaking changes are
always listed under **Changed** or **Removed**.

## [Unreleased]

### Removed

- `srv_moead` (SRV on a steady-state MOEA/D). It was an extrapolation rather
  than a transcription — liu2022 puts the SRV strategy on an NSGA-III carrier
  (§III-C) and says nothing about a steady-state one, so no part of it could be
  checked against a paper — and measured it did not converge: median IGD over 3
  seeds at 30 000 FE was 0.41 on DTLZ2 and 0.35 on ZDT1, against 0.060 and
  0.005 for `srv` on the same budget. It is maintained outside the public tree.
  The SRV strategy itself is unaffected: `srv`, `srv_nsga3` and the shared
  `srv_strategy.hpp` stay. The `SRVMOEAD_Individual` alias left `individuals.hpp`
  with it. The public list is 58 algorithms.
- `ibea_eplus_clean` (IBEA-ε+ with the mIBEA front filter). It was never a
  transcription of a paper but a composition kept as the control arm of an
  unpublished ablation study; it is now maintained outside the public tree.
  The public list is 58 algorithms. The paper-backed IBEA variants
  (`ibea_eplus`, `ibea_hd`, `mibea`, `r2ibea`) are unchanged.

### Changed

- README rewritten around the question "which way in do I want", at half the
  length: install, one quick start per interface, how to choose an algorithm,
  what the library is not. The algorithm tables moved to `docs/algorithms.md`,
  the TOML layer's reference to `docs/running.md`, embedding / warm start /
  the C ABI / the single header to `docs/embedding.md`. `README.ru.md` mirrors
  it section for section.
- The thirteen real-valued-only cores (CLIA, DCEA, DEA-GNG, DHEA, ETEA, HCCA,
  HLMEA, IF-MaOEA, ISDE+RD, MaOEA-3C, MOEA/D-DS, NAEMO, Two_Arch2) refuse a
  binary or mixed genome at `setup()` with a message. Twelve of them used to
  accept it and run with every offspring bit silently left at zero.

### Added

- `mootation.run.campaign`: a benchmark campaign runner — every selected
  algorithm on every selected problem, several seeds each, one
  `trajectory.jsonl` (IGD / IGD+ / HV against evaluations) plus `meta.json`
  and `final.csv` per run, resumable, shardable (`--shard i/n`, `--job k`,
  `--workers`), with `--emit-slurm N` writing a SLURM array script and a
  `jobs.txt` for GNU parallel. `mootation.run.metrics` carries the
  indicators (exact WFG hypervolume up to 5 objectives, Monte-Carlo above).
- The Python binding takes an observer: `Config.on_generation` /
  `Config.record_every` (and `minimize(on_generation=..., record_every=...)`)
  call back with the current answer set every k generations, which is how a
  convergence trajectory is recorded without re-running at several budgets.
- TUI: Campaign, Compare and Explore tabs when the config describes a builtin
  campaign (progress per problem x algorithm, median [q1, q3] table with CSV
  export, per-run trajectory plot in the terminal). `python/examples/campaign.toml`.
- `set_de_repair()` on MOEA/D-DE and MOEA/D-DRA selects clamping instead of
  the paper's random reset of out-of-box genes (see the headers for the
  measured difference).

### Fixed

- **Scale invariance.** A new test (`tests/test_scale_invariance.cpp`) runs
  every algorithm on DTLZ2 and on the same problem with every objective
  multiplied by 2^10, a power of two so the scaling is exact and a
  scale-equivariant algorithm has to reproduce its result bit for bit. 53 of
  the 59 did (the count before `srv_moead` was moved out). Of the six that did not:
  - `two_arch2` computed I_eps+ on raw objectives, so its fitness
    `-exp(-I/0.05)` underflowed to exactly zero for every pair once the
    objectives grew, and the convergence archive stopped ordering anything
    (IGD 0.0667 -> 0.2712 at 2^10, 0.5130 at 2^20 — the saturation value).
    The paper is silent on scaling while the IBEA fitness it reuses verbatim
    does scale (Alg.2 steps 2.1-2.2), so the scaled reading is now the
    default; `set_normalize(false)` restores Eq.1. At the native scale the two
    are indistinguishable (3 seeds, 30 000 FE: DTLZ2 0.0605 vs 0.0603, ZDT1
    0.00397 vs 0.00399).
  - `ar_moea` initialised the adapted reference set R' from the unit-simplex
    points while consuming it in translated objective coordinates, so the
    first generation compared a simplex against objectives of whatever
    magnitude the problem used. R' is now built in the coordinates it is used
    in. Free at the native scale (DTLZ2 0.05432, ZDT1 0.00395, unchanged).
  - `r2ibea` keeps the paper's letter: Eq.4 is written on raw objectives.
    `set_normalize(true)` transports the whole of Eq.5 into normalised
    coordinates — every normalised range is 1, so z* is -1 on every axis — and
    makes the run bit-identical at every scale factor. It is still not the
    default: 3 seeds, 30 000 FE, median IGD on ZDT1 is 0.0059 for the letter
    against 0.1489 for the scaled reading, with no overlap between seeds,
    while DTLZ2 is a tie. Eq.5's single shift by the largest range is exactly
    what makes an improvement on a narrow axis worth less than one on a wide
    axis, and normalising per axis discards that. The first version of this
    switch was wrong — it divided by the per-axis range but left z* shifted by
    the largest one, leaving a constant `max_range/range_j` inside the
    Tchebycheff max that was largest on the NARROWEST axis, inverting the skew
    instead of removing it. The verdict above is measured against the
    corrected version. The underflow of `exp(-I_R2/kappa)` is detected at run
    time and reported through `set_warn_handler`, on the FRACTION of dead
    pairs over the population rather than on all of them being dead: a total
    underflow is obvious from the output, a partial one looks like a working
    run and is not. Measured afterwards, the switch also has a positive
    indication and it is the mirror image: on DTLZ2 skewed by per-axis factors
    1, 10 and 100 the letter degrades to a median IGD of 0.0937 while the
    scaled reading holds at 0.0735, the value it reaches unskewed. So the rule
    is what the spread of ranges MEANS — units, and scaling helps; the
    problem, and it throws information away. On ZDT1 the wide axis is the
    direction of convergence, which is why the letter wins there by 25.
- `ibea_eplus` gains `set_normalize(false)`, an experiment switch only: the
  paper's own Alg.2 scaling stays the default. It answers whether the cost
  measured in R2-IBEA is a property of per-axis scaling in general. It is not
  — for IBEA the scaling is free (median IGD on ZDT1 0.00393 against 0.00398
  raw, a tie). Separating the two halves of IGD says why: on ZDT1 the scaled
  R2-IBEA converges exactly as well (mean distance to the front 1e-4 against
  8e-5) but covers half the front (span of f1 0.49 against 0.94 over three
  seeds), spread evenly inside that half. R2 allocates effort by direction and
  rescaling the axes moves which piece of the front each weight vector owns;
  IBEA has no such partition, it only decides an order. The caution therefore
  belongs to the MOEA/D and NSGA-III families, RVEA and θ-DEA rather than to
  indicator methods, which is also why nearly every algorithm in the skew list
  in `docs/algorithms.md` is a weight-vector or angle method. Which of the two cases
  you are in has a cheap test, now written up: the ratio between the
  per-objective ranges of the answer set holds steady when the spread is a
  unit and slides toward 1 when it is the problem (ZDT1 3.7 -> 1.03 over a
  run, DTLZ2 flat at 1.00). Freezing the divisor at the first pool's ranges
  was tried as a way to separate the two automatically and is recorded in the
  header as rejected: it halves the cost on ZDT1 but is worse than both
  alternatives on a skewed problem, because a frozen divisor freezes the
  resolution as well.
  - `adaw`, `moead_awa` and `mombi2` depend on the magnitude of the objectives
    through a constant of their own papers (z* = best - 1e-4 in AdaW's
    footnote 2, z* = min f - 1e-7 in MOEA/D-AWA's Step 1.2, eps = 1e-3
    compared against a raw range in MOMBI-II's Sec 5.1). Changing those would
    make the implementations unfaithful, so each header now states the
    dependence with the measured size and the test exempts them by name.

- The Python binding's `batch=` evaluator was called with ONE individual per
  call: `evaluate_batch` went through the per-individual `calc_objs` path,
  so "batched evaluation" cost the same number of Python round-trips as the
  plain function. The binding now installs a `BatchExecutor` on the vault,
  and a generational algorithm hands the batch function its whole offspring
  set in one call (steady-state cores still produce batches of one, as they
  do everywhere else).
- A constrained Python run seeded from a saved population silently started
  with every constraint reading as satisfied: the Python population file
  carries x, f and the aggregate cv only, and the binding passed no
  per-constraint values. `minimize(seed_population=..., constraints=...)`
  now recomputes the constraint values of the seed from `constraints`
  (no objective evaluations are spent), `Config.seed_limits` carries them,
  and the binding refuses a constrained warm start without them — the same
  rule `embed.hpp` already applied.
- `python -m mootation.run --check` validated `pop` only against an
  external problem's `n_objs`. For `kind = "builtin"` it now checks every
  algorithm against every objective count the benchmark selection contains,
  which is where a sweep with an exact-lattice algorithm actually breaks;
  the summary line no longer prints "about 0 evaluations" when `pop = 0` /
  `gens = 0` delegate the budget to the problems.
- `capi.h` guards its `MOO_API` definition, so `-DMOO_API=` on the
  preprocessor command line strips the decoration as the MATLAB wrapper's
  instructions require; `MOOtation.jl` exported a function that did not
  exist (`ask_tell`) and exports the real ask / tell / result API instead.
- CI: the `full-tests` pull-request label the comments promised now exists
  (the nightly job also runs on a labelled PR).
- Benchmarks (`python/mootation/benchmarks`), against the source papers:
  the DTLZ7/MaF7 reference front dropped the factor (1+g) = 2 (f_M was a
  full M too low); MaF2 evaluated plain DTLZ2; MaF6 used (1+g) instead of
  (1+100g); MaF7 used K = 10 instead of 20; MaF8/MaF9 used [-2, 2]^2 instead
  of [-10000, 10000]^2 and MaF9 forbade the whole exterior instead of the
  paper's regions; the WFG1 reference front used the convex instead of the
  mixed shape; ZDT4 had 30 instead of 10 variables; the SDTLZ scaling base
  was 10 at every M instead of Table VIII's per-M values; the BT5 and MOP4
  reference fronts kept the dominated part of the curve; the reference-frame
  cache was never read (file name mismatch). The cache is regenerated.
- Algorithms, from a second paper-fidelity pass plus a convergence audit on
  DTLZ2/ZDT1 (30000 evaluations, 3 seeds): MOMBI-II (the |.| in the ASF froze
  ZDT1), MOEA/D-AM2M (frozen normalisation stalled ZDT1), IF-MaOEA (reference
  set of ~2N points forced a biased trimming path every generation), DCEA
  (K-means iterations, V update period in NFE, z_min from the previous R),
  MaOEA-3C (front discard before angle selection), RD-EMO / mIBEA / EDV /
  DHEA (one vs two SBX children), ISDE+RD (partner drawn from the growing
  pool), HLMEA (c_old roll), R2-IBEA / MOMBI-II (tournaments with
  replacement), Two_Arch2 (eta = 15 per the paper). Headers of ~20 more
  algorithms corrected where they misdescribed the paper.
- Algorithms, from a third pass: one checklist per algorithm mapping every
  equation, pseudocode line and parameter of the full paper to a code location
  (2184 rows over the 60 files then in the tree; 47 of them ended in a code
  change). The behavioural ones: MOEA/D-DE and MOEA/D-DRA reproduce the
  papers' letter (DE without the j_rand gene, r2 / r3 drawn independently,
  unbounded polynomial mutation, random reset of the final out-of-box gene;
  `set_de_repair(Clip)` keeps the previous clamping); MOEA/D-M2M, SMS-M2M,
  HLMEA and ISDE+RD measure subregion angles from the origin and shift only
  objectives with a negative minimum (ISDE+RD keeps the ideal shift by default
  and puts the letter behind `set_ideal_shift(false)`, with the measured reason
  in its header); MOEA/D-M2M, SMS-M2M, MOEA/D-AM2M, APRD and Liu & Gu 2011
  draw mates once (self allowed) instead of rejection loops and top up without
  replacement; MOEA/D-AM2M builds the exact two equal-layer lattices of §V-C
  (110 = 55 + 55) instead of a silently truncated 220-point one; A-NSGA-III
  deletes reference points only in the paper's "ideal scenario" with a
  one-generation cooldown (`set_adaptation_mode`); APRD truncates its archive
  by angle, not Euclidean distance; NRV-MOEA sets P <- P_c as Alg.1 line 15
  says (`set_population_pruning(true)` restores the old behaviour; DTLZ2 IGD
  0.088 -> 0.061); DEA-GNG increments g after selection; SRV inserts the
  duplicate extreme twice and runs k-means for 2m iterations; R2-IBEA applies
  the P_c gate of Alg.2; EDV uses p_m = 1/pop_size; IF-MaOEA adjusts reference
  points per §2.4 of its cited source; MOEA/DDS increments G after the DDS and
  mates on the last DDS's d_2; DHEA normalises Div on the retained pool;
  MOEA/D, MOEA/DD, AdaW and MOEA/D-AWA draw mating indices as the papers do;
  GrEA defaults to div = 9; VaEA takes extremes over the whole last front;
  Liu & Gu 2011 uses S = ceil(sqrt(N)) exactly. Each change is declared in
  the file's header with the paper location and, where behaviour moved, the
  measured effect on DTLZ2 / ZDT1.

### Changed

- Long tests are labelled `nightly`. `test_convergence` and `test_constraints`
  each drive every algorithm and together are ~10 of the suite's ~11 minutes;
  every push now runs `ctest -LE nightly` and a scheduled job runs the full set,
  including a sanitized pass over the long ones. The trade is deliberate: those
  two guard slow rot, not the kind of breakage a compiler catches.

- **The project is now MOOtation** (was OptSearch). The C++ namespace is
  `mootation`, macros are `MOOTATION_*`, headers live under
  `include/mootation/`, and the CMake package and target are `MOOtation` /
  `MOOtation::MOOtation`.
- `algorithms.def` moved from `tests/` to `include/mootation/`. It is part of
  the public surface now: `embed.hpp` includes it three times to build the
  run-time dispatch, so it has to ship with the headers. The install rule
  matches `*.def` as well as `*.hpp` for the same reason.
- Citations corrected against Crossref. Three DOIs pointed at the wrong paper
  entirely — `hlmea` resolved to an unrelated fuzzy-clustering article,
  `if_maoea` to a bridge-tower reliability study, `r2ibea` to MOMBI — and three
  headers carried invented descriptive titles instead of the papers' own
  (`moead_de`, `moead_dra`, `moead_am2m`). All 54 unique DOIs now resolve to
  the work the header claims; SPEA2's is registered with DataCite rather than
  Crossref, which is noted in its header so a future sweep does not read the
  404 as an error.

### Fixed

- `io::save_population` wrote `active_n()` individuals — the third place in the
  codebase with this bug. Steady-state cores park a scratch slot at active
  index `pop_size()`, so the saved population contained one unselected
  offspring, which would then have been loaded back as if it had been chosen.

- The build used TWO different Python interpreters. `find_package(Python)` at
  the top level resolved 3.13 while pybind11, falling back to the deprecated
  `FindPythonInterp` modules, built the extension against 3.12 — and a `.pyd`
  carries an ABI tag, so the module the tests imported could never be the
  module that was built. `PYBIND11_FINDPYTHON=ON` makes pybind11 use the same
  `FindPython` result as everything else. The test's interpreter is also
  pinned explicitly now: `${Python_EXECUTABLE}` was empty in that scope, and an
  empty interpreter variable does not fail loudly — the command degrades to the
  script path alone and Windows picks something through the file association.
  None of this was visible while the extension was a flat module that any
  Python could import; it surfaced the moment it became a package with an
  ABI-tagged `_core` inside.
- `mootation.run` was two things at once: the dispatch function the binding
  exported, and the TOML subpackage. An attribute and a submodule fighting over
  one name resolves differently depending on what has been imported, which is a
  bug waiting for someone to hit it. The function is `run_raw` now; the name
  `run` belongs to the subpackage alone.
- The Python binding returned `pop_size + 1` solutions for every steady-state
  algorithm (MOEA/D-DE, MOEA/DD, HLMEA, AdaW and the rest). Those cores park a
  persistent scratch slot at active index `pop_size()`, so `active_n()` is one
  too many, and the extra row is an unselected offspring — enough to skew any
  IGD or hypervolume computed from Python, and invisible unless you count the
  rows. `embed.hpp` already handled this; the binding did not.

- `find_package(MOOtation)` handed out a target named `MOOtation::mootation`
  while the build tree, the README and the CI consumer all use
  `MOOtation::MOOtation`. A project that worked through `add_subdirectory` or
  `FetchContent` failed on the installed package with "target was not found".
  The exported name now matches the alias (`EXPORT_NAME`), for the C ABI target
  as well.
- The installed package carried no `INTERFACE_INCLUDE_DIRECTORIES` at all, so
  every consumer of `find_package(MOOtation)` failed with "cannot open include
  file". `GNUInstallDirs` was included in the install block, below the
  `target_include_directories` call that reads `CMAKE_INSTALL_INCLUDEDIR`, so
  `$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>` expanded to an empty
  generator expression. The build tree kept working throughout, which is what
  made it easy to miss.
- The install rule matched `*.hpp` only, so `algorithms.def` never reached the
  install prefix and `<mootation/embed.hpp>` did not compile from an installed
  tree.
- An unknown `algorithm` name was accepted by `Settings` and only surfaced once
  the run reached its dispatch — for a `Session`, on the worker thread at the
  first `ask()`. It is now rejected where the run is created, with the list of
  known names in the message.
- `Session::ask()` returned a reference into the batch the worker owns. A
  caller writing `const auto& X = ask(); tell(F); use(X);` was racing the
  worker. It returns by value now.

### Added

- **Saving a run and restarting from it** (`mootation.io` in C++,
  `mootation.persistence` in Python). Three optional files, all CSV with a `#`
  preamble that `pandas.read_csv(comment="#")` reads directly:
  - the **population** — what survived;
  - the **evaluation log** — every call, including the ones the optimizer
    discarded, which for an expensive evaluator is the bulk of the cost;
  - the settings that produced them, carried in the preamble, so one file
    answers both "what is this" and "what made it".

  From Python: `minimize(..., save_population=, log_evaluations=,
  seed_population=, on_size_mismatch=)`. Seeding costs ZERO evaluations — the
  objectives are read, not recomputed.

  This is a warm start, NOT a checkpoint: no RNG position, no per-algorithm
  state. That is a deliberate limit, and it is what makes a population saved by
  NSGA-II loadable by MOEA/D, since only the fields every algorithm shares are
  written. A resumed run does not reproduce what the uninterrupted one would
  have done; it starts from the same place.
- `python/examples/06_restart.py` walks all of it, including switching
  algorithm and population size between runs.

- **One Python package.** `mootation` now holds everything: the compiled
  extension (`mootation._core`), `minimize()`, `mootation.benchmarks`,
  `mootation.run` and `mootation.tui`. Nothing is imported eagerly — importing
  the parent must not require a compiler, or the pure-Python layers would stop
  working on the machines that most want them.
- `pyproject.toml`: `pip install .` builds and installs the extension;
  `.[bench]` adds NumPy, `.[tui]` adds Textual, `.[all]` both. A
  `mootation-run` console script comes with it.
- `mootation.minimize(fn, bounds, n_objs, ...)` — the whole API for the common
  case, with constraints, batched evaluation and per-algorithm knobs. An
  unknown knob is a `TypeError` naming the alternatives, not a silent no-op.
- `mootation.benchmarks.solve()` and `.igd()` — the bridge between the 216
  Python benchmark problems and the 59 C++ algorithms.
- Five worked examples in `python/examples/`, each runnable and each run in CI:
  basics, constraints, benchmark comparison, expensive evaluators, and
  many-objective (including the Das-Dennis population-size trap).
- HLMEA-9: the Alg.1 allocation fills every subregion to `floor(N/W)`, so the
  answer set is short of `pop_size` whenever W does not divide N — at the
  paper's own m=3 default W=15, a pop_size of 91 yields 90. Declared, and now
  warned about, because W is chosen by the library rather than by the user.

- `python/mootation/benchmarks/` — the standard test suites, 216 problems
  across 11 families (ZDT, DTLZ, WFG, MaF, Polygon, IPolygon, MOP, BT and the
  inverted/scaled/minus DTLZ variants), each with bounds, an evaluator and,
  where a closed form exists, a sampler of the true Pareto front. Ported from
  the private pipeline and verified against it: 648 random evaluations and 44
  reference fronts across all 216 problems, zero mismatches.
  Reference frames are resolved lazily — doing all of them at import took 136
  seconds — and nothing is ever written back into the package directory.
  Needs NumPy; the config, parser, journal and runner layers still need nothing.
- `python/mootation/tui/` — the terminal interface: four
  read-only screens (Config, Problems, Algorithms, Monitor) over the same
  `load`/`validate` the CLI uses, so the UI cannot disagree with `--check`.
  Needs Textual. `python/test_tui.py` renders every screen headless through
  `App.run_test`.
- `[benchmarks]` in the config: `families` x `objectives`, or an explicit
  `problems` list, resolved against the registry with near-miss hints. The
  spec's own example name `DTLZ2_M3` does not exist — it is `DTLZ2_3D` — which
  is the kind of thing this catches.
- `--problems` lists the registry; `--tui` opens the interface.

- `python/mootation/run/` — driving a run from a TOML file with objectives
  produced by external programs. Zero dependencies;
  `tomllib` is stdlib from 3.11.
  - `--check` validates before anything expensive starts: executables resolved
    for the current platform, `{x[i]}` within `n_vars`, bounds ordered and
    counted, algorithm names against `algorithms.def`, knobs against
    `knob_names()` in `settings.hpp`, and `pop` against each algorithm's own
    constraint (exact Das-Dennis lattice for the NSGA-III family, divisibility
    by `K` for M2M). Every complaint at once, exit 1, no UI.
  - Five output parsers — `csv`, `json`, `regex` (with `take = first|last|nth`),
    `columns`, and `python:module.function` for binary formats. A parse failure
    is an evaluation failure routed to `on_fail`; nothing is substituted.
  - Append-only JSONL journal, separate from the per-worker scratch, doubling
    as a content-addressed cache so re-proposed individuals are not
    re-evaluated. Survives a run killed mid-write: a torn final line costs that
    line, not the journal.
  - The registry and knob list are PARSED from the C++ headers rather than
    copied, so they cannot fall behind the library.
  65 tests, registered with CTest and run in CI on Python 3.11 and 3.13.

- `include/mootation/capi.h` + `capi/` — a C ABI, built as a shared library
  with `MOOTATION_BUILD_C_API=ON`. It is the bridge for every language that is
  not C++: ctypes/cffi, P/Invoke, Rust `extern "C"`, Julia `ccall`, MATLAB
  `loadlibrary`, Fortran `iso_c_binding`. Only C types cross the boundary —
  settings go in as one string, candidates and objective values move as flat
  row-major `double` arrays — so there is no struct to keep in sync between
  languages. No exception escapes: every fallible call returns a negative int
  or `NULL` and leaves a message in `moo_last_error()`. `capi/smoke.c` is
  compiled as C99 (a C++-ism in the header fails there, not downstream) and
  `capi/ctypes_demo.py` drives the same run from Python with `ctypes` alone.
- `include/mootation/embed.hpp` and `include/mootation/settings.hpp` — the
  embedding layer, for driving MOOtation from a program that already owns its
  evaluation loop. `Session::ask()` hands out candidates and `Session::tell(F)`
  takes their objective values and returns the next candidates; `run(settings,
  fn)` is the same thing with the loop inverted. Both pick the algorithm by
  name at run time and produce bit-identical results for the same seed.
  `Settings` is a plain struct with a plain `key = value` file format —
  unknown keys are an error rather than a silent no-op. `examples/embed_asktell.cpp`
  exercises both shapes and asserts they agree objective by objective.
  Batch size is the algorithm's choice, not `pop_size`: generational algorithms
  hand over a whole offspring generation, steady-state ones one candidate.
- `docs/writing-an-algorithm.md` — the Core contract for contributors: the four
  required entry points, the "active population is a valid answer after every
  step" invariant, the slot/function-evaluation rules of `DataVault`, the
  RNG-stream rule, and where constraint handling attaches.
- `include/mootation/problems/benchmarks.hpp` — DTLZ1-4, ZDT1-3 and a
  constrained C-DTLZ2 as compile-time specs, plus `MOOTATION_DEFINE_PROBLEM`,
  which generates the tag individual and its `Problem<>` specialization.
  `examples/benchmark_problem.cpp` runs one with and without its constraint.
- Optional pybind11 module (`MOOTATION_BUILD_PYTHON=ON`) exposing all 59
  algorithms. Its dispatch and per-algorithm individual types are generated from
  `include/mootation/algorithms.def`, so it cannot fall behind the library; optional setters
  are detected by SFINAE and a knob the chosen algorithm lacks is REPORTED in
  `result.ignored` rather than silently dropped.
- `tools/amalgamate.py` — flattens the library into a single header for
  environments with no include-path control. Generated, not committed; CI
  regenerates it, compiles it standalone at `-Werror`, and fails if a private
  file reached it.
- `tests/test_constraints.cpp` — runs all 59 algorithms twice on a constrained
  DTLZ2 (`x0 <= 0.5`), with the mode off and on, and fails if the two runs are
  identical (an inert `constraint_mode`) or if feasibility regresses. A CI step
  additionally greps for a declared-but-never-read `constraint_mode`.

### Fixed

- `hcca` — the MOEA/D update assigned every offspring to a UNIFORMLY RANDOM
  subproblem, discarding the home-subproblem index it had just computed and
  stored, which made the decomposition step undirected. DE children now update
  the neighbourhood of the subproblem they were bred for; SBX/PP children, which
  have no home subproblem, still fall back to a random one.
- `naemo` — the reference lattice was rounded UP by `generate_auto`, so on the
  default path `n` could exceed `pop_size`: at m=3, pop=100 it produced 105
  lines; at m=8, pop=150 it produced 330. Since one offspring is bred per line
  and every line keeps a member, this violated the invariant the header states
  and inflated the per-generation evaluation count. The default path now takes
  the largest attainable lattice <= pop_size (91 and 120 in those two cases).


## [0.1.0] — first public release

### Added

- 60 multi- and many-objective evolutionary algorithms, header-only, C++17, no
  dependencies outside the standard library:
  - **Pareto-dominance & diversity** — NSGA-II, SPEA2, SPEA2+SDE, GrEA, VaEA,
    AGE-MOEA, ETEA.
  - **Reference-point** — NSGA-III, A-NSGA-III, θ-DEA, AR-MOEA.
  - **Decomposition** — MOEA/D, MOEA/D-DE, MOEA/D-DRA, MOEA/DD, MOEA/D-AWA,
    AdaW, MOEA/D-M2M, MOEA/D-AM2M, SMS-M2M, MOEA/D-DS, RD-EMO, APRD,
    Liu–Gu 2011, I_SDE+ RD, DHEA, HLMEA.
  - **Indicator-based** — IBEA (ε+ and HD), IBEA-ε+ with front filter, mIBEA,
    R2-IBEA, HypE, MOMBI-II, NIMMO, IREA, EDV, IF-MaOEA, MaOEA-IAMD.
  - **Reference-vector** — RVEA, MaOEA-ARV, MBRA, NRV-MOEA, DEA-GNG,
    MaOEA/SRV, SRV-NSGA-III, SRV-MOEA/D.
  - **Clustering** — CA-MOEA, CAVA-MOEA, MaOEA/AC, MaOEA/C, MaOEA-3C, EMyO/C,
    crEA, DCEA, HCCA, LIS/LCS, CLIA.
  - **Archive-based** — Two_Arch2, NAEMO.
- Variation operators: SBX, polynomial mutation (NSGA-II reference variant),
  DE `rand/1/bin` with Clip and RandomReset repair, Liu–Li annealing arithmetic
  crossover and mutation, uniform binary crossover, bit-flip mutation.
- Reference-point generators: Das–Dennis simplex lattice and the two-layer
  variant.
- Genome types: real-valued in all 60; binary and mixed real + binary in 45.
  The other 15 have a real-valued reproduction operator and say so; those that
  could have silently dropped a binary genome now refuse it instead.
- Constraint handling: constraint violation with feasibility-first /
  constrained-domination handling in all 60, off by default to match the
  original unconstrained papers. Each header names the point where it attaches
  — the non-dominated sort, the scalarizing comparison, the archive admission
  test, the truncation rule — since only one comparison per algorithm is its
  actual preference relation.
- `mootation::set_warn_handler` — an opt-in, one-function diagnostic channel.
  Silent by default; when installed, it reports parameter substitutions a
  caller would otherwise only find by reading the header (a requested subregion
  count rounded up to an attainable lattice size, a weight set padded with
  duplicates).
- External and batch evaluation through a batch-executor interface, with lazy
  objective evaluation.
- Reproducibility: explicit `set_seed`, deterministic reference-point
  generation, paper-exact default parameters.
- Per-file fidelity documentation: primary source with DOI, a scheme of the
  generational cycle, paper defaults traced to their section, and a numbered
  list of declared deviations.
- CMake package with an `MOOtation::MOOtation` interface target, `install`
  rules, and `find_package` / `FetchContent` support.
- Test suite: convergence smoke tests across all algorithms and unit tests for
  the reference-point generators and variation operators.
- CI across g++ and clang, C++17 and C++20, with an AddressSanitizer and
  UndefinedBehaviorSanitizer job.

[Unreleased]: https://github.com/DedMoroz132/MOOtation/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/DedMoroz132/MOOtation/releases/tag/v0.1.0
