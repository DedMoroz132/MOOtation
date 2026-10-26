# Changelog

All notable changes to MOOtation are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Until 1.0.0 the public API may change in a minor release; breaking changes are
always listed under **Changed** or **Removed**.

## [Unreleased]

- `include/mootation/toml.hpp`: TOML 1.0 read into json.hpp's tree, as
  Python's `tomllib.loads` gives it — every string, number and key form,
  tables, arrays of tables, inline tables, tomllib's rules on defining and
  extending them, errors with their line and column; dates kept as their
  text; an integer past int64 an error (TOML 1.0's lossless rule). For the
  campaign engine in C++ (task 17, stage 5). `tests/test_toml.cpp`; the
  extension's `toml_to_json`.

- Changed (the owner's word, 26.10): `sqp_line_search = powell` is the
  default of the scalarization methods, after the A/B of stop, accept and
  powell; the runs of those with `local = sqp` change, `method_version = 4`.
- Added: the operator knobs where a core used SBX or polynomial mutation with a
  written-in probability (the owner's decision of 2026-10-26): `pm` in
  `a_nsga3`, `camoea`, `cava_moea`, `crea`, `emyo_c`, `grea`, `maoea_ac`,
  `maoea_c`, `maoeaarv`, `mbra`, `mombi2`, `nsga3`, `r2ibea`, `srv`,
  `srv_nsga3`, `vaea`; `pc` and `pm` in `adaw`, `edv`, `etea`, `hype`,
  `lis_lcs`, `maoea_3c`, `moead_awa`, `nrv_moea`; `pc` in `mosa` and `naemo`.
  Not set, each takes the value written in before (p_c = 1, p_m = 1/n; `edv`
  1/N, EDV-1): default runs unchanged, the golden fingerprints the same.
  `docs/embedding.md`'s table of operator knobs per algorithm gains `pc`.

- Added: the word knob `theta_ties` of `theta_dea` — `slot` (the default, TDEA-1:
  equal F_j in one cluster ordered by their slots) or `letter` (Definition 7's
  strict "<": copies share the first theta level); and the knobs `pc` and `pm`
  of `spea2` (not set: p_c = 1 and p_m = 1/n as before). Default runs unchanged
  (the golden fingerprints are the same).

- Changed (the owner's word, 25.10): `sqp_line_search = accept` is the
  default of the scalarization methods (`scalar` and its presets); the runs
  of those with `local = sqp` change, `method_version = 3`. A third value,
  `powell`: the merit function and the step length of Powell 1978, §4 (a
  penalty per constraint by (4.6), the step rule (4.7)-(4.8) with 0.1 and
  the change of the linearized merit; `slsqp.hpp`, SQP-7 to SQP-9).

- Changed (task 17, supplement 5, §3): C++ draws what the problems' specs
  held. The rotations of the rotated problems are Haar rotations from the
  library's generator (`suite/instances.hpp`: Householder QR of a normal
  matrix, R's diagonal positive), and the bbob-biobj and bbob-biobj-mixint
  instances are drawn by `suite/bbob.hpp` — new instances of 352 rotated
  problems and 220 bbob-biobj ones (version +1, `PROBLEM_CHANGED[4]`,
  reference version 4; ours, not COCO's). MaF8's and MaF9's polygons and
  MaF14's and MaF15's linkage are drawn in C++ too (MaF8_8D, MaF9_8D and
  MaF15 at 3, 5, 8, 15 objectives: version +1). `benchmarks/library_rng.py`
  draws the same in Python, the cross-check's; a test holds the rotations to
  the Haar distribution's moments.

- Fixed: `theta_dea` let copies of a solution crowd the first theta level.
  Definition 7 compares F_j with a strict "<", so two copies in one cluster
  were both theta-optimal; the first level then outgrew the clusters it
  covered, and the random choice of Step 20 dropped the only member of
  another cluster, leaving reference directions empty to the end of a run.
  Equal F_j are now ordered by their slots (TDEA-1, a declared deviation from the strict
  "<" of Definition 7).
  The default runs of `theta_dea` change: its three fingerprints are new.

- Fixed: a campaign entry of `nsga3`, `a_nsga3`, `theta_dea` or `srv_nsga3`
  with `n_ref_points` ran another population than it named: the config check
  accepted any `pop` with `n_ref_points` (the reference points no longer come
  from it, task 15, §2), but the job expansion still rounded `pop` to a
  Das-Dennis lattice (`campaign.fit_pop`). It now keeps the `pop`.

- `include/mootation/json.hpp`: JSON as Python's `json` module reads and
  writes it, byte for byte — a float as `repr()` (the shortest digits that
  read back, `std::to_chars`), `ensure_ascii` escapes, `json.dumps`'s layouts
  with and without `indent`, `json.loads`'s numbers, NaN and the infinities,
  a key given twice; for the store and the campaign engine in C++ (task 17,
  stage 5). `tests/test_json.cpp`; the extension's `json_reformat`,
  `json_float_repr`.

- The scalarization methods (`scalar` and its presets), `mosqp`, `boostdms`
  and `polish` in C++ (task 17, stage 6; `include/mootation/scalar/`,
  `_core.scalar`), with the library's own local solvers in place of SciPy's:
  an SQP (`slsqp.hpp`: the line-search SQP of Nocedal & Wright, Algorithm
  18.3, with the damped BFGS update and the l1 merit function, its quadratic
  subproblem solved as Lawson & Hanson's LSEI/LSI/LDP/NNLS, `lsq.hpp`), a
  Nelder-Mead (`nelder_mead.hpp`: Lagarias et al. 1998 with their tie rules and
  Gao & Han's adaptive coefficients), the CMA-ES and its eigendecomposition
  (Lawson & Hanson's symmetric QR algorithm, `eigen.hpp`); the random numbers
  of `rnd::`. Their runs change: `method_version = 2` in the settings
  (`polish_version` for the polishing). SciPy is no longer needed by them; it
  remains the reference of the tests (`own_slsqp_and_nelder_mead_against_scipy`).
  The Python modules `scalar/evaluator.py`, `subproblems.py`, `solvers.py`,
  `cmaes.py`, `mosqp.py`, `boostdms.py` are removed; tests/test_slsqp.cpp and
  tests/test_scalar.cpp check the solvers on problems with known solutions and
  the methods' answers bit for bit. The knob `sqp_line_search` (`stop`, then
  the default, or `accept`): what sqp does after 10 step lengths without
  Armijo's decrease (SQP-6).

- Changed: a run's clocks in `meta.json` are `timing_us`, whole
  microseconds (the store's integer codec keeps them exactly; the owner's
  word, task 17, supplement 5, §2.6), where they were `timing`, seconds
  rounded to 0.1 ms. `tools/perf` reads both.

- Store format 3 (task 17, stage 4; supplement 5, §2): the library's own
  codecs (`include/mootation/store/`). A float64 column: a value met before
  by its index, a new one by its top k bits through an adaptive binary range
  coder and the rest raw, or, where every value is an integer over 10^e
  (clocks in milliseconds, counts), the integers; integers by frame of
  reference, differences or runs; each block the shortest of these. A record's
  point sets (archive, population, selection, second answer, trajectory)
  go together, a column of one name through all of them, so that a point
  two of them hold is written once. A record's blocks read back bit for
  bit; formats 1 and 2 are read as before; format 3 needs the extension.
  Bytes (JSON, text) by the library's own LZ77 and range coder, no zlib;
  records checked by CRC-32C (`MOOTREC2`). A pack is one file,
  `pack-<family>.mpk`: a header, the runs' meta.json as one table (column by
  column), the trajectories' table, the records, an index, each at a
  multiple of 64 bytes, and a footer with their places and the file's
  CRC-32C; packs of before are read, and `compact` replaces them. A
  record's bytes — blocks, body, frame, the key's SHA-256 — and the files —
  a segment's record appended and synced, a pack written, read and checked
  — and what a record reads from its JSON (the point tables, meta.json's
  final once, a pack's meta and metrics tables) are the library's
  (`include/mootation/store/`); `store.py` finds the files and runs compact,
  merge and export over them. The format, byte by byte:
  `docs/store-format.md`.

- `dm::sin`, `cos` and `tan` for every finite argument (task 17, supplement
  5, §4): above 2^20 pi/2 Payne and Hanek's reduction by the bits of 2/pi
  (Muller, Elementary Functions, §11.4; the table from `gen_coeffs.py`),
  where the Cody-Waite reduction was not exact before and a debug build
  stopped at an assert. Within 0.86 ulp there (the reference file's new
  sections `sin_big`, `cos_big`, `tan_big`, with Gladman et al.'s hard inputs
  and Muller's worst case for the reduction); below 2^20 pi/2 the bits are
  the same. DOC4, whose sine takes arguments up to 3e8, is version 3.

- Fixed: the memory of `result.unique_evaluations`. Every distinct point a
  run evaluated was kept as a tuple of Python floats, about 3 kB a point of
  100 variables: 3 GB for a run of 10^6 evaluations, and a campaign of
  eighteen such runs at once exhausted the machine. The points are now
  remembered by a 16-byte digest (BLAKE2b; `unique.py`): the same count —
  -0.0 as 0.0, a NaN equal to nothing, as with the tuples — at about 90 MB
  per 10^6 points. The cache of `budget_counts = "unique"` keeps the points.

- Fixed: a campaign on a problem with a categorical variable (catZDT1).
  With a core, a decision-space indicator on the trajectory failed:
  `current_variables()` gave the real genome only (ten columns for eleven
  variables); it now gives the integer genome after it, and the campaign
  reads the rows as the problem's variables, the label's index. With a
  baseline, the run failed on a number where a label was wanted: a
  baseline's point now reaches the problem and the archive as its label, the
  nearest index, as the problem itself read the number. final.csv carries
  the eleven variables.

- In C++ (task 17, stage 2b): `metrics.compute` as a whole
  (`include/mootation/run_compute.hpp`), every indicator by name in the same
  order with the same values; the Python stays as the reference
  (`metrics._compute_numpy`).
  And a run's accounting of its evaluations (`include/mootation/
  run_ledger.hpp`, `campaign.USE_LEDGER`): the counts, the failures, the
  first feasible evaluation and every row offered to the run's archives, read
  at exactly the grid's counts; the same files.

- Task 17, supplement 3, stage 3a: DAS-CMOP1-9, LIR-CMOP1-14, MW1-14,
  DOC1-9, Polygon (with rcPolygon and c0Polygon), IPolygon, SYM-PART,
  Omni-test, MMF1-15, RE and CRE in the C++ suite (`include/mootation/suite`:
  `dascmop.hpp`, `lircmop.hpp`, `mw.hpp`, `doc.hpp`, `polygon.hpp`,
  `mmo.hpp`, `re.hpp`), the NumPy code the reference; the multimodal
  problems' and the polygons' Pareto sets too (`Native.pareto_set`). New in
  `detmath.hpp`: `asin` and `atan` (Cody & Waite 1980, ch. 10 and 11; MW5-8).
  Problem version 2 wherever the code calls an elementary function; three
  NumPy references now add in an order C++ repeats (DOC2's products, DOC7's
  and Omni-test's sums, rcPolygon's turn); IPolygon's front and Pareto set
  are drawn from the library's generator: reference version 3.

- Added (task 17, supplement 2, §1.3): a problem can say it is stochastic —
  `minimize(stochastic=True)`, an optional `is_stochastic()` in C++
  (`problem.hpp`), `BenchProblem.stochastic` (the noisy copies of kinds n and
  h, RE91), which a campaign passes on. There the caches of DMS, PAES and
  DMulti-MADS are off by default, and `result.settings` says why; the new
  value `"force"` of `dms_cache`, `paes_cache` and `mads_cache` keeps them.

- Changed (task 17, supplement 2, §2): the hypervolume's knobs. Exact on the
  trajectory up to `hv_grid_exact_max_m` (8), at the end up to
  `hv_exact_max_m` (10, was 5), Monte Carlo above with its standard error
  (`hv_se`, `hv_h_se`). New `hv_archive`: the exact hypervolume of the whole
  archive written (archive_cap x N), on the trajectory up to
  `hv_archive_grid_max_m` (6) objectives, read at exactly each count of the
  grid, and in `final_archive` up to `hv_archive_final_max_m` (8).

- The verification set (task 17, supplement 2, §3; `benchmarks/classic.py`,
  `include/mootation/suite/classic.hpp`): SCH, FON, POL, KUR (Deb et al. 2002,
  Table I), SPH_2D, SPH_3D, QV and KURZ, the errata's KUR (Zitzler et al.
  2001, n = 100), ZDT6_D100 — in the C++ suite from the start, family
  "verification", apart from the atlas. Fixed: the C++ ZDT6 divided the sum
  by 9, right at its ten variables only; now by n - 1 (the same values at
  ten).

- Changed (task 17, supplement 2): `dms_cache = "on"` by default (DMS-7; the
  same points in the same order, only the budget goes further). A point a
  core takes from its cache (DMS, PAES, DMulti-MADS) is brought to the
  vault's present — the equalities' tolerance, the constraint level, the
  failures' penalty — as its other copies are; before, it kept the values of
  the step it was evaluated at. `cache_check = "on"` evaluates every point
  taken from a cache again and compares. A run's `meta.json` says
  `seed_independent` where every seed gives the same run (DMS, MultiGLODS up
  to 100 variables, BoostDMS, MOSQP at their defaults).

- In C++ (task 17, stage 2a): the decision-space indicators igdx, ps_dmin and
  pairwise_distance (`include/mootation/decision_indicators.hpp`), the same
  values bit for bit; the second answer's spread_select. Changed: its first
  point is the library's generator's (`rnd::below` of a `std::mt19937` of its
  own, from the run's seed), not NumPy's: `meta["set_answer"]["version"]` 2.
  The Monte-Carlo hypervolume's points likewise (`rnd::uniform01` of a
  `std::mt19937`; `meta["hv_mc_version"]` 2 where it may be taken).
  CR (and so PSP) in C++, its root by the library's `pow` instead of the C
  library's, which differed between platforms in the last bit, and the
  hypervolume's ref_scale^M likewise, an ulp off at some ref_scale and M
  (`meta["indicator_version"]` 2). `reference_step` (gdp_min's) by every
  pair, in C++ or NumPy, the same values: no SciPy KD-tree. In C++ too:
  roi_dist, range_cover, nd_share, dup_share, nn_cv, the hypervolume of a set
  in its frame, np.quantile and np.median (`run_metrics.hpp`), the same
  values.

- WFG1-9, MaF10-12 (WFG1, WFG2, WFG9 with M - 1 positions) and rWFG1-9 in
  the C++ suite (`include/mootation/suite/wfg.hpp`; task 17, stage 1): 20-50
  times faster an evaluation; `benchmarks/wfg.py` stays the reference. They
  are version 2 of themselves (`registry.PROBLEM_VERSION`), their noisy and
  failing copies with them: the library's own `pow`, `sin` and `cos`, and
  rWFG's rotation the pairwise sums of `matvec_pairwise` instead of BLAS's
  product (in the reference too). Removed: numba (the WFG kernels' optional
  compiler) and the `numba` field of a run's `meta.json`.

- Faster, the same values (task 17, supplement 1, stage 0): `IncrementalFront`
  (IGD+ and epsilon of everything evaluated) skips a point weakly dominated
  by one offered before, up to five objectives; the MaF reference simplex is
  enumerated without the tuples whose sum is already above 1, the same grid
  (importing the registry took 4.7 s, now about 0.5 s).

- The store's format 2: a run's point sets go into its record straight from
  memory as float64 (no CSV text on the way), its trajectory column by
  column; the run folders and `mootation store export` still write ten
  significant digits, the same text. A store of format 1 is read as before.
  New beside the final population: `final_cv.csv` (its constraint violation,
  constrained problems) and `set_answer.csv` (the second answer's points).

- `[campaign] archive_cap` (x N; 0, the default, the whole archive): the run
  archive written is its first archive_cap x N points in DSS order, which
  begin with the archive scenario's N, so no indicator changes. A noisy
  copy's archive keeps the values its algorithm saw (`f1_seen` ...), which
  the archive scenario selects by. `python/examples/atlas_storage.toml`.

- `SetArchive` (the second answer's candidates) in C++
  (`include/mootation/set_archive.hpp`): a sorted list at two objectives, an
  ND-tree from three; the same points in the same order as the NumPy one.

- `dms_cache` and `paes_cache` (`off` by default, as the papers): a point
  evaluated before is taken from the run's cache, at no evaluation (DMS-7,
  PAES-7).

- Problem properties (`benchmarks/properties.py`) for MaF1-13 and their
  rotated copies (Cheng et al. 2017, Table 1), OMM, LOTZ, OJZJ and the
  uninformative problems: every registry problem now has a row.

- The result store (`mootation.store`, `[campaign] storage = "store"`): every
  worker appends its runs to its own segment (`store/seg-*.mhd`, `.raw`,
  `.idx`) instead of four files a run; a record is written whole and synced,
  a cut one is not read and `repair` removes it; `keep = "standard"` leaves
  the archives at the intermediate grid points out, `"full"` keeps them.
  `mootation store ls|check|repair|compact|merge|export`: packs per family
  of problems, byte for byte whatever segments the runs came in; tables of
  the trajectories per family (`mootation.store.metrics`), CSV export. The
  analyses (`--compare`, `--ranks`, `--cover`, ...) read a store as they read
  the run folders. `storage = "files"` (the default) is the layout of before.

- `mootation report <results>` (`mootation.run.campaign_report`): one HTML
  page on a campaign, per family and number of objectives — the ECDF over
  levels of the IGD+ floor, the median curves along the budget grid, a table
  per level, the archive beside the population, the runs not done.

- `[campaign] run_timeout`: a run past it is stopped by the runner and kept
  as "timeout", with its trajectory up to the last record and that record in
  `meta.json`; a resumed campaign does not run it again.

- `meta.json` says what a modified problem changes of which problem
  (`modification`: rotation, control, shift, noise, failing;
  `benchmarks.properties.modification_of`) and how many distinct points the
  run evaluated (`unique_fe`; the budget counts calls).

- An `[[algorithms]]` entry may run on a part of the campaign's problems
  (`problems = [...]`), as a grid of a knob beside the main runs;
  `python/examples/population_variants.toml` shows four ways of choosing the
  population size.

- `[campaign] set_answer` is `"evaluated"` by default and the second answer
  is kept only on the multi-modal problems (the problem property
  `multimodal_ps`: SYM-PART, Omni-test, MMF, IPolygon, minusDTLZ2); its time
  is `meta["timing"]["set_answer"]`. `SetArchive.add_many` compares a batch
  at once and keeps what adding its points one by one keeps.

- The budget loop stops without an error when the core ends by its own rule
  (`finished()`: DMS, DMulti-MADS, MultiGLODS) before the budget: the run is
  "done", and a campaign writes `meta["core_stopped"]` with the evaluations
  spent. Before, it waited 10^4 steps and failed.

- DMulti-MADS (DMM-9): a failed poll shrinks the frame of the member polled,
  found by its variables and its values. Found by the variables alone, a
  noisy problem without the cache (two list members with the same x, other
  values) shrank the first copy, never the incumbent, and the run idled.
  **Changed results**: only runs with such a copy (`mads_cache = "off"` on
  a noisy problem); the default runs and the fingerprints do not change.

- An algorithm that cannot run on a problem at all says "incompatible by
  construction" (a core without the integer genome on binary or categorical
  variables; BC-SPX and BC-REX when n + 1 parents exceed `pop_size`, with
  the numbers); a campaign marks such a run `meta["incompatible"]`.

- An `[[algorithms]]` entry with a key other than name, pop, gens, label,
  evaluations, init, init_share, params and problems is refused: a knob written beside
  `name` instead of in `params` was dropped without a word.

- `mootation refs handling FILE` (`refs.set_constraint_handling`) writes a
  reading of whether each constrained problem tests constraint handling —
  "tested", "untested", "unread" — into its constraints.json as `handling`,
  where `refs.constraint_properties` returns it; kept when the rest is
  computed again. The reading is the user's (a campaign's), not shipped.

- `ada = 1` in NSGA-III, θ-DEA and RVEA: Tanabe & Ishibuchi's ADA (IEEE TEVC
  2020; Algorithms S.10-S.12 of its supplement) for multi-modal problems — one
  child a step, compared only with its subproblem's neighbours in the
  variables, a growing population, the answer its best per subproblem
  (`include/mootation/operators/ada.hpp`). Off by default.

- `[campaign] set_answer = "evaluated"` or `"population"`: a second answer
  for the decision-space indicators — N points spread out in the variables
  (Tanabe & Ishibuchi 2020, §III-H) from every point evaluated that no other
  dominates (equal objectives with other variables kept apart) or from the
  final population; its IGDX, CR and PSP in `meta["set_answer"]`
  (`"evaluated"` by default, see above).

- DMulti-MADS keeps the points it evaluated (the cache V^k of Bigeon et
  al.'s Algorithm 1) and evaluates none twice (DMM-8); `mads_cache = "off"`
  is the run of before. **Changed results**: every DMulti-MADS run that met
  a point twice (without the cache up to a fifth of 10^4 evaluations went so
  on DTLZ2_5D and MaF1_5D). The fingerprints do not change: their short
  runs meet no point twice.

- `mootation.subset_select(F, k, method, ref)`: k of n points by `dss`,
  `hv_greedy` (lazy), `hv_ls_rn` (local search with candidate lists, Korogi &
  Tanabe 2025, from the greedy choice) or `igdp_greedy` (the IGD+ floor's
  greedy choice); the extreme points always kept. C++:
  `include/mootation/subset_select.hpp`.

- `constraint_mode = "none"`: the algorithm ranks by the objectives only,
  while the constraints are still evaluated and reported (`result.cv`), so a
  campaign measures the feasible part of the answer — the control of Tanabe &
  Oyama (CEC 2017) for whether a problem tests constraint handling at all.
  `mootation.run.refs.constraint_properties(name)`: a constrained problem's
  feasible share of the box, the constraints its infeasible points violate,
  and whether the constraints move the front (C-DTLZ), kept in the
  reference cache.

- NSGA-III, A-NSGA-III, θ-DEA and SRV-NSGA-III take the number of children
  (`n_offspring`, λ) and of reference points (`n_ref_points`, H) apart from
  the population size μ, after Tanabe & Oyama (EMO 2017): λ children a
  generation, μ survivors from the μ + λ; H reference points from
  `mootation.reference_points(n_objs, h)` (C++ `das_dennis::reference_points`):
  the lattice when h is a lattice size, else their greedy farthest-point
  choice from a dense lattice. Not set, λ = μ and H = μ's lattice, as before.

- MOEA/D-DE and MOEA/D-DRA take the three components of the DE mutation
  after Tanabe & Ishibuchi (Soft Computing 2019) as knobs: `de_base`
  (`current1`, the default; `rand1`; `current_to_pbest1` with `de_p`),
  `de_index` (`wr`, the default; `wor`; `wpr`) and `bound_repair` (`random`,
  the default; `clip`, `reflect`, the new `r_reflect`, and `resample`, which
  draws the mutant again). `bound_repair = r_reflect` (randomized reflection,
  their Eq. 11) is open to every operator. The defaults are the runs of
  before, bit for bit.

- `docs/deviations.md`: every departure from its paper that a core's header
  names, gathered by `tools/deviations.py` (algorithm, item, what, whether the
  default run departs); CI checks the table against the headers.
- `docs/algorithms.md`, "How each algorithm normalizes", and
  `mootation.run.algorithms.NORMALIZATION`: the way each of the 76 cores puts
  its objectives on one scale (PP, BP, the NSGA-III hyperplane, an ideal point
  only, none, or its own), after Tanabe 2024. No code changes.

- `mootation::dm` is faster under MSVC: the helpers that return a struct
  (log's split, the reduction of sin/cos/tan, pow's double-word log and exp)
  are force-inlined, which MSVC did not do on its own — log about 4x, cos 2x,
  sin, tan and pow 1.6x faster. No value changes (the same bits as before on
  every platform).
- **SMS-EMOA computes the hypervolume contributions exactly at every number
  of objectives by default** (`hv_exact_max_m` not set; it was 5): the race
  and Monte-Carlo estimates only with a threshold given and `hv_contrib =
  race` or `mc` — measured, they choose badly from five objectives on. From
  eight objectives SMS-EMOA and MO-CMA-ES warn that a run can take hours
  (`minimize()` and a campaign before its runs). **Changed results**:
  SMS-EMOA's runs at six objectives and more (the race before); the runs at
  up to five objectives and the fingerprints (two and three objectives) do
  not change.
- `rnd::gamma` at shape < 1 draws its uniform again when it is exactly 0
  (Marsaglia & Tsang take it on (0, 1)); no value of any other draw changes.
  `random.hpp` names the sources of the polar method (Marsaglia & Bray 1964)
  and of genrand_res53 (the reference code mt19937ar.c).
- The full fronts of DTLZ6 at ten and of DTLZ5 and DTLZ6 at fifteen
  objectives are no longer listed to build (`fronts_full.SIZES`): they would
  take hours.

- DMulti-MADS: the initial list keeps only its non-dominated points, as
  Algorithm 3 assumes of every list. The filter moved each kept point out of
  the list it was still comparing against, so a point dominated only by an
  earlier one stayed (on DTLZ7 and ZDT1 every point of the box's diagonal).
  A later copy of a listed point could then enter through such a dominated
  member, and the failed iteration shrank the frame of the other copy: the
  incumbent kept its frame, polled outside the box and evaluated nothing,
  forever (DTLZ7_3D, seed 13, 10 000 evaluations: "10000 steps in a row
  evaluated nothing"). **Changed results**: `dmulti_mads` runs take other
  trajectories wherever the diagonal holds a dominated point.
  With the list right, ZDT1's, ZDT3's and DTLZ7's diagonal leave one point,
  x = 0, a vertex of the box, and from there almost every poll point left the
  box: the run met §6.3's stop after some 40 evaluations. A poll step that
  would leave the box now stops at its last mesh point inside (DMM-7), and a
  repeated poll point is dropped; trajectories change wherever a poll point
  left the box. WFG4_3D does worse than before at 10 000 evaluations (IGD+
  about 0.6 against 0.3): the dominated diagonal points the filter kept lay
  near WFG4's optimal distance values, and the paper's start (the
  non-dominated points of the line, DMS) drops them. DMM-7 is our reading:
  the DMulti-MADS implementation is preliminary until its poll and bounds
  are checked against Audet, Le Digabel, Tribes (SIAM J. Optim. 2019, the
  granular mesh; the header had cited it as Optim. Eng., corrected) and
  OrthoMADS. Checked so far: Audet, Ianni, Le Digabel, Tribes (SIAM J.
  Optim. 2014) and the NOMAD 4.6.0 user guide give the poll directions as
  implemented (MADS(suc,neg), NOMAD's ORTHO N+1 NEG) and reject points
  outside the bounds without projecting them; neither cuts a poll step to
  the box, so DMM-7 stays ours.

- DTLZ1-7 at eight objectives (`DTLZ1_8D` ... `DTLZ7_8D`; 1357 problems):
  DTLZ was registered at 2-6, 10 and 15 objectives only. Evaluated in C++ as
  the other sizes; their frames in the shipped cache, and the full fronts of
  DTLZ5 and DTLZ6 at eight objectives (`fronts_full`, built and verified as
  the other sizes).

- `hv_contrib = iwfg`, SMS-EMOA and MO-CMA-ES (task 14 of
  2026-10-20, supplement 3, Д8.2): from six objectives the least
  hypervolume contributor by IWFG (While & Bradstreet, IEEE CEC 2012; the
  adaptive slicing and the Rank order of Cox & While, IEEE CEC 2016):
  every point's contribution in slabs, best first, until the least is
  complete; the points that may be the least are then counted by WFG as for
  every point. The removed point is WFG's bit for bit. Not the default: on
  the populations exact selection makes, the contributions grow alike and
  IWFG drops few points early. `hv::contributions_iwfg`,
  `hv::least_contributor_iwfg`.

- `hv_incremental` (SMS-EMOA, MO-CMA-ES; task 14 of 2026-10-20, supplement 3,
  Д8.3): the exact contributions by WFG keep every point's count from the
  step before and take it again when the point's input to WFG is the same
  bit for bit, so the run is the same and only the time differs. 1 by
  default (with `hv_contrib = hvc`, the default: WFG for every point);
  `Result.settings` reports the share of counts reused.

- **MO-CMA-ES computes the hypervolume contributions exactly at every number
  of objectives** (task 14 of 2026-10-20, supplement 3, Д8.1). Its step-size
  adaptation counts an offspring that survives as a success, so a wrongly
  removed point inflates sigma and the population drifts from the front;
  measured on DTLZ2 at six objectives, the race gave HV 0 and sigma 1.13,
  the exact contributions HV 0.71 and sigma 0.145. `n_samples` > 0 and
  `hv_contrib = race` or `mc` are now refused with that reason (a
  `ValueError`), `hv_exact_max_m` and `race_*` reported ignored.
  **Changed results**: MO-CMA-ES's runs above five objectives (the race
  before) are exact; the exact cost grows steeply with M.

- **The trajectories of all algorithms changed once (task 14 of 2026-10-20);
  a run at a given seed is now the same on Windows, Linux and macOS.**
  Version 2 of the trajectories (`trajectory_version` in every run's
  `meta.json`; a campaign refuses to mix versions in one folder or in
  `--merge`). Runs made before are not reproduced. The parts:
  - `random.hpp`: the library's own distributions and shuffle over
    `std::mt19937` (53-bit uniforms by genrand_res53, integers by Lemire's
    method, normals by the polar method, gamma by Marsaglia & Tsang), in
    place of the standard library's, which differ between implementations;
  - every `std::sort` stable, the tied `partial_sort`s and `nth_element`s
    with one result;
  - no contraction into fused multiply-adds: `-ffp-contract=off` /
    `/fp:precise` on the CMake target (it reaches the consumer's code in the
    translation units that link it) and pragmas in `fp_contract.hpp`; a build
    with `-ffast-math`, `-Ofast` or `/fp:fast` is warned of, 32-bit x86 with
    x87 arithmetic refused;
  - `_core.fp_env()` and `fp_env` in `meta.json`: denormals flushed to zero
    in the process are recorded and warned of;
  - `tests/golden_fingerprints.txt`, checked by the ctest
    `golden_fingerprints` on every platform; a probe with NaN objectives;
    `tools/compat_check.py --golden / --write-golden`;
  - `mootation selftest` / `mootation.selftest()`: does this build give the
    library's runs bit for bit;
  - README: "Reproducibility across platforms".

- `include/mootation/detmath.hpp` (task 14 of 2026-10-20, §2): the library's
  own `exp`, `log`, `pow`, `sin`, `cos`, `tan`, `acos` and `hypot`
  (namespace `mootation::dm`), built from +, −, ×, ÷ and sqrt alone, so the
  same bits on every platform: Tang's table-driven exp and log, Cody & Waite's
  reduction for the trigonometric ones, double-word arithmetic without FMA
  (Joldes et al. 2017), Remez coefficients from `tools/detmath/`. Within 1 ulp
  for exp, log, sin, cos, within 2 for the others (measured below 0.96 on
  10⁶ points each); special values as C23 Annex F. `_core.detmath.<f>` from
  Python. The cores, operators, `random.hpp`'s distributions, the hypervolume
  race and the C++ suite of problems call them instead of the C library's.
  **Changed results**: every core whose run reaches one of these functions
  (the operators SBX and polynomial mutation among them); the problems of the
  C++ suite that call them are one version up (`registry.PROBLEM_VERSION`;
  not ZDT1, MaF1, MaF9 and the bbob-biobj pairs of f1 and f8), and agree with
  their NumPy code to a few ulps instead of bit for bit.

- SMS-EMOA above `hv_exact_max_m` finds the least hypervolume
  contributor by the race of Bringmann & Friedrich (Theor. Comput. Sci. 425,
  2012, Algorithm 1; task 14 of 2026-10-20, §11): `hv_contrib = race`, the
  new default there. Samples in every point's bounding box in rounds, points
  dropped once surely not the least, until the answer is within (1 +
  `race_eps`) of the least contribution with probability 1 − `race_delta`
  (0.01, 1e-6), at most `race_max_samples` (10^6) samples a step; a race
  stopped there takes the point of the least upper bound on its
  contribution (its box's volume where it has no sample), not the first of
  the least estimates, which at five objectives and more are mostly 0 at
  the cap (supplement 3, Д8.5). The
  equal-share Monte-Carlo of before stays as `hv_contrib = mc`. A point
  known to contribute nothing (a copy, a weakly dominated one) ends the race
  without a sample; `race_abs_tol` (1e-12 of the region prod(r - ideal))
  ends it when the bounds are that close; `Result.settings` and `meta.json`
  count the steps stopped at `race_max_samples` (`race_capped_steps`,
  `race_capped_share`) and a campaign warns when a run's share exceeds 1 %.
  **Changed results**: SMS-EMOA at more than five objectives (MO-CMA-ES:
  exact, above).

- SMS-EMOA and MO-CMA-ES compute the hypervolume contributions exactly up to
  `hv_exact_max_m` objectives (new knob, 5 by default), Monte-Carlo above
  (task 13 of 2026-10-19, §3): at M = 3 and 4 by HVC3D / HVC4D
  (`hv_contrib = hvc` is now the default), at M = 5 by WFG. `n_samples`
  given decides from M = 4 (0 exact, N > 0 Monte-Carlo with N). **Changed
  results**: runs of both at M = 3 to 5 take other trajectories; at M = 4
  and 5 the contributions are exact instead of estimated. The previous
  defaults are `n_samples = 10000` with `hv_contrib = wfg`, which give the
  earlier runs again. `Result.settings` (and a campaign's `meta.json`) now
  names the contributions' knobs and method of the run. R2-EMOA reports
  `n_samples` and `hv_exact_max_m` ignored. The headers of `sms_emoa.hpp`,
  `mo_cma_es.hpp`, `hv_contribution.hpp` and `docs/running.md` describe what
  the Monte-Carlo estimate does near the front's edges.

- `hv_algo = "auto"`, the new default (task 13 of 2026-10-19, §3a). For a
  set's hypervolume (`metrics.hypervolume`, the campaign's `hv`/`hv_h`,
  `_core.hypervolume`): box decomposition at 3 to 8 objectives, where it is
  the faster, WFG at 2 and from 9 on (`hypervolume::automatic`, `hv_box.hpp`).
  The values agree with WFG's within 1e-11 relative, not bit for bit: the
  hypervolumes in metric files may differ from earlier runs in the last bits.
  For the contributions of SMS-EMOA and MO-CMA-ES `auto` is WFG, as before:
  the runs are unchanged. `wfg` and `box` stay available.

- `minimize` with `seed_population`, binary or categorical `var_types` and
  `constraints` failed before the run (IndexError in `Layout.to_user`): the
  seed points' constraint values were computed from the real variables
  alone. They are now computed from the whole genome, the bits and labels
  too, as the run evaluates the points (reported by a user).

- `nsga2_sdr` (task 13 of 2026-10-19, §1): NSGA-II/SDR, NSGA-II with Pareto
  dominance replaced by the strengthened dominance relation of Tian, Cheng,
  Zhang, Su & Jin (IEEE TEVC 23(2), 2019): R normalised by its own minima
  and maxima, Con = Σ f̄, the niche size θ̄ the ⌊|R|/2⌋-th distinct nearest
  angle, x ≺ y ⟺ Con(x)·max(1, θ_xy/θ̄) < Con(y), the layers by Deb's fast
  non-dominated sort with every pair compared (the relation is not
  transitive). In the family "Pareto-dominance & diversity", next to
  `nsga2`; NSGA-II's operators and defaults; feasibility first under
  constraints, SDR among the feasible points. The relation and the sort are
  `sdr.hpp` (in Python `_core.sdr_layers`); `tests/test_sdr.cpp` checks the
  paper's Fig. 4 and Property 3, the relation's properties and the core,
  and `test_run_layer.py` the layers and θ̄ against a reference written from
  the paper, bit for bit. The other algorithms are unchanged.

- The direct path (the core evaluates a problem of the C++ suite itself) also
  for integer and discrete variables (task 13 of 2026-10-19, §3b):
  bbob-biobj-mixint and any suite problem given such `var_types`. The binding
  decodes the core's genome as `minimize` does (an integer to the nearest,
  ties to even, within [ceil lo, floor hi]; a discrete value to the nearest,
  ties to the lower) and hands the decoded rows to `on_rows`, the points the
  Python path's evaluator receives. The same runs as `native = "off"`, file
  for file. Binary and categorical variables keep the Python path.

- RVEA* takes RVEA's random pairs of parents (task 13 of 2026-10-19; RVEA*-6
  in `rvea_star.hpp`, `rvea_parent_order` in `rvea.hpp`): the survivors
  shuffled anew for every round while they are fewer than N. **Changed
  results** only where its population fell below N; with N survivors, which
  RVEA*-1 keeps whenever the pool allows, the runs are the same.

- A campaign's runs of ZDT1_fail10b and ZDT1_fail30b end again (task 13 of
  2026-10-19): the failed band takes all of ZDT1's Pareto set, whose sample
  is then empty, and CR took the minimum of no rows. An empty sample now
  measures no decision-space indicator (IGDX, CR, PSP, ps_dmin: None, as for a
  problem without a Pareto set), and meta.json says why (`pareto_set_empty`).
  Every other run's files are unchanged.

- RVEA: the random pairs of parents (§III-B) are drawn anew for every round
  through the survivors when the population has shrunk below N (task 13 of
  2026-10-19, §2; RVEA-3 in `rvea.hpp`); before, one shuffle was gone through
  again and again, so the same pairs came back (at 20 of 126 survivors, 10
  distinct pairs of 63). **Changed results**: runs of `rvea` whose population
  shrank (irregular fronts) are not reproduced by earlier versions; with a
  full population the runs are the same. `tests/test_rvea.cpp`.

- R2-EMOA reports `hv_algo` ignored, as it does `hv_contrib` (task 13 of
  2026-10-19, §6): its selection computes no hypervolume. The same runs.
- MaF14 and MaF15: the header of `benchmarks/maf.py` records the cross-check
  with the LSMOP paper (task 13 of 2026-10-19, §4: the logistic map of
  Eq. 9-10, the linkages of Eq. 22-23, Table II's landscapes, the bounds; the
  groups rounded by the largest remainders, D = 20 M exactly, where LSMOP
  rounds up). The values are unchanged.
- The cache of references (`mootation refs`) also keeps the failing copies'
  and C2-ConvexDTLZ2's references (task 13 of 2026-10-19, §5): approximate
  fronts, but fixed by the problem alone, not by a campaign. The same arrays
  as the campaign's folder held, so the same runs; RE, CRE, RCM and their
  noisy copies stay in the campaign's folder. A failing copy is kept by its
  kind, not by `front_exact`, which it sets only once its sample is tested.
- The campaign's queue no longer counts MaF8 among the families evaluated in
  Python (task 13 of 2026-10-19, §7): it is C++ since task 12. Only the order
  of the queue may change, not the runs.

- `dss::order` (mosa's answer, DMS, the inner archive) sets the almost
  dominated points aside with less work and the same order, bit for bit
  (task 12 of 2026-10-19, §4): every node of its k-d tree holds the smallest
  sums of sqrt(q), q and q^2 of its points, and a point asks only the nodes
  whose sums a point that sets it aside could have; there is no round after
  one that leaves the kept points' range as it was; and the tree is built
  once, fitted to the points left after a round that sets points aside.
  `tests/test_dss.cpp`: 10 000 random sets against the order before and
  `order_plain`.
- The cache of references (task 12 of 2026-10-19, §3; `mootation.run.refs`,
  the console script `mootation refs build|check`): the frame, reference
  front, dense reference, Pareto-set sample and IGD+ floors of every registry
  problem with an exact front, built once per machine into the user's cache
  (`MOOTATION_REFS_DIR` to move it), only what is missing or old (problem
  version, builder version), `--rebuild`, `--extend`. A campaign builds the
  missing entries of its problems in parallel before its first run
  (`--no-build-refs` stops with the list instead) and reads them without
  locks; the same arrays as it built for itself before, so the same runs.
- `hv_contrib` (task 12 of 2026-10-19, §5): `hvc`, SMS-EMOA's and
  MO-CMA-ES's exact contributions at three and four objectives by HVC3D and
  HVC4D (Guerreiro & Fonseca, IEEE TEVC 22(3), 2018; `hvc.hpp`). At M = 3 the
  HVC3D structure is kept between steps and updated a point at a time (one
  offspring in, one point out), then every contribution in linear time; at
  M = 4 HVC4D in O(n²), only with `n_samples = 0` (M = 4 is Monte-Carlo by
  default). The values are WFG's up to the order of the sums, not bit for
  bit, so a run may take another trajectory; `wfg` stays the default.
  `tests/test_hvc.cpp` checks them against WFG on 10 000 random sets and the
  updated structure against one built from scratch, bit for bit.
- MaF14 and MaF15 (Cheng et al. 2017, Eq. 41–50; task 12 of 2026-10-19,
  §2): LSMOP3 and the inverted LSMOP8 at 3, 5, 8, 10 and 15 objectives, with
  the paper's D = 20 M variables, N_k = 2 and budget (max{100 000, 10 000 D}
  evaluations), their fronts (the simplex; 1 minus the sphere's octant) and
  Pareto sets, and Table 1's properties. Where the paper refers to the LSMOP
  paper (the sizes of the variable groups) or prints MaF14's positions in
  [0, 10], `benchmarks/maf.py` says how it is read. In the C++ suite from the
  start, the NumPy code (`maf.lsmop_rows`) the reference, bit for bit.
- Changed: MaF8 is computed in the C++ suite (§6) and is version 2 of itself:
  its distances are the C library's `hypot` (`np.hypot`, `std::hypot`), not
  CPython's `math.hypot`, an algorithm of its own: the values changed in the
  last bit (`registry.PROBLEM_VERSION`).
- bbob-biobj and bbob-biobj-mixint in the C++ suite (task 12 of 2026-10-19,
  §1; `include/mootation/suite/bbob.hpp`): the ten bbob functions of the
  pairs (Hansen et al., INRIA RR-6829), the 55 pairs (Brockhoff et al., Evol.
  Comput. 30(2), 2022) and the integer variables of the mixed-integer suite
  (Tušar et al., GECCO 2019), from the numbers the instances draw in Python;
  their values are the NumPy code's, bit for bit, on all 110 + 110 registered
  problems. The NumPy code's products of vectors and matrices go through BLAS,
  and the C++ writes out the order of OpenBLAS's kernels that NumPy 2.5 ships
  with, so the values are the same on every machine with the extension.
  bbob-biobj takes the direct path from the core; bbob-biobj-mixint, with its
  typed variables, does not.
- An ND-tree (Jaszkiewicz & Lust, IEEE TEVC 22(5), 2018; `nd_tree.hpp`)
  behind the run archive's dominance checks from three objectives and the
  nondominated filter from four (task 11 of 2026-10-18, §2): the same answers
  as comparing a point with every stored one, so the same archive and the same
  filter, bit for bit; without the pass over every stored point.
- mosa's core cheaper with the same runs (§3): the archive's objectives kept
  by columns for the energies' and the archive's passes, its span and ideal
  point updated as members come and go, the energies' ε only where it is the
  answer.
- Campaigns across machines (§5): `--shard i/k` takes a job's part from a
  hash of its problem, algorithm and seed (no longer its index), and
  `--merge PART ...` gathers the parts' results into the config's results
  directory after checking that every run is there once and that the parts
  ran the same code against the same references; `meta.json` has
  `reference_hash` and `code_stamp`. The queue runs the longest jobs first
  (§6.2).
- `hv_algo` (§6.1): `box`, the hypervolume by box decomposition (Lacour,
  Klamroth & Fonseca, Comput. Oper. Res. 79, 2017; `hv_box.hpp`), for
  SMS-EMOA's and MO-CMA-ES's exact contributions, `metrics.hypervolume` and
  `_core.hypervolume`; `wfg` stays the default.
- The C++ suite of benchmark problems (task 11 of 2026-10-18, §1): ZDT1–4 and
  6, DTLZ1–7 and its variants (wide, shift, I, minus, S), C-DTLZ, MaF1–7, MaF9
  and MaF13, ZCAT1–20, shiftZDT and the rotations over them are computed in
  C++ (`include/mootation/suite`, compiled without contraction into fused
  multiply-adds; `_core.Native`) when the extension is built. Their
  `evaluate`, `evaluate_batch` and constraints call it; the NumPy code stays
  as the reference (`BenchProblem.reference`), equal bit for bit wherever
  NumPy's `sin`, `cos`, `exp` and `pow` are the C library's, and runs without
  the extension. `minimize(native="auto")`, the default, has the core
  evaluate such a problem itself, without calling Python, where nothing of
  the caller's stands between (`minimize.direct_path`); `on_rows` observes
  those evaluations, `Result.native_seconds` times them; `native="off"` calls
  the problem as any function. Campaigns: `[campaign] native` (`"auto"`, the
  default, or `"off"`), the same files either way, `timing.native` in
  `meta.json`. The binding: `Problem.native` and `Problem.on_rows`.
- Changed: the rotated problems of these families whose matrix is not
  diagonal (rZDT, rDTLZ, rcDTLZ, rMaF, rcMaF, rZCAT) take A·d as NumPy's
  pairwise sums (`_rows.matvec_pairwise`), not BLAS's product, and MaF9's
  constraint from five objectives writes its dot products out: their values
  changed in the last bits, and they are version 2 of themselves
  (`registry.PROBLEM_VERSION`, `problem_version` in a run's `meta.json`;
  `campaign --list` marks the runs of version 1). The controls r0 and c0 are
  unchanged.

- Batches through the whole chain (task 10 of 2026-10-18). Every built-in
  problem has `evaluate_batch(X)` and `constraints_batch(X)` (n rows to n × M
  and n × K), and a constrained problem whose objectives and constraints come
  out of one computation `evaluate_with_constraints(X)` → `(F, G)`. ZDT, DTLZ
  and its variants, C-DTLZ, MaF (but MaF8 and MaF10–12), ZCAT, bbob-biobj and
  its mixint, DAS-CMOP, LIR-CMOP, MW, BT and MOP are written for batches: one
  implementation, `evaluate(x)` its batch of one row; the rotated, shifted and
  noisy copies take their base's batch; everything else loops over its point
  function. A row's values are those the point code gave, bit for bit
  (`benchmarks/_rows.py` names what has to stay one point at a time for that:
  BLAS's product per point, the C library's `pow`, and its `sin`, `cos` and
  `exp` of one float wherever NumPy's of an array differ on the machine), and
  a row that fails is NaN alone. `minimize` takes `batch_with_constraints`
  (the batch returns objectives and constraints of the rows, one call for
  both) and logs a batch (`log_evaluations` with `batch=`, refused before);
  `benchmarks.solve` and the campaign hand the batches on.
  `tools/compat_problems.py` compares every problem of the registry, point and
  batch, with a commit of before.
- The run layer's own work in C++ (`_core`): the nondominated filter, the run
  archive (`GridArchive`, with `add_many`), DSS and the greedy IGD+ selection,
  and the IGD, IGD+, GD+, ε+ and R2 indicators, each the Python version's
  values (the Python versions stay, the reference in the tests and the
  fallback without the extension). `_core.IncrementalFront` keeps IGD+ and ε+
  of every point a run evaluated up to date a batch at a time.
- Campaigns: `record_grid = "budgets"`, now the default for a new results
  directory (one started under another grid keeps it), records at 100, 150,
  200, 300, 500, 700, 1 000, … and 25 000, at exactly b evaluations (`fe_pop`
  where the population was complete before b); `all_evaluated` (on) gives
  every record and `final` `igdp_norm_all` and `eps_norm_all`, and
  `meta.json` the evaluation at which each of `all_levels` was first reached;
  `archive_mode = "posthoc"` fills the archive only when it is read, the same
  archive; the reference data of a problem (frame, front, dense reference,
  Pareto-set sample) are built once a campaign (by one of the workers that
  start on it together, the others waiting for its file) and kept in
  `<results>/_references/`, stamped with the reference version and a hash of
  the benchmark sources; `meta.json` has `timing`, the seconds of a run by
  component. `tools/perf/profile.py` and `tools/perf/batches.py` read where a
  campaign's time goes and which cores evaluate a generation at a time
  (`docs/algorithms.md`, "Batches and single points").
- `n_samples` reaches SMS-EMOA, MO-CMA-ES and HypE: the Monte-Carlo sample
  points of a step from four objectives (10 000; 0, exact contributions, for
  SMS-EMOA and MO-CMA-ES).
- Cheaper with the same results: the exact hypervolume contributions from
  three objectives (one WFG workspace for all the points) and the Monte-Carlo
  ones (each sample tested against the points that can cover it), SPEA2's and
  SPEA2+SDE's truncation (the distance rows computed once and kept sorted),
  and MaF9's constraint (the forbidden polygons tested for a whole batch).
- Changed: RCM02 (where EI / μ < 0), RCM24 (outside the domain of its
  logarithms) and RE22, RE22c (at x2 = 0) return NaN — a failed evaluation —
  where they raised; a campaign now counts those points in `n_failed`.
  `benchmarks.properties` has `undefined` and `undefined_share`.

- SMPSO (Nebro, Durillo, García-Nieto, Coello Coello, Luna & Alba, IEEE MCDM
  2009), `smpso`: a swarm with a bounded leaders archive by crowding distance,
  the constriction factor and the velocity bound of Eqs. 3-6, polynomial
  turbulence; Algorithm 1 in jMetal's order. What the paper leaves open is a
  preset, `smpso_preset`: `jmetal` (default), `platemo`, `fair` and `omopso`
  (OMOPSO, Reyes Sierra & Coello Coello, EMO 2005: no constriction, uniform
  and non-uniform mutation by thirds, only the particles that replaced their
  best offered as leaders and the leaders cut once a step, an
  epsilon-dominance archive as the answer), and each of its parts a knob
  (`chi_mode`, `vmax_frac`, `vel_on_bound_low`/`_up`, `w_min`/`w_max`,
  `c_min`/`c_max`, `r_per_dim`, `turbulence`, `turbulence_k`,
  `turbulence_p`, `archive_size`, `archive_prune`, `leader_select`,
  `leader_offer`, `eps_archive_eta`, with `eta_m`, `pm` and `bound_repair`).
  Real variables with finite bounds only.
- R2-EMOA (Trautmann, Wagner & Brockhoff, LION 2013; Brockhoff et al.,
  Evol. Comput. 2015), `r2_emoa`: SMS-EMOA's steady state with the worst
  front's least R2 contribution removed; the exact R2 at two objectives
  (Schäpermeier et al. 2024) or the paper's discrete weights (`r2_mode`,
  `r2_space`, `r2_weights`, `r2_offset`, `normalize`). SMS-EMOA gains a
  protected hook for it and is unchanged.
- BC (Tanabe & Ishibuchi, GECCO 2019), `bc_spx` and `bc_rex`: the non-elitist
  best-children selection with SPX or REX, the paper's BF and BA as
  `bc_selection`, `bc_lambda`.
- PAES (Knowles & Corne, Evol. Comput. 2000), `paes`: the (1 + 1) Pareto
  archived evolution strategy with its adaptive grid (`paes_depth`,
  `archive_size`).
- RM-MEDA (Zhang, Zhou & Jin, IEEE TEVC 2008), `rm_meda`: an estimation of
  distribution algorithm — an (m − 1)-dimensional local principal component
  analysis splits the population into clusters, each modelled by a box along
  its principal components plus Gaussian noise, the offspring sampled from
  the models and selected by non-dominated sorting with a crowding distance
  recomputed after every removal (`n_clusters`, `rm_extension`,
  `bound_repair`: clip by default, the text's random reset as `native`). A
  twelfth family, estimation of distribution. Real variables only.
- DMulti-MADS (Bigeon, Le Digabel & Salomon, Comput. Optim. Appl. 2021),
  `dmulti_mads`: a list of points with a frame size each, the incumbent of the
  largest frame chosen by spread, a speculative search, n + 1 poll directions
  on a mesh, the list updated by dominance and extent (`mads_tau`,
  `mads_delta0`, `mads_select`, `mads_poll`, `mads_search`).
- MultiGLODS (Custódio & Madeira, J. Glob. Optim. 2018), `multiglods`: DMS's
  list with comparison radii and active flags, searches started from a Sobol
  sequence and merged (`glods_search_every`, `glods_local_tol`).
  `include/mootation/sobol.hpp`: the Sobol sequence of Joe & Kuo's direction
  numbers, 100 dimensions.
- BoostDMS (Brás & Custódio, Comput. Optim. Appl. 2020), `boostdms`
  (Python): DMS with a search step on quadratic models, minimized by the
  adapted ε-constraint of Custódio, Pozzi & Silva (2026) or the paper's
  min-max levels (`boost_scalarization`).
- MOSQP (Fliege & Vaz, SIAM J. Optim. 2016), `mosqp` (Python): a set of
  points spread by single-objective SQP steps and refined by the
  all-objectives step, Armijo's rule on ℓ1 merit functions, a restoration for
  infeasible subproblems (`mosqp_tau`, `mosqp_mu`, `mosqp_beta`,
  `mosqp_spread`).
- Scalarization with a local solver (`python/mootation/scalar/`): the
  algorithm `scalar` and its presets `eps_sqp`, `tch_sqp`, `nbi_sqp`,
  `nnc_sqp`, `eps_nm`, `eps_cmaes`, run by name through `minimize()` and in
  campaigns. Anchors as lexicographic minima, then a plan of subproblems —
  ε-constraint, smooth Chebyshev, NBI (Das & Dennis 1998), NNC (Messac et al.
  2003), Pascoletti–Serafini — on a lattice, by the adaptive ε-constraint
  scheme of Laumanns, Thiele & Zitzler, or into the largest gap; each from the
  archive's best point for it, solved by SciPy's SLSQP with finite-difference
  gradients, SciPy's adaptive Nelder–Mead, or a CMA-ES written from Hansen's
  tutorial. Every evaluation, finite differences included, counted and
  archived; equalities to SLSQP as such; `result.solutions` lists the
  subproblems' solutions. Needs SciPy: the extra `scalar`.
- `polish` (`off`, `sqp`, `nm`) for every algorithm: `polish_share` of the
  budget kept for the end, `polish_points` points of the archive (DSS) each
  improved in one objective with the others no worse; the answer the
  non-dominated union of the algorithm's and the polished points, cut to
  pop_size without losing cover of the algorithm's answer. Off by default,
  and off changes nothing.
- `constraint_mode = "eps_level"` with `eps_schedule = {"tc": 0}`: the level is
  0 from the first step, as the rule says, also for a core whose setup spends
  fewer than `pop_size` evaluations (PAES starts from one point); it used to
  stay at eps(0) until the first `pop_size` evaluations were spent.
- `Result.settings`: every knob as the run used it, for a core whose knobs
  come from a preset (SMPSO); a campaign writes it to `meta.json` under
  `settings`. `run.budget.depends_on_budget(name, params)`: the budget
  dependence of a run by its settings as well (SMPSO's `turbulence = thirds`),
  which the campaign's `budget_dependent` and ladder now use.
- A point's constraint values were lost when a core rewrote its population.
  ETEA, MOEA/D-AM2M, MOEA/D-DS, RD-EMO, APRD, Liu & Gu 2011, ISDE+RD, DHEA,
  HLMEA, IF-MaOEA, DEA-GNG, MaOEA-3C, DCEA, HCCA, CLIA, Two_Arch2, NAEMO and
  DMS wrote it back with `seed_individual(v, vars, objs, {}, {})`, which took
  every constraint as 0: a violating point passed for a feasible one — for the
  core's own selection, the feasibility rules, the epsilon level, the archive,
  the answer and every feasibility measure. Every write-back also dropped a
  failed evaluation's mark (its penalty is finite, so it counted as a good
  point and widened the penalty's range), and |h| of the equalities. Now:
  - the vault keeps a point whole as a `VaultPoint` (variables, integer
    genome, objectives, constraint values, |h|, the failure): `point_of`,
    `archive_point_of`, `put_point`, `copy_point` (slot to slot, the MOEA/D
    family's move), `archive_push_point`; a point written back is brought to
    the vault's present — the equalities from |h| at the current tolerance, a
    failure to the current penalty, cv at the current epsilon level
    (`data_vault.hpp`, WRITING BACK);
  - `seed_individual` and `archive_push_data` refuse a point without its
    constraint values on a constrained problem, where they wrote zeros;
  - every core that writes back points it kept does so whole (the eighteen,
    and ADAW, MOEA/D, MOEA/D-DE, MOEA/D-DRA, MOEA/D-AWA, MOEA/D-M2M, SMS-M2M,
    NRV-MOEA, SPEA2, SPEA2+SDE, LIS-LCS, MOSA), and the copies a core compares
    again in a later step are brought to the vault's present first
    (`refresh_copies`; MOSA when `present()` moved);
  - the inner archive takes no failed point, and under constraint handling
    judges its members again when the tolerance or the level moves.
  Results on problems without constraints and failures are unchanged bit for
  bit; on constrained problems the eighteen cores change, on problems with
  failures the cores that write back.
- `mootation._core.current_points()`: the population and the vault's archive
  as the vault holds them (variables, objectives, constraint values,
  violation, failed), from an `on_generation` observer. A run-layer test
  checks with it every core after every step on C1-DTLZ1_3D, DOC3 (also under
  the linear tolerance schedule) and DTLZ2_3D_fail30r.

- The inner archive's cut after every addition (`inner_archive_cap`) is
  cheaper: `dss::order` holds the points in a k-d tree and computes only the
  pairs that can change the order (a point against those that could set it
  aside; a greedy step against the candidates whose cover it can lower), and
  returns, bit for bit, the order of `dss::order_plain`, the selection written
  out as before (tested on long archive streams and on arbitrary sets,
  `tests/test_survival.cpp`). DMS's answer and MOSA's archive use the same
  function. Every result is unchanged.
- The campaign's own reference (`--pooled-reference`, `campaign.pooled_reference`):
  per problem, the non-dominated points of all the campaign's finished runs
  (final answers, feasible rows only, and run archives) in
  `pooled_reference/<problem>.csv`, with a `.json` of the runs that went into
  it, the code that built it and its sha256, the pool's version; every run
  measured against it from its saved answers as `<indicator>_pooled`, in the
  pool's frame, the old values untouched, and measured again only when the
  pool changes. The dense reference's measurement shares the code
  (`_measure_saved`).
- Properties `front_ref` (`exact` / `approximate`, from
  `BenchProblem.front_exact`) and `front_built` (how an approximate reference
  was built). For a metric that reads the reference front or its frame, the
  rank table puts the synthetic problems with an approximate front in a group
  of their own, `approx_front`, where a campaign has both kinds. RCM28's front,
  two exact curves, is now `front_exact`.
- Reference values of other implementations live apart from the tests, in
  `tests/data/<source>_<subject>.json` (library, version, license, the
  command that made the file, date, points, tolerance), made by hand with
  `tools/refs/gen_pymoo_refs.py` and `gen_reproblems_refs.py`; CI never runs
  them and no test imports pymoo. Checks that had no stored reference now have
  one: DAS-CMOP1-9 at two triplets (36 points), Omni-test and SYM-PART simple
  (five each), WFG8's and WFG9's optimal distance variables (four vectors).
  Module headers and docs name, for every such check, the library and
  version, the number of points and the file.
- `rwmop.py`'s header lists where the competition's MATLAB code and the paper
  differ (RCM10-17, 19, 20, read at 200 points per problem; RCM30-50's data).
  Nothing of the code is used.
- The stochastic population update (NSGA-II, SMS-EMOA) checked against Bian
  et al.'s Algorithms 5 and 6: the subset sizes are ⌊|Q|/2⌋ and ⌊3|Q|/4⌋ drawn
  without replacement, as implemented.
- Run results and comparisons between algorithms moved out of the headers,
  `docs/` and this file; what remains describes the choices and their knobs.

- RWMOP's chemical and process problems RCM22, 23, 25, 26, 28, 29 (task 7,
  §8; `benchmarks/rwmop.py`): equalities counted as DOC's (|h| - 1e-4, n_eq),
  0/1 variables declared. RCM28's exp(0.5 x5) and exp(0.4 x6) read with the
  minus sign (as printed a reactor's output is <= 0 and h4 never holds);
  RCM29's "= 5.5 <= 0" as "- 5.5 <= 0". RCM24 and RCM27 are defined but not
  registered: their fronts are single points (derived in the header). The
  references: an epsilon-constraint method (SLSQP, every 0/1 setting, two
  seeds of starts, agreeing to 5e-4 in normalised IGD+) joined with the long
  runs' feasible points, RCM28's exact, RCM29's the best of its sixteen 0/1
  settings. RCM30-50 are not implementable from the paper (the inverters'
  frequencies and the networks' data are not given). 1340 problems.

- RWMOP's mechanical design problems (task 7, §8; `benchmarks/rwmop.py`):
  RCM01-18 and RCM21 of Kumar et al. (Swarm and Evolutionary Computation 67,
  2021), the formulas as the paper prints them, at the competition's budget.
  The header lists what cannot be read as printed and how it is read — RCM04's
  G and tau'', RCM14's rho (used, never given), RCM06's g11 and RCM08's g8
  (as printed nothing is feasible; the paper's own tables have every run
  feasible), RCM15's wire list out of order — and where the paper's problem is
  not the classic one (kept). RCM03-08 agree with the CRE port to 1e-12 (RCM08
  in the paper's order of constraints). RCM19 (no demands given) and RCM20 (no
  feasible point as printed) are not registered. Reference fronts from ten
  long runs per problem (`benchmarks/_rwmop/`, `front_exact = False`).
  `family_kind` "engineering" and `difficulty` "low" (the paper's level) in the
  property table. 1334 problems in 49 families.

- Multimodal problems (supplement 3, §5): SYM-PART simple and rotated,
  Omni-test and MMF1-15 with MMF1_z, MMF1_e, MMF14_a and MMF15_a, the 22
  problems of the CEC 2019 competition (`benchmarks/mmo.py`, from Liang, Qu,
  Gong and Yue's report), each with its front and every part of its Pareto
  set, the sets derived from the formulas and checked on a fine grid (the
  module's header lists where the report's figures and the formulas part);
  `multimodal_ps` and `ps_parts` in the property table. 1315 problems in 48
  families.
- `psp` (`run/metrics.py`): Yue, Qu and Liang's Pareto sets proximity, CR
  over IGDX (IEEE TEVC 22(5), 2018, Eq. 8). A campaign records `igdx`, `cr`
  and `psp` along the trajectory and in `final` wherever the problem has a
  Pareto-set sample (`[campaign] set_metrics`, on by default);
  `--cec2019-ranks` gives the competition's mean ranks by IGD, IGDX and 1/PSP.
- DN-NSGA-II (`algorithms/dn_nsga2.hpp`, `dn_nsga2`; supplement 3, §5.1):
  Liang, Yue and Qu's NSGA-II with a niched mating pool — a random member
  against the nearest of `dn_cf` others in the variables, the better of the
  two a parent — and crowding in the variables (CEC 2016). The crowding factor
  is left open by the paper: `dn_cf`, 20 by default. `nsga2` is unchanged, bit
  for bit. 67 algorithms; DN-NSGA-II refuses the integer genome (18 do).

- `i_cmop` (supplement 3, §4.4; `run/metrics.py`): Vodopija, Tušar and
  Filipič's indicator for a constrained problem (IEEE TEVC 29(1), 2025, Eq.
  12), one number for runs with and without a feasible point: at most 1 with
  one (the negative normalised hypervolume, or the distance to the region of
  interest), 1 plus the least violation without. The other feasibility
  measures of the supplement were there: `feas_share`, `fe_first_feas`, the
  runs without a feasible answer (`empty` in the tables), the archive's
  indicators on its feasible points at the hard budgets.
- CCMO (`algorithms/ccmo.hpp`; supplement 3, §4.2): Tian, Zhang, Xiao, Zhang
  and Jin's coevolution of the problem and its unconstrained helper (IEEE TEVC
  25(1), 2021, Algorithm 1), in the paper's NSGA-II form with SPEA2's
  truncation: N/2 children from each population, both cut from themselves and
  all N children, N evaluations a generation, 2N at the start. Not a wrapper
  over any core, as asked: none hands its children over before its selection.
  66 algorithms.
- C-TAEA (`algorithms/ctaea.hpp`; supplement 3, §4.3): Li, Chen, Fu and Yao's
  two-archive algorithm for constrained problems (IEEE TEVC 23(2), 2019),
  Algorithms 1-5, with the authors' code (as pymoo carries it) where it
  departs from the text: objectives taken from the ideal point without Eq. 8's
  division, one child per pair. 65 algorithms.
- The epsilon level (supplement 3, §4.1; `data_vault.hpp`, EPSILON LEVEL):
  `minimize(constraint_mode="eps_level")`. The vault counts a violation up to
  ε as none for the algorithm, so every feasibility-first core does the
  ε-level comparison; ε follows MOEA/D-IEpsilon's rule (Fan et al. 2019,
  Eq. 8) once per generation and is 0 from 80 % of the budget on
  (`eps_schedule`). With ε held at 0 every core's run is the feasibility one
  bit for bit; without the mode nothing changes. `_core.current_cv_level()`.
- bbob-biobj-mixint (`benchmarks/bbob_biobj_mixint.py`; supplement 3, §1.4):
  the 55 bbob-biobj pairs at n = 5 and 10 with four fifths of the variables
  integer (arities 2, 4, 8, 16), discretized as Tušar et al. (GECCO 2019,
  §3.1.1) do, each objective about its own optimum; and catZDT1
  (`benchmarks/categorical.py`), ZDT1 with a categorical choice of g. 1293
  problems in 45 families.
- Categorical variables (task 7, supplement 3, §1; `var_types.py`,
  `operators/bit_flip.hpp`): `("cat", labels)` or `("cat", k)`; `fn` gets the
  label. Bits and categorical variables form the integer genome
  (`ops::gene_levels`, `RunConfig.gene_levels`): uniform crossover, and a
  mutation that gives a gene one of its other values, each alike, with
  probability 1/(number of genes), at most 1/2 (pymoo's cap; this changes a
  run with a single bit, which used to flip in every child). The 47 cores
  that take the genome draw their first genes through `ops::random_gene`; with
  two or more bits a binary run is unchanged bit for bit. The 17 others refuse
  it, and `minimize` says so. `result.unique_evaluations` counts the distinct
  decoded points (`mootation/unique.py`); `budget_counts = "unique"` answers a
  repeat from memory and counts the budget in distinct points. The run
  journal's hash keys the decoded point, labels included.
- DOC1-9 (`benchmarks/doc.py`; task 7, §8): Liu and Wang's problems with
  constraints in both the decision and the objective space (IEEE TEVC 23(5),
  2019, supplementary file S-I), on CEC 2006's g04, g19, g23, g09, g21, g07,
  g14, g10 and g18; the equalities of DOC3, 5 and 7 as |h| - 1e-4 in
  `constraints` and as h in the new `BenchProblem.equalities` (`n_eq`), which
  `benchmarks.solve` hands to `eq_constraints`; the paper's fronts. 1182
  problems in 43 families.
- Equality constraints (task 7, supplement 3, §3; `data_vault.hpp`,
  EQUALITY CONSTRAINTS): `minimize(eq_constraints=..., n_eq=...)`; an
  equality holds where |h| <= `eq_tol` (1e-4) and adds max(|h| - eq_tol, 0)
  to cv, Liu and Wang's rule (IEEE TEVC 2019, Eq. 3). `eq_tol_schedule =
  "linear"` lowers the tolerance from `eq_tol0` to `eq_tol` over the
  evaluation budget, every point's violation rewritten from |h| before each
  step; `result.eq_tol`, `_core.current_eq_tol()`. The vault gains
  `set_equalities` and `set_eq_tol`, and `seed_individual` a flag `raw` for
  the caller's own values (`setup_with_seed` passes it). Every run without
  equalities is unchanged bit for bit.
- LIR-CMOP1-14 (`benchmarks/lir_cmop.py`; task 7, §8): Fan et al.'s
  constrained problems with large infeasible regions (Soft Computing 23(23),
  2019; the formulas of the preprint's Table 6), fronts built on first use
  from the problems' geometry and the frame read off them. 1173 problems in
  42 families.
- Failed evaluations (task 7, supplement 3, §2; `mootation/failures.py`,
  `data_vault.hpp`, FAILED EVALUATIONS). `minimize(on_failure="infeasible")`,
  the default: an exception of `failure_exceptions` (Exception), a non-finite
  objective or constraint, or a `None` row from `batch` fails the point; the
  core gives it a penalty above the worst finite value of every objective and
  a violation beyond the worst seen, the same for every failure and raised
  when the range moves, keeps it out of the answer and the observers' rows,
  and counts it (`result.n_failed`, in the budget). `on_failure="raise"`
  stops at the first. `max_failure_share`, `failure_window` stop a run that
  fails too often (`FailureShareExceeded`). The evaluation log gains the
  columns `failed` and `error` (format v2). A campaign records `n_failed` in
  `final` and `fail_share` in the trajectory, and keeps failed points out of
  the run archive and the oracle. Every default run without a failure is
  unchanged bit for bit.
- Failing copies (`benchmarks/failing.py`): ZDT1, DTLZ2_3D and WFG4_3D
  returning NaN in a region of 10 or 30 % — random balls (`_fail10r`), the
  band along the faces (`b`) or part of the Pareto set (`n`) —, 18 problems
  in the family `failing`; 1159 problems in 41 families.

- The maximin Latin hypercube (`init = "lhs"`, `lhs_search`) raises the
  distances to the power −p by multiplications and compares designs by the sum
  of the terms: numpy's power runs SIMD code on some CPUs (SVML on AVX-512)
  whose last bits differ, and the annealing turned that into another design
  from the same seed on another machine. A seed now gives the same design
  everywhere — a different one from before, once.
- Checked against the papers task 7 was written without (sources in the
  headers). `de_adapt = "shade"` with `crossover = "de_pbest"` draws p for
  every child from [2/N, 0.2] (SHADE's Eq. 20) instead of using `de_p`; the
  rest is SHADE's Algorithm 1 as it was. `mosa`'s header names DBMOSA for the
  unconstrained archive, the Laplace move, T_0 and the restarts from the
  archive, and says that AMOSA's bounded archive and informed reseeding are
  not used; with `sa_energy = "tch"` a chain's weight now rises on an
  objective where it is better than its neighbour or equal to it (it used to
  fall on a tie) and the weights are drawn at random when it has no
  non-dominated neighbour (they used to stay), as in PSA's algorithm.
  NSGA-III's `survival = "stochastic"` draws ⌈`spu_frac`·2N⌉ members, Opris's
  ⌈3µ/2⌉ (IJCAI 2025, Algorithm 1), where it took ⌊·⌋ as NSGA-II after Bian
  et al. does; the two differ at an odd N.
- DAS-CMOP at ζ = 1 (`_t13` … `_t16`): the feasibility constraint is the
  equality g = d within the paper's 10⁻⁶ (its Eq. 13); it was the product
  (e − g)(g − d), satisfied at g = d alone. The objectives and the other
  constraints, and WFG8's and WFG9's optimal distance variables, about which
  rWFG8 and rWFG9 turn: values checked against pymoo 0.6.2 at 36 points and
  four vectors of the optimum (`tests/data/`). Omni-test and SYM-PART simple:
  at five points each.
- RE22c's list of x1 is the source's (Amir & Hasegawa 1989, Table 2, read on
  the printed page): 77 areas, with 7.00 for seven #9 bars, whose area the
  table leaves blank and the original list leaves out, besides 3.10 for
  "3,10". Its reference front does not change. RE33c's header records Ray and
  Liew's box for x4, [2, 20], and their constraint x4 <= 11, which the suite's
  supplement read as x4 >= 11.

### Removed

- `srv_moead` (SRV on a steady-state MOEA/D). It was an extrapolation rather
  than a transcription — liu2022 puts the SRV strategy on an NSGA-III carrier
  (§III-C) and says nothing about a steady-state one, so no part of it could be
  checked against a paper. It is maintained outside the public tree.
  The SRV strategy itself is unaffected: `srv`, `srv_nsga3` and the shared
  `srv_strategy.hpp` stay. The `SRVMOEAD_Individual` alias left `individuals.hpp`
  with it. The public list is 58 algorithms.
- `ibea_eplus_clean` (IBEA-ε+ with the mIBEA front filter). It was never a
  transcription of a paper but a composition kept as the control arm of an
  unpublished ablation study; it is now maintained outside the public tree.
  The public list is 58 algorithms. The paper-backed IBEA variants
  (`ibea_eplus`, `ibea_hd`, `mibea`, `r2ibea`) are unchanged.

### Changed

- `archive_checkpoints` takes the archive scenario after exactly b evaluations
  of each `record_at` count, inside a generation if need be (task 7, §5.5),
  not at the first record at or past b: at 100 or 300 evaluations a generation
  of 91 is much of the budget. `record_at = [100, 300, 1000, ...]` now reads
  the archive at small budgets as it is.
- docs/running.md: GDE3 at `CR = 1` keeps the paper's F = 0.2, a small step
  along every variable at once; give it a larger F (task 7, §5.6).
- `mutation_scale` at 0 or below is an error (task 7, §5.2); it used to
  fall back to the default silently. "No mutation" is `mutation = "none"`.
- MSVC: the interface target adds `/bigobj` — `embed.hpp` instantiates every
  algorithm in one translation unit, which now overruns the section limit
  even in a Release build.
- `--cover` reads the coverage criterion against each problem's IGD+ floor:
  `igdp_norm ≤ (1 + τ) × floor`, τ = 1, 0.5, 0.25, 0.1 by default
  (`--cover-floor-taus`). N points on the front itself leave IGD+ above zero —
  the floor, the IGD+ of the N points DSS selects from the reference front
  itself, is about 0.002 at two objectives, 0.02 at three and 0.06 at five —
  so the absolute levels 0.01 and 0.003 could not be reached at three
  objectives, nor anything but 0.1 at five, whatever the algorithm. The
  progress criterion (`gdp_norm`, floor 0) and the hypervolume gap on problems
  without a front keep the absolute levels of `--cover-taus`; level k pairs the
  k-th values of both lists. The floors go to `igdp_floor.csv` and are read
  back by the next analysis; the CSV files gain a `tau_hv` column. Problems
  whose reference front is doubtful — DTLZ5, DTLZ6 and MaF6 from four
  objectives, WFG3, DTLZ1 at five objectives — are left out of the coverage
  (`cover_excluded.csv`, `--cover-keep-caveats` keeps them). `cover_meta.json`
  records the commit of the analysis code and the scheme of the levels, every
  CSV row its level in words, `cover_summary.csv` the commit too.
  `floor_review.csv` sets each floor beside the best `igdp_norm` of any run,
  the best algorithm's median and the value that decides coverage.
- `--cover`, after the review of the floor (2026-09-29): the floor is the
  greedy one — N reference points added one at a time for the smallest IGD+,
  then single swaps — since DSS picks for coverage, not IGD+. N is the
  population the algorithm ran with, so
  MOEA/D-M2M's and SMS-M2M's multiples of K have their own floors. The DSS
  floor stays beside it with their ratio, DSS's tax (`igdp_floor.csv`:
  `floor_greedy`, `floor_dss`, `dss_tax`, the reference version and its row
  counts), and the archive scenario, whose answer is itself a DSS selection,
  reads its levels from the DSS floor. 1.1 × floor is named the ideal level.
  Problems measured against IPolygon's and the aliased WFG1/WFG2 references of
  version 1, or against an older reference where it changed since, are left
  out; problems marked for their reference (other than n_ref rows, repeated
  rows, the Das-Dennis lattice from four objectives, a front of lower dimension
  than M − 1; `cover_marks.csv`) stay in, and every table is written twice, over
  all problems and over the unmarked ones (a `subset` column). `floor_review.csv`
  gains both floors, the level's floor and η = ratio^(−d). An analysis over no
  problem at all no longer fails.
- Reference fronts, version 2 (`registry.REFERENCE_VERSION`): every
  `pareto_front(n)` returns distinct rows, exactly n of them where its sampler
  can give as many — a shortfall drawn again from a larger sample, a surplus
  thinned at random with every objective's extremes kept, so the frame stays.
  Version 1 had rows repeated up to 216 times and 259 to 6 399 rows where
  1 000 were asked for. WFG1 and WFG2 sample their positions at
  random with the cube's corners instead of a grid whose x1 levels fell on the
  zeros of the last shape at five objectives (both fronts came out as the same
  linear slices). 82 of the 324 references changed (`REFERENCE_CHANGED`); IGD+,
  GD+ and the other reference-dependent values on them are not comparable with
  earlier ones until recomputed.
- IPolygon is a new instance (reference version 2): four polygons of radius 8,
  centred 50 apart in the four quarters of the box, as in the paper's figures;
  the two of radius 20, 40 apart, were too close for the paper's equivalent
  Pareto regions (Ishibuchi, Akedo & Nojima 2011, Section 2: "not too close").
  Checked on a dense grid: the non-dominated points are exactly the polygons'
  (and grid points within a step outside, which their projection dominates).
  Its frame is exact (0 at a vertex, the longest chord). Runs of the old
  instance answered another problem: run them again (`PROBLEM_CHANGED`).
- DSS (`archive.dss_order` and `dss.hpp`) sets almost-dominated points aside:
  a point some other point beats by more than 0.1 in an objective while it
  beats that point by at most 0.001 everywhere, in normalised coordinates — the
  best value of one objective bought with a much worse one of another. The
  per-objective seeds picked exactly these, and the max-min steps picked them
  next; now they come after every other point. Without the problem's frame the
  ideal and nadir are taken from the points kept, so an outlier no longer
  stretches the frame (the run archive's reduction and `--recompute --scenario
  archive` normalise so when the archive had no problem frame). A set with
  nothing to set aside is ordered exactly as before. What changes where such
  points exist: the archive scenario, the answers of `random_search`,
  `sobol_search` and `gsemo`, and DMS's answer — on DTLZ2_3D its list had
  some (tools/compat_check.py: DMS's DTLZ2_3D fingerprint moved, ZDT1's did
  not; DMS is younger than the compatibility baseline). The docs and both
  headers now say that DSS has no approximation guarantee. In the problem's
  frame a point is set aside or not whatever the others are, so
  `dss_order` asks only the points it is about to take — the same order,
  0.1 s instead of 12.6 s on a 12 758-point archive at five objectives.
- The tables and the TUI leave a budget ladder's rungs out, and `--at` reads a
  budget-dependent algorithm from its rung of that budget when there is one.
  `postprocess.BUDGET_SCHEDULED` is `budget.BUDGET_DEPENDENT`.
- `mo_cma_es`: p_target is 2/11, Table 1 of Igel, Hansen & Roth (Evol.
  Comput. 2007) at λ = 1, and c_p 1/12 with it; it was (5 + √(1/2))⁻¹ ≈
  0.1752, the value the 2010 paper prints while giving its defaults "as
  given in" that table — Table 1's value at λ = 2, a slip (MOCMA-9). The
  step size now aims at a slightly higher success rate, so MO-CMA-ES's
  populations differ from the previous version's; no other algorithm
  changes.
- `sms_emoa` at two objectives keeps the two extremes of the worst front
  always and compares the other points by their contributions from the
  neighbours alone, as EJOR 2007 (§2.1.3, "We decided to omit y_ref and
  always keep these extremal solutions") and the 2005 paper (§3.2) do; the
  order there no longer depends on the objectives' scale. It used the
  adaptive reference point nad + 1 of Beume's thesis at every M, so an
  extreme could go. From three objectives nothing changes. When the worst
  front has only the two extremes, nad + 1 still decides which goes (SMS-4).
- CI: `actions/checkout` v4 -> v5 and `actions/setup-python` v5 -> v6, the
  releases that run on Node 24. GitHub had begun forcing the old ones onto
  Node 24, with a deprecation warning on every job.
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
- Four reported defects that turned out to be the papers' own letter, each
  traced, measured and now described in its header rather than changed.
  **Two_Arch2** can return far fewer than n_DA solutions (on ZDT2, where the
  search collapses to x1 = 0): the diversity archive admits only nondominated
  points, and its output is §III-A's.
  **RVEA** keeps one solution per non-empty reference vector (§III-C, Alg. 2);
  on IDTLZ1_5D only 21 of the 126 vectors are the nearest to any point of the
  front, so the population stays far below N. **LIS/LCS** breeds N − K offspring
  with the old K and K with the new one (Alg. 1, lines 4-10), so a step costs
  N − K_old + K_new while line 16 books N; the binding counts what was spent.
  **MaOEA-IAMD** spends exactly N per step; its only extra is the 8·D
  variable-classification sample at setup (Alg. 1, line 2).
- NSGA-III's header: the quoted "Special care is taken to handle degenerate
  cases and nonnegative intercepts" is in the final journal version (§IV-C,
  p. 581 or 582), but it says only that care is taken, so the per-axis
  `nadir − z^min` fallback is now declared as this port's own rule (NSGA3-N2),
  next to the two places where the text and Algorithm 2 disagree — the ideal
  point and the extreme points, both kept historically as the text says
  (NSGA3-N1) — and the 1e-6 ASF weight, which Part I never states (NSGA3-N3).
- **The reference fronts of DTLZ5 at four to six and ten objectives, of DTLZ6
  at four to six, of WFG3 at three to five and of MaF6 at eight and ten are
  now their FULL Pareto fronts**; where no front is built yet — WFG3 at six
  and ten, DTLZ5 at fifteen, DTLZ6 at ten and fifteen, MaF6 at fifteen — the
  curve stays, with its run-time warning. Each of the six-objective builds took about six hours. The four
  were designed with a degenerate, curve-shaped front and the registry's
  reference sets were that curve, but at those sizes the true front also has a
  non-degenerate part (Ishibuchi, Masuda & Nojima, IEEE TEVC 20(5), 2016, for
  DTLZ5, DTLZ6 and WFG3), so the curve was a proper subset of it.
  `mootation.benchmarks.fronts_full` builds each front in the problem's
  reduced coordinates — the positions plus the distance scalar, M dimensions
  whatever the number of variables — sampling the box and its faces of every
  dimension (the far parts of the DTLZ5/6 fronts lie three and four
  coordinates deep); keeps a point only if it survives an independent sample
  of 10^6, a local search for a dominating point and an EXACT test against
  every attainable point, for which one value of the distance scalar is
  decided greedily in M steps and the scalar is scanned on a grid of 20 001;
  re-evaluates every survivor with the problem's own code; and stores the
  result in DSS order in `benchmarks/_fronts/` (`python -m
  mootation.benchmarks.make_fronts` rebuilds them). The exact test is what
  makes them clean: after every sampling check a quarter to a third of the
  DTLZ5/6 points, and 1-2 % of WFG3's, were still dominated, by 2e-4 to 4.5e-2
  of the range. The frame moves with the front. The DTLZ5 and DTLZ6 fronts run
  to the largest g the box allows (2.5 and 10), so their nadir reaches about
  3.4 and 11 in the objectives before the last, where the curve never
  exceeded 1; WFG3_3D's first objective reaches 3, the paper's counterexample
  (3, 1, 1) being on the front. So `hv`, `hv_h`, `igdp_norm`, `eps_norm` and
  every IGD-type value change on the campaign's DTLZ5_5D, DTLZ6_5D, WFG3_3D
  and WFG3_5D: `--recompute` updates finished runs' final values, but their
  trajectories need the runs again. MaF6 = DTLZ5(I = 2, M) with the factor
  (1 + 100 g) leaves the curve only from seven objectives — at five the build
  finds nothing off it, so `DEGENERATE_SUBSET_FROM` says 7 for it, not 4. And
  cos(pi/2) is 6.1e-17 in floating point, which let every point with x_1 = 1
  and g > 0 pass for nondominated and set the nadir of f_M to 1 + g_max (251
  on MaF6), while every point with f_M > 1 is dominated by the curve.
- `campaign_all.toml` records its trajectories on the new logarithmic grid,
  with GD+, ε, the range cover and the duplicate and nondominated shares on
  them, the hypervolume only up to three objectives, `roi_dist` on the final
  population, and the two baselines in its algorithm list.
- `campaign_all.toml` also records `igdx`, `cr` and `pdist` on the final
  population and a Monte-Carlo hypervolume (10 000 points) on the
  five-objective trajectories, and runs the two ablations: 62 930 jobs.
- `--instance-space` (task 6, item 13): the plane is PILOT's optimum computed
  exactly (`instance_space.pilot_exact`, the reduced-rank regression), with
  BFGS from `tries` starts kept as a check, no longer the BFGS end of the
  highest preservation; its axes are fixed — whitened, or with
  `--is-calibration preserve` stretched to the highest preservation — so that
  the footprints no longer depend on where BFGS stopped, and every footprint's
  area share comes with its range over six calibrations of the same optimum
  (`area_share_min`, `area_share_max`). The performance rows are weighted
  sqrt(q/a) (`--is-weight none` for the papers' loss); the bbob-biobj
  problems' features come from a uniform sample of their box, 19 of them
  (`instance_space.sample_features`; `--is-bbob-features properties` for the
  old ones). When SIFTED keeps no feature, all stay.

### Added

- DAS-CMOP1-9 (`benchmarks/dascmop.py`; task 7, §8): Fan et al.'s
  difficulty-adjustable constrained problems (Evolutionary Computation 28(3),
  2020, Table 2), at the sixteen difficulty triplets of their Table 3 and
  Figure 6's, with any other triplet through `dascmop.problem`; fronts built on
  first use from a dense sample with each point's g bisected to its
  feasibility boundary; the properties' new key `difficulty` names the
  triplet's kinds. 1141 problems in 40 families. LIR-CMOP, DOC and RWMOP wait
  for their papers.
- Typed variables (`mootation/var_types.py`; task 7, §10): `minimize(var_types=...)`
  declares each variable "real", "int", ("discrete", values) or "bin"; the vault
  rounds the integer and discrete ones after every write
  (`DataVault::set_snap`, `int_repair = "lamarck"`), or only the evaluation sees
  them rounded (`"baldwin"`); polynomial mutation takes `eta_int` (3) for them.
  Declared for RE22, RE23, RE25, RE35, RE36, CRE24, CRE25 and RE22c; a run
  records `dup_x_share`. The binary genome reaches Python (task 7, supplement
  2, §3): the bits go to the cores' binary genome and back
  (`result.binary_variables`, population files with b columns, warm starts),
  and the cores without one refuse it by name.
- OneMinMax, LOTZ and OneJumpZeroJump (`benchmarks/binary.py`; supplement 2,
  §3), with exact fronts checked against every string up to n = 12, and the
  paper's measure `cover_gen` / `cover_evals` (and for the inner archive) in a
  campaign's `final`. 994 problems in 39 families.
- RE22c and RE33c (task 7, §9): RE22 with x1's list as meant (3.10 for "3,10")
  and RE33 with x4 in [2, 11] (reproblems' issue #9), beside the originals;
  their reference fronts built here from an exhaustive grid refined locally
  (`benchmarks/re_fronts.py`, `_re/corrected/`).
- The stochastic population update and an inner archive (task 7, supplement
  2): `survival = "stochastic"` in SMS-EMOA, NSGA-II and NSGA-III after Bian,
  Zhou, Li & Qian (IJCAI 2023) — a random Q′ of ⌊`spu_frac`·|Q|⌋ members
  (1/2; 3/4) selected alone, the rest kept, so any member survives with
  probability at least 1/2 or 1/4 (tested); `spu_frac = 1` is greedy bit for
  bit. `inner_archive = "nd"`, `inner_archive_cap` and `archive_reuse_p` in
  SMS-EMOA, NSGA-II, NSGA-III and IBEA-ε+: the non-dominated points the
  algorithm evaluated, cut by DSS, and parents drawn from them; the result's
  `inner_archive`, `operator_stats`' `archive_parent_share`
  (`operators/inner_archive.hpp`). For the papers' theory runs:
  `bin_crossover = "none"`, NSGA-II's `mating = "each"`, and
  `crossover = "none"`. Every default is unchanged.
- `coords`: the variation in learned coordinates (`operators/learned_coords.hpp`,
  `variation.hpp`; task 7, supplement 1), in NSGA-II, IBEA-ε+, SPEA2+SDE,
  AGE-MOEA, SMS-EMOA and mosa's sbx step. `pca` crosses and mutates in
  y = (x − μ) R, R the eigenvectors of the population's covariance; `pca_nd`
  from its non-dominated members; `randrot` a fresh random rotation, the
  control; `axis`, the default, as before, bit for bit. R = I for the first
  `coords_warmup` generations (10), then updated every `coords_update` (1);
  no box in y (the crossovers unbounded, the mutation within the projected
  population's widened range), the children mapped back and repaired. The
  spectrum goes to `operator_stats`; `[campaign] coords_diag` adds the
  principal angles between the learned minor subspace and the problem's
  distance block (`run/coords_diag.py`); `_core.coords_state()` gives the
  basis. Tests: the basis complete and orthonormal, randrot reproducible, the
  offspring distribution turning with the population, the children in the
  box, an endless warm-up identical to axis in NSGA-II, SMS-EMOA and mosa.
- `--invariance-check` (`run/invariance.py`, task 7, §5.4): every algorithm
  entry of a campaign on its problems, on the objectives times 2^10 and times
  (2^-6, 1, 2^6, ...) — exact, so the trajectories are compared bit for bit —
  and shifted by 2^4 over 30 seeds — rounded, so compared by the rank-sum test
  with Holm's adjustment: "invariant", "rounding" or "not invariant". The tests
  hold the two cases known in advance: NSGA-II is invariant to a common
  power-of-two scale, SMS-EMOA at three objectives is not.
- Deterministic, step and relative noise, and three ways to measure a noisy
  run (`benchmarks/noisy.py`, task 7, §4). Besides `_n1`/`_n5`: `_d1`/`_d5`,
  the noise a hash of x, the same in every run and at every repeat;
  `_s1`/`_s5`, a hash of the cell ⌊(x − lb)/h⌋, h = `[campaign] noise_cell`
  (0.01) of the range — a step at every cell border; `_h1`/`_h5`,
  σ (f − ideal) N(0, 1), a relative error. Every kind at both levels of ZDT1,
  ZDT4, DTLZ1_3D, DTLZ2_3D, DTLZ2_5D, WFG4_3D, WFG7_3D, rcDTLZ2_3D, RE21,
  RE25 and RE34. `final` records every indicator as recommended (the
  noise-free values of the answer, as before), `<metric>_oracle` (the
  noise-free non-dominated set of every evaluation, reduced to the answer's
  size) and `<metric>_observed` (the noisy values of the answer);
  `meta["noise"]` the kind and the cell. `_n1`/`_n5` are unchanged bit for
  bit. The noisy copies of RE are ranked with the real-world problems.
  982 problems in 36 families.
- The rotations of the other families (`benchmarks/rotated.py`, task 7, §7),
  each turning only the distance block about its own optimum, conditioned
  only where g is isotropic, each with its Q = I control and three instances:
  rWFG1-3 (about 0.35, clipped); rWFG8 and rWFG9, whose optimum b_param makes
  depend on other variables, turned about it — 0.35^(1/e(u)), computed in
  order for the current positions — and checked on the front's sphere;
  rcDTLZ5 and c0DTLZ5; MaF by the form of its g (rcMaF1, rcMaF5, rcMaF6
  conditioned; rMaF2, whose per-objective chunk sums a rotation does change,
  rMaF3 and rMaF4 rotated; MaF7 not; the table in the header); rZCAT1-20 about
  the Pareto set y_II = g(y_I), clipped to the box, ZCAT19 and ZCAT20 with a
  matrix for each size of their distance block; rcPolygon with the optimum
  moved off the centre first, control c0Polygon. The header records how often
  rWFG and rZCAT clip. 904 problems in 36 families.
- Rotated ZDT, rotation instances and more objectives (`benchmarks/rotated.py`,
  task 7, §3): rZDT1-3 and rZDT6 evaluate ZDT at x' = |Q (z - c)| / m (the
  sum of absolute values in g is not isotropic, so the rotation changes the
  function; at Q = I it is shiftZDT, the control), rZDT4 at x' = Q (z - c),
  ZDT4's Rastrigin term rotated; unclipped. Three instances of Q,
  `<name>_i1` .. `_i3`, of rZDT1-4, rZDT6, rDTLZ1, rDTLZ3, rcDTLZ2 and
  rWFG4-7 at every size they have; instance 1 is the problem without the
  suffix, whose matrix the tests have held since task 6. rcDTLZ2 and c0DTLZ2
  at six and ten objectives too. The tests hold the hash of every new matrix.
  624 problems in 29 families.
- `mosa`, multi-objective simulated annealing (`algorithms/mosa.hpp`; task 7,
  §1, §1.2a and §6): chains over an archive of every non-dominated point,
  composed from AMOSA's archive and informed reseeding, DBMOSA's dominance
  energy and Pareto simulated annealing's Chebyshev chains with adaptive
  weights. The energies `cmb` (the default), `eps`, `dom`, `tch`; the steps
  `mix` (the default), `laplace`, `de`, `sbx` and `adaptive` (credit over a
  window); the temperature and the step scale fall over the whole budget, so it
  is budget-dependent (`run/budget.py`). One step is one evaluation; the answer
  is the archive reduced to `pop` by DSS. Its records go to `operator_stats`
  (`metric_names.SA_STATS`, `sa_w_mean_*`, `sa_w_sd_*`). The public list is 64
  algorithms. The header says which mechanism is whose and that AMOSA's,
  DBMOSA's and PSA's papers are still to be checked against.
- Differential evolution and an operator portfolio in the switchable
  algorithms (NSGA-II, IBEA-ε+, SPEA2+SDE, AGE-MOEA, SMS-EMOA; task 7, §2):
  `crossover = "de"` (DE/rand/1/bin, the host's pair and two more parents from
  its own selection, F = 0.5, CR = 1), `"de_pbest"` (JADE's
  current-to-pbest/1, x_pbest one of the host's best `de_p` by its own order,
  r2 from the population and the archive of displaced parents), with
  `de_adapt = "shade"` (a SHADE memory; success = the child survived the
  host's selection; weights `equal` or `contribution`), and
  `crossover = "portfolio"` (`portfolio = [...]`, `portfolio_credit = uniform |
  borg`). Their records — the SHADE memory, the CR and F drawn and kept, every
  operator's share of the children and of them kept — go to
  `operator_stats` and the trajectories (`metric_names.DE_STATS`,
  `portfolio_share_*`, `portfolio_kept_*`). `mutation = "none"`. The knobs
  `F` and `CR` now reach the five hosts; `pm` is a knob of NSGA-II,
  SPEA2+SDE, AGE-MOEA and SMS-EMOA too (it was fixed at 1/n and reported as
  ignored); `ref_delta` gives SMS-EMOA and HypE a reference point relative to
  the objectives' range (§5.3). The knobs added from now on reach a core
  through `set_param` / `set_word` (`settings.hpp`, `Config.params` /
  `Config.words`); docs/embedding.md lists which algorithm takes which
  operator knob. Every algorithm's defaults are unchanged
  (`tools/compat_check.py`: 0 fingerprints differ).
- shiftZDT1-4, shiftZDT6, shiftDTLZ6 and shiftDTLZ7 (`benchmarks/shifted.py`,
  task 6, item 2): ZDT1-3, ZDT6, DTLZ6 and DTLZ7 put the optimum of every
  distance variable on the lower bound, where clipping a variable lands for
  free, and ZDT4 in the middle of [-5, 5]. Each distance variable is read
  through h(x) = |x − c| / max(c, 1 − c), 0 at c and 1 at the farther bound,
  so g keeps its range (shiftZDT4: u = x − c in ZDT4's box); c from
  `shift_centres`, at least 0.15 of the range from both bounds and from the
  middle. Not shiftDTLZ1-4's cyclic shift: a linear g would jump across the
  wrap. The front, ideal and nadir are the base problem's, its reference front
  row for row; with c = 0 every problem is its base. 19 problems, 457 in all.
- Pareto-set samples for ZDT1-4 and ZDT6, DTLZ7, DTLZ5 and DTLZ6 up to three
  objectives (from four their fronts have a part with g > 0), IDTLZ1-2,
  SDTLZ1-2 and minusDTLZ2 (`pareto_sets.fill`, `minus_dtlz`), so `igdx`, `cr`
  and `ps_dmin` read them too.
- The problem property `boundary` (task 6, item 4): the optimum on a bound of
  the box, Huband et al.'s extremal parameters (Tables V, VII and XV) beside
  `centre`, their medial ones. `properties.measured()` reads both off a
  problem's Pareto-set sample (the share of the distance variables within
  1e-9 of a bound or of the middle, a flag from 99 %), and the tests hold the
  table's `boundary` to it on every problem with a sample. `--cover` writes
  every table a third time over the problems whose optimum is neither at the
  centre nor on the boundary (`subset` `interior`), and `--portfolio` gives
  those as the group `front_interior`; `--ranks --by boundary` groups by it.
- `--bias` brought to Kudela, van Stein, Bäck & Kononova (GECCO 2026; task 6,
  item 11). Their problems beside f1 (`uninformative_n02_2D`, unchanged, bit
  for bit): f2b, f3b and f4b, a disconnected, a stepped and a two-region front
  under a χ²-shaped noise (Table 1, variant β), and f5, [a, −a], every point
  non-dominated, in two and ten variables, from the run's seed and the
  evaluation's number. The problem counts, in 20 bins and exactly on a bound
  or at ½, the coordinates of every evaluated point and of the run's
  non-dominated ones, and the campaign writes the counts into `meta.json`
  (`structural_bias`). `--bias` gives the paper's binsizes at the bounds and in
  the centre for the final populations, the non-dominated and all evaluated
  points, the shares exactly on a bound and at ½, a χ² test on one point per
  run, the Clark-Evans index at two variables and a class by our thresholds,
  in place of the earlier ten-bin χ² per variable and the shares near the
  bounds. `structural_bias.toml` runs f1-f4b at the paper's 30 000
  evaluations; `structural_bias_f5.toml` runs f5 with ten seeds and DMS at
  10 000, whose run there otherwise takes over ten minutes.
- Indicators of convergence without coverage (task 6, item 8): `gdp_min`, the
  smallest d+ of any point of the set to the reference in `igdp_norm`'s frame,
  with the reference's resolution beside it (`gdp_min_ref_step`,
  `gdp_min_below_step`); `g_min` and `g_median`, the problem's own g(x) − g*
  over the set (`BenchProblem.distance_fn`, `benchmarks/distances.py`: ZDT,
  DTLZ and their variants), exact and without a reference; `ps_dmin`, the
  smallest distance from a solution to the Pareto-set sample.
- C-DTLZ (`benchmarks/c_dtlz.py`, task 6, item 5): Jain & Deb's constrained
  DTLZ (IEEE TEVC 18(4), 2014) — C1-DTLZ1, C1-DTLZ3, C2-DTLZ2,
  C2-ConvexDTLZ2, C3-DTLZ1, C3-DTLZ4 — at three and five objectives, the
  registry's first constrained family besides MaF9. Constraints in the
  library's sign (at most 0 when satisfied, the paper's c >= 0 turned); two
  equations read as pymoo reads them, written down in the module's header;
  values checked against pymoo 0.6.2 at 60 points to 1e-12 (`tests/data/`).
  Analytic fronts and exact frames; Pareto sets and g for C1 and C2-DTLZ2. The front of
  C2-ConvexDTLZ2 is larger than the paper's: behind the hole the cylinder
  cuts in the convex front, points of larger g are Pareto-optimal too; it is
  sampled and checked point by point (`BenchProblem.front_exact = False`, a
  `--cover` caveat).
- RE and CRE (`benchmarks/re.py`, task 6, item 6): Tanabe & Ishibuchi's
  real-world problems (Applied Soft Computing 89:106078, 2020), RE21-91 and
  CRE21-51, ported formula for formula from github.com/ryojitanabe/reproblems
  (MIT, commit 7876b4e); values checked against its Python version at 120
  points to 1e-12 (`tests/data/`). Defects kept on purpose and those fixed
  upstream are in the header. The reference fronts are the authors' approximations, shipped
  as they are in `benchmarks/_re/` with their license, and for CRE the
  feasible points of RE's; `BenchProblem.front_exact` is False there (and for
  C2-ConvexDTLZ2, replacing `registry.APPROXIMATE_FRONT`). RE91's normal
  draws come from the run's seed. The property `family_kind` (engineering /
  synthetic), and the rank table counts the real-world problems apart.
  535 problems in 27 families.
- MW1-14 (`benchmarks/mw.py`, task 6, item 5): Ma & Wang's constrained
  problems (IEEE TEVC 23(6), 2019), n = 15, MW4, MW8 and MW14 at three and
  five objectives (n = M + 12). Constraints in the library's sign; values
  checked against pymoo 0.6.2 at 85 points to 1e-12 (`tests/data/`; MW11's
  distance function g3 as in Table II, not the text's g2; the header lists
  the readings). The fronts are built from the problems' geometry, no file: the
  least feasible reachable point at every f1 or angle, exactly where the
  constraints allow it (factors linear in f2) and by bisection elsewhere,
  refined where the front is steep, ends or leaves the unconstrained one, then
  the non-dominated part — they match the paper's Figs. 4 and 5, every point
  is feasible to 1e-12 and reached by a decision vector, and no sampled
  feasible point beats them by more than the sample's spacing. Exact frames; Pareto sets and g for the six whose front
  lies at g = 1 (MW1, MW2, MW4, MW6, MW8, MW14). MW4, MW8, MW14 get dense
  references from four objectives. 552 problems in 28 families.
- `[campaign] answer_select` (task 6, item 13): how the run archive becomes
  the archive scenario's answer — `"dss"`, as before and the default, or
  `"igdp_self"`, the N points of the smallest IGD+ against the archive thinned
  by DSS to 1 000 points, chosen greedily without a reference
  (`archive.igdp_self_order`, `archive.greedy_igdp`); recorded in
  `meta["archive"]`, repeated by `--recompute --scenario archive`. And
  `--oracle`: the best N points of every run's archive under IGD+ against the
  reference, beside the population, the archive answer, igdp_self's choice and
  the whole archive (`run/oracle.py`, `<results>/_oracle/oracle.csv`).
- `python/examples/experiments/ablation_bound_repair.toml` (task 6, item 13),
  not run: the 17 algorithms with the `bound_repair` knob at each of its
  values — the 12 whose own operators leave the box at clip, reflect, random,
  midpoint and wrap (resample is not available to them), the 5 that vary by
  SBX and polynomial mutation, which stay in the box, with gaussian mutation at
  all six — on ZDT1, ZDT4, DTLZ1_3D, DTLZ3_3D, their shifted copies and
  rDTLZ1_3D and rDTLZ2_3D, ten seeds at 25 000 evaluations (9 000 runs).
- Rotated DTLZ and WFG (`benchmarks/rotated.py`, task 6, item 1 and its
  supplement 2): the distance variables rotated about their optimum, the
  position variables untouched, front and Pareto set exact — rDTLZ1-4 (y =
  1/2 + Q (z - c), unclipped), rcDTLZ2 (Lambda Q, condition number 10^3),
  rWFG4-7 (WFG's normalised distance variables about 0.35, clipped to
  [0, 1]), at three and five objectives, and the Q = I controls r0DTLZ1,
  r0DTLZ3 and c0DTLZ2. Q a Haar rotation from a fixed seed; the tests hold a
  hash of every matrix, bbob-biobj's included. The header records the
  literature and how the construction differs, and how often rWFG clips.
  `mootation.families` is the one family rule of the registry and the
  tables. 511 problems in 25 families.
- `properties`: minusDTLZ2 has a row; ZCAT's centre is False (was None).
- Noisy copies (`benchmarks/noisy.py`, task 6, item 7): `<base>_n1` and
  `<base>_n5` add 0.01 and 0.05 (nadir − ideal) N(0, 1) to each objective of
  ZDT1, ZDT4, DTLZ1_3D, DTLZ2_3D and WFG7_3D, the noise from the run's seed and
  the evaluation's number (`make_evaluator`). A campaign measures them on the
  noise-free objectives (`BenchProblem.true_objectives`), keeps those in its
  files and snapshots and the noisy ones in `final.csv` as `f_seen`, and
  records `true_dominated_share`. A family of their own, `noisy`. 487
  problems in 20 families.
- Dense references (task 6, item 9): `registry.dense_reference` gives, for the
  problems with an analytic front at 4 to 10 objectives, the first of 1 000
  ... 100 000 points whose step is at most a third of the population's;
  a campaign measures the front indicators against it beside the usual
  ones, as `<indicator>_ref_v3` (`[campaign] dense_reference`, on by default:
  the usual values do not change), and
  `--recompute-reference` adds those to a finished campaign from its saved
  answers, the old values untouched (`recompute_dense`); the tables read
  them. IGD, IGD+ and GD+ are computed in blocks of reference points now (ε,
  igdp_norm and the others were), with the same values; `reference_step`
  uses a KD-tree when SciPy is there, with the same value.
- `--reference` (task 6, item 10): a warning when the seeds and the number of
  problems leave no mark but `=` possible under Holm (at k seeds a side,
  beyond 0.025·C(2k, k) problems); `--adjust bh` for Benjamini–Hochberg over
  the problems; beside the counts the `=` with a large A12, how many
  problems fall in each class of A12 and the median A12 with its class;
  across problems, for `igdp_norm` and `eps_norm`, the
  signed-rank test on the logarithms of the medians, and for every metric
  the sign test and the Hodges–Lehmann estimate (a ratio of medians for those
  two). The header says the order is by `+` minus `-`, not a rating.
- An Owen-scrambled Sobol design (`mootation.designs`, task 6, item 12.2):
  the Sobol sequence with Joe & Kuo's direction numbers (new-joe-kuo-6.21201,
  shipped in `mootation/_sobol/` with its BSD licence; unscrambled, the
  points are SciPy's bit for bit) under Owen's nested uniform scrambling by
  Burley's hash (JCGT 9(4), 2020: bit reversal, the Laine-Karras
  permutation, bit reversal), keyed by the run's seed and the coordinate.
  `init = "sobol_owen"` and the baseline `sobol_owen_search`, not
  budget-dependent; the header records its correlations and discrepancy
  beside the randomized Halton's.
- A randomized Halton design (`mootation.designs`, task 6, item 12.3), after
  Owen, "A randomized Halton algorithm in R" (arXiv:1706.02808, 2017, Eq. 5
  and Algorithm 1): every base-b digit of every coordinate through an
  independent random permutation, as many digits as a double resolves,
  seeded by the run. `init = "halton_rand"` starts an algorithm from it and
  the baseline `halton_search` samples it; a sequence, so not
  budget-dependent. The header records its correlations and discrepancy at
  10, 30 and 50 variables. Tests hold `sobol`, `lhs` and the baselines' draws
  to their earlier values.
- Indicators of a constrained problem (task 6, item 5.2): every indicator is
  measured on the feasible part of the answer (`metrics.compute(...,
  constraints=)`), `feas_share` is recorded beside them and `fe_first_feas`,
  the first evaluation to give a feasible point, in `final`. An answer
  without a feasible point has no value — none, not 0 or infinity — and is
  counted apart: `--compare` writes "k empty" in its cell, `--ranks` (and its
  CSV) and `--ranks --by` an `empty` column, `--ci` and `--reference` their
  count; in the medians, the tests and the ranks it counts as the worst
  value. A
  constrained run keeps its variables in snapshots and checkpoints.
- Maximin Latin hypercubes (`mootation.designs`, after Morris & Mitchell,
  "Exploratory designs for computational experiments", 1995): their φ_p
  criterion and annealing search, once per p of their Section 5, the best by
  the maximin order; the search cut at 20 000 perturbations (2·10⁷/n above
  1 000 points) and the point drawn inside its Latin-hypercube cell instead of
  the paper's grid i/(n − 1), which holds the optima of DTLZ and ZDT. The
  tests check two designs of the paper's catalog. As a start, `init = "lhs"`;
  as a baseline, `lhs_search`, one hypercube of the whole budget
  (budget-dependent).
- Knobs for DMS and EMyO/C (task 5, runs A1 and A2),
  every default unchanged: DMS `dms_init = "random"` (a Latin hypercube of n
  points from the seed) and `dms_poll = "random"` (a poll centre drawn among
  those still worth polling, the directions in random order), so that its ten
  seeds are ten runs and its steps leave the dyadic grid that lands on the
  middle and the bounds of the box; EMyO/C `bound_repair`, `step_share`, `F`
  and `F_spread` (its offspring's repair, the clamp of the difference vector,
  and a factor on it, the paper's clip, 0.5 and 1 by default); and the problem
  `wideDTLZ3_3D`/`_5D`, DTLZ3 with the distance variables in [0, 1.2], whose
  middle is a local optimum of g (438 problems, 16 families).
- Small budgets (task 5, run B1): `[campaign] hard_budgets` reads a budget as
  the last record at or below it and the run archive after exactly that many
  evaluations reduced to `answer_k` points (`meta.json` `hard_at`);
  `pop_fit = "nearest"` fits a population to the nearest size its algorithm
  accepts; `ladder_min_gens` leaves out rungs of too few generations; an
  algorithm entry's `init = "sobol"` starts it from a scrambled Sobol design
  and `init_share` from a plan taking that share of the budget (the
  nondominated points of the plan first, the rest by DSS), through
  `minimize(init=..., init_share=...)` and the new `mootation.designs`.
- The analysis rules of task 5, item 2: an
  algorithm whose seeds are one run (DMS) is named by `--cover`,
  `--portfolio` and `--instance-space`, which add the sensitivity row "one run
  each" — every algorithm read from one seed at a time, the mean over the
  seeds (`cover_one_run.csv`, the portfolio's rule `one_run`, the footprints
  `good_one_run`); every budget reads the answer's size beside the
  indicators, a gdp level counts only for an answer of at least N/2 points
  (GD+ does not see how much of the front two points cover), and the answers
  shorter than that are marked: `cover_short_answers.csv`, the portfolio
  matrix's `seeds_short`, the rank tables' `short` column and `~` in
  `--compare`.
- `--instance-space` (task 4, item 5): instance space analysis written from
  Smith-Miles & Muñoz 2023 (PRELIM's relative performance, SIFTED's first
  step, PILOT by BFGS with the analytical solution and the exact optimum's loss
  beside it) and Muñoz & Smith-Miles 2017 (footprints by Delaunay triangulation
  with distance, density and purity limits; contradictions between the best
  algorithms' footprints removed), the problem properties as features, one
  space for the problems with a front and one for bbob-biobj; footprints at
  `--cover`'s levels and budgets, their areas as shares of the known region
  and of the convex hull. Checked against Muñoz et al. 2018 (Machine Learning
  107(1), Appendix A): the analytical solution of Corollary 1 is not the
  optimum, which is that of Corollary 2's rank-constrained form, a reduced-rank
  regression. Needs SciPy.
- `--portfolio` (task 4, item 5): the oracle and the single best solver, the
  Shapley values of the coverage game and the complementarity of every two
  algorithms at `--cover`'s levels and budgets, at 5, 7 and 10 seeds of ten,
  with the problems with a front and the bbob-biobj ones apart; the gap to the
  oracle per problem; and the matrix problem × algorithm × budget of the seeds
  reaching every level and the median and 7th best of four indicators.
  `cover.prepare` gathers the floors, the problems left out and the marked ones
  for both.
- `--recompute-reference`: a run measured against an older reference front
  than the current one, on a problem whose reference changed since, measured
  again from its saved answers — final, archive, archive checkpoints and
  trajectory — with the old values kept beside the new (`meta.json`
  `reference_v1`, `trajectory.ref_v1.jsonl`). Every run's `meta.json` now
  records `reference_version` and `reference_rows`.
- `front_dim` among the problem properties: the Pareto front's dimension (1 for
  DTLZ5/6 at three objectives and ZCAT14-16, 2 for Polygon and IPolygon, M − 1
  for a regular front), for reading ratios to the IGD+ floor across M.
- `--behaviour` (task 4, item 4): descriptors of every run from its trajectory
  — time to `gdp_norm` targets and to `igdp_norm` within a factor of the
  campaign's best, where both settle, the convergence slope, whether the
  coverage kept improving after the convergence stopped, the collapse of
  `range_cover`, stagnation at `nd_share` ≈ 1, `dup_share`, `igdx` and `cr` —
  their medians per problem and algorithm, and the algorithms grouped by
  those profiles (average linkage) beside the families `algorithms.def`
  declares. CSV under `<results>/_behaviour/`.
- Problem properties (task 4, item 5): `degenerate`, `disconnected`,
  `linkage`, `n_obj`, `n_vars` and `bbob_groups` beside Huband's keys, and rows
  for MOP1-7 (Liu, Gu & Zhang 2014), BT1-9 (Li, Zhang & Deng 2017), Polygon and
  IPolygon; every stage-3 problem has one. `--ranks --by` takes the new keys.
- What the stage-3 trajectories need (task 4). **Variables at the records**:
  `mootation._core.current_variables()` returns, from an `on_generation`
  observer, the variables of the rows it has just received, so a campaign
  records `pdist`, `igdx` and `cr` along the trajectory (the baselines pass
  theirs to the observer). **Snapshots**: `[campaign] snapshot_variables =
  "pareto_set"` (or `"all"`) keeps the variables in `snapshots.npz` too, and
  `--recompute-trajectory METRICS` computes indicators at every record from
  the snapshots into `trajectory.jsonl`, to about 1e-6 of the recorded values
  (float32 rows). **Archive checkpoints**: `archive_checkpoints = true` takes
  the archive scenario at the `record_at` counts below the budget —
  indicators in `meta.json` (`archive_at`), points in `archive_at.npz` — which
  `--at` and `--cover --scenario archive` read. `meta.json` also records
  `record_seconds` (what the records cost), `budget_nominal` and
  `has_pareto_set`. `record_at_metrics` names costly indicators recorded
  only at the `record_at` counts and at the end; `--recompute-trajectory`
  fills the other records later. `metrics.compute` reads `igdp_norm`,
  `gdp_norm`, `eps_norm`, `tau90` and `gap_max` off one pass over the
  normalised differences (`front_distances`) instead of one each — the same
  values bit for bit, half a trajectory record's cost at five objectives.
  The stage-3 configs record every indicator along the
  trajectory (the hypervolumes exact up to five objectives), the operator
  statistics, the checkpoints, and the variables in the snapshots where the
  problem has a Pareto-set sample. Measured on all 225 stage-3 problems
  (five algorithms, one seed, scaled to the stage): the records cost about
  4 % of the stage's CPU — 0.3 s a run at two objectives, 1.8 s at five —
  and the files come to about 32 GiB, of which the snapshots' variables are
  6.5 and the checkpoints 2.
- What stage 3 of the comparison campaign needs (task 3). **Budget ladder**:
  `[campaign] record_at` puts trajectory records at exact evaluation counts
  the decade grid misses (2 500, 5 000), and `ladder = [2500, 5000, 10000]`
  runs every budget-dependent algorithm separately at each of those budgets,
  filed as `<algorithm>@<budget>` after all other jobs; `meta.json` says
  `budget_dependent` and, for a rung, `ladder_of`. `mootation.run.budget`
  keeps the list read off the code beside the measured one and measures it
  (`python -m mootation.run.budget`): each algorithm at 10 000 and 25 000
  evaluations, every record compared bit for bit — exactly the seventeen with
  a t/t_max schedule differ. **`--cover`**: for levels τ of `igdp_norm` (and
  `gdp_norm`, the progress alone; the gap of `hv_h` to the campaign's best on
  bbob-biobj) and budgets, the problems nobody covers, the smallest sets of
  algorithms covering the rest — exact, by branch and bound, all of them up to
  a limit, with the greedy set — the curve k -> share covered with problems
  and with families weighted alike, and a bootstrap over seeds; CSV in
  `_cover/`. **Metrics**: `gap_max` (the largest d+ hole, tau90 at 100 %),
  `nn_cv` (the evenness of the non-dominated points, no front needed) and
  `gdp_norm` (GD+ in `igdp_norm`'s frame), with a table in docs/running.md of
  which indicators measure progress and which coverage. **Archive**:
  `[campaign] archive_variables = "selected"` keeps the variables only of the
  points the archive scenario selects — the objectives of every archive point
  in `archive.csv.gz`, the selected points in `final_archive.csv.gz`, which
  `--recompute --scenario archive` then reads — a tenth of the room of
  `archive.csv` over the stage-3 problems; `"all"`, the default, writes
  `archive.csv` as before. **Configs**: `campaign_stage3.toml` (every
  algorithm but HLMEA, HypE and R2-IBEA, the four baselines, 225 problems —
  stage 2's 203 and every BT, MOP, Polygon and IPolygon instance but
  IPolygon_8D — ten seeds, 25 000 evaluations, the ladder, the archive as
  `"selected"` and the population snapshots: about 32 GiB for the whole
  stage with task 4's trajectories, snapshot variables and archive
  checkpoints), `campaign_stage3_heavy.toml` (those three, the same settings and
  results tree) and `campaign_stage3_probe.toml`.
- RVEA* (`rvea_star`), the variant for irregular fronts that Cheng, Jin,
  Olhofer & Sendhoff give in §VI of the RVEA paper (IEEE TEVC 20(5), 2016,
  Algorithm 4). Beside RVEA's uniform vector set V it keeps a second set V*
  of N vectors; after every generation each vector of V* that is the nearest
  one to no non-dominated solution is replaced by a random direction inside
  the non-dominated solutions' range after the translation, where every
  minimum is 0 (footnote 6), and V ∪ V* guides the selection. The paper keeps
  one solution per non-empty vector, so its population can reach 2N; here
  every generation keeps N: all of V's best, then V*'s by APD, the runners-up
  when fewer than N vectors are occupied (RVEA*-1). Algorithm 4's "for j = 1
  to N" runs over the M objectives (RVEA*-2). `rvea` is unchanged. The public
  list is 63 algorithms.
- Set comparisons and two indicators (task 2, D4-D7). `r2`, the discrete R2 on
  `hv_h`'s Das-Dennis lattice, a final or trajectory metric for four
  objectives and more (`--recompute r2` adds it to a finished campaign).
  `--seed-distance` (D4): the chamfer distance between the non-dominated sets
  of two seeds in normalised variables, beside `pdist`. `--eps-table` (D6):
  the binary multiplicative epsilon between every two algorithms per problem,
  on raw non-negative objectives. `--magnitude` (D7): the magnitude of the
  dominated set (Emmerich 2026) against `hv_h` at up to three objectives,
  Kendall's tau per problem and the mean ranks — an experiment on
  `all58_v2`'s final populations, not a metric. `metrics.das_dennis`,
  `r2_discrete`, `magnitude`, `chamfer`, `eps_mult`.
- Multi-parent crossovers (task 2, B2): `spx`, `rex`, `undx` and `pcx`, words
  of the `crossover` knob in NSGA-II, IBEA-ε+, SPEA2+SDE, AGE-MOEA and
  SMS-EMOA (`operators/multi_parent.hpp`). SPX draws uniformly in the simplex
  of n + 1 parents expanded √(n + 2) times about its centre (Tsutsui,
  Yamamura & Higuchi, GECCO 1999, with the expansion rate of Higuchi, Tsutsui
  & Yamamura, PPSN 2000, as in Tanabe & Ishibuchi, GECCO 2019, under which
  the children's covariance is the parents'); REX
  adds Σ ξ_j(x_j − g), ξ_j ~ N(0, 1/n), to the centre g of n + 1 parents
  (Akimoto et al., 2009); UNDX draws the symmetric pair of Ono & Kobayashi
  about the midpoint of two parents, spread α = 0.5 along them and, by a
  third, β = 0.35 across; PCX draws about one of three parents with
  σ_ζ = σ_η = 0.1 (Deb, Anand & Joshi, Evol. Comput. 2002). The pair is the
  algorithm's own; the other parents come from its own mating selection, all
  different (MP-6), and are drawn only when the pair is crossed, with
  probability `pc`. `bound_repair` applies, `reflect` provisionally;
  `resample` is refused. Every algorithm at its defaults returns what it did
  (tools/compat_check.py).
- GDE3, SMS-EMOA and the steady-state MO-CMA-ES (task 2, C3, first queue).
  `gde3` (Kukkonen & Lampinen, CEC 2005): a DE/rand/1/bin trial per member,
  both kept when neither dominates, pruned by non-dominated sorting and a
  crowding distance recomputed after every removal; the constraint-domination
  of the paper (violations compared per constraint); `F` and `CR` knobs,
  0.2/0.2 by default (the paper's setting on its test problems), and
  `bound_repair` for its DE, `reflect` by default. `sms_emoa` (Emmerich, Beume
  & Naujoks, EMO 2005, and Beume, Naujoks & Emmerich, EJOR 2007, with Beume's
  2011 thesis for the adaptive reference point nad + 1, which EJOR uses from
  three objectives; at two the extremes are kept, see Changed): (μ+1), SBX
  η_c = 15 and PM η_m = 20, the least hypervolume contributor of the worst
  front removed (EJOR's basic Reduce, not its "dp" variant). `mo_cma_es`
  (Voß, Hansen & Igel, GECCO 2010, after Igel, Hansen & Roth, Evol. Comput.
  2007): (μ+1) with success meaning "selected", per-individual step size,
  evolution path and covariance matrix, the constants of the 2007 paper at
  λ = 1 that the 2010 paper restates (p_target 2/11, see Changed), penalised
  box constraints (Eq. 5), the search in box-normalised coordinates.
  Hypervolume contributions
  (`include/mootation/hv_contribution.hpp`, `_core.hv_contributions`) are
  exact at two and three objectives and Monte-Carlo, each point in its own
  bounding box, from four. Declared readings in each header. The public list
  is 62 algorithms. Experiment E5 (`python/examples/experiments/
  e5_rotation.toml`): GDE3 at CR = 1 against CR = 0.5, MO-CMA-ES and SMS-EMOA
  on E1's problems and the 110 bbob-biobj ones. All three are in
  `structural_bias.toml`, not in `campaign_all.toml`.
- DMS, Direct MultiSearch (Custódio, Madeira, Vaz & Vicente, SIAM J. Optim.
  21(3), 2011; task 2, C1): `dms`, the first algorithm here that is not
  evolutionary — a list of nondominated points with a step size each, polled
  along ±every coordinate, the step halved where a poll finds nothing. It is
  deterministic (the seed changes nothing; `test_reproducibility` holds it to
  that), stops on its own once every step is below 10⁻³ of the range (a
  `finished()` a core may now declare, which ends the loop early in `run`,
  `Session` and the binding), and answers with its list reduced to
  `pop_size` by DSS — `include/mootation/dss.hpp`, the C++ twin of
  `archive.dss_order`, reachable as `_core.dss_order`. `dms_init` chooses the
  initial list: `line` (the paper's best variant, the default) or `single`.
  Variables are normalised to the box (declared DMS-1). The public list is 59
  algorithms in eight families; DMS is in `structural_bias.toml`, not in
  `campaign_all.toml`, whose all58 runs it postdates.
- `crowding_space` (task 2, C2): NSGA-II's crowding distance over the
  decision variables instead of the objectives (`decision`; default
  `objectives`): one of the two changes of DN-NSGA-II (Liang, Yue & Qu, CEC
  2016), whose niched mating pool is not implemented; the header says which.
- `tools/compat_check.py` reports an algorithm the baseline does not have as
  new instead of changed.
- Operators (task 2, B1-B3): the mutations `gaussian`, `cauchy`,
  `uniform_reset`, `mixture` and `mixture_cauchy`
  (`operators/real_mutation.hpp`), the crossovers `uniform` and `blx_alpha`
  (`operators/real_crossover.hpp`), and DE/rand/2/bin and DE/best/1/bin
  (Storn & Price's DE/x/y/z notation). NSGA-II, IBEA-ε+, SPEA2+SDE and
  AGE-MOEA take `crossover` and `mutation`, MOEA/D-DE `mutation`, with
  `mutation_scale`, `mixture_q` and `blx_alpha`; the defaults are the papers'
  SBX and polynomial mutation, and every algorithm returns bit for bit what
  it did (tools/compat_check.py). `sbx_var_prob` sets SBX's share of crossed
  variables for one run (for experiment E1).
- Structural bias (task 2, A3): `uninformative_n02_2D` and
  `uninformative_n10_2D`, whose objective values are U(0, 1) draws from the
  run's seed and the evaluation's number, independent of x
  (`benchmarks/uninformative.py`; `BenchProblem.make_evaluator` gives each
  campaign run its own stream), `--bias` on the campaign command line (per
  problem and algorithm: a chi-square against uniform over ten bins per
  variable, and the shares near the bounds and in the centre) and
  `python/examples/structural_bias.toml`, every algorithm on both at thirty
  seeds. The registry holds 436 problems in 15 families.
- `bound_repair`: what an operator that can leave the box does with the
  variables it put outside — `clip`, `reflect`, `random`, `midpoint`,
  `resample`, `wrap` or `native` (`operators/bound_repair.hpp`) — a knob of
  the nine algorithms that have such an operator (DE in `moead_de`,
  `moead_dra`, `lis_lcs`; Liu & Li's in `liu_gu2011`, `moead_m2m`,
  `moead_am2m`; `dcea`, `hlmea`, `naemo`), in the binding, the campaign's
  `params` and the settings file. Each keeps the repair it had.
  `DERepair::Clip` / `RandomReset` remain and map onto `clip` / `random`.
- Operator records: every run's `meta.json` lists the variation operators it
  used with their repair (`operators`), and `operator_stats = true` in
  `[campaign]` adds what they did to the trajectory and to `final`:
  `oob_share` and `oob_var_share` (outside the box before repair),
  `survival_share`, `offspring_nd_share` and `step_mean`, relative to the
  parent population (`mootation._core.operator_stats()`,
  `minimize(operator_stats=True)`). Off by default; the populations are the
  same with it on.
- `tau90`, the 0.9 quantile of the IGD+ distances in `igdp_norm`'s
  normalisation, with the coverage curve at 0.01-0.2 beside it, and
  `n_final`, the size of the answer.
- `tools/compat_check.py` and `tests/compat_dump.cpp`: every algorithm at its
  defaults must return, bit for bit, the population it returned at the
  baseline commit in `tests/compat_baseline.txt`; the driver is compiled
  against both trees with the same compiler, and the CI job "default
  behaviour, bit for bit" runs it. `tests/test_bound_repair.cpp` checks the
  repairs themselves.

- The analysis layer of a campaign (`mootation.run.report`, `stats`,
  `postprocess`), on the campaign command line:
  `--reference ALG` tests every algorithm against a reference — the exact
  Wilcoxon rank-sum test per problem, Holm-corrected over the problems, with
  the Vargha-Delaney A12, and the exact signed-rank test on the per-problem
  medians across problems, Holm-corrected over the algorithms (both exact with
  ties, by dynamic programming over doubled ranks); `--ci` gives 95 %
  bootstrap intervals of the mean ranks; `--by KEY` ranks within groups of
  problems sharing a property; `--gap METRIC` prints the distance to the best
  value any run reached; `--zero-share` the share of seeds whose hypervolume
  is 0; `--ecdf METRIC` the COCO-style runtime ECDF, with the interpolation
  between records stated (`--interpolation step|linear`). Every table marks
  with `*` the sixteen algorithms that run on a schedule of the budget share
  spent, whose curves at different budgets must be overlaid by fraction.
- `mootation.benchmarks.properties`: front geometry, multimodality,
  deception, bias, scaled ranges, separability and a medial optimum for every
  problem campaign_all runs — Huband et al.'s Tables V, VII and XV for ZDT,
  DTLZ and WFG, argued in the module for IDTLZ, SDTLZ, shiftDTLZ, ZCAT and
  bbob-biobj; `?` where the source leaves it open.
- The archive scenario: every run with an archive also stores
  `final_archive`, the final indicators on the run archive reduced to the
  population size by DSS, and `--scenario archive` makes every table read it.
  `--recompute ... --scenario archive` rebuilds it from `archive.csv`, for
  campaigns run before it existed too; `meta.json` records the frame the
  selection normalised by, so the rebuild picks the same points.
- Two ablation baselines: `random_selection_ea`, NSGA-II's SBX and polynomial
  mutation with uniformly random parents and survival, and `gsemo`, global
  SEMO (Laumanns, Thiele & Zitzler 2004; Giel 2003) carried to real variables.
  Both are in `campaign_all.toml`.
- Pareto-set samples (`BenchProblem.pareto_set`) for Polygon, DTLZ1-4,
  shiftDTLZ1-4 (with `cyclic_vars`, the distance variables whose differences
  wrap) and ZCAT1-20 (the positions with every distance variable at g(y_I)),
  and three decision-space indicators on the final population: `igdx`
  (Tanabe & Ishibuchi 2019, Eq. 5, variables normalised by the bounds, cyclic
  ones wrapped), `cr` (the cover rate, Eqs. 7-8) and `pdist` (the mean
  pairwise distance, no reference needed).
- The exact hypervolume in C++: `mootation/hypervolume.hpp` (WFG, While,
  Bradstreet & Barone 2012, with sorting and slicing) and
  `mootation._core.hypervolume`, which `metrics.hypervolume` uses when the
  extension is built — about 20 ms for 210 points at five objectives, where
  the Python recursion took seconds. The Monte-Carlo estimate draws its points
  in NumPy and counts them in C++ (`_core.hv_covered`), so it is the same
  number either way. `tests/test_hypervolume.cpp` checks the exact value
  against inclusion-exclusion on 300 random sets.
- Hypervolume options in `[campaign]`: `hv_exact_max_m` (5) and
  `hv_mc_samples` (100 000) for the final value, `trajectory_hv_mc_samples` for
  a Monte-Carlo hypervolume on the trajectory above `trajectory_hv_max_m`
  instead of null.
- Variants: `[[algorithms]]` takes `label`, which names a variant and the
  directory its results are filed under, so two entries of one core with
  different parameters can share a campaign; and `evaluations`, a budget in
  evaluations that replaces `pop` x `gens`. Parameters can be booleans.
- `normalize` as a knob (`set_normalize` of `ibea_eplus`, `r2ibea`,
  `two_arch2`, `moead_am2m`) in the Python binding, the campaign's `params`
  and the settings file; each header says which setting is the paper's.
- `max_evaluations` in the C++ `Settings`, and so in `run()`, `Session`, the
  settings file and the C ABI: a budget in evaluations instead of `max_gen`
  steps, checked between steps, with the schedule's t_max corrected after the
  first step to the number of steps the budget buys.
- Provenance: every campaign run's `meta.json` records `revision` — the git
  commit and whether tracked files differed from it, for the source tree and
  for the compiled extension separately (`_core.__git_commit__`,
  `_core.__git_dirty__`, stamped by `python/git_stamp.cmake` on every build).
  The version string stays 0.1.0 across commits, so it could not tell a run
  made before a fix from one made after it; the pair also shows a checkout
  whose C++ was never rebuilt after a pull.
- New indicators: `gdp` (GD+, convergence only), `roi_dist` (distance to the
  box [ideal, nadir], which separates runs whose hypervolume is 0 and needs no
  reference front — the COCO bbob-biobj convention), `range_cover` (the worst
  objective's covered share of [ideal, nadir], per objective in
  `range_cover_each`), `nd_share` and `dup_share`.
- `--problems NAMES` on the campaign command line narrows `--list`, the run
  (with `--shard` and `--force`), the tables and `--recompute` to part of the
  selection: rerunning only the problems whose reference changed is
  `--problems ... --force`.
- Campaign recording: `record_grid = "log"` records at the fixed evaluation
  counts round(10^(j/10)) — the same at every budget, so the 10 000 / 25 000 /
  50 000 ladder compares point by point instead of through interpolation;
  `trajectory_hv_max_m` keeps the hypervolume on the trajectory only where it
  is cheap; `snapshots` (true or a list of problems) stores the population's
  objectives at every record in `snapshots.npz`.
- The run archive: every campaign run writes `archive.csv`, every
  nondominated point it evaluated whatever the algorithm kept, at most one per
  cell of a grid in normalized objectives (1e-3, or 1e-2 from five
  objectives), each objective's best point kept outside the grid so the ends
  of the front are never pruned (`mootation.run.archive.GridArchive`).
  `dss_order` selects N points of any set — distance-based subset selection
  on the IGD+ distance, indicator-neutral — and its order is incremental, so a
  set stored in that order is thinned by slicing.
- Baselines `random_search` and `sobol_search` (scrambled Sobol, SciPy):
  blind sampling with the run archive, answered by DSS at the problem's
  population size, spending the budget exactly. They go in the algorithm list
  like any core but are not C++ cores (`mootation.run.baselines`).

- bbob-biobj F1-F55 at 5 and 10 variables, the suite of Brockhoff, Auger,
  Hansen & Tusar (Evolutionary Computation 30(2):165-193, 2022), on top of the
  ten single-objective bbob functions of Hansen et al. (INRIA RR-6829, 2009
  definitions with the 2019 errata) in a new `benchmarks/bbob.py` — 110
  problems, and the registry is now 434 across 14 families. Each F is a pair
  of two bbob functions, two from each of the five bbob groups, giving
  C(11,2) = 55 pairs including the diagonal. Written from the papers.
  These are the first problems here with **no reference front**, which is the
  suite's nature rather than an omission: a pair of bbob functions has no
  closed-form Pareto set, and COCO itself estimates each instance's
  hypervolume from accumulated experiments. `pareto_front` is therefore None
  and IGD / IGD+ / ε report nothing on them instead of a number measured
  against an invented front; the hypervolume works, because the ideal and the
  nadir are exact — every base function's unique optimum is known by
  construction. Declared in the module headers: the 2019 errata arrive in the
  converted report as "A --> B" with the colour lost (read as B replacing A);
  f20's printed 2|x̂ᵒᵖᵗ| must be 2|xᵒᵖᵗ|, since only that puts the optimum at
  z = 420.96874633 where Schwefel's is; f17 lost a parenthesis; and the nadir
  formula in the bi-objective paper prints what is actually the ideal point,
  so its own defining sentence is followed instead. **The instances are ours.**
  The papers give the distributions an instance is drawn from but not the
  generator that turns an instance number into them, which exists only in
  COCO's code, so instance 1 here is not instance 1 in the COCO archives and
  results are not comparable with them.
- ZCAT1-20 at M = 2, 3, 5 and 10, the suite of Zapotecas-Martínez, Coello
  Coello, Aguirre & Tanaka (Swarm and Evolutionary Computation 81, 2023,
  101350) — 80 problems.
  Every problem is f_i = α_i(y_I) + β_i(y_II − g(y_I|m)), so the front (α) and
  the Pareto set (g, one of eleven topologies) are chosen independently and the
  difficulty of the distance term is a dial rather than a property of the
  problem: six levels, a bias, an imbalance between objectives, degenerate and
  region-switching fronts. Registered with the paper's own defaults (n = 10M,
  Level 1, complicated Pareto set, no bias, no imbalance) and its analytic
  ideal (0, ..., 0) and nadir (1, 4, ..., M²); the other dials are arguments of
  `mootation.benchmarks.zcat.evaluate`. Written from the paper and its
  supplement — the authors' reference implementation is GPL-3.0 and was not
  read or ported. The five points the paper leaves open (the missing absolute
  values in Z₂ and Z₄, the undefined constant A of ZCAT19, which m the topology
  is called with, the free index in ZCAT17/18's degenerate region, and whether
  exp(μ)⁸ means e^(8μ)) are resolved and declared in the module header.
- Campaign indicators: `igdp_norm` (IGD+ in nadir − ideal units, scale-free),
  `eps` and `eps_norm` (additive epsilon), and `hv_h` (hypervolume with the
  reference point at 1 + 1/H of the problem's default population, as
  Ishibuchi et al. propose). `final_metrics` computes its own list once on the
  final population, so the hypervolume at five objectives is not paid for at
  every trajectory point. `--at FRACTION` reads every run at that fraction of
  its budget for `--compare` and `--ranks`, and `--recompute` adds indicators
  to finished runs from their `final.csv` without rerunning them.
- The TUI can run a campaign. A campaign config opens on a dashboard —
  progress, speed and time left, the jobs in flight, every algorithm grouped
  by family with its own progress bar, every problem — with Start, Stop and a
  worker count that Apply changes while the campaign runs. The results are
  scanned in a background thread and a finished job is never re-read. The tab
  it replaces re-read every meta.json on the UI thread every three seconds and
  rebuilt a problems × algorithms table, which on a campaign of thousands of
  jobs is what made the interface lag.
- A running campaign follows `<results>/_workers.txt`, re-read every 1.5 s:
  more workers start at once, fewer let the surplus finish the job it is on,
  0 drains the campaign and stops it. `_runner.json` is its heartbeat. A
  worker that dies inside a core fails its one job and is replaced instead of
  leaving the pool a worker short, and each worker reads the config once
  instead of once per job.
- `algorithm_families()` in `mootation.run.algorithms`, read from the section
  comments of `algorithms.def`.
- `--ranks METRIC` on `mootation.run.campaign`, and the same view in the TUI's
  Compare tab (`t` switches between it and the medians, `e` exports either):
  each algorithm's rank on every problem by its median indicator, averaged
  overall, per family and per objective count, with the problems it won. A
  medians table 58 columns wide cannot be read; this is the view that says
  which algorithm is good where. `--compare` and `--ranks` reject an unknown
  metric instead of printing an empty table.
- `shiftDTLZ1`–`shiftDTLZ4` in `mootation.benchmarks`, at every size DTLZ has:
  DTLZ1-4 with the optimum of every distance variable moved off the centre of
  the box, where DTLZ puts it and where an operator or an initialisation that
  drifts toward the middle finds it for free. Each distance variable is read
  through a cyclic shift, so the landscape stays continuous and the fronts,
  ideal and nadir are DTLZ's; a large gap between an algorithm's DTLZ and
  shiftDTLZ results is centre bias. `campaign_all.toml` runs them at 3 and 5
  objectives, at the end of its problem list so existing job numbers hold.
- `python/examples/campaign_all.toml`: all 58 algorithms on 53 problems (ZDT;
  DTLZ1-7, WFG1-9, IDTLZ1-2, SDTLZ1-2 and shiftDTLZ1-4 at 3 and 5 objectives)
  with measured run times, and a recipe in `docs/running.md` for running it on
  another machine, or split across several.
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

- WFG1-9 without numba. `benchmarks/wfg.py` kept a plain-NumPy copy of every
  function for machines without numba, and its shape functions took one factor
  too few in the middle objectives — `prod(s[:M-2-m]) * c[M-1-m]` where Huband
  et al.'s Table X multiplies M − m sines, likewise for the convex and linear
  shapes — so f_2 … f_{M−1} were wrong from three objectives on (by up to 18 at
  ten objectives), while f_1, f_M and every value at two
  objectives were right. The copies are gone: every WFG function runs the
  kernels, compiled by numba where it is installed and interpreted where it is
  not, at the same cost. A new test holds both paths — the second in a child
  process with `NUMBA_DISABLE_JIT=1` — to pymoo 0.6.2's values at 36 points,
  WFG1-9 at 2, 3, 5 and 10 objectives (`tests/data/`), and runs in CI, where
  there is no numba. WFG values computed without numba before this fix are wrong from three
  objectives on. Every campaign run's `meta.json` now records `numba`.
- Clipping past the out-of-box counts (`oob_share` = 0 proved nothing where
  it happened): EMyO/C's offspring (`emyo_c_difference`), HCCA's DE/rand/1
  (`hcca_de_rand_1`, which was missing from the operator log altogether) and
  MaOEA-IAMD's noise (`maoea_iamd_gaussian`, clamped by the store) are now
  repaired, counted and logged where they leave the box, to the same values
  as before; and the store counts a clamp of its own as a repair, `vault` in
  the operator log, so that none can go unseen again. A run of every
  algorithm finds none left for the store.
- `sobol_search`'s header called SciPy's scrambling Owen-type; it is a linear
  matrix scramble with a digital shift (Matoušek), not Owen's nested one.
- The tables read a budget below the end (`--at`) the way `--cover` does, by
  one function (`cover.read_at`): the first trajectory record at or after the
  budget asked for (`budget_nominal`). `value_at` took the last record at or
  below `at × budget_fe`, and `budget_fe` is rounded up to whole generations:
  at a population of 91 "2 500" was read at 2 002 evaluations, while the
  budget-dependent algorithms' rungs had spent 2 548. The tables at 10 % of
  the budget therefore put the ordinary algorithms a fifth of a budget
  behind. A run that stopped before the budget (DMS, once
  its steps are below tolerance) is now read at its end, in `--cover` too,
  instead of having no value there; a rung is found by its nominal budget;
  and `report._value` (`--ci`, `--by`, `--reference`) reads the archive
  scenario at `--at` too, where it used to take the end.
- The IPolygon header claimed every point inside the two polygons is
  Pareto-optimal. Ishibuchi, Akedo & Nojima (2011, Section 2) state that only
  for polygons "not too close", and the registry's (radius 20, centres 40
  apart) are: at three objectives the Pareto set is a region between the
  triangles and 862 of the 1 000 reference points are dominated; at four only
  the squares' inner halves are Pareto-optimal (f2 = f4 there), and 523 of the
  reference points and half of `pareto_set`'s are dominated. The IPolygon
  values of IGD+, IGDX and PSP measured on that instance are not to be
  trusted; it is replaced in reference version 2 (see Changed). The Polygon
  header said the front is (M-1)-dimensional; it is 2-dimensional whatever M,
  degenerate from four objectives, as MaF8's.
- `--cover` and `--at` lost a budget-dependent algorithm on every problem
  whose population does not divide the budget: a run's `budget_fe` rounds the
  budget up to whole generations (25 025 at a population of 91), so `--cover`
  found no value at 25 000 and reported the full run as a missing rung, and
  `--at 0.1` looked for a rung of 2 502 evaluations and read the full run's
  trajectory instead of the 2 500 rung. Both now count from the budget asked
  for, `budget_nominal` in `meta.json` (new test at a population of 91).
- `sms_m2m` spent hours to days on a run at five and six objectives: its own
  hypervolume slicing carried every point of a slice down to the next
  objective, and each contribution was HV(F) − HV(F\{s}), n + 1 volumes for
  every removal. It now takes the contributions from `hv_contribution.hpp`,
  as SMS-EMOA and MO-CMA-ES do, exact at every M (SMSM2M-10). A 25 000-
  evaluation run on DTLZ3 with five objectives: 38-42 min -> 14 s; the whole
  algorithm in the stage-3 campaign: about 23 000 CPU-hours -> 10. The
  contributions are the same up to rounding, the default fingerprints did
  not move, and nine of ten finished probe runs returned the same population
  (the tenth parted in its last generation over points with objectives near
  1e-15, where the old difference of two volumes had lost the digits).

- A campaign's workers ran out of memory on a many-core Windows machine:
  importing NumPy made OpenBLAS reserve a buffer for every core in every
  process (756 MB each at 24 threads), and the stage-3 probe's twenty workers
  reached the commit limit of 31 GB of RAM plus page file — a MemoryError on
  a 5 MB array. The campaign pool, `--recompute`, `--cover` and the budget
  check now start their workers with one BLAS thread (17 MB), unless
  `OPENBLAS_NUM_THREADS`, `OMP_NUM_THREADS` or `MKL_NUM_THREADS` is set.

- README's family table (English and Russian) still added up to 58
  algorithms: Pareto-dominance 7 (GDE3 makes 8),
  indicator-based 11 (SMS-EMOA and MO-CMA-ES make 13), and no row for DMS.
  `rvea.hpp` said in one place that RVEA's population "recovers later" after
  a generation with empty subpopulations and, a few lines below, measured,
  that on an irregular front it stays below N for good; it now says the
  latter throughout.

- `test_constraints` decided its feasibility check — the constrained run ends
  with at least as many feasible solutions as the unconstrained one — on one
  seed, and under MSVC crEA's run on that seed fell short although its
  constraint handling works: the seed was unlucky. A regression on the
  suite's seed is now measured again on four more seeds and the mean over
  the five decides. The algorithms that pass on the
  suite's seed cost no more time; RVEA stays exempt.

- `MOOTATION_DEFINE_PROBLEM` tested its compile-time constraint count with a
  plain `if`, which MSVC 14.44 at `/W4 /WX` rejects as a constant condition
  (C4127), so `examples/benchmark_problem.cpp` did not build there; it is
  `if constexpr` now. The `crossover` and `mutation` knobs were documented
  for four algorithms (`settings.hpp`, the binding, docs/embedding.md);
  SMS-EMOA takes both as well.

- Headers and docstrings checked against the papers that were missing
  (Igel, Hansen & Roth 2007; Beume, Naujoks & Emmerich 2007; Deb, Anand &
  Joshi 2002; Higuchi, Tsutsui & Yamamura 2000; Eshelman & Schaffer 1993;
  Liang, Yue & Qu 2016; Singh, Bhattacharjee & Ray 2019; Ishibuchi et al.
  2015 and 2016; Brockhoff, Wagner & Trautmann 2015); no code changed.
  SMS-1 called EJOR's "dp" variant the EJOR version: EJOR has both, and the
  basic Reduce is the one implemented. MO-CMA-ES: the initial state is the 2007
  paper's. DSS departs from Singh et al.'s Algo. 1 in
  three ways (the M per-objective minima as seeds, the d+ distance, no
  filtering), now said in dss.hpp and archive.py, where "indicator-neutral"
  overstated it. GD+ is not Pareto-compliant, not even weakly, and the
  metric's description says so. `crowding_space = decision` is one of
  DN-NSGA-II's two changes, not DN-NSGA-II.

- `poly_mutation.hpp` cited Deb & Deb 2014 at "pp. 177-178"; the article is
  Int. J. AISC 4(1), pp. 1-28, and the settings are in its §5. `sbx.hpp`
  claimed that crossing every variable is "empirically stronger" on
  MaF3/DTLZ3; that was seen outside this repository and is now marked as
  unverified until experiment E1.

- The ZCAT reference fronts contained dominated points. The sampler kept the
  points of its own candidate set that no other candidate dominated, which is
  not Pareto-optimality: next to the gaps of a disconnected front a candidate
  the sample happened not to beat is beaten by a position vector it did not
  contain. Against 50 000 fresh points, 24 of ZCAT11_3D's 1000 reference
  points were dominated, 164 of ZCAT11_5D's (by up to 1.48 in f_5, whose span
  is 25), 98 of ZCAT12_5D's and 66 of ZCAT13_5D's (by up to 3.72). Every point
  is now checked against an independent sample and its own neighbours before
  the thinning, from a generator of its own, so the fronts in which nothing
  was dropped are bit-identical to what they were: 33 of the 40 in the
  campaign; ZCAT11-13 at three and five objectives and ZCAT16_3D changed. A
  fresh sample still finds 1 of ZCAT12_5D's points (by 0.37) and 2 of
  ZCAT13_5D's (0.27), where it found 98 and 66. IGD, IGD+, GD+ and ε change on
  the problems that changed; the frame, the paper's, does not.
- The same defect in the registry's own samplers for DTLZ7 (and MaF7), WFG2
  (and MaF11) and ZDT3, which take the nondominated images of a grid: 150 of
  WFG2_3D's 416 reference points were dominated by fresh front points (by up
  to 0.38), 256 of WFG2_4D's 700 (1.13), 140 of WFG2_10D's 844 (11.7); they
  get the same check (WFG1, whose front is connected, loses nothing). DTLZ7
  was worse: its grid ran over the cube, but the Pareto set is the product of
  the intervals [0, 0.2514] u [0.6316, 0.8594] — the left records of
  x (1 + sin 3 pi x) — so from eight objectives the grid points were almost
  all outside it (all but one of DTLZ7_15D's 16 384, whose grid was {0, 1}^14).
  The grid now runs over that set, with random draws from it where a grid
  would be its corners alone. The frame moves with it wherever a
  non-Pareto-optimal point had set it: the nadir of x_i is 0.8594, not 1, and
  the ideal of f_M is 2M − (M − 1) 1.6930 (3.228 at five objectives, where
  the grid had given 3.757). Thirteen frames changed (DTLZ7 at 2-6, 10 and 15,
  MaF7 at 3, 5, 8, 10 and 15, ZDT3 by 5e-4); WFG2's did not.
- AGE-MOEA, two corner cases of the geometry estimate. When the normalized
  central point's coordinates summed to M or more — which §3.3 says cannot
  happen, but the hyperplane normalization allows on irregular fronts — Eq. 8
  has no value and the guard drove p to its cap of 20; it is now p = 1, the
  answer the same function already gave when no central point existed (AGE-2).
  When one solution was the extreme of every objective, the second-minimum
  distance stayed at DBL_MAX and the first candidate in index order won with a
  score ranking it among the extremes (AGE-3). Neither case occurs on the
  campaign's problems at 10 000 evaluations: all 203, and 12 of them over five
  seeds, gave bit-identical results before and after.
- SPEA2 and SPEA2+SDE returned the wrong set. Their answer is the archive
  (Algorithm 1, Step 4), but the archive lived only in the vault's archive
  slots, and everything that reads a result — `minimize()`, the binding, `run()`
  and campaigns — reads the active population, which held the freshly bred
  offspring that no selection had seen. The archive is now copied into the
  active slots after `setup()` and every `step()`. Campaign results for both
  were measured on the offspring and need rerunning.
- RVEA's reference vector adaptation (Algorithm 3, lines 5-7) rescaled only as
  many vectors as there were survivors instead of all N, so whenever the
  population fell short of N the vectors at the end of the list never followed
  the front.
- MOEA/DD associated each offspring with its subregion once, against the
  ideal point of that moment, and never again, so the niche counts it compares
  mixed subregions measured from different ideal points. The population is now
  re-associated whenever an offspring moves the ideal point.
- NRV-MOEA's union of archive and population kept a solution twice when it
  was in both — every member at the first generation, since the archive starts
  as a copy of the population — and copies never dominate each other, so
  reproduction drew such a solution twice as often. The union now holds each
  solution once.
- A-NSGA-III's size cap on the reference set (an extension beyond the paper)
  trimmed the newest points first without looking at them, so it could drop a
  point that had just attracted a member while an empty one stayed. It now
  drops empty added points first.
- MOEA/D-DRA's utility could go negative and then climb. Step 5 multiplies
  π by 0.95 + 0.05·Δ/0.001, which is negative once Δ < −0.019, and Δ gets
  there: it compares scalarized values across a moving ideal point. A
  negative π then rose on every later update without an improvement, where
  the paper says it "will be reduced". The factor is floored at 0. MOEA/D-AWA
  takes this step from MOEA/D-DRA and gets the same floor.
- MOEA/D-AWA's reallocation (Algorithm 2, Step 1) gave each subproblem the
  best solution for its weight while reading from the population it was
  overwriting, so a slot already replaced could be copied again. It reads
  from a copy taken before the step.
- Algorithm headers that misdescribed the code or the paper, found by
  checking them against the papers: MOEA/D-AWA's note on corner subproblems
  (the code was right), a claim in MOEA/D-M2M's (withdrawn), and in the
  benchmark registry the conventions (no
  unbounded archive is kept; `K_runs` is 21) and the MaF attribution.
  Readings the papers leave open are now declared: MOEA/D's zero weights and
  result set, DEA-GNG's tie-breaking
  and sub-network expansion, the last generation of the Liu–Li operator,
  CLIA's reference-point relearning, HypE's reference point, and notes in
  CA-MOEA, VaEA, AR-MOEA, MOMBI-II, AdaW and Two_Arch2.
- Campaign budgets were counted in steps, and a step is not a generation for
  every core: NIMMO evaluates one offspring per step and MOEA/D-DRA and
  MOEA/D-AWA a fifth of the population, so at the same budget they spent 2 %
  and 21 % of what the other 55 algorithms did, and their campaign ranks meant
  nothing. It showed up by comparing the `fe` field of a finished campaign with
  its budget. `minimize()` and the binding's `Config` take `max_evaluations` —
  the run stops between steps once that many evaluations are spent, and
  `Result.evaluations` reports the count — and every campaign job runs on it.
  Schedules that anneal over `t_max` get the number of steps the budget
  actually buys, measured after the first step. The C++ `run()` / `Session`
  and the TOML runs of external solvers still count generations.
- A campaign on Windows died on the first collision between a worker writing a
  `meta.json` and the TUI or `--list` reading it. Windows refuses to replace a
  file another process has open, the error was not caught, and one collision
  stopped a campaign 8 400 jobs in. The write now retries for a few seconds,
  and an error outside a job's own run fails that job instead of the pool; the
  job reruns when the campaign is started again.
- A campaign on Windows could hang for good. The pool's workers shared two
  queues, whose semaphores are handed to a worker process while it is still
  starting, and on a loaded machine they could arrive dead in every worker of
  a run: each worker took a job, failed to report it ("The handle is invalid"
  or "Access is denied"), and died, and the runner waited for jobs nobody was
  running. It showed up in the test suite while the machine was compiling.
  Each worker now has its own pipe and asks the runner for one job at a time,
  so no semaphore crosses to it, and the runner always knows which job a
  worker holds: a worker that dies holding one fails exactly that job.
- Every Linux and macOS build, the C ABI job and the single-header check had
  been failing in CI since the binary-genome refusals were added. In HCCA the
  refusal shared a line with the two calls after it, which GCC and Clang flag
  as `-Wmisleading-indentation`, and CI builds with `-Werror`. Behaviour was
  unaffected — the `if` guarded only the `throw`, as intended — and MSVC does
  not warn, which is why the Windows jobs and local test runs stayed green.
  The bodies are reformatted, and an `-O3 -Wall -Wextra -Wpedantic` pass over
  every translation unit the CI builds is clean with GCC 15.
- A campaign job could spend more than ten minutes inside one hypervolume.
  Equal rows do not dominate each other, so the nondominated filter kept every
  copy, and the WFG recursion branches on each of them. The MOEA/D-M2M family
  carries copies in its population for the whole run: at generation 0 on
  WFG4 with five objectives a MOEA/D-AM2M population is 126 rows with 11
  distinct, and that single call never finished. The filter keeps each
  distinct row once. The values are unchanged — checked against the old code
  on eleven final populations at 3 and 5 objectives — the set above takes
  4 ms, and an ordinary well-spread 126-point set at five objectives still
  takes two to three seconds, which is why the shipped `campaign.toml` now
  records a trajectory point every 10 generations instead of every one.
- Every campaign job that set an integer parameter failed. The TOML layer
  turned every `params` value into a float; `T`, `nr`, `K`, `n_clusters` and
  `div` are integers in the binding, and pybind11 3 refuses 20.0 for an int.
  The shipped `campaign.toml` sets `T = 20` and `nr = 2` for MOEA/D-DE, so on
  a fresh install every one of its MOEA/D-DE jobs failed. TOML integers now
  stay integers, and `minimize()` takes a whole float such as `T=20.0` as the
  integer it means and names the parameter when given a fractional one.
- `pip install .` produced a package whose run layer could not start. The TOML
  layer reads the algorithm list out of `algorithms.def` and the knob list out
  of `settings.hpp`, and neither was in the wheel, so outside a checkout every
  campaign and run file failed validation with "cannot find
  include/mootation/algorithms.def", and the knob list silently fell back to a
  snapshot. Both files are now installed into the package, and the CI job
  that installs it exercises the run layer from outside the source tree.
- **Scale invariance.** A new test (`tests/test_scale_invariance.cpp`) runs
  every algorithm on DTLZ2 and on the same problem with every objective
  multiplied by 2^10, a power of two so the scaling is exact and a
  scale-equivariant algorithm has to reproduce its result bit for bit. Those
  that did not:
  - `two_arch2` computed I_eps+ on raw objectives, so its fitness
    `-exp(-I/0.05)` underflowed to exactly zero for every pair once the
    objectives grew, and the convergence archive stopped ordering anything.
    The paper is silent on scaling while the IBEA fitness it reuses verbatim
    does scale (Alg.2 steps 2.1-2.2), so the scaled reading is now the
    default; `set_normalize(false)` restores Eq.1.
  - `ar_moea` initialised the adapted reference set R' from the unit-simplex
    points while consuming it in translated objective coordinates, so the
    first generation compared a simplex against objectives of whatever
    magnitude the problem used. R' is now built in the coordinates it is used
    in.
  - `r2ibea` keeps the paper's letter: Eq.4 is written on raw objectives.
    `set_normalize(true)` transports the whole of Eq.5 into normalised
    coordinates — every normalised range is 1, so z* is -1 on every axis — and
    makes the run bit-identical at every scale factor. It is not the default:
    Eq.5's single shift by the largest range is exactly
    what makes an improvement on a narrow axis worth less than one on a wide
    axis, and normalising per axis discards that. The first version of this
    switch was wrong — it divided by the per-axis range but left z* shifted by
    the largest one, leaving a constant `max_range/range_j` inside the
    Tchebycheff max that was largest on the NARROWEST axis, inverting the skew
    instead of removing it. The underflow of `exp(-I_R2/kappa)` is detected at run
    time and reported through `set_warn_handler`, on the FRACTION of dead
    pairs over the population rather than on all of them being dead: a total
    underflow is obvious from the output, a partial one looks like a working
    run and is not. The rule is what the spread of ranges MEANS — units, and
    scaling helps; the problem, and it throws information away. On ZDT1 the
    wide axis is the direction of convergence.
- `ibea_eplus` gains `set_normalize(false)`, an experiment switch only: the
  paper's own Alg.2 scaling stays the default. R2 allocates effort by direction and
  rescaling the axes moves which piece of the front each weight vector owns;
  IBEA has no such partition, it only decides an order. The caution therefore
  belongs to the MOEA/D and NSGA-III families, RVEA and θ-DEA rather than to
  indicator methods. Which of the two cases
  you are in has a cheap test, now written up: the ratio between the
  per-objective ranges of the answer set holds steady when the spread is a
  unit and slides toward 1 when it is the problem. Freezing the divisor at
  the first pool's ranges is recorded in the header as rejected: a frozen
  divisor freezes the resolution as well.
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
- Algorithms, from a second paper-fidelity pass plus a convergence audit:
  MOMBI-II (the |.| in the ASF froze ZDT1), MOEA/D-AM2M (frozen normalisation
  stalled ZDT1), IF-MaOEA (reference
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
  and puts the letter behind `set_ideal_shift(false)`); MOEA/D-M2M, SMS-M2M,
  MOEA/D-AM2M, APRD and Liu & Gu 2011
  draw mates once (self allowed) instead of rejection loops and top up without
  replacement; MOEA/D-AM2M builds the exact two equal-layer lattices of §V-C
  (110 = 55 + 55) instead of a silently truncated 220-point one; A-NSGA-III
  deletes reference points only in the paper's "ideal scenario" with a
  one-generation cooldown (`set_adaptation_mode`); APRD truncates its archive
  by angle, not Euclidean distance; NRV-MOEA sets P <- P_c as Alg.1 line 15
  says (`set_population_pruning(true)` restores the old behaviour); DEA-GNG
  increments g after selection; SRV inserts the
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
