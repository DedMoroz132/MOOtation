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
  **Two_Arch2** returns 7, 5, 26, 7 and 51 solutions on ZDT2 over seeds 1-5 at
  10 000 evaluations: the diversity archive admits only nondominated points,
  and the search has collapsed to x1 = 0 in three of them — as it does for
  IBEA (largest f1 0.105 and 0.25 in two seeds) and NSGA-II (0.064) on the
  same budget. The archive's selection never ran and its output is §III-A's.
  **RVEA** keeps one solution per non-empty reference vector (§III-C, Alg. 2);
  on IDTLZ1_5D only 21 of the 126 vectors are the nearest to any point of the
  front, so the population stays at 8-27. **LIS/LCS** breeds N − K offspring
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

### Added

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
  active slots after `setup()` and every `step()`. Median IGD at 30 000
  evaluations, before → after: SPEA2 (3 seeds) DTLZ2 0.0788 → 0.0579, ZDT1
  0.0105 → 0.0040, DTLZ7 0.097 → 0.068, inverted DTLZ1 0.0437 → 0.0220, WFG4
  0.303 → 0.237; SPEA2+SDE (11 seeds) DTLZ2 0.085 → 0.077, ZDT1
  0.0115 → 0.0043, DTLZ7 0.082 → 0.063, inverted DTLZ1 0.0427 → 0.0218, but
  WFG4 0.315 → 0.326 (worse on 9 seeds) and scaled DTLZ2 2.53 → 2.96 (worse on
  10). Campaign results for both were measured on the offspring and need
  rerunning.
- RVEA's reference vector adaptation (Algorithm 3, lines 5-7) rescaled only as
  many vectors as there were survivors instead of all N, so whenever the
  population fell short of N the vectors at the end of the list never followed
  the front. ZDT1 0.0228 → 0.0202 and DTLZ7 0.148 → 0.103; DTLZ2, scaled
  DTLZ2, inverted DTLZ1 and WFG4 are unchanged.
- MOEA/DD associated each offspring with its subregion once, against the
  ideal point of that moment, and never again, so the niche counts it compares
  mixed subregions measured from different ideal points. The population is now
  re-associated whenever an offspring moves the ideal point. ZDT1 0.0052 →
  0.0040, inverted DTLZ1 0.056 → 0.039, DTLZ7 0.129 → 0.121; DTLZ2, scaled
  DTLZ2 and WFG4 are unchanged.
- NRV-MOEA's union of archive and population kept a solution twice when it
  was in both — every member at the first generation, since the archive starts
  as a copy of the population — and copies never dominate each other, so
  reproduction drew such a solution twice as often. The union now holds each
  solution once. No clear change in the results: DTLZ7 0.060 → 0.065, WFG4
  0.243 → 0.238, the other four within 1 %.
- A-NSGA-III's size cap on the reference set (an extension beyond the paper)
  trimmed the newest points first without looking at them, so it could drop a
  point that had just attracted a member while an empty one stayed. It now
  drops empty added points first. DTLZ2 0.0552 → 0.0542, inverted DTLZ1
  0.0230 → 0.0224, DTLZ7 0.069 → 0.071.
- MOEA/D-DRA's utility could go negative and then climb. Step 5 multiplies
  π by 0.95 + 0.05·Δ/0.001, which is negative once Δ < −0.019, and Δ gets
  there: it compares scalarized values across a moving ideal point. A
  negative π then rose on every later update without an improvement, where
  the paper says it "will be reduced". The factor is floored at 0. Median IGD
  over 11 seeds at 30 000 evaluations: ZDT1 0.259 → 0.209 (lower on every
  seed), WFG4 0.389 → 0.381; DTLZ2, DTLZ7 and scaled DTLZ2 stay within the
  seed scatter, and inverted DTLZ1 is bimodal either way. MOEA/D-AWA takes
  this step from MOEA/D-DRA and gets the same floor, which there changes
  nothing outside the seed scatter on the same six problems.
- MOEA/D-AWA's reallocation (Algorithm 2, Step 1) gave each subproblem the
  best solution for its weight while reading from the population it was
  overwriting, so a slot already replaced could be copied again. It reads
  from a copy taken before the step. The six measured problems are unchanged
  except DTLZ7, which moves within the seed scatter (0.103 → 0.107).
- Algorithm headers that misdescribed the code or the paper, found by
  checking a comparison with PlatEMO against the papers: MOEA/D-AWA's note
  on corner subproblems (the code was right), MOEA/D-M2M's claim about what
  PlatEMO does (withdrawn), and in the benchmark registry the conventions (no
  unbounded archive is kept; `K_runs` is 21) and the MaF attribution.
  Readings the papers leave open are now declared, with a measurement where
  one was taken: MOEA/D's zero weights and result set, DEA-GNG's tie-breaking
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
