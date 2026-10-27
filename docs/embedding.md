<!-- SPDX-License-Identifier: Apache-2.0 -->
# Embedding MOOtation in your own program

Three shapes, from the most static to the most portable: the algorithm fixed at
compile time (`Optimizer<Ind, Core>`), the algorithm chosen by name at run time
with either the library or you owning the loop (`embed.hpp`), and the same loop
behind a C ABI for other languages. All three run the same 76 cores from
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
you. The registry's ZDT, DTLZ, MaF, WFG, ZCAT, bbob-biobj, DAS-CMOP, LIR-CMOP,
MW, DOC, Polygon, multimodal (SYM-PART, Omni-test, MMF), RE/CRE, BT and MOP
families and their variants are in the C++ suite (`include/mootation/suite`, `suite::make(spec)`;
`benchmarks/native.py` writes each problem's spec); the other families live on
the Python side. A binary or mixed
genome is the same interface with `get_bin_vars_n() > 0`
([`problems/zdt1_mixed.hpp`](../problems/zdt1_mixed.hpp),
[`examples/ibea_mixed.cpp`](../examples/ibea_mixed.cpp)); 51 of the 76 cores
accept one, the other 18 refuse it at `setup()` with a message.

### External or parallel evaluation

`DataVault::set_batch_executor` replaces per-individual `calc_objs` with one
call per generation over the whole set of unevaluated individuals
(`BatchRequest` in, `BatchResponse` out; the contract is at the top of
[`batch_executor.hpp`](../include/mootation/batch_executor.hpp)). This is where
a thread pool, a process pool or a simulator queue plugs in;
[`examples/batch_demo.cpp`](../examples/batch_demo.cpp) shows it. Steady-state
cores still evaluate one candidate per `step()`, because that is how those
algorithms are defined.

An evaluation that writes a non-finite objective or constraint value — your
solver crashed and you wrote NaN — has failed. By default the vault gives the
point a penalty: above the worst finite value of every objective and a
violation beyond the worst seen, the same for every failure, so that every
algorithm discards it before any point that did not fail; `failed_n()` counts
them and `is_failed(v)` marks them, so leave those out of the answer you read.
`set_fail_raise(true)` throws at the first failure instead
([`data_vault.hpp`](../include/mootation/data_vault.hpp), FAILED EVALUATIONS).

Equalities h(x) = 0 are the last constraint values your problem writes, with
`set_equalities(n, delta)` on the vault before the first evaluation: the vault
turns each h into |h| - delta, so `limits_of` keeps the meaning "<= 0
satisfied" and cv adds max(|h| - delta, 0). `set_eq_tol(delta)` moves the
tolerance between steps and rewrites every evaluated point from its |h|.
Limits you hand to `setup_with_seed` are your problem's own, h included; an
algorithm writing values back through `seed_individual` writes the vault's
(EQUALITY CONSTRAINTS). `set_cv_level(eps)` makes a violation up to eps count
as none for the algorithm — the ε-level comparison on any feasibility-first
core — while `raw_cv(v)` and `max_raw_cv()` keep the violation itself
(EPSILON LEVEL); set it back to 0 before you read the answer's cv.

### Diagnostics

Some cores substitute a parameter that cannot be honoured: a subregion count
that is not a lattice size comes back rounded up, and the derived quota changes
with it. Each such case is named in the file's header and announced at run time
through `set_warn_handler` (silent by default). Where a core cannot honour the
configuration at all (an exact lattice size, `pop` divisible by `K`, a binary
genome) it throws `std::invalid_argument` from `setup()` with the nearest valid
sizes in the message.

## Run-time choice: `embed.hpp`

`#include <mootation/embed.hpp>` instantiates all 76 cores in that translation
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
| `crossover`, `mutation` | words: `sbx` / `uniform` / `blx_alpha` / `spx` / `rex` / `undx` / `pcx` / `de` / `de_pbest` / `portfolio`, and `polynomial` / `gaussian` / `cauchy` / `uniform_reset` / `mixture` / `mixture_cauchy` / `none` — NSGA-II, IBEA-ε+, SPEA2+SDE, AGE-MOEA and SMS-EMOA take both, MOEA/D-DE the mutation; the defaults are the papers' SBX and polynomial mutation |
| `mutation_scale`, `mixture_q`, `blx_alpha` | their parameters: s of the gaussian (0.1) and Cauchy (0.05) steps as a share of ub − lb (> 0: "no mutation" is `mutation = none`), the share of noisy steps in a mixture (0.1), BLX's α (0.5) |
| `F`, `CR`, `de_p`, `de_archive`, `de_adapt`, `de_adapt_weight`, `de_memory` | `crossover = de`: DE/rand/1/bin, `r0` and `r1` the host's pair and two more parents from its selection; `de_pbest`: current-to-pbest/1 (JADE), x_pbest one of the host's best `de_p` (0.1) by its own order, r2 from the population and the archive of displaced parents (`de_archive = yes`); F = 0.5, CR = 1. `de_adapt = shade`: F and CR from a SHADE memory of `de_memory` (5) cells, success = the child survived the host's selection, weights `equal` or `contribution` (hypervolume) — `operators/de_crossover.hpp`; the memory and the means of CR and F drawn and kept are in `operator_stats` |
| `dn_cf` | dn_nsga2: the crowding factor of its niched mating pool, a whole number ≥ 1 (20) |
| `hv_algo` | sms_emoa, mo_cma_es: the exact contributions' hypervolumes by `wfg` (default) or `box` (box decomposition, `hv_box.hpp`) |
| `hv_contrib` | sms_emoa, mo_cma_es: the exact contributions at M = 3 and 4 by `wfg` (default) or `hvc` (HVC3D with single-point updates, HVC4D; `hvc.hpp`); at M = 4 only with `n_samples = 0`; another trajectory than `wfg` possible |
| `n_samples` | sms_emoa, mo_cma_es, hype: the Monte-Carlo sample points of a step from four objectives (10 000); 0 the exact contributions at every M (sms_emoa, mo_cma_es) |
| `survival`, `spu_frac` | `greedy` or `stochastic` (sms_emoa, nsga2, nsga3): Bian et al.'s stochastic population update, Q′ of ⌊`spu_frac`·\|Q\|⌋ (1/2; 3/4) selected alone, the rest kept (docs/running.md) |
| `inner_archive`, `inner_archive_cap`, `archive_reuse_p` | `off` or `nd` (sms_emoa, nsga2, nsga3, ibea_eplus): an archive of the non-dominated points evaluated, its cap (unset: 10 μ), and the probability that a parent is drawn from it (`operators/inner_archive.hpp`) |
| `bin_crossover`, `mating` | `uniform` or `none` (sms_emoa, nsga2) for the binary genome; `tournament` or `each` (nsga2) |
| `coords`, `coords_warmup`, `coords_update` | the variation in learned coordinates of the switchable algorithms and mosa's sbx step: `axis` (default), `pca`, `pca_nd`, `randrot`; R = I for the first `coords_warmup` (10) generations, updated every `coords_update` (1) (`operators/learned_coords.hpp`; docs/running.md) |
| `portfolio`, `portfolio_credit` | `crossover = portfolio`: every mating's operator drawn from the list (`sbx, de, pcx, undx`), `uniform`ly or as in Borg (`borg`: ∝ the operator's children in the archive of the non-dominated survivors + 1); the share of each operator's children and of them kept are in `operator_stats` |
| `ref_delta` | SMS-EMOA and HypE: the reference point nad + δ·(nad − ideal), relative, instead of their papers' nad + 1 and max·(1 ± offset), which move with the objectives' scale and shift |
| `sbx_var_prob` | SBX's share of crossed variables (0.5) for this run |
| `bound_repair` | a word: `clip`, `reflect`, `random`, `midpoint`, `resample`, `wrap` or `native` — what an operator that can leave the box does with the variables it put outside (DE in MOEA/D-DE, MOEA/D-DRA, LIS/LCS and HCCA; Liu & Li's operators; DCEA, HLMEA, NAEMO and EMyO/C); each keeps its own default, and the others report it as ignored (`operators/bound_repair.hpp`). A value still outside when written is clamped by the store and counted as a repair of its own, `vault` in the operator log |
| `step_share`, `F_spread` | EMyO/C: the clamp of its difference vector as a share of ub − lb (0.5) and the spread of its factor F (0) |
| `smpso_preset`, `chi_mode`, `turbulence`, `r_per_dim`, `archive_prune`, `leader_select`, `leader_offer` | words for SMPSO: the preset `jmetal` / `platemo` / `fair` / `omopso` and the readings it fixes — χ `signed` / `abs` / `none`, the turbulence `every_k` / `prob` / `thirds` / `off`, r1 and r2 per particle or per variable (`no` / `yes`), the leaders' truncation `one_by_one` / `one_shot`, the leader `crowding_tournament` / `random`, the particles offered as leaders `all` / `improved` (`smpso.hpp`; [running.md](running.md)) |
| `vmax_frac`, `vel_on_bound_low`, `vel_on_bound_up`, `w_min`, `w_max`, `c_min`, `c_max`, `turbulence_k`, `turbulence_p`, `archive_size`, `eps_archive_eta` | SMPSO's numbers: the velocity bound as a share of ub − lb (0.5; 0: none), the velocity's factor at each bound (−1), the inertia weight's and C1, C2's ranges (0.1; 1.5–2.5), every k-th particle mutated (6) or each with a probability (0.15), the leaders' bound (pop_size), omopso's ε (0.0075); each overrides its preset |
| `r2_mode`, `r2_space`, `r2_weights`, `r2_offset` | R2-EMOA: the exact or the discrete R2, weights uniform in weight or angle space, their number (500), the utopian point's offset (0) (`r2_emoa.hpp`) |
| `bc_selection`, `bc_lambda` | BC: `bc` (default), `bf` or `ba`; the children of one step (0: 10n) (`bc_spx.hpp`) |
| `paes_depth` | PAES: the grid's recursive halvings per objective (5) (`paes.hpp`) |
| `mads_tau`, `mads_delta0`, `mads_select`, `mads_poll`, `mads_search` | DMulti-MADS: τ (1/2), Δ_0 (0.1), the incumbent `spread` / `first`, the poll `complete` / `opportunistic`, the search `speculative` / `none` (`dmulti_mads.hpp`) |
| `glods_search_every`, `glods_local_tol` | MultiGLODS: failures in a row before a search (3), the step below which a point is local (10⁻³) (`multiglods.hpp`) |
| `rm_extension` | RM-MEDA: how far each cluster's model extends past its points along each direction, a share of their range (0.25); its K is `n_clusters` (5) (`rm_meda.hpp`) |
| `scalarization`, `local`, `sub_plan`, `tch_weights`, `n_sub`, `sub_budget`, `fd_step`, `local_tol`, `cma_sigma0`, `eps_objective`, `boost_scalarization`, `mosqp_tau`, `mosqp_mu`, `mosqp_beta`, `mosqp_spread`, `polish`, `polish_share`, `polish_points` | Python only: the scalarization methods, boostdms, mosqp and the polishing (`python/mootation/scalar/`, [running.md](running.md)); a core reports them ignored |
| `dms_init`, `dms_poll` | words for DMS: the initial list, `line` / `single` / `random`, and the next poll centre, `first` / `random` (`dms.hpp`, DMS-6) |
| `sa_energy`, `sa_step`, `sa_weights`, `sa_tch_accept` | words for MOSA: the energy `cmb` / `eps` / `dom` / `tch`, the step `mix` / `laplace` / `de` / `sbx` / `adaptive`, with `tch` the weights `psa` / `fixed` and the acceptance `weighted_sum` / `chebyshev` (`mosa.hpp`; [running.md](running.md)) |
| `p_de`, `sa_chains`, `sa_reseed`, `sa_t_ratio`, `sa_s0`, `sa_s_end`, `sa_archive_cap`, `sa_psa_alpha`, `sa_adapt_window` | MOSA's numbers: the differential step's probability (0.7), the chains (1; 10 with `tch`), the reseeding probability (1/50), T's last over first (10⁻⁴), the Laplace scale first and last (0.2, 0.002), the archive's cap (0: none), PSA's α (1.05), the adaptive step's window (200) |

`Settings::from_file("run.cfg")` reads the file; an unknown key is an error, not
a silent no-op. [`examples/run.cfg`](../examples/run.cfg) is a commented copy
of every key. A knob the chosen algorithm does not have is reported in
`Result::ignored` rather than dropped.

Which algorithm takes the operator knobs (a knob it lacks is reported in
`Result::ignored`; measured by a short run of every algorithm with all six
set). `pm` is 1/n by default wherever it is a knob; NSGA-II, SPEA2+SDE,
AGE-MOEA and SMS-EMOA took it only from task 7 on.

<details><summary>pc, pm, eta_m, eta_c, mutation, crossover per algorithm</summary>

| algorithm | `pc` | `pm` | `eta_m` | `eta_c` | `mutation` | `crossover` |
|---|---|---|---|---|---|---|
| `nsga2` | yes | yes | yes | yes | yes | yes |
| `nsga2_sdr` | yes | yes | yes | yes | yes | yes |
| `spea2` | yes | yes | yes | yes | no | no |
| `spea2_sde` | yes | yes | yes | yes | yes | yes |
| `grea` | yes | yes | yes | yes | no | no |
| `vaea` | yes | yes | yes | yes | no | no |
| `agemoea` | yes | yes | yes | yes | yes | yes |
| `etea` | yes | yes | yes | yes | no | no |
| `gde3` | no | no | no | no | no | no |
| `nsga3` | yes | yes | yes | yes | no | no |
| `a_nsga3` | yes | yes | yes | yes | no | no |
| `theta_dea` | yes | yes | yes | yes | no | no |
| `ar_moea` | yes | yes | yes | yes | no | no |
| `moead` | yes | yes | yes | yes | no | no |
| `moead_de` | no | yes | yes | yes | yes | no |
| `moead_dra` | no | yes | yes | yes | no | no |
| `moead_dd` | yes | yes | yes | yes | no | no |
| `moead_awa` | yes | yes | yes | yes | no | no |
| `adaw` | yes | yes | yes | yes | no | no |
| `moead_m2m` | yes | yes | yes | yes | no | no |
| `moead_am2m` | yes | yes | yes | yes | no | no |
| `sms_m2m` | yes | yes | yes | yes | no | no |
| `moead_ds` | yes | yes | yes | yes | no | no |
| `rd_emo` | yes | yes | yes | yes | no | no |
| `aprd` | yes | yes | yes | yes | no | no |
| `liu_gu2011` | yes | yes | yes | yes | no | no |
| `isde_rd` | yes | yes | yes | yes | no | no |
| `dhea` | yes | yes | yes | yes | no | no |
| `hlmea` | no | yes | yes | yes | no | no |
| `ibea_eplus` | yes | yes | yes | yes | yes | yes |
| `ibea_hd` | yes | yes | yes | yes | no | no |
| `mibea` | yes | yes | yes | yes | no | no |
| `r2ibea` | yes | yes | yes | yes | no | no |
| `hype` | yes | yes | yes | yes | no | no |
| `mombi2` | yes | yes | yes | yes | no | no |
| `nimmo` | yes | yes | yes | yes | no | no |
| `irea` | yes | yes | yes | yes | no | no |
| `edv` | yes | yes | yes | yes | no | no |
| `if_maoea` | yes | yes | yes | yes | no | no |
| `maoea_iamd` | yes | no | yes | yes | no | no |
| `sms_emoa` | yes | yes | yes | yes | yes | yes |
| `mo_cma_es` | no | no | no | no | no | no |
| `rvea` | yes | yes | yes | yes | no | no |
| `rvea_star` | yes | yes | yes | yes | no | no |
| `maoeaarv` | yes | yes | yes | yes | no | no |
| `mbra` | yes | yes | yes | yes | no | no |
| `nrv_moea` | yes | yes | yes | yes | no | no |
| `dea_gng` | yes | yes | yes | yes | no | no |
| `srv` | yes | yes | yes | yes | no | no |
| `srv_nsga3` | yes | yes | yes | yes | no | no |
| `camoea` | yes | yes | yes | yes | no | no |
| `cava_moea` | yes | yes | yes | yes | no | no |
| `maoea_ac` | yes | yes | yes | yes | no | no |
| `maoea_c` | yes | yes | yes | yes | no | no |
| `maoea_3c` | yes | yes | yes | yes | no | no |
| `emyo_c` | no | yes | yes | yes | no | no |
| `crea` | yes | yes | yes | yes | no | no |
| `dcea` | no | yes | yes | yes | no | no |
| `hcca` | yes | yes | yes | yes | no | no |
| `lis_lcs` | yes | yes | yes | yes | no | no |
| `clia` | yes | yes | yes | yes | no | no |
| `two_arch2` | yes | yes | yes | yes | no | no |
| `naemo` | yes | yes | yes | no | no | no |
| `dms` | no | no | no | no | no | no |
| `mosa` | yes | yes | yes | yes | no | no |

</details>

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
