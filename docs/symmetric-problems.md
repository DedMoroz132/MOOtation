# Problems with symmetries

Single-objective problems whose parameters can be interchanged or moved along known directions without
changing the objective (task 19). They are least-squares teacher-student problems (but `lj_cluster`):
the data come from a teacher of the model class, the unknowns are the student's, the objective is
½‖r‖². Each is a family of the C++ suite (`include/mootation/suite/sym.hpp` and the headers it names,
with their sources and readings), evaluated by C++ and checked against an independent NumPy version
(`python/mootation/benchmarks/symmetric.py`).

## What a problem says about its symmetries

`suite::Native` (and `_core.Native` in Python) carries, for these families:

| member | meaning |
|---|---|
| `blocks` | `(offset, block_size, count)`: `count` blocks of `block_size` coordinates from `offset` that may be interchanged; other coordinates are free |
| `canon(x)` | a canonical representative: `f(canon(x)) = f(x)`, `canon(canon(x)) = canon(x)` bit for bit, `canon(P(x)) = canon(x)` for a permutation `P` of the blocks |
| `gauge_basis(x)` | `gauge_dim` directions (columns, `n_var` rows) along which the residuals (or the objective) do not change to first order |
| `linear` | the parameters the residuals are linear in (variable projection) |
| `res_blocks` | the sizes of the residuals' blocks, in order (`chaboche1d`: its three amplitudes, CH-4; `sumexp`: the grid's two halves, SE-4); empty: one block |
| `gradient(x)` | the objective's gradient where it is not a sum of squares |
| `start()` | the problem's own start, the registry's `x0` |

## The families

| family | model | symmetries | `f_opt` |
|---|---|---|---|
| `sumexp` | Σ a_k exp(−t/τ_k) on a grid; unknowns (a_k, log τ_k) | the K_s pairs (blocks); `canon` sorts by τ; a_k linear | 0 in the exact class; Lanczos's example (variant `lanczos`) |
| `lowrank` | Y = U Vᵀ on a mask | (U, V) → (UA, VA⁻ᵀ): r² gauge directions; the components (blocks) | 0 without noise |
| `cp3` | CP model of a 3-way tensor | the components (blocks), two scales per component (2R gauge directions); `canon`: equal norms, signs, order | 0 without noise |
| `cp3_degenerate` | de Silva and Lim's rank-3 tensor (1.1), student of rank 2 | as `cp3` | infimum 0, not attained (`f_opt_kind = infimum`) |
| `rbfnet` | Σ c_k exp(−‖u − μ_k‖² / 2s_k²) | the neurons (blocks) | 0 in the exact class |
| `mlp_plateau` | v_0 + Σ v_j tanh(w_j0 + w_jᵀu) | the units (blocks), sign flips; `canon` | 0; starts `random`, `duplicate` (γ_λ), `bias` (β_(0,w)) of Fukumizu and Amari |
| `chaboche1d` | uniaxial elastoplasticity, N Armstrong–Frederick back stresses, Voce hardening, strain control | the pairs (C_k, γ_k) (blocks); `canon` sorts by γ | 0 in the exact class |
| `lj_cluster` | Lennard-Jones energy of N atoms | the atoms (blocks), translations and rotations (6 gauge directions); `canon` | Wales and Doye's Table I (best known) |

The registry (`data/registry.json`) has 27 instances, `<family>_<numbers>_1D`; where the minimum is
not known, `f_opt` is the best value known (`f_opt_kind = best_known`, `SO_REFERENCE_VERSION`).

## How each is checked

| check | where |
|---|---|
| `f(x_opt) = 0` bit for bit at the teacher's point (exact class, no noise) | `tests/test_suite.cpp` |
| the Jacobian (gradient) against central differences, relative 10⁻⁶ | `tests/test_suite.cpp` |
| `canon`: `f(canon(x)) = f(x)` to 10⁻¹², idempotent and permutation-invariant bit for bit (sign flips of `cp3` and `mlp_plateau` included) | `tests/test_suite.cpp` |
| `gauge_basis`: ‖J g‖ ≤ 10⁻¹⁰ ‖J‖ ‖g‖ for every direction (`lowrank`, `cp3`, `cp3_degenerate`, `lj_cluster`) | `tests/test_suite.cpp` |
| `sumexp`: the resolution limit against Istratov and Vyvenko's Table I; Lanczos's f₂ within two decimals of f₃ on the grid | `tests/test_suite.cpp` |
| `cp3_degenerate`: n² f(A_n) equals ½‖S + Y/n‖² (the explicit residual) at n = 10, 100, 1000 while the norms grow like n | `tests/test_suite.cpp` |
| `mlp_plateau`: at either embedding the gradient is the smaller network's at its minimum, carried over | `tests/test_suite.cpp` |
| `lj_cluster`: the 13-atom icosahedron at its best radius gives Table I's energy (the other sizes: no published coordinates in the sources) | `tests/test_suite.cpp` |
| C++ against NumPy at every registered instance: residuals, Jacobian, objective, gradient, gauge directions, `canon`, to 10⁻¹² | `python/test_symmetric.py` |
| the box from C++ is the registry's | `tests/test_registry_bounds.cpp` |

## `lm` for such problems (C++, `scalar/lm.hpp`)

| what | how | checked by (`tests/test_lm.cpp`) |
|---|---|---|
| a gauge basis (`Problem::gauge`) | the step on the complement of the gauge directions, in Moré's scaling (readings LMN); `diagnose_gauge`: singular values, covariance and correlations on the complement; `Result::gauge_k`, `gauge_ratio` | `lowrank`: the steps orthogonal to N, the same minimum as without it, J Σ Jᵀ equal to the one with the gauge fixed explicitly (U's top block = I) |
| equalities (`solve_eq`) | linear ones without bounds: exact elimination; nonlinear ones, or any with bounds: the bound-constrained Lagrangian method, Nocedal & Wright's Algorithm 17.4, with `lm` for the subproblem (readings LMC) | a linear model with linear equalities against the KKT system (both paths); min ‖x − a‖ on the unit sphere (h's Jacobian given, by differences, with a box); `lowrank` with UᵀU = VᵀV |
| profiles (`profile`) | each value of x_j fixed by a linear equality, the rest refitted, continuation outward; Raue et al.'s threshold Δ_α = χ²(α, df) and classification (readings LMP); `chi2_quantile` | a linear model's profile is the parabola and its interval the classical one; `lowrank`'s is flat (structurally non-identifiable); χ² quantiles. Raue et al.'s ODE models are not in the library: the property stands in for their experiment |

None of these changes a run of `lm` without them (the iterates' checksum of `tests/test_lm.cpp`).

## Solvers (registered: `eki` in `minimize()` and campaigns, `scalar.hpp` LOCAL-5)

| what | source | checked by |
|---|---|---|
| `eki::minimize` (`scalar/eki.hpp`): ensemble Kalman inversion on the residuals | Iglesias, Law & Stuart 2013, Algorithm 1 (readings EKI) | `tests/test_lm.cpp`: the ensemble stays in the initial ensemble's affine hull (their Theorem 2.1, tighter); a linear problem's first mean is their (39), and near (41) with many members. Their Darcy and Navier–Stokes examples are not in the library |
| `CMAES::inject`, `inject_mean`, `elitist` (`scalar/cmaes.hpp`) | Hansen 2011, Figure 1, Table 1 (readings INJ) | `tests/test_slsqp.cpp`: runs without injection unchanged (a checksum of their points); a far injected point clipped, σ's change bounded; `inject_mean`; the elitist mode never loses its best |

## Symmetries in the population methods (`scalar/symmetry.hpp`)

A problem tells its symmetries to the single-objective algorithms by `sym_blocks` (a list of
`(offset, size, count)`), and optionally `canon(x)` and `align(x, ref)`: `minimize(...,
sym_blocks=, canon=, align=)`, and in campaigns the suite's `blocks` and `canon`. The built-in
`align(x, ref)` is `canon(x)` where given, then in every group the permutation of x's blocks with the
least sum of squared distances to ref's — the Hungarian method of Kuhn 1955 in Munkres's 1957 form
(readings SYM-1 to SYM-4); without a `canon` of the problem, `sym = canon` sorts each group's blocks
(SYM-5). A group's blocks must share their bounds (SYM-6). `mootation.symmetry` is the Python twin.

The knob `sym = off | canon | align` (`off` by default: every method runs as before, bit for bit):

| method | what is aligned (`align`) or made canonical (`canon`) |
|---|---|
| `ga_sbx`, `ssga` | the second parent to the first, or both canonical, before the crossover (GA-6) |
| `de`, `jade`, `shade`, `lshade` | the base and difference vectors (archive members too) to the target, or all of them and the target canonical (DE-6, PB-3) |
| `pso`, `spso2011` | the particle's own best and its neighbourhood's best to its position; the velocity stays in the particle's representation; `canon` refused (PSO-9, SPSO-9) |

`ga_sbx` and `ssga` also take `ga_block_uniform` (each block whole from one parent, with this
probability a pair; GA-7) and `ga_block_mutation` (one block's coordinates mutated, with this
probability a child; GA-8). The Evaluator's archive is by the raw x (SYM-10). Not wired yet: the
points injected into `cmaes` (§19.4.3) and grouping by distance (with task 18's stage 3).

| check | where |
|---|---|
| the Hungarian method's cost equals the least over every permutation, n = 1 … 7 (420 matrices, ties included); ties broken the same way every time | `tests/test_symmetry.cpp` |
| `align(P(x), x) = x` bit for bit for permutations P of two groups; `distance(P(x), x) = 0`; canon applied first | `tests/test_symmetry.cpp` |
| the built-in canon: `canon(P(x)) = canon(x)`, idempotent | `tests/test_symmetry.cpp` |
| each method's step with `align` on x and on P(x) (all points but the one aligned to permuted), and with `canon` on P(x), gives the same offspring bit for bit from the same generator state: `de::trial`, `pbest::trial`, `ga::cross`, `pso::move`, `spso2011::move`; with `off` it does not | `tests/test_symmetry.cpp` |
| `block_uniform`: each block whole from one parent; `block_mutation`: one block changed, in the box | `tests/test_symmetry.cpp` |
| `sym = off` given: every method as without the knob, bit for bit; the knobs in the settings only where given; the refusals | `tests/test_symmetry.cpp`, `python/test_run_layer.py` |
| without the knobs: the checksums of `tests/test_scalar.cpp` and the methods' own tests (twins `gde3`, `nsga2`, `sms_emoa` bit for bit) unchanged | `tests/test_scalar.cpp`, `test_de`, `test_ga`, … |
| C++ against Python: the Hungarian method, `align` and the built-in canon, bit for bit | `python/test_run_layer.py` |
| a campaign with `sym` on `sumexp` and `lj_cluster`: the same files from both engines | `python/test_run_layer.py` |

## Residual blocks as objectives (`suite/blocks_objectives.hpp`)

A problem whose residuals come in B blocks (`res_blocks`) is also a problem of B objectives
f_b = ½ w_b ‖r_b‖², one call of the residuals for all B — the calibration as a vector of measures
rather than one weighted sum (Gupta, Sorooshian & Yapo 1998, their (2) against (3); the paper has no
algorithm of its own to reproduce). Every multi-objective core runs it. From Python:
`minimize(None, bounds, None, residuals=r, objectives="blocks", residual_blocks=[...],
block_weights=None, algorithm=...)`, whose `result.sum_of_squares` is Σ_b f_b of every answer point;
or `_core.Native({**spec, "objectives": "blocks", "block_weights": [...]})` for a suite problem (its
spec and pickle carry the two keys). Readings BO-1 to BO-5 in the header: f_b's order of addition, Σ_b
f_b against ½‖r‖² (within 2γ_{n_res}; one block bit for bit), above 8 blocks a warning, what passes
through (residuals, symmetries, start), the weights the adapter's. Not yet: the problems as campaign
instances of their own, and over task 20's wrappers.

| check | where |
|---|---|
| the families' blocks: 80 + 160 + 240 (chaboche1d), the grid's halves (sumexp); C++ = NumPy | `tests/test_blocks_objectives.cpp`, `python/test_symmetric.py` |
| f_b from one call of the residuals, block by block, bit for bit; Σ_b f_b within 2γ_{n_res} of ½‖r‖²; one block of weight 1 the objective at M = 1 bit for bit | `tests/test_blocks_objectives.cpp` |
| the weights; residuals, canon, start passed through; the refusals | `tests/test_blocks_objectives.cpp` |
| f_b from C++ = NumPy's to 10⁻¹² at every registered instance with blocks; the spec and pickle make the adapter again | `python/test_symmetric.py` |
| `minimize(objectives="blocks")` under nsga2 and moead: every answer point's f_b and `sum_of_squares` as above; one block = the run at one objective bit for bit; the refusals and the warning | `python/test_run_layer.py` |
