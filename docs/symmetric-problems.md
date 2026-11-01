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
