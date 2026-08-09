#pragma once
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cmath>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "sbx.hpp"   // sbx_require_bound

namespace mootation::ops {

// ============================================================================
// Liu–Li operator: annealed arithmetic crossover + annealed mutation.
// FIX 2026-07-07 (source-fidelity review): new file —
// reproduction operators of the M2M/sub-regional family of Liu et al.
//
// Primary source:
//   H.-L. Liu, X. Li — "The multiobjective evolutionary algorithm based on
//   determined weight and sub-regional search", IEEE CEC 2009, §III-A,
//   Eq.(5)–(6).                              (source: liu2009)
// This operator is referenced by:
//   • liu2011  §II-C: "the crossover and mutation used in [1]" (§II-D Step 2
//     points at "equation(3)/(4)", which do not exist in that paper);
//   • liu2014 (MOEA/D-M2M) §III-A(1): "crossover and mutation operators with
//     the same control parameters in [20]".
//
// FORMULAS (letter of the paper):
//   Crossover (Eq.5):  x̃ᶜ = xⁱ + rc·(xⁱ − xʲ),
//     rc = rand·(1 − rand^((1 − gen/Max_gen)^0.7)),  rand ∈ [−1,1];
//     rc is ONE scalar per offspring (the formula is vector-wise).
//   Crossover bound repair:
//     x̃ᶜ_k < lb(k) → x̃ᶜ_k = lb(k) + 0.5·rnd·(xⁱ_k − lb(k)),  rnd ∈ [0,1];
//     x̃ᶜ_k > ub(k) → the paper prints "ub(k) − 0.5·rnd·(lb(k) − xⁱ_k)".
//   Mutation (Eq.6): each component with probability P_m, "x̃ᶜ is mutated
//     once at least"; xᶜ_h = x̃ᶜ_h + rm·(ub(h) − lb(h)),
//     rm = 0.15·rand·(1 − rand^(−(1 − gen/Max_gen)^0.7)),  rand ∈ [−1,1];
//   Mutation bound repair (from the PRE-mutation value x̃ᶜ_h):
//     < lb → lb + 0.5·rnd·(x̃ᶜ_h − lb);  > ub → ub − 0.5·rnd·(ub − x̃ᶜ_h).
//   Annealing: the exponent a = (1 − gen/Max_gen)^0.7 decays 1→0; as
//     gen→Max_gen, rc→0 and rm→0 ("similar to simulated annealing", §III-A).
//     Paper default P_m = 1/n (§V).
//
// ARBITRATION NOTES (ambiguities/typos of the primary source; resolved via
// the paper's internal consistency and the authors' canonical practice):
//   LL-1. In the rc/rm formulas the symbol "rand" appears twice. The base of
//     the power CANNOT come from [−1,1]: a negative base with a fractional
//     exponent is undefined in ℝ. We read it as TWO independent draws: a
//     sign factor u ~ U[−1,1] and a base r ~ U[0,1] (the authors' MATLAB
//     code of the M2M family does the same: (2*rand−1).*(1−rand.^…)).
//   LL-2. The upper crossover repair "ub − 0.5·rnd·(lb − xⁱ_k)" is an obvious
//     typo: with xⁱ_k ≥ lb the result is ≥ ub, i.e. outside the domain again.
//     The mirrored form ub − 0.5·rnd·(ub − xⁱ_k) is implemented — exactly
//     as in the same paper's own MUTATION repair formula.
//   LL-3. rm is drawn INDEPENDENTLY for each mutated component (the paper
//     defines rm once, but the mutation is component-wise; the authors'
//     code draws rm per component).
//   LL-4. "Mutated once at least": Bernoulli(P_m) over the components; if
//     none is selected, one uniformly random component is force-mutated.
//   LL-5. Numerical guard: the base r in rm is clamped from below at 1e-12
//     (r=0 with a negative exponent yields ±inf; guard precedent — hlmea.hpp).
//     The fraction ratio = gen/Max_gen is clamped to [0,1]; at gen ≥ Max_gen
//     the operator degenerates into copying the parent (rc=rm=0, the annealing
//     has "cooled down") — the calling algorithm must pass the real Max_gen
//     (set_t_max).
//
// gen — current generation (1-based, as in the paper: first mating at gen=1);
// max_gen ≥ 1 — maximum number of generations. Finite variable bounds are
// required (as for all bounded operators: sbx_require_bound).
// ============================================================================

// Annealing exponent a = (1 − gen/Max_gen)^0.7 (Eq.5–6), ratio ∈ [0,1].
inline double liuli_anneal(int gen, int max_gen)
{
    if (max_gen < 1)
        throw std::invalid_argument(
            "liuli: max_gen=" + std::to_string(max_gen) +
            " < 1 (a real generation budget is required, see set_t_max)");
    double ratio = static_cast<double>(gen) / static_cast<double>(max_gen);
    ratio = std::clamp(ratio, 0.0, 1.0);
    return std::pow(1.0 - ratio, 0.7);
}

// ── Crossover Eq.(5): one offspring from base parent x and partner y ─────────
// child = x + rc·(x − y); rc is one scalar per call; repair uses the base x.
template <typename RNG>
inline void liuli_crossover(const std::vector<double>& x,
                            const std::vector<double>& y,
                            std::vector<double>& child,
                            const std::vector<std::pair<std::optional<double>,
                                                        std::optional<double>>>& bounds,
                            int gen, int max_gen,
                            RNG& rng)
{
    std::uniform_real_distribution<double> uni01(0.0, 1.0);
    std::uniform_real_distribution<double> uni11(-1.0, 1.0);
    const double a = liuli_anneal(gen, max_gen);

    const double u  = uni11(rng);                       // rand ∈ [−1,1] (LL-1)
    const double r  = uni01(rng);                       // base ∈ [0,1] (LL-1)
    const double rc = u * (1.0 - std::pow(r, a));       // Eq.(5), exponent +a

    int nv = static_cast<int>(x.size());
    child.resize(nv);
    for (int k = 0; k < nv; ++k) {
        double lo = sbx_require_bound(bounds[k].first,  "lower", k);
        double hi = sbx_require_bound(bounds[k].second, "upper", k);
        double c  = x[k] + rc * (x[k] - y[k]);
        if (c < lo)      c = lo + 0.5 * uni01(rng) * (x[k] - lo);   // letter of the paper
        else if (c > hi) c = hi - 0.5 * uni01(rng) * (hi - x[k]);   // LL-2 (typo)
        child[k] = std::clamp(c, lo, hi);               // numerical guard
    }
}

// ── Mutation Eq.(6): component-wise with probability pm, at least one ────────
// x is modified in-place; repair uses the component's pre-mutation value.
template <typename RNG>
inline void liuli_mutation(std::vector<double>& x,
                           const std::vector<std::pair<std::optional<double>,
                                                       std::optional<double>>>& bounds,
                           double pm,
                           int gen, int max_gen,
                           RNG& rng)
{
    int nv = static_cast<int>(x.size());
    if (nv == 0) return;
    std::uniform_real_distribution<double> uni01(0.0, 1.0);
    std::uniform_real_distribution<double> uni11(-1.0, 1.0);
    const double a = liuli_anneal(gen, max_gen);

    auto mutate_one = [&](int h) {
        double lo  = sbx_require_bound(bounds[h].first,  "lower", h);
        double hi  = sbx_require_bound(bounds[h].second, "upper", h);
        double old = x[h];                              // x̃ᶜ_h (before mutation)
        double u   = uni11(rng);                        // rand ∈ [−1,1] (LL-1)
        double r   = std::max(uni01(rng), 1e-12);       // base, guard LL-5
        double rm  = 0.15 * u * (1.0 - std::pow(r, -a));// Eq.(6), exponent −a
        double c   = old + rm * (hi - lo);
        if (c < lo)      c = lo + 0.5 * uni01(rng) * (old - lo);
        else if (c > hi) c = hi - 0.5 * uni01(rng) * (hi - old);
        x[h] = std::clamp(c, lo, hi);                   // numerical guard
    };

    bool any = false;
    for (int h = 0; h < nv; ++h) {
        if (uni01(rng) <= pm) { mutate_one(h); any = true; }
    }
    if (!any) {                                         // "once at least" (LL-4)
        std::uniform_int_distribution<int> dh(0, nv - 1);
        mutate_one(dh(rng));
    }
}

} // namespace mootation::ops
