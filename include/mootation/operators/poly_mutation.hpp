#pragma once
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cmath>
#include <optional>
#include <random>
#include <utility>
#include <vector>

#include "sbx.hpp"   // sbx_require_bound

namespace mootation::ops {

// ============================================================================
// Polynomial mutation, bounded variant (highly-disruptive) — matches the
// NSGA-II reference code (K. Deb, mutation.c) line by line, including δ_q via
// δ1=(x−lo)/(hi−lo), δ2=(hi−x)/(hi−lo) and the factor 2(u−0.5)=2u−1; the
// de-facto standard of jMetal/PlatEMO/pymoo. The perturbation cannot leave
// the bounds by construction (clamp is a numerical guard).
//
// FIX 2026-07-08 (source-fidelity review):
//   corrected the ATTRIBUTION of the formulas. The previous header attributed
//   the δ1/δ2 formulas to the Deb & Deb 2014 paper ("Analysing mutation
//   schemes for real-parameter GAs", source: deb2014_) — this is WRONG.
//   deb2014_ (Eq.1-3, lines 51/61/63) defines a DIFFERENT, simpler variant:
//   δ̄_L=(2u)^{1/(1+η_m)}−1, δ̄_R=1−(2(1−u))^{1/(1+η_m)}, where the
//   perturbation is scaled by the distance to the bound (p−x^L / x^U−p) and
//   δ̄ does NOT depend on the position of x. The δ1/δ2 variant implemented
//   here is the NSGA-II code (mutation.c, Deb & Tiwari 2008), not the
//   deb2014_ formulas (they agree only at the endpoints: u=0→lo, u=0.5→x,
//   u=1→hi). The parameters p_m=1/n and η_m=20 follow the deb2014_ §5
//   settings (pp. 177-178). Formulas/code NOT changed — only the comment
//   text was edited.
//
// eta_m — distribution index; pm — mutation probability of EACH variable
// (taken from the specific algorithm's paper; typically 1/n).
// ============================================================================
template <typename RNG>
inline void polynomial_mutation(std::vector<double>& x,
                                const std::vector<std::pair<std::optional<double>,
                                                            std::optional<double>>>& bounds,
                                double eta_m,
                                double pm,
                                RNG& rng)
{
    std::uniform_real_distribution<double> uni(0.0, 1.0);
    for (std::size_t j = 0; j < x.size(); ++j) {
        if (uni(rng) > pm) continue;
        double lo = sbx_require_bound(bounds[j].first,  "lower", static_cast<int>(j));
        double hi = sbx_require_bound(bounds[j].second, "upper", static_cast<int>(j));
        double dx = hi - lo;
        if (dx < 1e-14) continue;
        double u = uni(rng);
        double dq;
        // FIX 2026-07-08: OP-2 (MINOR).
        // Clamp b to [0,1] before pow(1-b, ...): if x drifted slightly outside
        // [lo,hi] due to FP or a call on raw (unclipped) values, b>1 → 1-b<0 →
        // pow(negative, fractional)=NaN, propagating into x[j]. On normal
        // inputs b is already in [0,1], so the behavior does not change.
        if (u < 0.5) {
            double b   = std::clamp((x[j] - lo) / dx, 0.0, 1.0);
            double tmp = 2.0 * u + (1.0 - 2.0 * u) * std::pow(1.0 - b, eta_m + 1.0);
            dq = std::pow(tmp, 1.0 / (eta_m + 1.0)) - 1.0;
        } else {
            double b   = std::clamp((hi - x[j]) / dx, 0.0, 1.0);
            double tmp = 2.0 * (1.0 - u) + (2.0 * u - 1.0) * std::pow(1.0 - b, eta_m + 1.0);
            dq = 1.0 - std::pow(tmp, 1.0 / (eta_m + 1.0));
        }
        x[j] = std::clamp(x[j] + dq * dx, lo, hi);
    }
}

// Backward compatibility: the old signature = pm 1/n.
template <typename RNG>
inline void polynomial_mutation(std::vector<double>& x,
                                const std::vector<std::pair<std::optional<double>,
                                                            std::optional<double>>>& bounds,
                                double eta_m,
                                RNG& rng)
{
    double pm = x.empty() ? 0.0 : 1.0 / static_cast<double>(x.size());
    polynomial_mutation(x, bounds, eta_m, pm, rng);
}

} // namespace mootation::ops
