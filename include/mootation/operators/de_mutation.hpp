#pragma once
// SPDX-License-Identifier: Apache-2.0
// DE/rand/1/bin — Differential Evolution mutation + binomial crossover
// Storn & Price, 1997, Journal of Global Optimization (source: storn1997)
//
// Mutant:  v_j = x_a_j + F * (x_b_j - x_c_j)
// Trial:   y_j = v_j   if U(0,1) < CR or j == j_rand
//          y_j = x_i_j otherwise
//
// Repair when the mutant leaves the bounds:
//   DERepair::Clip        — clamp to the bound (classical variant);
//   DERepair::RandomReset — random value inside the domain.
//     MOEA/D-DE requirement (Li & Zhang 2009, Step 2.3, huili2009): "If an
//     element of y is out of the boundary of Ω, its value is reset to be a
//     randomly selected value inside the boundary". Clamping creates clusters
//     on the bounds — use RandomReset for the MOEA/D-DE family.
//
// x_a, x_b, x_c — three base vectors (caller ensures distinct where possible)
// x_i           — current subproblem solution (crossover base)
// F             — scale factor (paper default 0.5)
// CR            — crossover rate (paper default 1.0 for MOEA/D-DE)
// j_rand        — random dimension index guaranteeing at least one gene from v

#include <algorithm>
#include <optional>
#include <random>
#include <vector>

#include "sbx.hpp"   // sbx_require_bound

namespace mootation::ops {

enum class DERepair { Clip, RandomReset };

template <typename RNG>
inline void de_rand_1_bin(
    const std::vector<double>& x_a,
    const std::vector<double>& x_b,
    const std::vector<double>& x_c,
    const std::vector<double>& x_i,
    std::vector<double>&       y,
    const std::vector<std::pair<std::optional<double>,
                                std::optional<double>>>& bounds,
    double F,
    double CR,
    DERepair repair,
    RNG& rng)
{
    std::uniform_real_distribution<double> U01(0.0, 1.0);
    int nv = static_cast<int>(x_i.size());
    // FIX 2026-07-08: OP-1 (BUG).
    // Guard nv==0 (purely binary genome, num_vars=0): without it
    // uniform_int_distribution(0,-1) is UB (a<=b is required). Clear the
    // output and return (following the liuli_mutation pattern).
    if (nv == 0) { y.clear(); return; }
    std::uniform_int_distribution<int> rand_dim(0, nv - 1);
    int j_rand = rand_dim(rng);

    y.resize(nv);
    for (int j = 0; j < nv; ++j) {
        double lo = sbx_require_bound(bounds[j].first,  "lower", j);
        double hi = sbx_require_bound(bounds[j].second, "upper", j);
        double v_j = x_a[j] + F * (x_b[j] - x_c[j]);
        if (v_j < lo || v_j > hi) {
            if (repair == DERepair::RandomReset) {
                std::uniform_real_distribution<double> dom(lo, hi);
                v_j = dom(rng);
            } else {
                v_j = std::clamp(v_j, lo, hi);
            }
        }
        y[j] = (U01(rng) < CR || j == j_rand) ? v_j : x_i[j];
    }
}

// Backward compatibility: the old signature = Clip.
// The MOEA/D-DE family is migrated to RandomReset (group-fixes stage).
template <typename RNG>
inline void de_rand_1_bin(
    const std::vector<double>& x_a,
    const std::vector<double>& x_b,
    const std::vector<double>& x_c,
    const std::vector<double>& x_i,
    std::vector<double>&       y,
    const std::vector<std::pair<std::optional<double>,
                                std::optional<double>>>& bounds,
    double F,
    double CR,
    RNG& rng)
{
    de_rand_1_bin(x_a, x_b, x_c, x_i, y, bounds, F, CR, DERepair::Clip, rng);
}

} // namespace mootation::ops
