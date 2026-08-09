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

namespace mootation::ops {

// ── Global toggle for per-variable participation in SBX ──────────────────────
// Canonical form (Deb nsga2.c realcross / PlatEMO / jMetal): each variable is
// crossed with probability 0.5. The old MOOtation code crossed ALL variables
// (=1.0), which is empirically stronger on multimodal problems (MaF3/DTLZ3).
// The toggle enables an A/B test without touching every algorithm:
// ops::sbx_var_prob() = 1.0. Default 0.5 → behavior is unchanged relative to
// the current build.
inline double& sbx_var_prob() { static double v = 0.5; return v; }

// Require explicit bounds: the silent [0,1] default led to a silent
// clamp/NaN on problems with other domains (2026-06 audit).
inline double sbx_require_bound(const std::optional<double>& b,
                                const char* side, int j)
{
    if (!b)
        throw std::invalid_argument(
            std::string("SBX/PM: variable ") + std::to_string(j) +
            " has no " + side + " bound (bounded operators require finite bounds)");
    return *b;
}

// ============================================================================
// SBX — Simulated Binary Crossover, bounded.
//
// Primary sources:
//   - β-distribution: Deb & Agrawal 1995, "Simulated Binary Crossover for
//     Continuous Search Space", Complex Systems 9(2) (source: 09-2-2).
//   - Vector scheme — Deb's reference implementation (realcross, NSGA-II code;
//     the same scheme in jMetal and PlatEMO):
//       * the whole pair is crossed with probability pc (otherwise children = parents);
//       * each variable participates with probability 0.5 (otherwise it is copied);
//       * per-variable children are swapped with probability 0.5.
//
// eta_c — distribution index; pc — crossover probability of the pair (taken
// from the specific algorithm's paper: NSGA-II 0.9, NSGA-III 1.0, ...).
// ============================================================================
template <typename RNG>
inline void sbx(const std::vector<double>& p1,
                const std::vector<double>& p2,
                std::vector<double>& c1,
                std::vector<double>& c2,
                const std::vector<std::pair<std::optional<double>,
                                            std::optional<double>>>& bounds,
                double eta_c,
                double pc,
                RNG& rng)
{
    std::uniform_real_distribution<double> uni(0.0, 1.0);
    int nv = static_cast<int>(p1.size());
    c1 = p1; c2 = p2;

    // With pc >= 1.0 do NOT burn a random number (short-circuit). A naive
    // `uni(rng) > pc` is called even at pc = 1.0 (always false) and SHIFTS the
    // RNG stream by one draw per crossover. That shift is not neutral: on
    // bistable landscapes (MaF3) it measurably changes which basin a run
    // settles into. Short-circuiting keeps the stream identical to a
    // "cross everything" implementation with no pc check.
    if (pc < 1.0 && uni(rng) > pc) return;   // the pair is not crossed at all

    const double vp = sbx_var_prob();   // 0.5 canonical / 1.0 "cross everything"
    for (int j = 0; j < nv; ++j) {
        if (vp < 1.0 && uni(rng) > vp) continue;   // variable does not participate (copy)
        double lo = sbx_require_bound(bounds[j].first,  "lower", j);
        double hi = sbx_require_bound(bounds[j].second, "upper", j);
        if (std::abs(p1[j] - p2[j]) < 1e-14) continue;
        double y1 = std::min(p1[j], p2[j]);
        double y2 = std::max(p1[j], p2[j]);
        double dy = y2 - y1;
        double u  = uni(rng);
        // Lower child (near y1)
        {
            double bl = 1.0 + 2.0 * (y1 - lo) / dy;
            double al = 2.0 - std::pow(bl, -(eta_c + 1.0));
            double bq = (u <= 1.0 / al)
                ? std::pow(u * al,            1.0 / (eta_c + 1.0))
                : std::pow(1.0 / (2.0 - u * al), 1.0 / (eta_c + 1.0));
            c1[j] = std::clamp(0.5 * ((y1 + y2) - bq * dy), lo, hi);
        }
        // Upper child (near y2)
        {
            double bh = 1.0 + 2.0 * (hi - y2) / dy;
            double ah = 2.0 - std::pow(bh, -(eta_c + 1.0));
            double bq = (u <= 1.0 / ah)
                ? std::pow(u * ah,            1.0 / (eta_c + 1.0))
                : std::pow(1.0 / (2.0 - u * ah), 1.0 / (eta_c + 1.0));
            c2[j] = std::clamp(0.5 * ((y1 + y2) + bq * dy), lo, hi);
        }
        if (uni(rng) < 0.5) std::swap(c1[j], c2[j]);
    }
}

// Backward compatibility: the old signature = pc 1.0.
// Algorithms are migrated to an explicit pc per their papers (group-fixes stage).
template <typename RNG>
inline void sbx(const std::vector<double>& p1,
                const std::vector<double>& p2,
                std::vector<double>& c1,
                std::vector<double>& c2,
                const std::vector<std::pair<std::optional<double>,
                                            std::optional<double>>>& bounds,
                double eta_c,
                RNG& rng)
{
    sbx(p1, p2, c1, c2, bounds, eta_c, 1.0, rng);
}

} // namespace mootation::ops
