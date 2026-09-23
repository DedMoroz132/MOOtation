#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// Crossovers beyond SBX (task 2 of 2026-09-23, B2), two children per pair.
//
//   uniform    every variable of the first child from one parent, the
//              second child's from the other, the parent chosen by a fair
//              coin per variable — the shuffle_random mechanism of JEGA
//              (Dakota's MOGA) for two parents; cannot leave the box
//   blx_alpha  every variable drawn uniformly from [min − α·I, max + α·I],
//              I = |x1_i − x2_i|, α = 0.5 by default, each child on its own
//              draws (Eshelman & Schaffer, FOGA 2, 1993, pp. 187-202 — not in
//              this project's corpus; the formula is the task's); it can leave
//              the box, and a variable outside is repaired (bound_repair.hpp,
//              reflect by default, provisionally; resample redraws it; the
//              parent for midpoint is the child's own, x1 for the first)
//
// Both cross the pair with probability pc, as SBX does, and burn no random
// number for that when pc >= 1.
//
// PCX, UNDX and SPX will join them once their primary sources are in the
// corpus: their parameters have to come from the papers.
// ============================================================================

#include <algorithm>
#include <cmath>
#include <optional>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "bound_repair.hpp"
#include "sbx.hpp"

namespace mootation::ops {

enum class Crossover { SBX, Uniform, BlxAlpha };

inline const char* crossover_name(Crossover c)
{
    switch (c) {
        case Crossover::SBX:      return "sbx";
        case Crossover::Uniform:  return "uniform";
        case Crossover::BlxAlpha: return "blx_alpha";
    }
    return "?";
}

inline std::optional<Crossover> parse_crossover(const std::string& s)
{
    for (Crossover c : {Crossover::SBX, Crossover::Uniform, Crossover::BlxAlpha})
        if (s == crossover_name(c)) return c;
    return std::nullopt;
}

template <typename RNG>
inline void uniform_crossover(const std::vector<double>& p1, const std::vector<double>& p2,
                              std::vector<double>& c1, std::vector<double>& c2,
                              double pc, RNG& rng)
{
    note_operator("uniform", "none");
    std::uniform_real_distribution<double> uni(0.0, 1.0);
    c1 = p1; c2 = p2;
    if (pc < 1.0 && uni(rng) > pc) return;
    for (std::size_t j = 0; j < p1.size(); ++j)
        if (uni(rng) < 0.5) std::swap(c1[j], c2[j]);
}

template <typename RNG>
inline void blx_alpha_crossover(const std::vector<double>& p1, const std::vector<double>& p2,
                                std::vector<double>& c1, std::vector<double>& c2,
                                const std::vector<std::pair<std::optional<double>,
                                                            std::optional<double>>>& bounds,
                                double alpha, double pc, BoundRepair repair, RNG& rng)
{
    require_repair(repair, "blx_alpha", true, false);
    note_operator("blx_alpha", bound_repair_name(repair));
    std::uniform_real_distribution<double> uni(0.0, 1.0);
    c1 = p1; c2 = p2;
    if (pc < 1.0 && uni(rng) > pc) return;
    RepairTally t1, t2;
    for (std::size_t j = 0; j < p1.size(); ++j) {
        const double lo = sbx_require_bound(bounds[j].first,  "lower", static_cast<int>(j));
        const double hi = sbx_require_bound(bounds[j].second, "upper", static_cast<int>(j));
        const double I  = std::abs(p1[j] - p2[j]);
        const double a  = std::min(p1[j], p2[j]) - alpha * I;
        const double b  = std::max(p1[j], p2[j]) + alpha * I;
        auto child = [&](double parent, RepairTally& tally) {
            auto draw = [&] { return a + (b - a) * uni(rng); };
            double v = draw();
            if (v >= lo && v <= hi) return v;
            tally.out();
            if (repair == BoundRepair::Resample) {
                for (int t = 0; t < resample_tries() && (v < lo || v > hi); ++t) v = draw();
                return std::clamp(v, lo, hi);
            }
            return repair_value(v, lo, hi, parent, repair, rng);
        };
        c1[j] = child(p1[j], t1);
        c2[j] = child(p2[j], t2);
    }
}

// ── the choice, as the switchable algorithms hold it ────────────────────────
struct CrossoverSpec {
    Crossover   kind   = Crossover::SBX;
    double      alpha  = 0.5;                      // blx_alpha
    BoundRepair repair = BoundRepair::Reflect;     // provisional, see real_mutation.hpp

    template <typename RNG>
    void apply(const std::vector<double>& p1, const std::vector<double>& p2,
               std::vector<double>& c1, std::vector<double>& c2,
               const std::vector<std::pair<std::optional<double>,
                                           std::optional<double>>>& bounds,
               double eta_c, double pc, RNG& rng) const
    {
        switch (kind) {
            case Crossover::SBX:      sbx(p1, p2, c1, c2, bounds, eta_c, pc, rng); break;
            case Crossover::Uniform:  uniform_crossover(p1, p2, c1, c2, pc, rng); break;
            case Crossover::BlxAlpha:
                blx_alpha_crossover(p1, p2, c1, c2, bounds, alpha, pc, repair, rng); break;
        }
    }
};

}   // namespace mootation::ops
