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
//              I = |x1_i − x2_i| (Eshelman & Schaffer, "Real-Coded Genetic
//              Algorithms and Interval-Schemata", FOGA 2, 1993, pp. 187-202,
//              §2 and Fig. 1, in words; source eshelman1993); α = 0.5 by
//              default, the value at which a child is as likely outside its
//              parents as between them (§3.2), the one §4.2 runs. Two children
//              per mating (§4.1), each on its own draws — this library's
//              reading. The paper says nothing of a value outside the range:
//              here it is repaired (bound_repair.hpp, reflect by default,
//              provisionally; resample redraws it; the parent for midpoint is
//              the child's own, x1 for the first)
//
// Both cross the pair with probability pc, as SBX does, and burn no random
// number for that when pc >= 1.
//
// Multi-parent (multi_parent.hpp, formulas and sources there): spx and rex
// take n + 1 parents, undx and pcx three. A host with pairwise mating calls
// apply_any: the pair it selected is crossed with probability pc, as above,
// and only then are the other parents drawn with the host's own selection,
// each one different from those already taken (MP-6); two children come out
// (undx its symmetric pair, the others two independent draws). apply() takes
// two parents only and refuses these kinds. The default SBX path draws
// nothing more than before.
// ============================================================================

#include <algorithm>
#include <cmath>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "bound_repair.hpp"
#include "multi_parent.hpp"
#include "sbx.hpp"

namespace mootation::ops {

enum class Crossover { SBX, Uniform, BlxAlpha, SPX, REX, UNDX, PCX };

inline const char* crossover_name(Crossover c)
{
    switch (c) {
        case Crossover::SBX:      return "sbx";
        case Crossover::Uniform:  return "uniform";
        case Crossover::BlxAlpha: return "blx_alpha";
        case Crossover::SPX:      return "spx";
        case Crossover::REX:      return "rex";
        case Crossover::UNDX:     return "undx";
        case Crossover::PCX:      return "pcx";
    }
    return "?";
}

inline std::optional<Crossover> parse_crossover(const std::string& s)
{
    for (Crossover c : {Crossover::SBX, Crossover::Uniform, Crossover::BlxAlpha,
                        Crossover::SPX, Crossover::REX, Crossover::UNDX, Crossover::PCX})
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

    bool multi() const
    {
        return kind == Crossover::SPX || kind == Crossover::REX ||
               kind == Crossover::UNDX || kind == Crossover::PCX;
    }

    // Parents one application takes: n + 1 for spx and rex, 3 for undx and pcx.
    int parents(int n) const
    {
        if (kind == Crossover::SPX || kind == Crossover::REX) return n + 1;
        if (kind == Crossover::UNDX || kind == Crossover::PCX) return 3;
        return 2;
    }

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
            default:
                throw std::invalid_argument(std::string("crossover '") + crossover_name(kind) +
                                            "' takes more than two parents: call apply_any");
        }
    }

    // Two children of a multi-parent kind from P, P[0] and P[1] the pair and
    // at least parents(n) members in all.
    template <typename RNG>
    void cross_multi(const ParentSet& P, std::vector<double>& c1, std::vector<double>& c2,
                     const Bounds& bounds, RNG& rng) const
    {
        const int n = static_cast<int>(P[0].size());
        if (static_cast<int>(P.size()) < parents(n))
            throw std::invalid_argument(std::string("crossover '") + crossover_name(kind) +
                                        "': too few parents");
        switch (kind) {
            case Crossover::SPX: {
                const double eps = std::sqrt(static_cast<double>(n) + 2.0);
                const auto g = mp_detail::centre(P);
                c1 = spx_child(P, eps, rng); mp_detail::repair(c1, g, bounds, repair, rng);
                c2 = spx_child(P, eps, rng); mp_detail::repair(c2, g, bounds, repair, rng);
                break;
            }
            case Crossover::REX: {
                const auto g = mp_detail::centre(P);
                c1 = rex_child(P, rng); mp_detail::repair(c1, g, bounds, repair, rng);
                c2 = rex_child(P, rng); mp_detail::repair(c2, g, bounds, repair, rng);
                break;
            }
            case Crossover::UNDX: {
                auto pr = undx_pair(P[0], P[1], P[2], 0.5, 0.35, rng);
                std::vector<double> m(P[0].size());
                for (std::size_t j = 0; j < m.size(); ++j) m[j] = 0.5 * (P[0][j] + P[1][j]);
                c1 = std::move(pr.first);  mp_detail::repair(c1, m, bounds, repair, rng);
                c2 = std::move(pr.second); mp_detail::repair(c2, m, bounds, repair, rng);
                break;
            }
            case Crossover::PCX: {
                auto a = pcx_child(P, 0.1, 0.1, rng);
                c1 = std::move(a.first);  mp_detail::repair(c1, P[a.second], bounds, repair, rng);
                auto b = pcx_child(P, 0.1, 0.1, rng);
                c2 = std::move(b.first);  mp_detail::repair(c2, P[b.second], bounds, repair, rng);
                break;
            }
            default: break;
        }
    }

    // The hosts' entry: the pair they selected, and draw_parent(), their own
    // mating selection returning a member's variables, for the parents a
    // multi-parent kind takes beyond two. The pair is crossed with
    // probability pc; when it is not, no further parent is drawn. A drawn
    // parent equal to one already taken is drawn again, at most 10·k times
    // in one application (MP-6).
    template <typename RNG, typename Draw>
    void apply_any(const std::vector<double>& p1, const std::vector<double>& p2, Draw&& draw_parent,
                   std::vector<double>& c1, std::vector<double>& c2, const Bounds& bounds,
                   double eta_c, double pc, RNG& rng) const
    {
        if (!multi()) { apply(p1, p2, c1, c2, bounds, eta_c, pc, rng); return; }
        require_repair(repair, crossover_name(kind), false, false);
        note_operator(crossover_name(kind), bound_repair_name(repair));
        c1 = p1; c2 = p2;
        if (pc < 1.0) {
            std::uniform_real_distribution<double> uni(0.0, 1.0);
            if (uni(rng) > pc) return;
        }
        ParentSet P{p1, p2};
        const int k = parents(static_cast<int>(p1.size()));
        int redraws = 0;
        while (static_cast<int>(P.size()) < k) {
            std::vector<double> x = draw_parent();
            const bool taken = std::find(P.begin(), P.end(), x) != P.end();
            if (taken && redraws++ < 10 * k) continue;
            P.push_back(std::move(x));
        }
        cross_multi(P, c1, c2, bounds, rng);
    }
};

}   // namespace mootation::ops
