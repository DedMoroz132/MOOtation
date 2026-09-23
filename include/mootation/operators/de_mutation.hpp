#pragma once
// SPDX-License-Identifier: Apache-2.0
// DE/rand/1/bin — Differential Evolution mutation + binomial crossover
// Storn & Price, 1997, Journal of Global Optimization (source: storn1997)
//
// Mutant:  v_j = x_a_j + F * (x_b_j - x_c_j)
// Trial:   y_j = v_j   if U(0,1) < CR or j == j_rand
//          y_j = x_i_j otherwise
//
// Repair when the mutant leaves the bounds — any ops::BoundRepair
// (bound_repair.hpp) except resample and native, which DE has no way to do;
// the two historical names map onto it:
//   DERepair::Clip        — clamp to the bound (classical variant) = clip;
//   DERepair::RandomReset — random value inside the domain = random.
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

#include <stdexcept>
#include <string>

#include "bound_repair.hpp"
#include "sbx.hpp"   // sbx_require_bound

namespace mootation::ops {

enum class DERepair { Clip, RandomReset };

inline BoundRepair to_bound_repair(DERepair r)
{
    return r == DERepair::RandomReset ? BoundRepair::Random : BoundRepair::Clip;
}

// DE builds a variable from three others at once: it cannot redraw one of
// them alone, and it has no repair of its own.
inline void require_de_repair(BoundRepair b, const char* op)
{
    if (b == BoundRepair::Resample || b == BoundRepair::Native)
        throw std::invalid_argument(std::string(op) + ": bound_repair '" +
                                    bound_repair_name(b) + "' is not available for DE "
                                    "(clip, reflect, random, midpoint, wrap)");
}

// The mutant is repaired BEFORE the crossover coin, as it always was, so the
// random stream of every repair option is the historical one; a variable is
// counted as out of the box only when the trial takes it from the mutant.
// Midpoint uses the target x_i as the parent.
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
    BoundRepair repair,
    RNG& rng)
{
    require_de_repair(repair, "de_rand_1_bin");
    note_operator("de_rand_1_bin", bound_repair_name(repair));
    RepairTally tally;
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
        const bool out = v_j < lo || v_j > hi;
        if (out) v_j = repair_value(v_j, lo, hi, x_i[j], repair, rng);
        const bool take = U01(rng) < CR || j == j_rand;
        y[j] = take ? v_j : x_i[j];
        if (take && out) tally.out();
    }
}

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
    de_rand_1_bin(x_a, x_b, x_c, x_i, y, bounds, F, CR, to_bound_repair(repair), rng);
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

// ── Literal DE of Li & Zhang 2009 (huili2009 Eq.6) = Zhang, Liu, Li 2009
// (zhang2009 Eq.4):
//   ȳ_k = x^{r1}_k + F·(x^{r2}_k − x^{r3}_k)   with probability CR,
//   ȳ_k = x^{r1}_k                             otherwise.
// No j_rand term and no repair: both papers repair the FINAL y, after the
// mutation step, with a random reset (repair_out_of_box below). Added on
// 2026-09-05 (full-paper checklists of moead_de / moead_dra), which use it
// instead of de_rand_1_bin.
template <typename RNG>
inline void de_eq6(const std::vector<double>& x_r1,
                   const std::vector<double>& x_r2,
                   const std::vector<double>& x_r3,
                   std::vector<double>&       y,
                   double F, double CR, RNG& rng)
{
    note_operator("de_eq6", "box repair after mutation");
    std::uniform_real_distribution<double> U01(0.0, 1.0);
    std::size_t nv = x_r1.size();
    y.resize(nv);
    for (std::size_t k = 0; k < nv; ++k)
        y[k] = (U01(rng) < CR) ? x_r1[k] + F * (x_r2[k] - x_r3[k]) : x_r1[k];
}

// ── Repair of a finished offspring (MOEA/D-DE Step 2.3, MOEA/D-DRA Step 3.3):
// "If an element of y is out of the boundary of Ω, its value is reset to be a
// randomly selected value inside the boundary" (DERepair::RandomReset).
// DERepair::Clip is the PlatEMO / jMetal clamping convention, kept for
// experiments (see MDE-6 in moead_de.hpp).
// `parent` (optional) is what midpoint moves towards; without one it clips.
template <typename RNG>
inline void repair_out_of_box(std::vector<double>& y,
                              const std::vector<std::pair<std::optional<double>,
                                                          std::optional<double>>>& bounds,
                              BoundRepair repair, RNG& rng,
                              const std::vector<double>* parent = nullptr)
{
    require_de_repair(repair, "repair_out_of_box");
    note_operator("box repair", bound_repair_name(repair));
    RepairTally tally;
    for (std::size_t k = 0; k < y.size(); ++k) {
        double lo = sbx_require_bound(bounds[k].first,  "lower", static_cast<int>(k));
        double hi = sbx_require_bound(bounds[k].second, "upper", static_cast<int>(k));
        if (y[k] < lo || y[k] > hi) {
            tally.out();
            const BoundRepair how = (repair == BoundRepair::Midpoint && !parent)
                                    ? BoundRepair::Clip : repair;
            y[k] = repair_value(y[k], lo, hi, parent ? (*parent)[k] : y[k], how, rng);
        }
    }
}

template <typename RNG>
inline void repair_out_of_box(std::vector<double>& y,
                              const std::vector<std::pair<std::optional<double>,
                                                          std::optional<double>>>& bounds,
                              DERepair repair, RNG& rng)
{
    repair_out_of_box(y, bounds, to_bound_repair(repair), rng);
}

} // namespace mootation::ops
