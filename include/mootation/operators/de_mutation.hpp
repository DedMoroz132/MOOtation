#pragma once
// SPDX-License-Identifier: Apache-2.0
// DE/rand/1/bin — Differential Evolution mutation + binomial crossover
// Storn & Price, 1997, Journal of Global Optimization (source: de_storn1997)
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
#include "../random.hpp"

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
                                    "(clip, reflect, random, midpoint, wrap, r_reflect)");
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
    rnd::uniform_real_distribution<double> U01(0.0, 1.0);
    int nv = static_cast<int>(x_i.size());
    // FIX 2026-07-08: OP-1 (BUG).
    // Guard nv==0 (purely binary genome, num_vars=0): without it
    // uniform_int_distribution(0,-1) is UB (a<=b is required). Clear the
    // output and return (following the liuli_mutation pattern).
    if (nv == 0) { y.clear(); return; }
    rnd::uniform_int_distribution<int> rand_dim(0, nv - 1);
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

// ── More of the DE/x/y/z family (task 2 of 2026-09-23, B2) ──────────────────
// Storn & Price 1997, §2 (storn1997): x is the vector perturbed — "rand", a
// random population member, or "best", "the vector of lowest cost from the
// current population" — y the number of difference vectors, z the crossover,
// "bin" for the binomial one of de_rand_1_bin. Their Eq.5 writes DE/best/2 as
// v = x_best + F·(x_r1 + x_r2 − x_r3 − x_r4); by the same notation
//   DE/rand/2/bin   v = x_r1 + F·(x_r2 + x_r3 − x_r4 − x_r5)
//   DE/best/1/bin   v = x_best + F·(x_r1 − x_r2)
// with the trial built exactly as in de_rand_1_bin (j_rand, repair before the
// coin, midpoint towards the target x_i). Which vector is "best" in a
// multi-objective population is the caller's decision.
namespace detail {
template <typename RNG, typename Mutant>
inline void de_bin(Mutant mutant, const std::vector<double>& x_i, std::vector<double>& y,
                   const std::vector<std::pair<std::optional<double>,
                                               std::optional<double>>>& bounds,
                   double CR, BoundRepair repair, const char* op, RNG& rng)
{
    require_de_repair(repair, op);
    note_operator(op, bound_repair_name(repair));
    RepairTally tally;
    rnd::uniform_real_distribution<double> U01(0.0, 1.0);
    const int nv = static_cast<int>(x_i.size());
    if (nv == 0) { y.clear(); return; }
    rnd::uniform_int_distribution<int> rand_dim(0, nv - 1);
    const int j_rand = rand_dim(rng);
    y.resize(nv);
    for (int j = 0; j < nv; ++j) {
        const double lo = sbx_require_bound(bounds[j].first,  "lower", j);
        const double hi = sbx_require_bound(bounds[j].second, "upper", j);
        double v_j = mutant(j);
        const bool out = v_j < lo || v_j > hi;
        if (out) v_j = repair_value(v_j, lo, hi, x_i[j], repair, rng);
        const bool take = U01(rng) < CR || j == j_rand;
        y[j] = take ? v_j : x_i[j];
        if (take && out) tally.out();
    }
}
}   // namespace detail

template <typename RNG>
inline void de_rand_2_bin(const std::vector<double>& x_r1, const std::vector<double>& x_r2,
                          const std::vector<double>& x_r3, const std::vector<double>& x_r4,
                          const std::vector<double>& x_r5, const std::vector<double>& x_i,
                          std::vector<double>& y,
                          const std::vector<std::pair<std::optional<double>,
                                                      std::optional<double>>>& bounds,
                          double F, double CR, BoundRepair repair, RNG& rng)
{
    detail::de_bin([&](int j) { return x_r1[j] + F * (x_r2[j] + x_r3[j] - x_r4[j] - x_r5[j]); },
                   x_i, y, bounds, CR, repair, "de_rand_2_bin", rng);
}

template <typename RNG>
inline void de_best_1_bin(const std::vector<double>& x_best, const std::vector<double>& x_r1,
                          const std::vector<double>& x_r2, const std::vector<double>& x_i,
                          std::vector<double>& y,
                          const std::vector<std::pair<std::optional<double>,
                                                      std::optional<double>>>& bounds,
                          double F, double CR, BoundRepair repair, RNG& rng)
{
    detail::de_bin([&](int j) { return x_best[j] + F * (x_r1[j] - x_r2[j]); },
                   x_i, y, bounds, CR, repair, "de_best_1_bin", rng);
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
    rnd::uniform_real_distribution<double> U01(0.0, 1.0);
    std::size_t nv = x_r1.size();
    y.resize(nv);
    for (std::size_t k = 0; k < nv; ++k)
        y[k] = (U01(rng) < CR) ? x_r1[k] + F * (x_r2[k] - x_r3[k]) : x_r1[k];
}

// ── The three components of the DE mutation in MOEA/D-DE ────────────────────
// Tanabe & Ishibuchi, "Review and analysis of three components of the
// differential evolution mutation operator in MOEA/D-DE", Soft Computing,
// 2019 (source moead-de-mutation-components_tanabe2019, §3):
//   the strategy (de_base)
//     rand1              v = x_r1 + F·(x_r2 − x_r3)                  (Eq. 6)
//     current1           v = x_i + F·(x_r1 − x_r2)                   (Eq. 7)
//     current_to_pbest1  v = x_i + F·(x_pbest − x_i) + F·(x_r1 − x_r2), x_pbest
//                        one of the ⌈p·|R|⌉ best of R (JADE, Zhang & Sanderson
//                        2009, Eq. 6; source jade_zhang2009); "best" is the
//                        host's own order, no archive
//   the index selection (de_index), the indices drawn from the mating range R
//     wor   each different from i and from those drawn before   (Algorithm 2)
//     wr    independent draws: any may equal another or i       (Algorithm 3)
//     wpr   each different from those drawn before, any may equal i
//           (Algorithm 4; the authors' C++ code of MOEA/D-DE does this)
//   the bound handling: bound_repair (bound_repair.hpp) — clip is the paper's
//     replacement, random its reinitialization, reflect its reflection,
//     r_reflect its randomized reflection; its resampling is resample (the
//     host draws the indices and the mutant again).
// Algorithm 2 as printed reads "while r2 = i and r2 = r1" in line 6; its text
// ("r1 ≠ r2, r1 ≠ i, and r2 ≠ i") is what wor does.
enum class DEBase  { Rand1, Current1, CurrentToPBest1 };
enum class DEIndex { WOR, WR, WPR };

inline const char* de_base_name(DEBase b)
{
    switch (b) {
        case DEBase::Rand1:           return "rand1";
        case DEBase::Current1:        return "current1";
        case DEBase::CurrentToPBest1: return "current_to_pbest1";
    }
    return "?";
}
inline const char* de_index_name(DEIndex x)
{
    switch (x) {
        case DEIndex::WOR: return "wor";
        case DEIndex::WR:  return "wr";
        case DEIndex::WPR: return "wpr";
    }
    return "?";
}
inline std::optional<DEBase> parse_de_base(const std::string& s)
{
    for (DEBase b : {DEBase::Rand1, DEBase::Current1, DEBase::CurrentToPBest1})
        if (s == de_base_name(b)) return b;
    return std::nullopt;
}
inline std::optional<DEIndex> parse_de_index(const std::string& s)
{
    for (DEIndex x : {DEIndex::WOR, DEIndex::WR, DEIndex::WPR})
        if (s == de_index_name(x)) return x;
    return std::nullopt;
}

// k indices from R for the target i, in out[0..k). A rule R cannot satisfy
// (wor needs k members besides i, wpr k members) falls back to the next looser
// one, wor → wpr → wr, so a small R never loops forever.
template <typename RNG>
inline void de_pick(const std::vector<int>& R, int i, DEIndex how, int k, int* out, RNG& rng)
{
    const int n = static_cast<int>(R.size());
    const int others = n - static_cast<int>(std::count(R.begin(), R.end(), i));
    if (how == DEIndex::WOR && others < k) how = DEIndex::WPR;
    if (how == DEIndex::WPR && n < k) how = DEIndex::WR;
    rnd::uniform_int_distribution<int> pick(0, n - 1);
    for (int t = 0; t < k; ++t) {
        for (;;) {
            const int r = R[static_cast<std::size_t>(pick(rng))];
            bool ok = true;
            if (how == DEIndex::WOR && r == i) ok = false;
            if (how != DEIndex::WR)
                for (int s = 0; s < t; ++s) if (out[s] == r) ok = false;
            if (ok) { out[t] = r; break; }
        }
    }
}

// rand1 and current_to_pbest1 in the form of de_eq6 above (Li & Zhang Eq. 6): each variable from
// the mutant with probability CR, from the target x_i otherwise, no j_rand
// and no repair (the host repairs the finished offspring). current1 is
// de_eq6(x_i, x_r1, x_r2, ...).
template <typename RNG>
inline void de_eq6_rand1(const std::vector<double>& x_i, const std::vector<double>& x_r1,
                         const std::vector<double>& x_r2, const std::vector<double>& x_r3,
                         std::vector<double>& y, double F, double CR, RNG& rng)
{
    note_operator("de_eq6 rand1", "box repair after mutation");
    rnd::uniform_real_distribution<double> U01(0.0, 1.0);
    y.resize(x_i.size());
    for (std::size_t k = 0; k < x_i.size(); ++k)
        y[k] = (U01(rng) < CR) ? x_r1[k] + F * (x_r2[k] - x_r3[k]) : x_i[k];
}

template <typename RNG>
inline void de_eq6_pbest(const std::vector<double>& x_i, const std::vector<double>& x_pbest,
                         const std::vector<double>& x_r1, const std::vector<double>& x_r2,
                         std::vector<double>& y, double F, double CR, RNG& rng)
{
    note_operator("de_eq6 current_to_pbest1", "box repair after mutation");
    rnd::uniform_real_distribution<double> U01(0.0, 1.0);
    y.resize(x_i.size());
    for (std::size_t k = 0; k < x_i.size(); ++k)
        y[k] = (U01(rng) < CR) ? x_i[k] + F * (x_pbest[k] - x_i[k]) + F * (x_r1[k] - x_r2[k]) : x_i[k];
}

// Whether every variable of y is inside its bounds (resample: draw again).
inline bool inside_box(const std::vector<double>& y,
                       const std::vector<std::pair<std::optional<double>, std::optional<double>>>& bounds)
{
    for (std::size_t k = 0; k < y.size(); ++k) {
        if (bounds[k].first && y[k] < *bounds[k].first) return false;
        if (bounds[k].second && y[k] > *bounds[k].second) return false;
    }
    return true;
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
