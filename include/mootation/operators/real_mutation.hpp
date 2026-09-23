#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// Mutations beyond the polynomial one (task 2 of 2026-09-23, B1).
//
// Every one mutates each variable with probability p_m (1/n by default, as
// the polynomial mutation) and hands a variable it put outside [lb, ub] to a
// bound repair (bound_repair.hpp; reflect by default, PROVISIONALLY, until the
// comparison of E3 picks one):
//
//   gaussian       x' = x + s·(ub − lb)·N(0, 1),   s = 0.1 by default
//   cauchy         x' = x + s·(ub − lb)·C(0, 1),   s = 0.05 by default
//   uniform_reset  x' = U(lb, ub)                  (cannot leave the box)
//   mixture        per mutated variable: the polynomial step with
//                  probability 1 − q, the gaussian one with probability q
//                  (q = 0.1 by default); mixture_cauchy the same with the
//                  Cauchy step
//
// Why these (from the task): the pits of g sit 0.05 of the range apart on ZDT4
// and 0.1 on DTLZ1/3; the polynomial mutation with η = 20 steps further than
// 0.1 of the range with probability 0.11 in the middle of the range and about
// half as often near a bound, the gaussian with s = 0.1 with probability 0.32;
// Dakota's MOGA, with a gaussian of σ = 0.1 of the range, is 3rd of 58 on ZDT4
// and DTLZ3_3D at 10 000 evaluations, and 14th and 32nd with a uniform reset.
// The formulas are the task's; the distributions are the standard library's.
//
// resample redraws the variable's step, at most resample_tries() times, then
// clips. A caller that repairs the whole offspring itself afterwards (MOEA/D-DE's
// Step 2.3) passes no repair (std::nullopt) and gets the raw steps.
// ============================================================================

#include <algorithm>
#include <cmath>
#include <optional>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "bound_repair.hpp"
#include "poly_mutation.hpp"
#include "sbx.hpp"   // sbx_require_bound

namespace mootation::ops {

enum class Mutation { Polynomial, Gaussian, Cauchy, UniformReset, Mixture, MixtureCauchy };

inline const char* mutation_name(Mutation m)
{
    switch (m) {
        case Mutation::Polynomial:    return "polynomial";
        case Mutation::Gaussian:      return "gaussian";
        case Mutation::Cauchy:        return "cauchy";
        case Mutation::UniformReset:  return "uniform_reset";
        case Mutation::Mixture:       return "mixture";
        case Mutation::MixtureCauchy: return "mixture_cauchy";
    }
    return "?";
}

inline std::optional<Mutation> parse_mutation(const std::string& s)
{
    for (Mutation m : {Mutation::Polynomial, Mutation::Gaussian, Mutation::Cauchy,
                       Mutation::UniformReset, Mutation::Mixture, Mutation::MixtureCauchy})
        if (s == mutation_name(m)) return m;
    return std::nullopt;
}

// The step scale s when none is set: 0.1 for the gaussian, 0.05 for Cauchy's
// heavier tails.
inline double default_mutation_scale(bool cauchy) { return cauchy ? 0.05 : 0.1; }

namespace detail {

// The polynomial mutation's step for one variable, the same arithmetic as
// polynomial_mutation (the NSGA-II code's bounded variant).
inline double poly_step(double x, double lo, double hi, double eta_m, double u)
{
    const double dx = hi - lo;
    double dq;
    if (u < 0.5) {
        const double b   = std::clamp((x - lo) / dx, 0.0, 1.0);
        const double tmp = 2.0 * u + (1.0 - 2.0 * u) * std::pow(1.0 - b, eta_m + 1.0);
        dq = std::pow(tmp, 1.0 / (eta_m + 1.0)) - 1.0;
    } else {
        const double b   = std::clamp((hi - x) / dx, 0.0, 1.0);
        const double tmp = 2.0 * (1.0 - u) + (2.0 * u - 1.0) * std::pow(1.0 - b, eta_m + 1.0);
        dq = 1.0 - std::pow(tmp, 1.0 / (eta_m + 1.0));
    }
    return std::clamp(x + dq * dx, lo, hi);
}

// One additive step x + s(ub − lb)·Z, repaired: Z from `draw`.
template <typename RNG, typename Draw>
inline double noisy_step(double x, double lo, double hi, double s,
                         const std::optional<BoundRepair>& repair, RepairTally& tally,
                         Draw draw, RNG& rng)
{
    double v = x + s * (hi - lo) * draw();
    if (!repair || (v >= lo && v <= hi)) return v;
    tally.out();
    if (*repair == BoundRepair::Resample) {
        for (int t = 0; t < resample_tries() && (v < lo || v > hi); ++t)
            v = x + s * (hi - lo) * draw();
        return std::clamp(v, lo, hi);
    }
    return repair_value(v, lo, hi, x, *repair, rng);
}

}   // namespace detail

template <typename RNG>
inline void gaussian_mutation(std::vector<double>& x,
                              const std::vector<std::pair<std::optional<double>,
                                                          std::optional<double>>>& bounds,
                              double s, double pm, const std::optional<BoundRepair>& repair,
                              RNG& rng)
{
    note_operator("gaussian", repair ? bound_repair_name(*repair) : "repaired by the caller");
    RepairTally tally;
    std::uniform_real_distribution<double> uni(0.0, 1.0);
    std::normal_distribution<double> N01(0.0, 1.0);
    for (std::size_t j = 0; j < x.size(); ++j) {
        if (uni(rng) > pm) continue;
        const double lo = sbx_require_bound(bounds[j].first,  "lower", static_cast<int>(j));
        const double hi = sbx_require_bound(bounds[j].second, "upper", static_cast<int>(j));
        x[j] = detail::noisy_step(x[j], lo, hi, s, repair, tally, [&] { return N01(rng); }, rng);
    }
}

template <typename RNG>
inline void cauchy_mutation(std::vector<double>& x,
                            const std::vector<std::pair<std::optional<double>,
                                                        std::optional<double>>>& bounds,
                            double s, double pm, const std::optional<BoundRepair>& repair,
                            RNG& rng)
{
    note_operator("cauchy", repair ? bound_repair_name(*repair) : "repaired by the caller");
    RepairTally tally;
    std::uniform_real_distribution<double> uni(0.0, 1.0);
    std::cauchy_distribution<double> C01(0.0, 1.0);
    for (std::size_t j = 0; j < x.size(); ++j) {
        if (uni(rng) > pm) continue;
        const double lo = sbx_require_bound(bounds[j].first,  "lower", static_cast<int>(j));
        const double hi = sbx_require_bound(bounds[j].second, "upper", static_cast<int>(j));
        x[j] = detail::noisy_step(x[j], lo, hi, s, repair, tally, [&] { return C01(rng); }, rng);
    }
}

template <typename RNG>
inline void uniform_reset_mutation(std::vector<double>& x,
                                   const std::vector<std::pair<std::optional<double>,
                                                               std::optional<double>>>& bounds,
                                   double pm, RNG& rng)
{
    note_operator("uniform_reset", "none");
    std::uniform_real_distribution<double> uni(0.0, 1.0);
    for (std::size_t j = 0; j < x.size(); ++j) {
        if (uni(rng) > pm) continue;
        const double lo = sbx_require_bound(bounds[j].first,  "lower", static_cast<int>(j));
        const double hi = sbx_require_bound(bounds[j].second, "upper", static_cast<int>(j));
        std::uniform_real_distribution<double> dom(lo, hi);
        x[j] = dom(rng);
    }
}

template <typename RNG>
inline void mixture_mutation(std::vector<double>& x,
                             const std::vector<std::pair<std::optional<double>,
                                                         std::optional<double>>>& bounds,
                             double eta_m, double pm, double q, bool cauchy, double s,
                             const std::optional<BoundRepair>& repair, RNG& rng)
{
    note_operator(cauchy ? "mixture_cauchy" : "mixture",
                  repair ? bound_repair_name(*repair) : "repaired by the caller");
    RepairTally tally;
    std::uniform_real_distribution<double> uni(0.0, 1.0);
    std::normal_distribution<double> N01(0.0, 1.0);
    std::cauchy_distribution<double> C01(0.0, 1.0);
    for (std::size_t j = 0; j < x.size(); ++j) {
        if (uni(rng) > pm) continue;
        const double lo = sbx_require_bound(bounds[j].first,  "lower", static_cast<int>(j));
        const double hi = sbx_require_bound(bounds[j].second, "upper", static_cast<int>(j));
        if (uni(rng) >= q) {
            if (hi - lo >= 1e-14) x[j] = detail::poly_step(x[j], lo, hi, eta_m, uni(rng));
        } else if (cauchy) {
            x[j] = detail::noisy_step(x[j], lo, hi, s, repair, tally, [&] { return C01(rng); }, rng);
        } else {
            x[j] = detail::noisy_step(x[j], lo, hi, s, repair, tally, [&] { return N01(rng); }, rng);
        }
    }
}

// ── the choice, as the switchable algorithms hold it ────────────────────────
struct MutationSpec {
    Mutation kind  = Mutation::Polynomial;
    double   scale = -1.0;                 // < 0: default_mutation_scale()
    double   q     = 0.1;                  // mixture: the share of noisy steps
    BoundRepair repair = BoundRepair::Reflect;   // provisional, see the header

    double s() const {
        const bool c = kind == Mutation::Cauchy || kind == Mutation::MixtureCauchy;
        return scale > 0.0 ? scale : default_mutation_scale(c);
    }

    // `caller_repairs`: the caller repairs the whole offspring afterwards, so
    // the steps are left raw (MOEA/D-DE).
    template <typename RNG>
    void apply(std::vector<double>& x,
               const std::vector<std::pair<std::optional<double>,
                                           std::optional<double>>>& bounds,
               double eta_m, double pm, RNG& rng, bool caller_repairs = false) const
    {
        const std::optional<BoundRepair> rep =
            caller_repairs ? std::nullopt : std::optional<BoundRepair>(repair);
        switch (kind) {
            case Mutation::Polynomial:    polynomial_mutation(x, bounds, eta_m, pm, rng); break;
            case Mutation::Gaussian:      gaussian_mutation(x, bounds, s(), pm, rep, rng); break;
            case Mutation::Cauchy:        cauchy_mutation(x, bounds, s(), pm, rep, rng); break;
            case Mutation::UniformReset:  uniform_reset_mutation(x, bounds, pm, rng); break;
            case Mutation::Mixture:
                mixture_mutation(x, bounds, eta_m, pm, q, false, s(), rep, rng); break;
            case Mutation::MixtureCauchy:
                mixture_mutation(x, bounds, eta_m, pm, q, true, s(), rep, rng); break;
        }
    }
};

}   // namespace mootation::ops
