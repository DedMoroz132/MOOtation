#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// The cost to find an optimum with a prescribed confidence (task 25, item
// 24.7.3): the measures of D. Kavetski, Y. Qin, G. Kuczera, "The fast and the
// robust: trade-offs between optimization robustness and cost in the
// calibration of environmental models", Water Resour. Res. 54, 2018 (source
// fast-robust-tradeoffs_kavetski2018; "K18" below), and the distribution of
// the optima at fixed budgets of Y. Qin, D. Kavetski, G. Kuczera, P. Wang,
// "Can Gauss-Newton algorithms outperform stochastic optimization algorithms
// for parameter estimation of highly parameterized hydrological models?",
// Water Resour. Res. 58, 2022 (source can-gauss-newton-algorithms_qin2022;
// "Q22"). A run of an algorithm is K18's "invocation".
//
//   success(f, f_best, tol)    K18 (4): a run succeeds when its final value
//                              is within tol of the best known one
//   reliability(s)             K18 (6): the share of successful runs
//   batch_reliability(R, M)    K18 (9): 1 - (1 - R)^M, at least one success
//                              in M runs
//   characteristic_effort(R, alpha, n)  K18 (12): M_{X,alpha} = roundUp[log(1 - alpha)
//                              / log(1 - R)], R safeguarded to [eps_R,
//                              1 - eps_R], eps_R = 1/(n_runs + 1)
//   characteristic_cost(M, c)          K18 (13): C_{X,alpha} = M_{X,alpha} c, c the
//                              mean cost of a run
//   efficiency_ratio(CA, CB)   K18 (15): kappa^{A/B} = C^B / C^A, how many
//                              times fewer calls A needs than B
//   typ(Ms), spread(Ms)        K18 (16), (17): the median and the
//                              interquartile range of M_{X,alpha} over
//                              problems
//   values_at(runs, budget)    Q22 §3.4: the runs' best values at a budget,
//                              sorted (the empirical distribution of Q22's
//                              Figure 2); Q22's budgets 200, 500, 1000, 2000,
//                              3000, 5000 are q22_budgets()
//   share_at(runs, budget, t)  Q22 §3.4: the share of runs whose best value at
//                              the budget is at or below t (Q22's R_psi^B,
//                              there "NSE above psi")
//
// The cost of a run is so::cost of its records (evaluations + grad_cost *
// grad_calls, scalar/evaluator.hpp THE COUNTS); K18's case study takes the
// cost to termination (§3.2, c = N_Phi|T), which is the run's whole cost.
//
// READINGS:
//   KAV-1. Success is one-sided: f <= f_best + tol, tol = tau |f_best|
//     (relative, K18 (4)) or tau (absolute, "when psi_hat ~ 0", K18 after
//     (4)). K18's |psi_hat - psi| is two-sided because psi_hat is the best
//     known value and no run beats it; a run below a stale best known value
//     is a success here, not a failure.
//   KAV-2. roundUp[x] is taken of x rounded to the nearest integer when x is
//     within 64 ulp of it: log(1 - alpha) / log(1 - R) is an integer for
//     alpha = 0.75, R = 0.5 (x = 2), and the quotient of two rounded
//     logarithms may land a hair above it, which roundUp would turn into 3.
//   KAV-3. Q22's linear interpolation between iterations ((10)) is not taken:
//     a run's records carry the cost of each evaluation that improved the
//     best (scalar/improvements.hpp), so the best value at a budget is known
//     exactly (so::value_at); Q22 interpolate because an iteration of RGN
//     spends many runs at once.
//   KAV-4. The median and the quartiles are those of linear interpolation
//     between order statistics (Hyndman and Fan's definition 7, NumPy's
//     default): K18 do not say which; this one is what the Python reports
//     compute.
//   KAV-5. A run without a feasible record has no value at a budget: it is
//     not in values_at, and it counts as a miss in share_at (the share is
//     over all runs, as K18's reliability is).
// ============================================================================

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <optional>
#include <stdexcept>
#include <vector>

#include "detmath.hpp"
#include "so_metrics.hpp"

namespace mootation::so {

// KAV-1: within tol of the best known value, or below it
inline bool success(double f, double f_best, double tol) { return f <= f_best + tol; }

// The tolerance of K18 (4): tau |f_best| (relative) or tau (absolute)
inline double success_tolerance(double tau, double f_best, bool relative)
{
    return relative ? tau * std::fabs(f_best) : tau;
}

// K18 (6): the share of successful runs; 0 with no runs
inline double reliability(const std::vector<bool>& successes)
{
    if (successes.empty()) return 0.0;
    std::size_t k = 0;
    for (bool s : successes) k += s ? 1 : 0;
    return static_cast<double>(k) / static_cast<double>(successes.size());
}

// K18 (9): the probability of at least one success in m runs
inline double batch_reliability(double R, long long m)
{
    if (m <= 0) return 0.0;
    return 1.0 - dm::pow(1.0 - R, static_cast<double>(m));
}

// K18 (12) with its safeguard: R in [eps_R, 1 - eps_R], eps_R = 1/(n_runs +
// 1), n_runs the number of runs R was measured on (n_runs >= 1)
inline long long characteristic_effort(double R, double alpha, long long n_runs)
{
    if (!(alpha > 0.0 && alpha < 1.0)) throw std::invalid_argument("so::characteristic_effort: alpha in (0, 1)");
    if (n_runs < 1) throw std::invalid_argument("so::characteristic_effort: at least one run");
    if (!(R >= 0.0 && R <= 1.0)) throw std::invalid_argument("so::characteristic_effort: R in [0, 1]");
    const double eps = 1.0 / (static_cast<double>(n_runs) + 1.0);
    const double r = std::min(std::max(R, eps), 1.0 - eps);
    const double x = dm::log(1.0 - alpha) / dm::log(1.0 - r);
    const double nearest = std::round(x);
    const double ulp64 = 64.0 * std::numeric_limits<double>::epsilon() * std::max(1.0, std::fabs(x));
    const double up = std::fabs(x - nearest) <= ulp64 ? nearest : std::ceil(x);   // KAV-2
    return std::max(1LL, static_cast<long long>(up));
}

// K18 (13): the characteristic cost, c the mean cost of a run
inline double characteristic_cost(long long effort_m, double mean_run_cost) { return static_cast<double>(effort_m) * mean_run_cost; }

// K18 (15): kappa^{A/B} = C^B / C^A
inline double efficiency_ratio(double cost_a, double cost_b) { return cost_b / cost_a; }

// KAV-4: the q-quantile of values sorted in increasing order, linear between
// order statistics (Hyndman-Fan 7)
inline double quantile_sorted(const std::vector<double>& v, double q)
{
    if (v.empty()) throw std::invalid_argument("so::quantile: no values");
    const double h = (static_cast<double>(v.size()) - 1.0) * q;
    const std::size_t lo = static_cast<std::size_t>(std::floor(h));
    const std::size_t hi = std::min(lo + 1, v.size() - 1);
    return v[lo] + (h - static_cast<double>(lo)) * (v[hi] - v[lo]);
}

// The efforts as doubles, sorted (one vector, no copies of it below)
inline std::vector<double> sorted_doubles(const std::vector<long long>& efforts)
{
    std::vector<double> v;
    v.reserve(efforts.size());
    for (long long m : efforts) v.push_back(static_cast<double>(m));
    std::sort(v.begin(), v.end());
    return v;
}

// K18 (16): typ M = the median of M over problems
inline double typ(const std::vector<long long>& efforts) { return quantile_sorted(sorted_doubles(efforts), 0.5); }

// K18 (17): spread M = the interquartile range of M over problems
inline double spread(const std::vector<long long>& efforts)
{
    const std::vector<double> v = sorted_doubles(efforts);
    return quantile_sorted(v, 0.75) - quantile_sorted(v, 0.25);
}

// Q22's budgets (§3.4)
inline std::vector<double> q22_budgets() { return {200.0, 500.0, 1000.0, 2000.0, 3000.0, 5000.0}; }

// Q22 §3.4 (KAV-3, KAV-5): the runs' best feasible values at the budget,
// sorted; runs without one are left out
inline std::vector<double> values_at(const std::vector<std::vector<Improvement>>& runs, double budget,
                                     double grad_cost = 1.0)
{
    std::vector<double> out;
    for (const auto& r : runs) {
        const std::optional<double> v = value_at(r, budget, grad_cost);
        if (v) out.push_back(*v);
    }
    std::sort(out.begin(), out.end());
    return out;
}

// Q22's R_psi^B (KAV-5): the share of all runs whose best feasible value at
// the budget is at or below target
inline double share_at(const std::vector<std::vector<Improvement>>& runs, double budget, double target,
                       double grad_cost = 1.0)
{
    if (runs.empty()) return 0.0;
    std::size_t k = 0;
    for (const auto& r : runs) {
        const std::optional<double> v = value_at(r, budget, grad_cost);
        k += v && *v <= target ? 1 : 0;
    }
    return static_cast<double>(k) / static_cast<double>(runs.size());
}

}   // namespace mootation::so
