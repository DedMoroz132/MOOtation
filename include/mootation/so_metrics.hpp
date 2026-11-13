#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// The single-objective measures of a run (task 18, stage 1, §1.3): its
// precision Δf = f - f_opt, the targets, the runtime to a target, the average
// runtime (ERT) over runs, the empirical runtime distribution (ECDF) and the
// value at a fixed budget, all read off a run's improvement records
// (scalar/improvements.hpp: a record each time the best point improves).
// The one implementation: the campaign, the atlas and Python
// (_core.so_metrics; run/so_metrics.py is the cross-check) take these.
//
// Sources: N. Hansen, A. Auger, D. Brockhoff, D. Tušar, T. Tušar, "COCO:
// Performance Assessment", arXiv:1605.03560 (H16), and "Anytime Performance
// Assessment in Blackbox Optimization Benchmarking", IEEE TEVC 26(6) 2022
// (H22). J.J. Moré, S.M. Wild, "Benchmarking derivative-free optimization
// algorithms", SIAM J. Optim. 20(1) 2009, as H22 §III-F describes it.
//
// TARGETS (H22 §III-F (1)): tau(eps) = f_ref + eps.
//   absolute (bbob, CEC 2013): eps = 10^k, k = 2, 1.8, ..., -8 — COCO's 51
//     (H22 §III-F item 2, H16 §2); its tables take 1e1, 1e0, 1e-1, 1e-2,
//     1e-3, 1e-5, 1e-7 (H22 Table I); the final target is f_opt + 1e-8.
//   relative (MGH, NIST; the owner's decision of 01.11, В1): eps =
//     10^-k (f(x0) - f_ref), k = 1, 3, 5, 7 (Moré and Wild, H22 §III-F item
//     1), x0 the run's start.
// Which kind a problem takes is the registry's `targets` (RegistryProblem::
// targets); default_precisions gives each kind's eps.
// f_ref is f_opt where it is known, else the best known value, of version
// kReferenceVersion (SO_REFERENCE_VERSION; the registry's f_opt_kind says
// which).
//
// THE RUNTIME to a target (H22 §III-E): the cost at the first record whose
// point is feasible and at or below the target; none where no record is.
// The cost is evaluations + grad_cost * grad_calls (H22 §III-B: any
// accumulated cost; scalar/evaluator.hpp THE COUNTS), so another grad_cost
// reads the same records anew.
//
// ERT (H16 §4.1, H22 §V-A): (the runtimes of the runs that hit the target +
// the whole cost of those that did not) / the number that hit it; 0 / 0 is
// 0, and with no hit it is +inf.
//
// THE ECDF (H16 §4.3): over pairs (run, target), the share whose runtime is
// at most x; a pair without a runtime is in the denominator.
//
// THE VALUE AT A BUDGET: the best feasible f among the records of cost at
// most the budget (H22 §III-D: an ordinal measure only).
//
// SIMULATED RESTARTS (H16 §3.2, H22 §V-A): the runtime of the restarted
// algorithm off K runs at one target. A sample picks runs until one hit the
// target; its runtime is the whole cost of the runs that missed plus the
// runtime of the one that hit (H22 Eq. (2)). N samples bootstrap the
// distribution (H16 §3.2.1, H22 §V-A 1); their mean estimates ERT (H22
// §V-B), their ECDF is COCO's runtime distribution with simulated restarts
// (H22 §VI). With no run that hit, there is no sample (the procedure would
// never stop, H16 §3.2).
//
// READINGS:
//   SOM-1. "At or below": a record hits tau when f <= tau (H22's first
//     hitting of the target line).
//   SOM-2. Only a feasible record (cv = 0) hits a target or gives a value:
//     the problems of the registry at one objective have no constraints, and
//     H22's constrained indicator (its footnote 12) is not taken up here.
//   SOM-3. The absolute grid's exponents are k = k_max - i / 5, i = 0, 1,
//     ..., 5 (k_max - k_min), each 10^k by the library's pow (dm::), so the
//     same targets on every platform.
//   SOM-4. The simulated restarts are here (simulated_runtimes), per run
//     set and target; their aggregation (percentiles, the ECDF over
//     problems) is the report's (run/report_so.py, REP-4).
//   SOM-6. The first run of sample s, s = 0 ... N - 1, is run perm[s mod K]
//     of one random permutation of the K runs, drawn before the samples
//     (first = permuted, the default): H22 §V-A 1 picks the first trials
//     "without replacement", H16 §3.2.1 picks the (1 + (N mod K))-th and its
//     footnote 12 suggests a random permutation first. The further runs, and
//     every run with first = random, uniform with replacement.
//   SOM-7. The draws are std::mt19937(seed)'s words through rnd::below and
//     rnd::shuffle (random.hpp), so the same samples on every platform.
// ============================================================================

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "detmath.hpp"
#include "random.hpp"
#include "scalar/improvements.hpp"

namespace mootation::so {

using scalar::Improvement;

// SO_REFERENCE_VERSION: the version of the best known values that stand in
// for f_opt where it is not known
inline constexpr int kReferenceVersion = 1;

enum class TargetKind { Absolute, Relative };

inline TargetKind target_kind(const std::string& s)
{
    if (s == "absolute") return TargetKind::Absolute;
    if (s == "relative") return TargetKind::Relative;
    throw std::invalid_argument("so_metrics: targets 'absolute' or 'relative', not '" + s + "'");
}

// COCO's precisions 10^k, k = k_max, k_max - 0.2, ..., k_min (SOM-3)
inline std::vector<double> coco_precisions(int k_min = -8, int k_max = 2)
{
    if (k_min > k_max) throw std::invalid_argument("so_metrics: k_min above k_max");
    std::vector<double> out;
    for (int i = 0; i <= 5 * (k_max - k_min); ++i)
        out.push_back(dm::pow(10.0, static_cast<double>(5 * k_max - i) / 5.0));
    return out;
}

// the precisions of COCO's tables (H22 Table I)
inline std::vector<double> coco_table_precisions() { return {1e1, 1e0, 1e-1, 1e-2, 1e-3, 1e-5, 1e-7}; }

// Moré and Wild's factors 10^-k, k = 1, 3, 5, 7
inline std::vector<double> more_wild_factors() { return {1e-1, 1e-3, 1e-5, 1e-7}; }

// A kind's precisions: COCO's 51 (absolute), Moré and Wild's four (relative)
inline std::vector<double> default_precisions(TargetKind kind)
{
    return kind == TargetKind::Absolute ? coco_precisions() : more_wild_factors();
}

// The targets of a run: f_ref + eps (absolute) or f_ref + eps (f_x0 - f_ref)
// (relative), in the order of eps
inline std::vector<double> targets(TargetKind kind, const std::vector<double>& eps, double f_ref, double f_x0)
{
    std::vector<double> out;
    for (double e : eps) out.push_back(kind == TargetKind::Absolute ? f_ref + e : f_ref + e * (f_x0 - f_ref));
    return out;
}

inline double cost(const Improvement& r, double grad_cost)
{
    return static_cast<double>(r.evaluations) + grad_cost * static_cast<double>(r.grad_calls);
}

// The cost at the first feasible record at or below target (SOM-1, SOM-2)
inline std::optional<double> runtime(const std::vector<Improvement>& records, double target, double grad_cost = 1.0)
{
    for (const auto& r : records)
        if (r.cv <= 0.0 && r.f <= target) return cost(r, grad_cost);
    return std::nullopt;
}

// ERT over runs: runtimes[i] the run's runtime (none: missed), lengths[i] its
// whole cost
inline double ert(const std::vector<std::optional<double>>& runtimes, const std::vector<double>& lengths)
{
    if (runtimes.size() != lengths.size()) throw std::invalid_argument("so_metrics: a length per run");
    double total = 0.0;
    std::size_t hits = 0;
    for (std::size_t i = 0; i < runtimes.size(); ++i) {
        if (runtimes[i]) {
            total += *runtimes[i];
            ++hits;
        } else {
            total += lengths[i];
        }
    }
    if (hits == 0) return total == 0.0 ? 0.0 : std::numeric_limits<double>::infinity();
    return total / static_cast<double>(hits);
}

// The ECDF at each x of xs: the share of the runtimes (one per pair run x
// target; none: missed) at most x
inline std::vector<double> ecdf(const std::vector<std::optional<double>>& runtimes, const std::vector<double>& xs)
{
    std::vector<double> out;
    for (double x : xs) {
        std::size_t below = 0;
        for (const auto& t : runtimes) below += t && *t <= x ? 1 : 0;
        out.push_back(runtimes.empty() ? 0.0 : static_cast<double>(below) / static_cast<double>(runtimes.size()));
    }
    return out;
}

// N simulated runtimes off runs at one target (SIMULATED RESTARTS, SOM-6,
// SOM-7): runtimes[i] the run's runtime (none: missed), lengths[i] its whole
// cost; first_permuted false draws the first run of a sample like the others.
// Empty when no run hit the target.
inline std::vector<double> simulated_runtimes(const std::vector<std::optional<double>>& runtimes,
                                              const std::vector<double>& lengths, std::size_t n, unsigned seed,
                                              bool first_permuted = true)
{
    if (runtimes.size() != lengths.size()) throw std::invalid_argument("so_metrics: a length per run");
    const std::size_t k = runtimes.size();
    bool any = false;
    for (const auto& t : runtimes) any = any || t.has_value();
    std::vector<double> out;
    if (!any) return out;
    std::mt19937 g(seed);
    std::vector<std::size_t> perm(k);
    for (std::size_t i = 0; i < k; ++i) perm[i] = i;
    if (first_permuted) rnd::shuffle(perm.begin(), perm.end(), g);
    out.reserve(n);
    for (std::size_t s = 0; s < n; ++s) {
        double total = 0.0;
        std::size_t i = first_permuted ? perm[s % k] : static_cast<std::size_t>(rnd::below(g, k));
        while (!runtimes[i]) {
            total += lengths[i];
            i = static_cast<std::size_t>(rnd::below(g, k));
        }
        out.push_back(total + *runtimes[i]);
    }
    return out;
}

// THE PEAKS FOUND (task 18, stage 3, §3.5; the CEC 2013 niching report,
// Li, Engelbrecht, Epitropakis 2013, §III, Algorithm 1): the points sorted
// by F, largest first (F the report's, maximized); a point within eps of the
// peak height is a peak found when it is farther than r (Euclidean, in x)
// from every peak found before it. The count, at most the number of global
// optima. SOM-5: the order of equal F is the points' own (a stable sort).
inline int peaks_found(const std::vector<std::vector<double>>& X, const std::vector<double>& F, double height,
                       double eps, double r)
{
    std::vector<std::size_t> order(X.size());
    for (std::size_t i = 0; i < order.size(); ++i) order[i] = i;
    std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) { return F[a] > F[b]; });
    std::vector<std::size_t> seeds;
    for (std::size_t i : order) {
        if (!(std::fabs(height - F[i]) <= eps)) continue;
        bool far = true;
        for (std::size_t s : seeds) {
            double d2 = 0.0;
            for (std::size_t j = 0; j < X[i].size(); ++j) d2 += (X[i][j] - X[s][j]) * (X[i][j] - X[s][j]);
            far = far && std::sqrt(d2) > r;
        }
        if (far) seeds.push_back(i);
    }
    return static_cast<int>(seeds.size());
}

// The best feasible f among the records of cost at most budget; none before
// the first feasible record
inline std::optional<double> value_at(const std::vector<Improvement>& records, double budget, double grad_cost = 1.0)
{
    std::optional<double> best;
    for (const auto& r : records) {
        if (cost(r, grad_cost) > budget) break;
        if (r.cv <= 0.0 && (!best || r.f < *best)) best = r.f;
    }
    return best;
}

}   // namespace mootation::so
