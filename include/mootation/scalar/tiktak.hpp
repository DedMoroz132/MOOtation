#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// TikTak (task 19, §19.6; B13): the multistart of F. Arnoud, F. Guvenen, T.
// Kleineberg, "Benchmarking Global Optimizers", Federal Reserve Bank of
// Minneapolis Working Paper 801, 2023, §2.1 and Appendix B (source
// tiktak-benchmarking-global_arnoud2023_mplsfed; NBER WP 26340, 2019, §2.1
// and Appendix A.6, source tiktak-benchmarking-global_arnoud2019_nber, says
// the same, footnotes 28-29 there, 28 here). Included from strategy.hpp, a
// strategy of its frame (strategy.hpp, ROUNDS).
//
//   Step 0. N Sobol points in the box, each evaluated; the N* = share N of
//     them with the lowest values kept, s_1, ..., s_N*, f(s_1) <= ... <=
//     f(s_N*) (share = 0.1: their footnote 28 (2019) / §2.1 "for example,
//     1% to 10%").
//   Step 1. For i = 1 ... N*: the start S_i = (1 - theta_i) s_i + theta_i
//     p_low, p_low the parameters of the lowest local minimum found so far,
//     theta_i = min[max[0.1, (i / N*)^(1/2)], 0.995] (2023 footnote 28, 2019
//     footnote 29).
//   Step 2. A local search (the inner solver) from S_i to a local minimum,
//     recorded.
//   Step 3. After the N* searches, the lowest minimum is the answer (here:
//     the archive's best point, which it is or betters).
//
// Knobs: inner (a local solver: nelder_mead as their runs, lm, lbfgsb, ...),
// tiktak_n (N, 100), tiktak_share (0.1), start_points (sobol), start_budget
// (each search's allowance; 0: what is left shared among the searches left),
// min_dist, min_df (the minima log, strategy.hpp STRAT-4).
//
// Readings (OURS):
//   TT-1. Step 0's "order them in descending order, as s_1, ..., s_N*, with
//     f(s_1) <= ... <= f(s_N*)" (2019 A.6) is ascending, as §2.1 of both
//     says: the best point first. Ties keep the Sobol order (stable).
//   TT-2. i = 1: no minimum is known yet ("If i > 1", Step 1.2), S_1 = s_1.
//   TT-3. N* = max(1, round(share N)); failed or infeasible points of the
//     sample count as +inf and are kept last.
//   TT-4. The sample: start_points of designs.hpp (sobol: the Sobol
//     sequence under the seed's linear matrix scrambling), so runs of
//     different seeds differ; the paper's Sobol sequence is unscrambled.
//   TT-5. The searches are sequential (S_i needs p_low of searches 1 ... i -
//     1): a round of one start each, the generator start_seed(seed, i)
//     (STRAT-1). "Lowest" compares feasibility first, then f (search.hpp
//     SRCH-8). The paper's optional stopping heuristic (§2.1: the last two
//     different values of Z* close) is not taken: the run ends after the N*
//     searches or with the budget.
//   TT-6. The budget: Step 0 spends what its N points cost; search i gets
//     start_budget, or what is left shared among the N* - i + 1 searches
//     left, so a search that stops early leaves its rest to the later ones.
// ============================================================================

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "strategy.hpp"

namespace mootation::scalar {

namespace tiktak {

// theta_i of Step 1 (i = 1 ... N*)
inline double theta(long long i, long long n_star)
{
    const double t = std::sqrt(static_cast<double>(i) / static_cast<double>(n_star));
    return std::min(std::max(0.1, t), 0.995);
}

}   // namespace tiktak

inline Result run_tiktak(const Problem& P, double eq_tol, bool has_tol0, double eq_tol0, int pop_size,
                         long long budget, unsigned seed, const Knobs& knobs, Observer observer, int record_every)
{
    using namespace strategy;
    StrategyKnobs k("tiktak", knobs,
                    {"inner", "tiktak_n", "tiktak_share", "start_points", "start_budget", "min_dist", "min_df",
                     "keep_points"});
    const Inner inner = k.inner("inner");
    const long long N = k.whole("tiktak_n", 100, 1);
    const double share = k.number("tiktak_share", 0.1);
    const std::string points = k.word("start_points", "sobol", {"sobol", "lhs", "uniform", "halton_rand", "sobol_owen"});
    const long long cap = k.whole("start_budget", 0, 0);
    const double min_dist = k.number("min_dist", 1e-3), min_df = k.number("min_df", 1e-8);
    if (!(share > 0.0 && share <= 1.0)) throw std::invalid_argument("tiktak: tiktak_share in (0, 1]");
    search::finite_box("tiktak", P);
    const std::size_t n = P.lo.size();
    const long long n_star = std::max(1LL, std::llround(share * static_cast<double>(N)));            // TT-3

    Evaluator ev(P, eq_tol, has_tol0, eq_tol0, budget, pop_size, std::move(observer), record_every);
    ev.lean = lean_of(k);                                                            // strategy.hpp STRAT-12
    MinimaLog log(P.lo, P.hi, min_dist, min_df);
    strategy::group_by_symmetry(k, P, log);
    long long searches = 0;
    try {
        // Step 0: the sample, evaluated, its N* best in ascending order (TT-1, TT-3)
        const std::vector<double> U = designs::design(points, static_cast<std::size_t>(N), static_cast<int>(n), seed);
        std::vector<Vec> S(static_cast<std::size_t>(N), Vec(n));
        std::vector<double> f(static_cast<std::size_t>(N));
        for (std::size_t s = 0; s < S.size(); ++s) {
            for (std::size_t j = 0; j < n; ++j) S[s][j] = P.lo[j] + U[s * n + j] * search::width(P, j);
            const std::size_t i = ev.index(S[s]);
            f[s] = ev.failed[i] || ev.cv(i) > 0.0 ? std::numeric_limits<double>::infinity() : ev.F[i][0];
        }
        if (!ev.recorded) ev.record();
        std::vector<std::size_t> order(S.size());
        for (std::size_t s = 0; s < order.size(); ++s) order[s] = s;
        std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) { return f[a] < f[b]; });

        // Steps 1-2: the searches, one a round (TT-5)
        search::Found low;
        for (long long i = 1; i <= n_star; ++i) {
            const Vec& s = S[order[static_cast<std::size_t>(i - 1)]];
            Vec x = s;                                                                            // TT-2
            if (low.ok) {
                const double t = tiktak::theta(i, n_star);
                for (std::size_t j = 0; j < n; ++j) x[j] = (1.0 - t) * s[j] + t * low.x[j];
            }
            const long long rest = search::left(ev);
            if (rest <= 0) break;
            StartSpec sp;
            sp.seed = start_seed(seed, static_cast<std::uint64_t>(i));
            sp.start = search::at(x);
            sp.allowance = cap > 0 ? std::min(cap, rest) : std::max(1LL, rest / (n_star - i + 1));        // TT-6
            const std::vector<Result> res = strategy::round(ev, inner, {sp}, 1, pop_size);
            ++searches;
            const search::Found b = search::found(res[0]);
            if (!b.ok) continue;
            log.add(b.x, b.f, i);
            if (search::better(b, low)) low = b;
        }
    } catch (const BudgetSpent&) {
    }
    return search::finish(ev, k, log,
                          {{"tiktak_n_star", std::to_string(n_star)}, {"local_searches", std::to_string(searches)}});
}

}   // namespace mootation::scalar
