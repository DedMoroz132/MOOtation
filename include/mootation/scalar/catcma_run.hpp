#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// The solver catcma (task 26, the registration of task 24's §24.3.4): task
// 19's CatCMA (catcma.hpp, Hamano et al., GECCO 2024) on a problem whose last
// variables are categorical. Included by strategy.hpp.
//
// The problem's cat_levels (evaluator.hpp) give K_n for each of its last N_ca
// variables; their values are the category's index 0 ... K_n - 1 (as double),
// their bounds [0, K_n - 1]. The first n - N_ca variables are continuous,
// searched in the box scaled to [0, 1] (CCR-1). Every candidate CatCMA asks
// for is evaluated through the run's Evaluator (its budget, archive and
// answer), then told.
//
// Knobs: catcma_popsize (0: Table 1's lambda), catcma_sigma0 (0.3, in the
// scaled box), catcma_update (asng | printed, catcma.hpp CC-11), catcma_qmin
// (< 0: CC-2's margin).
//
// READINGS:
//   CCR-1. A continuous sample outside the scaled box is evaluated clipped to
//     it and told the value plus the squared distance clipped (in the scaled
//     box), as cmaes's samples (SCALAR-7).
//   CCR-2. The start: the continuous part of x0 where the run has one
//     (scaled), else the box's centre; the categories start uniform (CatCMA's
//     q^(0)).
//   CCR-3. Constraints are refused (CatCMA ranks by f alone); a problem
//     without categorical variables is refused (it is CMA-ES: cmaes).
//   CCR-4. The run ends when the budget is spent (the evaluator's).
// ============================================================================

#include <algorithm>
#include <cmath>
#include <limits>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "catcma.hpp"
#include "strategy.hpp"

namespace mootation::scalar {

inline Result run_catcma(const Problem& P, double eq_tol, bool has_tol0, double eq_tol0, int pop_size, long long budget,
                         unsigned seed, const Knobs& knobs, Observer observer, int record_every, const Start& start)
{
    using namespace strategy;
    StrategyKnobs k("catcma", knobs, {"catcma_popsize", "catcma_sigma0", "catcma_update", "catcma_qmin"});
    const std::size_t n = P.lo.size(), nca = P.cat_levels.size();
    if (nca == 0) throw std::invalid_argument("catcma needs categorical variables (cat_levels); without them: cmaes (CCR-3)");
    if (nca >= n) throw std::invalid_argument("catcma needs at least one continuous variable (CC-9)");
    if (P.n_cons > 0) throw std::invalid_argument("catcma: constraints are refused (CCR-3)");
    const std::size_t nco = n - nca;
    for (std::size_t j = 0; j < nco; ++j)
        if (!std::isfinite(P.lo[j]) || !std::isfinite(P.hi[j]) || !(P.hi[j] > P.lo[j]))
            throw std::invalid_argument("catcma needs a finite box for its continuous variables");
    for (std::size_t q = 0; q < nca; ++q)
        if (P.cat_levels[q] < 2) throw std::invalid_argument("catcma: every categorical variable needs K >= 2 levels");
    catcma::Options o;
    o.popsize = static_cast<int>(k.whole("catcma_popsize", 0, 0));
    const double sigma0 = k.number("catcma_sigma0", 0.3);
    o.update = k.word("catcma_update", "asng", {"asng", "printed"}) == "printed" ? catcma::Update::printed
                                                                                 : catcma::Update::asng;
    o.q_min = k.number("catcma_qmin", -1.0);
    Vec mean(nco, 0.5);                                                              // CCR-2
    if (start.has_x0) {
        if (start.x0.size() != n) throw std::invalid_argument("catcma: x0 has another length");
        for (std::size_t j = 0; j < nco; ++j)
            mean[j] = std::min(1.0, std::max(0.0, (start.x0[j] - P.lo[j]) / (P.hi[j] - P.lo[j])));
    }
    Evaluator ev(P, eq_tol, has_tol0, eq_tol0, budget, pop_size, std::move(observer), record_every);
    std::mt19937 rng(seed);
    catcma::CatCMA cma(mean, sigma0, P.cat_levels, rng, o);
    long long generations = 0;
    try {
        for (;;) {
            const std::vector<catcma::Point> pts = cma.ask();
            Vec f(pts.size());
            for (std::size_t i = 0; i < pts.size(); ++i) {
                Vec x(n);
                double out = 0.0;                                                    // CCR-1
                for (std::size_t j = 0; j < nco; ++j) {
                    const double u = pts[i].x[j], c = std::min(1.0, std::max(0.0, u));
                    out += (u - c) * (u - c);
                    x[j] = P.lo[j] + c * (P.hi[j] - P.lo[j]);
                }
                for (std::size_t q = 0; q < nca; ++q) x[nco + q] = static_cast<double>(pts[i].c[q]);
                const std::size_t at = ev.index(x);
                const double v = ev.F[at][0];
                f[i] = std::isfinite(v) ? v + out : std::numeric_limits<double>::max();   // a failure last
            }
            cma.tell(f);
            ++generations;
        }
    } catch (const BudgetSpent&) {                                                   // CCR-4
    }
    MinimaLog log(P.lo, P.hi, 1e-3, 1e-8);
    return search::finish(ev, k, log, {{"generations", std::to_string(generations)}});
}

}   // namespace mootation::scalar
