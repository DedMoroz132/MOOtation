#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// reformulation: local searches in turn in several linked formulations of
// one problem (task 26, item 24.2.11) —
// reformulation descent, N. Mladenović, F. Plastria, D. Urošević,
// "Reformulation descent applied to circle packing problems", Comput. Oper.
// Res. 32 (2005) 2419-2434 (source reformulation-descent-applied_
// mladenovic2004), §2, steps 2-4: a local search in
// the current formulation; where it improves, back to the first
// formulation; where it does not, on to the next; the end when none
// improves. Included by strategy.hpp.
//
// The formulations: 0 is the problem itself; 1 ... n_formulations the
// problem's formulation(k), in variables y of their own, with to_formulation
// (x -> y) and from_formulation (y -> x) (evaluator.hpp). A search in
// formulation k starts from the current x mapped to y, runs the inner
// solver on formulation(k) (an Evaluator of its own; its evaluations and
// gradient calls added to the run's), and its answer mapped back is
// evaluated once in the problem: it improves when that value is lower
// (RF-2).
//
// THE DEGENERATE CASE: without other formulations the run is one local
// search in the problem from the run's start with the whole budget: the
// inner solver's run, bit for bit (RF-1).
//
// Knobs: inner, start_budget, rf_passes (0: until no formulation improves).
//
// READINGS:
//   RF-1. The first search is in formulation 0 from the run's start; then k
//     = 1; a search in k that improves sets k = 0 (step 3: "restart the full
//     loop ... from the first formulation"), else k + 1; k past
//     n_formulations ends. Step 3 read literally would search formulation 0
//     once more after its own first improvement; its answer is stationary
//     there already ("when only one formulation is available, RD simply
//     reduces to a single local optimization step", §2), so that repeat is
//     not made.
//   RF-2. Improvement is measured in the problem (formulation 0): the mapped
//     answer evaluated there, one evaluation counted. Constraints are
//     refused (a formulation's feasibility is its own).
//   RF-3. A search's budget: start_budget, else what is left over the
//     formulations' count plus one.
//   Not here: Locatelli and Schoen's "direct moves", a problem's
//   own composite move.
// ============================================================================

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "strategy.hpp"

namespace mootation::scalar {

inline Result run_reformulation(const Problem& P, double eq_tol, bool has_tol0, double eq_tol0, int pop_size,
                                long long budget, unsigned seed, const Knobs& knobs, Observer observer,
                                int record_every, const Start& start)
{
    using namespace strategy;
    (void)observer;
    (void)record_every;
    StrategyKnobs k("reformulation", knobs, {"inner", "start_budget", "rf_passes"});
    const Inner inner = k.inner("inner");
    const long long cap = k.whole("start_budget", 0, 0), passes = k.whole("rf_passes", 0, 0);
    const std::size_t K = P.n_formulations;
    if (K > 0 && (!P.formulation || !P.to_formulation || !P.from_formulation))
        throw std::invalid_argument("reformulation: n_formulations > 0 needs formulation, to_formulation and from_formulation");
    if (P.n_cons > 0) throw std::invalid_argument("reformulation: constraints are refused (RF-2)");
    long long spent = 0, grads = 0, searches = 0, improved = 0;
    auto allowance = [&]() {                                                         // RF-3
        const long long left = budget - spent;
        return std::max(1LL, cap > 0 ? std::min(cap, left) : left / static_cast<long long>(K + 1));
    };
    // 0. the first search, in the problem from the run's start
    const long long a0 = K == 0 ? budget : allowance();
    Result best = run(inner.name, P, eq_tol, has_tol0, eq_tol0, pop_size, a0, seed, inner.knobs, {}, 0, start);
    spent += best.evaluations;
    grads += best.grad_calls;
    ++searches;
    if (!best.objectives.empty() && K > 0) {
        Vec x = best.variables[0];
        double f = best.objectives[0][0];
        std::size_t q = 1;
        long long pass = 0;
        while (q <= K && budget - spent > 0 && (passes == 0 || pass < passes)) {
            Result r;
            Vec back;
            if (q == 0) {
                r = run(inner.name, P, eq_tol, has_tol0, eq_tol0, pop_size, allowance(),
                        start_seed(seed, static_cast<std::uint64_t>(searches)), inner.knobs, {}, 0, search::at(x));
                if (!r.variables.empty()) back = r.variables[0];
            } else {
                const Problem F = P.formulation(q);
                Vec y;
                P.to_formulation(q, x, y);
                if (y.size() != F.lo.size()) throw std::runtime_error("reformulation: to_formulation gave y of another length");
                r = run(inner.name, F, eq_tol, has_tol0, eq_tol0, pop_size, allowance(),
                        start_seed(seed, static_cast<std::uint64_t>(searches)), inner.knobs, {}, 0, search::at(y));
                if (!r.variables.empty()) {
                    P.from_formulation(q, r.variables[0], back);
                    if (back.size() != x.size()) throw std::runtime_error("reformulation: from_formulation gave x of another length");
                    for (std::size_t j = 0; j < back.size(); ++j) back[j] = std::min(P.hi[j], std::max(P.lo[j], back[j]));
                }
            }
            spent += r.evaluations;
            grads += r.grad_calls;
            ++searches;
            double fb = std::numeric_limits<double>::infinity();
            if (!back.empty()) {                                                     // RF-2
                Vec fv, gv;
                P.evaluate(back, fv, gv);
                ++spent;
                fb = fv.empty() ? fb : fv[0];
            }
            if (fb < f) {
                f = fb;
                x = back;
                ++improved;
                best.objectives = {Vec{fb}};
                best.variables = {back};
                q = 0;
                ++pass;
            } else {
                ++q;
            }
        }
    }
    best.evaluations = spent;
    best.grad_calls = grads;
    best.ignored = k.ignored();
    best.settings.push_back({"searches", std::to_string(searches)});
    best.settings.push_back({"improved", std::to_string(improved)});
    return best;
}

}   // namespace mootation::scalar
