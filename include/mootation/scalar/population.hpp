#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// The population methods of stage 4 (de, jade, shade, lshade, ga, g3, chc,
// pso, spso2011, sep_cmaes) on the evaluator (task 18, stage 4 wiring; group
// 2 of stage 1's plan): every point a method asks for goes through
// Evaluator::index, so it is counted against the budget, kept in the archive
// and reported to the observer, and the best point's records are the
// evaluator's (improvements.hpp). The methods themselves stay pure ask/tell
// numerics (evo.hpp); scalar.hpp's run_population builds one from its knobs
// and hands it to drive() below.
//
// THE LOOP: ask() a generation, evaluate its points in order, tell() their
// values f and violations v (evaluator.hpp's cv), until the method's own stop
// or the budget.
//
// READINGS (declared):
//   POP-1. The budget can end inside a generation: the points evaluated so
//     far are in the archive (and may be the answer), the method is not told
//     the rest. A run is the same for every budget up to where the budget
//     stops it, except lshade's, whose population schedule is laid out over
//     MAX_NFE, the run's budget by default (lshade_max_nfe).
//   POP-2. A point the archive has already (a repeated point) is not
//     evaluated again and costs nothing (evaluator.hpp, THE BUDGET); the
//     method is told its value as for any other. So a generation can cost
//     less than its size, and one of points the archive has costs nothing.
//     After kIdleGenerations such generations in a row the run stops: the
//     method can only repeat itself (chc, whose generations can be empty
//     while its threshold falls, restarts well before).
//   POP-3. A failed evaluation (evaluator.hpp: a value not finite) is told as
//     NaN, f and v, which evo.hpp counts as +infinity (EVO-1).
//   POP-4. sep_cmaes works in the box normalized to [0, 1]^n, as cmaes does
//     in run_single (SINGLE-2), from the mean y0 with step cma_sigma0; a
//     point outside the box is evaluated at its clip and told the clip's f
//     plus the squared distance to it times (1 + |f|), cmaes's SCALAR-7 on
//     the problem's own f (no frame: the method only ranks). It stops when
//     sigma times the largest standard deviation is below local_tol, as
//     cmaes does.
// ============================================================================

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

#include "../operators/bound_repair.hpp"
#include "evaluator.hpp"

namespace mootation::scalar::population {

using Vec = std::vector<double>;

inline constexpr int kIdleGenerations = 1000;   // POP-2

// The values of a generation, point by point through the evaluator (POP-3);
// true when one of them was a new evaluation.
inline bool values(Evaluator& ev, const Rows& X, Vec& f, Vec& v)
{
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const long long before = ev.spent;
    f.assign(X.size(), nan);
    v.assign(X.size(), nan);
    for (std::size_t k = 0; k < X.size(); ++k) {
        const std::size_t i = ev.index(X[k]);       // BudgetSpent past the budget (POP-1)
        if (ev.failed[i]) continue;
        f[k] = ev.F[i][0];
        v[k] = ev.cv(i);
    }
    return ev.spent > before;
}

// ask/tell until stop() or the budget (BudgetSpent goes to the caller).
template <class Method, class Stop>
void drive(Method& method, Evaluator& ev, Stop stop)
{
    int idle = 0;
    bool first = true;
    while (!stop()) {
        const Rows X = method.ask();
        Vec f, v;
        const bool fresh = values(ev, X, f, v);
        if (first && !ev.recorded) ev.record();          // gen 0: the initial population
        first = false;
        idle = fresh ? 0 : idle + 1;
        if (idle >= kIdleGenerations) return;            // POP-2
        method.tell(f, v);
    }
}

// sep_cmaes in the normalized box (POP-4).
template <class Method>
void drive_normalized(Method& es, Evaluator& ev, double tol)
{
    const std::size_t n = ev.lo.size();
    int idle = 0;
    bool first = true;
    while (es.spread() > tol) {
        const Rows Y = es.ask();
        Rows X(Y.size(), Vec(n));
        Vec d2(Y.size(), 0.0);
        for (std::size_t k = 0; k < Y.size(); ++k) {
            ops::RepairTally tally;                              // the sample (task 20, §20.2.2)
            for (std::size_t j = 0; j < n; ++j) {
                const double yc = std::min(1.0, std::max(0.0, Y[k][j]));
                if (yc != Y[k][j]) tally.out();
                d2[k] += (Y[k][j] - yc) * (Y[k][j] - yc);
                X[k][j] = ev.lo[j] + yc * (ev.hi[j] - ev.lo[j]);
            }
        }
        Vec f, v;
        const bool fresh = values(ev, X, f, v);
        if (first && !ev.recorded) ev.record();
        first = false;
        idle = fresh ? 0 : idle + 1;
        if (idle >= kIdleGenerations) return;
        for (std::size_t k = 0; k < f.size(); ++k)
            if (std::isfinite(f[k])) f[k] += d2[k] * (1.0 + std::fabs(f[k]));
        es.tell(f, v);
    }
}

}   // namespace mootation::scalar::population
