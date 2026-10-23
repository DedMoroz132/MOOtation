#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// Polishing at the end of the budget (task 9, A2): any algorithm, then a
// local solver on some of the points it found.
//
//   polish         off (the default: nothing changes) | sqp | nm
//   polish_share   the share of max_evaluations kept for the polishing (0.2);
//                  the algorithm runs on the rest
//   polish_points  how many points to polish; 0: from what the share buys, as
//                  n_sub (scalar, SCALAR-1), at most pop_size
//
// THE POINTS are the feasible non-dominated points of every evaluation the
// algorithm made, chosen by DSS (dss.hpp) when there are more; without a
// feasible point, the least violating ones.
//
// POLISHING a point a is the ε-constraint subproblem "improve one objective,
// the others no worse than at a": min f_k  s.t.  f_i <= f_i(a) for every i,
// started at a, with k going round the objectives from one point to the next
// (POLISH-1). Every point it evaluates goes to the archive; a itself is
// feasible for it, and its result is no worse than a up to the subproblem's
// slack (POLISH-3).
//
// THE ANSWER is the non-dominated points of the algorithm's answer and the
// polished points; when there are more than pop_size, first one point weakly
// dominating each point of the algorithm's answer, then the rest in DSS order
// (POLISH-2). A polished point replaces what it dominates and nothing of the
// algorithm's answer is lost. The observer goes on after the algorithm's last
// record, every record_every * pop_size evaluations.
//
// READINGS:
//   POLISH-1. The objective improved goes round the objectives (the first
//     point f_1, the second f_2, ...) in DSS order; the objectives are
//     normalized by the range of the points chosen.
//   POLISH-2. The answer after polishing is the non-dominated union of the
//     algorithm's answer and the polished points, not the archive's; cut to
//     pop_size, it keeps a point weakly dominating each point of the
//     algorithm's answer first (so it can hold more than pop_size when the
//     algorithm's answer does).
//   POLISH-3. The subproblem's constraints f_i <= f_i(a) hold to a slack
//     (SLACK, in the normalized objectives): a result a hair worse than a in
//     one objective does not dominate a, and both stay.
// ============================================================================

#include <algorithm>
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include "../dss.hpp"
#include "../fp_contract.hpp"
#include "solvers.hpp"

namespace mootation::scalar {

struct PolishResult {
    Rows objectives, variables;
    Vec cv;
    long long spent = 0;          // the polishing's evaluations
    std::size_t chosen = 0;       // the points polished
};

// The algorithm's evaluations (points), its answer's variables (base), the
// evaluations it spent (base_evaluations); how: "sqp" or "nm".
inline PolishResult polish(const Problem& P, const std::vector<Point>& points, const Rows& base,
                           long long base_evaluations, const std::string& how, long long n_points,
                           double eq_tol, int pop_size, long long budget, Observer observer,
                           int record_every, double fd_step, double local_tol)
{
    const std::size_t m = static_cast<std::size_t>(P.m), n = P.lo.size();
    Evaluator ev(P, eq_tol, false, 0.0, budget, pop_size, std::move(observer), record_every);
    ev.recorded = true;                              // the algorithm has had gen 0
    const long long p = std::max(1, pop_size);
    ev.gen0 = (base_evaluations + p - 1) / p;
    for (const Point& pt : points) ev.plant(pt.x, pt.f, pt.g);
    std::vector<std::size_t> base_idx;
    for (const Vec& x : base)
        if (ev.known(x)) base_idx.push_back(ev.find(x));
    std::vector<std::size_t> chosen = ev.answer();
    const long long k_pts = n_points ? n_points
        : std::min<long long>(pop_size, std::max<long long>(1, budget / (10 * static_cast<long long>(n + 1))));
    if (static_cast<long long>(chosen.size()) > k_pts) chosen.resize(static_cast<std::size_t>(k_pts));
    std::vector<long long> polished;
    if (!chosen.empty()) {
        Vec lo = ev.F[chosen[0]], hi = ev.F[chosen[0]];
        for (std::size_t i : chosen)
            for (std::size_t j = 0; j < m; ++j) {
                lo[j] = std::min(lo[j], ev.F[i][j]);
                hi[j] = std::max(hi[j], ev.F[i][j]);
            }
        const Frame frame = Frame::from_anchors({lo, hi});
        try {
            for (std::size_t j = 0; j < chosen.size(); ++j) {
                const std::size_t a = chosen[j];
                const Vec Fa = frame.bar(ev.F[a]);
                Sub sub;
                sub.kind = "polish";
                sub.a0 = unit(m, j % m);
                for (std::size_t i = 0; i < m; ++i) sub.ineq.push_back({unit(m, i, -1.0), 0.0, Fa[i]});
                const long long left = ev.budget - ev.spent;
                ev.limit = ev.spent + std::max<long long>(1, left / static_cast<long long>(chosen.size() - j));
                Run run(sub, ev, frame);
                run.touched.push_back(a);
                const Vec y0 = run.y_of(a);
                try {
                    if (how == "sqp") sqp(run, y0, 0.0, fd_step, local_tol);
                    else nm(run, y0, {a}, local_tol);
                } catch (const SubBudgetSpent&) {
                } catch (...) {
                    ev.limit = ev.budget;
                    throw;
                }
                ev.limit = ev.budget;
                polished.push_back(best_of(run, run.touched));
            }
        } catch (const BudgetSpent&) {
        }
    }
    ev.over = true;
    // the answer: the algorithm's answer and the polished points, non-dominated
    std::vector<std::size_t> pool;
    auto add = [&](std::size_t i) {
        if (std::find(pool.begin(), pool.end(), i) == pool.end()) pool.push_back(i);
    };
    for (std::size_t i : base_idx) add(i);
    for (long long i : polished)
        if (i >= 0) add(static_cast<std::size_t>(i));
    std::vector<std::size_t> ok, feas;
    for (std::size_t i : pool)
        if (!ev.failed[i]) ok.push_back(i);
    for (std::size_t i : ok)
        if (ev.cv(i) <= 0.0) feas.push_back(i);
    auto weakly_below = [&](const Vec& a, const Vec& b) {    // a <= b everywhere
        for (std::size_t j = 0; j < m; ++j)
            if (!(a[j] <= b[j])) return false;
        return true;
    };
    std::vector<std::size_t> idx;
    if (!feas.empty()) {
        std::vector<std::size_t> nd;
        for (std::size_t a : feas) {
            bool dominated = false;
            for (std::size_t b : feas)
                if (weakly_below(ev.F[b], ev.F[a]) && ev.F[b] != ev.F[a]) { dominated = true; break; }
            if (!dominated && std::find(nd.begin(), nd.end(), a) == nd.end()) nd.push_back(a);
        }
        if (nd.size() > static_cast<std::size_t>(pop_size)) {
            // first, for every feasible point of the algorithm's answer, one
            // point that weakly dominates it; the rest in DSS order (POLISH-2)
            Rows Fn;
            for (std::size_t i : nd) Fn.push_back(ev.F[i]);
            std::vector<std::size_t> keep;
            for (std::size_t b : base_idx) {
                if (ev.failed[b] || ev.cv(b) > 0.0) continue;
                std::vector<std::size_t> cover;
                for (std::size_t j = 0; j < nd.size(); ++j)
                    if (weakly_below(Fn[j], ev.F[b])) cover.push_back(j);
                bool have = false;
                for (std::size_t j : cover) have = have || std::find(keep.begin(), keep.end(), j) != keep.end();
                if (!have) keep.push_back(cover[0]);
            }
            std::vector<std::size_t> rest;
            for (std::size_t j : dss::order(Fn, Fn.size()))
                if (std::find(keep.begin(), keep.end(), j) == keep.end()) rest.push_back(j);
            const std::size_t room = keep.size() < static_cast<std::size_t>(pop_size)
                                         ? static_cast<std::size_t>(pop_size) - keep.size() : 0;
            std::vector<std::size_t> pick = keep;
            for (std::size_t t = 0; t < std::min(room, rest.size()); ++t) pick.push_back(rest[t]);
            std::vector<std::size_t> cut;
            for (std::size_t j : pick) cut.push_back(nd[j]);
            nd = cut;
        }
        idx = nd;
    } else {
        idx = ok;
        std::stable_sort(idx.begin(), idx.end(), [&](std::size_t a, std::size_t b) {
            const double ca = ev.cv(a), cb = ev.cv(b);
            return ca < cb || (ca == cb && a < b);
        });
        if (idx.size() > static_cast<std::size_t>(pop_size)) idx.resize(static_cast<std::size_t>(pop_size));
    }
    PolishResult r;
    for (std::size_t i : idx) {
        r.objectives.push_back(ev.F[i]);
        r.variables.push_back(ev.X[i]);
        r.cv.push_back(ev.cv(i));
    }
    if (ev.observer) ev.observer(ev.gen0 + (ev.spent + p - 1) / p, r.objectives, r.variables);
    r.spent = ev.spent;
    r.chosen = chosen.size();
    return r;
}

}   // namespace mootation::scalar
