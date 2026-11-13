#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// Strategies over subspaces of the variables (task 26, items 24.2.10 and
// 24.2.9's windows), on the subproblems of task 24 (hooks.hpp: restricted,
// embed; the problem's neighbors and contributions). Included by
// strategy.hpp.
//
// coevolution — cooperative coevolution with groups that overlap (M. A.
//   Potter, K. A. De Jong, "A cooperative coevolutionary approach to
//   function optimization", PPSN III, 1994; the groups of variables and
//   their overlap after S. Strasser, J. Sheppard, N. Fortier, R. Goodman,
//   "Factored evolutionary algorithms", IEEE TEVC 21(2), 2017 (not checked
//   on its text): a group is a variable and all the variables it
//   interacts with directly; overlapping sliding windows are the same idea
//   for a band of interactions). A context vector x holds the best point;
//   a cycle runs the inner solver on every group in turn, on the subproblem
//   "group free, the rest frozen at x" from x's values of the group, and
//   writes its answer into x when it is better (CC-1). Knobs: inner,
//   cc_groups = window (cc_window variables from every cc_stride-th on, the
//   last window ending at the last variable) | neighbors (the problem's
//   neighbors hook, CC-2), cc_window (10), cc_stride (0: half the window,
//   at least 1), cc_cycles (0: until the budget), start_budget (each
//   group's run; 0: what is left shared among the cycle's groups), min_dist,
//   min_df, keep_points.
//
// lns — large neighbourhood search by windows: from the run's start (its
//   x0, else a uniform point, CC-3), at every step a window of variables is
//   destroyed (drawn anew, uniformly in the box) and repaired by the inner
//   solver on the subproblem "window free, the rest frozen"; the result
//   replaces the current point when it is better (a monotone acceptance,
//   LNS-1). The window: lns_window = random (a variable drawn uniformly,
//   with its neighbors where the problem gives the hook, else the
//   lns_size - 1 variables after it, cyclically) | worst (the variable of
//   the largest contribution, the problem's contributions hook, LNS-2).
//   Knobs: inner, lns_window, lns_size (8), lns_steps (0: the budget),
//   start_budget, min_dist, min_df, keep_points.
//
// THE DEGENERATE CASES. coevolution with one group of all the variables
// (cc_window >= n, or cc_groups = neighbors where every variable meets
// all) and cc_cycles = 1, from an x0, is the inner solver from x0 with the
// whole budget, bit for bit (hooks.hpp RES-1: the subproblem of every
// variable in order is the problem itself; the seed is the run's, STRAT-1).
//
// READINGS:
//   CC-1. The groups are optimized one after another (Gauss-Seidel, as
//     Potter and De Jong's round robin), each from the context as the
//     earlier groups of the cycle left it; a variable in two groups takes
//     the value of the later one that improved. Strasser et al.'s
//     competition and sharing steps (a shared variable's value chosen among
//     the groups) are not made: with one context vector the later group
//     already starts from the earlier's value.
//   CC-2. cc_groups = neighbors: group j is {j} and neighbors(x)[j], sorted;
//     a group equal to an earlier one is dropped; the groups are taken anew
//     at the start of every cycle, from the context.
//   CC-3. Without an x0 the first context (and lns's first point) is a
//     uniform point of the box, from the run's stream; it is evaluated as
//     the first group's (or step's) start.
//   CC-4. A group's run is the inner solver's on the subproblem, its seed
//     start_seed(seed, s) for the s-th run (0 the first); its points are the
//     run's, in the run's budget (the views of strategy.hpp). The cycle ends
//     early when the budget is spent; the run ends on a cycle that spends
//     nothing (SRCH-10's idea, STRAT-8).
//   LNS-1. The destroyed point is the repair's start: the inner solver
//     starts from it, and the current point is replaced only by a better
//     answer (SRCH-8: feasibility, then f).
//   LNS-2. lns_window = worst needs the contributions hook; the window is
//     that variable with its neighbors (the hook) or the lns_size - 1 after
//     it; a tie for the largest contribution takes the first.
//   LNS-3. The draws per step: the window's variable (random) and then the
//     window's new values, from one stream seeded by the run's seed.
//   LNS-4. Without start_budget the first run (every variable, from the
//     start) gets a tenth of the budget and every repair a twentieth of
//     what is left, plus one: the paper-free choice of this library, so
//     that the first point is a local minimum and the repairs are many.
// ============================================================================

#include <algorithm>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "hooks.hpp"
#include "strategy.hpp"

namespace mootation::scalar {

namespace subspace {

using strategy::Inner;
using strategy::View;
using Groups = std::vector<std::vector<std::size_t>>;

// The inner solver on "idx free, the rest frozen at context", from the
// context's values (CC-4): its answer in the whole space.
inline search::Found sub_run(Evaluator& ev, const Inner& inner, const Vec& context, const Start& from,
                             const std::vector<std::size_t>& idx, long long allowance, unsigned seed,
                             int pop_size)
{
    View view(ev, static_cast<double>(allowance));
    const Problem Q = restricted(view.problem, context, idx);
    Start st = from;
    if (idx.size() != context.size() || !from.has_x0) {
        Vec z(idx.size());
        for (std::size_t k = 0; k < idx.size(); ++k) z[k] = context[idx[k]];
        st = search::at(z);
    }
    Result r;
    ++strategy::start_depth();
    try {
        r = run(inner.name, Q, ev.eq_tol_end, ev.has_tol0, ev.eq_tol0, pop_size, allowance, seed, inner.knobs, {},
                0, st);
    } catch (...) {
        --strategy::start_depth();
        throw;                                        // as strategy::round: an error is not merged
    }
    --strategy::start_depth();
    view.merge_into(ev);
    ev.hook_calls += r.hook_calls;
    ev.hook_spent += r.hook_spent;
    ev.hook_seconds += r.hook_seconds;
    ev.cache_hits += view.cache_hits();
    ev.delta_calls += r.delta_calls;
    ev.delta_spent += r.delta_spent;
    search::Found f = search::found(r);
    if (f.ok) f.x = embed(context, idx, f.x);
    return f;
}

inline Groups windows(std::size_t n, std::size_t w, std::size_t stride)
{
    Groups g;
    if (w >= n) {
        std::vector<std::size_t> all(n);
        for (std::size_t j = 0; j < n; ++j) all[j] = j;
        g.push_back(all);
        return g;
    }
    for (std::size_t s = 0;; s += stride) {
        const std::size_t a = std::min(s, n - w);              // the last window ends at the last variable
        std::vector<std::size_t> one(w);
        for (std::size_t k = 0; k < w; ++k) one[k] = a + k;
        g.push_back(one);
        if (a + w >= n) break;
    }
    return g;
}

inline Groups by_neighbors(const Problem& P, const Vec& x)                       // CC-2
{
    const std::vector<std::vector<std::size_t>> nb = P.neighbors(x);
    const std::size_t n = x.size();
    if (nb.size() != n) throw std::runtime_error("coevolution: the neighbors hook gave " + std::to_string(nb.size()) +
                                                 " lists for " + std::to_string(n) + " variables");
    Groups g;
    for (std::size_t j = 0; j < n; ++j) {
        std::vector<std::size_t> one = nb[j];
        one.push_back(j);
        std::sort(one.begin(), one.end());
        one.erase(std::unique(one.begin(), one.end()), one.end());
        for (std::size_t v : one)
            if (v >= n) throw std::runtime_error("coevolution: the neighbors hook named a variable past n");
        if (std::find(g.begin(), g.end(), one) == g.end()) g.push_back(one);
    }
    return g;
}

inline Vec first_point(const Problem& P, const Start& start, std::mt19937& rng)  // CC-3
{
    if (start.has_x0) return start.x0;
    Vec x(P.lo.size());
    for (std::size_t j = 0; j < x.size(); ++j) x[j] = P.lo[j] + rnd::uniform01(rng) * search::width(P, j);
    return x;
}

}   // namespace subspace

inline Result run_coevolution(const Problem& P, double eq_tol, bool has_tol0, double eq_tol0, int pop_size,
                              long long budget, unsigned seed, const Knobs& knobs, Observer observer, int record_every,
                              const Start& start)
{
    using namespace strategy;
    StrategyKnobs k("coevolution", knobs, {"inner", "cc_groups", "cc_window", "cc_stride", "cc_cycles",
                                           "start_budget", "min_dist", "min_df", "keep_points"});
    const Inner inner = k.inner("inner");
    const bool by_nb = k.word("cc_groups", "window", {"window", "neighbors"}) == "neighbors";
    const long long w = k.whole("cc_window", 10, 1), stride0 = k.whole("cc_stride", 0, 0);
    const long long cycles = k.whole("cc_cycles", 0, 0), cap = k.whole("start_budget", 0, 0);
    const double min_dist = k.number("min_dist", 1e-3), min_df = k.number("min_df", 1e-8);
    if (by_nb && !P.neighbors) throw std::invalid_argument("coevolution: cc_groups = neighbors needs the neighbors hook");
    const long long stride = stride0 > 0 ? stride0 : std::max(1LL, w / 2);
    if (stride > w) throw std::invalid_argument("coevolution: cc_stride must not exceed cc_window (the groups overlap)");
    search::finite_box("coevolution", P);
    Evaluator ev(P, eq_tol, has_tol0, eq_tol0, budget, pop_size, std::move(observer), record_every);
    ev.lean = lean_of(k);                                                            // STRAT-12
    MinimaLog log(P.lo, P.hi, min_dist, min_df);
    group_by_symmetry(k, P, log);
    std::mt19937 rng(start_seed(seed, 0xC0E0u));
    Vec context = subspace::first_point(P, start, rng);
    search::Found cur;
    long long runs = 0, done_cycles = 0, accepted = 0, groups_n = 0;
    try {
        while (search::left(ev) > 0 && (cycles == 0 || done_cycles < cycles)) {
            const double before = ev.used();
            const subspace::Groups G =
                by_nb ? subspace::by_neighbors(P, context)
                      : subspace::windows(context.size(), static_cast<std::size_t>(w), static_cast<std::size_t>(stride));
            groups_n = static_cast<long long>(G.size());
            for (std::size_t g = 0; g < G.size(); ++g) {
                const long long rest = search::left(ev);
                if (rest <= 0) break;
                const long long share = std::max(1LL, rest / static_cast<long long>(G.size() - g));
                const long long allowance = cap > 0 ? std::min(cap, rest) : share;
                const search::Found r = subspace::sub_run(ev, inner, context, runs == 0 ? start : Start{}, G[g],
                                                          allowance, start_seed(seed, static_cast<std::uint64_t>(runs)),
                                                          pop_size);
                if (r.ok) log.add(r.x, r.f, runs);
                ++runs;
                if (search::better(r, cur)) {                                        // CC-1
                    cur = r;
                    context = r.x;
                    ++accepted;
                }
            }
            ++done_cycles;
            if (ev.used() == before) break;                                          // CC-4
        }
    } catch (const BudgetSpent&) {
    }
    return search::finish(ev, k, log, {{"cycles", std::to_string(done_cycles)}, {"runs", std::to_string(runs)},
                                       {"accepted", std::to_string(accepted)}, {"groups", std::to_string(groups_n)}});
}

inline Result run_lns(const Problem& P, double eq_tol, bool has_tol0, double eq_tol0, int pop_size, long long budget,
                      unsigned seed, const Knobs& knobs, Observer observer, int record_every, const Start& start)
{
    using namespace strategy;
    StrategyKnobs k("lns", knobs, {"inner", "lns_window", "lns_size", "lns_steps", "start_budget", "min_dist",
                                   "min_df", "keep_points"});
    const Inner inner = k.inner("inner");
    const bool worst = k.word("lns_window", "random", {"random", "worst"}) == "worst";
    const long long size = k.whole("lns_size", 8, 1), steps = k.whole("lns_steps", 0, 0);
    const long long cap = k.whole("start_budget", 0, 0);
    const double min_dist = k.number("min_dist", 1e-3), min_df = k.number("min_df", 1e-8);
    if (worst && !P.contributions) throw std::invalid_argument("lns: lns_window = worst needs the contributions hook");
    search::finite_box("lns", P);
    const std::size_t n = P.lo.size();
    Evaluator ev(P, eq_tol, has_tol0, eq_tol0, budget, pop_size, std::move(observer), record_every);
    ev.lean = lean_of(k);
    MinimaLog log(P.lo, P.hi, min_dist, min_df);
    group_by_symmetry(k, P, log);
    std::mt19937 rng(start_seed(seed, 0x1A5u));                                      // LNS-3
    search::Found cur;
    long long runs = 0, done = 0, accepted = 0;
    auto window_at = [&](std::size_t j, const Vec& x) {
        std::vector<std::size_t> win;
        if (P.neighbors) {
            const auto nb = P.neighbors(x);
            if (nb.size() != n) throw std::runtime_error("lns: the neighbors hook gave a list per variable short");
            win = nb[j];
            win.push_back(j);
        } else {
            for (long long k2 = 0; k2 < size && static_cast<std::size_t>(k2) < n; ++k2)
                win.push_back((j + static_cast<std::size_t>(k2)) % n);
        }
        std::sort(win.begin(), win.end());
        win.erase(std::unique(win.begin(), win.end()), win.end());
        return win;
    };
    try {
        // the first point: the inner solver over every variable from the start (CC-3)
        const Vec x0 = subspace::first_point(P, start, rng);
        std::vector<std::size_t> all(n);
        for (std::size_t j = 0; j < n; ++j) all[j] = j;
        const long long first_allowance = cap > 0 ? std::min(cap, search::left(ev)) : std::max(1LL, search::left(ev) / 10);
        cur = subspace::sub_run(ev, inner, x0, start.has_x0 ? start : search::at(x0), all, first_allowance,
                                start_seed(seed, 0), pop_size);
        if (cur.ok) log.add(cur.x, cur.f, 0);
        runs = 1;
        if (!cur.ok) throw BudgetSpent();
        long long idle = 0;
        while (search::left(ev) > 0 && idle < 100 && (steps == 0 || done < steps)) {    // SRCH-10
            const double before = ev.used();
            std::size_t j = 0;
            if (worst) {                                                             // LNS-2
                Vec c(n, 0.0);
                P.contributions(cur.x, c);
                for (std::size_t q = 1; q < c.size() && q < n; ++q)
                    if (c[q] > c[j]) j = q;
            } else {
                j = std::min(n - 1, static_cast<std::size_t>(rnd::uniform01(rng) * static_cast<double>(n)));
            }
            const std::vector<std::size_t> win = window_at(j, cur.x);
            Vec destroyed = cur.x;                                                   // LNS-1
            for (std::size_t v : win) destroyed[v] = P.lo[v] + rnd::uniform01(rng) * search::width(P, v);
            const long long rest = search::left(ev);
            const long long allowance = cap > 0 ? std::min(cap, rest) : std::max(1LL, std::min(rest, rest / 20 + 1));
            const search::Found r = subspace::sub_run(ev, inner, destroyed, Start{}, win, allowance,
                                                      start_seed(seed, static_cast<std::uint64_t>(runs)), pop_size);
            ++runs;
            ++done;
            if (r.ok) log.add(r.x, r.f, runs - 1);
            if (search::better(r, cur)) {
                cur = r;
                ++accepted;
            }
            idle = ev.used() == before ? idle + 1 : 0;
        }
    } catch (const BudgetSpent&) {
    }
    return search::finish(ev, k, log, {{"steps", std::to_string(done)}, {"runs", std::to_string(runs)},
                                       {"accepted", std::to_string(accepted)}});
}

}   // namespace mootation::scalar
