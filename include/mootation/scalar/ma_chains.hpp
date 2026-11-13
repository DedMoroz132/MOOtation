#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// ma_chains: memetic algorithms with local search chains (task 25, §24.2.1;
// АП-1 of the preparation). D. Molina, M. Lozano, C. García-Martínez, F.
// Herrera, "Memetic algorithms for continuous optimisation based on local
// search chains", Evol. Comput. 18(1) (2010) (source memetic-algorithms-
// continuous_molina2010; "M10"), §4, Figure 4; D. Molina, M. Lozano, A.M.
// Sánchez, F. Herrera, "MA-SSW-Chains", Soft Comput. 15 (2011) (source
// memetic-algorithms-local_molina2010; "M11"), §4. Included by strategy.hpp.
//
//   1. A population of ma_pop points (the first the run's start where it has
//      one, the others uniform in the box).
//   2. The steady-state GA for n_freq = I_str (1 - r_LG) / r_LG evaluations:
//      a parent at random, n_ass more at random and the one farthest from the
//      first (Euclidean, the box normalized) the second (negative assortative
//      mating, M10 §4.4.3); a child by BLX-alpha (alpha = 0.5, §4.4.2); with
//      probability ma_pmut (0.125) one random gene by BGA's mutation c +- rang
//      sum_{k=0}^{15} alpha_k 2^-k, rang = 0.1 (b - a), P(alpha_k = 1) =
//      1/16 (§4.4.4); the child replaces the worst member when better (§4.1).
//   3. S_LS: the members never improved by a local search, or improved by more
//      than delta_min at their last one (§4.3). The best of S_LS is searched;
//      if S_LS is empty the population restarts (MAC-1).
//   4. The local search from it for I_str evaluations, continuing its chain
//      (its state kept with the member) where it has one; the result written
//      back (lamarck = on), or only its value (lamarck = off, Baldwinian), or
//      written with probability lamarck_share (share; Houck et al. 1997).
//   5. Again from 2 until the budget.
// The local searches: ma_local = sw (Solis-Wets, MA-SW-Chains), ssw (M11's
// subgrouping Solis-Wets), mts_ls1, mts_ls2 (chain_ls.hpp, in the box
// normalized to [0, 1]^n, their state packed with the member), or inner (the
// strategy's inner solver from the member with I_str evaluations; with inner
// = {"solver": "local_search"} the problem's own local search, its state
// passed through Start::ls_state and Result::ls_state, task 24 §24.1.1).
//
// Knobs and defaults (M10 §4.4.7, M11 §4.1.3): ma_local (sw), ma_pop (60;
// M11: 100), ma_istr (500), ma_rlg (0.5; M11: 0.8 for D >= 100), ma_delta_min
// (1e-8; M11: 0), ma_restart (on | off), ma_nass (3), ma_alpha (0.5),
// ma_pmut (0.125), lamarck (on | off | share), lamarck_share (0.25), inner
// (for ma_local = inner), min_dist, min_df, keep_points.
//
// READINGS:
//   MAC-1. An empty S_LS restarts the population at random keeping the best
//     (CEC 2010 and M11; M10's ECJ text searches the best member again):
//     ma_restart = on by default, off takes M10's rule.
//   MAC-2. BGA mutates one gene, chosen uniformly ("the gene to be mutated",
//     M10 §4.4.4); the first parent and the n_ass candidates are uniform (M10
//     §4.4.3: "in our experiments all the parents are selected at random").
//   MAC-3. The first value of a chain's step size (M11 §4.2, §4.4: half the
//     distance to the nearest member) is taken in the normalized box, at
//     least 1e-12; a member without a chain (never searched, or a new child)
//     starts one.
//   MAC-4. n_freq rounds to the nearest whole number, at least 1; the GA's
//     share is counted in evaluations spent (a repeated point is free).
//   MAC-5. A child or a restarted member starts without a chain; a member
//     replaced by a local search's result keeps the chain of that search.
//   MAC-7. A GA phase stops early after 100 n_freq children in a row that
//     spend nothing (all repeats), and a phase (GA and local search) that
//     spends nothing ends the run (as SRCH-9, SRCH-10). A local search's
//     improvement (S_LS) is its decrease of f when the violation stays, else
//     infinite if it became feasible-better, else none; a violating point
//     counts as worse than every feasible one in chain_ls's searches.
//   MAC-6. The strategy's draws come from mt19937(start_seed(seed,
//     0x4D41434Cu)), the coins of lamarck = share from start_seed(seed, 1 <<
//     32) as memetic's (SRCH-12); ma_local = inner's s-th search has the seed
//     start_seed(seed, s).
// ============================================================================

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "chain_ls.hpp"
#include "strategy.hpp"

namespace mootation::scalar {

inline Result run_ma_chains(const Problem& P, double eq_tol, bool has_tol0, double eq_tol0, int pop_size,
                            long long budget, unsigned seed, const Knobs& knobs, Observer observer, int record_every,
                            const Start& start)
{
    using namespace strategy;
    StrategyKnobs k("ma_chains", knobs,
                    {"inner", "ma_local", "ma_pop", "ma_istr", "ma_rlg", "ma_delta_min", "ma_restart", "ma_nass",
                     "ma_alpha", "ma_pmut", "lamarck", "lamarck_share", "min_dist", "min_df", "keep_points"});
    const std::string ls = k.word("ma_local", "sw", {"sw", "ssw", "mts_ls1", "mts_ls2", "inner"});
    const long long N = k.whole("ma_pop", 60, 2), istr = k.whole("ma_istr", 500, 1), nass = k.whole("ma_nass", 3, 1);
    const double rlg = k.number("ma_rlg", 0.5), delta_min = k.number("ma_delta_min", 1e-8);
    const bool restart = k.word("ma_restart", "on", {"on", "off"}) == "on";
    const double alpha = k.number("ma_alpha", 0.5), pmut = k.number("ma_pmut", 0.125);
    const std::string lamarck = k.word("lamarck", "on", {"off", "on", "share"});
    const double share = lamarck == "share" ? k.number("lamarck_share", 0.25) : 1.0;
    const double min_dist = k.number("min_dist", 1e-3), min_df = k.number("min_df", 1e-8);
    if (!(rlg > 0.0 && rlg < 1.0)) throw std::invalid_argument("ma_chains: ma_rlg must be in (0, 1)");
    if (!(share >= 0.0 && share <= 1.0)) throw std::invalid_argument("ma_chains: lamarck_share must be in [0, 1]");
    if (!(alpha >= 0.0) || !(pmut >= 0.0 && pmut <= 1.0))
        throw std::invalid_argument("ma_chains: ma_alpha >= 0 and ma_pmut in [0, 1]");
    Inner inner;
    if (ls == "inner") inner = k.inner("inner");
    search::finite_box("ma_chains", P);
    const std::size_t n = P.lo.size();
    const long long n_freq = std::max(1LL, std::llround(static_cast<double>(istr) * (1.0 - rlg) / rlg));   // MAC-4
    Evaluator ev(P, eq_tol, has_tol0, eq_tol0, budget, pop_size, std::move(observer), record_every);
    ev.lean = lean_of(k);                                                            // STRAT-12
    MinimaLog log(P.lo, P.hi, min_dist, min_df);
    strategy::group_by_symmetry(k, P, log);
    std::mt19937 rng(start_seed(seed, 0x4D41434Cu)), coins(start_seed(seed, std::uint64_t{1} << 32));   // MAC-6

    struct Member {
        Vec x;
        double f = 0.0, cv = 0.0;
        Vec state;                       // the local search's chain (MAC-5)
        bool searched = false;
        double gain = 0.0;
    };
    auto better = [](const Member& a, const Member& b) { return a.cv != b.cv ? a.cv < b.cv : a.f < b.f; };
    auto eval = [&](const Vec& x, Member& m) {
        const std::size_t i = ev.index(x);
        m.x = x;
        m.f = ev.F[i][0];
        m.cv = ev.cv(i);
    };
    auto w = [&](std::size_t j) { return search::width(P, j); };
    auto clip = [&](Vec& y) {
        for (std::size_t j = 0; j < n; ++j)
            y[j] = periodic::is(P.periodic, j) ? periodic::wrap_into(y[j], P.lo[j], P.hi[j])
                                               : std::min(P.hi[j], std::max(P.lo[j], y[j]));
    };
    auto ndist = [&](const Vec& a, const Vec& b) {                                  // the normalized box
        double s = 0.0;
        for (std::size_t j = 0; j < n; ++j) {
            const double g = periodic::gap(a[j], b[j], periodic::is(P.periodic, j), w(j)) / w(j);
            s += g * g;
        }
        return std::sqrt(s);
    };
    auto uniform_point = [&]() {
        Vec x(n);
        for (std::size_t j = 0; j < n; ++j) x[j] = P.lo[j] + rnd::uniform01(rng) * w(j);
        return x;
    };
    std::vector<Member> pop;
    long long applications = 0, restarts = 0, searches = 0;

    // 4. one local search from member b (chained), written back by lamarck
    auto local_search = [&](std::size_t b) {
        Member& m = pop[b];
        const double before_f = m.f;
        Vec x = m.x;
        double fx = m.f;
        Vec state = m.state;
        if (ls == "inner") {
            StartSpec sp;
            sp.seed = start_seed(seed, static_cast<std::uint64_t>(searches++));
            sp.start = search::at(m.x);
            sp.start.ls_state = state;
            const std::vector<Result> r = search::starts(ev, inner, {sp}, istr, 1, pop_size);
            if (r.empty()) throw BudgetSpent();
            const search::Found fo = search::found(r[0]);
            if (!fo.ok) throw BudgetSpent();
            x = fo.x;
            fx = fo.f;
            state = r[0].ls_state;
        } else {
            // chain_ls in the normalized box: u = (x - lo) / w
            Vec u(n), lo0(n, 0.0), hi1(n, 1.0);
            for (std::size_t j = 0; j < n; ++j) u[j] = (m.x[j] - P.lo[j]) / w(j);
            auto f = [&](const Vec& v) {
                Vec y(n);
                for (std::size_t j = 0; j < n; ++j) y[j] = P.lo[j] + v[j] * w(j);
                const std::size_t i = ev.index(y);
                // feasibility first: a violating point is worse than any feasible one
                return ev.cv(i) > 0.0 ? std::numeric_limits<double>::max() / 4.0 + ev.cv(i) : ev.F[i][0];
            };
            double fu = m.cv > 0.0 ? std::numeric_limits<double>::max() / 4.0 + m.cv : m.f;
            double nearest = std::numeric_limits<double>::infinity();               // MAC-3
            for (std::size_t q = 0; q < pop.size(); ++q)
                if (q != b) nearest = std::min(nearest, ndist(pop[q].x, m.x));
            const double first = std::max(1e-12, std::isfinite(nearest) ? nearest / 2.0 : 0.1);
            if (ls == "sw" || ls == "ssw") {
                chain_ls::SWOptions o;
                o.subgroups = ls == "ssw";
                chain_ls::SWState st = chain_ls::SWState::unpack(state, n);
                if (st.rho < 0.0) o.rho0 = first;
                chain_ls::solis_wets(f, u, fu, st, istr, lo0, hi1, rng, o);
                state = st.pack();
            } else {
                chain_ls::MTSOptions o;
                chain_ls::MTSState st = chain_ls::MTSState::unpack(state, n);
                if (st.sr < 0.0) o.sr_init = first;
                if (ls == "mts_ls1") chain_ls::mts_ls1(f, u, fu, st, istr, lo0, hi1, o);
                else chain_ls::mts_ls2(f, u, fu, st, istr, lo0, hi1, rng, o);
                state = st.pack();
            }
            for (std::size_t j = 0; j < n; ++j) x[j] = P.lo[j] + u[j] * w(j);
        }
        Member r;
        eval(x, r);                                                                  // known: free
        ++applications;
        log.add(r.x, r.f, applications);
        m.gain = r.cv == m.cv ? before_f - r.f : (better(r, m) ? std::numeric_limits<double>::infinity() : 0.0);
        m.searched = true;
        m.state = state;                                                             // MAC-5
        const bool write = lamarck == "on" || (lamarck == "share" && rnd::uniform01(coins) < share);
        if (better(r, m)) {
            if (write) m.x = r.x;                                                    // Lamarck
            m.f = r.f;                                                               // Baldwin: the value only
            m.cv = r.cv;
        }
        (void)fx;
    };

    try {
        // 1.
        for (long long p = 0; p < N; ++p) {
            Member m;
            eval(p == 0 && start.has_x0 ? start.x0 : uniform_point(), m);
            pop.push_back(m);
        }
        for (;;) {
            // 2. the steady-state GA for n_freq evaluations
            const double ga_from = ev.used();
            long long ga_idle = 0;                                                     // MAC-4: repeats are free
            while (ev.used() - ga_from < static_cast<double>(n_freq) && ga_idle < 100 * n_freq) {
                const double step_from = ev.used();
                const std::size_t a = static_cast<std::size_t>(rnd::below(rng, static_cast<std::uint64_t>(N)));
                std::size_t b = a;
                double far = -1.0;
                for (long long c = 0; c < nass; ++c) {                                 // MAC-2
                    std::size_t q = static_cast<std::size_t>(rnd::below(rng, static_cast<std::uint64_t>(N)));
                    const double d = ndist(pop[q].x, pop[a].x);
                    if (d > far) {
                        far = d;
                        b = q;
                    }
                }
                Vec z(n);
                for (std::size_t j = 0; j < n; ++j) {                                   // BLX-alpha
                    const double lo = std::min(pop[a].x[j], pop[b].x[j]), hi = std::max(pop[a].x[j], pop[b].x[j]);
                    const double I = hi - lo;
                    z[j] = lo - I * alpha + rnd::uniform01(rng) * (I * (1.0 + 2.0 * alpha));
                }
                if (rnd::uniform01(rng) < pmut) {                                       // BGA (MAC-2)
                    const std::size_t j = static_cast<std::size_t>(rnd::below(rng, static_cast<std::uint64_t>(n)));
                    double s = 0.0;
                    for (int kk = 0; kk <= 15; ++kk)
                        if (rnd::below(rng, 16) == 0) s += std::ldexp(1.0, -kk);
                    const double sign = rnd::below(rng, 2) == 0 ? 1.0 : -1.0;
                    z[j] += sign * 0.1 * w(j) * s;
                }
                clip(z);
                Member child;
                eval(z, child);
                std::size_t worst = 0;
                for (std::size_t q = 1; q < pop.size(); ++q)
                    if (better(pop[worst], pop[q])) worst = q;
                if (better(child, pop[worst])) pop[worst] = child;                     // MAC-5: no chain
                ga_idle = ev.used() == step_from ? ga_idle + 1 : 0;
            }
            // 3. S_LS and its best
            std::size_t pick = pop.size();
            for (std::size_t q = 0; q < pop.size(); ++q)
                if ((!pop[q].searched || pop[q].gain > delta_min) && (pick == pop.size() || better(pop[q], pop[pick])))
                    pick = q;
            if (pick == pop.size()) {
                std::size_t best = 0;
                for (std::size_t q = 1; q < pop.size(); ++q)
                    if (better(pop[q], pop[best])) best = q;
                if (restart) {                                                          // MAC-1
                    ++restarts;
                    for (std::size_t q = 0; q < pop.size(); ++q) {
                        if (q == best) continue;
                        Member m;
                        eval(uniform_point(), m);
                        pop[q] = m;
                    }
                    pick = best == 0 ? 1 : 0;
                    for (std::size_t q = 0; q < pop.size(); ++q)
                        if (q != best && better(pop[q], pop[pick])) pick = q;
                } else {
                    pick = best;
                }
            }
            const double ls_from = ev.used();
            local_search(pick);                                                         // 4.
            if (ev.used() == ga_from && ev.used() == ls_from) break;                    // a phase that spends nothing (SRCH-9)
        }
    } catch (const BudgetSpent&) {
    }
    return search::finish(ev, k, log,
                          {{"ls_applications", std::to_string(applications)}, {"restarts", std::to_string(restarts)},
                           {"n_freq", std::to_string(n_freq)}});
}

}   // namespace mootation::scalar
