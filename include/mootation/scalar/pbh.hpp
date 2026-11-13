#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// pbh: Population Basin Hopping (task 25, §24.2.2; АП-8 of the preparation):
// M. Grosso, M. Locatelli, F. Schoen, "A population-based approach for hard
// global optimization problems based on dissimilarity measures", Math.
// Program. 110 (2007), §2.3 (source population-dissimilarity-global-opt_
// grosso2007_preprint; "GLS07"), on the rounds of strategy.hpp. Included by
// strategy.hpp.
//
//   0. K local minima: the first from the run's start (as ils), the others
//      from uniform points of the box.
//   1. Every member X_p gives a child Y_p = L(X_p + Delta): Delta the move of
//      ils (every coordinate by U[-1, 1] times pbh_step times the box's width,
//      or in the problem's units with step_units = absolute; clipped, or
//      wrapped on a periodic coordinate), L the inner solver.
//   2. In the members' order, each child against the population as it then
//      is: X_q the member nearest to Y_p by the dissimilarity d; if d < d_cut
//      and Y_p is better than X_q, Y_p replaces X_q; if d >= d_cut and Y_p
//      is better than the worst member, Y_p replaces the worst; else it is
//      dropped.
//   3. Again from 1 until the budget, pbh_maxstep iterations, or 100
//      iterations in a row that spend nothing (SRCH-10).
// With K = 1 and d_cut = inf it is monotonic basin hopping, MBH: ils with
// ils_accept = better (GLS07 §2.3), and the run is ils's bit for bit.
//
// Knobs: inner, pbh_K (10), pbh_step (0.1), step_units (box | absolute),
// pbh_distance = fdiff (|f(X) - f(Y)|, GLS07's measure for Schwefel's
// function) | euclid (the Euclidean distance in the box normalized to [0,
// 1]^n, the short way round a periodic coordinate) | aligned (the problem's
// align to y, else symmetry::distance on sym_blocks and canon, SYM-11),
// pbh_dcut (0: inf; c > 0: c times the mean pairwise d of the first
// population, GLS07 §4: 0.5 for d_ord^3, 1.5 for d_H), pbh_dcut_anneal (off |
// on: d_cut times 0.85 every 50 iterations in the first 250, 2P-PBH*),
// pbh_maxstep (0: the budget), threads, start_budget, min_dist, min_df,
// keep_points.
//
// READINGS:
//   PBH-1. Step 2 against the worst compares f(Y_p) (the preprint prints
//     f(Y_q)).
//   PBH-2. "Near" is d < d_cut, as the preprint (the journal's text has <=).
//   PBH-3. The children are made from the population at the iteration's
//     start (their local searches one round of K starts) and placed one
//     after the other in the members' order.
//   PBH-4. The draws: per child its n coordinates' moves and then one more
//     uniform, as ils draws its acceptance's u at every step, so that K = 1
//     follows ils's stream; the extra members' uniform points come from the
//     same stream before the first iteration. The local search of the s-th
//     search (0 the first member's) has the seed start_seed(seed, s), as ils.
//   PBH-5. Ties: the nearest and the worst are the first in the members'
//     order; "better" is SRCH-8's (feasibility, then f).
// ============================================================================

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "strategy.hpp"

namespace mootation::scalar {

inline Result run_pbh(const Problem& P, double eq_tol, bool has_tol0, double eq_tol0, int pop_size, long long budget,
                      unsigned seed, const Knobs& knobs, Observer observer, int record_every, const Start& start)
{
    using namespace strategy;
    StrategyKnobs k("pbh", knobs,
                    {"inner", "pbh_K", "pbh_step", "step_units", "pbh_distance", "pbh_dcut", "pbh_dcut_anneal",
                     "pbh_maxstep", "threads", "start_budget", "min_dist", "min_df", "keep_points"});
    const Inner inner = k.inner("inner");
    const long long K = k.whole("pbh_K", 10, 1), threads = k.whole("threads", 1, 1);
    const double step = k.number("pbh_step", 0.1);
    const bool absolute = k.word("step_units", "box", {"box", "absolute"}) == "absolute";
    const std::string dist = k.word("pbh_distance", "fdiff", {"fdiff", "euclid", "aligned"});
    const double dcut_c = k.number("pbh_dcut", 0.0);
    const bool anneal = k.word("pbh_dcut_anneal", "off", {"off", "on"}) == "on";
    const long long maxstep = k.whole("pbh_maxstep", 0, 0);
    const long long cap = k.whole("start_budget", 0, 0);
    const double min_dist = k.number("min_dist", 1e-3), min_df = k.number("min_df", 1e-8);
    if (!(step > 0.0)) throw std::invalid_argument("pbh: pbh_step must be > 0");
    if (!(dcut_c >= 0.0)) throw std::invalid_argument("pbh: pbh_dcut must be >= 0 (0: inf)");
    search::finite_box("pbh", P);
    const std::size_t n = P.lo.size();
    Evaluator ev(P, eq_tol, has_tol0, eq_tol0, budget, pop_size, std::move(observer), record_every);
    ev.lean = lean_of(k);                                                            // STRAT-12
    MinimaLog log(P.lo, P.hi, min_dist, min_df);
    strategy::group_by_symmetry(k, P, log);
    std::mt19937 rng(start_seed(seed, 0x9E3779B9u));                                 // PBH-4: ils's stream
    const Vec period = symmetry::periods(P.periodic, P.lo, P.hi);

    auto d = [&](const search::Found& a, const search::Found& b) {
        if (dist == "fdiff") return std::fabs(a.f - b.f);
        Vec x = a.x;
        if (dist == "aligned") {
            if (P.align) {
                Vec out;
                P.align(a.x, b.x, out);
                x = out;
            } else if (!P.sym_blocks.empty()) {
                return symmetry::distance(a.x, b.x, P.sym_blocks, P.canon, period);
            }
        }
        double s = 0.0;
        for (std::size_t j = 0; j < n; ++j) {
            const double w = search::width(P, j);
            const double g = periodic::gap(x[j], b.x[j], periodic::is(P.periodic, j), w) / (w > 0.0 ? w : 1.0);
            s += g * g;
        }
        return std::sqrt(s);
    };
    long long searches = 0, steps = 0, accepted = 0, iterations = 0;
    auto spec = [&](const Start& st) {
        StartSpec sp;
        sp.seed = start_seed(seed, static_cast<std::uint64_t>(searches++));
        sp.start = st;
        return sp;
    };
    auto run_round = [&](std::vector<StartSpec> specs) {
        const std::vector<Result> r = search::starts(ev, inner, std::move(specs), cap, static_cast<int>(threads), pop_size);
        std::vector<search::Found> out;
        for (const Result& q : r) out.push_back(search::found(q));
        return out;
    };
    std::vector<search::Found> pop;
    double dcut = std::numeric_limits<double>::infinity();
    try {
        // 0. the first population (PBH-4: the uniform points drawn first)
        std::vector<StartSpec> init{spec(start)};
        for (long long p = 1; p < K; ++p) {
            Vec x(n);
            for (std::size_t j = 0; j < n; ++j) x[j] = P.lo[j] + rnd::uniform01(rng) * search::width(P, j);
            init.push_back(spec(search::at(x)));
        }
        const long long first = searches - K;
        pop = run_round(init);
        for (std::size_t p = 0; p < pop.size(); ++p)
            if (pop[p].ok) log.add(pop[p].x, pop[p].f, first + static_cast<long long>(p));
        if (static_cast<long long>(pop.size()) < K) throw BudgetSpent();
        for (const auto& m : pop)
            if (!m.ok) throw BudgetSpent();
        if (dcut_c > 0.0 && K > 1) {                                                 // GLS07 §4: c d_ave
            double sum = 0.0;
            long long pairs = 0;
            for (long long a = 0; a < K; ++a)
                for (long long b = a + 1; b < K; ++b, ++pairs)
                    sum += d(pop[static_cast<std::size_t>(a)], pop[static_cast<std::size_t>(b)]);
            dcut = dcut_c * sum / static_cast<double>(pairs);
        }
        long long idle = 0;
        while (search::left(ev) > 0 && idle < 100 && (maxstep == 0 || iterations < maxstep)) {   // SRCH-10
            const double before = ev.used();
            // 1. the children's starts (PBH-3, PBH-4)
            std::vector<StartSpec> kids;
            const long long first_kid = searches;
            for (long long p = 0; p < K; ++p) {
                Vec y = pop[static_cast<std::size_t>(p)].x;
                for (std::size_t j = 0; j < n; ++j) {
                    const double moved = y[j] + (2.0 * rnd::uniform01(rng) - 1.0) * step *
                                                    (absolute ? 1.0 : search::width(P, j));
                    y[j] = periodic::is(P.periodic, j) ? periodic::wrap_into(moved, P.lo[j], P.hi[j])
                                                       : std::min(P.hi[j], std::max(P.lo[j], moved));
                }
                (void)rnd::uniform01(rng);                                           // PBH-4
                kids.push_back(spec(search::at(y)));
            }
            const std::vector<search::Found> Y = run_round(kids);
            ++iterations;
            // 2. placed one after the other (PBH-1, PBH-2, PBH-5)
            bool all = static_cast<long long>(Y.size()) == K;
            for (std::size_t p = 0; p < Y.size(); ++p) {
                const search::Found& c = Y[p];
                if (!c.ok) {
                    all = false;
                    continue;
                }
                ++steps;
                log.add(c.x, c.f, first_kid + static_cast<long long>(p));
                std::size_t near = 0, worst = 0;
                double dn = std::numeric_limits<double>::infinity();
                for (std::size_t q = 0; q < pop.size(); ++q) {
                    const double dq = d(c, pop[q]);
                    if (dq < dn) {
                        dn = dq;
                        near = q;
                    }
                    if (search::better(pop[worst], pop[q])) worst = q;
                }
                const std::size_t target = dn < dcut ? near : worst;
                if (search::better(c, pop[target])) {
                    pop[target] = c;
                    ++accepted;
                }
            }
            if (anneal && iterations <= 250 && iterations % 50 == 0) dcut *= 0.85;   // 2P-PBH*
            if (!all) break;
            idle = ev.used() == before ? idle + 1 : 0;
        }
    } catch (const BudgetSpent&) {
    }
    std::vector<std::pair<std::string, std::string>> extra{{"steps", std::to_string(steps)},
                                                           {"accepted", std::to_string(accepted)},
                                                           {"iterations", std::to_string(iterations)}};
    if (std::isfinite(dcut)) extra.push_back({"pbh_dcut_end", detail::g_format(dcut)});
    return search::finish(ev, k, log, extra);
}

}   // namespace mootation::scalar
