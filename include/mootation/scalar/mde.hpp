#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// mde: differential evolution over local minima (task 25, §24.2.3; АП-9 of
// the preparation). Included by strategy.hpp after pbh.hpp.
//
//   mde       M. Locatelli, M. Maischberger, F. Schoen, "Differential
//             evolution methods based on local searches", Comput. Oper. Res.
//             43 (2014) (source differential-evolution-methods_locatelli2013;
//             "LMS14"), Algorithm 2: for each member i, k1, k2, k3 != i all
//             different, Trial = x_k1 + F (x_k2 - x_k3), crossover, Trial =
//             L(Trial), x_i replaced when the trial is better;
//   greedy    G-MDE, G. Cabassi, M. Locatelli, "Computational investigation
//             of simple memetic approaches for continuous global
//             optimization", Comput. Oper. Res. 72 (2016) (source
//             computational-investigation-simple_cabassi2016; "CL16"),
//             Algorithm 6: r != i, phi = +1 when f(p_i) > f(p_r), else -1, y =
//             p_i + phi F (p_r - p_i), the candidate against p_i;
//   distance  D-MDE, CL16 Algorithm 7: G-MDE's candidate against the member
//             nearest to it by mde_distance (fdiff, CL16's |f(s) - f(u)|;
//             euclid; aligned: pbh.hpp's dissimilarity);
//   hybrid    H-MDE, CL16 Algorithm 8: D-MDE's selection when phi = -1,
//             G-MDE's when phi = +1.
// As CL16 Algorithm 1 the members are taken in order and each candidate is
// placed before the next is made. The first population: mde_pop local minima,
// the first from the run's start, the others from uniform points (CL16
// Algorithm 2). The run ends (CL16 Algorithm 3, after each sweep over the
// members) when the best has not improved for mde_noimprove local searches
// (0: 100 n, CL16 §4), the population's values have not changed for
// mde_idle (100) sweeps, the sum of |f(p_i) - f(p_j)| over all pairs is at
// most mde_ftol (1e-4, LMS14 §3's threshold), or the budget is spent.
//
// Knobs: inner, mde_variant (mde | greedy | distance | hybrid), mde_pop
// (40), mde_F (0.5), mde_CR (1), mde_match (off | align | greedy),
// mde_distance (fdiff | euclid | aligned), mde_noimprove, mde_idle, mde_ftol,
// start_budget, min_dist, min_df, keep_points.
//
// READINGS:
//   MDE-1. CR in the classical sense (§24.9 item 17): the probability that a
//     coordinate is the mutant's, so mde_CR = 1 is LMS14's CR = 0 (their test
//     "U < CR: keep x_i's coordinate", no crossover) and CL16's CR = 1. Below
//     1, every coordinate draws U(0, 1) and takes the mutant's when U < CR
//     (CL16: <=), with no forced coordinate (neither text has one).
//   MDE-2. k1, k2, k3 all different and different from i (LMS14 Algorithm 2;
//     its §3 allows repeats "without significant change"): mde needs mde_pop
//     >= 4, the others >= 2.
//   MDE-3. mde_match relabels at the start of every sweep each member to the
//     best one (the record): align — the problem's align, else
//     symmetry::align on sym_blocks and canon; greedy — LMS14 §4, Algorithm
//     4: within each group of interchangeable blocks, the block pairs (member,
//     record) sorted by squared distance, taken greedily, the relabelling
//     applied once the matching is complete.
//   MDE-4. The local search of the s-th search has the seed start_seed(seed,
//     s); the strategy's own draws (the indices, the coins) come from
//     mt19937(start_seed(seed, 0x4D444531)).
//   MDE-5. "The population does not change": the members' values (f, cv) are
//     the same as at the sweep's start (LMS14 §3: "the objective values").
//   MDE-6. A trial is clipped to the box (wrapped on a periodic coordinate).
// ============================================================================

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "strategy.hpp"

namespace mootation::scalar {

namespace search {

// LMS14 Algorithm 4 within each group of interchangeable blocks: x relabelled
// towards ref greedily (MDE-3)
inline Vec greedy_match(const Vec& x, const Vec& ref, const std::vector<symmetry::Blocks>& groups)
{
    Vec out = x;
    for (const auto& g : groups) {
        std::vector<std::pair<double, std::pair<std::size_t, std::size_t>>> pairs;   // (d^2, (block of x, block of ref))
        for (std::size_t a = 0; a < g.count; ++a)
            for (std::size_t b = 0; b < g.count; ++b) {
                double s = 0.0;
                for (std::size_t c = 0; c < g.size; ++c) {
                    const double t = x[g.offset + a * g.size + c] - ref[g.offset + b * g.size + c];
                    s += t * t;
                }
                pairs.push_back({s, {a, b}});
            }
        std::stable_sort(pairs.begin(), pairs.end(),
                         [](const auto& p, const auto& q) { return p.first < q.first; });
        std::vector<char> used_x(g.count, 0), used_r(g.count, 0);
        std::vector<std::size_t> to(g.count, 0);                 // block of ref -> block of x
        for (const auto& p : pairs) {
            const std::size_t a = p.second.first, b = p.second.second;
            if (used_x[a] || used_r[b]) continue;
            used_x[a] = used_r[b] = 1;
            to[b] = a;
        }
        for (std::size_t b = 0; b < g.count; ++b)
            for (std::size_t c = 0; c < g.size; ++c) out[g.offset + b * g.size + c] = x[g.offset + to[b] * g.size + c];
    }
    return out;
}

}   // namespace search

inline Result run_mde(const Problem& P, double eq_tol, bool has_tol0, double eq_tol0, int pop_size, long long budget,
                      unsigned seed, const Knobs& knobs, Observer observer, int record_every, const Start& start)
{
    using namespace strategy;
    StrategyKnobs k("mde", knobs,
                    {"inner", "mde_variant", "mde_pop", "mde_F", "mde_CR", "mde_match", "mde_distance",
                     "mde_noimprove", "mde_idle", "mde_ftol", "start_budget", "min_dist", "min_df", "keep_points"});
    const Inner inner = k.inner("inner");
    const std::string variant = k.word("mde_variant", "mde", {"mde", "greedy", "distance", "hybrid"});
    const long long K = k.whole("mde_pop", 40, 2);
    const double F = k.number("mde_F", 0.5), CR = k.number("mde_CR", 1.0);
    const std::string match = k.word("mde_match", "off", {"off", "align", "greedy"});
    const std::string dist = k.word("mde_distance", "fdiff", {"fdiff", "euclid", "aligned"});
    const std::size_t n = P.lo.size();
    const long long noimprove = k.whole("mde_noimprove", 0, 0) > 0 ? k.whole("mde_noimprove", 0, 0)
                                                                    : 100 * static_cast<long long>(n);
    const long long idle_max = k.whole("mde_idle", 100, 1);
    const double ftol = k.number("mde_ftol", 1e-4);
    const long long cap = k.whole("start_budget", 0, 0);
    const double min_dist = k.number("min_dist", 1e-3), min_df = k.number("min_df", 1e-8);
    if (variant == "mde" && K < 4) throw std::invalid_argument("mde: mde_variant = mde needs mde_pop >= 4 (MDE-2)");
    if (!(F > 0.0 && F < 2.0)) throw std::invalid_argument("mde: mde_F must be in (0, 2)");
    if (!(CR >= 0.0 && CR <= 1.0)) throw std::invalid_argument("mde: mde_CR must be in [0, 1]");
    if (match == "greedy" && P.sym_blocks.empty())
        throw std::invalid_argument("mde: mde_match = greedy needs the problem's sym_blocks");
    search::finite_box("mde", P);
    Evaluator ev(P, eq_tol, has_tol0, eq_tol0, budget, pop_size, std::move(observer), record_every);
    ev.lean = lean_of(k);                                                            // STRAT-12
    MinimaLog log(P.lo, P.hi, min_dist, min_df);
    strategy::group_by_symmetry(k, P, log);
    std::mt19937 rng(start_seed(seed, 0x4D444531u));                                 // MDE-4
    const Vec period = symmetry::periods(P.periodic, P.lo, P.hi);
    long long searches = 0, sweeps = 0, replaced = 0;
    auto local = [&](const Start& st) {
        StartSpec sp;
        sp.seed = start_seed(seed, static_cast<std::uint64_t>(searches));
        sp.start = st;
        const std::vector<Result> r = search::starts(ev, inner, {sp}, cap, 1, pop_size);
        const search::Found b = r.empty() ? search::Found() : search::found(r[0]);
        if (b.ok) log.add(b.x, b.f, searches);
        ++searches;
        return b;
    };
    auto into_box = [&](Vec& y) {                                                    // MDE-6
        for (std::size_t j = 0; j < n; ++j)
            y[j] = periodic::is(P.periodic, j) ? periodic::wrap_into(y[j], P.lo[j], P.hi[j])
                                               : std::min(P.hi[j], std::max(P.lo[j], y[j]));
    };
    auto pick_other = [&](long long i, const std::vector<long long>& taken) {       // MDE-2
        for (;;) {
            const long long r = static_cast<long long>(rnd::below(rng, static_cast<std::uint64_t>(K)));
            if (r == i || std::find(taken.begin(), taken.end(), r) != taken.end()) continue;
            return r;
        }
    };
    std::vector<search::Found> pop;
    try {
        pop.push_back(local(start));                                                 // CL16 Algorithm 2
        for (long long p = 1; p < K; ++p) {
            Vec x(n);
            for (std::size_t j = 0; j < n; ++j) x[j] = P.lo[j] + rnd::uniform01(rng) * search::width(P, j);
            pop.push_back(local(search::at(x)));
        }
        for (const auto& m : pop)
            if (!m.ok) throw BudgetSpent();
        auto best_of = [&]() {
            std::size_t b = 0;
            for (std::size_t q = 1; q < pop.size(); ++q)
                if (search::better(pop[q], pop[b])) b = q;
            return b;
        };
        search::Found best = pop[best_of()];
        long long no_imp = 0, unchanged = 0;
        while (search::left(ev) > 0) {
            // MDE-3: the members relabelled to the record
            if (match != "off") {
                const Vec& ref = pop[best_of()].x;
                for (auto& m : pop) {
                    if (match == "greedy") m.x = search::greedy_match(m.x, ref, P.sym_blocks);
                    else if (P.align) {
                        Vec out;
                        P.align(m.x, ref, out);
                        m.x = out;
                    } else if (!P.sym_blocks.empty()) {
                        m.x = symmetry::align(m.x, ref, P.sym_blocks, P.canon, period);
                    }
                }
            }
            std::vector<std::pair<double, double>> before;
            for (const auto& m : pop) before.push_back({m.f, m.cv});
            for (long long i = 0; i < K; ++i) {
                const search::Found& xi = pop[static_cast<std::size_t>(i)];
                Vec y(n);
                double phi = 1.0;
                if (variant == "mde") {                                                // LMS14 Algorithm 2
                    const long long k1 = pick_other(i, {}), k2 = pick_other(i, {k1}), k3 = pick_other(i, {k1, k2});
                    for (std::size_t j = 0; j < n; ++j)
                        y[j] = pop[static_cast<std::size_t>(k1)].x[j] +
                               F * (pop[static_cast<std::size_t>(k2)].x[j] - pop[static_cast<std::size_t>(k3)].x[j]);
                } else {                                                               // CL16 Algorithm 6
                    const long long r = pick_other(i, {});
                    const search::Found& pr = pop[static_cast<std::size_t>(r)];
                    phi = search::better(pr, xi) ? 1.0 : -1.0;                         // f(p_i) > f(p_r)
                    for (std::size_t j = 0; j < n; ++j) y[j] = xi.x[j] + phi * F * (pr.x[j] - xi.x[j]);
                }
                if (CR < 1.0)                                                          // MDE-1
                    for (std::size_t j = 0; j < n; ++j)
                        if (!(rnd::uniform01(rng) < CR)) y[j] = xi.x[j];
                into_box(y);
                const search::Found c = local(search::at(y));
                if (!c.ok) throw BudgetSpent();
                // the selection
                std::size_t target = static_cast<std::size_t>(i);
                const bool by_distance = variant == "distance" || (variant == "hybrid" && phi < 0.0);
                if (by_distance) {                                                     // CL16 Algorithm 7
                    double dn = std::numeric_limits<double>::infinity();
                    for (std::size_t q = 0; q < pop.size(); ++q) {
                        const double dq = search::dissimilarity(P, dist, c, pop[q], period);
                        if (dq < dn) {
                            dn = dq;
                            target = q;
                        }
                    }
                }
                if (search::better(c, pop[target])) {
                    pop[target] = c;
                    ++replaced;
                }
            }
            ++sweeps;
            // CL16 Algorithm 3
            const search::Found& now = pop[best_of()];
            if (search::better(now, best)) {
                best = now;
                no_imp = 0;
            } else {
                no_imp += K;
            }
            bool same = true;
            for (std::size_t q = 0; q < pop.size(); ++q) same = same && pop[q].f == before[q].first && pop[q].cv == before[q].second;
            unchanged = same ? unchanged + 1 : 0;                                      // MDE-5
            double fvar = 0.0;
            for (std::size_t a = 0; a < pop.size(); ++a)
                for (std::size_t b = 0; b < pop.size(); ++b) fvar += std::fabs(pop[a].f - pop[b].f);
            if (no_imp >= noimprove || unchanged >= idle_max || fvar <= ftol) break;
        }
    } catch (const BudgetSpent&) {
    }
    return search::finish(ev, k, log,
                          {{"sweeps", std::to_string(sweeps)}, {"local_searches", std::to_string(searches)},
                           {"replaced", std::to_string(replaced)}});
}

}   // namespace mootation::scalar
