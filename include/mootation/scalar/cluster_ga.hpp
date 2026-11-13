#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// cluster_ga: a genetic algorithm over local minima with cut-and-splice
// (task 26, item 24.2.9), on the rounds of
// strategy.hpp. Included by strategy.hpp.
//
// D. M. Deaven, K. M. Ho, "Molecular geometry optimization with a genetic
// algorithm", Phys. Rev. Lett. 75 (1995) 288 (DH95), and D. M. Deaven, N.
// Tit, J. R. Morris, K. M. Ho, "Structural optimization of Lennard-Jones
// clusters by a genetic algorithm", Chem. Phys. Lett. 256 (1996) 195
// (DTMH96), the steps as this library takes them (not checked on DTMH96's text):
//   0. cga_pop members, local minima: the first from the run's start, the
//      others from uniform points of the box (as pbh, PBH-4).
//   1. Children: with cga_mating = boltzmann (DH95) cga_children of them,
//      each from two parents drawn with probability proportional to
//      exp(-(E - E_min) / T_m), T_m the range of the members' energies
//      (uniform where the range is 0); with all_pairs (DTMH96) one child of
//      every pair. A child is the crossover of its parents (the problem's
//      crossover hook, task 24, its first child; else the built-in cut and
//      splice, CGA-1) and, with probability cga_mut_rate, mutated (the
//      problem's mutate hook; else every coordinate moved by U[-1, 1]
//      times cga_step times the box's width, pbh's move).
//   2. Every child to a local minimum by the inner solver; it is the child
//      (Lamarck).
//   3. In the children's order: a child replaces the highest member when it
//      is lower and no member is within cga_de of it in energy (DTMH96:
//      "children successively replace higher ones"; DH95's δE).
//   4. Again from 1 until the budget, cga_maxgen generations, or 100 in a
//      row that spend nothing (SRCH-10).
// THE DEGENERATE CASE: cga_pop = 1 makes every child a mutant of the one
// member (CGA-2) by pbh's move, and with cga_de = 0 the run is pbh's with
// pbh_K = 1 — monotonic basin hopping, ils with ils_accept = better — bit for
// bit (the same stream, the same seeds, PBH-4).
//
// THE REPLACEMENT (task 26, item 24.5.2, A2 and A3; cga_replace):
//   worst_if_better (the default) step 3 above;
//   multi_dynamic   C. Segura, C. A. Coello Coello, E. Segredo, A. Hernández-
//     Aguirre, "A novel diversity-based replacement strategy for
//     evolutionary algorithms", IEEE Trans. Cybern. 46(12), 2016 (source
//     novel-diversity-replacement_segura2015), Algorithms 2 and 3: parents
//     by binary tournament on f; the members and the children joined; the
//     best by f into the new population; until it has cga_pop members: the
//     DCN of every candidate (its distance to the nearest member selected),
//     a candidate with DCN < D penalized (f = +inf), the non-dominated
//     candidates in (f, -DCN) without repetitions, one of them at random;
//     D = D_I (1 - used / (md_end budget)), not below 0 (MD-1, MD-2);
//   afpo            M. Schmidt, H. Lipson, "Age-fitness Pareto optimization",
//     Genetic Programming Theory and Practice VIII, 2011, ch. 8 (source
//     age-fitness-pareto_schmidt2010): random pairs of members; one new
//     random individual every generation; every individual's age: 1 at its
//     creation, +1 every generation it stays, a child the largest of its
//     parents' (Hornby's rule, the paper's); the selection: groups of
//     afpo_tournament (2) drawn at random, their dominated (in f and age,
//     both minimized) removed, until cga_pop are left or none is dominated
//     (AF-1, AF-2).
// Every mode searches every child locally (A7 in the single-objective
// track: Segura's Algorithm 3, "the local search of every offspring").
//
// Knobs: inner, cga_pop (8), cga_mating (boltzmann | all_pairs),
// cga_children (0: cga_pop), cga_mut_rate (0.05), cga_step (0.1), cga_de
// (1e-6), cga_maxgen (0: the budget), crossover (builtin | user), mutation
// (builtin | user), cga_replace (worst_if_better | multi_dynamic | afpo),
// md_di (0.5), md_end (1), md_distance (euclid | aligned), afpo_tournament
// (2), threads, start_budget, min_dist, min_df, keep_points.
//
// READINGS:
//   CGA-1. The built-in cut and splice (DH95) needs the problem's first
//     symmetry group to be of points (blocks of 2 or 3 coordinates): a
//     random direction (uniform in the ball [-1, 1]^d, by rejection), each
//     parent's points about its own centre, k = the first parent's points
//     on the positive side of the plane through its centre (k = 0 or all:
//     half, rounded down); the child takes those k of the first parent and
//     the N - k of the second with the lowest projection (DH95's plane
//     shifted until the count is right), in the first parent's frame,
//     clipped to the box; the coordinates in no group are the first
//     parent's. Without such a group the built-in crossover is uniform:
//     every coordinate from either parent with probability 1/2, one uniform
//     each.
//   CGA-2. With one member there is no pair: every child is the member
//     mutated (cga_mut_rate is not drawn).
//   CGA-3. The draws per child: the parents (boltzmann: two uniforms; a
//     parent drawn twice is kept, a child of one parent and itself), the
//     crossover's, one uniform for the mutation's choice (not with one
//     member), the mutation's n moves, and one more uniform (PBH-4), all
//     from one stream seeded as pbh's.
//   CGA-4. cga_de's default 1e-6 is Johnston's duplicate threshold
//     (Birmingham GA, Dalton Trans. 2003, §3; not checked on its text);
//     DTMH96's 5e-4 for LJ is a problem's choice.
//   MD-1. D_I = md_di G, G the largest distance of two points: sqrt(n) in
//     the box scaled to [0, 1]^n (md_distance = euclid), else (aligned:
//     hooks.hpp distance_of) the largest distance between two members of the
//     first population. T_End is the budget (evaluations), as Segura et
//     al.'s generations variant; md_end the share of it at which D is 0.
//   MD-2. A child is in the joined set as it is; the members' and the
//     children's ties in the non-dominated choice ("without repetitions")
//     are the first of each (f, DCN) pair; the random choice is one uniform.
//   AF-1. The newcomer is a uniform point of the box, searched locally like
//     every child, age 1; the ages of the members grow by one after the
//     selection.
//   AF-2. A Pareto tournament draws afpo_tournament distinct members (a
//     partial Fisher-Yates draw) and removes those another of them
//     dominates; the selection stops at cga_pop or when no member of the
//     whole population is dominated.
//   Not here: the pool mode, the
//   stagnation trigger of mutation, the predator, the outlier parents of
//   Jørgensen et al. and USPEX's antiseeds; Wolf and Landman's two-stage
//   relaxation (WL-1: E - |grad E|, read on the PDF by task 25).
// ============================================================================

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "hooks.hpp"
#include "strategy.hpp"

namespace mootation::scalar {

namespace cga {

// CGA-1: the cut and splice of a and b over the point blocks of g
inline Vec cut_and_splice(const Problem& P, const symmetry::Blocks& g, const Vec& a, const Vec& b, std::mt19937& rng)
{
    const std::size_t d = g.size, N = g.count;
    Vec dir(d);
    for (;;) {                                                       // uniform in the ball, by rejection
        double s = 0.0;
        for (double& v : dir) {
            v = 2.0 * rnd::uniform01(rng) - 1.0;
            s += v * v;
        }
        if (s <= 1.0 && s > 1e-24) {
            const double r = std::sqrt(s);
            for (double& v : dir) v /= r;
            break;
        }
    }
    auto centre = [&](const Vec& x) {
        Vec c(d, 0.0);
        for (std::size_t i = 0; i < N; ++i)
            for (std::size_t q = 0; q < d; ++q) c[q] += x[g.offset + i * d + q];
        for (double& v : c) v /= static_cast<double>(N);
        return c;
    };
    const Vec ca = centre(a), cb = centre(b);
    auto proj = [&](const Vec& x, const Vec& c, std::size_t i) {
        double s = 0.0;
        for (std::size_t q = 0; q < d; ++q) s += (x[g.offset + i * d + q] - c[q]) * dir[q];
        return s;
    };
    std::vector<std::size_t> ia(N), ib(N);
    for (std::size_t i = 0; i < N; ++i) ia[i] = ib[i] = i;
    std::stable_sort(ia.begin(), ia.end(), [&](std::size_t p, std::size_t q) { return proj(a, ca, p) > proj(a, ca, q); });
    std::stable_sort(ib.begin(), ib.end(), [&](std::size_t p, std::size_t q) { return proj(b, cb, p) < proj(b, cb, q); });
    std::size_t k = 0;
    for (std::size_t i = 0; i < N; ++i) k += proj(a, ca, i) > 0.0 ? 1 : 0;
    if (k == 0 || k == N) k = N / 2;
    Vec child = a;
    for (std::size_t s = 0; s < N; ++s) {
        const bool from_a = s < k;
        const std::size_t atom = from_a ? ia[s] : ib[s - k];
        for (std::size_t q = 0; q < d; ++q) {
            const std::size_t j = g.offset + s * d + q;
            const double v = from_a ? a[g.offset + atom * d + q] : b[g.offset + atom * d + q] - cb[q] + ca[q];
            child[j] = std::min(P.hi[j], std::max(P.lo[j], v));
        }
    }
    return child;
}

}   // namespace cga

inline Result run_cluster_ga(const Problem& P, double eq_tol, bool has_tol0, double eq_tol0, int pop_size,
                             long long budget, unsigned seed, const Knobs& knobs, Observer observer, int record_every,
                             const Start& start)
{
    using namespace strategy;
    StrategyKnobs k("cluster_ga", knobs,
                    {"inner", "cga_pop", "cga_mating", "cga_children", "cga_mut_rate", "cga_step", "cga_de",
                     "cga_maxgen", "crossover", "mutation", "threads", "start_budget", "min_dist", "min_df",
                     "keep_points", "cga_replace", "md_di", "md_end", "md_distance", "afpo_tournament"});
    const Inner inner = k.inner("inner");
    const long long K = k.whole("cga_pop", 8, 1), threads = k.whole("threads", 1, 1);
    const bool pairs = k.word("cga_mating", "boltzmann", {"boltzmann", "all_pairs"}) == "all_pairs";
    const long long kids0 = k.whole("cga_children", 0, 0);
    const double mut_rate = k.number("cga_mut_rate", 0.05), step = k.number("cga_step", 0.1);
    const double de = k.number("cga_de", 1e-6);
    const long long maxgen = k.whole("cga_maxgen", 0, 0), cap = k.whole("start_budget", 0, 0);
    const bool user_x = k.find("crossover") && k.word("crossover", "builtin", {"builtin", "user"}) == "user";
    const bool user_m = k.find("mutation") && k.word("mutation", "builtin", {"builtin", "user"}) == "user";
    const double min_dist = k.number("min_dist", 1e-3), min_df = k.number("min_df", 1e-8);
    const std::string mode = k.word("cga_replace", "worst_if_better", {"worst_if_better", "multi_dynamic", "afpo"});
    const bool md = mode == "multi_dynamic", afpo = mode == "afpo";
    const double md_di = md ? k.number("md_di", 0.5) : 0.0, md_end = md ? k.number("md_end", 1.0) : 1.0;
    const bool md_aligned = md && k.word("md_distance", "euclid", {"euclid", "aligned"}) == "aligned";
    const long long tour = afpo ? k.whole("afpo_tournament", 2, 2) : 2;
    if (md && !(md_di >= 0.0)) throw std::invalid_argument("cluster_ga: md_di must be >= 0");
    if (md && !(md_end > 0.0)) throw std::invalid_argument("cluster_ga: md_end must be > 0");
    if (!(mut_rate >= 0.0 && mut_rate <= 1.0)) throw std::invalid_argument("cluster_ga: cga_mut_rate in [0, 1]");
    if (!(step > 0.0)) throw std::invalid_argument("cluster_ga: cga_step must be > 0");
    if (!(de >= 0.0)) throw std::invalid_argument("cluster_ga: cga_de must be >= 0");
    if (user_x && !P.crossover) throw std::invalid_argument("cluster_ga: crossover = user needs the crossover hook");
    if (user_m && !P.mutate) throw std::invalid_argument("cluster_ga: mutation = user needs the mutate hook");
    const bool points = !P.sym_blocks.empty() && (P.sym_blocks[0].size == 2 || P.sym_blocks[0].size == 3);
    search::finite_box("cluster_ga", P);
    const std::size_t n = P.lo.size();
    Evaluator ev(P, eq_tol, has_tol0, eq_tol0, budget, pop_size, std::move(observer), record_every);
    ev.lean = lean_of(k);
    MinimaLog log(P.lo, P.hi, min_dist, min_df);
    group_by_symmetry(k, P, log);
    std::mt19937 rng(start_seed(seed, 0x9E3779B9u));                                 // CGA-3: pbh's stream
    long long searches = 0, generations = 0, crossings = 0, mutations = 0, accepted = 0;
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
    auto move = [&](Vec& y) {                                                        // pbh's move
        if (user_m) {
            P.mutate(y, rng);
            if (y.size() != n) throw std::runtime_error("cluster_ga: the mutate hook changed the point's length");
        } else {
            for (std::size_t j = 0; j < n; ++j) y[j] += (2.0 * rnd::uniform01(rng) - 1.0) * step * search::width(P, j);
        }
        for (std::size_t j = 0; j < n; ++j)
            y[j] = periodic::is(P.periodic, j) ? periodic::wrap_into(y[j], P.lo[j], P.hi[j])
                                               : std::min(P.hi[j], std::max(P.lo[j], y[j]));
        ++mutations;
    };
    std::vector<search::Found> pop;
    std::vector<long long> age;                                                      // AF-1
    auto dist = [&](const Vec& a, const Vec& b) {                                    // MD-1
        if (md_aligned) return distance_of(P, a, b);
        double q = 0.0;
        for (std::size_t j = 0; j < n; ++j) {
            const double w = search::width(P, j);
            const double g = periodic::gap(a[j], b[j], periodic::is(P.periodic, j), w) / (w > 0.0 ? w : 1.0);
            q += g * g;
        }
        return std::sqrt(q);
    };
    double D_I = 0.0;
    try {
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
        age.assign(pop.size(), 1);
        if (md) {
            double G = std::sqrt(static_cast<double>(n));
            if (md_aligned) {
                G = 0.0;
                for (std::size_t a = 0; a < pop.size(); ++a)
                    for (std::size_t b = a + 1; b < pop.size(); ++b) G = std::max(G, dist(pop[a].x, pop[b].x));
            }
            D_I = md_di * G;
        }
        long long idle = 0;
        while (search::left(ev) > 0 && idle < 100 && (maxgen == 0 || generations < maxgen)) {   // SRCH-10
            const double before = ev.used();
            // 1. the children (CGA-2, CGA-3)
            std::vector<std::pair<std::size_t, std::size_t>> parents;
            const long long n_kids = kids0 > 0 ? kids0 : K;
            if (K == 1) {
                parents.push_back({0, 0});
            } else if (md || afpo) {
                auto any = [&]() {
                    return std::min(pop.size() - 1, static_cast<std::size_t>(rnd::uniform01(rng) * static_cast<double>(pop.size())));
                };
                auto tournament = [&]() {                                            // Segura: on f
                    const std::size_t a = any(), b = any();
                    return search::better(pop[b], pop[a]) ? b : a;
                };
                for (long long c = 0; c < n_kids; ++c) {
                    const std::size_t a = md ? tournament() : any();
                    const std::size_t b = md ? tournament() : any();
                    parents.push_back({a, b});
                }
            } else if (pairs) {
                for (std::size_t a = 0; a < pop.size(); ++a)
                    for (std::size_t b = a + 1; b < pop.size(); ++b) parents.push_back({a, b});
            } else {
                double lo_e = pop[0].f, hi_e = pop[0].f;
                for (const auto& m : pop) {
                    lo_e = std::min(lo_e, m.f);
                    hi_e = std::max(hi_e, m.f);
                }
                const double T = hi_e - lo_e;
                std::vector<double> w(pop.size());
                double sum = 0.0;
                for (std::size_t q = 0; q < pop.size(); ++q) sum += w[q] = T > 0.0 ? dm::exp(-(pop[q].f - lo_e) / T) : 1.0;
                auto draw = [&]() {
                    double u = rnd::uniform01(rng) * sum;
                    for (std::size_t q = 0; q < w.size(); ++q) {
                        if (u < w[q]) return q;
                        u -= w[q];
                    }
                    return w.size() - 1;
                };
                for (long long c = 0; c < n_kids; ++c) {
                    const std::size_t a = draw();
                    const std::size_t b = draw();
                    parents.push_back({a, b});
                }
            }
            std::vector<StartSpec> kids;
            const long long first_kid = searches;
            for (const auto& pr : parents) {
                Vec y;
                if (K == 1) {
                    y = pop[0].x;
                    move(y);                                                         // CGA-2
                } else {
                    const Vec& a = pop[pr.first].x;
                    const Vec& b = pop[pr.second].x;
                    if (user_x) {
                        Vec c1(n), c2(n);
                        P.crossover(a, b, c1, c2, rng);
                        if (c1.size() != n) throw std::runtime_error("cluster_ga: the crossover hook gave a child of another length");
                        y = c1;
                    } else if (points) {
                        y = cga::cut_and_splice(P, P.sym_blocks[0], a, b, rng);
                    } else {                                                         // CGA-1: uniform
                        y = a;
                        for (std::size_t j = 0; j < n; ++j)
                            if (rnd::uniform01(rng) < 0.5) y[j] = b[j];
                    }
                    ++crossings;
                    if (rnd::uniform01(rng) < mut_rate) move(y);
                }
                (void)rnd::uniform01(rng);                                           // PBH-4
                kids.push_back(spec(search::at(y)));
            }
            if (afpo) {                                                              // AF-1: the newcomer
                Vec x(n);
                for (std::size_t j = 0; j < n; ++j) x[j] = P.lo[j] + rnd::uniform01(rng) * search::width(P, j);
                kids.push_back(spec(search::at(x)));
            }
            const std::vector<search::Found> Y = run_round(kids);
            ++generations;
            // 3. placed one after the other
            bool all = Y.size() == kids.size();
            if (md || afpo) {
                std::vector<search::Found> U = pop;
                std::vector<long long> A = age;
                for (std::size_t p = 0; p < Y.size(); ++p) {
                    if (!Y[p].ok) {
                        all = false;
                        continue;
                    }
                    log.add(Y[p].x, Y[p].f, first_kid + static_cast<long long>(p));
                    U.push_back(Y[p]);
                    A.push_back(p < parents.size() ? std::max(age[parents[p].first], age[parents[p].second]) : 1);
                }
                if (md) {                                                            // Segura, Algorithm 2
                    const double frac = std::max(0.0, 1.0 - ev.used() / (md_end * static_cast<double>(budget)));
                    const double D = D_I * frac;
                    std::vector<char> taken(U.size(), 0);
                    std::vector<search::Found> next;
                    std::size_t best = 0;
                    for (std::size_t q = 1; q < U.size(); ++q)
                        if (search::better(U[q], U[best])) best = q;
                    taken[best] = 1;
                    next.push_back(U[best]);
                    while (static_cast<long long>(next.size()) < K) {
                        std::vector<std::size_t> cand;
                        std::vector<double> fv, dcn;
                        for (std::size_t q = 0; q < U.size(); ++q) {
                            if (taken[q]) continue;
                            double dmin = std::numeric_limits<double>::infinity();
                            for (const auto& m : next) dmin = std::min(dmin, dist(U[q].x, m.x));
                            cand.push_back(q);
                            dcn.push_back(dmin);
                            const bool feasible = U[q].cv <= 0.0;
                            fv.push_back(dmin < D ? std::numeric_limits<double>::infinity()
                                                  : (feasible ? U[q].f : std::numeric_limits<double>::infinity()));
                        }
                        if (cand.empty()) break;
                        std::vector<std::size_t> front;                              // (f min, DCN max), MD-2
                        for (std::size_t a = 0; a < cand.size(); ++a) {
                            bool dominated = false, repeat = false;
                            for (std::size_t b = 0; b < cand.size() && !dominated; ++b)
                                dominated = (fv[b] <= fv[a] && dcn[b] >= dcn[a]) && (fv[b] < fv[a] || dcn[b] > dcn[a]);
                            for (std::size_t f : front) repeat = repeat || (fv[f] == fv[a] && dcn[f] == dcn[a]);
                            if (!dominated && !repeat) front.push_back(a);
                        }
                        const std::size_t pick = front[std::min(front.size() - 1,
                                                                static_cast<std::size_t>(rnd::uniform01(rng) *
                                                                                         static_cast<double>(front.size())))];
                        taken[cand[pick]] = 1;
                        next.push_back(U[cand[pick]]);
                    }
                    accepted += static_cast<long long>(next.size());
                    pop = next;
                    age.assign(pop.size(), 1);
                } else {                                                             // AFPO, AF-2
                    auto dominates = [&](std::size_t a, std::size_t b) {
                        const bool fa = search::better(U[a], U[b]) || (!search::better(U[b], U[a]));
                        return fa && A[a] <= A[b] && (search::better(U[a], U[b]) || A[a] < A[b]);
                    };
                    std::vector<std::size_t> live(U.size());
                    for (std::size_t q = 0; q < U.size(); ++q) live[q] = q;
                    while (static_cast<long long>(live.size()) > K) {
                        bool any_dominated = false;
                        for (std::size_t a = 0; a < live.size() && !any_dominated; ++a)
                            for (std::size_t b = 0; b < live.size() && !any_dominated; ++b)
                                any_dominated = a != b && dominates(live[b], live[a]);
                        if (!any_dominated) break;
                        const std::size_t t = std::min<std::size_t>(static_cast<std::size_t>(tour), live.size());
                        std::vector<std::size_t> group(live);
                        for (std::size_t g = 0; g < t; ++g) {                        // a partial Fisher-Yates draw
                            const std::size_t r = g + std::min(group.size() - g - 1,
                                                               static_cast<std::size_t>(rnd::uniform01(rng) *
                                                                                        static_cast<double>(group.size() - g)));
                            std::swap(group[g], group[r]);
                        }
                        std::vector<std::size_t> out;
                        for (std::size_t a = 0; a < t; ++a)
                            for (std::size_t b = 0; b < t; ++b)
                                if (a != b && dominates(group[b], group[a])) {
                                    out.push_back(group[a]);
                                    break;
                                }
                        for (std::size_t o : out) {
                            if (static_cast<long long>(live.size()) <= K) break;
                            live.erase(std::find(live.begin(), live.end(), o));
                        }
                    }
                    std::vector<search::Found> next;
                    std::vector<long long> nage;
                    for (std::size_t q : live) {
                        next.push_back(U[q]);
                        nage.push_back(A[q] + 1);                                    // AF-1
                    }
                    accepted += static_cast<long long>(live.size());
                    pop = next;
                    age = nage;
                }
                if (!all) break;
                idle = ev.used() == before ? idle + 1 : 0;
                continue;
            }
            for (std::size_t p = 0; p < Y.size(); ++p) {
                const search::Found& c = Y[p];
                if (!c.ok) {
                    all = false;
                    continue;
                }
                log.add(c.x, c.f, first_kid + static_cast<long long>(p));
                std::size_t worst = 0;
                bool near = false;
                for (std::size_t q = 0; q < pop.size(); ++q) {
                    if (search::better(pop[worst], pop[q])) worst = q;
                    if (std::fabs(pop[q].f - c.f) < de) near = true;
                }
                if (!near && search::better(c, pop[worst])) {
                    pop[worst] = c;
                    ++accepted;
                }
            }
            if (!all) break;
            idle = ev.used() == before ? idle + 1 : 0;
        }
    } catch (const BudgetSpent&) {
    }
    std::vector<std::pair<std::string, std::string>> extra{{"generations", std::to_string(generations)},
                                                           {"crossings", std::to_string(crossings)},
                                                           {"mutations", std::to_string(mutations)},
                                                           {"accepted", std::to_string(accepted)}};
    if (afpo) {
        std::string ages;
        for (std::size_t q = 0; q < age.size(); ++q) ages += (q ? "," : "") + std::to_string(age[q]);
        extra.push_back({"ages", ages});
    }
    if (md) extra.push_back({"md_d_initial", detail::g_format(D_I)});
    return search::finish(ev, k, log, extra);
}

}   // namespace mootation::scalar
