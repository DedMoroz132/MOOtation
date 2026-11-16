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
//      row that spend nothing (SRCH-10), or the stop rule's stall (task 30,
//      §30.13's cga_stall: generations in a row without a better best) or
//      max_steps (stop_rule.hpp; task 31, layer 1).
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
// (2), threads, start_budget, min_dist, min_df, keep_points, round_stop
// (target | off; strategy.hpp STRAT-16), minima_log (STRAT-15), cga_init
// (uniform | around_x0 | seed_points), cga_init_moves (1; CGA-5),
// cga_islands (1), cga_migrate_every (5), cga_migrants (1; CGA-6),
// crossover / mutation = ops, op_select (fixed | probability_matching |
// adaptive_pursuit), op_pmin (1/(2K)), op_alpha (0.8), op_beta (0.8; CGA-7).
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
//   CGA-5. cga_init = uniform | around_x0 | seed_points (task 30, §30.5;
//     off by default: uniform, step 0 above). around_x0: members 2 ... K
//     are x0 (clipped to the box) moved cga_init_moves (1) times by the
//     mutation's move (the mutate hook with mutation = user, else pbh's),
//     from the same stream, member by member; the moves are counted apart
//     (init_moves), not in mutations. seed_points: members 2 ... K are the
//     start's seed points in their order (their x, clipped; their values
//     are not used: each is a local search's start), the rest uniform as
//     before; init_seeded says how many. Member 1 is the start, as before
//     (with seed_points, without its seed points: they are members).
//   CGA-6. cga_islands = I (task 30, §30.8; off by default: 1, the one
//     population above). I populations of cga_pop members on a ring, with
//     cga_replace = worst_if_better: island 0 is the one population (its
//     stream, its first population from the start), island i has its own
//     stream, mt19937(start_seed(seed, 0x9E3779B9 + i)), and cga_pop first
//     members of its own by the rule of CGA-5. A generation: every island's
//     children from its own members and stream, all of them in one round
//     (threads: 1 thread and N give the same run), each placed into its
//     island (step 3). Every cga_migrate_every (5) generations the
//     cga_migrants (1) best members of island i - 1 (as they were before
//     the migration) are placed into island i by step 3's rule (migrated
//     counts those taken). The answer is the run's archive's best, as
//     before; Result::population holds every island's members. Not here:
//     an island's own parameter of the problem (at_param).
//   CGA-7. The operator portfolios (task 30, §30.7; off by default: the one
//     crossover and mutation above). crossover = ops, mutation = ops: the
//     problem's crossover_ops, mutate_ops (named hooks of crossover's and
//     mutate's form), one chosen for each crossover and each move by
//     ProportionalSelectOperator(P) (one uniform of the stream, none with
//     one operator, so one operator is crossover = user with it, bit for
//     bit), the allocation rule op_select (D. Thierens, GECCO 2005, source
//     thierens2005; probability matching after D. E. Goldberg, Machine
//     Learning 5, 1990): fixed P = 1/K (§4, the equal-probability rule);
//     probability_matching (§2, (1) and (2)); adaptive_pursuit (§3, (1) and
//     (3)-(5)); P(i) = 1/K and Q(i) = 1.0 at the start, the algorithms' line
//     2; op_alpha (0.8), op_beta (0.8) and op_pmin (1/(2K), §3's value, the
//     one of §4's runs) as in §4; op_pmin < 1/K refused otherwise
//     (footnote 2). Readings: TH-1. The reward of an application (GetReward)
//     is 1 where its child entered the population (step 3), else 0, so Q
//     is the running share of its children taken; a child moved after a
//     crossover rewards both. TH-2. The rewards come when the generation's
//     round is over, child by child in their order, so a generation's
//     children are chosen with the P of the generation before (the paper
//     rewards each application at once). TH-3. (2)'s Q is the one just
//     updated, as the algorithm's lines 6-8 run; all Q zero gives P = 1/K
//     ((2) divides by 0). TH-4. Ties of the best Q in (3) are broken by one
//     uniform of the stream ("ties are broken at random"). With cga_replace
//     = worst_if_better only. In the settings, operators: each operator's
//     [name, kind, calls, children taken, mean of the better parent's f -
//     the child's f over its children that came back, P at the end]; the
//     crossovers' calls add up to the run's crossings.
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
#include <utility>
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

// A portfolio of operators and its allocation rule (CGA-7; D. Thierens, "An
// adaptive pursuit strategy for allocating operator probabilities", GECCO
// 2005, §2-3): fixed, probability matching (2), adaptive pursuit (3)-(5)
struct Portfolio {
    std::vector<std::string> names;
    std::vector<double> P, Q;                 // the probabilities, the running estimates of the reward
    std::vector<long long> calls, taken;      // applications; children that entered the population
    std::vector<double> gain;                 // the sum of (the better parent's f - the child's f)
    std::vector<long long> scored;            // the children whose f came back
    int rule = 0;                             // 0 fixed, 1 probability matching, 2 adaptive pursuit
    double pmin = 0.0, alpha = 0.8, beta = 0.8;

    void start(std::vector<std::string> ns, int r, double p_min, double a, double b)
    {
        names = std::move(ns);
        const std::size_t K = names.size();
        P.assign(K, 1.0 / static_cast<double>(K));   // the algorithms' line 2: P(i) = 1/K, Q(i) = 1.0
        Q.assign(K, 1.0);
        calls.assign(K, 0);
        taken.assign(K, 0);
        gain.assign(K, 0.0);
        scored.assign(K, 0);
        rule = r;
        pmin = p_min;
        alpha = a;
        beta = b;
    }

    // ProportionalSelectOperator(P): one uniform of g, none with one operator
    std::size_t pick(std::mt19937& g) const
    {
        if (P.size() == 1) return 0;
        const double u = rnd::uniform01(g);
        double s = 0.0;
        for (std::size_t q = 0; q + 1 < P.size(); ++q) {
            s += P[q];
            if (u < s) return q;
        }
        return P.size() - 1;
    }

    // the reward R of operator a: Q by (1), then P by (2) or (3)-(5); ties
    // of the best Q broken at random, by one uniform of g (CGA-7)
    void reward(std::size_t a, double R, std::mt19937& g)
    {
        if (rule == 0) return;
        const std::size_t K = P.size();
        Q[a] = Q[a] + alpha * (R - Q[a]);
        if (rule == 1) {
            double sum = 0.0;
            for (double q : Q) sum += q;
            for (std::size_t i = 0; i < K; ++i)
                P[i] = sum > 0.0 ? pmin + (1.0 - static_cast<double>(K) * pmin) * Q[i] / sum : 1.0 / static_cast<double>(K);
            return;
        }
        double top = Q[0];
        for (double q : Q) top = std::max(top, q);
        std::vector<std::size_t> best;
        for (std::size_t i = 0; i < K; ++i)
            if (Q[i] == top) best.push_back(i);
        const std::size_t star =
            best.size() == 1 ? best[0]
                             : best[std::min(best.size() - 1,
                                             static_cast<std::size_t>(rnd::uniform01(g) * static_cast<double>(best.size())))];
        const double pmax = 1.0 - static_cast<double>(K - 1) * pmin;                     // (5)
        for (std::size_t i = 0; i < K; ++i)
            P[i] = i == star ? P[i] + beta * (pmax - P[i]) : P[i] + beta * (pmin - P[i]);   // (3), (4)
    }
};

}   // namespace cga

inline Result run_cluster_ga(const Problem& P, double eq_tol, bool has_tol0, double eq_tol0, int pop_size,
                             long long budget, unsigned seed, const Knobs& knobs, Observer observer, int record_every,
                             const Start& start)
{
    using namespace strategy;
    StrategyKnobs k("cluster_ga", knobs,
                    {"inner", "cga_pop", "cga_mating", "cga_children", "cga_mut_rate", "cga_step", "cga_de",
                     "cga_maxgen", "crossover", "mutation", "threads", "start_budget", "min_dist", "min_df",
                     "keep_points", "cga_replace", "md_di", "md_end", "md_distance", "afpo_tournament",
                     "cga_init", "cga_init_moves", "cga_islands", "cga_migrate_every", "cga_migrants",
                     "op_select", "op_pmin", "op_alpha", "op_beta"});
    const Inner inner = k.inner("inner");
    const long long K = k.whole("cga_pop", 8, 1), threads = k.whole("threads", 1, 1);
    const bool pairs = k.word("cga_mating", "boltzmann", {"boltzmann", "all_pairs"}) == "all_pairs";
    const long long kids0 = k.whole("cga_children", 0, 0);
    const double mut_rate = k.number("cga_mut_rate", 0.05), step = k.number("cga_step", 0.1);
    const double de = k.number("cga_de", 1e-6);
    const long long maxgen = k.whole("cga_maxgen", 0, 0), cap = k.whole("start_budget", 0, 0);
    const std::string x_how = k.find("crossover") ? k.word("crossover", "builtin", {"builtin", "user", "ops"}) : "builtin";
    const std::string m_how = k.find("mutation") ? k.word("mutation", "builtin", {"builtin", "user", "ops"}) : "builtin";
    const bool user_x = x_how == "user", user_m = m_how == "user";
    const bool ops_x = x_how == "ops", ops_m = m_how == "ops";                       // CGA-7
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
    if (ops_x && P.crossover_ops.empty()) throw std::invalid_argument("cluster_ga: crossover = ops needs crossover_ops");
    if (ops_m && P.mutate_ops.empty()) throw std::invalid_argument("cluster_ga: mutation = ops needs mutate_ops");
    cga::Portfolio xport, mport;                                                     // CGA-7
    if (ops_x || ops_m) {
        if (md || afpo) throw std::invalid_argument("cluster_ga: crossover or mutation = ops needs cga_replace = worst_if_better");
        const std::string sel = k.find("op_select")
                                    ? k.word("op_select", "fixed", {"fixed", "probability_matching", "adaptive_pursuit"})
                                    : "fixed";
        const int rule = sel == "fixed" ? 0 : sel == "probability_matching" ? 1 : 2;
        const double alpha = rule ? k.number("op_alpha", 0.8) : 0.8, beta = rule == 2 ? k.number("op_beta", 0.8) : 0.8;
        if (!(alpha > 0.0 && alpha <= 1.0) || !(beta > 0.0 && beta <= 1.0))
            throw std::invalid_argument("cluster_ga: op_alpha and op_beta in (0, 1]");
        auto names_of = [](const auto& ops) {
            std::vector<std::string> v;
            for (const auto& o : ops) v.push_back(o.first);
            return v;
        };
        auto pmin_of = [&](std::size_t K_ops) {                                      // Thierens's 1/(2K)
            const double p = rule && K_ops > 1 ? k.number("op_pmin", 0.5 / static_cast<double>(K_ops)) : 0.0;
            if (rule && K_ops > 1 && !(p > 0.0 && p < 1.0 / static_cast<double>(K_ops)))
                throw std::invalid_argument("cluster_ga: op_pmin in (0, 1/K) for K operators (Thierens, footnote 2)");
            return p;
        };
        if (ops_x) xport.start(names_of(P.crossover_ops), rule, pmin_of(P.crossover_ops.size()), alpha, beta);
        if (ops_m) mport.start(names_of(P.mutate_ops), rule, pmin_of(P.mutate_ops.size()), alpha, beta);
    }
    // a generation's children: their operators, the better parent's f, what became of them (CGA-7)
    std::vector<long long> kid_x, kid_m;
    std::vector<double> kid_pf;
    struct Placed {
        std::size_t kid;
        bool taken;
        double f;
    };
    std::vector<Placed> placed;
    long long kids_from = 0, last_m = -1;
    const bool points = !P.sym_blocks.empty() && (P.sym_blocks[0].size == 2 || P.sym_blocks[0].size == 3);
    search::finite_box("cluster_ga", P);
    const std::size_t n = P.lo.size();
    const std::string init_mode =                                                    // CGA-5
        k.find("cga_init") ? k.word("cga_init", "uniform", {"uniform", "around_x0", "seed_points"}) : "uniform";
    const long long init_moves = init_mode == "around_x0" ? k.whole("cga_init_moves", 1, 1) : 0;
    if (init_mode == "around_x0" && !(start.has_x0 && start.x0.size() == n))
        throw std::invalid_argument("cluster_ga: cga_init = around_x0 needs x0 (of n values)");
    const long long I = k.find("cga_islands") ? k.whole("cga_islands", 1, 1) : 1;      // CGA-6
    const long long mig_every = I > 1 ? k.whole("cga_migrate_every", 5, 1) : 0;
    const long long migrants = I > 1 ? k.whole("cga_migrants", 1, 1) : 0;
    if (I > 1 && (md || afpo))
        throw std::invalid_argument("cluster_ga: cga_islands > 1 needs cga_replace = worst_if_better");
    if (I > 1 && migrants > K) throw std::invalid_argument("cluster_ga: cga_migrants at most cga_pop");
    Evaluator ev(P, eq_tol, has_tol0, eq_tol0, budget, pop_size, std::move(observer), record_every);
    strategy::round_stop_of(k, ev);                                                  // STRAT-16
    StopRule stop = strategy::stop_of(k, ev);                                        // stop_rule.hpp: a step a generation
    long long idle = 0;
    ev.lean = lean_of(k);
    MinimaLog log(P.lo, P.hi, min_dist, min_df);
    group_by_symmetry(k, P, log);
    std::mt19937 rng(start_seed(seed, 0x9E3779B9u));                                 // CGA-3: pbh's stream
    long long searches = 0, generations = 0, crossings = 0, mutations = 0, accepted = 0, migrated = 0;
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
    auto shift = [&](Vec& y, std::mt19937& g) {                                      // pbh's move
        if (user_m) {
            P.mutate(y, g);
            if (y.size() != n) throw std::runtime_error("cluster_ga: the mutate hook changed the point's length");
        } else if (ops_m) {                                                          // CGA-7
            const std::size_t o = mport.pick(g);
            P.mutate_ops[o].second(y, g);
            if (y.size() != n) throw std::runtime_error("cluster_ga: a mutate operator changed the point's length");
            last_m = static_cast<long long>(o);
        } else {
            for (std::size_t j = 0; j < n; ++j) y[j] += (2.0 * rnd::uniform01(g) - 1.0) * step * search::width(P, j);
        }
        for (std::size_t j = 0; j < n; ++j)
            y[j] = periodic::is(P.periodic, j) ? periodic::wrap_into(y[j], P.lo[j], P.hi[j])
                                               : std::min(P.hi[j], std::max(P.lo[j], y[j]));
    };
    auto move = [&](Vec& y, std::mt19937& g) {
        shift(y, g);
        ++mutations;
    };
    auto clipped = [&](Vec y) {
        for (std::size_t j = 0; j < n; ++j)
            y[j] = periodic::is(P.periodic, j) ? periodic::wrap_into(y[j], P.lo[j], P.hi[j])
                                               : std::min(P.hi[j], std::max(P.lo[j], y[j]));
        return y;
    };
    long long init_shifts = 0, init_seeded = 0;                                      // CGA-5
    Start head = start;                                                              // member 1's start
    if (init_mode == "seed_points") {
        head.has_seed_points = false;                                                // the seeds are members
        head.seed_points.clear();
    }
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
    auto member = [&](std::mt19937& g) {                                             // CGA-5
        Vec x(n);
        if (init_mode == "around_x0") {
            x = clipped(start.x0);
            for (long long m = 0; m < init_moves; ++m, ++init_shifts) shift(x, g);
        } else if (init_mode == "seed_points" && start.has_seed_points &&
                   static_cast<std::size_t>(init_seeded) < start.seed_points.size()) {
            x = start.seed_points[static_cast<std::size_t>(init_seeded++)].x;
            if (x.size() != n) throw std::invalid_argument("cluster_ga: a seed point is not of n values");
            x = clipped(x);
        } else {
            for (std::size_t j = 0; j < n; ++j) x[j] = P.lo[j] + rnd::uniform01(g) * search::width(P, j);
        }
        return x;
    };
    // 1. the children of ps (CGA-2, CGA-3) from generator g: their parents and starts
    auto breed = [&](const std::vector<search::Found>& ps, std::mt19937& g,
                     std::vector<std::pair<std::size_t, std::size_t>>& parents, std::vector<StartSpec>& kids) {
        const long long n_kids = kids0 > 0 ? kids0 : K;
        if (K == 1) {
            parents.push_back({0, 0});
        } else if (md || afpo) {
            auto any = [&]() {
                return std::min(ps.size() - 1, static_cast<std::size_t>(rnd::uniform01(g) * static_cast<double>(ps.size())));
            };
            auto tournament = [&]() {                                            // Segura: on f
                const std::size_t a = any(), b = any();
                return search::better(ps[b], ps[a]) ? b : a;
            };
            for (long long c = 0; c < n_kids; ++c) {
                const std::size_t a = md ? tournament() : any();
                const std::size_t b = md ? tournament() : any();
                parents.push_back({a, b});
            }
        } else if (pairs) {
            for (std::size_t a = 0; a < ps.size(); ++a)
                for (std::size_t b = a + 1; b < ps.size(); ++b) parents.push_back({a, b});
        } else {
            double lo_e = ps[0].f, hi_e = ps[0].f;
            for (const auto& m : ps) {
                lo_e = std::min(lo_e, m.f);
                hi_e = std::max(hi_e, m.f);
            }
            const double T = hi_e - lo_e;
            std::vector<double> w(ps.size());
            double sum = 0.0;
            for (std::size_t q = 0; q < ps.size(); ++q) sum += w[q] = T > 0.0 ? dm::exp(-(ps[q].f - lo_e) / T) : 1.0;
            auto draw = [&]() {
                double u = rnd::uniform01(g) * sum;
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
        for (const auto& pr : parents) {
            Vec y;
            long long xo = -1;
            bool moved = false;
            last_m = -1;
            if (K == 1) {
                y = ps[0].x;
                move(y, g);                                                         // CGA-2
                moved = true;
            } else {
                const Vec& a = ps[pr.first].x;
                const Vec& b = ps[pr.second].x;
                if (ops_x) {                                                         // CGA-7
                    const std::size_t o = xport.pick(g);
                    Vec c1(n), c2(n);
                    P.crossover_ops[o].second(a, b, c1, c2, g);
                    if (c1.size() != n) throw std::runtime_error("cluster_ga: a crossover operator gave a child of another length");
                    y = c1;
                    xo = static_cast<long long>(o);
                    ++xport.calls[o];
                } else if (user_x) {
                    Vec c1(n), c2(n);
                    P.crossover(a, b, c1, c2, g);
                    if (c1.size() != n) throw std::runtime_error("cluster_ga: the crossover hook gave a child of another length");
                    y = c1;
                } else if (points) {
                    y = cga::cut_and_splice(P, P.sym_blocks[0], a, b, g);
                } else {                                                         // CGA-1: uniform
                    y = a;
                    for (std::size_t j = 0; j < n; ++j)
                        if (rnd::uniform01(g) < 0.5) y[j] = b[j];
                }
                ++crossings;
                if (rnd::uniform01(g) < mut_rate) {
                    move(y, g);
                    moved = true;
                }
            }
            (void)rnd::uniform01(g);                                           // PBH-4
            kids.push_back(spec(search::at(y)));
            const long long mo = moved ? last_m : -1;
            if (mo >= 0) ++mport.calls[static_cast<std::size_t>(mo)];
            kid_x.push_back(xo);
            kid_m.push_back(mo);
            kid_pf.push_back(K == 1 ? ps[0].f : std::min(ps[pr.first].f, ps[pr.second].f));
        }
    };
    // 3. c in place of the highest member of ps where lower and none within cga_de of it
    auto settle = [&](std::vector<search::Found>& ps, const search::Found& c) {
        std::size_t worst = 0;
        bool near = false;
        for (std::size_t q = 0; q < ps.size(); ++q) {
            if (search::better(ps[worst], ps[q])) worst = q;
            if (std::fabs(ps[q].f - c.f) < de) near = true;
        }
        if (near || !search::better(c, ps[worst])) return false;
        ps[worst] = c;
        return true;
    };
    // ... the children Y placed into ps one after the other (worst_if_better); false where one failed
    auto place = [&](std::vector<search::Found>& ps, const std::vector<search::Found>& Y, long long first_kid) {
        bool all = true;
        for (std::size_t p = 0; p < Y.size(); ++p) {
            const search::Found& c = Y[p];
            if (!c.ok) {
                all = false;
                continue;
            }
            log.add(c.x, c.f, first_kid + static_cast<long long>(p));
            const bool in = settle(ps, c);
            if (in) ++accepted;
            placed.push_back({static_cast<std::size_t>(first_kid + static_cast<long long>(p) - kids_from), in, c.f});
        }
        return all;
    };
    // CGA-7: a generation's rewards, child by child in their order: 1 where the
    // child entered the population, else 0 — its crossover's and its mutation's
    auto credit = [&](std::mt19937& g) {
        for (const Placed& c : placed) {
            const double R = c.taken ? 1.0 : 0.0;
            const double d = kid_pf[c.kid] - c.f;
            if (kid_x[c.kid] >= 0) {
                const std::size_t o = static_cast<std::size_t>(kid_x[c.kid]);
                xport.taken[o] += c.taken ? 1 : 0;
                xport.gain[o] += d;
                ++xport.scored[o];
                xport.reward(o, R, g);
            }
            if (kid_m[c.kid] >= 0) {
                const std::size_t o = static_cast<std::size_t>(kid_m[c.kid]);
                mport.taken[o] += c.taken ? 1 : 0;
                mport.gain[o] += d;
                ++mport.scored[o];
                mport.reward(o, R, g);
            }
        }
    };
    auto fresh_generation = [&]() {
        kid_x.clear();
        kid_m.clear();
        kid_pf.clear();
        placed.clear();
        kids_from = searches;
    };
    double D_I = 0.0;
    try {
        if (I > 1) {                                                                 // CGA-6: the islands
            std::vector<std::mt19937> gens{rng};
            for (long long i = 1; i < I; ++i)
                gens.emplace_back(start_seed(seed, 0x9E3779B9u + static_cast<std::uint64_t>(i)));
            const std::size_t KK = static_cast<std::size_t>(K);
            std::vector<StartSpec> init{spec(head)};
            for (long long p = 1; p < K; ++p) init.push_back(spec(search::at(member(gens[0]))));
            for (std::size_t i = 1; i < gens.size(); ++i)
                for (long long p = 0; p < K; ++p) init.push_back(spec(search::at(member(gens[i]))));
            const long long first = searches - K * I;
            const std::vector<search::Found> all0 = run_round(init);
            for (std::size_t p = 0; p < all0.size(); ++p)
                if (all0[p].ok) log.add(all0[p].x, all0[p].f, first + static_cast<long long>(p));
            if (all0.size() < KK * gens.size()) throw BudgetSpent();
            for (const auto& m : all0)
                if (!m.ok) throw BudgetSpent();
            std::vector<std::vector<search::Found>> isles;
            for (std::size_t i = 0; i < gens.size(); ++i)
                isles.emplace_back(all0.begin() + static_cast<std::ptrdiff_t>(i * KK),
                                   all0.begin() + static_cast<std::ptrdiff_t>((i + 1) * KK));
            auto gather = [&]() {
                pop.clear();
                for (const auto& is : isles) pop.insert(pop.end(), is.begin(), is.end());
            };
            gather();
            stop.mark(ev);
            while (idle < 100 && (maxgen == 0 || generations < maxgen) && stop.go(ev)) {   // SRCH-10
                const double before = ev.used();
                std::vector<StartSpec> kids;
                std::vector<std::size_t> made;
                fresh_generation();
                const long long first_kid = searches;
                for (std::size_t i = 0; i < isles.size(); ++i) {
                    std::vector<std::pair<std::size_t, std::size_t>> parents;
                    const std::size_t had = kids.size();
                    breed(isles[i], gens[i], parents, kids);
                    made.push_back(kids.size() - had);
                }
                const std::vector<search::Found> Y = run_round(kids);
                ++generations;
                stop.step(ev);
                bool all = Y.size() == kids.size();
                std::size_t at = 0;
                for (std::size_t i = 0; i < isles.size(); ++i) {
                    const std::size_t from = std::min(at, Y.size()), to = std::min(at + made[i], Y.size());
                    const std::vector<search::Found> Yi(Y.begin() + static_cast<std::ptrdiff_t>(from),
                                                        Y.begin() + static_cast<std::ptrdiff_t>(to));
                    all = place(isles[i], Yi, first_kid + static_cast<long long>(at)) && all;
                    at += made[i];
                }
                credit(gens[0]);
                gather();
                if (!all) break;
                if (generations % mig_every == 0) {                                  // the ring
                    std::vector<std::vector<search::Found>> best(isles.size());
                    for (std::size_t i = 0; i < isles.size(); ++i) {
                        best[i] = isles[i];
                        std::stable_sort(best[i].begin(), best[i].end(),
                                         [](const search::Found& u, const search::Found& v) { return search::better(u, v); });
                        best[i].resize(static_cast<std::size_t>(migrants));
                    }
                    for (std::size_t i = 0; i < isles.size(); ++i)
                        for (const search::Found& c : best[(i + isles.size() - 1) % isles.size()])
                            if (settle(isles[i], c)) ++migrated;
                    gather();
                }
                idle = ev.used() == before ? idle + 1 : 0;
            }
        } else {
            std::vector<StartSpec> init{spec(head)};
            for (long long p = 1; p < K; ++p) init.push_back(spec(search::at(member(rng))));
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
            stop.mark(ev);
            while (idle < 100 && (maxgen == 0 || generations < maxgen) && stop.go(ev)) {   // SRCH-10
                const double before = ev.used();
                // 1. the children (CGA-2, CGA-3)
                std::vector<std::pair<std::size_t, std::size_t>> parents;
                std::vector<StartSpec> kids;
                fresh_generation();
                const long long first_kid = searches;
                breed(pop, rng, parents, kids);
                if (afpo) {                                                              // AF-1: the newcomer
                    Vec x(n);
                    for (std::size_t j = 0; j < n; ++j) x[j] = P.lo[j] + rnd::uniform01(rng) * search::width(P, j);
                    kids.push_back(spec(search::at(x)));
                }
                const std::vector<search::Found> Y = run_round(kids);
                ++generations;
                stop.step(ev);
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
                all = place(pop, Y, first_kid) && all;
                credit(rng);
                if (!all) break;
                idle = ev.used() == before ? idle + 1 : 0;
            }
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
    if (init_mode == "around_x0") extra.push_back({"init_moves", std::to_string(init_shifts)});      // CGA-5
    if (init_mode == "seed_points") extra.push_back({"init_seeded", std::to_string(init_seeded)});
    if (I > 1) extra.push_back({"migrated", std::to_string(migrated)});                       // CGA-6
    if (ops_x || ops_m) {                                                            // CGA-7
        json::Value rows = json::Value::array();
        auto rows_of = [&](const cga::Portfolio& pf, const char* kind) {
            for (std::size_t o = 0; o < pf.names.size(); ++o) {
                json::Value row = json::Value::array();
                row.items.push_back(json::Value::string(pf.names[o]));
                row.items.push_back(json::Value::string(kind));
                row.items.push_back(json::Value::integer(pf.calls[o]));
                row.items.push_back(json::Value::integer(pf.taken[o]));
                row.items.push_back(json::Value::number(pf.scored[o] ? pf.gain[o] / static_cast<double>(pf.scored[o]) : 0.0));
                row.items.push_back(json::Value::number(pf.P[o]));
                rows.items.push_back(row);
            }
        };
        rows_of(xport, "crossover");
        rows_of(mport, "mutation");
        extra.push_back({"operators", json::dump(rows, -1, ",", ":")});
    }
    Result r = search::finish(ev, k, log, extra, stop.stopped_by(ev, idle >= 100 ? "idle" : "rule"));
    for (const auto& m : pop)                                                        // param.hpp PARAM-9
        if (m.ok) r.population.push_back(m.x);
    return r;
}

}   // namespace mootation::scalar
