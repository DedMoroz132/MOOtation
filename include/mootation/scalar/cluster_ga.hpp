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
//     crossovers' calls add up to the run's crossings. Since task 31's layer
//     2 the portfolio is every strategy's (portfolio.hpp), the move is
//     moves.hpp's (mutation = builtin | user | ops are its box | user |
//     portfolio) and every child's outcome goes to Problem::on_child
//     (MOVE-4), in every cga_replace; the runs are those of before, bit for
//     bit.
//   CGA-8 (only when asked: the resumable API). cluster_ga is resumable
//     (task 31, layer 3; resumable.hpp): class ClusterGa keeps the
//     generations' state between pieces, a step a generation; run_cluster_ga
//     is ClusterGa in one piece. Its members are the population (every
//     island's); admit needs one population (cga_islands = 1), and
//     worst_if_better is step 3's rule with cga_de, as the islands'
//     migrants; a member admitted into afpo's population is of age 1.
//   Not here: the pool mode, the
//   stagnation trigger of mutation, the predator, the outlier parents of
//   Jørgensen et al. and USPEX's antiseeds; Wolf and Landman's two-stage
//   relaxation (WL-1: E - |grad E|, read on the PDF by task 25).
// ============================================================================

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "hooks.hpp"
#include "resumable.hpp"
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
            child[j] = (std::min)(P.hi[j], (std::max)(P.lo[j], v));
        }
    }
    return child;
}

// the operator portfolio of CGA-7, now every strategy's (task 31, layer 2;
// portfolio.hpp)
using Portfolio = moves::Portfolio;

}   // namespace cga

// cluster_ga as a resumable run (task 31, layer 3; resumable.hpp): the state
// of the generations' loop kept between pieces, a step one generation.
// run_cluster_ga is ClusterGa in one piece (RES-1).
class ClusterGa : public Resumable {
public:
    std::unique_ptr<Problem> own;                       // RES-8: the run's copy of the problem, where it has one
    std::function<void(Result&)> tail;                  // ... and what scalar::run adds to the result

    ClusterGa(const Problem& P, double eq_tol, bool has_tol0, double eq_tol0, int pop_size, long long budget,
              unsigned seed, const Knobs& knobs, Observer observer, int record_every, const Start& start)
        : P_(P), knobs_(knobs),
          k_("cluster_ga", knobs_,
             {"inner", "cga_pop", "cga_mating", "cga_children", "cga_mut_rate", "cga_step", "cga_de", "cga_maxgen",
              "crossover", "mutation", "threads", "start_budget", "min_dist", "min_df", "keep_points", "cga_replace",
              "md_di", "md_end", "md_distance", "afpo_tournament", "cga_init", "cga_init_moves", "cga_islands",
              "cga_migrate_every", "cga_migrants", "op_select", "op_pmin", "op_alpha", "op_beta"}),
          seed_(seed), pop_size_(pop_size), budget_(budget), start_(start)
    {
        using namespace strategy;
        inner_ = k_.inner("inner");
        K_ = k_.whole("cga_pop", 8, 1);
        threads_ = k_.whole("threads", 1, 1);
        pairs_ = k_.word("cga_mating", "boltzmann", {"boltzmann", "all_pairs"}) == "all_pairs";
        kids0_ = k_.whole("cga_children", 0, 0);
        mut_rate_ = k_.number("cga_mut_rate", 0.05);
        step_ = k_.number("cga_step", 0.1);
        de_ = k_.number("cga_de", 1e-6);
        maxgen_ = k_.whole("cga_maxgen", 0, 0);
        cap_ = k_.whole("start_budget", 0, 0);
        const std::string x_how =
            k_.find("crossover") ? k_.word("crossover", "builtin", {"builtin", "user", "ops"}) : "builtin";
        const std::string m_how =
            k_.find("mutation") ? k_.word("mutation", "builtin", {"builtin", "user", "ops"}) : "builtin";
        user_x_ = x_how == "user";
        const bool user_m = m_how == "user";
        ops_x_ = x_how == "ops";
        ops_m_ = m_how == "ops";                                                     // CGA-7
        const double min_dist = k_.number("min_dist", 1e-3), min_df = k_.number("min_df", 1e-8);
        const std::string mode = k_.word("cga_replace", "worst_if_better", {"worst_if_better", "multi_dynamic", "afpo"});
        md_ = mode == "multi_dynamic";
        afpo_ = mode == "afpo";
        md_di_ = md_ ? k_.number("md_di", 0.5) : 0.0;
        md_end_ = md_ ? k_.number("md_end", 1.0) : 1.0;
        md_aligned_ = md_ && k_.word("md_distance", "euclid", {"euclid", "aligned"}) == "aligned";
        tour_ = afpo_ ? k_.whole("afpo_tournament", 2, 2) : 2;
        if (md_ && !(md_di_ >= 0.0)) throw std::invalid_argument("cluster_ga: md_di must be >= 0");
        if (md_ && !(md_end_ > 0.0)) throw std::invalid_argument("cluster_ga: md_end must be > 0");
        if (!(mut_rate_ >= 0.0 && mut_rate_ <= 1.0)) throw std::invalid_argument("cluster_ga: cga_mut_rate in [0, 1]");
        if (!(step_ > 0.0)) throw std::invalid_argument("cluster_ga: cga_step must be > 0");
        if (!(de_ >= 0.0)) throw std::invalid_argument("cluster_ga: cga_de must be >= 0");
        if (user_x_ && !P.crossover) throw std::invalid_argument("cluster_ga: crossover = user needs the crossover hook");
        if (user_m && !P.mutate) throw std::invalid_argument("cluster_ga: mutation = user needs the mutate hook");
        if (ops_x_ && P.crossover_ops.empty()) throw std::invalid_argument("cluster_ga: crossover = ops needs crossover_ops");
        if (ops_m_ && P.mutate_ops.empty()) throw std::invalid_argument("cluster_ga: mutation = ops needs mutate_ops");
        mv_.kind = user_m ? moves::Kind::user : ops_m_ ? moves::Kind::portfolio : moves::Kind::box;   // the move (moves.hpp)
        if (ops_x_ || ops_m_) {                                                      // CGA-7
            if (md_ || afpo_)
                throw std::invalid_argument("cluster_ga: crossover or mutation = ops needs cga_replace = worst_if_better");
            double alpha = 0.8, beta = 0.8;
            const int rule = moves::rule_of(k_, "cluster_ga", alpha, beta);
            if (ops_x_)
                xport_.start(moves::names_of(P.crossover_ops), rule,
                             moves::pmin_of(k_, "cluster_ga", rule, P.crossover_ops.size()), alpha, beta);
            if (ops_m_)
                mv_.ops.start(moves::names_of(P.mutate_ops), rule,
                              moves::pmin_of(k_, "cluster_ga", rule, P.mutate_ops.size()), alpha, beta);
        }
        points_ = !P.sym_blocks.empty() && (P.sym_blocks[0].size == 2 || P.sym_blocks[0].size == 3);
        search::finite_box("cluster_ga", P);
        n_ = P.lo.size();
        init_mode_ =                                                                 // CGA-5
            k_.find("cga_init") ? k_.word("cga_init", "uniform", {"uniform", "around_x0", "seed_points"}) : "uniform";
        init_moves_ = init_mode_ == "around_x0" ? k_.whole("cga_init_moves", 1, 1) : 0;
        if (init_mode_ == "around_x0" && !(start.has_x0 && start.x0.size() == n_))
            throw std::invalid_argument("cluster_ga: cga_init = around_x0 needs x0 (of n values)");
        I_ = k_.find("cga_islands") ? k_.whole("cga_islands", 1, 1) : 1;              // CGA-6
        mig_every_ = I_ > 1 ? k_.whole("cga_migrate_every", 5, 1) : 0;
        migrants_ = I_ > 1 ? k_.whole("cga_migrants", 1, 1) : 0;
        if (I_ > 1 && (md_ || afpo_))
            throw std::invalid_argument("cluster_ga: cga_islands > 1 needs cga_replace = worst_if_better");
        if (I_ > 1 && migrants_ > K_) throw std::invalid_argument("cluster_ga: cga_migrants at most cga_pop");
        ev_ = std::make_unique<Evaluator>(P, eq_tol, has_tol0, eq_tol0, budget, pop_size, std::move(observer),
                                          record_every);
        strategy::round_stop_of(k_, *ev_);                                           // STRAT-16
        stop_ = strategy::stop_of(k_, *ev_);                                         // stop_rule.hpp: a step a generation
        ev_->lean = lean_of(k_);
        log_ = std::make_unique<MinimaLog>(P.lo, P.hi, min_dist, min_df);
        group_by_symmetry(k_, P, *log_);
        rng_.seed(start_seed(seed, 0x9E3779B9u));                                    // CGA-3: pbh's stream
        head_ = start;                                                               // member 1's start
        if (init_mode_ == "seed_points") {
            head_.has_seed_points = false;                                           // the seeds are members
            head_.seed_points.clear();
        }
    }

    const std::string& name() const override { return name_; }

    void begin() override
    {
        if (begun_) return;
        begun_ = true;
        try {
            if (I_ > 1) begin_islands();
            else begin_one();
        } catch (const BudgetSpent&) {
            over_ = true;
        }
    }

    bool advance(const Piece& piece = Piece()) override
    {
        const double from = ev_->counted();
        begin();
        long long taken = 0;
        while (!over_) {
            if ((piece.steps > 0 && taken >= piece.steps) || (piece.spend > 0.0 && ev_->counted() - from >= piece.spend)) {
                if (!can_step()) over_ = true;                                        // RES-2: the run's end found now
                break;
            }
            if (!can_step()) {
                over_ = true;
                break;
            }
            try {
                if (I_ > 1) step_islands();
                else step_one();
            } catch (const BudgetSpent&) {
                over_ = true;
            }
            ++taken;
        }
        return !over_;
    }

    bool done() const override { return over_; }
    long long steps() const override { return stop_.steps(); }
    const Evaluator& ledger() const override { return *ev_; }

    void set_budget(long long budget) override                                      // RES-5
    {
        ev_->limit = ev_->limit == ev_->budget ? budget : (std::min)(ev_->limit, budget);
        ev_->budget = budget;
    }

    std::vector<Member> members() const override
    {
        std::vector<Member> out;
        for (const search::Found& m : pop_)
            if (m.ok) out.push_back(member_of(*ev_, m.x, m.f, m.cv));
        return out;
    }

    // RES-7: the k best members
    std::vector<Member> best_points(std::size_t k) const override
    {
        std::vector<search::Found> ps;
        for (const search::Found& m : pop_)
            if (m.ok) ps.push_back(m);
        std::stable_sort(ps.begin(), ps.end(), [](const search::Found& u, const search::Found& v) { return search::better(u, v); });
        std::vector<Member> out;
        for (std::size_t q = 0; q < ps.size() && q < k; ++q) out.push_back(member_of(*ev_, ps[q].x, ps[q].f, ps[q].cv));
        return out;
    }

    // RES-3, RES-4: into the one population (not with cga_islands > 1)
    long long admit(const std::vector<Member>& in, Place how) override
    {
        one_population("admit");
        long long taken = 0;
        std::vector<char> fresh(pop_.size(), 0);
        for (std::size_t q = 0; q < in.size(); ++q) {
            const search::Found c = planted(in[q]);
            if (!c.ok) break;
            if (how == Place::all) {
                if (q >= pop_.size()) break;
                put(q, c);
                ++taken;
                continue;
            }
            if (how == Place::worst_if_better) {
                if (settle(pop_, c)) ++taken;                                         // step 3, with cga_de
                continue;
            }
            std::size_t worst = pop_.size();
            for (std::size_t p = 0; p < pop_.size(); ++p)
                if (!fresh[p] && (worst == pop_.size() || search::better(pop_[worst], pop_[p]))) worst = p;
            if (worst == pop_.size()) break;
            put(worst, c);
            fresh[worst] = 1;
            ++taken;
        }
        return taken;
    }

    long long admit(const std::vector<Member>& in, const PlaceRule& rule) override
    {
        one_population("admit");
        long long taken = 0;
        for (const Member& m : in) {
            const long long at = rule(members(), m);
            if (at < 0 || at >= static_cast<long long>(pop_.size())) continue;
            const search::Found c = planted(m);
            if (!c.ok) break;
            put(static_cast<std::size_t>(at), c);
            ++taken;
        }
        return taken;
    }

    Operators operators() const override                                         // RES-6
    {
        Operators o;
        o.mutation = mv_.ops;
        o.crossover = xport_;
        return o;
    }

    void set_operators(const Operators& o) override
    {
        if (o.mutation.on()) {
            if (o.mutation.names != mv_.ops.names)
                throw std::invalid_argument("cluster_ga: set_operators: a portfolio of other operators (RES-6)");
            mv_.ops = o.mutation;
        }
        if (o.crossover.on()) {
            if (o.crossover.names != xport_.names)
                throw std::invalid_argument("cluster_ga: set_operators: a portfolio of other operators (RES-6)");
            xport_ = o.crossover;
        }
    }

    Member best() const override { return best_member(*ev_); }

    std::vector<std::pair<std::string, std::string>> summary() const override
    {
        std::vector<std::pair<std::string, std::string>> extra{{"generations", std::to_string(generations_)},
                                                               {"crossings", std::to_string(crossings_)},
                                                               {"mutations", std::to_string(mutations_)},
                                                               {"accepted", std::to_string(accepted_)}};
        if (afpo_) {
            std::string ages;
            for (std::size_t q = 0; q < age_.size(); ++q) ages += (q ? "," : "") + std::to_string(age_[q]);
            extra.push_back({"ages", ages});
        }
        if (md_) extra.push_back({"md_d_initial", detail::g_format(D_I_)});
        if (init_mode_ == "around_x0") extra.push_back({"init_moves", std::to_string(init_shifts_)});      // CGA-5
        if (init_mode_ == "seed_points") extra.push_back({"init_seeded", std::to_string(init_seeded_)});
        if (I_ > 1) extra.push_back({"migrated", std::to_string(migrated_)});                       // CGA-6
        if (ops_x_ || ops_m_) extra.push_back({"operators", moves::operators_row(&xport_, &mv_.ops)});   // CGA-7
        return extra;
    }

    Result result() override
    {
        over_ = true;
        Result r = search::finish(*ev_, k_, *log_, summary(), stop_.stopped_by(*ev_, idle_ >= 100 ? "idle" : "rule"));
        for (const auto& m : pop_)                                                   // param.hpp PARAM-9
            if (m.ok) r.population.push_back(m.x);
        if (tail) tail(r);
        return r;
    }

private:
    struct Placed {
        std::size_t kid;
        bool taken;
        double f;
    };

    std::string name_ = "cluster_ga";
    const Problem& P_;
    Knobs knobs_;
    strategy::StrategyKnobs k_;
    unsigned seed_ = 0;
    int pop_size_ = 1;
    long long budget_ = 0;                            // the budget at the start (MD-1's T_End; RES-5)
    Start start_, head_;
    strategy::Inner inner_;
    long long K_ = 8, threads_ = 1, kids0_ = 0, maxgen_ = 0, cap_ = 0, tour_ = 2;
    bool pairs_ = false, user_x_ = false, ops_x_ = false, ops_m_ = false;
    double mut_rate_ = 0.05, step_ = 0.1, de_ = 1e-6;
    bool md_ = false, afpo_ = false, md_aligned_ = false;
    double md_di_ = 0.0, md_end_ = 1.0;
    moves::Move mv_;                                  // the move; its portfolio is the mutations' (CGA-7)
    cga::Portfolio xport_;                            // CGA-7
    std::vector<moves::Child> kid_;                   // a generation's children (MOVE-4)
    std::vector<Placed> placed_;                      // ... what became of them (CGA-7)
    long long kids_from_ = 0, last_m_ = -1;
    bool points_ = false;
    std::size_t n_ = 0;
    std::string init_mode_ = "uniform";
    long long init_moves_ = 0;
    long long I_ = 1, mig_every_ = 0, migrants_ = 0;
    std::unique_ptr<Evaluator> ev_;
    strategy::StopRule stop_;
    long long idle_ = 0;
    std::unique_ptr<strategy::MinimaLog> log_;
    std::mt19937 rng_;
    long long searches_ = 0, generations_ = 0, crossings_ = 0, mutations_ = 0, accepted_ = 0, migrated_ = 0;
    long long init_shifts_ = 0, init_seeded_ = 0;                                    // CGA-5
    std::vector<search::Found> pop_;
    std::vector<long long> age_;                                                     // AF-1
    double D_I_ = 0.0;
    std::vector<std::mt19937> gens_;                                                 // CGA-6: the islands
    std::vector<std::vector<search::Found>> isles_;
    bool begun_ = false, over_ = false;

    strategy::StartSpec spec(const Start& st)
    {
        strategy::StartSpec sp;
        sp.seed = strategy::start_seed(seed_, static_cast<std::uint64_t>(searches_++));
        sp.start = st;
        return sp;
    }

    std::vector<search::Found> run_round(std::vector<strategy::StartSpec> specs)
    {
        const std::vector<Result> r =
            search::starts(*ev_, inner_, std::move(specs), cap_, static_cast<int>(threads_), pop_size_);
        std::vector<search::Found> out;
        for (const Result& q : r) out.push_back(search::found(q));
        return out;
    }

    void shift(Vec& y, std::mt19937& g)                                             // pbh's move (moves.hpp)
    {
        last_m_ = mv_.apply(P_, y, step_, g, "cluster_ga");
    }

    void move(Vec& y, std::mt19937& g)
    {
        shift(y, g);
        ++mutations_;
    }

    Vec clipped(Vec y) const
    {
        for (std::size_t j = 0; j < n_; ++j)
            y[j] = periodic::is(P_.periodic, j) ? periodic::wrap_into(y[j], P_.lo[j], P_.hi[j])
                                                : (std::min)(P_.hi[j], (std::max)(P_.lo[j], y[j]));
        return y;
    }

    double dist(const Vec& a, const Vec& b) const                                  // MD-1
    {
        if (md_aligned_) return distance_of(P_, a, b);
        double q = 0.0;
        for (std::size_t j = 0; j < n_; ++j) {
            const double w = search::width(P_, j);
            const double g = periodic::gap(a[j], b[j], periodic::is(P_.periodic, j), w) / (w > 0.0 ? w : 1.0);
            q += g * g;
        }
        return std::sqrt(q);
    }

    Vec member(std::mt19937& g)                                                     // CGA-5
    {
        Vec x(n_);
        if (init_mode_ == "around_x0") {
            x = clipped(start_.x0);
            for (long long m = 0; m < init_moves_; ++m, ++init_shifts_) shift(x, g);
        } else if (init_mode_ == "seed_points" && start_.has_seed_points &&
                   static_cast<std::size_t>(init_seeded_) < start_.seed_points.size()) {
            x = start_.seed_points[static_cast<std::size_t>(init_seeded_++)].x;
            if (x.size() != n_) throw std::invalid_argument("cluster_ga: a seed point is not of n values");
            x = clipped(x);
        } else {
            for (std::size_t j = 0; j < n_; ++j) x[j] = P_.lo[j] + rnd::uniform01(g) * search::width(P_, j);
        }
        return x;
    }

    // 1. the children of ps (CGA-2, CGA-3) from generator g: their parents and starts
    void breed(const std::vector<search::Found>& ps, std::mt19937& g, std::vector<std::pair<std::size_t, std::size_t>>& parents,
               std::vector<strategy::StartSpec>& kids)
    {
        const long long n_kids = kids0_ > 0 ? kids0_ : K_;
        if (K_ == 1) {
            parents.push_back({0, 0});
        } else if (md_ || afpo_) {
            auto any = [&]() {
                return (std::min)(ps.size() - 1, static_cast<std::size_t>(rnd::uniform01(g) * static_cast<double>(ps.size())));
            };
            auto tournament = [&]() {                                            // Segura: on f
                const std::size_t a = any(), b = any();
                return search::better(ps[b], ps[a]) ? b : a;
            };
            for (long long c = 0; c < n_kids; ++c) {
                const std::size_t a = md_ ? tournament() : any();
                const std::size_t b = md_ ? tournament() : any();
                parents.push_back({a, b});
            }
        } else if (pairs_) {
            for (std::size_t a = 0; a < ps.size(); ++a)
                for (std::size_t b = a + 1; b < ps.size(); ++b) parents.push_back({a, b});
        } else {
            double lo_e = ps[0].f, hi_e = ps[0].f;
            for (const auto& m : ps) {
                lo_e = (std::min)(lo_e, m.f);
                hi_e = (std::max)(hi_e, m.f);
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
            last_m_ = -1;
            if (K_ == 1) {
                y = ps[0].x;
                move(y, g);                                                         // CGA-2
                moved = true;
            } else {
                const Vec& a = ps[pr.first].x;
                const Vec& b = ps[pr.second].x;
                if (ops_x_) {                                                        // CGA-7
                    const std::size_t o = xport_.pick(g);
                    Vec c1(n_), c2(n_);
                    P_.crossover_ops[o].second(a, b, c1, c2, g);
                    if (c1.size() != n_) throw std::runtime_error("cluster_ga: a crossover operator gave a child of another length");
                    y = c1;
                    xo = static_cast<long long>(o);
                    ++xport_.calls[o];
                } else if (user_x_) {
                    Vec c1(n_), c2(n_);
                    P_.crossover(a, b, c1, c2, g);
                    if (c1.size() != n_) throw std::runtime_error("cluster_ga: the crossover hook gave a child of another length");
                    y = c1;
                    xo = 0;                                                          // MOVE-4: the hook's tag
                } else if (points_) {
                    y = cga::cut_and_splice(P_, P_.sym_blocks[0], a, b, g);
                } else {                                                         // CGA-1: uniform
                    y = a;
                    for (std::size_t j = 0; j < n_; ++j)
                        if (rnd::uniform01(g) < 0.5) y[j] = b[j];
                }
                ++crossings_;
                if (rnd::uniform01(g) < mut_rate_) {
                    move(y, g);
                    moved = true;
                }
            }
            (void)rnd::uniform01(g);                                           // PBH-4
            kids.push_back(spec(search::at(y)));
            const long long mo = moved ? last_m_ : -1;
            if (mo >= 0 && mv_.ops.on()) ++mv_.ops.calls[static_cast<std::size_t>(mo)];
            moves::Child ch;                                                         // MOVE-4: the child's record
            ch.crossover = xo;
            ch.mutation = mo;
            ch.parents_f = K_ == 1 ? Vec{ps[0].f} : Vec{ps[pr.first].f, ps[pr.second].f};
            kid_.push_back(ch);
        }
    }

    // 3. c in place of the highest member of ps where lower and none within cga_de of it
    bool settle(std::vector<search::Found>& ps, const search::Found& c) const
    {
        std::size_t worst = 0;
        bool nearby = false;
        for (std::size_t q = 0; q < ps.size(); ++q) {
            if (search::better(ps[worst], ps[q])) worst = q;
            if (std::fabs(ps[q].f - c.f) < de_) nearby = true;
        }
        if (nearby || !search::better(c, ps[worst])) return false;
        ps[worst] = c;
        return true;
    }

    // ... the children Y placed into ps one after the other (worst_if_better); false where one failed
    bool place(std::vector<search::Found>& ps, const std::vector<search::Found>& Y, long long first_kid)
    {
        bool all = true;
        for (std::size_t p = 0; p < Y.size(); ++p) {
            const search::Found& c = Y[p];
            if (!c.ok) {
                all = false;
                continue;
            }
            log_->add(c.x, c.f, first_kid + static_cast<long long>(p));
            const bool in = settle(ps, c);
            if (in) ++accepted_;
            placed_.push_back({static_cast<std::size_t>(first_kid + static_cast<long long>(p) - kids_from_), in, c.f});
        }
        return all;
    }

    // CGA-7: a generation's rewards, child by child in their order: 1 where the
    // child entered the population, else 0 — its crossover's and its mutation's;
    // then on_child (moves.hpp MOVE-4, MOVE-5)
    void credit(std::mt19937& g)
    {
        for (const Placed& c : placed_) moves::told(P_, &xport_, &mv_.ops, kid_[c.kid], c.f, c.taken, g);
    }

    void fresh_generation()
    {
        kid_.clear();
        placed_.clear();
        kids_from_ = searches_;
    }

    void gather()
    {
        pop_.clear();
        for (const auto& is : isles_) pop_.insert(pop_.end(), is.begin(), is.end());
    }

    // the loop's test before a generation (SRCH-10)
    bool can_step() { return idle_ < 100 && (maxgen_ == 0 || generations_ < maxgen_) && stop_.go(*ev_); }

    // 0. CGA-6: every island's first population
    void begin_islands()
    {
        gens_.assign(1, rng_);
        for (long long i = 1; i < I_; ++i)
            gens_.emplace_back(strategy::start_seed(seed_, 0x9E3779B9u + static_cast<std::uint64_t>(i)));
        const std::size_t KK = static_cast<std::size_t>(K_);
        std::vector<strategy::StartSpec> init{spec(head_)};
        for (long long p = 1; p < K_; ++p) init.push_back(spec(search::at(member(gens_[0]))));
        for (std::size_t i = 1; i < gens_.size(); ++i)
            for (long long p = 0; p < K_; ++p) init.push_back(spec(search::at(member(gens_[i]))));
        const long long first = searches_ - K_ * I_;
        const std::vector<search::Found> all0 = run_round(init);
        for (std::size_t p = 0; p < all0.size(); ++p)
            if (all0[p].ok) log_->add(all0[p].x, all0[p].f, first + static_cast<long long>(p));
        if (all0.size() < KK * gens_.size()) throw BudgetSpent();
        for (const auto& m : all0)
            if (!m.ok) throw BudgetSpent();
        for (std::size_t i = 0; i < gens_.size(); ++i)
            isles_.emplace_back(all0.begin() + static_cast<std::ptrdiff_t>(i * KK),
                                all0.begin() + static_cast<std::ptrdiff_t>((i + 1) * KK));
        gather();
        stop_.mark(*ev_);
    }

    // a generation of the islands (CGA-6)
    void step_islands()
    {
        const double before = ev_->used();
        std::vector<strategy::StartSpec> kids;
        std::vector<std::size_t> made;
        fresh_generation();
        const long long first_kid = searches_;
        for (std::size_t i = 0; i < isles_.size(); ++i) {
            std::vector<std::pair<std::size_t, std::size_t>> parents;
            const std::size_t had = kids.size();
            breed(isles_[i], gens_[i], parents, kids);
            made.push_back(kids.size() - had);
        }
        const std::vector<search::Found> Y = run_round(kids);
        ++generations_;
        stop_.step(*ev_);
        bool all = Y.size() == kids.size();
        std::size_t at = 0;
        for (std::size_t i = 0; i < isles_.size(); ++i) {
            const std::size_t from = (std::min)(at, Y.size()), to = (std::min)(at + made[i], Y.size());
            const std::vector<search::Found> Yi(Y.begin() + static_cast<std::ptrdiff_t>(from),
                                                Y.begin() + static_cast<std::ptrdiff_t>(to));
            all = place(isles_[i], Yi, first_kid + static_cast<long long>(at)) && all;
            at += made[i];
        }
        credit(gens_[0]);
        gather();
        if (!all) {
            over_ = true;
            return;
        }
        if (generations_ % mig_every_ == 0) {                                        // the ring
            std::vector<std::vector<search::Found>> top(isles_.size());
            for (std::size_t i = 0; i < isles_.size(); ++i) {
                top[i] = isles_[i];
                std::stable_sort(top[i].begin(), top[i].end(),
                                 [](const search::Found& u, const search::Found& v) { return search::better(u, v); });
                top[i].resize(static_cast<std::size_t>(migrants_));
            }
            for (std::size_t i = 0; i < isles_.size(); ++i)
                for (const search::Found& c : top[(i + isles_.size() - 1) % isles_.size()])
                    if (settle(isles_[i], c)) ++migrated_;
            gather();
        }
        idle_ = ev_->used() == before ? idle_ + 1 : 0;
    }

    // 0. the one population's first members
    void begin_one()
    {
        std::vector<strategy::StartSpec> init{spec(head_)};
        for (long long p = 1; p < K_; ++p) init.push_back(spec(search::at(member(rng_))));
        const long long first = searches_ - K_;
        pop_ = run_round(init);
        for (std::size_t p = 0; p < pop_.size(); ++p)
            if (pop_[p].ok) log_->add(pop_[p].x, pop_[p].f, first + static_cast<long long>(p));
        if (static_cast<long long>(pop_.size()) < K_) throw BudgetSpent();
        for (const auto& m : pop_)
            if (!m.ok) throw BudgetSpent();
        age_.assign(pop_.size(), 1);
        if (md_) {
            double G = std::sqrt(static_cast<double>(n_));
            if (md_aligned_) {
                G = 0.0;
                for (std::size_t a = 0; a < pop_.size(); ++a)
                    for (std::size_t b = a + 1; b < pop_.size(); ++b) G = (std::max)(G, dist(pop_[a].x, pop_[b].x));
            }
            D_I_ = md_di_ * G;
        }
        stop_.mark(*ev_);
    }

    // a generation of the one population
    void step_one()
    {
        const double before = ev_->used();
        // 1. the children (CGA-2, CGA-3)
        std::vector<std::pair<std::size_t, std::size_t>> parents;
        std::vector<strategy::StartSpec> kids;
        fresh_generation();
        const long long first_kid = searches_;
        breed(pop_, rng_, parents, kids);
        if (afpo_) {                                                                 // AF-1: the newcomer
            Vec x(n_);
            for (std::size_t j = 0; j < n_; ++j) x[j] = P_.lo[j] + rnd::uniform01(rng_) * search::width(P_, j);
            kids.push_back(spec(search::at(x)));
        }
        const std::vector<search::Found> Y = run_round(kids);
        ++generations_;
        stop_.step(*ev_);
        // 3. placed one after the other
        bool all = Y.size() == kids.size();
        if (md_ || afpo_) {
            std::vector<search::Found> U = pop_;
            std::vector<long long> A = age_;
            std::vector<char> in_next;                                               // MOVE-4: by U's index
            std::vector<std::size_t> u_of(Y.size(), 0);
            for (std::size_t p = 0; p < Y.size(); ++p) {
                if (!Y[p].ok) {
                    all = false;
                    continue;
                }
                log_->add(Y[p].x, Y[p].f, first_kid + static_cast<long long>(p));
                u_of[p] = U.size();
                U.push_back(Y[p]);
                A.push_back(p < parents.size() ? (std::max)(age_[parents[p].first], age_[parents[p].second]) : 1);
            }
            if (md_) {                                                               // Segura, Algorithm 2
                const double frac = (std::max)(0.0, 1.0 - ev_->used() / (md_end_ * static_cast<double>(budget_)));
                const double D = D_I_ * frac;
                std::vector<char> taken(U.size(), 0);
                std::vector<search::Found> next;
                std::size_t top = 0;
                for (std::size_t q = 1; q < U.size(); ++q)
                    if (search::better(U[q], U[top])) top = q;
                taken[top] = 1;
                next.push_back(U[top]);
                while (static_cast<long long>(next.size()) < K_) {
                    std::vector<std::size_t> cand;
                    std::vector<double> fv, dcn;
                    for (std::size_t q = 0; q < U.size(); ++q) {
                        if (taken[q]) continue;
                        double dmin = std::numeric_limits<double>::infinity();
                        for (const auto& m : next) dmin = (std::min)(dmin, dist(U[q].x, m.x));
                        cand.push_back(q);
                        dcn.push_back(dmin);
                        const bool feasible = U[q].cv <= 0.0;
                        fv.push_back(dmin < D ? std::numeric_limits<double>::infinity()
                                              : (feasible ? U[q].f : std::numeric_limits<double>::infinity()));
                    }
                    if (cand.empty()) break;
                    std::vector<std::size_t> front;                                  // (f min, DCN max), MD-2
                    for (std::size_t a = 0; a < cand.size(); ++a) {
                        bool dominated = false, repeat = false;
                        for (std::size_t b = 0; b < cand.size() && !dominated; ++b)
                            dominated = (fv[b] <= fv[a] && dcn[b] >= dcn[a]) && (fv[b] < fv[a] || dcn[b] > dcn[a]);
                        for (std::size_t f : front) repeat = repeat || (fv[f] == fv[a] && dcn[f] == dcn[a]);
                        if (!dominated && !repeat) front.push_back(a);
                    }
                    const std::size_t pick = front[(std::min)(front.size() - 1,
                                                            static_cast<std::size_t>(rnd::uniform01(rng_) *
                                                                                     static_cast<double>(front.size())))];
                    taken[cand[pick]] = 1;
                    next.push_back(U[cand[pick]]);
                }
                accepted_ += static_cast<long long>(next.size());
                in_next = taken;
                pop_ = next;
                age_.assign(pop_.size(), 1);
            } else {                                                                 // AFPO, AF-2
                auto dominates = [&](std::size_t a, std::size_t b) {
                    const bool fa = search::better(U[a], U[b]) || (!search::better(U[b], U[a]));
                    return fa && A[a] <= A[b] && (search::better(U[a], U[b]) || A[a] < A[b]);
                };
                std::vector<std::size_t> live(U.size());
                for (std::size_t q = 0; q < U.size(); ++q) live[q] = q;
                while (static_cast<long long>(live.size()) > K_) {
                    bool any_dominated = false;
                    for (std::size_t a = 0; a < live.size() && !any_dominated; ++a)
                        for (std::size_t b = 0; b < live.size() && !any_dominated; ++b)
                            any_dominated = a != b && dominates(live[b], live[a]);
                    if (!any_dominated) break;
                    const std::size_t t = std::min<std::size_t>(static_cast<std::size_t>(tour_), live.size());
                    std::vector<std::size_t> group(live);
                    for (std::size_t g = 0; g < t; ++g) {                            // a partial Fisher-Yates draw
                        const std::size_t r = g + (std::min)(group.size() - g - 1,
                                                           static_cast<std::size_t>(rnd::uniform01(rng_) *
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
                        if (static_cast<long long>(live.size()) <= K_) break;
                        live.erase(std::find(live.begin(), live.end(), o));
                    }
                }
                std::vector<search::Found> next;
                std::vector<long long> nage;
                in_next.assign(U.size(), 0);
                for (std::size_t q : live) {
                    next.push_back(U[q]);
                    nage.push_back(A[q] + 1);                                        // AF-1
                    in_next[q] = 1;
                }
                accepted_ += static_cast<long long>(live.size());
                pop_ = next;
                age_ = nage;
            }
            if (P_.on_child)                                                         // MOVE-4: afpo's newcomer is no child
                for (std::size_t p = 0; p < Y.size() && p < parents.size(); ++p)
                    if (Y[p].ok) moves::told(P_, nullptr, nullptr, kid_[p], Y[p].f, in_next[u_of[p]] != 0, rng_);
            if (!all) {
                over_ = true;
                return;
            }
            idle_ = ev_->used() == before ? idle_ + 1 : 0;
            return;
        }
        all = place(pop_, Y, first_kid) && all;
        credit(rng_);
        if (!all) {
            over_ = true;
            return;
        }
        idle_ = ev_->used() == before ? idle_ + 1 : 0;
    }

    void one_population(const char* what) const
    {
        if (I_ > 1) throw std::invalid_argument(std::string("cluster_ga: ") + what + " needs one population (cga_islands = 1)");
    }

    // member q of the population becomes c (an afpo member's age 1)
    void put(std::size_t q, const search::Found& c)
    {
        pop_[q] = c;
        if (q < age_.size()) age_[q] = 1;
    }

    // a member from elsewhere, planted in the archive (RES-3; evaluated where its values are not whole,
    // SEED-1: not ok where that evaluation ended the run)
    search::Found planted(const Member& m)
    {
        std::size_t i = 0;
        search::Found c;
        try {
            i = ev_->seed(m.x, {m.f}, m.g);                                         // RES-3, SEED-1
        } catch (const BudgetSpent&) {
            over_ = true;                                                           // its evaluation ended the run
            return c;
        }
        c.x = ev_->X[i];
        c.f = ev_->F[i][0];
        c.cv = ev_->cv(i);
        c.ok = true;
        return c;
    }
};

inline Result run_cluster_ga(const Problem& P, double eq_tol, bool has_tol0, double eq_tol0, int pop_size,
                             long long budget, unsigned seed, const Knobs& knobs, Observer observer, int record_every,
                             const Start& start)
{
    ClusterGa c(P, eq_tol, has_tol0, eq_tol0, pop_size, budget, seed, knobs, std::move(observer), record_every, start);
    c.advance();                                                                     // RES-1: one piece
    return c.result();
}

}   // namespace mootation::scalar
