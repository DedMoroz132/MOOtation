#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// THE MOVE of the strategies over local minima (task 31, layer 2; task 30's
// §30.14, §30.15 and §30.7 made general): the perturbation of a point that
// basin_hopping, ils (search.hpp), pbh (pbh.hpp) and cluster_ga (cluster_ga.hpp,
// its mutation) make before a local search, and ga_sbx and ssga (ga.hpp, the
// mutation of a child). Included by strategy.hpp.
//
// THE KINDS of the move (the knob step of basin_hopping, ils and pbh;
// mutation of cluster_ga, ga_mutation of ga_sbx and ssga):
//   box        the strategy's own move, as before: every coordinate by
//              U[-1, 1] times the step times the box's width (or in the
//              problem's units, step_units = absolute; SRCH-15); cluster_ga's
//              builtin, ga's polynomial mutation;
//   (perturb)  task 25's perturb = user (SRCH-19): the problem's perturb(y,
//              step, rng), the step its strength;
//   user       the problem's mutate(y, rng) (task 24's hook);
//   portfolio  the problem's mutate_ops, one chosen for each move by the
//              portfolio's rule op_select (portfolio.hpp: fixed,
//              probability_matching, adaptive_pursuit; op_pmin, op_alpha,
//              op_beta): cluster_ga's mutation = ops of task 30, §30.7.
// Every moved point is clipped to the box, a periodic coordinate wrapped
// (SRCH-4, PER-2), whatever the kind.
//
// READINGS (all of them off by default: step = box is the run of before):
//   MOVE-1. Off by default. step = user, portfolio: the hooks have no
//     strength, so basin_hopping's step adaptation (SRCH-5) and the strength
//     ladder (SRCH-18) are off — the ladder's knobs are refused, bh_step,
//     ils_step, pbh_step and step_units are not read (the settings list them
//     as ignored where given), and bh_step_end is not in the settings. Where
//     the move needs the adapted step as its scale, perturb = user (SRCH-19)
//     hands it to the problem's perturb hook as the strength; perturb = user
//     beside step = user or portfolio is refused (two moves).
//   MOVE-2. Off by default. The draws: the user's hook and a portfolio's
//     operator draw from the strategy's stream (rnd:: only), in the place of
//     the box move's n uniforms; the portfolio's choice is one more uniform
//     before them (none with one operator). The strategy's other draws (the
//     walk's u, pbh's and cluster_ga's extra uniform, PBH-4, CGA-3) stay
//     where they were, so step = portfolio with one operator and op_select =
//     fixed is step = user with that operator, bit for bit.
//   MOVE-3. Off by default. The operators' calls count the children made by
//     a move, as they are made, whether or not their outcome comes back (a
//     generation cut by the budget's end: its children were made, their
//     calls counted, as cluster_ga's crossings; so the calls may exceed the
//     children told by up to a generation). cluster_ga's around_x0 moves of
//     the first population are not children; ga's second child of a pair
//     dropped by GA-3 is not one.
//   MOVE-4. Off by default (Problem::on_child given). The tag travels with
//     the child in the strategy's record of its children (Child: its
//     operators and its parents' f), made where the child is made, never
//     looked up by the child's x. The library calls on_child(parents_f,
//     child_f, tag) once for every child whose outcome came back, on the
//     run's thread, after the strategy has decided its fate: basin_hopping
//     and ils after the acceptance (parents_f the current minimum's f,
//     taken: the walk moved to it), pbh and cluster_ga after the
//     generation's placement, child by child in their order (taken: it
//     entered the population; cluster_ga's parents_f the two parents' f,
//     one where it has one member; multi_dynamic and afpo: it is in the
//     next population; afpo's newcomer is not a child), ga after tell
//     (taken: it survived; child_f its f, there is no local search). A
//     child whose local search did not come back (the budget's end) has no
//     outcome and no call. The rounds' threads only search, so 1 thread
//     and N give the same calls in the same order; the calls add up to the
//     children with an outcome. An exception of the hook ends the run with
//     it.
//   MOVE-5. Off by default. A portfolio's reward is 1 where the child was
//     taken (MOVE-4's taken), else 0, its gain the better parent's f less
//     the child's; it is rewarded when on_child would be called, before
//     the call (cluster_ga: TH-2, a generation's children with the P of the
//     generation before; basin_hopping and ils: step by step; pbh: after
//     the generation's placement; ga: after tell). The settings' operators
//     row as CGA-7's: [name, kind, calls, taken, mean gain, P at the end].
//   MOVE-6. The crossover hook's children (Problem::crossover and each of
//     crossover_ops): every strategy hands c1 and c2 of n values each, zeros
//     (cluster_ga; ga_sbx and ssga with ga_crossover = user or ops), and the
//     hook writes all n of each (it may index them, or assign n values;
//     it must not append). Before (the owner's choice of 18.11) ga's user
//     crossover (GA-9) got the vectors as they were — empty at the first
//     pair, the last pair's children after — so a hook that indexed them
//     wrote past the end; a hook that wrote them whole runs as before, bit
//     for bit (an appending hook was refused by GA-10 from the second pair
//     on, and is now at the first).
// ============================================================================

#include <cstddef>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "evaluator.hpp"
#include "portfolio.hpp"

namespace mootation::scalar::moves {

using Vec = std::vector<double>;

enum class Kind { box, perturb, user, portfolio };

// a moved point back into the box: clipped, a periodic coordinate wrapped (SRCH-4, PER-2)
inline void into_box(const Problem& P, Vec& y)
{
    for (std::size_t j = 0; j < y.size(); ++j)
        y[j] = periodic::is(P.periodic, j) ? periodic::wrap_into(y[j], P.lo[j], P.hi[j])
                                           : (std::min)(P.hi[j], (std::max)(P.lo[j], y[j]));
}

// The move of a strategy: its kind, the box move's units, the portfolio of
// the problem's mutate_ops
struct Move {
    Kind kind = Kind::box;
    bool absolute = false;       // step_units = absolute (SRCH-15)
    Portfolio ops;               // kind portfolio

    bool scaled() const { return kind == Kind::box || kind == Kind::perturb; }

    // y moved in place and put back into the box (MOVE-2); its tag: the
    // portfolio's index, 0 the mutate hook, -1 the box move or perturb
    long long apply(const Problem& P, Vec& y, double step, std::mt19937& g, const std::string& who) const
    {
        const std::size_t n = y.size();
        long long tag = -1;
        if (kind == Kind::box) {
            for (std::size_t j = 0; j < n; ++j)
                y[j] += (2.0 * rnd::uniform01(g) - 1.0) * step * (absolute ? 1.0 : P.hi[j] - P.lo[j]);
        } else if (kind == Kind::perturb) {
            P.perturb(y, step, g);
            if (y.size() != n) throw std::runtime_error(who + ": the perturb hook changed the point's length");
        } else if (kind == Kind::user) {
            P.mutate(y, g);
            if (y.size() != n) throw std::runtime_error(who + ": the mutate hook changed the point's length");
            tag = 0;
        } else {
            const std::size_t o = ops.pick(g);
            P.mutate_ops[o].second(y, g);
            if (y.size() != n) throw std::runtime_error(who + ": a mutate operator changed the point's length");
            tag = static_cast<long long>(o);
        }
        into_box(P, y);
        return tag;
    }
};

// The child's outcome on the problem's on_child (portfolio.hpp told)
inline void told(const Problem& P, Portfolio* xport, Portfolio* mport, const Child& c, double f, bool taken,
                 std::mt19937& g)
{
    told(P.on_child, xport, mport, c, f, taken, g);
}

// op_select and its alpha, beta (portfolio.hpp), read in cluster_ga's order of task 30
template <class Knobs_>
int rule_of(Knobs_& k, const std::string& who, double& alpha, double& beta)
{
    const std::string sel =
        k.given("op_select") ? k.word("op_select", "fixed", {"fixed", "probability_matching", "adaptive_pursuit"}) : "fixed";
    const int rule = sel == "fixed" ? 0 : sel == "probability_matching" ? 1 : 2;
    alpha = rule ? k.number("op_alpha", 0.8) : 0.8;
    beta = rule == 2 ? k.number("op_beta", 0.8) : 0.8;
    if (!(alpha > 0.0 && alpha <= 1.0) || !(beta > 0.0 && beta <= 1.0))
        throw std::invalid_argument(who + ": op_alpha and op_beta in (0, 1]");
    return rule;
}

// op_pmin for K operators: Thierens's 1/(2K) by default, in (0, 1/K)
template <class Knobs_>
double pmin_of(Knobs_& k, const std::string& who, int rule, std::size_t K)
{
    const double p = rule && K > 1 ? k.number("op_pmin", 0.5 / static_cast<double>(K)) : 0.0;
    if (rule && K > 1 && !(p > 0.0 && p < 1.0 / static_cast<double>(K)))
        throw std::invalid_argument(who + ": op_pmin in (0, 1/K) for K operators (Thierens, footnote 2)");
    return p;
}

template <class Ops>
std::vector<std::string> names_of(const Ops& ops)
{
    std::vector<std::string> v;
    for (const auto& o : ops) v.push_back(o.first);
    return v;
}

// step = box | user | portfolio of basin_hopping, ils and pbh, where given
// (MOVE-1): the move's kind, its portfolio started; perturb = user (task 25)
// is the scaled user move
template <class Knobs_>
Move step_of(Knobs_& k, const Problem& P, const std::string& who, bool perturb_user)
{
    Move m;
    const std::string s = k.given("step") ? k.word("step", "box", {"box", "user", "portfolio"}) : "box";
    if (perturb_user && s != "box") throw std::invalid_argument(who + ": perturb = user and step = " + s + " are two moves");
    if (s == "user" && !P.mutate) throw std::invalid_argument(who + ": step = user needs the problem's mutate hook");
    if (s == "portfolio" && P.mutate_ops.empty())
        throw std::invalid_argument(who + ": step = portfolio needs the problem's mutate_ops");
    m.kind = perturb_user ? Kind::perturb : s == "user" ? Kind::user : s == "portfolio" ? Kind::portfolio : Kind::box;
    if (m.kind == Kind::portfolio) {
        double alpha = 0.8, beta = 0.8;
        const int rule = rule_of(k, who, alpha, beta);
        m.ops.start(names_of(P.mutate_ops), rule, pmin_of(k, who, rule, P.mutate_ops.size()), alpha, beta);
    }
    return m;
}

}   // namespace mootation::scalar::moves
