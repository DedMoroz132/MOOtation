#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// portfolio_run: a portfolio of DIFFERENT resumable runs (task 32, layer 1).
// The members are resumable runs (resumable.hpp) of any resumable strategies,
// each with its knobs, seed and start points; they share one budget, and the
// next piece of it goes to the member chosen by the portfolio's rule, which
// learns from what the pieces gave: the improvement of the best per unit of
// the budget spent (used(), stop_rule.hpp STOP-1). Included at the end of
// strategy.hpp, after ensemble.hpp. The ensemble (ensemble.hpp) holds K copies
// of one strategy; the operator portfolio (portfolio.hpp) chooses operators
// within a run; this one chooses between runs.
//
// THE MEMBERS. members = a JSON list of solvers, {"solver": one of
// resumable_names(), its knobs}, one a member. Member m has
//   its seed     member_rng[m] (a JSON list of K whole numbers or nulls),
//                else start_seed(seed, m) (strategy.hpp STRAT-1: member 0 has
//                the run's seed);
//   its start    member_seeds[m] (a JSON list of K lists of points, or
//                nulls; as ensemble's): x0 the first point, seed points the
//                others (resumable.hpp carry_start); without them the run's
//                start;
//   its share    split_budget(budget, K) (resumable.hpp): the members' shares
//                add up to the run's budget.
//
// THE RULE (op_select):
//   fixed                 (the default) every member runs within its share,
//                         the pieces drawn with P = 1/K among the members still
//                         running (portfolio.hpp's fixed rule; Thierens 2005,
//                         §4);
//   probability_matching, adaptive_pursuit
//                         the budget is one pool: a member may spend past its
//                         share, as far as the pool goes; each piece's member
//                         drawn by P, and P learns from the pieces' rewards by
//                         portfolio.hpp's rules (Thierens 2005, §2 (1)-(2), §3
//                         (1), (3)-(5); op_alpha, op_beta, op_pmin, the
//                         defaults there);
//   successive_halving    K. Jamieson, A. Talwalkar, AISTATS 2016 (source
//                         successive-halving_jamieson2016), §2, Figure 3
//                         (Successive Halving, analysed in §3.1), the arms
//                         the members: in round k each member left gets
//                         r_k = ⌊B / (|S_k| ⌈log2 K⌉)⌋ more of the budget (B
//                         the run's budget), then the better half stays (PF-6).
// A piece (the portfolio rules) is piece_budget of the budget's units or
// piece_steps of the member's steps (resumable.hpp RES-2: a generation, a step
// of the walk); neither given: one step. round_pieces (1) pieces are drawn a
// round, from the same P; a member drawn c times in a round has one piece of c
// pieces; the round's members advance in `threads` threads.
//
// THE REWARD (PF-2). A piece's outcome is the improvement of the member's own
// best (reward = own, the default) or of the best of all members (reward =
// all), and the budget it spent: reward = improvement / spent. The settings
// keep every round's outcomes.
//
// THE EXCHANGE (share_best = off | on; off by default). With on, before its
// piece every member whose best is worse than the best of all members is
// handed that point (admit, Place::worst_if_better: it takes the point only in
// place of a worse member; resumable.hpp RES-4, the ensemble's
// migrant_place = worst_if_better). The point's values are the giver's (RES-3:
// planted, not counted).
//
// THE MEMBER'S END is its own (its stall, max_steps, target, rule; the
// member's knobs). leftover = keep | share (the owner's decision of 20.11):
// keep (the default) leaves the rest of its share unspent, as ensemble's
// ENS-1; share hands it to the members still running.
//
// Knobs: members, member_seeds, member_rng, op_select (fixed |
// probability_matching | adaptive_pursuit | successive_halving), op_pmin,
// op_alpha, op_beta, piece_budget, piece_steps, round_pieces (1), reward (own
// | all), share_best (off | on), leftover (keep | share), threads (1); stall,
// max_steps, budget_units (stop_rule.hpp; the step is a round; budget_units
// goes to every member that does not give its own).
//
// READINGS:
//   PF-1 (only when asked: the portfolio_run strategy). The budget: with
//     op_select = fixed each member's run has its share as its budget from
//     the start, so that K members with share_best = off are the K
//     independent runs with the budget split evenly (RES-1: a run cut into
//     pieces is the run in one piece; the order of the pieces does not
//     matter). With the other rules the budget is a pool: before its piece a
//     member's budget is what it has spent and what is left of the pool
//     (Resumable::set_budget, RES-5; budget units rounded up a member,
//     Evaluator::left), so it can spend past its share and the members
//     together never past the run's budget; with round_pieces > 1 what is left
//     is split among the round's pieces (split_budget), a member's part the
//     sum of its pieces'. A member whose part ran out has ended (its run does
//     not take a step past its budget), as at the end of the pool.
//   PF-2 (only when asked: the portfolio_run strategy). The improvement from
//     the best before the piece (after share_best's point) to the best after
//     it: the violation's decrease where the point before violates, else the
//     decrease of f; none (not rewarded) where there was no point before (the
//     member's first piece, reward = own; the run's first, reward = all).
//     reward = all measures from the best of all members before the round to
//     the better of it and the member's best after. The reward is the
//     improvement over the budget's units the member spent in the piece
//     (Evaluator::counted), 0 where it spent nothing; one reward a member a
//     round, in the members' order.
//   PF-3 (only when asked: the portfolio_run strategy). The draw (ProportionalSelectOperator,
//     Thierens 2005) is among the members still running: P of each over the
//     sum of theirs, one uniform of the run's stream, mt19937(start_seed(seed,
//     0x9F01)); none where one member runs. The rules update P of every
//     member (an ended member's too: Thierens's update is over all
//     operators); adaptive pursuit's ties drawn from the same stream (TH-4).
//   PF-4 (only when asked: the portfolio_run strategy). A member's end and
//     its rest: a member ended by its own rule (stall, max_steps, rule, idle;
//     not by its budget or the target) leaves rest = its share - what it
//     spent (none below 0). leftover = keep: the rest is not spent by anyone
//     (with the pool rules the pool is that much smaller); leftover = share:
//     with op_select = fixed the rest is split among the members still
//     running (split_budget, their budgets raised), with the pool rules it
//     simply stays in the pool.
//   PF-5 (only when asked: the portfolio_run strategy). The end: the stop
//     rule before a round (budget: what is left of the pool after the rests
//     kept; deadline; stall: rounds without a better best of all; max_steps:
//     rounds); a member's target ends the run after its round; every member
//     ended; 100 rounds in a row that spend nothing (idle, SRCH-10). stopped_by
//     as ensemble's ENS-6. The answer: the best of the members (less
//     violating, then lower f; the first of equals), its points, improvements
//     and settings rows; the counts every member's; population every
//     member's, in their order (ENS-5).
//   PF-6 (only when asked: the portfolio_run strategy). successive_halving:
//     Jamieson and Talwalkar's §2, Figure 3: rounds k = 0 ... ⌈log2 K⌉ - 1
//     over the members as arms, a
//     pull one unit of the budget: a member of S_k gets a piece of r_k units
//     (at least 1), its budget the pool's (PF-1), so a step that crosses r_k
//     finishes (RES-1) and the next members find the pool that much smaller.
//     The members are ranked by their best (less violating, then lower f;
//     the earlier member first of equals) and the better half rounded up
//     stays (as halving's SRCH-6), so ⌈log2 K⌉ rounds leave one. Figure 3
//     ends there; here the last member then runs on to the end of the pool
//     (what the floors of r_k and the members ended early left). A member
//     ended early is ranked by its best and gets no piece.
//   PF-7 (only when asked: the portfolio_run strategy). threads > 1 advances
//     a round's members in that many threads (ensemble.hpp advance_members,
//     ENS-7: the deferred count replayed member by member), members whose
//     pieces are alike together, in the members' order; the round's plan
//     (draws, budgets, points handed) is made before it and the rewards after
//     it, so 1 thread and N give the same run.
//   PF-8 (only when asked: the portfolio_run strategy). The observer and
//     record_every go to the member only where there is one (ENS-8).
//   PF-9 (only when asked: the portfolio_run strategy). One member is the
//     member's run, bit for bit, with every rule: its budget is the run's,
//     the draws need no uniform, and its pieces are the run in one piece
//     (RES-1).
// ============================================================================

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "ensemble.hpp"
#include "portfolio.hpp"
#include "resumable.hpp"
#include "strategy.hpp"

namespace mootation::scalar {

namespace pf {

enum class Rule { fixed, matching, pursuit, halving };

// a member's run: what start_resumable gets but its budget
struct Spec {
    const Problem* problem = nullptr;
    std::string name;                     // its strategy (the portfolio's operator's name)
    Knobs knobs;
    unsigned seed = 0;
    Start start;
};

// the portfolio's run
struct Plan {
    std::vector<Spec> members;
    Rule rule = Rule::fixed;
    double pmin = 0.0, alpha = 0.8, beta = 0.8;
    Piece piece;                          // a draw's piece (the portfolio rules)
    long long round_pieces = 1;
    bool reward_all = false;              // reward = all
    bool share_best = false;
    bool leftover_share = false;          // leftover = share
    unsigned stream = 0;                  // the draws (PF-3)
    int threads = 1;
};

// what the run did
struct Outcome {
    std::vector<Result> results;          // a member without budget: empty
    std::vector<char> ran;
    std::vector<long long> shares;        // split_budget(budget, K)
    std::vector<long long> pieces;        // the pieces each member had
    std::vector<double> P, Q;             // the portfolio's at the end
    long long rounds = 0;
    long long offered = 0, accepted = 0;  // share_best
    long long withheld = 0;               // the rests kept (PF-4)
    json::Value history = json::Value::array();   // [round, member, pieces, spent, improvement, reward]
    json::Value halving = json::Value::array();   // S_k of each round (PF-6)
    std::string why;                      // the rule's end, target, deadline or idle; empty: the members'
};

// a member's resumable run from its spec, with the budget it starts with
using Make = std::function<std::unique_ptr<Resumable>(const Spec& spec, long long budget, std::size_t m)>;

// the improvement from a to b (PF-2); none (-1) where a has no point
inline double improvement(const Member& a, const Member& b)
{
    if (a.x.empty()) return -1.0;
    if (b.x.empty()) return 0.0;
    const double d = a.cv > 0.0 ? a.cv - b.cv : a.f - b.f;
    return std::isfinite(d) ? (std::max)(0.0, d) : 0.0;
}

// ProportionalSelectOperator(P) among the members still running (PF-3)
inline std::size_t draw(const std::vector<double>& P, const std::vector<std::size_t>& live, std::mt19937& g)
{
    if (live.size() == 1) return live[0];
    double sum = 0.0;
    for (std::size_t m : live) sum += P[m];
    const double u = rnd::uniform01(g) * sum;
    double s = 0.0;
    for (std::size_t q = 0; q + 1 < live.size(); ++q) {
        s += P[live[q]];
        if (u < s) return live[q];
    }
    return live.back();
}

// ⌈log2 n⌉ for n >= 1
inline long long ceil_log2(std::size_t n)
{
    long long r = 0;
    std::size_t p = 1;
    while (p < n) {
        p *= 2;
        ++r;
    }
    return r;
}

// THE PORTFOLIO's run: the members made by `make`, the rounds; the members' results at the end
inline Outcome run(const Problem& P, long long budget, const Plan& plan, strategy::StopRule& stop, const Make& make)
{
    const std::size_t K = plan.members.size();
    const bool pooled = plan.rule != Rule::fixed;                                      // PF-1
    Outcome out;
    out.results.resize(K);
    out.ran.assign(K, 0);
    out.pieces.assign(K, 0);
    out.shares = split_budget(budget, K);
    std::vector<long long> cap = out.shares;                                          // fixed: the members' budgets
    std::vector<std::unique_ptr<Resumable>> R(K);
    for (std::size_t m = 0; m < K; ++m) {
        const long long b = pooled ? budget : cap[m];
        if (b > 0) R[m] = make(plan.members[m], b, m);
    }
    moves::Portfolio port;
    {
        std::vector<std::string> names;
        for (const Spec& s : plan.members) names.push_back(s.name);
        const int r = plan.rule == Rule::matching ? 1 : plan.rule == Rule::pursuit ? 2 : 0;
        port.start(names, r, plan.pmin, plan.alpha, plan.beta);
    }
    std::mt19937 rng(plan.stream);
    std::vector<char> settled(K, 0);                  // a member's end seen (PF-4)
    auto used = [&](std::size_t m) {
        return R[m] ? static_cast<long long>(std::ceil(R[m]->ledger().counted())) : 0LL;
    };
    auto pool = [&]() {
        long long s = budget - out.withheld;
        for (std::size_t m = 0; m < K; ++m) s -= used(m);
        return s;
    };
    auto live = [&](std::size_t m) { return R[m] && !R[m]->done(); };
    auto best_of_all = [&]() {
        Member b;
        for (std::size_t m = 0; m < K; ++m)
            if (R[m]) {
                const Member x = R[m]->best();
                if (ens::better(x, b)) b = x;
            }
        return b;
    };
    // a round: member L[q] advanced by a piece of c[q] pieces (piece[q]); returns whether a member hit the target
    auto play = [&](const std::vector<std::size_t>& L, const std::vector<long long>& c, const std::vector<Piece>& piece) {
        if (plan.share_best) {                                                       // THE EXCHANGE
            const Member g = best_of_all();
            for (std::size_t m : L)
                if (ens::better(g, R[m]->best())) {
                    ++out.offered;
                    out.accepted += R[m]->admit({g}, Place::worst_if_better);
                }
        }
        if (pooled) {                                                                 // PF-1
            long long n = 0;
            for (long long x : c) n += x;
            const std::vector<long long> parts = split_budget((std::max)(0LL, pool()), static_cast<std::size_t>(n));
            std::size_t at = 0;
            for (std::size_t q = 0; q < L.size(); ++q) {
                long long part = 0;
                for (long long i = 0; i < c[q]; ++i) part += parts[at++];
                const long long b = used(L[q]) + part;
                if (R[L[q]]->ledger().budget != b) R[L[q]]->set_budget(b);
            }
        }
        std::vector<Member> own(L.size());
        std::vector<double> from(L.size());
        for (std::size_t q = 0; q < L.size(); ++q) {
            own[q] = R[L[q]]->best();
            from[q] = R[L[q]]->ledger().counted();
        }
        const Member all = best_of_all();
        std::vector<char> done_(L.size(), 0);                                         // PF-7: alike pieces together
        for (std::size_t q = 0; q < L.size(); ++q) {
            if (done_[q]) continue;
            std::vector<std::size_t> same;
            for (std::size_t r = q; r < L.size(); ++r)
                if (!done_[r] && piece[r].spend == piece[q].spend && piece[r].steps == piece[q].steps) {
                    same.push_back(L[r]);
                    done_[r] = 1;
                }
            ens::advance_members(R, same, piece[q], plan.threads);
        }
        bool target = false;
        for (std::size_t q = 0; q < L.size(); ++q) {                                   // PF-2: the outcomes
            const std::size_t m = L[q];
            const Member now = R[m]->best();
            const double spent = R[m]->ledger().counted() - from[q];
            const double d = plan.reward_all ? improvement(all, ens::better(now, all) ? now : all) : improvement(own[q], now);
            const double reward = d < 0.0 ? 0.0 : spent > 0.0 ? d / spent : 0.0;
            if (d >= 0.0 && (plan.rule == Rule::matching || plan.rule == Rule::pursuit)) port.reward(m, reward, rng);
            out.pieces[m] += c[q];
            json::Value row = json::Value::array();
            row.items.push_back(json::Value::integer(out.rounds));
            row.items.push_back(json::Value::integer(static_cast<long long>(m)));
            row.items.push_back(json::Value::integer(c[q]));
            row.items.push_back(json::Value::number(spent));
            row.items.push_back(d < 0.0 ? json::Value::null() : json::Value::number(d));
            row.items.push_back(d < 0.0 ? json::Value::null() : json::Value::number(reward));
            out.history.items.push_back(row);
            target = target || R[m]->ledger().target_reached;
        }
        for (std::size_t m : L) {                                                     // PF-4: a member's end
            if (!R[m]->done() || settled[m]) continue;
            settled[m] = 1;
            const Evaluator& ev = R[m]->ledger();
            if (ev.target_reached || ev.left() <= 0) continue;                         // its budget's end: no rest
            const long long rest = (std::max)(0LL, (pooled ? out.shares[m] : cap[m]) - used(m));
            if (rest == 0) continue;
            if (!plan.leftover_share) {
                out.withheld += rest;
                continue;
            }
            if (pooled) continue;                                                     // it stays in the pool
            std::vector<std::size_t> on;
            for (std::size_t r = 0; r < K; ++r)
                if (live(r)) on.push_back(r);
            const std::vector<long long> parts = split_budget(rest, on.size());
            for (std::size_t q = 0; q < on.size(); ++q) {
                cap[on[q]] += parts[q];
                R[on[q]]->set_budget(cap[on[q]]);
            }
        }
        ++out.rounds;
        return target;
    };
    auto counted = [&]() {
        double s = 0.0;
        for (const auto& r : R)
            if (r) s += r->ledger().counted();
        return s;
    };
    Member so_far;
    long long idle = 0;
    // after a round: the stop rule's step, the target, idle; whether the run goes on
    auto after = [&](bool target, double before) {
        const Member now = best_of_all();
        const bool improved = ens::better(now, so_far);
        stop.step(improved);
        if (improved) so_far = now;
        if (target) {
            out.why = "target";
            return false;
        }
        idle = counted() == before ? idle + 1 : 0;
        if (idle >= 100) {
            out.why = "idle";
            return false;
        }
        return true;
    };
    try {
        if (plan.rule == Rule::halving) {                                            // PF-6
            const long long rounds = ceil_log2(K);
            std::vector<std::size_t> S;
            for (std::size_t m = 0; m < K; ++m)
                if (R[m]) S.push_back(m);
            bool on = true;
            for (long long k = 0; k < rounds && on && S.size() > 1; ++k) {
                if (!stop.go(pool(), P)) {
                    on = false;
                    break;
                }
                json::Value s = json::Value::array();
                for (std::size_t m : S) s.items.push_back(json::Value::integer(static_cast<long long>(m)));
                out.halving.items.push_back(s);
                const long long r_k = (std::max)(1LL, budget / (static_cast<long long>(S.size()) * rounds));
                std::vector<std::size_t> L;
                for (std::size_t m : S)
                    if (live(m)) L.push_back(m);
                const double before = counted();
                bool target = false;
                if (!L.empty()) {
                    Piece p;
                    p.spend = static_cast<double>(r_k);
                    target = play(L, std::vector<long long>(L.size(), 1), std::vector<Piece>(L.size(), p));
                }
                on = after(target, before);
                std::vector<Member> b(S.size());
                for (std::size_t q = 0; q < S.size(); ++q) b[q] = R[S[q]]->best();
                std::vector<std::size_t> order(S.size());
                for (std::size_t q = 0; q < order.size(); ++q) order[q] = q;
                std::stable_sort(order.begin(), order.end(), [&](std::size_t x, std::size_t y) { return ens::better(b[x], b[y]); });
                std::vector<std::size_t> kept;
                for (std::size_t q = 0; q < (S.size() + 1) / 2; ++q) kept.push_back(S[order[q]]);
                std::sort(kept.begin(), kept.end());
                S = kept;
            }
            if (on && !S.empty() && live(S[0]) && stop.go(pool(), P)) {              // the last member to the end
                json::Value s = json::Value::array();
                s.items.push_back(json::Value::integer(static_cast<long long>(S[0])));
                out.halving.items.push_back(s);
                const double before = counted();
                after(play({S[0]}, {1}, {Piece()}), before);
            }
        } else {
            while (stop.go(pool(), P)) {
                std::vector<std::size_t> on;
                for (std::size_t m = 0; m < K; ++m)
                    if (live(m)) on.push_back(m);
                if (on.empty()) break;
                std::vector<long long> n(K, 0);
                for (long long q = 0; q < plan.round_pieces; ++q) ++n[draw(port.P, on, rng)];
                std::vector<std::size_t> L;
                std::vector<long long> c;
                std::vector<Piece> piece;
                for (std::size_t m = 0; m < K; ++m) {
                    if (n[m] == 0) continue;
                    L.push_back(m);
                    c.push_back(n[m]);
                    Piece p;
                    p.spend = plan.piece.spend * static_cast<double>(n[m]);
                    p.steps = plan.piece.steps * n[m];
                    piece.push_back(p);
                }
                const double before = counted();
                if (!after(play(L, c, piece), before)) break;
                bool any = false;
                for (std::size_t m = 0; m < K; ++m) any = any || live(m);
                if (!any) break;
            }
        }
    } catch (const BudgetSpent&) {                    // a point handed over and evaluated (RES-3) past the end
        out.why = past_deadline(P) ? "deadline" : "budget";
    }
    if (out.why.empty()) out.why = stop.stopped_by("");
    out.P = port.P;
    out.Q = port.Q;
    for (std::size_t m = 0; m < K; ++m)
        if (R[m]) {
            out.results[m] = R[m]->result();
            out.ran[m] = 1;
        }
    return out;
}

// a JSON list of the knob `name`, one entry a member; Null where not given
inline json::Value list_knob(strategy::StrategyKnobs& k, const std::string& name)
{
    const Knob* v = k.find(name);
    if (v == nullptr) return json::Value::null();
    if (!v->is_word) throw std::invalid_argument("portfolio_run: " + name + " must be a JSON list, one entry a member");
    json::Value o = json::parse(v->word);
    if (o.kind != json::Value::Kind::Array || o.items.empty())
        throw std::invalid_argument("portfolio_run: " + name + " must be a non-empty JSON list, one entry a member");
    k.settings.push_back({name, v->word});
    return o;
}

}   // namespace pf

inline Result run_portfolio_of_runs(const Problem& P, double eq_tol, bool has_tol0, double eq_tol0, int pop_size,
                                    long long budget, unsigned seed, const Knobs& knobs, Observer observer,
                                    int record_every, const Start& start)
{
    using namespace strategy;
    StrategyKnobs k("portfolio_run", knobs,
                    {"members", "member_seeds", "member_rng", "op_select", "op_pmin", "op_alpha", "op_beta",
                     "piece_budget", "piece_steps", "round_pieces", "reward", "share_best", "leftover", "threads"});
    const json::Value list = pf::list_knob(k, "members");
    if (list.kind == json::Value::Kind::Null)
        throw std::invalid_argument("portfolio_run needs its members: members = [{\"solver\": ...}, ...]");
    const std::size_t K = list.items.size();
    std::vector<Inner> inner;
    for (std::size_t m = 0; m < K; ++m) {
        const std::string at = "portfolio_run: member " + std::to_string(m);
        if (list.items[m].kind != json::Value::Kind::Object)
            throw std::invalid_argument(at + " must be a solver: {\"solver\": name, its knobs ...}");
        Knob w;
        w.is_word = true;
        w.word = json::dump(list.items[m], -1, ",", ":");
        inner.push_back(inner_of(at, w));
        if (!detail::listed(resumable_names(), inner.back().name))
            throw std::invalid_argument(at + " must be a resumable strategy (resumable_names()), not " +
                                        inner.back().name);
    }
    const json::Value seeds = pf::list_knob(k, "member_seeds");
    const json::Value rngs = pf::list_knob(k, "member_rng");
    for (const json::Value* l : {&seeds, &rngs})
        if (l->kind != json::Value::Kind::Null && l->items.size() != K)
            throw std::invalid_argument("portfolio_run: member_seeds and member_rng need one entry a member (" +
                                        std::to_string(K) + ")");
    pf::Plan plan;
    const std::string sel = k.given("op_select") ? k.word("op_select", "fixed",
                                                          {"fixed", "probability_matching", "adaptive_pursuit",
                                                           "successive_halving"})
                                                 : "fixed";
    plan.rule = sel == "probability_matching" ? pf::Rule::matching
                : sel == "adaptive_pursuit"   ? pf::Rule::pursuit
                : sel == "successive_halving" ? pf::Rule::halving
                                              : pf::Rule::fixed;
    if (plan.rule == pf::Rule::matching || plan.rule == pf::Rule::pursuit) {        // portfolio.hpp's knobs
        plan.alpha = k.number("op_alpha", 0.8);
        plan.beta = plan.rule == pf::Rule::pursuit ? k.number("op_beta", 0.8) : 0.8;
        if (!(plan.alpha > 0.0 && plan.alpha <= 1.0) || !(plan.beta > 0.0 && plan.beta <= 1.0))
            throw std::invalid_argument("portfolio_run: op_alpha and op_beta in (0, 1]");
        plan.pmin = K > 1 ? k.number("op_pmin", 0.5 / static_cast<double>(K)) : 0.0;
        if (K > 1 && !(plan.pmin > 0.0 && plan.pmin < 1.0 / static_cast<double>(K)))
            throw std::invalid_argument("portfolio_run: op_pmin in (0, 1/K) for K members (Thierens, footnote 2)");
    }
    if (plan.rule == pf::Rule::halving) {
        if (k.given("piece_budget") || k.given("piece_steps") || k.given("round_pieces"))
            throw std::invalid_argument("portfolio_run: successive_halving's pieces are its rounds' r_k (PF-6): "
                                        "piece_budget, piece_steps and round_pieces are not read");
    } else {
        if (k.given("piece_budget")) {
            plan.piece.spend = k.number("piece_budget", 0.0);
            if (!(plan.piece.spend >= 0.0)) throw std::invalid_argument("portfolio_run: piece_budget must be >= 0");
        }
        if (k.given("piece_steps")) plan.piece.steps = k.whole("piece_steps", 0, 0);
        if (!(plan.piece.spend > 0.0) && plan.piece.steps <= 0) {
            plan.piece.steps = 1;
            k.settings.push_back({"piece_steps", "1"});
        }
        plan.round_pieces = k.whole("round_pieces", 1, 1);
    }
    plan.reward_all = k.word("reward", "own", {"own", "all"}) == "all";
    plan.share_best = k.word("share_best", "off", {"off", "on"}) == "on";
    plan.leftover_share = k.word("leftover", "keep", {"keep", "share"}) == "share";
    plan.threads = static_cast<int>(k.whole("threads", 1, 1));
    plan.stream = start_seed(seed, 0x9F01u);                                          // PF-3
    StopRule stop = stop_of(k);                                                       // a step a round
    const Knob* units = k.find("budget_units");
    plan.members.resize(K);
    for (std::size_t m = 0; m < K; ++m) {
        pf::Spec& s = plan.members[m];
        const std::string at = "portfolio_run: member " + std::to_string(m);
        s.problem = &P;
        s.name = inner[m].name;
        s.knobs = inner[m].knobs;
        if (units) s.knobs = ens::over({{"budget_units", *units}}, s.knobs);         // the member's own first
        s.seed = start_seed(seed, static_cast<std::uint64_t>(m));
        if (rngs.kind != json::Value::Kind::Null && rngs.items[m].kind != json::Value::Kind::Null) {
            const json::Value& v = rngs.items[m];
            if (v.kind != json::Value::Kind::Int || v.i < 0 || v.i > 0xFFFFFFFFLL)
                throw std::invalid_argument(at + "'s member_rng must be a whole number in [0, 2^32), or null");
            s.seed = static_cast<unsigned>(v.i);
        }
        s.start = start;
        if (seeds.kind != json::Value::Kind::Null && seeds.items[m].kind != json::Value::Kind::Null) {
            const json::Value& l = seeds.items[m];
            if (l.kind != json::Value::Kind::Array)
                throw std::invalid_argument(at + "'s member_seeds must be a list of points, or null");
            std::vector<Vec> pts;
            for (const json::Value& p : l.items) {
                if (p.kind != json::Value::Kind::Array || p.items.size() != P.lo.size())
                    throw std::invalid_argument(at + "'s member_seeds: a point is a list of n numbers");
                Vec x;
                for (const json::Value& v : p.items) x.push_back(ens::number_of(v, at + "'s member_seeds"));
                pts.push_back(x);
            }
            if (!pts.empty()) s.start = carry_start(pts);
        }
    }
    const pf::Make make = [&](const pf::Spec& s, long long b, std::size_t) {
        return start_resumable(s.name, *s.problem, eq_tol, has_tol0, eq_tol0, pop_size, b, s.seed, s.knobs,
                               K == 1 ? observer : Observer{}, K == 1 ? record_every : 0, s.start);   // PF-8
    };
    const pf::Outcome o = pf::run(P, budget, plan, stop, make);
    // PF-5: the answer
    std::size_t pick = K;
    for (std::size_t m = 0; m < K; ++m) {
        if (!o.ran[m]) continue;
        const Member a = ens::answer_of(o.results[m]);
        if (a.x.empty()) continue;
        if (pick == K || ens::better(a, ens::answer_of(o.results[pick]))) pick = m;
    }
    Result out = pick < K ? o.results[pick] : Result();
    out.evaluations = out.grad_calls = out.n_failed = out.hook_calls = out.success_calls = out.delta_calls = 0;
    out.cache_hits = out.decode_invalid = out.outside_evals = out.outside_grads = 0;
    out.hook_spent = out.hook_seconds = out.success_spent = out.delta_spent = 0.0;
    out.used = 0.0;
    out.population.clear();
    json::Value best = json::Value::array(), share = json::Value::array(), pieces = json::Value::array(),
                why_m = json::Value::array();
    double total = 0.0;
    for (std::size_t m = 0; m < K; ++m)
        if (o.ran[m]) total += o.results[m].used;
    for (std::size_t m = 0; m < K; ++m) {
        const Member a = o.ran[m] ? ens::answer_of(o.results[m]) : Member();
        best.items.push_back(a.x.empty() ? json::Value::null() : json::Value::number(a.f));
        share.items.push_back(json::Value::number(o.ran[m] && total > 0.0 ? o.results[m].used / total : 0.0));
        pieces.items.push_back(json::Value::integer(o.pieces[m]));
        why_m.items.push_back(o.ran[m] ? json::Value::string(o.results[m].stopped_by) : json::Value::null());
        if (!o.ran[m]) continue;
        const Result& r = o.results[m];
        out.evaluations += r.evaluations;
        out.grad_calls += r.grad_calls;
        out.n_failed += r.n_failed;
        out.hook_calls += r.hook_calls;
        out.hook_spent += r.hook_spent;
        out.hook_seconds += r.hook_seconds;
        out.success_calls += r.success_calls;
        out.success_spent += r.success_spent;
        out.delta_calls += r.delta_calls;
        out.delta_spent += r.delta_spent;
        out.cache_hits += r.cache_hits;
        out.decode_invalid += r.decode_invalid;
        out.outside_evals += r.outside_evals;
        out.outside_grads += r.outside_grads;
        out.used += r.used;
        out.population.insert(out.population.end(), r.population.begin(), r.population.end());
    }
    // PF-5: what ended the run (ensemble's ENS-6)
    std::string why = o.why;
    for (std::size_t m = 0; m < K; ++m)
        if (o.ran[m] && o.results[m].stopped_by == "target") why = "target";
    if (why.empty()) {
        for (std::size_t m = 0; m < K; ++m) {
            if (!o.ran[m]) continue;
            if (why.empty()) why = o.results[m].stopped_by;
            else if (why != o.results[m].stopped_by) why = "rule";
        }
        if (why.empty()) why = "rule";
    }
    out.stopped_by = why;
    out.ignored = k.ignored();
    auto list_of = [](const std::vector<double>& v) {
        json::Value a = json::Value::array();
        for (double x : v) a.items.push_back(json::Value::number(x));
        return json::dump(a, -1, ",", ":");
    };
    auto dump = [](const json::Value& v) { return json::dump(v, -1, ",", ":"); };
    out.settings.insert(out.settings.end(), k.settings.begin(), k.settings.end());
    out.settings.push_back({"pf_best", dump(best)});
    out.settings.push_back({"pf_answer", pick < K ? std::to_string(pick) : std::string("none")});
    out.settings.push_back({"pf_share", dump(share)});
    out.settings.push_back({"pf_pieces", dump(pieces)});
    out.settings.push_back({"pf_stopped_by", dump(why_m)});
    out.settings.push_back({"rounds", std::to_string(o.rounds)});
    if (plan.rule != pf::Rule::halving) out.settings.push_back({"pf_P", list_of(o.P)});
    if (plan.rule == pf::Rule::matching || plan.rule == pf::Rule::pursuit) out.settings.push_back({"pf_Q", list_of(o.Q)});
    if (plan.rule == pf::Rule::halving) out.settings.push_back({"pf_halving", dump(o.halving)});
    if (plan.share_best) {
        out.settings.push_back({"pf_shared_offered", std::to_string(o.offered)});
        out.settings.push_back({"pf_shared_accepted", std::to_string(o.accepted)});
    }
    if (o.withheld > 0) out.settings.push_back({"pf_withheld", std::to_string(o.withheld)});
    out.settings.push_back({"pf_rewards", dump(o.history)});
    return out;
}

}   // namespace mootation::scalar
