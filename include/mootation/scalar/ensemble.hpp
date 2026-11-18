#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// ensemble: K resumable runs in one run, with an exchange between them every
// exchange_every steps (task 31, layer 4; task 30's §30.8 and §30.16: the
// islands without restarts). Included at the end of strategy.hpp, after
// resumable_run.hpp.
//
// THE MEMBERS. Member m (0 ... K - 1) is the resumable run (resumable.hpp) of
// the inner strategy (inner = {"solver": one of resumable_names(), its
// knobs}) with
//   its problem  the run's, or with member_params (a JSON list of K numbers
//                or nulls) the level L_m of the problem's parameter
//                (param.hpp problem_at; the optional part of 30.8);
//   its knobs    inner's, with member_knobs[m] (a JSON list of K objects or
//                nulls) over them: a knob of the same name replaced, a new
//                one added;
//   its seed     start_seed(seed, m) (strategy.hpp STRAT-1: member 0 has the
//                run's seed);
//   its start    member_seeds[m] (a JSON list of K lists of points, or
//                nulls): x0 the first point, seed points the others
//                (resumable.hpp carry_start); without them the run's start;
//   its budget   split_budget(budget, K) (resumable.hpp): the members'
//                budgets add up to the run's.
// K is ens_K, else the length of the member lists given, else 1.
//
// THE ROUNDS. A round advances every member that has not ended by a piece of
// exchange_every of its own steps (resumable.hpp RES-2: a generation of
// cluster_ga or ga, a step of the walk), member by member in their order,
// or in `threads` threads; then the exchange. exchange_every = 0 (the
// default): one round, every member to its end, no exchange.
//
// THE EXCHANGE (exchange = migrate | swap):
//   migrate  the islands. Every member that has not ended receives migrants
//            from its sources by topology: ring (member m - 1), full (every
//            other member, m + 1 first, round the ring) or random (one other
//            member, drawn). From each source `migrants` (1) of its members
//            as they were before the exchange, chosen by migrant_pick: best
//            (the less violating, then the lower f), random (drawn without
//            repetition) or diverse (the farthest from the receiver's elite,
//            its best member, by the problem's distance, hooks.hpp
//            distance_of); placed by migrant_place: worst (each in place of
//            the receiver's worst, resumable.hpp RES-4) or worst_if_better.
//   swap     replica exchange (K. Hukushima, K. Nemoto, J. Phys. Soc. Jpn.
//            65 (1996) 1604, §2, (2.5)-(2.7); as replica.hpp): the neighbours
//            m, m + 1 in the order 0-1, 1-2, ... exchange their
//            configurations with probability min(1, exp(-swap_fs Delta)).
//            swap_over = knob: member m's temperature T_m is its value of the
//            knob swap_knob (e.g. basin_hopping's bh_temperature, given by
//            member_knobs), every member on the problem itself, Delta = (1 /
//            T_{m+1} - 1 / T_m)(E_m - E_{m+1}); swap_over = param: member m at
//            the level member_params[m], Delta = swap_beta [H_m(X_{m+1}) +
//            H_{m+1}(X_m) - H_m(X_m) - H_{m+1}(X_{m+1})], the carried
//            configurations evaluated (counted).
//
// THE BUDGET AND THE TIME are the run's (stop_rule.hpp; task 31, layer 1):
// the members' ledgers and the exchanges' evaluations add up to what the run
// spent, and never pass its budget; the members' problems keep the run's
// deadline and clock (evaluator.hpp derived). The stop rule's step is a round
// (stall: rounds without a better best over the members; max_steps: rounds).
//
// THE DEGENERATE CASES: one member without an exchange is the inner
// strategy's run, bit for bit (its seed, start, budget, knobs and observer);
// K members with exchange_every = 0, exchange = migrate with migrants = 0,
// or exchange = none are K independent runs with the members' seeds,
// starts, knobs and budgets (ENS-1).
//
// cluster_ga's islands (cga_islands; CGA-6) are not moved here: they share
// one archive (a point one island evaluated is free for the others) and one
// round of local searches, the islands' streams are their own (start_seed(
// seed, 0x9E3779B9 + i)) and the searches' seeds one count over the islands.
// The ensemble of cluster_ga members is the islands of 30.16 with an archive
// each; its numbers are not cga_islands', so cga_islands stays as it is, its
// fingerprints unchanged. replica_ladder takes the ensemble with rl_mode =
// resume (replica.hpp RL-5; off by default: rl_mode = pieces, the runs of
// before).
//
// Knobs: inner, ens_K, exchange (migrate | swap | none), exchange_every (0),
// topology (ring | full | random), migrants (1), migrant_pick (best | random
// | diverse), migrant_place (worst | worst_if_better), swap_over (knob |
// param), swap_knob, swap_beta (1), swap_fs (1), member_seeds, member_knobs,
// member_params, threads (1); stall, max_steps, budget_units (stop_rule.hpp;
// budget_units goes to every member that does not give its own).
//
// READINGS:
//   ENS-1 (only when asked: the ensemble strategy). The budget is split once
//     (split_budget) and a member keeps its share: one that ends early (its
//     target, its stall, its own rule) leaves the rest unspent, not handed
//     to the others, so that K members without an exchange are the K
//     independent runs (resumable.hpp RES-1: a run cut into pieces is the run
//     in one piece).
//   ENS-2 (only when asked: the ensemble strategy). An exchange's evaluations
//     (a migrant carried to another level, a configuration carried over a
//     parameter) are charged to the member whose problem evaluated them: its
//     budget is lowered by one each (Resumable::set_budget, RES-5), and only
//     where its budget has the room (a pair without it is not tried, a
//     migrant without it not sent), so the run never spends past its
//     budget. They are counted in the result's evaluations, as replica.hpp's
//     (RL-1), and checked against the deadline before each (TIME-2).
//   ENS-3 (only when asked: the ensemble strategy). A migrant between members
//     on one problem keeps the giver's values (RES-3: planted, not counted);
//     between levels of the parameter it is carried (param.hpp carried) and
//     evaluated at the receiver's level (ENS-2), a failed value not sent. A
//     member without member_params is at the problem's own level (its param);
//     where that is unknown (NaN) a migrant to or from it is only brought into
//     the receiver's box, not rescaled.
//     best ranks the giver's members by the giver's values; diverse measures
//     the carried points from the receiver's elite at the receiver's level.
//     The draws (random topology, random pick) come from one stream,
//     mt19937(start_seed(seed, 0xE45E)), receiver by receiver: one uniform
//     for a source, a partial Fisher-Yates draw of migrants (as AF-2).
//   ENS-4 (only when asked: the ensemble strategy). A configuration (swap) is
//     a member's members (a walk's point, a population), its energy E the f
//     of its best member; an accepted exchange hands each member the other's
//     (admit, Place::all). Over a parameter every member of X_{m+1} is
//     evaluated at level m and of X_m at level m + 1 (H of a carried
//     configuration: its best), m's first. The uniform is drawn only where
//     Delta > 0 (RL-3); a carried point whose value failed refuses the
//     exchange without a draw.
//   ENS-5 (only when asked: the ensemble strategy). The answer: the best
//     (less violating, then lower f; the first of equals) of the members on
//     the problem itself; with every member at a level of the parameter, the
//     last member's (the ladder ends at the level wanted, RL-4). Its points,
//     improvements, success_at and settings rows are the answer member's;
//     the counts are every member's and the exchanges' (evaluations, gradient
//     calls, hooks', used, outside); population is every member's, in their
//     order. The settings add ens_best (each member's best f), ens_answer,
//     rounds and the exchanges' counts (migrants_offered, migrants_accepted;
//     swaps_tried, swaps_accepted; exchange_evaluations).
//   ENS-6 (only when asked: the ensemble strategy). The end: the stop rule
//     before a round (budget, deadline, stall, max_steps); a member's target
//     ends the run after its round; every member ended; 100 rounds in a row
//     that spend nothing (idle, SRCH-10). stopped_by: target where a member
//     reached it, else the rule's end, else the members' where they agree,
//     else rule.
//   ENS-7 (only when asked: the ensemble strategy). threads > 1 advances the
//     members of a round in that many threads, each member's run in one
//     thread (a round within it runs in that thread, STRAT-7); what the
//     problem does besides evaluating is replayed member by member, in their
//     order (strategy.hpp THE DEFERRED COUNT), and the exchanges run after
//     the round, so 1 thread and N give the same run.
//   ENS-8 (only when asked: the ensemble strategy). The observer and
//     record_every go to the member only where there is one (PARAM-5's rule
//     for the levels of shrink): K runs would report K counts of their own.
// ============================================================================

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <exception>
#include <functional>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "hooks.hpp"
#include "param.hpp"
#include "resumable.hpp"
#include "strategy.hpp"

namespace mootation::scalar {

namespace ens {

enum class Exchange { none, migrate, swap };
enum class Topology { ring, full, random };
enum class Pick { best, random, diverse };

// a member's run: what start_resumable gets
struct Spec {
    const Problem* problem = nullptr;     // the run's problem, or own's (a level)
    std::shared_ptr<Problem> own;         // the level's problem (member_params)
    double level = std::numeric_limits<double>::quiet_NaN();   // NaN: the problem itself
    Knobs knobs;
    unsigned seed = 0;
    Start start;
    long long budget = 0;
    double temperature = 0.0;             // swap_over = knob: its value of swap_knob
};

// the ensemble's run
struct Plan {
    std::vector<Spec> members;
    Piece piece;                          // a round's piece; none: one round to the end
    Exchange exchange = Exchange::none;
    Topology topology = Topology::ring;
    long long migrants = 1;
    Pick pick = Pick::best;
    Place place = Place::worst;
    bool over_param = false;
    double beta = 1.0, fs = 1.0;
    unsigned stream = 0;                  // the exchanges' generator (ENS-3, RL-3)
    int threads = 1;
};

// what the run did: each member's result and the exchanges' counts
struct Outcome {
    std::vector<Result> results;          // a member without budget: empty
    std::vector<char> ran;
    long long rounds = 0, pieces = 0;
    long long offered = 0, accepted = 0, tried = 0, swapped = 0;
    long long own_evaluations = 0;        // ENS-2
    std::string why;                      // the rule's end, target, deadline or idle; empty: the members'
};

// a member's resumable run from its spec
using Make = std::function<std::unique_ptr<Resumable>(const Spec& spec, std::size_t m)>;

// a better than b: the less violating, then the lower f (search.hpp SRCH-8); no point is worst
inline bool better(const Member& a, const Member& b)
{
    if (a.x.empty()) return false;
    if (b.x.empty()) return true;
    if (a.cv != b.cv) return a.cv < b.cv;
    return a.f < b.f;
}

// the index of the best of X, the first of equals
inline std::size_t best_of(const std::vector<Member>& X)
{
    std::size_t b = 0;
    for (std::size_t q = 1; q < X.size(); ++q)
        if (better(X[q], X[b])) b = q;
    return b;
}

// a member's level of the parameter: its own, else the problem's (param; NaN where it has none)
inline double level_of(const Problem& P, const Spec& s) { return std::isnan(s.level) ? P.param : s.level; }

inline bool same_problem(const Problem& P, const Spec& a, const Spec& b)
{
    const double la = level_of(P, a), lb = level_of(P, b);
    return (std::isnan(la) && std::isnan(lb)) || la == lb;
}

// x of member `from` as a point of member `to`'s problem (ENS-3): carried
// between two levels (param.hpp carried, the problem's rescale), else only
// brought into the receiver's box (a level unknown: nothing to rescale by)
inline Vec carry(const Problem& P, const Spec& from, const Spec& to, const Vec& x)
{
    const double lf = level_of(P, from), lt = level_of(P, to);
    if (std::isfinite(lf) && std::isfinite(lt)) return carried(*from.problem, *to.problem, x, lf, lt);
    const Problem& Q = *to.problem;
    Vec y = x;
    for (std::size_t j = 0; j < y.size() && j < Q.lo.size(); ++j)
        y[j] = periodic::is(Q.periodic, j) ? periodic::wrap_into(y[j], Q.lo[j], Q.hi[j])
                                           : (std::min)(Q.hi[j], (std::max)(Q.lo[j], y[j]));
    return y;
}

// the members L of a round advanced by a piece each, in `threads` threads (ENS-7)
inline void advance_members(std::vector<std::unique_ptr<Resumable>>& R, const std::vector<std::size_t>& L,
                            const Piece& piece, int threads)
{
    using namespace strategy;
    const std::size_t n = L.size();
    const std::size_t T = start_depth() > 0 ? 1                                       // STRAT-7
                                            : (std::min)(n, static_cast<std::size_t>((std::max)(1, threads)));
    if (T <= 1) {
        for (std::size_t m : L) R[m]->advance(piece);
        return;
    }
    std::vector<std::exception_ptr> error(n);
    std::vector<Deferred> later(n);                   // THE DEFERRED COUNT, member by member
    std::vector<ops::RepairCounts> made(n);
    std::atomic<std::size_t> next{0};
    std::vector<std::thread> pool;
    for (std::size_t t = 0; t < T; ++t)
        pool.emplace_back([&]() {
            for (std::size_t k = next++; k < n; k = next++) {
                deferred_here() = &later[k];
                start_counts0() = ops::repair_counts();
                ++start_depth();
                try {
                    R[L[k]]->advance(piece);
                } catch (...) {
                    error[k] = std::current_exception();
                }
                --start_depth();
                made[k] = ops::repair_counts() - start_counts0();
                deferred_here() = nullptr;
            }
        });
    for (auto& th : pool) th.join();
    const ops::RepairCounts base = ops::repair_counts();
    ops::RepairCounts before{};                       // the earlier members' totals
    for (std::size_t k = 0; k < n; ++k) {             // in the members' order, as one thread runs them
        for (DeferredCall& call : later[k]) {
            ops::repair_counts() = base;
            ops::repair_counts() += before;
            ops::repair_counts() += call.at;
            defer_or_run(std::move(call.what));
        }
        before += made[k];
    }
    ops::repair_counts() = base;
    ops::repair_counts() += before;
    for (std::size_t k = 0; k < n; ++k)
        if (error[k]) std::rethrow_exception(error[k]);
}

// THE ENSEMBLE's run: the members made by `make`, the rounds, the exchanges;
// the members' results at the end (their result())
inline Outcome run(const Problem& P, long long budget, const Plan& plan, strategy::StopRule& stop, const Make& make)
{
    const std::size_t K = plan.members.size();
    Outcome out;
    out.results.resize(K);
    out.ran.assign(K, 0);
    std::vector<std::unique_ptr<Resumable>> R(K);
    std::vector<long long> bud(K, 0);
    for (std::size_t m = 0; m < K; ++m) {
        bud[m] = plan.members[m].budget;
        if (bud[m] > 0) R[m] = make(plan.members[m], m);
    }
    std::mt19937 rng(plan.stream);
    auto counted = [&]() {
        double c = static_cast<double>(out.own_evaluations);
        for (const auto& r : R)
            if (r) c += r->ledger().counted();
        return c;
    };
    auto left = [&]() { return budget - static_cast<long long>(std::ceil(counted())); };
    auto live = [&](std::size_t m) { return R[m] && !R[m]->done(); };
    // y evaluated in member m's problem, charged to its budget (ENS-2)
    auto value_at = [&](std::size_t m, const Vec& y) {
        const Problem& Q = *plan.members[m].problem;
        if (past_deadline(Q)) throw BudgetSpent{};                                   // TIME-2
        Vec f, g;
        Q.evaluate(y, f, g);
        ++out.own_evaluations;
        bud[m] -= 1;
        R[m]->set_budget(bud[m]);
        Member v;
        v.x = y;
        v.g = g;
        bool ok = !f.empty() && std::isfinite(f[0]);
        for (double gi : g) ok = ok && std::isfinite(gi);
        v.f = ok ? f[0] : std::numeric_limits<double>::infinity();
        v.cv = ok ? R[m]->ledger().violation(g, R[m]->ledger().eq_tol()) : std::numeric_limits<double>::infinity();
        return std::make_pair(v, ok);
    };
    auto room = [&](std::size_t m) { return R[m]->ledger().left(); };
    // migrate: the islands (ENS-3)
    auto migrate = [&]() {
        std::vector<std::vector<Member>> S(K);                                        // as before the exchange
        for (std::size_t m = 0; m < K; ++m)
            if (R[m]) S[m] = R[m]->members();
        for (std::size_t r = 0; r < K && K > 1; ++r) {
            if (!live(r)) continue;
            std::vector<std::size_t> from;
            if (plan.topology == Topology::ring) {
                from.push_back((r + K - 1) % K);
            } else if (plan.topology == Topology::full) {
                for (std::size_t q = 1; q < K; ++q) from.push_back((r + q) % K);
            } else {
                const std::size_t i = (std::min)(K - 2, static_cast<std::size_t>(rnd::uniform01(rng) * static_cast<double>(K - 1)));
                from.push_back(i < r ? i : i + 1);
            }
            const Spec& to = plan.members[r];
            for (std::size_t s : from) {
                if (!R[s] || S[s].empty() || plan.migrants <= 0) continue;
                const Spec& sp = plan.members[s];
                const bool same = same_problem(P, sp, to);
                std::vector<Vec> xs;                                                  // at the receiver's level
                for (const Member& c : S[s]) xs.push_back(same ? c.x : carry(P, sp, to, c.x));
                std::vector<std::size_t> order(S[s].size());
                for (std::size_t q = 0; q < order.size(); ++q) order[q] = q;
                const std::size_t take = (std::min)(order.size(), static_cast<std::size_t>(plan.migrants));
                if (plan.pick == Pick::best) {
                    std::stable_sort(order.begin(), order.end(),
                                     [&](std::size_t a, std::size_t b) { return better(S[s][a], S[s][b]); });
                } else if (plan.pick == Pick::random) {
                    for (std::size_t g = 0; g < take; ++g) {                          // a partial Fisher-Yates draw
                        const std::size_t q = g + (std::min)(order.size() - g - 1,
                                                             static_cast<std::size_t>(rnd::uniform01(rng) *
                                                                                      static_cast<double>(order.size() - g)));
                        std::swap(order[g], order[q]);
                    }
                } else {                                                              // diverse
                    std::vector<double> d(order.size(), 0.0);
                    if (!S[r].empty()) {
                        const Vec& elite = S[r][best_of(S[r])].x;
                        for (std::size_t q = 0; q < d.size(); ++q) d[q] = distance_of(*to.problem, xs[q], elite);
                    }
                    std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) { return d[a] > d[b]; });
                }
                std::vector<Member> in;
                for (std::size_t g = 0; g < take; ++g) {
                    const std::size_t q = order[g];
                    if (same) {
                        in.push_back(S[s][q]);
                        continue;
                    }
                    if (room(r) < 1) break;                                           // ENS-2
                    const auto v = value_at(r, xs[q]);
                    if (v.second) in.push_back(v.first);
                }
                if (in.empty()) continue;
                out.offered += static_cast<long long>(in.size());
                out.accepted += R[r]->admit(in, plan.place);
            }
        }
    };
    // swap: the neighbours' configurations (ENS-4)
    auto swap = [&]() {
        for (std::size_t a = 0; a + 1 < K; ++a) {
            const std::size_t b = a + 1;
            if (!live(a) || !live(b)) continue;
            const std::vector<Member> Xa = R[a]->members(), Xb = R[b]->members();
            if (Xa.empty() || Xb.empty()) continue;
            const Spec &A = plan.members[a], &B = plan.members[b];
            const double Ea = Xa[best_of(Xa)].f, Eb = Xb[best_of(Xb)].f;
            std::vector<Member> Ya, Yb;                                               // what a and b get
            double delta = 0.0;
            if (plan.over_param) {
                if (room(a) < static_cast<long long>(Xb.size()) || room(b) < static_cast<long long>(Xa.size())) continue;
                ++out.tried;
                bool ok = true;
                for (const Member& x : Xb) {
                    const auto v = value_at(a, carry(P, B, A, x.x));
                    ok = ok && v.second;
                    Ya.push_back(v.first);
                }
                for (const Member& x : Xa) {
                    const auto v = value_at(b, carry(P, A, B, x.x));
                    ok = ok && v.second;
                    Yb.push_back(v.first);
                }
                if (!ok) continue;
                delta = plan.beta * (Ya[best_of(Ya)].f + Yb[best_of(Yb)].f - Ea - Eb);
            } else {
                ++out.tried;
                delta = (1.0 / B.temperature - 1.0 / A.temperature) * (Ea - Eb);
                Ya = Xb;
                Yb = Xa;
            }
            const bool accept = !(delta > 0.0) || rnd::uniform01(rng) < dm::exp(-plan.fs * delta);   // RL-3
            if (accept) {
                R[a]->admit(Ya, Place::all);
                R[b]->admit(Yb, Place::all);
                ++out.swapped;
            }
        }
    };
    const bool one_round = !(plan.piece.spend > 0.0) && plan.piece.steps <= 0;
    Member best_so_far;
    long long idle = 0;
    try {
        while (stop.go(left(), P)) {
            std::vector<std::size_t> L;
            for (std::size_t m = 0; m < K; ++m)
                if (live(m)) L.push_back(m);
            if (L.empty()) break;
            const double before = counted();
            advance_members(R, L, plan.piece, plan.threads);
            ++out.rounds;
            out.pieces += static_cast<long long>(L.size());
            Member now;
            bool target = false;
            for (std::size_t m = 0; m < K; ++m) {
                if (!R[m]) continue;
                const Member b = R[m]->best();
                if (better(b, now)) now = b;
                target = target || R[m]->ledger().target_reached;
            }
            const bool improved = better(now, best_so_far);
            stop.step(improved);
            if (improved) best_so_far = now;
            if (target) {
                out.why = "target";
                break;
            }
            if (one_round) break;
            bool any = false;
            for (std::size_t m = 0; m < K; ++m) any = any || live(m);
            if (!any) break;
            if (plan.exchange == Exchange::migrate) migrate();
            if (plan.exchange == Exchange::swap) swap();
            idle = counted() == before ? idle + 1 : 0;
            if (idle >= 100) {
                out.why = "idle";
                break;
            }
        }
    } catch (const BudgetSpent&) {                    // an exchange's evaluation past the deadline (TIME-2)
        out.why = "deadline";
    }
    if (out.why.empty()) out.why = stop.stopped_by("");
    for (std::size_t m = 0; m < K; ++m)
        if (R[m]) {
            out.results[m] = R[m]->result();
            out.ran[m] = 1;
        }
    return out;
}

// the knob of a JSON value (as strategy.hpp inner_of reads a solver's)
inline Knob knob_of(const std::string& what, const json::Value& x)
{
    Knob kv;
    if (x.kind == json::Value::Kind::String) {
        kv.is_word = true;
        kv.word = x.s;
    } else if (x.kind == json::Value::Kind::Int) {
        kv.number = static_cast<double>(x.i);
    } else if (x.kind == json::Value::Kind::Float) {
        kv.number = x.f;
    } else if (x.kind == json::Value::Kind::Object) {
        kv.is_word = true;                                // a solver within: its canonical text
        kv.word = json::dump(x, -1, ",", ":");
    } else {
        throw std::invalid_argument(what + ": a word, a number or a solver");
    }
    return kv;
}

// a member list's JSON (a knob whose word is a JSON list), Null where not given
inline json::Value list_knob(strategy::StrategyKnobs& k, const std::string& name)
{
    const Knob* v = k.find(name);
    if (v == nullptr) return json::Value::null();
    if (!v->is_word) throw std::invalid_argument("ensemble: " + name + " must be a JSON list, one entry a member");
    json::Value o = json::parse(v->word);
    if (o.kind != json::Value::Kind::Array || o.items.empty())
        throw std::invalid_argument("ensemble: " + name + " must be a non-empty JSON list, one entry a member");
    k.settings.push_back({name, v->word});
    return o;
}

inline double number_of(const json::Value& x, const std::string& what)
{
    if (x.kind == json::Value::Kind::Int) return static_cast<double>(x.i);
    if (x.kind == json::Value::Kind::Float) return x.f;
    throw std::invalid_argument(what + " must be a number");
}

// knobs over others: a knob of the same name replaced, a new one added
inline Knobs over(Knobs base, const Knobs& top)
{
    for (const auto& kv : top) {
        bool set = false;
        for (auto& b : base)
            if (b.first == kv.first) {
                b.second = kv.second;
                set = true;
            }
        if (!set) base.push_back(kv);
    }
    return base;
}

// the f and cv of a result's answer (its first row); none: no point
inline Member answer_of(const Result& r)
{
    Member a;
    if (r.variables.empty() || r.objectives.empty() || r.objectives[0].empty()) return a;
    a.x = r.variables[0];
    a.f = r.objectives[0][0];
    a.cv = r.cv.empty() ? 0.0 : r.cv[0];
    return a;
}

}   // namespace ens

inline Result run_ensemble(const Problem& P, double eq_tol, bool has_tol0, double eq_tol0, int pop_size,
                           long long budget, unsigned seed, const Knobs& knobs, Observer observer, int record_every,
                           const Start& start)
{
    using namespace strategy;
    StrategyKnobs k("ensemble", knobs,
                    {"inner", "ens_K", "exchange", "exchange_every", "topology", "migrants", "migrant_pick",
                     "migrant_place", "swap_over", "swap_knob", "swap_beta", "swap_fs", "member_seeds", "member_knobs",
                     "member_params", "threads"});
    const Inner inner = k.inner("inner");
    if (!detail::listed(resumable_names(), inner.name))
        throw std::invalid_argument("ensemble: the inner strategy must be resumable (resumable_names()), not " +
                                    inner.name);
    const json::Value seeds = ens::list_knob(k, "member_seeds");
    const json::Value mknobs = ens::list_knob(k, "member_knobs");
    const json::Value params = ens::list_knob(k, "member_params");
    std::size_t K = 0;
    if (k.given("ens_K")) K = static_cast<std::size_t>(k.whole("ens_K", 1, 1));
    for (const json::Value* l : {&seeds, &mknobs, &params}) {
        if (l->kind == json::Value::Kind::Null) continue;
        if (K == 0) K = l->items.size();
        if (l->items.size() != K)
            throw std::invalid_argument("ensemble: member_seeds, member_knobs and member_params need one entry a member (" +
                                        std::to_string(K) + ")");
    }
    if (K == 0) K = 1;
    if (!k.given("ens_K")) k.settings.push_back({"ens_K", std::to_string(K)});
    const std::string how = k.word("exchange", "migrate", {"migrate", "swap", "none"});
    const long long every = k.whole("exchange_every", 0, 0);
    ens::Plan plan;
    plan.exchange = every == 0 || how == "none" ? ens::Exchange::none
                    : how == "swap"             ? ens::Exchange::swap
                                                : ens::Exchange::migrate;
    if (every > 0) plan.piece.steps = every;
    plan.threads = static_cast<int>(k.whole("threads", 1, 1));
    plan.stream = start_seed(seed, 0xE45Eu);                                          // ENS-3
    std::string swap_knob;
    if (how == "migrate") {
        const std::string t = k.word("topology", "ring", {"ring", "full", "random"});
        plan.topology = t == "full" ? ens::Topology::full : t == "random" ? ens::Topology::random : ens::Topology::ring;
        plan.migrants = k.whole("migrants", 1, 0);
        const std::string p = k.word("migrant_pick", "best", {"best", "random", "diverse"});
        plan.pick = p == "random" ? ens::Pick::random : p == "diverse" ? ens::Pick::diverse : ens::Pick::best;
        plan.place = k.word("migrant_place", "worst", {"worst", "worst_if_better"}) == "worst" ? Place::worst
                                                                                         : Place::worst_if_better;
    } else if (how == "swap") {
        plan.over_param = k.word("swap_over", "knob", {"knob", "param"}) == "param";
        plan.beta = k.number("swap_beta", 1.0);
        plan.fs = k.number("swap_fs", 1.0);
        if (!(plan.fs > 0.0)) throw std::invalid_argument("ensemble: swap_fs must be > 0");
        if (plan.over_param && params.kind == json::Value::Kind::Null)
            throw std::invalid_argument("ensemble: swap_over = param needs member_params, a level a member");
        if (!plan.over_param) {
            const Knob* sk = k.find("swap_knob");
            if (!sk || !sk->is_word || sk->word.empty())
                throw std::invalid_argument("ensemble: swap_over = knob needs swap_knob, the inner strategy's knob");
            swap_knob = sk->word;
            k.settings.push_back({"swap_knob", swap_knob});
        }
    }
    StopRule stop = stop_of(k);                                                       // a step a round
    const Knob* units = k.find("budget_units");
    const std::vector<long long> share = split_budget(budget, K);                     // ENS-1
    plan.members.resize(K);
    for (std::size_t m = 0; m < K; ++m) {
        ens::Spec& s = plan.members[m];
        const std::string at = "ensemble: member " + std::to_string(m);
        s.problem = &P;
        if (params.kind != json::Value::Kind::Null && params.items[m].kind != json::Value::Kind::Null) {
            s.level = ens::number_of(params.items[m], at + "'s member_params");
            s.own = std::make_shared<Problem>(problem_at(P, s.level));
            s.problem = s.own.get();
        }
        if (plan.over_param && std::isnan(s.level))
            throw std::invalid_argument(at + ": swap_over = param needs its level in member_params");
        Knobs top;
        if (mknobs.kind != json::Value::Kind::Null) {
            const json::Value& o = mknobs.items[m];
            if (o.kind != json::Value::Kind::Null && o.kind != json::Value::Kind::Object)
                throw std::invalid_argument(at + "'s member_knobs must be an object of knobs, or null");
            for (std::size_t q = 0; q < o.keys.size(); ++q)
                top.push_back({o.keys[q], ens::knob_of(at + "." + o.keys[q], o.items[q])});
        }
        s.knobs = ens::over(inner.knobs, top);
        if (units) s.knobs = ens::over({{"budget_units", *units}}, s.knobs);         // the member's own first
        if (!swap_knob.empty()) {
            bool found = false;
            for (const auto& kv : s.knobs)
                if (kv.first == swap_knob && !kv.second.is_word) {
                    s.temperature = kv.second.number;
                    found = true;
                }
            if (!found || !(s.temperature > 0.0))
                throw std::invalid_argument(at + ": swap_over = knob needs its " + swap_knob + " > 0 (member_knobs)");
        }
        s.seed = start_seed(seed, static_cast<std::uint64_t>(m));
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
        s.budget = share[m];
    }
    const ens::Make make = [&](const ens::Spec& s, std::size_t) {
        return start_resumable(inner.name, *s.problem, eq_tol, has_tol0, eq_tol0, pop_size, s.budget, s.seed,
                               s.knobs, K == 1 ? observer : Observer{}, K == 1 ? record_every : 0, s.start);   // ENS-8
    };
    const ens::Outcome o = ens::run(P, budget, plan, stop, make);
    // ENS-5: the answer
    std::size_t pick = K;
    for (std::size_t m = 0; m < K; ++m) {
        if (!o.ran[m] || !std::isnan(plan.members[m].level)) continue;
        const Member a = ens::answer_of(o.results[m]);
        if (a.x.empty()) continue;
        if (pick == K || ens::better(a, ens::answer_of(o.results[pick]))) pick = m;
    }
    if (pick == K)
        for (std::size_t m = K; m-- > 0;)
            if (o.ran[m]) {
                pick = m;
                break;
            }
    Result out = pick < K ? o.results[pick] : Result();
    out.evaluations = o.own_evaluations;
    out.grad_calls = out.n_failed = out.hook_calls = out.success_calls = out.delta_calls = 0;
    out.cache_hits = out.decode_invalid = out.outside_evals = out.outside_grads = 0;
    out.hook_spent = out.hook_seconds = out.success_spent = out.delta_spent = 0.0;
    out.used = static_cast<double>(o.own_evaluations);
    out.population.clear();
    std::string per = "[";
    for (std::size_t m = 0; m < K; ++m) {
        const Member a = o.ran[m] ? ens::answer_of(o.results[m]) : Member();
        per += (m ? "," : "") + (a.x.empty() ? std::string("null") : detail::g_format(a.f));
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
    per += "]";
    // ENS-6: what ended the run
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
    out.settings.insert(out.settings.end(), k.settings.begin(), k.settings.end());
    out.settings.push_back({"ens_best", per});
    out.settings.push_back({"ens_answer", pick < K ? std::to_string(pick) : std::string("none")});
    out.settings.push_back({"rounds", std::to_string(o.rounds)});
    if (plan.exchange == ens::Exchange::migrate) {
        out.settings.push_back({"migrants_offered", std::to_string(o.offered)});
        out.settings.push_back({"migrants_accepted", std::to_string(o.accepted)});
    }
    if (plan.exchange == ens::Exchange::swap) {
        out.settings.push_back({"swaps_tried", std::to_string(o.tried)});
        out.settings.push_back({"swaps_accepted", std::to_string(o.swapped)});
    }
    if (o.own_evaluations > 0) out.settings.push_back({"exchange_evaluations", std::to_string(o.own_evaluations)});
    return out;
}

// replica_ladder with rl_mode = resume (replica.hpp RL-5): the replicas as the
// members of an ensemble, swap over the ladder every swap_every of the budget
inline Result replica_resumed(const Problem& P, double eq_tol, bool has_tol0, double eq_tol0, int pop_size,
                              long long budget, unsigned seed, const Start& start, const std::string& inner,
                              const std::vector<Problem>& Q, const std::vector<Knobs>& kn, const std::vector<double>& lv,
                              bool over_param, double beta, double fs, long long every, strategy::StopRule& stop,
                              strategy::Ledger& ledger, double grad_cost, std::vector<std::string> ignored)
{
    using namespace strategy;
    const std::size_t K = Q.size();
    ens::Plan plan;
    plan.exchange = every > 0 ? ens::Exchange::swap : ens::Exchange::none;
    if (every > 0) plan.piece.spend = static_cast<double>(every);
    plan.over_param = over_param;
    plan.beta = beta;
    plan.fs = fs;
    plan.stream = start_seed(seed, 0x5EA1u);                                          // RL-3's stream
    const long long each = budget / static_cast<long long>(K);
    plan.members.resize(K);
    for (std::size_t m = 0; m < K; ++m) {
        ens::Spec& s = plan.members[m];
        s.problem = &Q[m];
        s.level = over_param ? lv[m] : std::numeric_limits<double>::quiet_NaN();
        s.knobs = kn[m];
        s.temperature = over_param ? 0.0 : lv[m];
        s.seed = start_seed(seed, static_cast<std::uint64_t>(m));                     // the first pieces' seeds
        s.start = start;
        s.budget = each > 0 ? each : (static_cast<long long>(m) < budget ? 1 : 0);    // the pieces of swap_every = 0
    }
    const ens::Make make = [&](const ens::Spec& s, std::size_t) {
        return start_resumable(inner, *s.problem, eq_tol, has_tol0, eq_tol0, pop_size, s.budget, s.seed, s.knobs, {},
                               0, s.start);
    };
    const ens::Outcome o = ens::run(P, budget, plan, stop, make);
    std::vector<char> has(K, 0);
    for (std::size_t m = 0; m < K; ++m) {
        if (!o.ran[m]) continue;
        ledger.add(o.results[m], grad_cost);
        has[m] = o.results[m].variables.empty() ? 0 : 1;
    }
    ledger.evaluations += o.own_evaluations;
    std::size_t pick = K - 1;                                                        // RL-4
    if (!over_param)
        for (std::size_t m = 0; m < K; ++m)
            if (has[m] && (!has[pick] || o.results[m].objectives[0][0] < o.results[pick].objectives[0][0])) pick = m;
    Result out = o.results[pick];
    ledger.into(out);
    std::string why = o.why;
    for (std::size_t m = 0; m < K; ++m)
        if (o.ran[m] && o.results[m].stopped_by == "target") why = "target";
    out.stopped_by = why.empty() ? "rule" : why;
    out.ignored = std::move(ignored);
    std::string per = "[";
    for (std::size_t m = 0; m < K; ++m)
        per += (m ? "," : "") + (has[m] ? detail::g_format(o.results[m].objectives[0][0]) : std::string("null"));
    per += "]";
    out.settings.push_back({"replica_best", per});
    out.settings.push_back({"pieces", std::to_string(o.pieces)});
    out.settings.push_back({"swaps_tried", std::to_string(o.tried)});
    out.settings.push_back({"swaps_accepted", std::to_string(o.swapped)});
    out.settings.push_back({"rl_mode", "resume"});
    return out;
}

}   // namespace mootation::scalar
