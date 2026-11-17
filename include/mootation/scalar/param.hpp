#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// An outer loop over a parameter of the problem (task 26, item 24.2.7): the
// problem's param, at_param(L) and rescale
// (evaluator.hpp), and the strategy shrink. Included by strategy.hpp.
//
// THE LEVELS. problem_at(P, L) is P.at_param(L), checked (PARAM-1); a level
// is run by the inner solver on an Evaluator of its own (the problem at L
// evaluates directly, outside any start's view), its evaluations and
// gradient calls added to the run's. carried(P_from, P_to, x, from, to) is
// P_from's rescale on a copy of x, then clipped to P_to's box (a periodic
// coordinate wrapped) — the warm start of the next level (PARAM-2). A level
// succeeds when its run reached the problem-at-L's target: a feasible
// answer at or below its stop_at, or its accept_success (PARAM-3).
//
// shrink: the levels from L_0 = P.param, each from the last success's point
// carried to the new L (the first from the run's start):
//   shrink_step = geometric  L <- rho L after a success (shrink_rho 0.99),
//                            the first failure ends;
//                 bisect     between L_lo = shrink_lo (infeasible, given) and
//                            L_hi = L_0 (feasible): the middle, a success
//                            moves L_hi, a failure L_lo, until L_hi - L_lo
//                            <= shrink_tol L_hi (1e-6) — Huang & Ye 2011's
//                            rule (not checked on its text);
//                 ilsqn      Imamichi, Yagiura, Nagamochi 2009, §2.2: after a
//                            success L := (1 - r_dec) L_best, after a failure
//                            L := (1 + r_inc) L and, when L >= L_best, back
//                            to (1 - r_dec) L_best from the best point
//                            (ilsqn_dec 0.04, ilsqn_inc 0.01);
//                 idts       Lai et al. 2022, §2.2: L = L_best - Delta,
//                            Delta <- max(theta Delta, Delta_min) after every
//                            level (idts_theta 0.7, idts_delta 0.1,
//                            idts_min 1e-4);
//                 ladder     the values of shrink_ladder (a JSON list) in
//                            order, every one from the last level's point,
//                            success or not: continuation along a ladder
//                            (smoothing a model's thresholds down to the
//                            model itself, Kavetski & Kuczera 2007 §8.4).
//   The answer is the run of the best level: the least L that succeeded
//   (ladder: the last level), else the first level; its settings name L.
// THE DEGENERATE CASE: one level (shrink_rounds = 1) is the inner solver on
// the problem at L_0 from the run's start with the whole budget, bit for bit
// (the seed is the run's, STRAT-1; the counts are that run's).
//
// Knobs: inner, shrink_step (geometric), shrink_rho, shrink_lo, shrink_tol,
// ilsqn_dec, ilsqn_inc, idts_theta, idts_delta, idts_min, shrink_ladder,
// shrink_rounds (0: until the budget or the rule ends), start_budget,
// shrink_count (budget; PARAM-8), shrink_carry (point; PARAM-9).
//
// READINGS:
//   PARAM-1. problem_at refuses a problem without at_param, a non-finite L,
//     a problem at L whose n differs or whose param is not L (std::isnan,
//     never ==, for "none"); the message names L.
//   PARAM-2. The rescale is the source problem's (from and to are its
//     terms); without one x goes unchanged into the new box, clipped.
//   PARAM-3. Success is the level's own target (stop_at, accept_success);
//     a level's success cost belongs to its Evaluator (SUCC-2).
//   PARAM-4. A level's budget: start_budget, else what is left over the
//     levels still allowed (shrink_rounds), else a tenth of what is left
//     (the rules other than ladder do not know their number of levels);
//     ladder shares what is left among the levels left. At least one
//     evaluation; the loop ends when nothing is left.
//   PARAM-5. No observer across the levels: each level's run has its own
//     archive; the run's trajectory is the best level's.
//   PARAM-6. The levels in the settings (shrink_trail, param_best) carry 17
//     significant digits, so that they read back as the values tried.
//   PARAM-7 (only when asked: max_seconds). The run's deadline (scalar.hpp
//     TIME-1; task 30, §30.1): problem_at gives the problem at L the earlier of
//     its own deadline and P's, so a level's run ends at the run's
//     deadline; shrink starts no level once the deadline has passed (a
//     level it cut short ends the levels, not as the rule's failure), and
//     the settings say why the levels ended (stopped_by: deadline, budget,
//     rounds, or rule — the rule's own end). Without a deadline nothing
//     changes, stopped_by included.
//   PARAM-8. shrink_count = budget | evaluations (task 30, §30.2; budget by
//     default since the owner's choice of 16.11, method version 5;
//     evaluations: the runs before). A level's run spends its allowance in
//     the budget's units: evaluations + grad_cost gradient calls + the
//     success, hook and delta costs (evaluator.hpp THE COUNTS). budget
//     charges each level all of it (grad_cost the inner solver's knob where
//     it takes one, else 1), so the levels together stay within the run's
//     budget; evaluations charged the evaluations alone, so with gradients
//     the levels went on past the budget (up to shrink_rounds times
//     start_budget). Without gradients and costs the two are the same run.
//     The settings have shrink_counts: every level's [evaluations, gradient
//     calls, charge]; evaluations in the result is the sum of the levels'
//     evaluations, grad_calls of their gradient calls.
//   PARAM-9. shrink_carry = point | population (task 30, §30.6; off by
//     default: point, the level from one point). population: the inner
//     solver is cluster_ga with cga_init = seed_points (else refused); a
//     level starts from the carried point as before and from the last
//     population of the level it is carried from (Result::population),
//     every member carried as the point is, the member equal to the point
//     (else the last) left out, as the start's seed points: members 2 ... K
//     of the next level's first population (cluster_ga CGA-5), so the
//     number of members is kept. A level whose first round did not end
//     (its target reached in it, or its allowance spent) has no population:
//     the next starts from the point alone, its members 2 ... K uniform.
//     shrink_carried: every level's carried members with the point (0 for
//     the first level).
// ============================================================================

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "periodic.hpp"
#include "strategy.hpp"

namespace mootation::scalar {

// a parameter's value in the settings, every digit (PARAM-6)
inline std::string param_text(double v)
{
    char buf[64];
    std::snprintf(buf, sizeof buf, "%.17g", v);
    return buf;
}

// PARAM-1
inline Problem problem_at(const Problem& P, double L)
{
    if (!P.at_param) throw std::invalid_argument("the problem has no at_param (a parameter's levels)");
    if (!std::isfinite(L)) throw std::invalid_argument("at_param: the level L = " + std::to_string(L) + " is not finite");
    Problem Q = P.at_param(L);
    if (Q.lo.size() != P.lo.size() || Q.hi.size() != P.hi.size())
        throw std::invalid_argument("at_param(" + std::to_string(L) + ") changed the number of variables");
    if (std::isnan(Q.param) || Q.param != L)
        throw std::invalid_argument("at_param(" + std::to_string(L) + ") gave a problem whose param is not L");
    return derived(P, std::move(Q));                         // PARAM-7: the deadline, the clock (stop_rule.hpp STOP-6)
}

// PARAM-2
inline Vec carried(const Problem& P_from, const Problem& P_to, const Vec& x, double from, double to)
{
    Vec y = x;
    if (P_from.rescale) P_from.rescale(y, from, to);
    if (y.size() != P_to.lo.size()) throw std::runtime_error("rescale changed the point's length");
    for (std::size_t j = 0; j < y.size(); ++j)
        y[j] = periodic::is(P_to.periodic, j) ? periodic::wrap_into(y[j], P_to.lo[j], P_to.hi[j])
                                              : (std::min)(P_to.hi[j], (std::max)(P_to.lo[j], y[j]));
    return y;
}

// PARAM-3
inline bool reached(const Problem& Q, const Result& r)
{
    if (r.objectives.empty()) return false;
    if (r.success_at >= 0) return true;
    const bool feasible = r.cv.empty() || r.cv[0] <= 0.0;
    return feasible && r.objectives[0][0] <= Q.stop_at;
}

inline Result run_shrink(const Problem& P, double eq_tol, bool has_tol0, double eq_tol0, int pop_size, long long budget,
                         unsigned seed, const Knobs& knobs, Observer observer, int record_every, const Start& start)
{
    using namespace strategy;
    (void)observer;
    (void)record_every;                                                              // PARAM-5
    StrategyKnobs k("shrink", knobs, {"inner", "shrink_step", "shrink_rho", "shrink_lo", "shrink_tol", "ilsqn_dec",
                                      "ilsqn_inc", "idts_theta", "idts_delta", "idts_min", "shrink_ladder",
                                      "shrink_rounds", "start_budget", "shrink_count", "shrink_carry"});
    const Inner inner = k.inner("inner");
    const std::string rule = k.word("shrink_step", "geometric", {"geometric", "bisect", "ilsqn", "idts", "ladder"});
    const double rho = k.number("shrink_rho", 0.99), tol = k.number("shrink_tol", 1e-6);
    const double dec = k.number("ilsqn_dec", 0.04), inc = k.number("ilsqn_inc", 0.01);
    const double theta = k.number("idts_theta", 0.7), delta0 = k.number("idts_delta", 0.1), dmin = k.number("idts_min", 1e-4);
    const long long rounds = k.whole("shrink_rounds", 0, 0), cap = k.whole("start_budget", 0, 0);
    if (std::isnan(P.param) || !P.at_param) throw std::invalid_argument("shrink needs the problem's param and at_param");
    if (!(rho > 0.0 && rho < 1.0)) throw std::invalid_argument("shrink: shrink_rho in (0, 1)");
    std::vector<double> ladder;
    if (rule == "ladder") {
        const Knob* kl = k.find("shrink_ladder");
        if (!kl || !kl->is_word) throw std::invalid_argument("shrink: shrink_step = ladder needs shrink_ladder, a JSON list");
        const json::Value v = json::parse(kl->word);
        if (v.kind != json::Value::Kind::Array || v.items.empty())
            throw std::invalid_argument("shrink: shrink_ladder must be a non-empty JSON list of numbers");
        for (const auto& it : v.items)
            ladder.push_back(it.kind == json::Value::Kind::Int ? static_cast<double>(it.i) : it.f);
    }
    // PARAM-8: what a level is charged; the default is this one line's
    const std::string count_default = "budget";
    const bool by_budget = (k.find("shrink_count") ? k.word("shrink_count", count_default, {"budget", "evaluations"})
                                                   : count_default) == "budget";
    const bool carry_pop =                                                           // PARAM-9
        k.find("shrink_carry") && k.word("shrink_carry", "point", {"point", "population"}) == "population";
    if (carry_pop) {
        bool seeded = false;
        for (const auto& kv : inner.knobs) seeded = seeded || (kv.first == "cga_init" && kv.second.word == "seed_points");
        if (inner.name != "cluster_ga" || !seeded)
            throw std::invalid_argument("shrink: shrink_carry = population needs inner cluster_ga with cga_init = "
                                        "seed_points (PARAM-9)");
    }
    double lo_b = 0.0;
    if (rule == "bisect") {
        lo_b = k.number("shrink_lo", std::numeric_limits<double>::quiet_NaN());
        if (!(lo_b < P.param)) throw std::invalid_argument("shrink: bisect needs shrink_lo below the problem's param");
    }

    StopRule stop = stop_of(k);                                                      // stop_rule.hpp: a step a level
    Ledger ledger;                                                                   // STOP-6: the levels' spending
    long long spent = 0, grads = 0, levels = 0;
    double L = rule == "ladder" ? ladder[0] : P.param, L_best = std::numeric_limits<double>::quiet_NaN();
    double delta = delta0, hi_b = P.param;
    Result best, first;
    bool have_best = false;
    Vec x_best, x_last;
    Rows pop_best, pop_last;                                                         // PARAM-9
    std::string carried_sizes = "[";
    double L_last = P.param;
    std::string trail = "[";
    std::string stopped = "rule";                                                    // PARAM-7
    double charged = 0.0;                                                            // PARAM-8
    std::string counts = "[";
    for (;;) {
        const long long left = budget - (by_budget ? static_cast<long long>(std::ceil(charged)) : spent);
        if (past_deadline(P)) {
            stopped = "deadline";
            break;
        }
        if (left <= 0) {
            stopped = "budget";
            break;
        }
        if (rounds > 0 && levels >= rounds) {
            stopped = "rounds";
            break;
        }
        if (rule == "ladder" && static_cast<std::size_t>(levels) >= ladder.size()) break;
        if (!stop.go(left, P)) {                                                     // stall, max_steps
            stopped = stop.stopped_by("rule");
            break;
        }
        long long allowance = cap > 0 ? (std::min)(cap, left)                                        // PARAM-4
                              : rounds > 0 ? left / (rounds - levels)
                              : rule == "ladder" ? left / static_cast<long long>(ladder.size() - static_cast<std::size_t>(levels))
                                                 : left / 10;
        allowance = (std::max)(1LL, (std::min)(allowance, left));
        const Problem Q = problem_at(P, L);
        Start st = start;
        if (levels > 0) {
            const Vec& from_x = rule == "ladder" ? x_last : (have_best ? x_best : x_last);
            const double from_L = rule == "ladder" ? L_last : (have_best ? L_best : L_last);
            st = search::at(carried(P, Q, from_x, from_L, L));
            if (carry_pop) {                                                         // PARAM-9
                const Rows& from_pop = rule == "ladder" ? pop_last : (have_best ? pop_best : pop_last);
                std::size_t skip = from_pop.size();
                for (std::size_t q = 0; q < from_pop.size() && skip == from_pop.size(); ++q)
                    if (from_pop[q] == from_x) skip = q;
                if (skip == from_pop.size() && !from_pop.empty()) skip = from_pop.size() - 1;
                st.has_seed_points = true;
                for (std::size_t q = 0; q < from_pop.size(); ++q)
                    if (q != skip) st.seed_points.push_back(Point{carried(P, Q, from_pop[q], from_L, L), {}, {}});
            }
        }
        if (carry_pop)
            carried_sizes += (levels ? "," : "") + std::to_string(levels ? st.seed_points.size() + 1 : 0);
        const Result r = run(inner.name, Q, eq_tol, has_tol0, eq_tol0, pop_size, allowance,
                             start_seed(seed, static_cast<std::uint64_t>(levels)), inner.knobs, {}, 0, st);
        spent += r.evaluations;
        grads += r.grad_calls;
        double grad_cost = 1.0;                                                      // PARAM-8
        if (std::find(r.ignored.begin(), r.ignored.end(), "grad_cost") == r.ignored.end())
            for (const auto& kv : inner.knobs)
                if (kv.first == "grad_cost" && !kv.second.is_word) grad_cost = kv.second.number;
        const double cost = static_cast<double>(r.evaluations) + grad_cost * static_cast<double>(r.grad_calls) +
                            r.success_spent + r.hook_spent + r.delta_spent;
        charged += cost;
        ledger.add(r, grad_cost);
        counts += (levels ? "," : "") + std::string("[") + std::to_string(r.evaluations) + "," +
                  std::to_string(r.grad_calls) + "," + param_text(cost) + "]";
        const bool ok = reached(Q, r);
        trail += (levels ? "," : "") + std::string("[") + param_text(L) + "," + (ok ? "1" : "0") + "]";
        if (levels == 0) first = r;
        ++levels;
        if (!r.variables.empty()) {
            x_last = r.variables[0];
            pop_last = r.population;
            L_last = L;
        }
        stop.step(ok && (!have_best || L < L_best || rule == "ladder"));            // STOP-3: a lower level that succeeded
        if (rule == "ladder") {
            best = r;
            have_best = true;
            L_best = L;
            if (static_cast<std::size_t>(levels) < ladder.size()) L = ladder[static_cast<std::size_t>(levels)];
            continue;
        }
        if (ok && (!have_best || L < L_best)) {
            best = r;
            have_best = true;
            L_best = L;
            x_best = r.variables[0];
            pop_best = r.population;
        }
        if (past_deadline(P)) {                                                      // PARAM-7: not the rule's end
            stopped = "deadline";
            break;
        }
        if (rule == "geometric") {
            if (!ok) break;
            L = rho * L;
        } else if (rule == "bisect") {
            if (ok) hi_b = L;
            else lo_b = L;
            if (hi_b - lo_b <= tol * std::fabs(hi_b)) break;
            L = 0.5 * (lo_b + hi_b);
        } else if (rule == "ilsqn") {
            if (ok) {
                L = (1.0 - dec) * L_best;
            } else {
                L = (1.0 + inc) * L;
                if (have_best && L >= L_best) L = (1.0 - dec) * L_best;
            }
        } else {                                                                     // idts
            delta = (std::max)(theta * delta, dmin);
            L = (have_best ? L_best : L) - delta;
        }
    }
    trail += "]";
    Result out = have_best ? best : first;
    ledger.into(out);                                                                // the levels' counts and costs
    out.evaluations = spent;
    out.grad_calls = grads;
    out.stopped_by = stopped;
    out.ignored = k.ignored();
    out.settings.push_back({"shrink_levels", std::to_string(levels)});
    out.settings.push_back({"shrink_trail", trail});
    if (have_best) out.settings.push_back({"param_best", param_text(L_best)});
    if (P.deadline < std::numeric_limits<double>::infinity()) out.settings.push_back({"stopped_by", stopped});
    out.settings.push_back({"shrink_counts", counts + "]"});                                 // PARAM-8
    if (carry_pop) out.settings.push_back({"shrink_carried", carried_sizes + "]"});           // PARAM-9
    return out;
}

}   // namespace mootation::scalar
