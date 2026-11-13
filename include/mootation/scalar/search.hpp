#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// Strategies around a local solver (task 18, stage 3, §3.3), on the rounds of
// strategy.hpp: every local search is a start of its round, the inner solver
// on a view of the problem. Included by strategy.hpp.
//
//   mlsl           Multi Level Single Linkage: A.H.G. Rinnooy Kan, G.T.
//                  Timmer, "Stochastic global optimization methods", Math.
//                  Program. 39 (1987), part I (source mlsl-part1_
//                  rinnooykan1987, (35)) and part II (mlsl-part2_
//                  rinnooykan1987, §2 and §4). A uniform sample grows by
//                  mlsl_n points an iteration; in iteration k the local
//                  search starts from every point of the reduced sample (the
//                  γkN of least f) from which none has started yet and that
//                  has no sample point of smaller f within the critical
//                  distance r_k = π^(-1/2) (Γ(1 + n/2) m(S) σ log(kN) /
//                  kN)^(1/n), and that is not within mlsl_tau of the box's
//                  boundary (Q_τ) nor within mlsl_nu of a minimum found (X*_ν).
//                  Knobs: inner, mlsl_n (100), mlsl_sigma (4), mlsl_gamma
//                  (0.2), mlsl_tau (0), mlsl_nu (0), threads.
//   ils            Iterated local search: H.R. Lourenço, O. Martin, T.
//                  Stützle, "Iterated Local Search" (2001; source iterated-
//                  local-search_lourenco2001), §2-3: a local search from the
//                  start, then over and over a perturbation of the current
//                  local minimum (every coordinate moved by U[-1, 1] times
//                  ils_step times the box's width) and a local search from
//                  there, and the acceptance criterion (ils_accept): better
//                  (1), rw (2), restart (3; a uniform point after ils_restart
//                  iterations without improvement), lsmc (better, else with
//                  probability exp((C(s*) - C(s*'))/T), T = temperature).
//   basin_hopping  D.J. Wales, J.P.K. Doye, J. Phys. Chem. A 101 (1997) 5111
//                  (source basin-hopping_wales1997_arxiv), p. 6: Monte Carlo
//                  on the transformed surface, i.e. ils with lsmc at a
//                  constant temperature, every coordinate displaced by U[-1,
//                  1] times the step, the step adjusted to an acceptance
//                  ratio of 0.5 (bh_accept_rate).
//   halving        Successive halving: K. Jamieson, A. Talwalkar, AISTATS
//                  2016 (source successive-halving_jamieson2016), Algorithm
//                  1: n_starts arms; in round k each arm left gets
//                  ⌊B/(|S_k| ⌈log2 n⌉)⌋ more evaluations, then the better half
//                  stays. An arm is the inner solver from its start (start 0
//                  its own, the others from start_points, as multistart).
//   memetic        A global solver (inner) and a local one (local_solver) as
//                  it goes: every memetic_every evaluations of the global
//                  solver, local searches from memetic_points points of the
//                  archive (memetic_pick: best, or spread: the best point
//                  and then those farthest from the starts so far among the
//                  best tenth), together local_share of the budget.
//
// Common knobs: start_budget (a local search's allowance; 0 what is left,
// shared among the round's starts), min_dist, min_df (the minima log).
//
// READINGS (declared):
//   SRCH-1. MLSL's r_k: part II's (2) prints it without Γ; it refers to part
//     I's Single Linkage analysis, whose (35) has Γ(1 + n/2) (the volume of
//     the unit ball): taken with Γ. m(S) = 1, distances in the box
//     normalized to [0, 1]^n.
//   SRCH-2. MLSL as part II's experiments (§4): τ = ν = 0 by default (no
//     boundary set Q_τ, no neighbourhood X*_ν of the minima found), σ = 4,
//     γ = 0.2. With ν = 0 a sample point lower than every other within r_k
//     near a minimum found starts a search again, so the searches grow slowly
//     with the sample; ν > 0 keeps them finite (part I, §3; Theorem 1). The
//     sample grows by mlsl_n points an iteration until the budget, not by
//     the Bayesian rule (3)-(4): the budget is the stopping rule here.
//   SRCH-3. MLSL: the local searches of an iteration are one round; a point
//     that started one never starts another; the sample is the uniform points
//     only (a local search's points are in the archive, not in the sample).
//   SRCH-4. A perturbed point is clipped to the box (ils, basin_hopping).
//   SRCH-5. basin_hopping's step: adjusted every bh_interval (10) steps,
//     divided by 0.9 when the acceptance ratio was above bh_accept_rate,
//     multiplied by 0.9 otherwise (the paper gives the target, not the rule).
//     The step is a share of the box's width (bh_step, 0.05); the temperature
//     (bh_temperature, 1) is in the units of f.
//   SRCH-6. halving: a round reruns each arm with its whole allowance so far;
//     its earlier points come from the archive, free, and only what goes
//     beyond costs (a solver whose run does not depend on the budget repeats
//     itself exactly: halving's arms continue). Arms are ranked by the best
//     point they found, the better half rounded up stays.
//   SRCH-7. memetic: the global solver is rerun with its allowance grown by
//     memetic_every, its earlier points free (as SRCH-6); it does not see the
//     local searches' points (no Lamarckian step), they only enter the answer.
//     With memetic_every as large as the budget it is the global solver and
//     then local searches at the end, as polish. lamarck (SRCH-12) writes
//     them back.
//   SRCH-8. A local search's result is the inner solver's answer; worse means
//     more violating, then larger f (Deb).
//   SRCH-12. memetic's lamarck (task 24, §24.1.1; off by default, and then
//     SRCH-7 holds): on — every local search's answer x' (its values from
//     the archive) is a seed of the global solver's later runs, the newest
//     first, with warm_start = on handed to it (scalar.hpp POP-7: the first
//     population's first rows; a single solver starts from the best seed);
//     the strategy's x0 is no longer the global run's x0 (the oldest seed
//     where it was evaluated);
//     share — each x' is written with probability lamarck_share (0.25;
//     Houck et al. 1997: 20-40 %), the coins from mt19937(start_seed(seed,
//     2^32)) in the order of the searches. The global run then differs from
//     the last phase's, and its earlier points are free only where met
//     again. lamarck and lamarck_share go to the settings only where given.
//   SRCH-13. mlsl_stop = bayes (task 24, §24.2.5; off by default: budget):
//     mlsl stops after the iteration where w (gamma k N - 1) / (gamma k N - w
//     - 2) <= w + 1/2 (part II, (3)), w the minima log's minima (at least
//     one), gamma k N the reduced sample (floor, as the iteration's); the
//     sample grows by mlsl_n as before, not to part II's (4).
//   SRCH-9. memetic: a phase whose global run spends nothing is the last (as
//     RST-8): the global solver has stopped by its own criteria (rerun, it
//     evaluates nothing new), and the local searches alone would go on in
//     phases of a few evaluations each (5380 phases of a budget of 20000 on a
//     linear f with its minimum in the box's corner), or of none at all once
//     every archive point has been a local start.
//   SRCH-10. ils and basin_hopping: 100 steps in a row that spend nothing end
//     the walk (as RST-8, SRCH-9); mlsl: an iteration that spends nothing
//     ends it (its sample would grow without end); basin_hopping's step grows
//     to the box's width at most (a larger one is only clipped more, SRCH-4).
//     With the minima on the box's bounds (CEC 2013 F1) the step grew without
//     end, every point was clipped onto a known one, and the walk went on
//     below its budget without evaluating anything.
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

namespace search {

using strategy::Inner;
using strategy::StartSpec;

constexpr double kPi = 3.14159265358979323846;

struct Found {
    Vec x;
    double f = std::numeric_limits<double>::infinity();
    double cv = std::numeric_limits<double>::infinity();
    bool ok = false;
};

inline Found found(const Result& r)
{
    Found b;
    if (r.objectives.empty()) return b;
    b.x = r.variables[0];
    b.f = r.objectives[0][0];
    b.cv = r.cv.empty() ? 0.0 : r.cv[0];
    b.ok = true;
    return b;
}

// SRCH-8
inline bool better(const Found& a, const Found& b)
{
    if (!a.ok) return false;
    if (!b.ok) return true;
    if (a.cv != b.cv) return a.cv < b.cv;
    return a.f < b.f;
}

inline long long left(const Evaluator& ev) { return ev.budget - ev.spent; }

// Γ(1 + n/2): (n/2)! for n even, (n/2)(n/2 - 1) ... (1/2) √π for n odd
inline double gamma_one_plus_half(std::size_t n)
{
    double g = n % 2 == 0 ? 1.0 : std::sqrt(kPi);
    for (double a = static_cast<double>(n) / 2.0; a > 0.0; a -= 1.0) g *= a;
    return g;
}

// A round of starts with their allowances: start_budget each, else what is
// left shared among them. Returns their answers ({} when no budget is left).
inline std::vector<Result> starts(Evaluator& ev, const Inner& inner, std::vector<StartSpec> s, long long cap,
                                  int threads, int pop_size)
{
    const long long rest = left(ev);
    if (rest <= 0 || s.empty()) return {};
    const long long each = cap > 0 ? std::min(cap, rest) : std::max(1LL, rest / static_cast<long long>(s.size()));
    long long given = 0;
    for (auto& sp : s) {
        sp.allowance = std::min(each, rest - given);
        given += sp.allowance;
    }
    while (!s.empty() && s.back().allowance <= 0) s.pop_back();
    return strategy::round(ev, inner, s, threads, pop_size);
}

inline Start at(const Vec& x)
{
    Start st;
    st.has_x0 = true;
    st.x0 = x;
    return st;
}

inline double width(const Problem& P, std::size_t j) { return P.hi[j] - P.lo[j]; }

inline void finite_box(const std::string& name, const Problem& P)
{
    for (std::size_t j = 0; j < P.lo.size(); ++j)
        if (!std::isfinite(P.lo[j]) || !std::isfinite(P.hi[j]))
            throw std::invalid_argument(name + " needs a box: every bound must be finite");
}

inline Result finish(Evaluator& ev, strategy::StrategyKnobs& k, const strategy::MinimaLog& log,
                     std::vector<std::pair<std::string, std::string>> extra = {})
{
    if (!ev.recorded) ev.record();
    ev.over = true;
    const std::vector<std::size_t> idx = ev.answer();
    ev.record(true);
    Result r = detail::result_of(ev, idx);
    r.ignored = k.ignored();
    r.settings = k.settings;
    for (auto& e : extra) r.settings.push_back(std::move(e));
    r.settings.push_back({"minima", log.json_text()});
    r.settings.push_back({"method_version", std::to_string(METHOD_VERSION)});
    return r;
}

}   // namespace search

// ── mlsl ────────────────────────────────────────────────────────────────────
inline Result run_mlsl(const Problem& P, double eq_tol, bool has_tol0, double eq_tol0, int pop_size, long long budget,
                       unsigned seed, const Knobs& knobs, Observer observer, int record_every)
{
    using namespace strategy;
    StrategyKnobs k("mlsl", knobs,
                    {"inner", "mlsl_n", "mlsl_sigma", "mlsl_gamma", "mlsl_tau", "mlsl_nu", "threads", "start_budget",
                     "min_dist", "min_df", "keep_points", "mlsl_stop", "minima_bayes"});
    const Inner inner = k.inner("inner");
    const long long N = k.whole("mlsl_n", 100, 1);
    const double sigma = k.number("mlsl_sigma", 4.0), gamma = k.number("mlsl_gamma", 0.2);
    const double tau = k.number("mlsl_tau", 0.0), nu = k.number("mlsl_nu", 0.0);
    const long long threads = k.whole("threads", 1, 1), cap = k.whole("start_budget", 0, 0);
    const double min_dist = k.number("min_dist", 1e-3), min_df = k.number("min_df", 1e-8);
    if (!(sigma > 0.0) || !(gamma > 0.0 && gamma <= 1.0))
        throw std::invalid_argument("mlsl: mlsl_sigma must be > 0 and mlsl_gamma in (0, 1]");
    const bool bayes_stop =                                                          // SRCH-13
        k.find("mlsl_stop") && k.word("mlsl_stop", "budget", {"budget", "bayes"}) == "bayes";
    const bool bayes = bayes_of(k);                                                  // strategy.hpp STRAT-13
    long long reduced_last = 0;
    search::finite_box("mlsl", P);
    const std::size_t n = P.lo.size();
    const double dn = static_cast<double>(n);
    Evaluator ev(P, eq_tol, has_tol0, eq_tol0, budget, pop_size, std::move(observer), record_every);
    ev.lean = lean_of(k);                                                            // STRAT-12
    MinimaLog log(P.lo, P.hi, min_dist, min_df);
    strategy::group_by_symmetry(k, P, log);
    std::mt19937 sampler(seed);
    std::vector<Vec> U;                       // the sample, normalized
    std::vector<double> f;                    // its values (failed or infeasible: +inf)
    std::vector<char> used;
    std::vector<Vec> minima;                  // X*, normalized
    long long searches = 0, iterations = 0;
    try {
        for (long long it = 1;; ++it) {
            const double before = ev.used();
            for (long long s = 0; s < N; ++s) {                                // the sample grows
                Vec u(n), x(n);
                for (std::size_t j = 0; j < n; ++j) {
                    u[j] = rnd::uniform01(sampler);
                    x[j] = P.lo[j] + u[j] * search::width(P, j);
                }
                const std::size_t i = ev.index(x);
                U.push_back(u);
                f.push_back(ev.failed[i] || ev.cv(i) > 0.0 ? std::numeric_limits<double>::infinity() : ev.F[i][0]);
                used.push_back(0);
            }
            if (!ev.recorded) ev.record();
            iterations = it;
            const double kN = static_cast<double>(U.size());
            const double rk = dm::pow(search::gamma_one_plus_half(n) * sigma * dm::log(kN) / kN, 1.0 / dn) /
                              std::sqrt(search::kPi);                                                   // SRCH-1
            std::vector<std::size_t> order(U.size());
            for (std::size_t i = 0; i < order.size(); ++i) order[i] = i;
            std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) { return f[a] < f[b]; });
            const std::size_t reduced = std::max<std::size_t>(1, static_cast<std::size_t>(gamma * kN));
            std::vector<StartSpec> run;
            std::vector<std::size_t> from;
            for (std::size_t q = 0; q < reduced && q < order.size(); ++q) {
                const std::size_t i = order[q];
                if (used[i] || !std::isfinite(f[i])) continue;
                bool excluded = false;                                                         // Q_τ, X*_ν
                for (std::size_t c = 0; c < n && !excluded; ++c) excluded = std::min(U[i][c], 1.0 - U[i][c]) < tau;
                for (std::size_t q2 = 0; q2 < minima.size() && !excluded; ++q2) {
                    double d2 = 0.0;
                    for (std::size_t c = 0; c < n; ++c) d2 += (minima[q2][c] - U[i][c]) * (minima[q2][c] - U[i][c]);
                    excluded = d2 < nu * nu;
                }
                if (excluded) continue;
                bool lower_near = false;
                for (std::size_t j = 0; j < U.size() && !lower_near; ++j) {
                    if (!(f[j] < f[i])) continue;
                    double d2 = 0.0;
                    for (std::size_t c = 0; c < n; ++c) d2 += (U[j][c] - U[i][c]) * (U[j][c] - U[i][c]);
                    lower_near = d2 <= rk * rk;
                }
                if (lower_near) continue;
                used[i] = 1;
                StartSpec sp;
                sp.seed = start_seed(seed, static_cast<std::uint64_t>(1 + searches + static_cast<long long>(run.size())));
                Vec x(n);
                for (std::size_t c = 0; c < n; ++c) x[c] = P.lo[c] + U[i][c] * search::width(P, c);
                sp.start = search::at(x);
                run.push_back(sp);
                from.push_back(i);
            }
            const std::vector<Result> res = search::starts(ev, inner, run, cap, static_cast<int>(threads), pop_size);
            for (std::size_t q = 0; q < res.size(); ++q) {
                const search::Found b = search::found(res[q]);
                if (!b.ok) continue;
                log.add(b.x, b.f, searches + static_cast<long long>(q));
                Vec u(n);
                for (std::size_t c = 0; c < n; ++c) u[c] = (b.x[c] - P.lo[c]) / search::width(P, c);
                minima.push_back(u);
            }
            searches += static_cast<long long>(res.size());
            reduced_last = static_cast<long long>(reduced);
            if (bayes_stop) {                                                                    // SRCH-13
                const double w = static_cast<double>(log.list().size()), gN = static_cast<double>(reduced);
                if (w >= 1.0 && gN > w + 2.0 && w * (gN - 1.0) / (gN - w - 2.0) <= w + 0.5) break;
            }
            if (search::left(ev) <= 0 || ev.used() == before) break;                             // SRCH-10
        }
    } catch (const BudgetSpent&) {
    }
    std::vector<std::pair<std::string, std::string>> extra{{"iterations", std::to_string(iterations)},
                                                           {"local_searches", std::to_string(searches)}};
    if (bayes)                                                                                    // STRAT-13
        for (auto& e : bayes_estimates(static_cast<long long>(log.list().size()), reduced_last)) extra.push_back(e);
    return search::finish(ev, k, log, extra);
}

// ── ils, basin_hopping ──────────────────────────────────────────────────────
inline Result run_walk(const std::string& name, const Problem& P, double eq_tol, bool has_tol0, double eq_tol0,
                       int pop_size, long long budget, unsigned seed, const Knobs& knobs, Observer observer,
                       int record_every, const Start& start)
{
    using namespace strategy;
    const bool bh = name == "basin_hopping";
    std::vector<std::string> own{"inner", "start_budget", "min_dist", "min_df", "keep_points"};
    if (bh) own.insert(own.end(), {"bh_temperature", "bh_step", "bh_accept_rate", "bh_interval"});
    else own.insert(own.end(), {"ils_accept", "ils_step", "ils_restart", "ils_temperature"});
    StrategyKnobs k(name, knobs, own);
    const Inner inner = k.inner("inner");
    const long long cap = k.whole("start_budget", 0, 0);
    const double min_dist = k.number("min_dist", 1e-3), min_df = k.number("min_df", 1e-8);
    std::string accept = "lsmc";
    double step = 0.05, T = 1.0, target = 0.5;
    long long interval = 10, restart_after = 0;
    if (bh) {
        T = k.number("bh_temperature", 1.0);
        step = k.number("bh_step", 0.05);
        target = k.number("bh_accept_rate", 0.5);
        interval = k.whole("bh_interval", 10, 1);
    } else {
        accept = k.word("ils_accept", "better", {"better", "rw", "restart", "lsmc"});
        step = k.number("ils_step", 0.1);
        restart_after = k.whole("ils_restart", 10, 1);
        T = k.number("ils_temperature", 1.0);
    }
    if (!(step > 0.0) || !(T > 0.0)) throw std::invalid_argument(name + ": the step and the temperature must be > 0");
    search::finite_box(name, P);
    const std::size_t n = P.lo.size();
    Evaluator ev(P, eq_tol, has_tol0, eq_tol0, budget, pop_size, std::move(observer), record_every);
    ev.lean = lean_of(k);                                                            // STRAT-12
    MinimaLog log(P.lo, P.hi, min_dist, min_df);
    strategy::group_by_symmetry(k, P, log);
    std::mt19937 rng(start_seed(seed, 0x9E3779B9u));             // the walk's own draws
    long long steps = 0, accepted_all = 0;
    auto local = [&](const Start& st, long long s) {
        StartSpec sp;
        sp.seed = start_seed(seed, static_cast<std::uint64_t>(s));
        sp.start = st;
        const std::vector<Result> r = search::starts(ev, inner, {sp}, cap, 1, pop_size);
        const search::Found b = r.empty() ? search::Found() : search::found(r[0]);
        if (b.ok) log.add(b.x, b.f, s);
        return b;
    };
    try {
        search::Found cur = local(start, 0), best = cur;
        long long since = 0, accepted = 0, tried = 0, idle = 0;
        for (long long s = 1; cur.ok && search::left(ev) > 0 && idle < 100; ++s) {                  // SRCH-10
            const double before = ev.used();
            Vec y = cur.x;
            for (std::size_t j = 0; j < n; ++j) {                                                // SRCH-4
                const double moved = y[j] + (2.0 * rnd::uniform01(rng) - 1.0) * step * search::width(P, j);
                y[j] = periodic::is(P.periodic, j) ? periodic::wrap_into(moved, P.lo[j], P.hi[j])  // PER-2
                                                   : std::min(P.hi[j], std::max(P.lo[j], moved));
            }
            const double u = rnd::uniform01(rng);
            const search::Found next = local(search::at(y), s);
            if (!next.ok) break;
            ++steps;
            bool take;
            if (accept == "rw") take = true;
            else if (accept == "lsmc")
                take = search::better(next, cur) ||
                       (next.cv == cur.cv && u < dm::exp((cur.f - next.f) / T));
            else take = search::better(next, cur);
            ++tried;
            if (take) {
                ++accepted;
                ++accepted_all;
                cur = next;
            }
            if (search::better(next, best)) {
                best = next;
                since = 0;
            } else if (accept == "restart" && ++since > restart_after) {                         // (3)
                Vec x(n);
                for (std::size_t j = 0; j < n; ++j) x[j] = P.lo[j] + rnd::uniform01(rng) * search::width(P, j);
                const search::Found fresh = local(search::at(x), ++s);
                if (fresh.ok) cur = fresh;
                since = 0;
            }
            if (bh && tried == interval) {                                                       // SRCH-5
                step = static_cast<double>(accepted) / static_cast<double>(tried) > target ? step / 0.9 : step * 0.9;
                step = std::min(step, 1.0);                                                       // SRCH-10
                accepted = tried = 0;
            }
            idle = ev.used() == before ? idle + 1 : 0;
        }
    } catch (const BudgetSpent&) {
    }
    std::vector<std::pair<std::string, std::string>> extra{{"steps", std::to_string(steps)},
                                                           {"accepted", std::to_string(accepted_all)}};
    if (bh) extra.push_back({"bh_step_end", detail::g_format(step)});
    return search::finish(ev, k, log, extra);
}

// ── halving ─────────────────────────────────────────────────────────────────
inline Result run_halving(const Problem& P, double eq_tol, bool has_tol0, double eq_tol0, int pop_size,
                          long long budget, unsigned seed, const Knobs& knobs, Observer observer, int record_every,
                          const Start& start)
{
    using namespace strategy;
    StrategyKnobs k("halving", knobs,
                    {"inner", "n_starts", "start_points", "threads", "min_dist", "min_df", "keep_points"});
    if (lean_of(k))                                                                  // STRAT-12
        throw std::invalid_argument("halving: keep_points = best is refused: its arms' reruns are free only "
                                    "through the archive of every point (SRCH-6)");
    const Inner inner = k.inner("inner");
    const long long n_arms = k.whole("n_starts", 16, 1);
    const std::string points = k.word("start_points", "sobol", {"sobol", "lhs", "uniform", "halton_rand", "sobol_owen"});
    const long long threads = k.whole("threads", 1, 1);
    const double min_dist = k.number("min_dist", 1e-3), min_df = k.number("min_df", 1e-8);
    const std::size_t n = P.lo.size();
    if (n_arms > 1) search::finite_box("halving", P);
    Evaluator ev(P, eq_tol, has_tol0, eq_tol0, budget, pop_size, std::move(observer), record_every);
    MinimaLog log(P.lo, P.hi, min_dist, min_df);
    strategy::group_by_symmetry(k, P, log);
    const std::vector<double> D =
        n_arms > 1 ? designs::design(points, static_cast<std::size_t>(n_arms - 1), static_cast<int>(n), seed)
                   : std::vector<double>();
    struct Arm {
        StartSpec spec;
        search::Found best;
        long long index = 0;
    };
    std::vector<Arm> arms(static_cast<std::size_t>(n_arms));
    for (std::size_t a = 0; a < arms.size(); ++a) {
        arms[a].index = static_cast<long long>(a);
        arms[a].spec.seed = start_seed(seed, a);
        if (a == 0) {
            arms[a].spec.start = start;
        } else {
            Vec x(n);
            for (std::size_t j = 0; j < n; ++j) x[j] = P.lo[j] + D[(a - 1) * n + j] * search::width(P, j);
            arms[a].spec.start = search::at(x);
        }
    }
    long long rounds = 1;
    while ((1LL << rounds) < n_arms) ++rounds;                       // ⌈log2 n⌉, at least 1
    json::Value plan = json::Value::array();
    try {
        for (long long r = 0; r < rounds && !arms.empty(); ++r) {
            const long long more = budget / (static_cast<long long>(arms.size()) * rounds);
            if (more <= 0) break;
            std::vector<StartSpec> specs;
            const long long fresh = std::min(more, search::left(ev) / static_cast<long long>(arms.size()));
            if (fresh <= 0) break;
            for (Arm& a : arms) {
                a.spec.allowance += more;                                                         // SRCH-6
                a.spec.fresh = fresh;
                specs.push_back(a.spec);
            }
            const std::vector<Result> res = strategy::round(ev, inner, specs, static_cast<int>(threads), pop_size);
            for (std::size_t a = 0; a < arms.size(); ++a) arms[a].best = search::found(res[a]);
            json::Value row = json::Value::array();
            row.items.push_back(json::Value::integer(static_cast<std::int64_t>(arms.size())));
            row.items.push_back(json::Value::integer(arms[0].spec.allowance));
            plan.items.push_back(row);
            if (r + 1 == rounds) break;
            std::stable_sort(arms.begin(), arms.end(),
                             [](const Arm& x, const Arm& y) { return search::better(x.best, y.best); });
            arms.resize((arms.size() + 1) / 2);
        }
    } catch (const BudgetSpent&) {
    }
    for (const Arm& a : arms)
        if (a.best.ok) log.add(a.best.x, a.best.f, a.index);
    return search::finish(ev, k, log, {{"rounds", json::dump(plan, -1, ",", ":")}});
}

// ── memetic ─────────────────────────────────────────────────────────────────
inline Result run_memetic(const Problem& P, double eq_tol, bool has_tol0, double eq_tol0, int pop_size,
                          long long budget, unsigned seed, const Knobs& knobs, Observer observer, int record_every,
                          const Start& start)
{
    using namespace strategy;
    StrategyKnobs k("memetic", knobs,
                    {"inner", "local_solver", "memetic_every", "memetic_points", "memetic_pick", "local_share",
                     "start_budget", "threads", "min_dist", "min_df", "lamarck", "lamarck_share", "keep_points"});
    if (lean_of(k))                                                                  // STRAT-12
        throw std::invalid_argument("memetic: keep_points = best is refused: its global reruns are free only "
                                    "through the archive of every point, and its local starts are drawn from it "
                                    "(SRCH-7)");
    const Inner global = k.inner("inner");
    const Inner local = k.inner("local_solver");
    const long long every = k.whole("memetic_every", 1000, 1), n_points = k.whole("memetic_points", 1, 1);
    const std::string pick = k.word("memetic_pick", "best", {"best", "spread"});
    const double share = k.number("local_share", 0.2);
    const long long cap = k.whole("start_budget", 0, 0), threads = k.whole("threads", 1, 1);
    const double min_dist = k.number("min_dist", 1e-3), min_df = k.number("min_df", 1e-8);
    if (!(share > 0.0 && share < 1.0)) throw std::invalid_argument("memetic: local_share must be in (0, 1)");
    const std::string lamarck =                                                                 // SRCH-12
        k.find("lamarck") ? k.word("lamarck", "off", {"off", "on", "share"}) : std::string("off");
    const double lamarck_share = lamarck == "share" ? k.number("lamarck_share", 0.25) : 1.0;
    if (!(lamarck_share >= 0.0 && lamarck_share <= 1.0))
        throw std::invalid_argument("memetic: lamarck_share must be in [0, 1]");
    std::mt19937 coins(start_seed(seed, std::uint64_t{1} << 32));
    std::vector<Point> written;                       // the local searches' answers, the newest first
    Inner seeded = global;
    if (lamarck != "off") seeded.knobs.push_back({"warm_start", Knob{true, "on", 0.0}});
    const std::size_t n = P.lo.size();
    Evaluator ev(P, eq_tol, has_tol0, eq_tol0, budget, pop_size, std::move(observer), record_every);
    MinimaLog log(P.lo, P.hi, min_dist, min_df);
    strategy::group_by_symmetry(k, P, log);
    StartSpec g;
    g.seed = seed;
    g.start = start;
    std::vector<Vec> started;
    long long phases = 0, searches = 0;
    auto scale = [&](std::size_t j) {
        const double w = search::width(P, j);
        return std::isfinite(w) && w > 0.0 ? w : 1.0;
    };
    try {
        while (search::left(ev) > 0) {
            const double before = ev.used();
            g.fresh = std::min(every, search::left(ev));                                          // SRCH-7
            g.allowance += g.fresh;
            if (!written.empty()) {                                                               // SRCH-12
                g.start = start;
                g.start.has_seed_points = true;
                g.start.seed_points = written;
                if (start.has_x0 && ev.known(start.x0)) {     // x0, if evaluated: the oldest seed
                    const std::size_t i = ev.index(start.x0);
                    g.start.seed_points.push_back(Point{ev.X[i], ev.F[i], ev.G[i]});
                }
                g.start.has_x0 = false;
            }
            strategy::round(ev, lamarck == "off" ? global : seeded, {g}, 1, pop_size);
            const bool global_done = ev.used() == before;                                         // SRCH-9
            ++phases;
            // the local searches' starts: the archive's best, or the best and then the farthest among the best tenth
            std::vector<std::size_t> order;
            for (std::size_t i = 0; i < ev.X.size(); ++i)
                if (!ev.failed[i]) order.push_back(i);
            std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
                const double ca = ev.cv(a), cb = ev.cv(b);
                return ca != cb ? ca < cb : ev.F[a][0] < ev.F[b][0];
            });
            if (pick == "spread") order.resize(std::max<std::size_t>(1, order.size() / 10));
            std::vector<StartSpec> specs;
            auto far = [&](std::size_t i) {
                double d = std::numeric_limits<double>::infinity();
                for (const Vec& s : started) {
                    double m = 0.0;
                    for (std::size_t j = 0; j < n; ++j) m = std::max(m, std::fabs(ev.X[i][j] - s[j]) / scale(j));
                    d = std::min(d, m);
                }
                return d;
            };
            for (std::size_t c = 0; c < order.size() && static_cast<long long>(specs.size()) < n_points; ++c) {
                std::size_t i = order[c];
                if (pick == "spread" && !specs.empty()) {
                    double best_d = -1.0;
                    for (std::size_t q : order) {
                        const double d = far(q);
                        if (d > best_d) {
                            best_d = d;
                            i = q;
                        }
                    }
                }
                bool again = false;
                for (const Vec& s : started) again = again || s == ev.X[i];
                if (again) continue;
                started.push_back(ev.X[i]);
                StartSpec sp;
                sp.seed = start_seed(seed, static_cast<std::uint64_t>(1 + searches + static_cast<long long>(specs.size())));
                sp.start = search::at(ev.X[i]);
                specs.push_back(sp);
            }
            const long long share_cap =
                std::max(1LL, static_cast<long long>(share / (1.0 - share) * static_cast<double>(every)) /
                                  std::max<long long>(1, static_cast<long long>(specs.size())));
            const std::vector<Result> res = search::starts(ev, local, specs, cap > 0 ? std::min(cap, share_cap) : share_cap,
                                                           static_cast<int>(threads), pop_size);
            for (std::size_t q = 0; q < res.size(); ++q) {
                const search::Found b = search::found(res[q]);
                if (b.ok) log.add(b.x, b.f, searches + static_cast<long long>(q));
                if (lamarck == "off" || !b.ok) continue;                                          // SRCH-12
                if (lamarck == "share" && !(rnd::uniform01(coins) < lamarck_share)) continue;
                const std::size_t i = ev.index(b.x);                                             // in the archive
                written.insert(written.begin(), Point{ev.X[i], ev.F[i], ev.G[i]});
            }
            searches += static_cast<long long>(res.size());
            if (global_done) break;                                                               // SRCH-9
        }
    } catch (const BudgetSpent&) {
    }
    return search::finish(ev, k, log, {{"phases", std::to_string(phases)}, {"local_searches", std::to_string(searches)}});
}

}   // namespace mootation::scalar
