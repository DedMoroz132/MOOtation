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
// shared among the round's starts), min_dist, min_df (the minima log), and
// the stop rule's stall, max_steps, budget_units (stop_rule.hpp; what is
// left is Evaluator::left(), STOP-1).
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
//   SRCH-14. mlsl_cells = on (task 24, §24.2.5; off by default): the check
//     "a sample point of lower f within r_k" looks in the cell of the point
//     and its 3^n neighbours only — the sample in cubes of side r_k (1 +
//     1e-9) of the normalized box, rebuilt each iteration (part II, §3, the
//     static form; not Bentley's dynamic one) — with the full scan's test,
//     term by term: any point within r_k differs by less than a side in
//     every coordinate, so the decisions are the full scan's. n at most 10
//     (3^10 cells a point).
//   SRCH-9. memetic: a phase whose global run spends nothing is the last (as
//     RST-8): the global solver has stopped by its own criteria (rerun, it
//     evaluates nothing new), and the local searches alone would go on in
//     phases of a few evaluations each (5380 phases of a budget of 20000 on a
//     linear f with its minimum in the box's corner), or of none at all once
//     every archive point has been a local start.
//   SRCH-15. step_units (task 25, §24.2.4; box by default, the old move):
//     absolute moves every coordinate by U[-1, 1] times the step in the
//     problem's own units instead of times the box's width.
//   SRCH-16. Acceptance by a threshold (ils_accept = threshold, bh_accept =
//     threshold; task 25, §24.2.4): X. Lai, J.-K. Hao, R. Yue, Z. Lü, Z.-H.
//     Fu, EJOR 299 (2022), §3.4, Algorithm 3 (source iterated-dynamic-
//     thresholding_lai2021, DTS): the new minimum is taken when dE = E(new) -
//     E(current) < Th and dE != 0; then Th <- mu Th when more were taken than
//     refused so far, else Th <- Th / mu; Th_0 = walk_th0 (1e-4), mu =
//     walk_th_mu (0.75). The
//     counts run over the whole walk (DTS-1, as Algorithm 3). Points of
//     different violation are compared by feasibility first (SRCH-8).
//     A new minimum equal to the current one (dE = 0) is a refusal, as in
//     Algorithm 3: a step so small that the local search mostly returns to
//     the current minimum drives Th up without end (basin_hopping's shrinking
//     step does that on Rastrigin) — Lai et al.'s move is a fixed one.
//   SRCH-17. Metropolis on a relative scale (ils_accept = lsmc_rel, bh_accept
//     = metropolis_rel; task 25): the first walk_t0_trials (10) steps accept
//     the better only and record |dE| != 0; then T = walk_t_rel (1) times
//     their median, fixed for the rest of the walk; with no nonzero |dE|
//     recorded, T = 1e-12 (1 + |f|) of the current minimum.
//   SRCH-18. The strength ladder (task 25, §24.2.4; IDLS, Lai et al. 2025,
//     source structural-optimization-atomic_lai2025, Algorithm 1): with
//     ladder_max > 0 the step is ladder_min after a new
//     best, else grows by ladder_step, back to ladder_min past ladder_max;
//     ladder_min = ladder_max, ladder_step = 0 is a constant step. With the
//     ladder basin_hopping's step adaptation (SRCH-5) is off.
//   SRCH-19. perturb = user (task 25, §24.2.4; task 24's Problem::perturb,
//     §24.1.3): the move is the problem's, perturb(y, step, rng) on a copy of
//     the current minimum with the walk's own generator and the step (the
//     ladder's where there is one) as its strength; the result is clipped to
//     the box, a periodic coordinate wrapped (PER-2), as the builtin move. A
//     hook that repeats the builtin move gives the builtin run bit for bit.
//   SRCH-22. Off by default. step = box | user | portfolio (task 31, layer 2; moves.hpp
//     MOVE-1 ... MOVE-5; box by default, the moves above): user the problem's
//     mutate hook, portfolio its mutate_ops by op_select; then the step's
//     knobs and its adaptation are off (MOVE-1). Problem::on_child is told
//     every step's child after the acceptance (taken: the walk moved to it),
//     the restart's uniform point not being a child. bh_temperature_schedule,
//     ils_temperature_schedule = const | geometric and *_end anneal the
//     Metropolis temperature (stop_rule.hpp STOP-7; bh_accept = metropolis,
//     ils_accept = lsmc only); the settings then say the last step's
//     temperature (*_reached).
//   SRCH-20. basin_hopping with bh_accept = threshold and its adaptive step
//     (no ladder) warns once (mootation::warn): DTS is meant for a fixed move
//     (SRCH-16); the run itself is unchanged.
//   SRCH-21. memetic on a problem with decode (task 24, §24.1.1, §24.1.2;
//     off by default): a builtin local solver searches only the variables
//     its start's genome has active (hooks.hpp active_of; the subproblem
//     restricted(P, x, active), RES-1, its points through the run's
//     evaluator and embedded back), so it does not move a structure's slots;
//     a start with no active variable, or that does not decode, is skipped.
//     The user's local_search gets the whole x and the mask instead
//     (scalar.hpp ULS-7).
//   SRCH-23 (only when asked: the resumable API). ils and basin_hopping are
//     resumable (task 31, layer 3; resumable.hpp): class Walk keeps the
//     walk's state between pieces, a step one pass of the loop above;
//     run_walk is Walk in one piece. Its member is the current minimum
//     (admit sets it, planted), its best_points the minima log's lowest,
//     its operators the step (bh_step) and the move's portfolio.
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
#include <functional>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "../warn.hpp"
#include "hooks.hpp"
#include "resumable.hpp"
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

// what the strategy may still spend (stop_rule.hpp STOP-1)
inline long long left(const Evaluator& ev) { return ev.left(); }

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
    const long long each = cap > 0 ? (std::min)(cap, rest) : (std::max)(1LL, rest / static_cast<long long>(s.size()));
    long long given = 0;
    for (auto& sp : s) {
        sp.allowance = (std::min)(each, rest - given);
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
                     std::vector<std::pair<std::string, std::string>> extra = {}, const std::string& stopped_by = "")
{
    if (!ev.recorded) ev.record();
    ev.over = true;
    const std::vector<std::size_t> idx = ev.answer();
    ev.record(true);
    Result r = detail::result_of(ev, idx);
    r.ignored = k.ignored();
    r.settings = k.settings;
    for (auto& e : extra) r.settings.push_back(std::move(e));
    if (k.find("round_stop")) r.settings.push_back({"round_skipped", std::to_string(ev.round_skipped)});   // STRAT-16
    if (log.on()) r.settings.push_back({"minima", log.json_text()});                // STRAT-15
    r.settings.push_back({"method_version", std::to_string(METHOD_VERSION)});
    r.stopped_by = stopped_by;                                                      // stop_rule.hpp STOP-4
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
                     "min_dist", "min_df", "keep_points", "mlsl_stop", "minima_bayes", "mlsl_cells"});
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
    const bool cells = k.find("mlsl_cells") && k.word("mlsl_cells", "off", {"off", "on"}) == "on";   // SRCH-14
    if (cells && P.lo.size() > 10) throw std::invalid_argument("mlsl: mlsl_cells for n <= 10 (3^n cells a point)");
    search::finite_box("mlsl", P);
    const std::size_t n = P.lo.size();
    const double dn = static_cast<double>(n);
    Evaluator ev(P, eq_tol, has_tol0, eq_tol0, budget, pop_size, std::move(observer), record_every);
    strategy::round_stop_of(k, ev);                                                  // STRAT-16
    StopRule stop = strategy::stop_of(k, ev);                                        // stop_rule.hpp: a step an iteration
    std::string own = "rule";
    ev.lean = lean_of(k);                                                            // STRAT-12
    MinimaLog log(P.lo, P.hi, min_dist, min_df);
    strategy::group_by_symmetry(k, P, log);
    strategy::whole_log(log, bayes || bayes_stop, "mlsl: minima_bayes = on or mlsl_stop = bayes");
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
            // SRCH-14: the sample by cell, a cell's key its integer coordinates' bytes
            const double side = rk * (1.0 + 1e-9);
            std::unordered_map<std::string, std::vector<std::size_t>> grid;
            auto cell_of = [&](const Vec& u) {
                std::vector<long long> c(n);
                for (std::size_t w = 0; w < n; ++w) c[w] = static_cast<long long>(std::floor(u[w] / side));
                return c;
            };
            auto key_of_cell = [](const std::vector<long long>& c) {
                return std::string(reinterpret_cast<const char*>(c.data()), c.size() * sizeof(long long));
            };
            if (cells)
                for (std::size_t j = 0; j < U.size(); ++j) grid[key_of_cell(cell_of(U[j]))].push_back(j);
            std::vector<StartSpec> run;
            std::vector<std::size_t> from;
            for (std::size_t q = 0; q < reduced && q < order.size(); ++q) {
                const std::size_t i = order[q];
                if (used[i] || !std::isfinite(f[i])) continue;
                bool excluded = false;                                                         // Q_τ, X*_ν
                for (std::size_t c = 0; c < n && !excluded; ++c) excluded = (std::min)(U[i][c], 1.0 - U[i][c]) < tau;
                for (std::size_t q2 = 0; q2 < minima.size() && !excluded; ++q2) {
                    double d2 = 0.0;
                    for (std::size_t c = 0; c < n; ++c) d2 += (minima[q2][c] - U[i][c]) * (minima[q2][c] - U[i][c]);
                    excluded = d2 < nu * nu;
                }
                if (excluded) continue;
                bool lower_near = false;
                auto near_lower = [&](std::size_t j) {         // the full scan's test, term by term
                    if (!(f[j] < f[i])) return false;
                    double d2 = 0.0;
                    for (std::size_t c = 0; c < n; ++c) d2 += (U[j][c] - U[i][c]) * (U[j][c] - U[i][c]);
                    return d2 <= rk * rk;
                };
                if (!cells) {
                    for (std::size_t j = 0; j < U.size() && !lower_near; ++j) lower_near = near_lower(j);
                } else {                                                                   // SRCH-14
                    const std::vector<long long> home = cell_of(U[i]);
                    std::vector<int> off(n, -1);
                    for (bool more = true; more && !lower_near;) {
                        std::vector<long long> c = home;
                        for (std::size_t w = 0; w < n; ++w) c[w] += off[w];
                        const auto hit = grid.find(key_of_cell(c));
                        if (hit != grid.end())
                            for (std::size_t j : hit->second)
                                if (near_lower(j)) {
                                    lower_near = true;
                                    break;
                                }
                        more = false;                          // the next offset in {-1, 0, 1}^n
                        for (std::size_t w = 0; w < n && !more; ++w) {
                            if (off[w] < 1) {
                                ++off[w];
                                more = true;
                            } else {
                                off[w] = -1;
                            }
                        }
                    }
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
            stop.step(ev);
            if (ev.used() == before) {                                                           // SRCH-10
                own = "idle";
                break;
            }
            if (!stop.go(ev)) break;
        }
    } catch (const BudgetSpent&) {
    }
    std::vector<std::pair<std::string, std::string>> extra{{"iterations", std::to_string(iterations)},
                                                           {"local_searches", std::to_string(searches)}};
    if (bayes)                                                                                    // STRAT-13
        for (auto& e : bayes_estimates(static_cast<long long>(log.list().size()), reduced_last)) extra.push_back(e);
    return search::finish(ev, k, log, extra, stop.stopped_by(ev, own));
}

// ── ils, basin_hopping ──────────────────────────────────────────────────────
// The walk as a resumable run (task 31, layer 3; resumable.hpp): the state of
// run_walk's loop kept between pieces, a step one pass of the loop. run_walk
// is Walk in one piece (RES-1).
class Walk : public Resumable {
public:
    std::unique_ptr<Problem> own;                       // RES-8: the run's copy of the problem, where it has one
    std::function<void(Result&)> tail;                  // ... and what scalar::run adds to the result

    Walk(const std::string& name, const Problem& P, double eq_tol, bool has_tol0, double eq_tol0, int pop_size,
         long long budget, unsigned seed, const Knobs& knobs, Observer observer, int record_every, const Start& start)
        : name_(name), bh_(name == "basin_hopping"), P_(P), knobs_(knobs), k_(name, knobs_, own_knobs(name)),
          seed_(seed), pop_size_(pop_size), start_(start)
    {
        using namespace strategy;
        inner_ = k_.inner("inner");
        cap_ = k_.whole("start_budget", 0, 0);
        const double min_dist = k_.number("min_dist", 1e-3), min_df = k_.number("min_df", 1e-8);
        // MOVE-1: the step's knobs only where the move has a strength (step = box, the default)
        const Knob* step_knob = k_.find("step");
        const bool scaled = step_knob == nullptr || (step_knob->is_word && step_knob->word == "box");
        if (bh_) {
            T_ = k_.number("bh_temperature", 1.0);
            if (scaled) {
                step_ = k_.number("bh_step", 0.05);
                target_ = k_.number("bh_accept_rate", 0.5);
                interval_ = k_.whole("bh_interval", 10, 1);
            }
            if (k_.find("bh_accept")) {                                                   // SRCH-16, SRCH-17
                const std::string a = k_.word("bh_accept", "metropolis", {"metropolis", "threshold", "metropolis_rel"});
                accept_ = a == "metropolis" ? "lsmc" : a == "threshold" ? "threshold" : "lsmc_rel";
            }
        } else {
            accept_ = k_.word("ils_accept", "better", {"better", "rw", "restart", "lsmc", "threshold", "lsmc_rel"});
            if (scaled) step_ = k_.number("ils_step", 0.1);
            restart_after_ = k_.whole("ils_restart", 10, 1);
            T_ = k_.number("ils_temperature", 1.0);
        }
        if (!(step_ > 0.0) || !(T_ > 0.0)) throw std::invalid_argument(name + ": the step and the temperature must be > 0");
        // task 25, §24.2.4: each knob in the settings only where given
        const bool absolute =
            scaled && k_.find("step_units") && k_.word("step_units", "box", {"box", "absolute"}) == "absolute";
        th_ = k_.find("walk_th0") ? k_.number("walk_th0", 1e-4) : 1e-4;                      // SRCH-16
        th_mu_ = k_.find("walk_th_mu") ? k_.number("walk_th_mu", 0.75) : 0.75;
        t0_trials_ = k_.find("walk_t0_trials") ? k_.whole("walk_t0_trials", 10, 1) : 10;      // SRCH-17
        t_rel_ = k_.find("walk_t_rel") ? k_.number("walk_t_rel", 1.0) : 1.0;
        ladder_max_ = k_.find("ladder_max") ? k_.number("ladder_max", 0.0) : 0.0;             // SRCH-18
        ladder_min_ = k_.find("ladder_min") ? k_.number("ladder_min", ladder_max_) : ladder_max_;
        ladder_step_ = k_.find("ladder_step") ? k_.number("ladder_step", 0.0) : 0.0;
        ladder_ = ladder_max_ > 0.0;
        if (accept_ == "threshold" && !(th_ > 0.0 && th_mu_ > 0.0 && th_mu_ <= 1.0))
            throw std::invalid_argument(name + ": walk_th0 > 0 and walk_th_mu in (0, 1]");
        if (accept_ == "lsmc_rel" && !(t_rel_ > 0.0)) throw std::invalid_argument(name + ": walk_t_rel must be > 0");
        if (ladder_ && !(ladder_min_ > 0.0 && ladder_min_ <= ladder_max_ && ladder_step_ >= 0.0))
            throw std::invalid_argument(name + ": 0 < ladder_min <= ladder_max and ladder_step >= 0");
        if (ladder_) step_ = ladder_min_;
        const bool user_move = k_.find("perturb") && k_.word("perturb", "builtin", {"builtin", "user"}) == "user";   // SRCH-19
        if (user_move && !P.perturb) throw std::invalid_argument(name + ": perturb = user needs the problem's perturb hook");
        mv_ = moves::step_of(k_, P, name, user_move);                                 // the move (moves.hpp)
        mv_.absolute = absolute;
        if (ladder_ && !mv_.scaled())                                                 // MOVE-1
            throw std::invalid_argument(name + ": the strength ladder needs a move with a strength (step = box)");
        if (bh_ && accept_ == "threshold" && !ladder_ && mv_.scaled())                  // SRCH-20
            mootation::warn("basin_hopping: bh_accept = threshold (DTS) is meant for a fixed move; the adaptive step "
                            "(bh_accept_rate) shrinks it until most new minima are the current one, which counts as a "
                            "refusal and drives the threshold up. A fixed step: ladder_min = ladder_max = the step.");
        search::finite_box(name, P);
        n_ = P.lo.size();
        ev_ = std::make_unique<Evaluator>(P, eq_tol, has_tol0, eq_tol0, budget, pop_size, std::move(observer),
                                          record_every);
        strategy::round_stop_of(k_, *ev_);                                            // STRAT-16
        stop_ = strategy::stop_of(k_, *ev_);                                          // stop_rule.hpp: a step of the walk
        ev_->lean = lean_of(k_);                                                      // STRAT-12
        log_ = std::make_unique<MinimaLog>(P.lo, P.hi, min_dist, min_df);
        strategy::group_by_symmetry(k_, P, *log_);
        t_knob_ = bh_ ? "bh_temperature" : "ils_temperature";                        // STOP-7: annealing
        anneal_ = schedule_of(k_, name, t_knob_, T_, *ev_);
        if (anneal_.geometric && accept_ != "lsmc")
            throw std::invalid_argument(name + ": " + t_knob_ + "_schedule is of the Metropolis acceptance's temperature (" +
                                        (bh_ ? "bh_accept = metropolis" : "ils_accept = lsmc") + ")");
        rng_.seed(start_seed(seed, 0x9E3779B9u));                                    // the walk's own draws
    }

    const std::string& name() const override { return name_; }

    void begin() override
    {
        if (begun_) return;
        begun_ = true;
        try {
            cur_ = local(start_, 0);
            best_ = cur_;
            stop_.mark(*ev_);
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
                step();
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
        if (!cur_.ok) return {};
        return {member_of(*ev_, cur_.x, cur_.f, cur_.cv)};
    }

    // RES-7: the walk's k best minima (its minima log), else its best
    std::vector<Member> best_points(std::size_t k) const override
    {
        std::vector<Member> out;
        if (log_->on()) {
            std::vector<const strategy::Minimum*> ms;
            for (const strategy::Minimum& m : log_->list()) ms.push_back(&m);
            std::stable_sort(ms.begin(), ms.end(), [](const strategy::Minimum* a, const strategy::Minimum* b) { return a->f < b->f; });
            for (std::size_t q = 0; q < ms.size() && out.size() < k; ++q)
                out.push_back(member_of(*ev_, ms[q]->x, ms[q]->f, 0.0));
        } else if (best_.ok && k > 0) {
            out.push_back(member_of(*ev_, best_.x, best_.f, best_.cv));
        }
        return out;
    }

    long long admit(const std::vector<Member>& in, Place how) override             // RES-3, RES-4: one member
    {
        if (in.empty()) return 0;
        const search::Found c = planted(in[0]);
        if (!c.ok || (how == Place::worst_if_better && !search::better(c, cur_))) return 0;
        cur_ = c;
        return 1;
    }

    long long admit(const std::vector<Member>& in, const PlaceRule& rule) override
    {
        long long taken = 0;
        for (const Member& m : in)
            if (rule(members(), m) == 0) {
                const search::Found c = planted(m);
                if (!c.ok) break;
                cur_ = c;
                ++taken;
            }
        return taken;
    }

    Operators operators() const override                                         // RES-6
    {
        Operators o;
        o.has_step = mv_.scaled();
        o.step = step_;
        o.mutation = mv_.ops;
        return o;
    }

    void set_operators(const Operators& o) override
    {
        if (o.has_step && mv_.scaled()) step_ = o.step;
        if (o.mutation.on()) {
            if (o.mutation.names != mv_.ops.names)
                throw std::invalid_argument(name_ + ": set_operators: a portfolio of other operators (RES-6)");
            mv_.ops = o.mutation;
        }
    }

    Member best() const override { return best_member(*ev_); }

    std::vector<std::pair<std::string, std::string>> summary() const override
    {
        std::vector<std::pair<std::string, std::string>> extra{{"steps", std::to_string(steps_)},
                                                               {"accepted", std::to_string(accepted_all_)}};
        if (bh_ && mv_.scaled()) extra.push_back({"bh_step_end", detail::g_format(step_)});      // MOVE-1
        if (accept_ == "threshold") extra.push_back({"walk_th_end", detail::g_format(th_)});
        if (accept_ == "lsmc_rel") extra.push_back({"walk_t_rel_end", detail::g_format(T_rel_)});
        if (anneal_.geometric) extra.push_back({t_knob_ + "_reached", detail::g_format(T_)});      // STOP-7
        if (mv_.ops.on()) extra.push_back({"operators", moves::operators_row(nullptr, &mv_.ops)});    // MOVE-5
        return extra;
    }

    Result result() override
    {
        over_ = true;
        Result r = search::finish(*ev_, k_, *log_, summary(), stop_.stopped_by(*ev_, idle_ >= 100 ? "idle" : "rule"));
        if (tail) tail(r);
        return r;
    }

private:
    std::string name_;
    bool bh_ = false;
    const Problem& P_;
    Knobs knobs_;
    strategy::StrategyKnobs k_;
    unsigned seed_ = 0;
    int pop_size_ = 1;
    Start start_;
    strategy::Inner inner_;
    long long cap_ = 0;
    std::string accept_ = "lsmc";
    double step_ = 0.05, T_ = 1.0, target_ = 0.5;
    long long interval_ = 10, restart_after_ = 0;
    double th_ = 1e-4, th_mu_ = 0.75, t_rel_ = 1.0;
    long long t0_trials_ = 10;
    double ladder_max_ = 0.0, ladder_min_ = 0.0, ladder_step_ = 0.0;
    bool ladder_ = false;
    moves::Move mv_;
    std::size_t n_ = 0;
    std::unique_ptr<Evaluator> ev_;
    strategy::StopRule stop_;
    long long idle_ = 0;
    std::unique_ptr<strategy::MinimaLog> log_;
    std::string t_knob_;
    strategy::Schedule anneal_;
    std::mt19937 rng_;
    long long steps_ = 0, accepted_all_ = 0;
    long long th_taken_ = 0, th_refused_ = 0;                                        // SRCH-16 (DTS-1)
    std::vector<double> t0_seen_;                                                    // SRCH-17
    double T_rel_ = 0.0;
    search::Found cur_, best_;
    long long since_ = 0, accepted_ = 0, tried_ = 0, s_ = 1;
    bool begun_ = false, over_ = false;

    static std::vector<std::string> own_knobs(const std::string& name)
    {
        const bool bh = name == "basin_hopping";
        std::vector<std::string> own{"inner", "start_budget", "min_dist", "min_df", "keep_points"};
        if (bh) own.insert(own.end(), {"bh_temperature", "bh_step", "bh_accept_rate", "bh_interval", "bh_accept"});
        else own.insert(own.end(), {"ils_accept", "ils_step", "ils_restart", "ils_temperature"});
        own.insert(own.end(), {"step_units", "walk_th0", "walk_th_mu", "walk_t0_trials", "walk_t_rel", "ladder_min",
                               "ladder_max", "ladder_step", "perturb"});                      // task 25, §24.2.4
        own.insert(own.end(), {"step", "op_select", "op_pmin", "op_alpha", "op_beta"});        // task 31, layer 2
        own.insert(own.end(), {bh ? "bh_temperature_schedule" : "ils_temperature_schedule",
                               bh ? "bh_temperature_end" : "ils_temperature_end"});          // STOP-7
        return own;
    }

    // the local search from st, the walk's s-th
    search::Found local(const Start& st, long long s)
    {
        strategy::StartSpec sp;
        sp.seed = strategy::start_seed(seed_, static_cast<std::uint64_t>(s));
        sp.start = st;
        const std::vector<Result> r = search::starts(*ev_, inner_, {sp}, cap_, 1, pop_size_);
        const search::Found b = r.empty() ? search::Found() : search::found(r[0]);
        if (b.ok) log_->add(b.x, b.f, s);
        return b;
    }

    // the loop's test before a step (SRCH-10)
    bool can_step() { return cur_.ok && idle_ < 100 && stop_.go(*ev_); }

    // one pass of the loop: a move, its local search, the acceptance
    void step()
    {
        const double before = ev_->used();
        Vec y = cur_.x;
        // SRCH-4, SRCH-15, SRCH-19, PER-2; MOVE-2
        const long long tag = mv_.apply(P_, y, step_, rng_, name_);
        if (tag >= 0 && mv_.ops.on()) ++mv_.ops.calls[static_cast<std::size_t>(tag)];        // MOVE-3
        const double u = rnd::uniform01(rng_);
        const search::Found next = local(search::at(y), s_);
        if (!next.ok) {
            over_ = true;
            return;
        }
        ++steps_;
        if (anneal_.geometric) T_ = anneal_.at(*ev_, stop_);                                   // STOP-7
        bool take;
        if (accept_ == "rw") take = true;
        else if (accept_ == "lsmc")
            take = search::better(next, cur_) || (next.cv == cur_.cv && u < dm::exp((cur_.f - next.f) / T_));
        else if (accept_ == "threshold") {                                                    // SRCH-16
            const double dE = next.f - cur_.f;
            take = next.cv != cur_.cv ? search::better(next, cur_) : (dE < th_ && dE != 0.0);
            (take ? th_taken_ : th_refused_) += 1;
            th_ = th_taken_ > th_refused_ ? th_ * th_mu_ : th_ / th_mu_;
        } else if (accept_ == "lsmc_rel") {                                                   // SRCH-17
            if (static_cast<long long>(steps_) <= t0_trials_) {
                if (next.cv == cur_.cv && next.f != cur_.f) t0_seen_.push_back(std::fabs(next.f - cur_.f));
                take = search::better(next, cur_);
                if (static_cast<long long>(steps_) == t0_trials_) {
                    std::vector<double> v = t0_seen_;
                    std::sort(v.begin(), v.end());
                    const std::size_t m = v.size();
                    const double med = m == 0 ? 0.0 : (m % 2 ? v[m / 2] : (v[m / 2 - 1] + v[m / 2]) / 2.0);
                    T_rel_ = med > 0.0 ? t_rel_ * med : 1e-12 * (1.0 + std::fabs(cur_.f));
                }
            } else {
                take = search::better(next, cur_) || (next.cv == cur_.cv && u < dm::exp((cur_.f - next.f) / T_rel_));
            }
        } else take = search::better(next, cur_);
        if (mv_.ops.on() || P_.on_child) {                                                   // MOVE-4, MOVE-5
            moves::Child ch;
            ch.mutation = tag;
            ch.parents_f = {cur_.f};
            moves::told(P_, nullptr, &mv_.ops, ch, next.f, take, rng_);
        }
        ++tried_;
        if (take) {
            ++accepted_;
            ++accepted_all_;
            cur_ = next;
        }
        if (ladder_) step_ = search::better(next, best_) ? ladder_min_                       // SRCH-18
                             : (step_ + ladder_step_ > ladder_max_ ? ladder_min_ : step_ + ladder_step_);
        if (search::better(next, best_)) {
            best_ = next;
            since_ = 0;
        } else if (accept_ == "restart" && ++since_ > restart_after_) {                        // (3)
            Vec x(n_);
            for (std::size_t j = 0; j < n_; ++j) x[j] = P_.lo[j] + rnd::uniform01(rng_) * search::width(P_, j);
            const search::Found fresh = local(search::at(x), ++s_);
            if (fresh.ok) cur_ = fresh;
            since_ = 0;
        }
        if (bh_ && !ladder_ && mv_.scaled() && tried_ == interval_) {                       // SRCH-5, SRCH-18, MOVE-1
            step_ = static_cast<double>(accepted_) / static_cast<double>(tried_) > target_ ? step_ / 0.9 : step_ * 0.9;
            step_ = (std::min)(step_, 1.0);                                                      // SRCH-10
            accepted_ = tried_ = 0;
        }
        idle_ = ev_->used() == before ? idle_ + 1 : 0;
        stop_.step(*ev_);
        ++s_;
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

inline Result run_walk(const std::string& name, const Problem& P, double eq_tol, bool has_tol0, double eq_tol0,
                       int pop_size, long long budget, unsigned seed, const Knobs& knobs, Observer observer,
                       int record_every, const Start& start)
{
    Walk w(name, P, eq_tol, has_tol0, eq_tol0, pop_size, budget, seed, knobs, std::move(observer), record_every, start);
    w.advance();                                                                     // RES-1: one piece
    return w.result();
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
    strategy::round_stop_of(k, ev);                                                  // STRAT-16
    StopRule stop = strategy::stop_of(k, ev);                                        // stop_rule.hpp: a step a round
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
        for (long long r = 0; r < rounds && !arms.empty() && stop.go(ev); ++r) {
            const long long more = budget / (static_cast<long long>(arms.size()) * rounds);
            if (more <= 0) break;
            std::vector<StartSpec> specs;
            const long long fresh = (std::min)(more, search::left(ev) / static_cast<long long>(arms.size()));
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
            stop.step(ev);
            if (r + 1 == rounds) break;
            std::stable_sort(arms.begin(), arms.end(),
                             [](const Arm& x, const Arm& y) { return search::better(x.best, y.best); });
            arms.resize((arms.size() + 1) / 2);
        }
    } catch (const BudgetSpent&) {
    }
    for (const Arm& a : arms)
        if (a.best.ok) log.add(a.best.x, a.best.f, a.index);
    return search::finish(ev, k, log, {{"rounds", json::dump(plan, -1, ",", ":")}}, stop.stopped_by(ev, "rule"));
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
    strategy::round_stop_of(k, ev);                                                  // STRAT-16
    StopRule stop = strategy::stop_of(k, ev);                                        // stop_rule.hpp: a step a phase
    std::string own = "rule";
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
        while (stop.go(ev)) {
            const double before = ev.used();
            g.fresh = (std::min)(every, search::left(ev));                                          // SRCH-7
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
            auto distant = [&](std::size_t i) {
                double d = std::numeric_limits<double>::infinity();
                for (const Vec& s : started) {
                    double m = 0.0;
                    for (std::size_t j = 0; j < n; ++j) m = (std::max)(m, std::fabs(ev.X[i][j] - s[j]) / scale(j));
                    d = (std::min)(d, m);
                }
                return d;
            };
            for (std::size_t c = 0; c < order.size() && static_cast<long long>(specs.size()) < n_points; ++c) {
                std::size_t i = order[c];
                if (pick == "spread" && !specs.empty()) {
                    double best_d = -1.0;
                    for (std::size_t q : order) {
                        const double d = distant(q);
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
                if (P.decode && local.name != "local_search") {                              // SRCH-21
                    const std::vector<char> active = active_of(P, ev.X[i], "memetic");
                    for (std::size_t j = 0; j < active.size(); ++j)
                        if (active[j]) sp.free.push_back(j);
                    if (!active.empty() && sp.free.empty()) continue;      // nothing to move
                    if (sp.free.size() == n) sp.free.clear();
                }
                sp.seed = start_seed(seed, static_cast<std::uint64_t>(1 + searches + static_cast<long long>(specs.size())));
                sp.start = search::at(ev.X[i]);
                specs.push_back(sp);
            }
            const long long share_cap =
                (std::max)(1LL, static_cast<long long>(share / (1.0 - share) * static_cast<double>(every)) /
                                  std::max<long long>(1, static_cast<long long>(specs.size())));
            const std::vector<Result> res = search::starts(ev, local, specs, cap > 0 ? (std::min)(cap, share_cap) : share_cap,
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
            stop.step(ev);
            if (global_done) {                                                                    // SRCH-9
                own = "idle";
                break;
            }
        }
    } catch (const BudgetSpent&) {
    }
    return search::finish(ev, k, log, {{"phases", std::to_string(phases)}, {"local_searches", std::to_string(searches)}},
                          stop.stopped_by(ev, own));
}

}   // namespace mootation::scalar
