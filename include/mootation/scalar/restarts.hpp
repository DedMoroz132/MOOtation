#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// Restarts of CMA-ES (task 18, stage 3, §3.3; owner's В4 and В5): the
// strategies ipop and bipop, and the preset bipop_2009. Each run is the
// library's CMA-ES (cmaes.hpp) in the problem's own coordinates, through the
// evaluator, until one of the termination criteria of Hansen 2009 (cmaes.hpp,
// stop()) holds or the budget is spent; then the next run. Included by
// strategy.hpp.
//
//   ipop        A. Auger, N. Hansen, "A Restart CMA Evolution Strategy With
//               Increasing Population Size", CEC 2005 (source
//               ipop-cma-es_auger2005, AH05), §2-3: each restart independent,
//               its population twice the last (λ_def 2^k); the start mean
//               uniform in the box, σ⁰ = (B − A)/2, until the budget.
//   bipop       N. Hansen, "Benchmarking a BI-Population CMA-ES on the
//               BBOB-2009 Function Testbed", GECCO 2009 (source
//               bipop-cma-es_hansen2009, H09), §2.1: a first run with λ_def,
//               then two regimes, each with its budget, and the one whose
//               budget is smaller runs next. Regime 1: λ = 2^i λ_def (i = 1 ...
//               9), σ⁰ = 2 (1/5 of the domain's width). Regime 2: λ_s =
//               ⌊λ_def (½ λ_ℓ/λ_def)^(U²)⌋, σ⁰ = 2·10^(−2U'), at most half of
//               the last large run's evaluations. The first and the last
//               restart are of regime 1. The start mean uniform in
//               [−4, 4]^D, i.e. the box less a tenth of its width each side.
//   bipop_2009  bipop with H09's CMA-ES: cma_weights = positive, cma_cmu =
//               hansen2009 (cmaes.hpp).
//
// Knobs: cma_weights, cma_cmu (cmaes.hpp); cma_sigma0 (σ⁰ as a share of the
// box's width: ipop 0.5, bipop 0.2); start_share (the share of the width left
// out each side of the box for the start mean: ipop 0, bipop 0.1); cma_bounds
// (penalty: a point outside the box is evaluated at its clip plus the squared
// distance times 1 + |f|, as SCALAR-7; none: evaluated where it is, for a
// function defined beyond the box, as bbob is, В5); bipop_large (9, the
// restarts of regime 1); min_dist, min_df (the minima log, strategy.hpp).
// Every run is in the settings ("restarts": regime, λ, σ⁰, evaluations, the
// criterion that stopped it).
//
// READINGS (declared):
//   RST-1. One set of termination criteria for both, H09's (§2.2): AH05's are
//     the same in substance (equalfunvalhist, TolFun, TolX, noeffectaxis,
//     noeffectcoord, conditioncov) but for TolFun over the last generation's
//     values; H09 adds MaxIter, TolUpSigma and Stagnation.
//   RST-2. U[0,1]² of λ_s is the square of one uniform number; the second
//     draw (σ⁰) is another (H09: "independent uniformly distributed numbers").
//   RST-3. "Half of the recent large budget": half of the evaluations of the
//     last run of regime 1.
//   RST-4. The regimes' budgets are the evaluations their runs spent; the
//     first run (λ_def) counts in neither (H09: "the first single run ... is
//     disregarded"). Regime 2 runs next when its budget is smaller than
//     regime 1's, else regime 1.
//   RST-5. "The first and last restart are conducted under the first regime":
//     after regime 1's last run (λ = 2^bipop_large λ_def) the strategy stops.
//   RST-6. The box gives the start: a bound that is not finite is refused, and
//     so are constraints (bounds only).
//   RST-7. A run's generator and its start mean come from start_seed(seed, k)
//     (strategy.hpp, STRAT-1), k the run's number from 0; bipop draws λ_s and
//     σ⁰ of a run of regime 2 from it before the start mean.
//   RST-8. A run that evaluates nothing ends the strategy: the budget, or the
//     run's target (evaluator.hpp, THE TARGET), is spent, whatever regime 2's
//     own allowance says.
// ============================================================================

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "cmaes.hpp"
#include "strategy.hpp"

namespace mootation::scalar {

namespace restarts {

struct Run {
    int regime = 0;                   // bipop: 0 the first run, 1 large, 2 small; ipop: 0
    std::size_t lambda = 0;
    double sigma0 = 0.0;
    long long evaluations = 0;
    CMAES::Stop stop = CMAES::Stop::none;
    bool budget_out = false;          // the budget (or the run's allowance) ended it
    double best_f = std::numeric_limits<double>::infinity();
    Vec best_x;
};

// One run of CMA-ES from m0 with step sigma0 and population lambda, until a
// criterion of H09 or the budget; at most `allowance` evaluations (0: no cap
// but the budget's).
inline Run run_cma(Evaluator& ev, const Vec& m0, double sigma0, std::size_t lambda, const CMAESOptions& options,
                   bool clip, long long allowance, std::mt19937& rng)
{
    Run r;
    r.lambda = lambda;
    r.sigma0 = sigma0;
    const long long before = ev.spent;
    const long long limit = ev.limit;
    if (allowance > 0) ev.limit = std::min(limit, ev.spent + allowance);
    CMAES es(m0, sigma0, rng, static_cast<int>(lambda), options);
    const std::size_t n = m0.size();
    auto value = [&](const Vec& y) {
        Vec x = y;
        double d2 = 0.0;
        if (clip)
            for (std::size_t j = 0; j < n; ++j) {
                x[j] = std::min(ev.hi[j], std::max(ev.lo[j], y[j]));
                d2 += (y[j] - x[j]) * (y[j] - x[j]);
            }
        const std::size_t i = ev.index(x);
        if (ev.failed[i]) return std::numeric_limits<double>::infinity();
        const double f = ev.F[i][0];
        if (f < r.best_f) {
            r.best_f = f;
            r.best_x = x;
        }
        return f + d2 * (1.0 + std::fabs(f));
    };
    try {
        while (true) {
            const auto Y = es.ask();
            Vec v;
            v.reserve(Y.size());
            for (const auto& y : Y) v.push_back(value(y));
            if (!ev.recorded) ev.record();
            es.tell(v);
            r.stop = es.stop();
            if (r.stop != CMAES::Stop::none) break;
        }
    } catch (const SubBudgetSpent&) {
        r.budget_out = true;
    } catch (const BudgetSpent&) {
        r.budget_out = true;
    }
    ev.limit = limit;
    r.evaluations = ev.spent - before;
    return r;
}

// the start mean: uniform in the box less `share` of its width each side
inline Vec start_mean(const Vec& lo, const Vec& hi, double share, std::mt19937& rng)
{
    Vec m(lo.size());
    for (std::size_t j = 0; j < lo.size(); ++j) {
        const double w = hi[j] - lo[j];
        m[j] = lo[j] + share * w + rnd::uniform01(rng) * (1.0 - 2.0 * share) * w;
    }
    return m;
}

}   // namespace restarts

inline Result run_restarts(const std::string& name, const Problem& P, double eq_tol, bool has_tol0, double eq_tol0,
                           int pop_size, long long budget, unsigned seed, const Knobs& knobs, Observer observer,
                           int record_every, const Start& start)
{
    using namespace strategy;
    using restarts::Run;
    const bool bipop = name != "ipop";
    std::vector<std::string> own{"cma_weights", "cma_cmu", "cma_sigma0", "start_share", "cma_bounds", "min_dist",
                                 "min_df"};
    if (bipop) own.push_back("bipop_large");
    StrategyKnobs k(name, knobs, own);
    CMAESOptions o;
    const bool h09 = name == "bipop_2009";
    if (k.word("cma_weights", h09 ? "positive" : "active", {"active", "positive"}) == "positive")
        o.weights = CMAESOptions::Weights::positive;
    if (k.word("cma_cmu", h09 ? "hansen2009" : "tutorial2016", {"tutorial2016", "hansen2009"}) == "hansen2009")
        o.cmu = CMAESOptions::Cmu::hansen2009;
    const double sigma_share = k.number("cma_sigma0", bipop ? 0.2 : 0.5);
    const double share = k.number("start_share", bipop ? 0.1 : 0.0);
    const bool clip = k.word("cma_bounds", "penalty", {"penalty", "none"}) == "penalty";
    const long long large = bipop ? k.whole("bipop_large", 9, 1) : 0;
    const double min_dist = k.number("min_dist", 1e-3), min_df = k.number("min_df", 1e-8);
    if (!(sigma_share > 0.0)) throw std::invalid_argument(name + ": cma_sigma0 must be > 0");
    if (!(share >= 0.0 && share < 0.5)) throw std::invalid_argument(name + ": start_share must be in [0, 0.5)");
    if (P.m != 1)
        throw std::invalid_argument(name + " is a single-objective algorithm: n_objs must be 1, got " +
                                    std::to_string(P.m));
    if (P.n_cons != 0) throw std::invalid_argument(name + " takes bounds only (RST-6)");
    const std::size_t n = P.lo.size();
    double width = 0.0;
    for (std::size_t j = 0; j < n; ++j) {
        if (!std::isfinite(P.lo[j]) || !std::isfinite(P.hi[j]))
            throw std::invalid_argument(name + " draws its start in the box: every bound must be finite (RST-6)");
        width = std::max(width, P.hi[j] - P.lo[j]);
    }
    std::vector<std::string> ignored = k.ignored();
    if (start.has_x0) ignored.push_back("x0");
    std::sort(ignored.begin(), ignored.end());

    Evaluator ev(P, eq_tol, has_tol0, eq_tol0, budget, pop_size, std::move(observer), record_every);
    const std::size_t lam_def = 4 + static_cast<std::size_t>(std::floor(3.0 * dm::log(static_cast<double>(n))));
    const double sigma_large = sigma_share * width;
    std::vector<Run> runs;
    auto one = [&](int regime, std::size_t lambda, double sigma0, long long allowance, std::mt19937& rng) {
        const Vec m0 = restarts::start_mean(P.lo, P.hi, share, rng);
        Run r = restarts::run_cma(ev, m0, sigma0, lambda, o, clip, allowance, rng);
        r.regime = regime;
        runs.push_back(r);
        return r;
    };
    for (unsigned long long q = 0;; ++q) {
        std::mt19937 rng(start_seed(seed, q));                                          // RST-7
        if (ev.spent >= ev.budget) break;                        // ev.budget: THE TARGET shortens it
        if (!bipop) {
            const Run r = one(0, lam_def << q, sigma_large, 0, rng);
            if (r.budget_out) break;
            continue;
        }
        // bipop (H09 §2.1; RST-3 ... RST-5)
        long long used_large = 0, used_small = 0, last_large = 0, i_large = 0;
        std::size_t lam_last = lam_def;
        for (const Run& done : runs) {
            if (done.regime == 1) {
                used_large += done.evaluations;
                last_large = done.evaluations;
                ++i_large;
                lam_last = done.lambda;
            }
            if (done.regime == 2) used_small += done.evaluations;
        }
        Run r;
        if (q == 0) {
            r = one(0, lam_def, sigma_large, 0, rng);
        } else if (i_large == 0 || used_small >= used_large) {
            r = one(1, lam_def << (i_large + 1), sigma_large, 0, rng);
            if (i_large + 1 >= large) break;                                              // RST-5
        } else {
            const double u = rnd::uniform01(rng), u2 = rnd::uniform01(rng);              // RST-2
            const double ratio = 0.5 * static_cast<double>(lam_last) / static_cast<double>(lam_def);
            const auto lam_s = static_cast<std::size_t>(std::floor(static_cast<double>(lam_def) * dm::pow(ratio, u * u)));
            r = one(2, std::max(lam_s, lam_def), sigma_large * dm::pow(10.0, -2.0 * u2), std::max(1LL, last_large / 2), rng);
            if (r.budget_out && ev.spent < ev.budget) r.budget_out = false;               // its own allowance ended
        }
        if (r.budget_out || r.evaluations == 0) break;                                    // RST-8
    }
    if (!ev.recorded) ev.record();
    MinimaLog log(P.lo, P.hi, min_dist, min_df);
    json::Value list = json::Value::array();
    for (std::size_t q = 0; q < runs.size(); ++q) {
        const Run& r = runs[q];
        if (!r.best_x.empty()) log.add(r.best_x, r.best_f, static_cast<long long>(q));
        json::Value row = json::Value::array();
        row.items.push_back(json::Value::integer(r.regime));
        row.items.push_back(json::Value::integer(static_cast<std::int64_t>(r.lambda)));
        row.items.push_back(json::Value::number(r.sigma0));
        row.items.push_back(json::Value::integer(r.evaluations));
        row.items.push_back(json::Value::string(r.budget_out ? "budget" : CMAES::stop_name(r.stop)));
        list.items.push_back(row);
    }
    ev.over = true;
    const std::vector<std::size_t> idx = ev.answer();
    ev.record(true);
    Result res = detail::result_of(ev, idx);
    res.ignored = ignored;
    res.settings = k.settings;
    res.settings.push_back({"restarts", json::dump(list, -1, ",", ":")});
    res.settings.push_back({"minima", log.json_text()});
    res.settings.push_back({"method_version", std::to_string(METHOD_VERSION)});
    return res;
}

}   // namespace mootation::scalar
