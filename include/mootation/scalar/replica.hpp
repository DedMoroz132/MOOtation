#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// replica_ladder: replica exchange over a ladder of a parameter (task 26,
// item 24.2.8). Included by strategy.hpp.
//
// K. Hukushima, K. Nemoto, "Exchange Monte Carlo method and application to
// spin glass simulations", J. Phys. Soc. Jpn. 65 (1996) 1604 (source
// exchange-monte-carlo_hukushima1996), §2: K replicas, replica m at its own
// level; after the replicas' own steps the configurations of neighbours m,
// m + 1 are exchanged with probability W = min(1, exp(-Delta)), (2.5)-(2.7),
// Delta = (beta_{m+1} - beta_m)(H(X_m) - H(X_{m+1})) on a temperature
// ladder. G. Basurto et al. 2025 (source densest-packings-accelerated_
// basurto2025), Eq. (6): the exponent times f_s; where the exponent is
// negative (the exchanges balance would refuse) f_s < 1 raises the
// acceptance beyond balance (rl_fs, default 1: the physical exchange,
// decision 16 of the package; f_s < 1 only when asked).
//
// A replica is the inner solver's run at its level, in pieces of
// swap_every evaluations, each piece from the replica's current point (the
// answer of its last piece; its seed start_seed(seed, s), s counting the
// pieces of the run). The ladder:
//   rl_over = knob   the inner solver's knob rl_knob (e.g. bh_temperature
//                    of basin_hopping) takes the levels as temperatures T;
//                    beta = 1/T, every replica on the problem itself,
//                    Delta as above (Hukushima and Nemoto's);
//   rl_over = param  the problem's parameter (param.hpp: problem_at), the
//                    levels are values L of it; the exchange of
//                    configurations between two problems H_m, H_{m+1} at
//                    one inverse temperature rl_beta: Delta = rl_beta
//                    [H_m(X_{m+1}) + H_{m+1}(X_m) - H_m(X_m) - H_{m+1}
//                    (X_{m+1})] (RL-1), two evaluations, counted.
// The levels: rl_levels (a JSON list), else rl_K levels geometric from
// rl_lo to rl_hi (RL-2). Neighbour pairs are tried in the order 0-1, 1-2,
// ... after every round of pieces (RL-3).
//
// THE DEGENERATE CASES: one level and no exchange (swap_every = 0: one
// piece with the whole budget) is the inner solver at that level, bit for
// bit (param: on problem_at(P, L), the seed the run's); with K levels and
// swap_every = 0 the replicas are K independent runs, the budget shared
// (their bests in the settings, replica_best).
//
// Knobs: inner, rl_over (knob | param), rl_knob, rl_levels, rl_K (4),
// rl_lo, rl_hi, rl_beta (1), rl_fs (1), swap_every (0: no exchange).
//
// READINGS:
//   RL-1. Over a parameter the replicas carry configurations between two
//     problems (a Hamiltonian exchange): Hukushima and Nemoto's detailed
//     balance with H_m(X) in place of beta_m H(X); the configuration moved
//     to another level is carried (param.hpp carried: the source's rescale,
//     clipped) before it is evaluated there.
//   RL-2. The geometric ladder is this library's default where no list is
//     given (Hukushima and Nemoto choose their temperatures by the
//     acceptance; their rule is not taken up).
//   RL-3. An exchange's uniform comes from one stream seeded by the run's
//     seed, drawn only where Delta > 0 (W = 1 otherwise).
//   RL-4. The answer: over a parameter the last level's best piece (the
//     ladder ends at the level wanted); over a knob the best piece of any
//     replica (all solve the problem itself).
// ============================================================================

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "param.hpp"
#include "strategy.hpp"

namespace mootation::scalar {

inline Result run_replica_ladder(const Problem& P, double eq_tol, bool has_tol0, double eq_tol0, int pop_size,
                                 long long budget, unsigned seed, const Knobs& knobs, Observer observer,
                                 int record_every, const Start& start)
{
    using namespace strategy;
    (void)observer;
    (void)record_every;
    StrategyKnobs k("replica_ladder", knobs, {"inner", "rl_over", "rl_knob", "rl_levels", "rl_K", "rl_lo", "rl_hi",
                                              "rl_beta", "rl_fs", "swap_every"});
    const Inner inner = k.inner("inner");
    const bool over_param = k.word("rl_over", "knob", {"knob", "param"}) == "param";
    const double beta = k.number("rl_beta", 1.0), fs = k.number("rl_fs", 1.0);
    const long long every = k.whole("swap_every", 0, 0);
    if (!(fs > 0.0)) throw std::invalid_argument("replica_ladder: rl_fs must be > 0");
    std::string knob;
    if (!over_param) {
        const Knob* kk = k.find("rl_knob");
        if (!kk || !kk->is_word || kk->word.empty())
            throw std::invalid_argument("replica_ladder: rl_over = knob needs rl_knob, the inner solver's knob");
        knob = kk->word;
    }
    std::vector<double> lv;
    if (const Knob* kl = k.find("rl_levels")) {
        if (!kl->is_word) throw std::invalid_argument("replica_ladder: rl_levels must be a JSON list");
        const json::Value v = json::parse(kl->word);
        if (v.kind != json::Value::Kind::Array || v.items.empty())
            throw std::invalid_argument("replica_ladder: rl_levels must be a non-empty JSON list of numbers");
        for (const auto& it : v.items) lv.push_back(it.kind == json::Value::Kind::Int ? static_cast<double>(it.i) : it.f);
    } else {                                                                         // RL-2
        const long long K = k.whole("rl_K", 4, 1);
        const double lo = k.number("rl_lo", std::numeric_limits<double>::quiet_NaN());
        const double hi = k.number("rl_hi", std::numeric_limits<double>::quiet_NaN());
        if (!(lo > 0.0 && hi > 0.0)) throw std::invalid_argument("replica_ladder: rl_lo and rl_hi (> 0), or rl_levels");
        for (long long q = 0; q < K; ++q)
            lv.push_back(K == 1 ? hi : lo * dm::pow(hi / lo, static_cast<double>(q) / static_cast<double>(K - 1)));
    }
    if (!over_param)
        for (double t : lv)
            if (!(t > 0.0)) throw std::invalid_argument("replica_ladder: a temperature level must be > 0");
    const std::size_t K = lv.size();
    std::vector<Problem> Q;
    for (double L : lv) Q.push_back(over_param ? problem_at(P, L) : P);
    auto knobs_at = [&](std::size_t m) {
        Knobs kn = inner.knobs;
        if (!over_param) {
            bool set = false;
            for (auto& kv : kn)
                if (kv.first == knob) {
                    kv.second = Knob{false, "", lv[m]};
                    set = true;
                }
            if (!set) kn.push_back({knob, Knob{false, "", lv[m]}});
        }
        return kn;
    };
    std::mt19937 rng(start_seed(seed, 0x5EA1u));                                     // RL-3
    long long spent = 0, grads = 0, pieces = 0, tried = 0, swapped = 0;
    std::vector<Vec> x(K);
    std::vector<double> E(K, std::numeric_limits<double>::infinity());
    std::vector<Result> best(K);
    std::vector<char> has(K, 0);
    std::vector<char> started(K, 0);
    auto value = [&](std::size_t m, const Vec& y) {
        Vec f, g;
        Q[m].evaluate(y, f, g);
        ++spent;
        return f.empty() ? std::numeric_limits<double>::infinity() : f[0];
    };
    while (budget - spent > 0) {
        const long long left = budget - spent;
        const long long piece = every > 0 ? std::max(1LL, std::min(every, left / static_cast<long long>(K)))
                                          : std::max(1LL, left / static_cast<long long>(K));
        bool any = false;
        for (std::size_t m = 0; m < K && budget - spent > 0; ++m) {
            const long long allowance = std::min(piece, budget - spent);
            const Start st = started[m] ? search::at(x[m]) : start;
            const Result r = run(inner.name, Q[m], eq_tol, has_tol0, eq_tol0, pop_size, allowance,
                                 start_seed(seed, static_cast<std::uint64_t>(pieces)), knobs_at(m), {}, 0, st);
            ++pieces;
            spent += r.evaluations;
            grads += r.grad_calls;
            any = any || r.evaluations > 0;
            if (r.variables.empty()) continue;
            started[m] = 1;
            x[m] = r.variables[0];
            E[m] = r.objectives[0][0];
            if (!has[m] || r.objectives[0][0] < best[m].objectives[0][0]) {
                best[m] = r;
                has[m] = 1;
            }
        }
        if (every == 0 || !any) break;
        for (std::size_t m = 0; m + 1 < K && budget - spent > 1; ++m) {          // RL-3: the neighbours
            if (!started[m] || !started[m + 1]) continue;
            ++tried;
            double delta;
            Vec xa = x[m], xb = x[m + 1];
            double Ea = E[m], Eb = E[m + 1];
            if (over_param) {                                                        // RL-1
                xa = carried(Q[m + 1], Q[m], x[m + 1], lv[m + 1], lv[m]);
                xb = carried(Q[m], Q[m + 1], x[m], lv[m], lv[m + 1]);
                Ea = value(m, xa);
                Eb = value(m + 1, xb);
                delta = beta * (Ea + Eb - E[m] - E[m + 1]);
            } else {
                delta = (1.0 / lv[m + 1] - 1.0 / lv[m]) * (E[m] - E[m + 1]);
                xa = x[m + 1];
                xb = x[m];
                Ea = E[m + 1];
                Eb = E[m];
            }
            const bool accept = !(delta > 0.0) || rnd::uniform01(rng) < dm::exp(-fs * delta);
            if (accept) {
                x[m] = xa;
                x[m + 1] = xb;
                E[m] = Ea;
                E[m + 1] = Eb;
                ++swapped;
            }
        }
    }
    std::size_t pick = K - 1;                                                        // RL-4
    if (!over_param)
        for (std::size_t m = 0; m < K; ++m)
            if (has[m] && (!has[pick] || best[m].objectives[0][0] < best[pick].objectives[0][0])) pick = m;
    Result out = best[pick];
    out.evaluations = spent;
    out.grad_calls = grads;
    out.ignored = k.ignored();
    std::string per = "[";
    for (std::size_t m = 0; m < K; ++m)
        per += (m ? "," : "") + (has[m] ? detail::g_format(best[m].objectives[0][0]) : std::string("null"));
    per += "]";
    out.settings.push_back({"replica_best", per});
    out.settings.push_back({"pieces", std::to_string(pieces)});
    out.settings.push_back({"swaps_tried", std::to_string(tried)});
    out.settings.push_back({"swaps_accepted", std::to_string(swapped)});
    return out;
}

}   // namespace mootation::scalar
