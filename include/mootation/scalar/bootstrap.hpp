#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// The bootstrap by residual blocks (task 28, §28.2: bagging, out-of-sample
// errors, the responses' spread): K runs of any method of scalar::run on the problem with
// its blocks resampled. Member k draws B' blocks with replacement from the
// B' blocks that are not probes (rnd::below, the generator seeded once with
// Options::seed, the draws of every member before any run); a block's weight
// in member k is its weight times the times it was drawn, so a block drawn
// twice counts twice and one not drawn (out of bag) not at all. Then:
//
//   * the member's run: scalar::run(method, the reweighted problem, budget,
//     seed + 1 + k, knobs, start);
//   * its out-of-bag error: the mean square per row of the weighted
//     residuals of its out-of-bag blocks at its answer's best point;
//   * its residuals on the probe blocks: blocks the caller names,
//     weight 0 in every member (never fitted), and the mean and the standard
//     deviation of each probe row over the members (the residual is the
//     response minus the datum: the spread of the residuals is the
//     responses').
//
// Out of bag a block is with probability (1 - 1/B')^B' -> 1/e = 0.368.
// Bagging a smooth least-squares fit gains little (P. Bühlmann, B. Yu,
// "Analyzing bagging", Ann. Statist. 30(4) 2002; the round of 11.11, §2.4):
// Mode::whole_pipeline (the default) resamples the whole method — a
// strategy's starts and its structure — not only its last fit; Mode::refit
// runs a plain fit first and starts every member from its answer (x0).
//
// Readings (OURS):
//   BT-1. The members' seeds: Options::seed + 1 + k (k = 0 .. K - 1); the
//     draws' generator: std::mt19937(Options::seed). Members run one after
//     another; a member's run is scalar::run's, so with every count 1
//     (Options::unit_counts, a check) member k is the plain run of seed
//     seed + 1 + k, bit for bit.
//   BT-2. The out-of-bag error and the probes' residuals are computed by the
//     problem's residuals at the member's answer, outside its budget: one
//     call a member, counted in Result::extra_evaluations. The problem's
//     weights w_b are kept in the out-of-bag error (sum w_b ||r_b||^2 / rows);
//     the probes' residuals are the problem's own (unweighted).
//   BT-3. A member whose answer is empty (a failed run) or whose out-of-bag
//     set is empty has oob = NaN; the mean and deviation skip no member (a
//     failed member makes the probe statistics NaN).
//   BT-4. Mode::refit: the plain fit is scalar::run with seed Options::seed,
//     the caller's start and the probes at weight 0; its result is
//     Result::base, its evaluations not the members'.
//   BT-5. Refused: a problem without residuals or without two residual
//     blocks or more, K < 1, a probe block out of range or named twice, no
//     block left to draw.
// ============================================================================

#include <cmath>
#include <cstddef>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "../fp_contract.hpp"
#include "../random.hpp"
#include "scalar.hpp"

namespace mootation::scalar::bootstrap {

struct Options {
    int members = 20;                       // K
    unsigned seed = 0;                      // BT-1
    std::vector<std::size_t> probe;         // the probe blocks, held out of every member
    enum class Mode { whole_pipeline, refit } mode = Mode::whole_pipeline;
    bool unit_counts = false;               // every count 1 (BT-1's check)
};

struct Member {
    std::vector<int> counts;                // a block's draws (0: out of bag; probes 0)
    unsigned seed = 0;
    scalar::Result run;
    Vec x;                                  // its answer's best point (empty: none)
    double oob = std::numeric_limits<double>::quiet_NaN();   // BT-2
    std::size_t oob_blocks = 0;
    Vec probe_r;                            // its residuals on the probe blocks' rows
};

struct Result {
    std::vector<Member> members;
    Vec probe_mean, probe_sd;               // over the members, a probe row each
    double oob_share = 0.0;                 // the mean share of the drawable blocks out of bag
    long long extra_evaluations = 0;        // BT-2
    scalar::Result base;                    // Mode::refit's plain fit (BT-4)
};

namespace detail {

// the problem's residuals at x, outside any evaluator (BT-2)
inline Vec residuals_at(const Problem& P, const Vec& x)
{
    Vec r;
    if (P.residuals) {
        P.residuals(x, r);
    } else {
        Rows J;
        P.residuals_jac(x, r, J);
    }
    return r;
}

}   // namespace detail

inline Result fit(const std::string& method, const Problem& P, long long budget, const Knobs& knobs,
                  const Options& o, const Start& start = {}, int pop_size = 1)
{
    if (!P.has_residuals()) throw std::invalid_argument("bootstrap: the problem gives no residuals (BT-5)");
    const std::size_t B = P.blocks.size();
    if (B < 2) throw std::invalid_argument("bootstrap: two residual blocks or more are needed (BT-5)");
    if (o.members < 1) throw std::invalid_argument("bootstrap: members >= 1 (BT-5)");
    std::vector<char> is_probe(B, 0);
    for (std::size_t b : o.probe) {
        if (b >= B) throw std::invalid_argument("bootstrap: a probe block out of range (BT-5)");
        if (is_probe[b]) throw std::invalid_argument("bootstrap: a probe block named twice (BT-5)");
        is_probe[b] = 1;
    }
    std::vector<std::size_t> pool;                     // the drawable blocks
    for (std::size_t b = 0; b < B; ++b)
        if (!is_probe[b]) pool.push_back(b);
    if (pool.empty()) throw std::invalid_argument("bootstrap: no block left to draw (BT-5)");
    const Vec w0 = P.weights.empty() ? Vec(B, 1.0) : P.weights;
    std::vector<std::size_t> start_row(B + 1, 0);
    for (std::size_t b = 0; b < B; ++b) start_row[b + 1] = start_row[b] + static_cast<std::size_t>(P.blocks[b]);

    Result res;
    std::mt19937 draws(o.seed);                        // BT-1: every member's draws first
    res.members.resize(static_cast<std::size_t>(o.members));
    for (std::size_t k = 0; k < res.members.size(); ++k) {
        Member& mb = res.members[k];
        mb.seed = o.seed + 1u + static_cast<unsigned>(k);
        mb.counts.assign(B, 0);
        for (std::size_t d = 0; d < pool.size(); ++d) {
            const std::size_t b = o.unit_counts ? pool[d] : pool[rnd::below(draws, pool.size())];
            ++mb.counts[b];
        }
    }
    Start st = start;
    if (o.mode == Options::Mode::refit) {              // BT-4
        Problem P0 = P;                                // the probes held out of it too
        P0.weights = w0;
        for (std::size_t b : o.probe) P0.weights[b] = 0.0;
        res.base = scalar::run(method, P0, 1e-6, false, 0.0, pop_size, budget, o.seed, knobs, {}, 0, start);
        if (res.base.variables.empty()) throw std::runtime_error("bootstrap: the plain fit has no answer (BT-4)");
        st.has_x0 = true;
        st.x0 = res.base.variables[0];
    }
    double oob_sum = 0.0;
    std::size_t probe_rows = 0;
    for (std::size_t b : o.probe) probe_rows += static_cast<std::size_t>(P.blocks[b]);
    for (Member& mb : res.members) {
        Problem Pk = P;
        Pk.weights.assign(B, 0.0);
        for (std::size_t b = 0; b < B; ++b) Pk.weights[b] = w0[b] * mb.counts[b];
        mb.run = scalar::run(method, Pk, 1e-6, false, 0.0, pop_size, budget, mb.seed, knobs, {}, 0, st);
        std::size_t out = 0;
        for (std::size_t b : pool) out += mb.counts[b] == 0;
        oob_sum += static_cast<double>(out) / static_cast<double>(pool.size());
        if (mb.run.variables.empty()) {               // BT-3
            mb.probe_r.assign(probe_rows, std::numeric_limits<double>::quiet_NaN());
            continue;
        }
        mb.x = mb.run.variables[0];
        const Vec r = detail::residuals_at(P, mb.x);
        ++res.extra_evaluations;
        if (r.size() != start_row[B]) throw std::invalid_argument("bootstrap: the residuals' count changed");
        double s = 0.0;
        std::size_t rows = 0;
        for (std::size_t b : pool)
            if (mb.counts[b] == 0) {
                ++mb.oob_blocks;
                for (std::size_t i = start_row[b]; i < start_row[b + 1]; ++i) s += w0[b] * r[i] * r[i];
                rows += static_cast<std::size_t>(P.blocks[b]);
            }
        if (rows > 0) mb.oob = s / static_cast<double>(rows);
        for (std::size_t b : o.probe)
            for (std::size_t i = start_row[b]; i < start_row[b + 1]; ++i) mb.probe_r.push_back(r[i]);
    }
    res.oob_share = oob_sum / static_cast<double>(res.members.size());
    const double K = static_cast<double>(res.members.size());
    res.probe_mean.assign(probe_rows, 0.0);
    res.probe_sd.assign(probe_rows, 0.0);
    for (std::size_t i = 0; i < probe_rows; ++i) {
        double m = 0.0;
        for (const Member& mb : res.members) m += mb.probe_r[i];
        m /= K;
        double v = 0.0;
        for (const Member& mb : res.members) v += (mb.probe_r[i] - m) * (mb.probe_r[i] - m);
        res.probe_mean[i] = m;
        res.probe_sd[i] = res.members.size() > 1 ? std::sqrt(v / (K - 1.0)) : 0.0;
    }
    return res;
}

}   // namespace mootation::scalar::bootstrap
