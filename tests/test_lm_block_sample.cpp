// SPDX-License-Identifier: Apache-2.0
// lm on a subsample of the residual blocks (task 28, §28.1; scalar.hpp
// LOCAL-11, BS; lm.hpp LMS):
//   * the draw: exactly k blocks of B every time, a block's rows alike,
//     sqrt(B / k) with reweight (1 without), each block drawn k / B of the
//     time; with reweight the subsample's cost is the full cost on average
//     (an unbiased estimate);
//   * lm_block_sample = frac with q = 1, and count with k = B: plain lm,
//     bit for bit (BS-2);
//   * count k = B / 2 on an exact model: the answer reaches the model (the
//     evaluator's full cost below 1e-20), the draws counted, no saving;
//   * one seed one run; another seed another;
//   * the problem's residuals_blocks: called with the drawn blocks' mask
//     (k of B), each call charged k/B of an evaluation, lm's points
//     evaluated in full, the answer the model's (BS-6, BS-7);
//   * refused: one block, lm_loss = huber, k out of range, lm_block_size
//     without lm_block_sample (BS-5).

#include <cmath>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <mootation/scalar/scalar.hpp>

namespace sc = mootation::scalar;
using Vec = std::vector<double>;

static int failures = 0;

static void check(bool ok, const std::string& what)
{
    std::printf("%s  %s\n", ok ? "ok  " : "FAIL", what.c_str());
    if (!ok) ++failures;
}

// y = a exp(-b t) + c observed in 8 blocks of 5 points (block i at t = i + 0.2 j), exact at (2, 0.3, 0.5)
static sc::Problem decay(bool blocks = true)
{
    sc::Problem P;
    P.m = 1;
    P.lo = {-10.0, 0.0, -10.0};
    P.hi = {10.0, 5.0, 10.0};
    P.n_res = 40;
    if (blocks) {
        P.blocks.assign(8, 5);
        P.weights.assign(8, 1.0);
    }
    P.residuals = [](const Vec& x, Vec& r) {
        r.assign(40, 0.0);
        for (int i = 0; i < 40; ++i) {
            const double t = (i / 5) + 0.2 * (i % 5);
            const double y = 2.0 * mootation::dm::exp(-0.3 * t) + 0.5;
            r[static_cast<std::size_t>(i)] = x[0] * mootation::dm::exp(-x[1] * t) + x[2] - y;
        }
    };
    return P;
}

static sc::Knob word(const std::string& w)
{
    sc::Knob k;
    k.is_word = true;
    k.word = w;
    return k;
}

static sc::Knob number(double v)
{
    sc::Knob k;
    k.number = v;
    return k;
}

static sc::Result run(const sc::Problem& P, const sc::Knobs& k, unsigned seed = 1)
{
    sc::Start s;
    s.has_x0 = true;
    s.x0 = {1.0, 1.0, 0.0};
    return sc::run("lm", P, 1e-6, false, 0.0, 1, 3000, seed, k, {}, 0, s);
}

static std::string setting(const sc::Result& r, const std::string& k)
{
    for (const auto& [a, b] : r.settings)
        if (a == k) return b;
    return "";
}

static bool same(const sc::Result& a, const sc::Result& b)
{
    bool ok = a.variables == b.variables && a.objectives == b.objectives && a.evaluations == b.evaluations &&
              a.grad_calls == b.grad_calls && a.improvements.size() == b.improvements.size();
    for (std::size_t k = 0; ok && k < a.improvements.size(); ++k)
        ok = a.improvements[k].f == b.improvements[k].f && a.improvements[k].evaluations == b.improvements[k].evaluations;
    return ok;
}

static void draws()
{
    const std::vector<int> blocks{3, 1, 4, 1, 5, 9, 2, 6};
    std::mt19937 rng(7);
    bool exact = true, alike = true;
    std::vector<int> hits(8, 0);
    const int N = 4000;
    double full = 0.0, mean = 0.0;
    Vec r;
    for (int b = 0, at = 0; b < 8; ++b)
        for (int q = 0; q < blocks[static_cast<std::size_t>(b)]; ++q, ++at) r.push_back(0.1 * (at % 7) - 0.25);
    for (double v : r) full += v * v;
    for (int d = 0; d < N; ++d) {
        const Vec rows = sc::detail::block_rows(blocks, 3, true, rng);
        std::size_t at = 0;
        int in = 0;
        double c = 0.0;
        for (std::size_t b = 0; b < 8; ++b) {
            const double s = rows[at];
            for (int q = 0; q < blocks[b]; ++q, ++at) {
                alike = alike && rows[at] == s;
                c += rows[at] * rows[at] * r[at] * r[at];
            }
            exact = exact && (s == 0.0 || std::fabs(s - std::sqrt(8.0 / 3.0)) < 1e-15);
            if (s != 0.0) {
                ++in;
                ++hits[b];
            }
        }
        exact = exact && in == 3;
        mean += c / N;
    }
    double worst = 0.0;
    for (int h : hits) worst = std::max(worst, std::fabs(h / static_cast<double>(N) - 3.0 / 8.0));
    char line[200];
    std::snprintf(line, sizeof line,
                  "the draw: 3 blocks of 8 each time, rows alike, sqrt(8/3); shares 3/8 within %.3f; mean cost / "
                  "full %.4f",
                  worst, mean / full);
    check(exact && alike && worst < 0.03 && std::fabs(mean / full - 1.0) < 0.03, line);
    std::mt19937 g2(7);
    const Vec plain = sc::detail::block_rows(blocks, 3, false, g2);
    bool ones = true;
    for (double v : plain) ones = ones && (v == 0.0 || v == 1.0);
    check(ones, "lm_block_reweight = off: the drawn rows times 1");
}

static void plain_lm()
{
    const sc::Problem P = decay();
    const sc::Result a = run(P, {});
    const sc::Result b = run(P, {{"lm_block_sample", word("frac")}, {"lm_block_size", number(1.0)}});
    const sc::Result c = run(P, {{"lm_block_sample", word("count")}, {"lm_block_size", number(8.0)}});
    check(same(a, b) && same(a, c) && setting(b, "lm_resamples") == "0" && setting(a, "lm_block_sample").empty(),
          "frac q = 1 and count k = B: plain lm, bit for bit; off adds no setting (BS-2)");
}

static void subsample()
{
    const sc::Problem P = decay();
    const sc::Knobs k{{"lm_block_sample", word("count")}, {"lm_block_size", number(4.0)}};
    const sc::Result a = run(P, k), b = run(P, k), c = run(P, k, 2);
    const double f = a.objectives.empty() ? 1e300 : a.objectives[0][0];
    char line[240];
    std::snprintf(line, sizeof line,
                  "count k = 4 of 8 on an exact model: full cost %.2e at the answer, %s draws after the first, "
                  "lm_block_k %s, saving %s",
                  f, setting(a, "lm_resamples").c_str(), setting(a, "lm_block_k").c_str(),
                  setting(a, "lm_block_saving").c_str());
    check(f < 1e-20 && std::stoll(setting(a, "lm_resamples")) > 0 && setting(a, "lm_block_k") == "4" &&
              setting(a, "lm_block_saving") == "none",
          line);
    check(same(a, b) && !same(a, c), "one seed one run, another seed another");
    const sc::Result e = run(P, {{"lm_block_sample", word("frac")}, {"lm_block_size", number(0.3)},
                                 {"lm_resample_every", number(3.0)}, {"lm_block_reweight", word("off")}});
    check(setting(e, "lm_block_k") == "2" && setting(e, "lm_resample_every") == "3" &&
              setting(e, "lm_block_reweight") == "off",
          "frac q = 0.3 of 8: k = round(2.4) = 2; lm_resample_every and lm_block_reweight in the settings");
}

static void hook()
{
    sc::Problem P = decay();
    long long calls = 0, wrong = 0;
    P.residuals_blocks = [&](const Vec& x, const std::vector<char>& mask, Vec& r) {   // the drawn blocks alone
        ++calls;
        int on = 0;
        for (char c : mask) on += c != 0;
        if (mask.size() != 8 || on != 4) ++wrong;
        for (int i = 0; i < 40; ++i) {
            if (!mask[static_cast<std::size_t>(i / 5)]) continue;
            const double t = (i / 5) + 0.2 * (i % 5);
            const double y = 2.0 * mootation::dm::exp(-0.3 * t) + 0.5;
            r[static_cast<std::size_t>(i)] = x[0] * mootation::dm::exp(-x[1] * t) + x[2] - y;
        }
    };
    const sc::Knobs k{{"lm_block_sample", word("count")}, {"lm_block_size", number(4.0)}};
    const sc::Result a = run(P, k), b = run(P, k);
    const double f = a.objectives.empty() ? 1e300 : a.objectives[0][0];
    const long long bc = std::stoll(setting(a, "lm_block_calls"));
    const double spent = std::stod(setting(a, "lm_block_spent"));
    char line[260];
    std::snprintf(line, sizeof line,
                  "residuals_blocks: %lld calls of 4 blocks each, %.1f evaluations charged for them (half each); "
                  "%lld full evaluations of lm's points; full cost %.2e at the answer",
                  bc, spent, a.evaluations, f);
    check(setting(a, "lm_block_saving") == "hook" && wrong == 0 && bc == calls / 2 && spent == 0.5 * bc &&
              f < 1e-20 && a.evaluations + spent <= 3000.0,
          line);
    check(same(a, b), "residuals_blocks: one seed one run");
    sc::Problem Q = P;
    Q.periodic = {1, 0, 0};
    bool refused = false;
    try {
        run(Q, k);
    } catch (const std::invalid_argument& e) {
        refused = std::string(e.what()).find("BS-6") != std::string::npos;
    }
    check(refused, "residuals_blocks with a periodic variable refused (BS-6)");
}

static void refusals()
{
    auto refused = [](const sc::Problem& P, const sc::Knobs& k) {
        try {
            run(P, k);
        } catch (const std::invalid_argument&) {
            return true;
        }
        return false;
    };
    const sc::Problem P = decay(), one = decay(false);
    check(refused(one, {{"lm_block_sample", word("count")}, {"lm_block_size", number(1.0)}}) &&
              refused(P, {{"lm_block_sample", word("count")}, {"lm_block_size", number(4.0)},
                          {"lm_loss", word("huber")}}) &&
              refused(P, {{"lm_block_sample", word("count")}, {"lm_block_size", number(9.0)}}) &&
              refused(P, {{"lm_block_sample", word("count")}, {"lm_block_size", number(2.5)}}) &&
              refused(P, {{"lm_block_sample", word("frac")}, {"lm_block_size", number(0.0)}}) &&
              refused(P, {{"lm_block_size", number(0.5)}}) &&
              refused(P, {{"lm_block_sample", word("count")}, {"lm_block_size", number(4.0)},
                          {"lm_resample_every", number(0.0)}}),
          "refused: one block, lm_loss = huber, k = 9 of 8, k = 2.5, q = 0, lm_block_size alone, "
          "lm_resample_every = 0 (BS-5)");
}

int main()
{
    draws();
    plain_lm();
    subsample();
    hook();
    refusals();
    std::printf(failures ? "FAILED\n" : "all passed\n");
    return failures ? 1 : 0;
}
