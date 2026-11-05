// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// scalar/cmaes.hpp's options and termination criteria of BIPOP-CMA-ES (task
// 18, stage 3; N. Hansen, GECCO 2009, §2 and §2.2, "H09"):
//   - weights = positive and cmu = hansen2009: the weights, μ_w and c_μ by
//     H09's formulas, computed here by hand for D = 10, λ = 10 and λ = 15;
//   - the defaults unchanged: test_cmaes_inject's checksum of six runs;
//   - stop(): each criterion on a run made to meet it first — a constant f
//     (EqualFunVals at iteration D), values falling at every call (MaxIter at
//     ⌈100 + 50(D + 3)²/√λ⌉), a shifted sphere (TolHistFun), a sphere scaled
//     by 1e30 (TolX), noise (Stagnation), a mean of 1e20 (NoEffectAxis) or
//     with one coordinate of 1e20 (NoEffectCoor);
//   - H09's CMA-ES on Rosenbrock, D = 10: below 1e-8, then stopped by a
//     criterion; a checksum of its points, the same with MSVC and clang.
// ============================================================================

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <vector>

#include <mootation/scalar/cmaes.hpp>

namespace {

int g_failed = 0;

void check(bool ok, const std::string& what)
{
    if (!ok) {
        ++g_failed;
        std::printf("FAIL  %s\n", what.c_str());
    } else {
        std::printf("ok    %s\n", what.c_str());
    }
}

using mootation::scalar::CMAES;
using mootation::scalar::CMAESOptions;
using Vec = std::vector<double>;

CMAESOptions h09()
{
    CMAESOptions o;
    o.weights = CMAESOptions::Weights::positive;
    o.cmu = CMAESOptions::Cmu::hansen2009;
    return o;
}

// H09 §2 by hand: w_i = (ln(μ+1) − ln i)/Σ_j (ln(μ+1) − ln j), μ = ⌊λ/2⌋
void options()
{
    for (int lam : {10, 15}) {
        std::mt19937 rng(1);
        CMAES es(Vec(10, 0.0), 1.0, rng, lam, h09());
        const int mu = lam / 2;
        double sum = 0.0, sum2 = 0.0;
        for (int i = 1; i <= mu; ++i) sum += std::log(mu + 1.0) - std::log(static_cast<double>(i));
        Vec w(static_cast<std::size_t>(lam), 0.0);
        for (int i = 1; i <= mu; ++i) {
            w[static_cast<std::size_t>(i - 1)] = (std::log(mu + 1.0) - std::log(static_cast<double>(i))) / sum;
            sum2 += w[static_cast<std::size_t>(i - 1)] * w[static_cast<std::size_t>(i - 1)];
        }
        const double mueff = 1.0 / sum2, n = 10.0;
        const double c1 = 2.0 / ((n + 1.3) * (n + 1.3) + mueff);
        const double cmu = std::min(1.0 - c1, 2.0 * (mueff - 2.0 + 1.0 / mueff) / ((n + 2.0) * (n + 2.0) + mueff));
        double dw = 0.0;
        for (std::size_t i = 0; i < w.size(); ++i) dw = std::max(dw, std::fabs(es.weights()[i] - w[i]));
        check(dw < 1e-15 && std::fabs(es.mu_eff() - mueff) < 1e-13 && std::fabs(es.c_mu() - cmu) < 1e-14 * cmu,
              "H09 weights, mu_w and c_mu, lambda = " + std::to_string(lam) + " (|dw| " + std::to_string(dw) +
                  ", c_mu " + std::to_string(es.c_mu()) + ")");
    }
    std::mt19937 rng(1);
    CMAES es(Vec(10, 0.0), 1.0, rng, 10);
    bool negative = false;
    for (double w : es.weights()) negative = negative || w < 0.0;
    check(negative && es.c_mu() != CMAES(Vec(10, 0.0), 1.0, rng, 10, h09()).c_mu(),
          "the defaults: negative weights and the tutorial's c_mu");
}

struct Run {
    CMAES::Stop stop = CMAES::Stop::none;
    long long t = 0;
    double best = 1e300;
};

// ask/tell until stop() names a criterion or `limit` iterations
Run run(CMAES& es, const std::function<double(const Vec&)>& f, long long limit = 100000)
{
    Run r;
    for (long long it = 0; it < limit; ++it) {
        const auto X = es.ask();
        Vec v;
        for (const auto& x : X) {
            v.push_back(f(x));
            r.best = std::min(r.best, v.back());
        }
        es.tell(v);
        r.stop = es.stop();
        r.t = es.iterations();
        if (r.stop != CMAES::Stop::none) break;
    }
    return r;
}

std::string said(const Run& r)
{
    return std::string(CMAES::stop_name(r.stop)) + " at t = " + std::to_string(r.t);
}

void criteria()
{
    {   // a constant f: f(x_1:λ) = f(x_k:λ) at every iteration, so EqualFunVals once there are D of them
        std::mt19937 rng(2);
        CMAES es(Vec(10, 0.0), 1.0, rng, 0, h09());
        const Run r = run(es, [](const Vec&) { return 1.0; });
        check(r.stop == CMAES::Stop::equal_fun_vals && r.t == 10, "constant f: EqualFunVals at t = D = 10 (" + said(r) + ")");
    }
    {   // values falling at every call: nothing but MaxIter, ⌈100 + 50·25/√6⌉ = 611 at D = 2, λ = 6
        std::mt19937 rng(3);
        CMAES es(Vec(2, 0.0), 1.0, rng, 0, h09());
        double c = 0.0;
        const Run r = run(es, [&](const Vec&) { return c -= 1.0; });
        check(r.stop == CMAES::Stop::max_iter && r.t == 611, "falling values: MaxIter at t = 611 (" + said(r) + ")");
    }
    {   // a shifted sphere: the best values flatten below 1e-12 long before the steps are 1e-12
        std::mt19937 rng(4);
        CMAES es(Vec(5, 1.0), 1.0, rng, 0, h09());
        const Run r = run(es, [](const Vec& x) {
            double s = 1.0;
            for (double t : x) s += t * t;
            return s;
        });
        check(r.stop == CMAES::Stop::tol_hist_fun && r.best - 1.0 < 1e-11, "1 + |x|^2: TolHistFun (" + said(r) + ")");
    }
    {   // 1e30 |x|^2: the values keep spreading while the steps fall below 1e-12 σ⁰
        std::mt19937 rng(5);
        CMAES es(Vec(5, 1.0), 1.0, rng, 0, h09());
        const Run r = run(es, [](const Vec& x) {
            double s = 0.0;
            for (double t : x) s += t * t;
            return 1e30 * s;
        });
        check(r.stop == CMAES::Stop::tol_x, "1e30 |x|^2: TolX (" + said(r) + ")");
    }
    {   // noise: no progress, the medians do not fall
        std::mt19937 rng(6), noise(60);
        std::uniform_real_distribution<double> u(0.0, 1.0);
        CMAES es(Vec(3, 0.0), 1.0, rng, 0, h09());
        const Run r = run(es, [&](const Vec&) { return u(noise); });
        const double window = std::ceil(0.2 * static_cast<double>(r.t) + 120.0 + 30.0 * 3.0 / 7.0);
        check(r.stop == CMAES::Stop::stagnation && static_cast<double>(r.t) >= window,
              "noise: Stagnation, not before its window (" + said(r) + ")");
    }
    {   // a mean of 1e20: a step of 0.1σ√l along any axis leaves it as it is
        std::mt19937 rng(7);
        CMAES es(Vec(4, 1e20), 1.0, rng, 0, h09());
        const Run r = run(es, [](const Vec& x) { return x[0] + 2.0 * x[1] + 3.0 * x[2] + 4.0 * x[3]; });
        check(r.stop == CMAES::Stop::no_effect_axis && r.t == 1, "mean 1e20: NoEffectAxis at t = 1 (" + said(r) + ")");
    }
    {   // one coordinate of 1e20: the axis moves the others, coordinate 0 stays
        std::mt19937 rng(8);
        Vec m(4, 0.0);
        m[0] = 1e20;
        CMAES es(m, 1.0, rng, 0, h09());
        const Run r = run(es, [](const Vec& x) { return x[1] + 2.0 * x[2] + 3.0 * x[3]; });
        check(r.stop == CMAES::Stop::no_effect_coor && r.t == 1, "m_0 = 1e20: NoEffectCoor at t = 1 (" + said(r) + ")");
    }
}

// H09's CMA-ES on Rosenbrock, D = 10, from m = 0.5, σ⁰ = 0.5: below 1e-8, then a criterion
void rosenbrock()
{
    std::uint64_t h = 1469598103934665603ull;
    auto mix = [&](double v) {
        std::uint64_t b;
        std::memcpy(&b, &v, sizeof b);
        h = (h ^ b) * 1099511628211ull;
    };
    std::mt19937 rng(9);
    CMAES es(Vec(10, 0.5), 0.5, rng, 0, h09());
    const Run r = run(es, [&](const Vec& x) {
        double s = 0.0;
        for (std::size_t i = 0; i + 1 < x.size(); ++i)
            s += 100.0 * (x[i + 1] - x[i] * x[i]) * (x[i + 1] - x[i] * x[i]) + (1.0 - x[i]) * (1.0 - x[i]);
        for (double e : x) mix(e);
        return s;
    });
    check(r.best < 1e-8 && r.stop != CMAES::Stop::none && r.stop != CMAES::Stop::max_iter,
          "H09 CMA-ES, Rosenbrock D = 10: best " + std::to_string(r.best) + ", " + said(r));
    mix(es.sigma());
    char buf[32];
    std::snprintf(buf, sizeof buf, "%016llx", static_cast<unsigned long long>(h));
    check(h == 0x3137571e9df4d9d1ull, std::string("H09 CMA-ES, Rosenbrock D = 10: checksum ") + buf);
}

}   // namespace

int main()
{
    options();
    criteria();
    rosenbrock();
    std::printf("%s (%d failed)\n", g_failed == 0 ? "PASS" : "FAIL", g_failed);
    return g_failed == 0 ? 0 : 1;
}
