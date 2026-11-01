// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// scalar/cmaes.hpp's injection of external points (task 19, §19.4.3; N.
// Hansen, "Injecting external solutions into CMA-ES", arXiv:1110.4181, 2011):
//   - without injection the runs are those before injection came: a checksum
//     of the points of six runs (Rosenbrock and an ellipsoid, n = 2, 5, 10),
//     computed with the header before the change;
//   - a far injected point that ranks first: its step clipped to c_y (3),
//     sigma's change at most e-fold (11); unclipped, sigma explodes;
//   - inject_mean: within c_y^m the mean moves to x_m (4), (6);
//   - elitist: each iteration's best never worsens.
// ============================================================================

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
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

void cma_injection()
{
    using mootation::scalar::CMAES;
    // without injection the runs are those before injection came: the points' checksum
    std::uint64_t h = 1469598103934665603ull;
    auto mix = [&](double v) {
        std::uint64_t b;
        std::memcpy(&b, &v, sizeof b);
        h = (h ^ b) * 1099511628211ull;
    };
    auto value = [](const std::vector<double>& x, int f) {
        const std::size_t n = x.size();
        double s = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            if (f == 0 && i + 1 < n) s += 100 * (x[i + 1] - x[i] * x[i]) * (x[i + 1] - x[i] * x[i]) + (1 - x[i]) * (1 - x[i]);
            if (f == 1) s += (static_cast<double>(i) + 1.0) * x[i] * x[i];
        }
        return s;
    };
    for (int n : {2, 5, 10})
        for (int f = 0; f < 2; ++f) {
            std::mt19937 rng(static_cast<unsigned>(100 + n + f));
            CMAES es(std::vector<double>(static_cast<std::size_t>(n), 0.5), 0.3, rng);
            for (int it = 0; it < 150; ++it) {
                const auto X = es.ask();
                std::vector<double> v;
                for (const auto& x : X) {
                    v.push_back(value(x, f));
                    for (double e : x) mix(e);
                }
                es.tell(v);
            }
            mix(es.spread());
        }
    char buf[32];
    std::snprintf(buf, sizeof buf, "%016llx", static_cast<unsigned long long>(h));
    check(h == 0xee4cc9846d4f14d8ull, std::string("cmaes: without injection the runs as before (checksum ") + buf + ")");

    // a far injected point that ranks first: its step clipped to c_y (3), sigma's change below e^1 (11)
    for (bool clip : {true, false}) {
        std::mt19937 rng(5);
        CMAES es(std::vector<double>(5, 0.5), 0.3, rng);
        for (int it = 0; it < 20; ++it) {
            const auto X = es.ask();
            std::vector<double> v;
            for (const auto& x : X) v.push_back(value(x, 1));
            es.tell(v);
        }
        if (!clip) es.injection({1e300, 1e300, 1e300});
        std::vector<double> far = es.mean();
        far[0] += 1e6;
        es.inject({far});
        const double before = es.sigma();
        const auto X = es.ask();
        std::vector<double> v;
        for (const auto& x : X) v.push_back(value(x, 1));
        v[0] = -1.0;                                    // the injected point ranks first
        es.tell(v);
        const double ratio = es.sigma() / before;
        check(clip ? X[0] == far && ratio <= std::exp(1.0) * (1.0 + 1e-15) : ratio > 1e3,
              std::string("cmaes inject: a far point that ranks first, ") +
                  (clip ? "sigma grows at most e-fold (" : "unclipped sigma explodes (") + std::to_string(ratio) + ")");
    }
    // inject_mean: within c_y^m the mean moves to x_m
    {
        std::mt19937 rng(6);
        CMAES es(std::vector<double>(4, 0.5), 0.3, rng);
        std::vector<double> xm = es.mean();
        for (double& t : xm) t += 0.01;
        es.inject_mean(xm);
        const auto X = es.ask();
        std::vector<double> v;
        for (const auto& x : X) v.push_back(value(x, 1));
        es.tell(v);
        double d = 0.0;
        for (std::size_t i = 0; i < xm.size(); ++i) d = std::max(d, std::fabs(es.mean()[i] - xm[i]));
        check(d <= 1e-15, "cmaes inject_mean: the mean moves to x_m (" + std::to_string(d) + ")");
    }
    // elitist: the best point told so far comes back every iteration, so each iteration's best never worsens
    {
        std::mt19937 rng(8);
        CMAES es(std::vector<double>(6, 2.0), 1.0, rng);
        es.elitist(true);
        double last = 1e300;
        bool monotone = true;
        for (int it = 0; it < 60; ++it) {
            const auto X = es.ask();
            std::vector<double> v;
            for (const auto& x : X) v.push_back(value(x, 0));
            const double best = *std::min_element(v.begin(), v.end());
            monotone = monotone && best <= last;
            last = best;
            es.tell(v);
        }
        check(monotone, "cmaes elitist: each iteration's best never worsens (Rosenbrock, n = 6)");
    }
}

}   // namespace

int main()
{
    cma_injection();
    if (g_failed) {
        std::printf("%d check(s) failed\n", g_failed);
        return 1;
    }
    std::printf("all checks passed\n");
    return 0;
}
