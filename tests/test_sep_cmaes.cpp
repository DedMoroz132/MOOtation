// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// scalar/sep_cmaes.hpp:
//   - Ros & Hansen 2008, Tables 2-4: the mean evaluations to the target, 5
//     runs where the paper has 3, with the tables' starts, step-sizes and
//     default lambda — ellipsoid (beta = 1e6) and Rosenbrock at n = 20, the
//     hyper-ellipsoid, Diff-Pow (beta = n - 1) and Rosenbrock at n = 30, and
//     Table 4's random starts: within 10 % of the tables (they come out at
//     0.95-1.03 of them);
//   - Fig. 2's direction: on the fully rotated ellipsoid sep-CMA-ES does not
//     reach the target (n = 10, 10^5 evaluations), the full CMA-ES of
//     cmaes.hpp does;
//   - c_cov = (n + 2)/3 c_cov^def;
//   - a checksum of the runs, the same bits on every platform.
// ============================================================================

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

#include <mootation/detmath.hpp>
#include <mootation/random.hpp>
#include <mootation/scalar/cmaes.hpp>
#include <mootation/scalar/sep_cmaes.hpp>

namespace {

namespace S = mootation::scalar::sep_cmaes;
namespace dm = mootation::dm;
using S::Vec;

int g_failed = 0;
std::uint64_t g_hash = 1469598103934665603ull;     // FNV-1a over the runs' bits

void mix(double v)
{
    std::uint64_t b;
    std::memcpy(&b, &v, sizeof b);
    g_hash = (g_hash ^ b) * 1099511628211ull;
}

void check(bool ok, const std::string& what)
{
    if (!ok) ++g_failed;
    std::printf("%s  %s\n", ok ? "ok  " : "FAIL", what.c_str());
}

double elli(const Vec& x)
{
    double s = 0.0;
    const std::size_t n = x.size();
    for (std::size_t i = 0; i < n; ++i)
        s += dm::pow(1e6, static_cast<double>(i) / static_cast<double>(n - 1)) * x[i] * x[i];
    return s;
}
double hyperelli(const Vec& x)
{
    double s = 0.0;
    for (std::size_t i = 0; i < x.size(); ++i) {
        const double t = static_cast<double>(i + 1) * x[i];
        s += t * t;
    }
    return s;
}
double diffpow(const Vec& x)                                    // beta = n - 1
{
    double s = 0.0;
    const std::size_t n = x.size();
    for (std::size_t i = 0; i < n; ++i)
        s += dm::pow(std::fabs(x[i]), 2.0 + static_cast<double>(n - 1) * static_cast<double>(i) / static_cast<double>(n - 1));
    return s;
}
double rosen(const Vec& x)
{
    double s = 0.0;
    for (std::size_t i = 0; i + 1 < x.size(); ++i) {
        const double a = x[i] * x[i] - x[i + 1];
        s += 100.0 * a * a + (x[i] - 1.0) * (x[i] - 1.0);
    }
    return s;
}

void tables()
{
    struct Case {
        const char* name;
        std::function<double(const Vec&)> f;
        std::size_t n;
        double start;     // >= 2: uniform in [-start, start]^n, else every coordinate = start
        double sigma, target, paper;
    };
    const Case cases[] = {
        {"Table 3, ellipsoid, n = 20, to 1e-9", elli, 20, 1, 1, 1e-9, 5.4e3},
        {"Table 3, Rosenbrock, n = 20, to 1e-9", rosen, 20, 0, 0.1, 1e-9, 116e3},
        {"Table 2, hyper-ellipsoid, n = 30, to 1e-10", hyperelli, 30, 1, 1, 1e-10, 5.9e3},
        {"Table 2, Diff-Pow, n = 30, to 1e-20", diffpow, 30, 1, 1, 1e-20, 9.6e3},
        {"Table 2, Rosenbrock, n = 30, to 1e-6", rosen, 30, 0, 0.1, 1e-6, 106e3},
        {"Table 4, ellipsoid, n = 30, to 1e-14", elli, 30, 5, 5, 1e-14, 11e3},
        {"Table 4, Rosenbrock, n = 30, to 1e-14", rosen, 30, 2, 2, 1e-14, 191e3},
    };
    for (const Case& c : cases) {
        double sum = 0.0;
        int hit = 0;
        for (unsigned r = 0; r < 5; ++r) {
            std::mt19937 rng(400u + r);
            mootation::rnd::uniform_real_distribution<double> U(0.0, 1.0);
            Vec m(c.n, c.start);
            if (c.start >= 2) for (double& t : m) t = -c.start + 2.0 * c.start * U(rng);
            S::SepCMAES es(m, c.sigma, rng);
            long long nfe = 0;
            bool found = false;
            while (nfe < 3000000 && !found) {
                const auto X = es.ask();
                Vec f(X.size());
                for (std::size_t i = 0; i < X.size(); ++i) {
                    f[i] = c.f(X[i]);
                    ++nfe;
                    if (f[i] < c.target && !found) {
                        found = true;
                        sum += static_cast<double>(nfe);
                        ++hit;
                    }
                }
                es.tell(f);
            }
            mix(static_cast<double>(nfe));
        }
        const double mean = hit ? sum / hit : 0.0;
        char line[200];
        std::snprintf(line, sizeof line, "%s: %d of 5, mean %.0f, paper %.0f (%.2f)", c.name, hit, mean, c.paper, mean / c.paper);
        check(hit == 5 && std::fabs(mean / c.paper - 1.0) <= 0.10, line);
    }
}

void rotated()
{
    const std::size_t n = 10;
    std::mt19937 qg(4242);
    std::vector<Vec> Q(n, Vec(n));
    for (std::size_t i = 0; i < n; ++i) {
        for (double& t : Q[i]) t = mootation::rnd::normal01(qg);
        for (std::size_t k = 0; k < i; ++k) {
            double d = 0.0;
            for (std::size_t j = 0; j < n; ++j) d += Q[i][j] * Q[k][j];
            for (std::size_t j = 0; j < n; ++j) Q[i][j] -= d * Q[k][j];
        }
        double s = 0.0;
        for (double t : Q[i]) s += t * t;
        for (double& t : Q[i]) t /= std::sqrt(s);
    }
    auto f = [&](const Vec& x) {
        Vec y(n, 0.0);
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < n; ++j) y[i] += Q[i][j] * x[j];
        return elli(y);
    };
    auto reach = [&](auto& es) {
        long long nfe = 0;
        while (nfe < 100000) {
            const auto X = es.ask();
            Vec v(X.size());
            for (std::size_t i = 0; i < X.size(); ++i) {
                v[i] = f(X[i]);
                ++nfe;
                if (v[i] < 1e-9) return nfe;
            }
            es.tell(v);
        }
        return -1LL;
    };
    std::mt19937 r1(11), r2(11);
    S::SepCMAES sep(Vec(n, 1.0), 1.0, r1);
    mootation::scalar::CMAES full(Vec(n, 1.0), 1.0, r2);
    const long long a = reach(sep), b = reach(full);
    mix(static_cast<double>(a));
    mix(static_cast<double>(b));
    char line[200];
    std::snprintf(line, sizeof line,
                  "rotated ellipsoid, n = 10, to 1e-9 in 1e5 evaluations: sep-CMA-ES %s, CMA-ES at %lld (Fig. 2)",
                  a < 0 ? "does not reach it" : "reaches it", b);
    check(a < 0 && b > 0, line);
}

}   // namespace

int main()
{
    std::mt19937 rng(1);
    const S::SepCMAES es(Vec(20, 0.0), 1.0, rng);
    {   // Algorithm 1's parameter setting at n = 20, by hand: lambda = 12, mu = 6
        const double n = 20.0;
        double w[6], sw = 0.0, sw2 = 0.0;
        for (int i = 0; i < 6; ++i) sw += w[i] = std::log(7.0) - std::log(i + 1.0);
        for (double& t : w) sw2 += (t / sw) * (t / sw);
        const double mu = 1.0 / sw2;
        const double def = 1.0 / mu * 2.0 / ((n + std::sqrt(2.0)) * (n + std::sqrt(2.0))) +
                           (1.0 - 1.0 / mu) * std::min(1.0, (2.0 * mu - 1.0) / ((n + 2.0) * (n + 2.0) + mu));
        check(es.lambda() == 12 && std::fabs(es.c_cov() / ((n + 2.0) / 3.0 * def) - 1.0) < 1e-12,
              "lambda = 4 + floor(3 ln 20) = 12; c_cov = (n + 2)/3 c_cov^def");
    }
    tables();
    rotated();
    std::printf("checksum %016llx\n", static_cast<unsigned long long>(g_hash));
    check(g_hash == 0x1dcfe22a9336ae29ull, "the checksum of the runs, the same bits on every platform");
    std::printf(g_failed ? "FAILED\n" : "all passed\n");
    return g_failed ? 1 : 0;
}
