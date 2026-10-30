// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// scalar/spso2011.hpp. The paper's experiment is CEC 2013, not in the library
// (task 18, supplement 3, §4: the direction of its findings on bbob comes
// with the heavy runs). Here:
//   - the sphere, CEC 2013's f1 without its shift, D = 10, [-100, 100]^10,
//     10000 D evaluations: every run of 10 below 1e-8, as Table I's 51 runs;
//   - rotational invariance, the point of SPSO-2011 (§II-C): an ellipsoid of
//     condition 1e4 at D = 10, axis-parallel and rotated by a fixed
//     orthogonal matrix, 10 seeds each, final bests after 10000 D evaluations:
//     two-sided Mann-Whitney at 0.05, no difference;
//   - the absorbing walls: with the optimum outside, every position stays in
//     the box and the swarm ends on its corner;
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
#include <mootation/scalar/spso2011.hpp>

namespace {

namespace S = mootation::scalar::spso2011;
namespace evo = mootation::scalar::evo;
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

// Two-sided Mann-Whitney U for two samples of 10: a difference at 0.05 when
// min(U1, U2) <= 23; ties take their mean rank.
bool differ(const Vec& a, const Vec& b)
{
    std::vector<std::pair<double, int>> all;
    for (double v : a) all.push_back({v, 0});
    for (double v : b) all.push_back({v, 1});
    std::sort(all.begin(), all.end());
    double r1 = 0.0;
    for (std::size_t i = 0; i < all.size();) {
        std::size_t j = i;
        while (j < all.size() && all[j].first == all[i].first) ++j;
        const double rank = (static_cast<double>(i + 1) + static_cast<double>(j)) / 2.0;
        for (std::size_t k = i; k < j; ++k)
            if (all[k].second == 0) r1 += rank;
        i = j;
    }
    const double u1 = r1 - 55.0, u2 = 100.0 - u1;
    return std::min(u1, u2) <= 23.0;
}

double run(const std::function<double(const Vec&)>& f, std::size_t n, unsigned seed, long long budget, bool* walls = nullptr)
{
    std::mt19937 rng(seed);
    S::SPSO2011 swarm(Vec(n, -100.0), Vec(n, 100.0), S::Options{}, rng);
    long long nfe = 0;
    double best = INFINITY;
    while (nfe < budget) {
        const evo::Rows X = swarm.ask();
        Vec v(X.size());
        for (std::size_t i = 0; i < X.size(); ++i) {
            v[i] = f(X[i]);
            best = std::min(best, v[i]);
            if (walls)
                for (std::size_t j = 0; j < n; ++j) *walls = *walls && X[i][j] >= -100.0 && X[i][j] <= 100.0;
        }
        nfe += static_cast<long long>(X.size());
        swarm.tell(v);
    }
    mix(best);
    return best;
}

double sphere(const Vec& x)
{
    double s = 0.0;
    for (double v : x) s += v * v;
    return s;
}

}   // namespace

int main()
{
    int below = 0;
    double worst = 0.0;
    for (unsigned r = 0; r < 10; ++r) {
        const double b = run(sphere, 10, 70u + r, 100000);
        below += b < 1e-8;
        worst = std::max(worst, b);
    }
    char line[200];
    std::snprintf(line, sizeof line, "sphere, D = 10, 10000 D evaluations: %d of 10 below 1e-8 (worst %.2e)", below, worst);
    check(below == 10, line);

    // a fixed orthogonal matrix: Gram-Schmidt on normal vectors of its own generator
    const std::size_t n = 10;
    std::mt19937 qg(12345);
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
    auto ellipsoid = [&](const Vec& x) {
        double s = 0.0;
        for (std::size_t i = 0; i < n; ++i) s += mootation::dm::pow(1e4, static_cast<double>(i) / static_cast<double>(n - 1)) * x[i] * x[i];
        return s;
    };
    auto rotated = [&](const Vec& x) {
        Vec y(n, 0.0);
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < n; ++j) y[i] += Q[i][j] * x[j];
        return ellipsoid(y);
    };
    Vec a, b;
    for (unsigned r = 0; r < 10; ++r) {
        a.push_back(run(ellipsoid, n, 80u + r, 100000));
        b.push_back(run(rotated, n, 90u + r, 100000));
    }
    std::sort(a.begin(), a.end());
    std::sort(b.begin(), b.end());
    std::snprintf(line, sizeof line,
                  "ellipsoid 1e4, D = 10, axis-parallel and rotated: medians %.3g and %.3g, Mann-Whitney 0.05: no difference",
                  0.5 * (a[4] + a[5]), 0.5 * (b[4] + b[5]));
    check(!differ(a, b), line);

    bool walls = true;
    const double corner =
        run([](const Vec& x) { double s = 0.0; for (double v : x) s += (v - 150.0) * (v - 150.0); return s; }, 5, 7u, 8000, &walls);
    std::snprintf(line, sizeof line, "absorbing walls: the optimum at 150 outside; every position in the box, the best at the corner (f = %.9g)", corner);
    check(walls && corner == 12500.0, line);

    std::printf("checksum %016llx\n", static_cast<unsigned long long>(g_hash));
    check(g_hash == 0xf7c3caa1461c6f0dull, "the checksum of the runs, the same bits on every platform");
    std::printf(g_failed ? "FAILED\n" : "all passed\n");
    return g_failed ? 1 : 0;
}
